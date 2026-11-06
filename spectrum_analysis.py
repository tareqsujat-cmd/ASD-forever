"""
Spectral structure of the connectome classifier.

Two distinct notions of "low rank" arise for connectome classification, and they concern
different objects:

  data rank    the participants' feature vectors lie close to a low-dimensional subspace
               of the 19,900-dimensional feature space (a property of the sample);
  weight rank  the fitted weight matrix W, viewed as a region-by-region matrix, is well
               approximated by a few components u_k u_k^T (a property of the classifier).

A model may exhibit one without the other, so both are measured here. For the data, the
explained variance of the feature matrix is accumulated over components. For the weights,
the eigenvalue spectrum of W is computed inside each cross-validation fold, and the share
of spectral mass captured by the leading components is reported alongside the accuracy
obtained when the classifier is restricted to them.

The script emits a JSON record and a figure with three panels: cumulative spectral mass
for both objects, AUROC as a function of retained components, and the eigenvalue profile
of W averaged over folds.

Usage:
    python spectrum_analysis.py --n_folds 5 --out results/ablations
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import warnings
warnings.filterwarnings("ignore")
import numpy as np

from run_benchmark import load_dataset, _outer_splits
from gpu_tangent import GPUTangentFC
from models.lowrank_spectral import vec_to_sym, batch_vec_to_sym, n_rois_from_dim


def analyse(covs, y, sites, splits, ranks, C=0.01):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    gt = GPUTangentFC(device="cpu", max_iter=10)
    weight_spec, data_spec = [], []
    auroc_by_rank = {r: [] for r in ranks}

    for k, (tr, te) in enumerate(splits):
        gt.fit(covs[tr])
        Xtr, Xte = gt.transform(covs[tr]), gt.transform(covs[te])
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        p = n_rois_from_dim(Xtr.shape[1])

        # ---- data spectrum: singular values of the centred feature matrix ----
        s = np.linalg.svd(Xtr - Xtr.mean(0), compute_uv=False)
        data_spec.append((s ** 2) / (s ** 2).sum())

        # ---- weight spectrum: eigenvalues of the fitted W ----
        dense = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(Xtr, y[tr])
        W = vec_to_sym(dense.coef_.ravel(), p, halve=True)
        evals, V = np.linalg.eigh(W)
        order = np.argsort(np.abs(evals))[::-1]
        ev = np.abs(evals[order])
        weight_spec.append(ev / ev.sum())

        # ---- accuracy when restricted to the leading components ----
        Str, Ste = batch_vec_to_sym(Xtr, p), batch_vec_to_sym(Xte, p)
        for r in ranks:
            U = V[:, order[:r]]
            Ztr = np.einsum("nij,ik,jk->nk", Str, U, U, optimize=True)
            Zte = np.einsum("nij,ik,jk->nk", Ste, U, U, optimize=True)
            head = LogisticRegression(C=1.0, solver="liblinear",
                                      max_iter=2000).fit(Ztr, y[tr])
            auroc_by_rank[r].append(
                roc_auc_score(y[te], head.predict_proba(Zte)[:, 1]))
        print(f"  fold {k + 1}/{len(splits)} done", flush=True)

    m = min(len(v) for v in data_spec)
    return {
        "weight_spectrum": np.mean(np.stack(weight_spec), axis=0),
        "data_spectrum": np.mean(np.stack([d[:m] for d in data_spec]), axis=0),
        "auroc_by_rank": {r: float(np.mean(v)) for r, v in auroc_by_rank.items()},
        "auroc_std_by_rank": {r: float(np.std(v)) for r, v in auroc_by_rank.items()},
    }


def figure(res, ranks, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.style import ieee_style, COLORS

    ws = np.cumsum(res["weight_spectrum"])
    ds = np.cumsum(res["data_spectrum"])
    with ieee_style():
        fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))

        ax[0].plot(np.arange(1, len(ws) + 1), ws, lw=2.4,
                   color=COLORS.get("asd", "#c0392b"), label="classifier weights $W$")
        ax[0].plot(np.arange(1, len(ds) + 1), ds, lw=2.4, ls="--",
                   color="#5b8db8", label="feature matrix")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("components retained")
        ax[0].set_ylabel("cumulative spectral mass")
        ax[0].set_title("How concentrated is the spectrum?")
        ax[0].legend(fontsize=7.5, loc="lower right")
        ax[0].set_ylim(0, 1.02)

        rs = sorted(res["auroc_by_rank"])
        mu = [res["auroc_by_rank"][r] for r in rs]
        sd = [res["auroc_std_by_rank"][r] for r in rs]
        ax[1].errorbar(rs, mu, yerr=sd, fmt="-o", lw=2.4, capsize=3,
                       color=COLORS.get("asd", "#c0392b"))
        ax[1].set_xscale("log")
        ax[1].set_xlabel("components retained")
        ax[1].set_ylabel("AUROC")
        ax[1].set_title("Accuracy under compression")

        ev = res["weight_spectrum"]
        ax[2].plot(np.arange(1, len(ev) + 1), ev, lw=2.2, color="#444444")
        ax[2].set_yscale("log"); ax[2].set_xlabel("component index")
        ax[2].set_ylabel("normalised $|\\lambda_k|$")
        ax[2].set_title("Eigenvalue profile of $W$")

        fig.tight_layout()
        out.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"spectrum_analysis.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("figure ->", out / "spectrum_analysis.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Spectral structure of data and weights")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_dir", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranks", nargs="+", type=int,
                    default=[1, 2, 5, 10, 20, 50, 100, 150, 200])
    ap.add_argument("--out", default="results/ablations")
    args = ap.parse_args()

    ts, y, sites, _ = load_dataset(Path(args.processed_dir), Path(args.ts_dir))
    gt = GPUTangentFC(device="cpu", max_iter=10)
    covs = gt.compute_covariances(ts)
    splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)

    res = analyse(covs, y, sites, splits, args.ranks)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ws, ds = np.cumsum(res["weight_spectrum"]), np.cumsum(res["data_spectrum"])
    summary = {
        "weight_mass_top10": float(ws[9]), "weight_mass_top50": float(ws[49]),
        "weight_components_for_90pct": int(np.searchsorted(ws, 0.90) + 1),
        "data_mass_top10": float(ds[9]) if len(ds) > 9 else None,
        "data_mass_top50": float(ds[49]) if len(ds) > 49 else None,
        "data_components_for_90pct": int(np.searchsorted(ds, 0.90) + 1),
        "auroc_by_rank": res["auroc_by_rank"],
        "auroc_std_by_rank": res["auroc_std_by_rank"],
    }
    (out / "spectrum_analysis.json").write_text(json.dumps(summary, indent=2))

    print("\n=== spectral structure ===")
    print(f"  classifier weights: top-10 = {summary['weight_mass_top10']:.3f} of mass, "
          f"top-50 = {summary['weight_mass_top50']:.3f}, "
          f"{summary['weight_components_for_90pct']} components reach 90%")
    print(f"  feature matrix:     top-10 = {summary['data_mass_top10']:.3f}, "
          f"top-50 = {summary['data_mass_top50']:.3f}, "
          f"{summary['data_components_for_90pct']} components reach 90%")
    print("\n  AUROC by components retained:")
    for r in sorted(res["auroc_by_rank"]):
        print(f"    {r:4d}: {res['auroc_by_rank'][r]:.4f}")
    figure(res, args.ranks, out)
    print("report ->", out / "spectrum_analysis.json")


if __name__ == "__main__":
    main()
