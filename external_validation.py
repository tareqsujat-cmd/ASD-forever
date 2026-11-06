"""
External-validation harness for the ASD connectome model.

The adopted model (multi-atlas tangent-space FC + phenotype late fusion, no ComBat,
no post-hoc recalibration) is trained on a *source* cohort and then applied, frozen,
to an *external* cohort. Every learned component — the per-atlas tangent reference,
the feature scaler, the logistic weights, and the phenotype model — is fit on the
source only; the external cohort is touched exactly once, at prediction time.

Two modes are provided:

  * ``--mock_holdout_sites S1 S2 ...``  — uses ABIDE-I and treats the named sites as a
    stand-in external cohort (train on the rest, apply to the held-out sites through the
    external code path). This validates the harness end-to-end without a second dataset.

  * ``--external_ts_dir DIR --external_labels CSV``  — real external validation. DIR holds
    ROI time series named ``<...>_rois_<atlas>.1D`` (same convention as ABIDE-I); CSV has
    columns ``subject_id,label`` and, optionally, the phenotype columns
    ``age,sex,fiq,viq,piq,handedness``. Intended for ABIDE-II once its derivatives are
    downloaded in this format.

Usage:
    python external_validation.py --mock_holdout_sites STANFORD OLIN
    python external_validation.py --external_ts_dir ./abide2/ts --external_labels ./abide2/labels.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")

import warnings
warnings.filterwarnings("ignore")
import numpy as np

from train_sota import load_multi_atlas, load_phenotype, ATLAS_INFO
from gpu_tangent import GPUTangentFC


# --------------------------- metrics ---------------------------

def _metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "auroc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y, yhat)),
        "sensitivity": float(sens), "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2),
    }


def _ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(e)


# --------------------------- freeze-train-and-apply ---------------------------

def fit_apply(ts_by_atlas, y, pheno, tr, ex, atlases, seed=42, device="cpu"):
    """Train the frozen model on `tr` and return ensemble probabilities on `ex`."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.impute import SimpleImputer

    ext_probs = []
    for a in atlases:
        gt = GPUTangentFC(device=device, max_iter=10)
        covs = gt.compute_covariances(ts_by_atlas[a])       # all subjects
        gt.fit(covs[tr])                                    # reference on SOURCE only
        Xtr, Xex = gt.transform(covs[tr]), gt.transform(covs[ex])
        sc = StandardScaler().fit(Xtr)
        Xtr, Xex = sc.transform(Xtr), sc.transform(Xex)
        gs = GridSearchCV(LogisticRegression(solver="liblinear", max_iter=2000),
                          {"C": [0.001, 0.01, 0.1, 1.0]}, scoring="roc_auc",
                          cv=3, n_jobs=-1).fit(Xtr, y[tr])
        ext_probs.append(gs.predict_proba(Xex)[:, 1])
        print(f"  atlas={a}: source-trained, applied to {len(ex)} external subjects")

    if pheno is not None:
        imp = SimpleImputer(strategy="median").fit(pheno[tr])
        psc = StandardScaler().fit(imp.transform(pheno[tr]))
        pgs = GridSearchCV(LogisticRegression(max_iter=2000),
                           {"C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc",
                           cv=3, n_jobs=-1).fit(psc.transform(imp.transform(pheno[tr])), y[tr])
        ext_probs.append(pgs.predict_proba(psc.transform(imp.transform(pheno[ex])))[:, 1])
        print(f"  phenotype member applied")

    return np.mean(ext_probs, axis=0)


# --------------------------- external file loader (ABIDE-II) ---------------------------

def load_external(ts_dir: Path, labels_csv: Path, atlases):
    """Load an external cohort: ROI .1D per atlas + a labels/phenotype CSV."""
    import pandas as pd
    df = pd.read_csv(labels_csv)
    ids = df["subject_id"].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()
    ts_by_atlas = {}
    for a in atlases:
        suffix = ATLAS_INFO[a][0]                            # e.g. "rois_cc200"
        series = []
        for sid in ids:
            hits = list(ts_dir.glob(f"*{sid}*{suffix}.1D"))
            if not hits:
                raise FileNotFoundError(f"no {suffix} file for subject {sid} in {ts_dir}")
            series.append(np.loadtxt(hits[0]))
        ts_by_atlas[a] = series
    phe_cols = ["age", "sex", "fiq", "viq", "piq", "handedness"]
    pheno = df[phe_cols].to_numpy(float) if all(c in df for c in phe_cols) else None
    return ts_by_atlas, y, np.array(ids), pheno


# --------------------------- main ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="External validation (source ABIDE-I)")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--atlases", nargs="+", default=["cc200", "aal", "ho"])
    ap.add_argument("--mock_holdout_sites", nargs="+", default=None,
                    help="ABIDE-I sites to treat as a stand-in external cohort")
    ap.add_argument("--external_ts_dir", default=None)
    ap.add_argument("--external_labels", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/external_validation")
    args = ap.parse_args()

    # source = ABIDE-I
    src_ts, y_src, sites_src, sids_src = load_multi_atlas(
        Path(args.processed_dir), Path(args.ts_root), args.atlases)
    pheno_src = load_phenotype(args.processed_dir, sids_src)

    if args.mock_holdout_sites:
        hold = set(args.mock_holdout_sites)
        ex = np.array([i for i, s in enumerate(sites_src) if s in hold])
        tr = np.array([i for i, s in enumerate(sites_src) if s not in hold])
        if len(ex) == 0:
            raise SystemExit(f"no subjects from sites {hold}; available: {sorted(set(sites_src))}")
        print(f"MOCK external validation: hold out {sorted(hold)} "
              f"({len(ex)} subjects) | train on {len(tr)}")
        ts_all, y_all, pheno_all = src_ts, y_src, pheno_src
        y_ex = y_all[ex]

    elif args.external_ts_dir and args.external_labels:
        ext_ts, y_ext, ids_ext, pheno_ext = load_external(
            Path(args.external_ts_dir), Path(args.external_labels), args.atlases)
        # concatenate source + external so the tangent transform is shared
        n_src = len(y_src)
        ts_all = {a: list(src_ts[a]) + list(ext_ts[a]) for a in args.atlases}
        y_all = np.concatenate([y_src, y_ext])
        pheno_all = (np.vstack([pheno_src, pheno_ext])
                     if (pheno_src is not None and pheno_ext is not None) else None)
        tr = np.arange(n_src); ex = np.arange(n_src, n_src + len(y_ext))
        y_ex = y_ext
        print(f"EXTERNAL validation: train on {n_src} ABIDE-I | test on {len(y_ext)} external")
    else:
        raise SystemExit("provide --mock_holdout_sites OR (--external_ts_dir and --external_labels)")

    prob = fit_apply(ts_all, y_all, pheno_all, tr, ex, args.atlases, args.seed, args.device)
    m = _metrics(y_ex, prob); m["ece"] = _ece(y_ex, prob); m["n_external"] = int(len(ex))

    print("\n=== external-validation metrics ===")
    for k, v in m.items():
        print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "external_report.json").write_text(json.dumps(
        {"mode": "mock" if args.mock_holdout_sites else "external",
         "holdout_sites": args.mock_holdout_sites, "metrics": m,
         "y_true": y_ex.tolist(), "y_prob": prob.tolist()}, indent=2))
    print("report ->", out / "external_report.json")


if __name__ == "__main__":
    main()
