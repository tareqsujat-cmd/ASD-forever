"""
Choosing the shrinkage honestly, and checking that the gain survives.

A sweep that reports its best value has chosen that value by looking at held-out
performance, which is the practice this work criticises elsewhere. To claim that
calibrating shrinkage to the classification problem helps, the amount of shrinkage must be
selected inside the training partition and then applied unchanged to data that played no
part in choosing it.

Each outer fold therefore selects the shrinkage by a cross-validation run entirely within
its own training participants, and the selected value is applied once to the held-out
fold. The comparison is against the same pipeline with no extra shrinkage, on identical
folds, so the difference is attributable to the selection and nothing else.

Usage:
    python exp_ridge_nested.py --study results/<study>
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
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Nested selection of the shrinkage level")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--grid", nargs="+", type=float,
                    default=[0.0, 0.10, 0.20, 0.30, 0.40, 0.50])
    ap.add_argument("--protocols", nargs="+", default=["pooled", "loso"])
    ap.add_argument("--cohort", default="strict", choices=["strict", "func"])
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--study", required=True)
    args = ap.parse_args()

    import torch
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    from run_benchmark import load_dataset, _outer_splits
    from run_ablations import logreg_factory, _scaled
    from gpu_tangent import GPUTangentFC
    from evaluation.stat_compare import delong_roc_test

    if args.cohort == "func":
        from qc_cohort import load_from_phenotype
        ts, y, sites, _ = load_from_phenotype(
            Path("abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"),
            Path(args.ts_root), "rois_cc200")
    else:
        ts, y, sites, _ = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    y = np.asarray(y); sites = np.asarray(sites)
    print(f"cohort {args.cohort}: {len(y)} participants, {len(set(sites))} sites", flush=True)

    # covariances at every candidate level, computed once
    covs = {}
    for r in args.grid:
        gt = GPUTangentFC(device="cpu", max_iter=10, ridge=r, dtype=torch.float64)
        covs[r] = gt.compute_covariances(ts)
        print(f"  covariances at ridge {r:.2f}", flush=True)

    def fit_score(r, tr, te):
        gt = GPUTangentFC(device="cpu", max_iter=10, ridge=r, dtype=torch.float64)
        gt.fit(covs[r][tr])
        Xtr, Xte = _scaled(gt.transform(covs[r][tr]), gt.transform(covs[r][te]))
        clf = logreg_factory()().fit(Xtr, y[tr])
        return clf.predict_proba(Xte)[:, 1]

    rows, records = [], []
    for proto in args.protocols:
        splits, _ = _outer_splits(proto, y, sites, args.n_folds, args.seed)
        oof = {"nested": ([], [], []), "baseline": ([], [], [])}
        chosen = []
        for k, (tr, te) in enumerate(splits):
            # choose the level using only the training participants of this fold
            inner = StratifiedKFold(args.inner_folds, shuffle=True, random_state=args.seed)
            scores = {r: [] for r in args.grid}
            for itr, iva in inner.split(np.zeros(len(tr)), y[tr]):
                for r in args.grid:
                    try:
                        p = fit_score(r, tr[itr], tr[iva])
                        scores[r].append(roc_auc_score(y[tr[iva]], p))
                    except Exception:
                        scores[r].append(np.nan)
            mean = {r: np.nanmean(v) for r, v in scores.items()}
            r_hat = max(mean, key=mean.get)
            chosen.append(r_hat)

            for name, r in (("nested", r_hat), ("baseline", 0.0)):
                try:
                    p = fit_score(r, tr, te)
                except Exception as exc:
                    print(f"  fold {k} {name} failed: {type(exc).__name__}"); continue
                oof[name][0].extend(y[te].tolist())
                oof[name][1].extend(p.tolist())
                oof[name][2].extend(sites[te].tolist())
            print(f"  {proto} fold {k+1}/{len(splits)}: chose ridge {r_hat:.2f}", flush=True)

        res = {}
        for name in ("nested", "baseline"):
            yy = np.asarray(oof[name][0]); pp = np.asarray(oof[name][1])
            if not len(yy):
                continue
            res[name] = (yy, pp)
            rows.append({"protocol": proto, "condition": name, "n": int(len(yy)),
                         "auroc": float(roc_auc_score(yy, pp)),
                         "accuracy": float(accuracy_score(yy, (pp >= 0.5).astype(int)))})
            records.append({"condition": f"{name} ridge [{proto}]", "protocol": proto,
                            "oof_true": yy.tolist(), "oof_prob": pp.tolist(),
                            "oof_site": oof[name][2]})
        if "nested" in res and "baseline" in res:
            yy = res["nested"][0]
            a, b, p = delong_roc_test(yy, res["nested"][1], res["baseline"][1])
            rows[-2]["delong_p_vs_baseline"] = float(p)
            print(f"\n  {proto}: nested {a:.4f} vs unridged {b:.4f} -> "
                  f"{a-b:+.4f} (DeLong p={p:.4f})")
            vals, cnts = np.unique(chosen, return_counts=True)
            print(f"  levels chosen: " +
                  ", ".join(f"{v:.2f}x{c}" for v, c in zip(vals, cnts)), flush=True)

    df = pd.DataFrame(rows)
    study = Path(args.study)
    t = study / "tables"; t.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.cohort == "strict" else f"_{args.cohort}"
    df.to_csv(t / f"ridge_nested{suffix}.csv", index=False)
    (study / "raw").mkdir(parents=True, exist_ok=True)
    (study / "raw" / f"ridge_nested{suffix}.json").write_text(json.dumps(records))
    print("\n", df.to_string(index=False))
    print("\n->", t / f"ridge_nested{suffix}.csv")


if __name__ == "__main__":
    main()
