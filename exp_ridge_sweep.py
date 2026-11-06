"""
Static and windowed connectivity compared at matched regularisation.

The windowed representation in ``dynamic_fc`` applies an isotropic ridge to every window
covariance, because a window is far shorter than the number of regions.  The whole-scan
representation in ``gpu_tangent`` applies no such ridge.  A comparison between the two
therefore varies the amount of shrinkage at the same time as the representation, and any
difference between them confounds the two.

This sweeps the ridge on the *static* representation across the same range, so the
windowed result can be read against a static baseline regularised the same way.

Usage:
    python exp_ridge_sweep.py --out results/<study>/ablations
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
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Static vs windowed at matched regularisation")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--ridges", nargs="+", type=float,
                    default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--n_windows", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from run_benchmark import load_dataset, _outer_splits
    from run_ablations import cv_evaluate, tangent_features, logreg_factory, _scaled
    from gpu_tangent import GPUTangentFC
    from dynamic_fc import DynamicTangentFC
    from evaluation.stat_compare import delong_roc_test

    ts, y, sites, _ = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)
    print(f"{len(y)} participants | {args.n_folds} folds", flush=True)

    records, rows = [], []

    for r in args.ridges:
        gt = GPUTangentFC(device="cpu", max_iter=10, ridge=r)
        covs = gt.compute_covariances(ts)
        res = cv_evaluate(tangent_features(gt, covs), logreg_factory(), splits, y, sites,
                          f"static ridge={r:.2f}", protocol="pooled")
        records.append(res)
        rows.append({"condition": f"static ridge={r:.2f}", "representation": "static",
                     "ridge": r, "auroc": res["fold_mean"]["auroc"],
                     "auroc_std": res["fold_std"]["auroc"],
                     "accuracy": res["fold_mean"]["accuracy"]})
        print(f"  static ridge={r:.2f}  AUROC {res['fold_mean']['auroc']:.4f}", flush=True)

    # the windowed representation at its own ridge, for reference
    dfc = DynamicTangentFC(window=args.window, n_windows=args.n_windows)
    wcovs = dfc.compute_window_covariances(ts)

    def dyn_features(tr, te, k=None):
        # same standardisation the static path applies, so the two differ only in
        # representation and shrinkage
        dfc.fit(wcovs, tr)
        B_tr = dfc.transform(wcovs, tr, blocks=("mean",))
        B_te = dfc.transform(wcovs, te, blocks=("mean",))
        return _scaled(B_tr["mean"], B_te["mean"])

    res = cv_evaluate(dyn_features, logreg_factory(), splits, y, sites,
                      f"dynamic: mean (ridge={dfc.ridge:.2f})", protocol="pooled")
    records.append(res)
    rows.append({"condition": f"dynamic: mean (ridge={dfc.ridge:.2f})",
                 "representation": "windowed", "ridge": dfc.ridge,
                 "auroc": res["fold_mean"]["auroc"], "auroc_std": res["fold_std"]["auroc"],
                 "accuracy": res["fold_mean"]["accuracy"]})
    print(f"  dynamic: mean ridge={dfc.ridge:.2f}  AUROC {res['fold_mean']['auroc']:.4f}",
          flush=True)

    # the comparisons that decide the claim
    by = {r["condition"]: r for r in records}
    base = f"static ridge={args.ridges[0]:.2f}"
    dyn = f"dynamic: mean (ridge={dfc.ridge:.2f})"
    matched = f"static ridge={dfc.ridge:.2f}"
    tests = []
    for a, b, meaning in (
        (matched, base, "ridge alone, on the static representation"),
        (dyn, base, "the windowed representation against an unridged static baseline"),
        (dyn, matched, "the windowed representation against a matched static baseline"),
    ):
        if a not in by or b not in by:
            continue
        ya, pa = np.asarray(by[a]["oof_true"]), np.asarray(by[a]["oof_prob"])
        pb = np.asarray(by[b]["oof_prob"])
        auc_a, auc_b, pv = delong_roc_test(ya, pa, pb)
        tests.append({"comparison": f"{a} vs {b}", "meaning": meaning,
                      "auroc_a": float(auc_a), "auroc_b": float(auc_b),
                      "difference": float(auc_a - auc_b), "delong_p": float(pv)})
        print(f"  {meaning:58s} {auc_a - auc_b:+.4f}  p={pv:.4f}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "ridge_sweep.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "ridge_sweep_tests.csv", index=False)
    (out / "ridge_sweep.json").write_text(json.dumps(records, indent=2))
    print("->", out / "ridge_sweep.csv")


if __name__ == "__main__":
    main()
