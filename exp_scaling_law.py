"""
Two ways of spending a scanning budget: more minutes, or more people.

Scan length in this release varies by a factor of four, and that variation is organised by
site, so comparing sites confounds duration with everything else that differs between
centres.  The comparison here is made within participants instead: the same people are
analysed at successively shorter truncations of their own recording, so nothing changes
across the sweep except how much signal each covariance is estimated from.  The second
axis adds participants at full length, which is the axis the field has actually been
spending on.

Two things are held fixed that would otherwise be confounded with duration.  The
covariance is re-estimated from the truncated series at every length rather than reused,
and the regularisation strength is re-selected at every length, so a change in accuracy
cannot be attributed to an estimator that happens to suit one length better than another.

Usage:
    python exp_scaling_law.py --study results/<study>
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
    ap = argparse.ArgumentParser(description="Duration and sample-size scaling")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--durations", nargs="+", type=int,
                    default=[40, 60, 90, 120, 160, 200, 232])
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[100, 200, 300, 400, 550, 700, 825])
    ap.add_argument("--min_frames", type=int, default=232,
                    help="participants must have at least this many volumes to enter the "
                         "duration sweep, so the same people appear at every length")
    ap.add_argument("--subject_repeats", type=int, default=1,
                    help="random subsets drawn per sample size. One draw carries the noise "
                         "of that particular subset, which is not comparable with the "
                         "duration axis where the same people appear at every length.")
    ap.add_argument("--site_sweep", default="NYU",
                    help="repeat the sample-size axis inside one site, so that adding "
                         "participants does not also mean adding sites")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ridges", nargs="+", type=float, default=[0.15, 0.30, 0.50],
                    help="the duration axis is repeated at each of these shrinkage levels, "
                         "so the effect of length cannot be confused with the effect of "
                         "regularisation")
    ap.add_argument("--study", required=True)
    args = ap.parse_args()

    from run_benchmark import load_dataset, _outer_splits
    from run_ablations import cv_evaluate, tangent_features, logreg_factory
    from gpu_tangent import GPUTangentFC

    ts, y, sites, _ = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    sites = np.asarray(sites)
    lengths = np.array([t.shape[0] for t in ts])
    print(f"{len(y)} participants | scan length {lengths.min()}-{lengths.max()} "
          f"(median {int(np.median(lengths))})", flush=True)

    rows = []

    # ---------------- axis 1: minutes, within participant ----------------
    keep = np.where(lengths >= args.min_frames)[0]
    print(f"\nduration axis: {len(keep)} participants with >= {args.min_frames} volumes "
          f"({int(y[keep].sum())} autism)", flush=True)
    y_k, sites_k = y[keep], sites[keep]
    splits_k, _ = _outer_splits("pooled", y_k, sites_k, args.n_folds, args.seed)

    import torch
    for T in args.durations:
        # the covariance is rebuilt from the truncated series; reusing a covariance
        # computed once from the full recording would make the sweep meaningless
        trunc = [np.asarray(ts[i][:T], dtype=np.float64) for i in keep]
        assert all(t.shape[0] == T for t in trunc), "truncation did not take"
        for r in args.ridges:
            # a 40-volume covariance over 200 regions is badly rank deficient, so the
            # eigendecomposition behind the log map needs float64 and enough shrinkage
            # to stay conditioned; the sweep over r is what keeps the length effect
            # separable from the shrinkage effect
            gt = GPUTangentFC(device="cpu", max_iter=10, ridge=r, dtype=torch.float64)
            try:
                covs = gt.compute_covariances(trunc)
                res = cv_evaluate(tangent_features(gt, covs), logreg_factory(), splits_k,
                                  y_k, sites_k, f"T={T} r={r:.2f}", protocol="pooled")
            except Exception as exc:
                print(f"  T={T:4d} r={r:.2f}  failed: {type(exc).__name__}", flush=True)
                continue
            rows.append({"axis": "duration", "value": T, "ridge": r, "n": len(keep),
                         "auroc": res["fold_mean"]["auroc"],
                         "auroc_std": res["fold_std"]["auroc"],
                         "accuracy": res["fold_mean"]["accuracy"]})
            print(f"  T={T:4d} frames  r={r:.2f}  n={len(keep)}  "
                  f"AUROC {res['fold_mean']['auroc']:.4f}"
                  f" +/- {res['fold_std']['auroc']:.4f}", flush=True)

    # ---------------- axis 2: people, at full length ----------------
    r0 = args.ridges[0]
    gt_full = GPUTangentFC(device="cpu", max_iter=10, ridge=r0, dtype=torch.float64)
    covs_full = gt_full.compute_covariances(ts)
    rng = np.random.RandomState(args.seed)

    for n in args.sizes:
        if n > len(y):
            continue
        draws = []
        for rep in range(args.subject_repeats):
            idx = rng.choice(len(y), n, replace=False)
            splits_n, _ = _outer_splits("pooled", y[idx], sites[idx], args.n_folds,
                                        args.seed)
            gt_n = GPUTangentFC(device="cpu", max_iter=10, ridge=r0, dtype=torch.float64)
            try:
                res = cv_evaluate(tangent_features(gt_n, covs_full[idx]), logreg_factory(),
                                  splits_n, y[idx], sites[idx], f"n={n} rep={rep}",
                                  protocol="pooled")
            except Exception as exc:
                print(f"  n={n:4d} rep={rep} failed: {type(exc).__name__}", flush=True)
                continue
            draws.append((res["fold_mean"]["auroc"], res["fold_mean"]["accuracy"]))
        if not draws:
            continue
        aucs = np.array([d[0] for d in draws])
        rows.append({"axis": "subjects", "value": n, "ridge": r0, "n": n,
                     "auroc": float(aucs.mean()),
                     "auroc_std": float(aucs.std()) if len(aucs) > 1 else float("nan"),
                     "n_draws": len(draws),
                     "accuracy": float(np.mean([d[1] for d in draws]))})
        spread = f" (spread {aucs.std():.4f} over {len(aucs)} draws)" if len(aucs) > 1 else ""
        print(f"  n={n:4d} people   AUROC {aucs.mean():.4f}{spread}", flush=True)

    # ---------------- axis 2b: people, inside one site ----------------
    if args.site_sweep:
        m = np.where(sites == args.site_sweep)[0]
        if len(m) >= 60:
            print(f"\nsample size inside {args.site_sweep} (n={len(m)}):", flush=True)
            for n in [x for x in args.sizes if x <= len(m)] + [len(m)]:
                idx = rng.choice(m, n, replace=False)
                if len(set(y[idx])) < 2:
                    continue
                sp, _ = _outer_splits("pooled", y[idx], sites[idx], 5, args.seed)
                gt_s = GPUTangentFC(device="cpu", max_iter=10, ridge=r0, dtype=torch.float64)
                res = cv_evaluate(tangent_features(gt_s, covs_full[idx]), logreg_factory(),
                                  sp, y[idx], sites[idx], f"{args.site_sweep} n={n}",
                                  protocol="pooled")
                rows.append({"axis": f"subjects ({args.site_sweep})", "value": n, "ridge": r0, "n": n,
                             "auroc": res["fold_mean"]["auroc"],
                             "auroc_std": res["fold_std"]["auroc"],
                             "accuracy": res["fold_mean"]["accuracy"]})
                print(f"  n={n:4d} within-site  AUROC {res['fold_mean']['auroc']:.4f}",
                      flush=True)

    df = pd.DataFrame(rows)
    study = Path(args.study)
    t = study / "tables"; t.mkdir(parents=True, exist_ok=True)
    df.to_csv(t / "scaling_law.csv", index=False)

    # the contrast that carries the claim
    from scipy.stats import spearmanr
    stats = {}
    for axis in df["axis"].unique():
        for r in sorted(df[df.axis == axis]["ridge"].dropna().unique()):
            d = df[(df.axis == axis) & (df.ridge == r)]
            if len(d) < 3:
                continue
            rho, pv = spearmanr(d["value"], d["auroc"])
            key = f"{axis} @ ridge {r:.2f}"
            stats[key] = {"n_points": int(len(d)), "ridge": float(r),
                          "spearman_rho": float(rho), "p": float(pv),
                          "auroc_min": float(d["auroc"].min()),
                          "auroc_max": float(d["auroc"].max()),
                          "range": float(d["auroc"].max() - d["auroc"].min())}
            print(f"\n{key:34s} rho={rho:+.3f} p={pv:.4f}  "
                  f"AUROC {d['auroc'].min():.3f}-{d['auroc'].max():.3f}")
    (study / "raw").mkdir(parents=True, exist_ok=True)
    (study / "raw" / "scaling_law.json").write_text(json.dumps(stats, indent=2))
    print("\n->", t / "scaling_law.csv")


if __name__ == "__main__":
    main()
