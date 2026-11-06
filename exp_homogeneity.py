"""
Is data from one scanner worth more, or is testing on one scanner merely easier?

Training on participants from a single site and testing on held-out participants from
that same site gives a much higher figure than training and testing across sites. That
is often read as evidence that heterogeneity is costly. It admits a second reading: the
within-site figure is measured on a test set drawn from the same distribution as the
training set, so the task itself is easier, and nothing has been learned about the value
of the data.

The two readings differ in a checkable way. If homogeneous training data is genuinely
more informative, a model trained at one site should also transfer better to *other*
sites. If the advantage is only easier evaluation, it should vanish the moment the test
set comes from elsewhere.

Training-set size is held equal in every condition, so the comparison is about where the
participants come from and not how many there are.

Usage:
    python exp_homogeneity.py --study results/<study>
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
    ap = argparse.ArgumentParser(description="Homogeneity of training data vs of the test set")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--site", default="NYU", help="the single site used as the homogeneous source")
    ap.add_argument("--n_train", type=int, default=120)
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--ridge", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--study", required=True)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import roc_auc_score
    from run_benchmark import load_dataset
    from run_ablations import logreg_factory, _scaled
    from gpu_tangent import GPUTangentFC

    ts, y, sites, _ = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    sites = np.asarray(sites)
    gt0 = GPUTangentFC(device="cpu", max_iter=10, ridge=args.ridge, dtype=torch.float64)
    covs = gt0.compute_covariances(ts)

    at_site = np.where(sites == args.site)[0]
    elsewhere = np.where(sites != args.site)[0]
    print(f"{args.site}: {len(at_site)} participants | elsewhere: {len(elsewhere)}")
    if len(at_site) < args.n_train + args.n_test:
        raise SystemExit(f"{args.site} has too few participants for "
                         f"{args.n_train} train + {args.n_test} test")

    def run(tr_idx, te_idx):
        gt = GPUTangentFC(device="cpu", max_iter=10, ridge=args.ridge, dtype=torch.float64)
        gt.fit(covs[tr_idx])
        Xtr, Xte = _scaled(gt.transform(covs[tr_idx]), gt.transform(covs[te_idx]))
        clf = logreg_factory()().fit(Xtr, y[tr_idx])   # the factory returns a builder
        return roc_auc_score(y[te_idx], clf.predict_proba(Xte)[:, 1])

    rng = np.random.RandomState(args.seed)
    rows = []
    for rep in range(args.repeats):
        # a homogeneous training set, and a matched heterogeneous one
        site_perm = rng.permutation(at_site)
        tr_site, te_site = site_perm[:args.n_train], site_perm[args.n_train:args.n_train + args.n_test]
        tr_mixed = rng.choice(elsewhere, args.n_train, replace=False)
        te_mixed = rng.choice(np.setdiff1d(elsewhere, tr_mixed), args.n_test, replace=False)

        for tr, te, tr_kind, te_kind in (
            (tr_site,  te_site,  "one site", "the same site"),
            (tr_site,  te_mixed, "one site", "other sites"),
            (tr_mixed, te_site,  "many sites", "one unseen site"),
            (tr_mixed, te_mixed, "many sites", "other sites"),
        ):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            try:
                auc = run(tr, te)
            except Exception as exc:
                print(f"  rep {rep} {tr_kind}->{te_kind} failed: "
                      f"{type(exc).__name__}: {exc}")
                continue
            rows.append({"repeat": rep, "train": tr_kind, "test": te_kind,
                         "condition": f"train {tr_kind} -> test {te_kind}",
                         "n_train": len(tr), "n_test": len(te), "auroc": float(auc)})
        if (rep + 1) % 5 == 0:
            print(f"  {rep + 1}/{args.repeats} repeats", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("every condition failed -- nothing to summarise. "
                         "Check the errors printed above rather than the aggregation.")
    summ = (df.groupby(["train", "test", "condition"])["auroc"]
              .agg(["mean", "std", "count"]).reset_index()
              .sort_values("mean", ascending=False))
    print(f"\n{'condition':44s}{'AUROC':>9s}{'sd':>8s}{'n':>5s}")
    print("-" * 66)
    for _, r in summ.iterrows():
        print(f"{r['condition']:44s}{r['mean']:>9.4f}{r['std']:>8.4f}{int(r['count']):>5d}")

    # the contrast that separates the two readings
    def m(tr, te):
        s = df[(df.train == tr) & (df.test == te)]["auroc"]
        return (s.mean(), s) if len(s) else (float("nan"), s)
    same, s1 = m("one site", "the same site")
    trans, s2 = m("one site", "other sites")
    mixed, s3 = m("many sites", "other sites")
    from scipy.stats import ttest_ind
    note = {}
    if len(s2) and len(s3):
        t, p = ttest_ind(s2, s3, equal_var=False)
        note["transfer_difference"] = float(trans - mixed)
        note["transfer_p"] = float(p)
        print(f"\ntesting on other sites, trained on one site vs on many: "
              f"{trans - mixed:+.4f} (p={p:.4f})")
        print("  if training homogeneity carried real information this would be positive;")
        print("  if the within-site advantage was only easier evaluation it would be zero "
              "or negative.")
    note.update({"within_site": float(same), "one_site_to_others": float(trans),
                 "many_sites_to_others": float(mixed),
                 "easier_evaluation_gap": float(same - trans)})
    print(f"\nsame-site testing minus other-site testing, both trained at one site: "
          f"{same - trans:+.4f}")
    print("  that gap is the part attributable to the test set, not the training data.")

    study = Path(args.study)
    t = study / "tables"; t.mkdir(parents=True, exist_ok=True)
    summ.to_csv(t / "homogeneity.csv", index=False)
    df.to_csv(t / "homogeneity_repeats.csv", index=False)
    (study / "raw").mkdir(parents=True, exist_ok=True)
    (study / "raw" / "homogeneity.json").write_text(json.dumps(note, indent=2))
    print("\n->", t / "homogeneity.csv")


if __name__ == "__main__":
    main()
