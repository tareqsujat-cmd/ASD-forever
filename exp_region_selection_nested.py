"""
Region selection with the nuisance model refitted inside each outer-training partition.

In the published training-fold condition the regions are ranked inside each fold, but the
nuisance adjustment used for that ranking is fitted once on the complete analysis cohort.  The
held-out participants therefore contribute to the residuals from which the ranking is derived,
so the condition is not a strictly nested comparator.

Here the nuisance coefficients are estimated from the outer-training rows alone and applied to
those rows only, so every data-dependent step of the selection sees training participants and
nothing else.  Folds, features, tangent projection, scaling and classifier are unchanged, so the
difference from the published condition is attributable to the nuisance fit.

Usage:
    .venv/bin/python exp_region_selection_nested.py --study results/latest_study
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

MISSING = -9999


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--roi_table", default="atlas/roi_table.csv")
    ap.add_argument("--pheno", default="abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--top_k", nargs="+", type=int, default=[10, 20, 30, 50, 100])
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ridge", type=float, default=0.30)
    ap.add_argument("--study", required=True)
    args = ap.parse_args()

    from run_benchmark import load_dataset, _outer_splits
    from run_ablations import logreg_factory, _scaled
    from gpu_tangent import GPUTangentFC

    roi = pd.read_csv(args.roi_table)
    keep = roi.loc[~roi["excluded"], "roi_id"].to_numpy() - 1

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    y = np.asarray(y)
    sites = np.asarray(sites)
    ok = np.array([bool((np.asarray(t)[:, keep].std(axis=0) > 0).all()) for t in ts])
    ts = [np.asarray(ts[i])[:, keep] for i in np.where(ok)[0]]
    y, sites = y[ok], sites[ok]
    sids = [sids[i] for i in np.where(ok)[0]]
    P = len(keep)
    print(f"{len(y)} participants, {P} regions", flush=True)

    iu = np.triu_indices(P, k=1)
    FC = np.stack([np.arctanh(np.clip(np.corrcoef(t.T), -0.999, 0.999))[iu] for t in ts])

    ph = pd.read_csv(args.pheno)
    ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    ph = ph.set_index("key")
    sub = ph.reindex([str(s).split("_")[-1].lstrip("0") for s in sids])

    def clean(c):
        v = pd.to_numeric(sub[c], errors="coerce")
        return v.mask(v <= MISSING + 1).fillna(v[v > MISSING + 1].median()).to_numpy()

    C = np.column_stack([np.ones(len(y)), clean("AGE_AT_SCAN"), clean("SEX"),
                         clean("func_mean_fd"),
                         pd.get_dummies(pd.Series(sites), drop_first=True).to_numpy(float)])

    def rank_in_train(tr):
        """Rank regions using a nuisance model fitted on the training rows alone."""
        b, *_ = np.linalg.lstsq(C[tr], FC[tr], rcond=None)
        R = FC[tr] - C[tr] @ b
        yy = y[tr]
        sd = R.std(axis=0, ddof=1)
        d = np.abs((R[yy == 1].mean(0) - R[yy == 0].mean(0)) / np.where(sd > 0, sd, np.nan))
        score = np.array([np.nanmean(d[(iu[0] == r) | (iu[1] == r)]) for r in range(P)])
        return np.argsort(-score)

    splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)

    def evaluate(select):
        yt, pt = [], []
        for tr, te in splits:
            cols = np.arange(P) if select is None else select(tr)
            gt = GPUTangentFC(device="cpu", max_iter=10, ridge=args.ridge,
                              dtype=torch.float64)
            covs = gt.compute_covariances([t[:, cols] for t in ts])
            gt.fit(covs[tr])
            Xtr, Xte = _scaled(gt.transform(covs[tr]), gt.transform(covs[te]))
            clf = logreg_factory()().fit(Xtr, y[tr])
            yt.extend(y[te].tolist())
            pt.extend(clf.predict_proba(Xte)[:, 1].tolist())
        yt, pt = np.asarray(yt), np.asarray(pt)
        return roc_auc_score(yt, pt), accuracy_score(yt, (pt >= 0.5).astype(int))

    rows = []
    auc, acc = evaluate(None)
    rows.append(dict(condition="all regions", k=P, selection="none", auroc=auc, accuracy=acc))
    print(f"  all {P} regions                      AUROC {auc:.4f}  acc {acc:.4f}", flush=True)

    for k in args.top_k:
        if k >= P:
            continue
        auc, acc = evaluate(lambda tr, k=k: rank_in_train(tr)[:k])
        rows.append(dict(condition=f"top {k}", k=k,
                         selection="inside each fold, nuisance refitted in-fold",
                         auroc=auc, accuracy=acc))
        print(f"  top {k:3d} in-fold, nested nuisance    AUROC {auc:.4f}  acc {acc:.4f}",
              flush=True)

    out = Path(args.study) / "review"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "region_selection_nested.csv", index=False)

    # compare with the published training-fold condition
    pub = pd.read_csv(Path(args.study) / "tables" / "region_selection.csv")
    pub = pub[pub.selection.str.startswith("inside each fold")]
    merged = df[df.k < P].merge(pub[["k", "auroc"]], on="k", suffixes=("_nested", "_published"))
    merged["delta"] = merged.auroc_nested - merged.auroc_published
    merged.to_csv(out / "region_selection_nested_vs_published.csv", index=False)
    print("\n== nested nuisance versus the published training-fold condition ==")
    for _, r in merged.iterrows():
        print(f"  top {int(r.k):3d}  nested {r.auroc_nested:.4f}  published "
              f"{r.auroc_published:.4f}  delta {r.delta:+.4f}")
    print(f"  mean |delta| = {merged.delta.abs().mean():.4f}")
    json.dump({"mean_abs_delta": float(merged.delta.abs().mean()),
               "max_abs_delta": float(merged.delta.abs().max()),
               "all_regions_auroc": float(df.loc[df.condition == "all regions", "auroc"].iloc[0])},
              open(out / "region_selection_nested_summary.json", "w"), indent=2)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
