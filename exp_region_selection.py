"""
Does keeping only the most discriminating regions improve screening?

The question is natural and the obvious way of answering it is wrong. If the regions are
chosen by looking at how well they separate the groups, and the model is then scored on
those same participants, the choice has already used the answer. The apparent gain is a
measure of how much information leaked, not of how much the regions help.

Three conditions are therefore compared on identical folds. In the first the regions are
chosen once from everyone, which is the flawed procedure. In the second they are chosen
inside each training partition and applied unchanged to the held-out participants, which is
the honest one. The third keeps every region. The difference between the first two is the
size of the illusion; the difference between the second and third is the real effect.

Usage:
    python exp_region_selection.py --study results/<study>
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

MISSING = -9999.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Region selection, done wrongly and correctly")
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

    import torch
    from sklearn.metrics import roc_auc_score, accuracy_score
    from run_benchmark import load_dataset, _outer_splits
    from run_ablations import logreg_factory, _scaled
    from gpu_tangent import GPUTangentFC
    from evaluation.stat_compare import delong_roc_test

    roi = pd.read_csv(args.roi_table)
    keep_tbl = roi[~roi["excluded"]].reset_index(drop=True)
    keep = keep_tbl["roi_id"].to_numpy() - 1

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    y = np.asarray(y); sites = np.asarray(sites)
    ok = np.array([bool((np.asarray(t)[:, keep].std(axis=0) > 0).all()) for t in ts])
    ts = [np.asarray(ts[i])[:, keep] for i in np.where(ok)[0]]
    y, sites = y[ok], sites[ok]
    sids = [sids[i] for i in np.where(ok)[0]]
    P = len(keep)
    print(f"{len(y)} participants, {P} regions available", flush=True)

    # nuisance-adjusted connectivity, used only for RANKING regions (never for the model)
    iu = np.triu_indices(P, k=1)
    FC = np.stack([np.arctanh(np.clip(np.corrcoef(t.T), -0.999, 0.999))[iu] for t in ts])
    ph = pd.read_csv(args.pheno); ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    ph = ph.set_index("key")
    sub = ph.reindex([str(s).split("_")[-1].lstrip("0") for s in sids])

    def clean(c):
        v = pd.to_numeric(sub[c], errors="coerce")
        return v.mask(v <= MISSING + 1).fillna(v[v > MISSING + 1].median()).to_numpy()

    C = np.column_stack([np.ones(len(y)), clean("AGE_AT_SCAN"), clean("SEX"),
                         clean("func_mean_fd"),
                         pd.get_dummies(pd.Series(sites), drop_first=True).to_numpy(float)])
    beta, *_ = np.linalg.lstsq(C, FC, rcond=None)
    RES = FC - C @ beta

    def rank_regions(rows):
        """Order regions by how strongly their connections separate the groups in `rows`."""
        R, yy = RES[rows], y[rows]
        sd = R.std(axis=0, ddof=1)
        d = np.abs((R[yy == 1].mean(0) - R[yy == 0].mean(0)) / np.where(sd > 0, sd, np.nan))
        score = np.array([np.nanmean(d[(iu[0] == r) | (iu[1] == r)]) for r in range(P)])
        return np.argsort(-score)

    all_rows = np.arange(len(y))
    global_order = rank_regions(all_rows)          # the flawed choice, made once from everyone
    splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)

    def evaluate(select):
        """select(train_rows) -> region indices; None means keep every region."""
        yt, pt = [], []
        for tr, te in splits:
            cols = np.arange(P) if select is None else select(tr)
            gt = GPUTangentFC(device="cpu", max_iter=10, ridge=args.ridge, dtype=torch.float64)
            covs = gt.compute_covariances([t[:, cols] for t in ts])
            gt.fit(covs[tr])
            Xtr, Xte = _scaled(gt.transform(covs[tr]), gt.transform(covs[te]))
            clf = logreg_factory()().fit(Xtr, y[tr])
            yt.extend(y[te].tolist()); pt.extend(clf.predict_proba(Xte)[:, 1].tolist())
        yt, pt = np.asarray(yt), np.asarray(pt)
        return yt, pt, roc_auc_score(yt, pt), accuracy_score(yt, (pt >= 0.5).astype(int))

    rows, oof = [], {}
    yt, pt, auc, acc = evaluate(None)
    oof["all regions"] = (yt, pt)
    rows.append({"condition": "all regions", "k": P, "selection": "none",
                 "auroc": auc, "accuracy": acc})
    print(f"  all {P} regions           AUROC {auc:.4f}  acc {acc:.4f}", flush=True)

    for k in args.top_k:
        if k >= P:
            continue
        yt, pt, auc, acc = evaluate(lambda tr, k=k: global_order[:k])
        oof[f"top {k} (chosen from everyone)"] = (yt, pt)
        rows.append({"condition": f"top {k}", "k": k, "selection": "from everyone (flawed)",
                     "auroc": auc, "accuracy": acc})
        print(f"  top {k:3d} chosen from everyone  AUROC {auc:.4f}  acc {acc:.4f}", flush=True)

        yt, pt, auc, acc = evaluate(lambda tr, k=k: rank_regions(tr)[:k])
        oof[f"top {k} (chosen in-fold)"] = (yt, pt)
        rows.append({"condition": f"top {k}", "k": k, "selection": "inside each fold (honest)",
                     "auroc": auc, "accuracy": acc})
        print(f"  top {k:3d} chosen in-fold        AUROC {auc:.4f}  acc {acc:.4f}", flush=True)

    df = pd.DataFrame(rows)
    base_y, base_p = oof["all regions"]
    tests = []
    for name, (yy, pp) in oof.items():
        if name == "all regions":
            continue
        a, b, pv = delong_roc_test(base_y, pp, base_p)
        tests.append({"condition": name, "auroc": float(a), "vs_all_regions": float(a - b),
                      "delong_p": float(pv)})
    tdf = pd.DataFrame(tests)

    study = Path(args.study)
    t = study / "tables"; t.mkdir(parents=True, exist_ok=True)
    df.to_csv(t / "region_selection.csv", index=False)
    tdf.to_csv(t / "region_selection_tests.csv", index=False)

    print(f"\n{'condition':40s}{'AUROC':>9s}{'vs all':>9s}{'p':>9s}")
    print("-" * 68)
    for _, r in tdf.iterrows():
        print(f"{r['condition'][:39]:40s}{r.auroc:>9.4f}{r.vs_all_regions:>+9.4f}{r.delong_p:>9.4f}")

    for k in args.top_k:
        f = df[(df.k == k) & (df.selection.str.startswith("from"))]
        h = df[(df.k == k) & (df.selection.str.startswith("inside"))]
        if len(f) and len(h):
            print(f"\ntop {k}: choosing from everyone reads "
                  f"{f.auroc.iloc[0] - h.auroc.iloc[0]:+.4f} AUROC higher than choosing "
                  f"honestly — that gap is the leak, not a gain.")
    print("\n->", t / "region_selection.csv")


if __name__ == "__main__":
    main()
