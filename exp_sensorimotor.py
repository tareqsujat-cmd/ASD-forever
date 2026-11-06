"""
Inside the sensorimotor system: which regions carry the difference, and does the ranking hold?

The system-level contrast places the largest difference within the sensorimotor network.
That block averages over two hundred connections, so it says nothing about which parts of
the system are responsible. Twenty-one regions is a small enough family to ask, provided
the answer is checked rather than assumed: the same data is split in half repeatedly and
the two halves' region rankings compared, which measures directly how much of any ordering
would survive collecting a second cohort.

The same nuisance adjustment as the system-level analysis is applied — acquisition site,
head motion, age and sex — and significance again comes from permuting diagnosis within
site.

Usage:
    python exp_sensorimotor.py --study results/<study> --system sensorimotor
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
    ap = argparse.ArgumentParser(description="Region-level detail inside one system")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--roi_table", default="atlas/roi_table.csv")
    ap.add_argument("--pheno", default="abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--system", default="sensorimotor")
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--n_splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--match_motion", action="store_true",
                    help="restrict to within-site pairs matched on mean FD")
    ap.add_argument("--caliper", type=float, default=0.02)
    ap.add_argument("--tag", default="")
    ap.add_argument("--study", required=True)
    args = ap.parse_args()

    from run_benchmark import load_dataset
    from scipy.stats import false_discovery_control, spearmanr

    roi = pd.read_csv(args.roi_table)
    keep_tbl = roi[~roi["excluded"]].reset_index(drop=True)
    keep = keep_tbl["roi_id"].to_numpy() - 1
    in_sys = (keep_tbl["system"] == args.system).to_numpy()
    n_sys = int(in_sys.sum())
    print(f"{args.system}: {n_sys} regions of {len(keep_tbl)} retained")
    if n_sys < 5:
        raise SystemExit("too few regions in that system")

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    y = np.asarray(y); sites = np.asarray(sites)
    ok = np.array([bool((np.asarray(t)[:, keep].std(axis=0) > 0).all()) for t in ts])
    ts = [ts[i] for i in np.where(ok)[0]]
    y, sites = y[ok], sites[ok]
    sids = [sids[i] for i in np.where(ok)[0]]
    print(f"{len(y)} participants ({int(y.sum())} autistic)")

    # connectivity restricted to this system
    sys_cols = keep[in_sys]
    iu = np.triu_indices(n_sys, k=1)
    X = np.stack([np.arctanh(np.clip(np.corrcoef(np.asarray(t)[:, sys_cols].T), -0.999, 0.999))[iu]
                  for t in ts])
    print(f"{X.shape[1]} connections within the system")

    ph = pd.read_csv(args.pheno)
    ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    ph = ph.set_index("key")
    sub = ph.reindex([str(s).split("_")[-1].lstrip("0") for s in sids])

    def clean(c):
        v = pd.to_numeric(sub[c], errors="coerce")
        return v.mask(v <= MISSING + 1).fillna(v[v > MISSING + 1].median()).to_numpy()

    age, sex, fd = clean("AGE_AT_SCAN"), clean("SEX"), clean("func_mean_fd")

    if args.match_motion:
        # The regions carrying this effect sit along the central sulcus, which is where head
        # motion writes its signature, and autistic participants move more.  Regressing FD
        # out assumes the relation is linear in one scalar; pairing each autistic participant
        # with a control scanned at the same site and moving almost exactly as much removes
        # the difference by design instead.
        from scipy.stats import mannwhitneyu
        chosen = []
        for st in np.unique(sites):
            at = np.where((sites == st) & (y == 1))[0]
            ct = list(np.where((sites == st) & (y == 0))[0])
            for i in sorted(at, key=lambda k: fd[k]):
                if not ct:
                    break
                j = min(ct, key=lambda k: abs(fd[k] - fd[i]))
                if abs(fd[j] - fd[i]) <= args.caliper:
                    chosen += [i, j]; ct.remove(j)
        sel = np.array(sorted(chosen))
        if len(sel) < 100:
            raise SystemExit(f"only {len(sel)//2} pairs within caliper {args.caliper}")
        before = mannwhitneyu(fd[y == 1], fd[y == 0]).pvalue
        X, y, sites = X[sel], y[sel], sites[sel]
        age, sex, fd = age[sel], sex[sel], fd[sel]
        ts = [ts[i] for i in sel]
        after = mannwhitneyu(fd[y == 1], fd[y == 0]).pvalue
        print(f"motion-matched: {len(sel)//2} pairs, {len(sel)} participants "
              f"({int(y.sum())} autistic)")
        print(f"  head-motion difference between groups: p={before:.2e} before matching, "
              f"p={after:.3f} after", flush=True)

    C = np.column_stack([np.ones(len(y)), age, sex, fd,
                         pd.get_dummies(pd.Series(sites), drop_first=True).to_numpy(float)])
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    R = X - C @ beta
    sd = R.std(axis=0, ddof=1)

    names = keep_tbl.loc[in_sys, "display_name"].to_numpy()
    # parcels whose centroid sits on the midline get "M": forcing them to a side is where
    # the hemisphere column and the atlas name disagree
    cx = keep_tbl.loc[in_sys, "com_x"].to_numpy()
    hemi = np.where(np.abs(cx) < 5.0, "M", np.where(cx < 0, "L", "R"))

    def region_effect(labels):
        """Mean standardised difference over each region's connections within the system."""
        d = (R[labels == 1].mean(0) - R[labels == 0].mean(0)) / np.where(sd > 0, sd, np.nan)
        out = np.zeros(n_sys)
        for r in range(n_sys):
            m = (iu[0] == r) | (iu[1] == r)
            out[r] = np.nanmean(d[m])
        return out, d

    obs, edge_d = region_effect(y)

    # permute diagnosis within site
    rng = np.random.RandomState(args.seed)
    null = np.zeros((args.n_perm, n_sys))
    for i in range(args.n_perm):
        perm = y.copy()
        for s in np.unique(sites):
            m = sites == s
            perm[m] = rng.permutation(perm[m])
        null[i], _ = region_effect(perm)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{args.n_perm} permutations", flush=True)

    p = (1 + (np.abs(null) >= np.abs(obs)).sum(axis=0)) / (1 + args.n_perm)
    q = false_discovery_control(p)

    # would this ranking survive a second cohort? split the data and compare orderings.
    # nuisance coefficients are refit inside each half: fitting them once on everyone and
    # reusing them couples the halves and inflates the agreement.
    print(f"\nchecking reproducibility over {args.n_splits} random splits ...", flush=True)
    idx = np.arange(len(y))
    splits = []
    rng = np.random.RandomState(args.seed + 1)
    for _ in range(args.n_splits):
        perm = rng.permutation(idx)
        splits.append((perm[: len(idx) // 2], perm[len(idx) // 2:]))

    def half_effects(Xb, iu_b, n_reg, half):
        """Standardised group difference inside one half, residualised on that half alone."""
        Ch, Xh, yh = C[half], Xb[half], y[half]
        Rh = Xh - Ch @ np.linalg.lstsq(Ch, Xh, rcond=None)[0]
        sdh = Rh.std(axis=0, ddof=1)
        dh = (Rh[yh == 1].mean(0) - Rh[yh == 0].mean(0)) / np.where(sdh > 0, sdh, np.nan)
        reg = np.array([np.nanmean(dh[(iu_b[0] == r) | (iu_b[1] == r)]) for r in range(n_reg)])
        return reg, dh

    def reproducibility(Xb, iu_b, n_reg, use_splits, edges=False):
        """Mean split-half rank agreement of the region ranking (and optionally the edges)."""
        reg_r, edge_r, top_r = [], [], []
        for a, b in use_splits:
            ra, da = half_effects(Xb, iu_b, n_reg, a)
            rb, db = half_effects(Xb, iu_b, n_reg, b)
            reg_r.append(spearmanr(ra, rb).statistic)
            if edges:
                edge_r.append(spearmanr(da, db).statistic)
                sel = np.argsort(-np.abs(da))[:min(50, len(da))]
                top_r.append(spearmanr(da[sel], db[sel]).statistic)
        return (float(np.nanmean(reg_r)),
                float(np.nanmean(edge_r)) if edges else float("nan"),
                float(np.nanmean(top_r)) if edges else float("nan"))

    rho, rho_edges, rho_edges_top = reproducibility(X, iu, n_sys, splits, edges=True)

    # is that agreement particular to this system, or would any block of the same size
    # reproduce as well? the honest comparator is a cardinality-matched null.
    n_null, null_splits = 100, splits[: min(50, len(splits))]
    print(f"comparing against {n_null} random {n_sys}-region blocks ...", flush=True)
    iu_n = np.triu_indices(n_sys, k=1)
    rng_b = np.random.RandomState(args.seed + 2)
    null_rho = []
    for j in range(n_null):
        cols = keep[rng_b.choice(len(keep_tbl), n_sys, replace=False)]
        Xb = np.stack([np.arctanh(np.clip(np.corrcoef(np.asarray(t)[:, cols].T),
                                          -0.999, 0.999))[iu_n] for t in ts])
        null_rho.append(reproducibility(Xb, iu_n, n_sys, null_splits)[0])
        if (j + 1) % 25 == 0:
            print(f"  {j+1}/{n_null} blocks", flush=True)
    null_rho = np.array(null_rho, dtype=float)
    null_med = float(np.nanmedian(null_rho))
    null_pct = float((null_rho >= rho).mean())

    df = pd.DataFrame({
        "region": names, "hemisphere": hemi,
        "roi_id": keep_tbl.loc[in_sys, "roi_id"].to_numpy(),
        "effect": obs, "z": obs / null.std(axis=0), "p": p, "q_fdr": q,
    }).sort_values("effect")

    study = Path(args.study)
    t = study / "tables"; t.mkdir(parents=True, exist_ok=True)
    df.to_csv(t / f"{args.system.replace(' ','_')}_regions{args.tag}.csv", index=False)

    print(f"\n{'region':46s}{'hem':>5s}{'effect':>9s}{'z':>7s}{'q':>8s}")
    print("-" * 78)
    for _, r in df.head(12).iterrows():
        print(f"{str(r.region)[:45]:46s}{r.hemisphere:>5s}{r.effect:>+9.4f}{r.z:>7.2f}{r.q_fdr:>8.4f}")
    n_sig = int((df.q_fdr < 0.05).sum())
    print(f"\n{n_sig} of {n_sys} regions significant after correction")
    print(f"how concentrated: strongest region accounts for "
          f"{abs(df.effect.iloc[0]) / abs(df.effect).sum() * 100:.1f}% of the total effect")
    print(f"left vs right mean effect: "
          f"{df[df.hemisphere=='L'].effect.mean():+.4f} / {df[df.hemisphere=='R'].effect.mean():+.4f}")
    print(f"\nSPLIT-HALF REPRODUCIBILITY, all measured in the same {args.n_splits} splits:")
    print(f"  ranking of the {n_sys} regions in this system : rho = {rho:.3f}")
    print(f"  ranking of all {len(iu[0])} connections within it: rho = {rho_edges:.3f}")
    print(f"  ranking of its top-50 connections           : rho = {rho_edges_top:.3f}")
    print(f"\n  random {n_sys}-region blocks, same splits: median rho = {null_med:.3f} "
          f"(range {np.nanmin(null_rho):.3f}-{np.nanmax(null_rho):.3f}); "
          f"{null_pct * 100:.0f}% reach this system's {rho:.3f}")

    (study / "raw").mkdir(parents=True, exist_ok=True)
    (study / "raw" / f"{args.system.replace(' ','_')}_regions{args.tag}.json").write_text(json.dumps(
        {"system": args.system, "n_regions": n_sys, "n_significant": n_sig,
         "split_half_rho": rho, "split_half_rho_all_edges": rho_edges,
         "split_half_rho_top50_edges": rho_edges_top,
         "null_block_rho_median": null_med, "null_block_rho_pct_ge": null_pct,
         "null_block_rho_min": float(np.nanmin(null_rho)),
         "null_block_rho_max": float(np.nanmax(null_rho)), "n_null_blocks": n_null,
         "n_splits": int(args.n_splits), "n_participants": int(len(y))}, indent=2))
    print("\n->", t / f"{args.system.replace(' ','_')}_regions{args.tag}.csv")


if __name__ == "__main__":
    main()
