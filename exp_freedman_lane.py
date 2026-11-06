"""
Freedman--Lane residual permutation for the system-level and regional group contrasts.

The published tests permute the diagnosis labels within site and recompute the statistic on
residuals that were formed once under the nuisance model.  Under that scheme the permuted
labels are independent of age, sex and head motion, whereas the real diagnosis is not, so the
null is not the covariate-adjusted null that the claim requires.

Freedman--Lane instead leaves the design alone and permutes the residuals of the reduced
model:

    reduced   Y = C b + e                     (C: intercept, age, sex, head motion, site)
    permute   Y* = C b_hat + P e              (P permutes rows within site)
    full      Y* = d a + C b + e*             (d: diagnosis, kept in its true position)

so the association between diagnosis and the covariates is preserved in every permutation.

Because the diagnosis column is orthogonalised against C, the fitted coefficient reduces to
a single inner product, a_c = w . e_c with w the normalised residualised diagnosis vector.
The block statistic is a weighted mean of those coefficients, so each permutation costs one
vector-matrix product and the whole null is a single matmul.

Usage:
    .venv/bin/python exp_freedman_lane.py --study results/latest_study [--match_motion]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MISSING = -9999


def build(args):
    """Connectivity, diagnosis, nuisance design - the same inputs as the published tests."""
    from run_benchmark import load_dataset

    roi = pd.read_csv(args.roi_table)
    keep = roi.loc[~roi["excluded"], "roi_id"].to_numpy() - 1
    system = roi.set_index("roi_id")["system"]

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    y = np.asarray(y)
    sites = np.asarray(sites)
    ok = np.array([bool((np.asarray(t)[:, keep].std(axis=0) > 0).all()) for t in ts])
    ts = [ts[i] for i in np.where(ok)[0]]
    y, sites = y[ok], sites[ok]
    sids = [sids[i] for i in np.where(ok)[0]]

    iu = np.triu_indices(len(keep), k=1)
    X = np.stack([np.arctanh(np.clip(np.corrcoef(np.asarray(t)[:, keep].T), -0.999, 0.999))[iu]
                  for t in ts]).astype(np.float64)

    ph = pd.read_csv(args.pheno)
    ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    ph = ph.set_index("key")
    sub = ph.reindex([str(s).split("_")[-1].lstrip("0") for s in sids])

    def clean(col):
        v = pd.to_numeric(sub[col], errors="coerce")
        return v.mask(v <= MISSING + 1).fillna(v[v > MISSING + 1].median()).to_numpy()

    age, sex, fd = clean("AGE_AT_SCAN"), clean("SEX"), clean("func_mean_fd")

    if args.match_motion:
        chosen = []
        for st in np.unique(sites):
            at = np.where((sites == st) & (y == 1))[0]
            ct = list(np.where((sites == st) & (y == 0))[0])
            for i in sorted(at, key=lambda k: fd[k]):
                if not ct:
                    break
                j = min(ct, key=lambda k: abs(fd[k] - fd[i]))
                if abs(fd[j] - fd[i]) <= args.caliper:
                    chosen += [i, j]
                    ct.remove(j)
        sel = np.array(sorted(chosen))
        X, y, sites = X[sel], y[sel], sites[sel]
        age, sex, fd = age[sel], sex[sel], fd[sel]
        print(f"motion-matched: {len(sel)//2} pairs, {len(sel)} participants")

    site_d = pd.get_dummies(pd.Series(sites), drop_first=True).to_numpy(dtype=float)
    C = np.column_stack([np.ones(len(y)), age, sex, fd, site_d])
    return X, y.astype(float), sites, C, roi, keep, system, iu


def freedman_lane(X, y, sites, C, blocks, n_perm, seed):
    """Return observed block statistics and their Freedman--Lane p-values.

    blocks: dict name -> boolean mask over connections.
    """
    # reduced model: residuals carry everything the nuisance variables do not explain
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    e = X - C @ beta
    sd = e.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)

    # diagnosis orthogonalised against the nuisance design
    g, *_ = np.linalg.lstsq(C, y, rcond=None)
    d_perp = y - C @ g
    w = d_perp / (d_perp @ d_perp)                      # a_c = w . e_c

    # each block statistic is a weighted mean of the per-connection coefficients,
    # so it is a single inner product of w with a precomputed block vector
    names = list(blocks)
    E = np.column_stack([np.nanmean(e[:, m] / sd[m], axis=1) for m in blocks.values()])
    obs = w @ E

    rng = np.random.default_rng(seed)
    idx_by_site = [np.flatnonzero(sites == s) for s in np.unique(sites)]
    W = np.empty((n_perm, len(w)))
    for j in range(n_perm):
        perm = np.arange(len(w))
        for idx in idx_by_site:
            perm[idx] = rng.permutation(idx)
        W[j] = w[perm]
    null = W @ E                                        # (n_perm, n_blocks)

    p = (1.0 + (np.abs(null) >= np.abs(obs)).sum(axis=0)) / (n_perm + 1.0)
    return names, obs, p, null


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        prev = min(prev, p[i] * m / (m - rank))
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--roi_table", default="atlas/roi_table.csv")
    ap.add_argument("--pheno", default="abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--match_motion", action="store_true")
    ap.add_argument("--caliper", type=float, default=0.01)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--study", required=True)
    args = ap.parse_args()

    X, y, sites, C, roi, keep, system, iu = build(args)
    print(f"connectivity {X.shape}; {int(y.sum())} autistic of {len(y)}")

    sysname = system.reindex(roi.loc[~roi["excluded"], "roi_id"]).to_numpy()
    si, sj = sysname[iu[0]], sysname[iu[1]]

    # --- system pairs -------------------------------------------------------------------
    sysblocks = {}
    for a in np.unique(sysname):
        for b in np.unique(sysname):
            if a > b:
                continue
            m = ((si == a) & (sj == b)) | ((si == b) & (sj == a))
            if m.sum() >= 20:
                sysblocks[f"{a}|{b}"] = m
    names, obs, p, _ = freedman_lane(X, y, sites, C, sysblocks, args.n_perm, args.seed)
    q = bh(p)
    sysdf = pd.DataFrame(dict(pair=names, effect=obs, p=p, q_fdr=q,
                              n_edges=[int(sysblocks[n].sum()) for n in names]))

    # --- somatomotor regions ------------------------------------------------------------
    roi_keep = roi.loc[~roi["excluded"]].reset_index(drop=True)
    # the published regional effect averages each region's connections WITHIN the
    # somatomotor system, so the same restriction is applied here
    sm = np.flatnonzero(sysname == "sensorimotor")
    in_sm = np.isin(iu[0], sm) & np.isin(iu[1], sm)
    regblocks = {}
    for r in sm:
        m = in_sm & ((iu[0] == r) | (iu[1] == r))
        regblocks[str(roi_keep.loc[r, "roi_id"])] = m
    rnames, robs, rp, _ = freedman_lane(X, y, sites, C, regblocks, args.n_perm, args.seed)
    rq = bh(rp)
    regdf = pd.DataFrame(dict(roi_id=rnames, effect=robs, p=rp, q_fdr=rq))

    out = Path(args.study) / "review"
    out.mkdir(parents=True, exist_ok=True)
    tag = "matched" if args.match_motion else "all"
    sysdf.to_csv(out / f"freedman_lane_systems_{tag}.csv", index=False)
    regdf.to_csv(out / f"freedman_lane_regions_{tag}.csv", index=False)

    print(f"\n== system pairs ({tag}) ==")
    print(f"  {len(sysdf)} pairs; q<0.05: {(sysdf.q_fdr < 0.05).sum()}; "
          f"negative and q<0.05: {((sysdf.q_fdr < 0.05) & (sysdf.effect < 0)).sum()}")
    smr = sysdf[sysdf.pair.str.contains("sensorimotor")]
    print("  somatomotor pairs:")
    for _, r in smr.iterrows():
        print(f"    {r.pair:34s} d={r.effect:+.3f}  p={r.p:.4f}  q={r.q_fdr:.4f}")
    print(f"\n== somatomotor regions ({tag}) ==")
    print(f"  {len(regdf)} regions; q<0.05: {(regdf.q_fdr < 0.05).sum()}; "
          f"mean effect {regdf.effect.mean():+.3f}")
    json.dump({"tag": tag, "n": int(len(y)), "n_perm": args.n_perm,
               "systems_sig": int((sysdf.q_fdr < 0.05).sum()),
               "systems_neg_sig": int(((sysdf.q_fdr < 0.05) & (sysdf.effect < 0)).sum()),
               "regions_sig": int((regdf.q_fdr < 0.05).sum()),
               "region_mean_effect": float(regdf.effect.mean())},
              open(out / f"freedman_lane_summary_{tag}.json", "w"), indent=2)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
