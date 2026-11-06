"""
Replication of the fitted patterns between independent halves of the cohort.

The agreement between outer training partitions, computed in exp_interpretation.py, compares
fits that share most of their participants.  Such fits agree closely even when the labels are
noise, because a regularised linear model fitted to overlapping data with the same labels
returns nearly the same weights.  Replication is therefore judged here on disjoint halves.

The cohort is divided R times into two halves of nearly equal size, stratified by site and
diagnosis.  Each half is analysed on its own, from the tangent reference and the scaler to
the classifier, with the penalty fixed at the value that was selected in every outer fold of
the real analysis.  The weights and the structure coefficients of the two halves are compared
at the level of connections, regions and system pairs.  The same procedure is repeated with
the diagnosis permuted within site, which gives the agreement expected when the labels carry
no information about connectivity beyond the prevalence of each site.

Usage:
    .venv/bin/python exp_interpretation_splithalf.py --workers 4 --splits 50 --n-perm 40
    .venv/bin/python exp_interpretation_splithalf.py --analyse
"""

from __future__ import annotations

import exp_nested_outer as eno          # sets thread caps before numpy is imported  # isort: skip

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = eno.ROOT
OUT = ROOT / "results" / "latest_study" / "review"
DIR = OUT / "splithalf"
SEED = 42
LEVELS = ["connection", "region", "region (magnitude)", "system pair", "system pair (magnitude)"]
QUANT = ["coef", "struct"]


def halves(y, sites, r):
    """Two disjoint halves, stratified by site and diagnosis."""
    rng = np.random.default_rng(SEED * 7919 + r)
    a, b = [], []
    for s in np.unique(sites):
        for c in (0, 1):
            idx = np.flatnonzero((sites == s) & (y == c))
            idx = rng.permutation(idx)
            a.append(idx[::2])
            b.append(idx[1::2])
    return np.sort(np.concatenate(a)), np.sort(np.concatenate(b))


def permuted(y, sites, rng):
    yp = y.copy()
    for s in np.unique(sites):
        idx = np.flatnonzero(sites == s)
        yp[idx] = rng.permutation(y[idx])
    return yp


def fit_structure(Z, yy, c):
    """Weights and structure coefficients of one l2-logistic fit on standardised features."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=c, solver="liblinear", max_iter=2000).fit(Z, yy)
    w, b = clf.coef_[0], float(clf.intercept_[0])
    Z64 = np.asarray(Z, np.float64)
    s = Z64 @ w + b
    return w, (Z64 - Z64.mean(0)).T @ (s - s.mean()) / (len(yy) * s.std())


def one_split(r: int, n_perm: int, c: float) -> str:
    import exp_interpretation as ei

    path = DIR / f"split_{r:03d}.npz"
    if path.exists():
        return f"{path.name} exists"
    D = eno.data()
    y, sites = D.y, D.sites
    blk = ei.system_blocks()
    canon, to_canon_corr = ei.edge_tables()
    keep = blk["block"] < len(blk["names"]) - 1
    A, B = halves(y, sites, r)
    rng = np.random.default_rng(SEED * 104729 + r)
    perms = [permuted(y, sites, rng) for _ in range(n_perm)]
    t0 = time.time()

    # stats[v, q, level, 0] is the real-label agreement; 1..n_perm the permuted-label agreement
    stats = np.full((3, len(QUANT), len(LEVELS), 1 + n_perm), np.nan)
    sign = np.full((3, len(QUANT), 3, 1 + n_perm), np.nan)        # connection, region, pair
    pair_pattern = np.zeros((3, 2, len(blk["names"]) - 1))        # real-label pair structure
    for v, view in enumerate(eno.VIEWS):
        ZA, _ = eno.view_features(view, A, A[:2], D)
        ZB, _ = eno.view_features(view, B, B[:2], D)
        for k in range(1 + n_perm):
            lab = y if k == 0 else perms[k - 1]
            wa, sa = fit_structure(ZA, lab[A], c)
            wb, sb = fit_structure(ZB, lab[B], c)
            for qi, (xa, xb) in enumerate(((wa, wb), (sa, sb))):
                M = np.stack([ei.to_canonical(xa.astype(np.float64), v, to_canon_corr),
                              ei.to_canonical(xb.astype(np.float64), v, to_canon_corr)])
                Ag = ei.aggregate(M, canon, blk)
                stats[v, qi, 0, k] = ei.pairwise_rho(M[:, keep])[0]
                stats[v, qi, 1, k] = ei.pairwise_rho(Ag["region_signed"])[0]
                stats[v, qi, 2, k] = ei.pairwise_rho(Ag["region_abs"])[0]
                stats[v, qi, 3, k] = ei.pairwise_rho(Ag["block_signed"])[0]
                stats[v, qi, 4, k] = ei.pairwise_rho(Ag["block_sq"])[0]
                sign[v, qi, 0, k] = ei.sign_agreement(M[:, keep])
                sign[v, qi, 1, k] = ei.sign_agreement(Ag["region_signed"])
                sign[v, qi, 2, k] = ei.sign_agreement(Ag["block_signed"])
                if k == 0 and qi == 1:
                    pair_pattern[v] = Ag["block_signed"]
        print(f"  [split {r:03d}] {view:18s} done ({time.time() - t0:.0f}s)", flush=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, stats=stats, sign=sign, pair_pattern=pair_pattern,
                        n_A=len(A), n_B=len(B))
    os.replace(tmp, path)
    return f"{path.name} done in {time.time() - t0:.0f}s"


# =========================================================================================
def analyse() -> None:
    files = [f for f in sorted(DIR.glob("split_*.npz")) if ".tmp" not in f.name]
    if not files:
        raise SystemExit("no split files")
    S = np.stack([np.load(f)["stats"] for f in files])            # (splits, 3, 2, 5, 1+P)
    G = np.stack([np.load(f)["sign"] for f in files])
    n_perm = S.shape[-1] - 1
    rows = []
    for v, vn in enumerate(eno.VIEWS):
        for qi, q in enumerate(QUANT):
            for li, lev in enumerate(LEVELS):
                obs = S[:, v, qi, li, 0]
                null = S[:, v, qi, li, 1:].mean(0)                # split-averaged, one per permutation
                row = dict(view=vn, quantity=q, level=lev, n_splits=len(files),
                           observed_rho=float(obs.mean()), observed_sd=float(obs.std(ddof=1)),
                           observed_min=float(obs.min()),
                           null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                           null_p95=float(np.percentile(null, 95)), null_max=float(null.max()),
                           n_perm=n_perm,
                           p_value=float((1 + (null >= obs.mean()).sum()) / (1 + n_perm)))
                if li in (0, 1, 3):
                    gi = {0: 0, 1: 1, 3: 2}[li]
                    row["sign_agreement"] = float(G[:, v, qi, gi, 0].mean())
                    row["sign_agreement_null"] = float(G[:, v, qi, gi, 1:].mean())
                rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "interpretation_splithalf.csv", index=False)
    pat = np.stack([np.load(f)["pair_pattern"] for f in files])   # (splits, 3, 2, 36)
    np.savez(OUT / "interpretation_splithalf_patterns.npz", pair_pattern=pat)
    print(f"{len(files)} splits, {n_perm} permutations -> interpretation_splithalf.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--n-perm", type=int, default=40)
    ap.add_argument("--c", type=float, default=0.001)
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.analyse:
        analyse()
        return
    DIR.mkdir(parents=True, exist_ok=True)
    # the penalty is fixed at the value selected in every outer fold of the real analysis
    chosen = {float(c) for f in eno.OUT.glob("fold_pooled_*.npz") if ".tmp" not in f.name
              for c in np.load(f)["c_best"]}
    if chosen and chosen != {a.c}:
        raise SystemExit(f"outer folds selected {sorted(chosen)}, not only {a.c}")
    t0 = time.time()
    eno.run_parallel([(r, one_split, (r, a.n_perm, a.c)) for r in range(a.splits)], a.workers,
                     lambda r, msg, mins: print(f"{time.strftime('%H:%M:%S')} split {r:03d}: {msg} "
                                                f"(elapsed {mins:.1f} min)", flush=True))
    (DIR / "run.json").write_text(json.dumps(dict(splits=a.splits, n_perm=a.n_perm, c=a.c,
                                                   seed=SEED, minutes=(time.time() - t0) / 60)))


if __name__ == "__main__":
    main()
