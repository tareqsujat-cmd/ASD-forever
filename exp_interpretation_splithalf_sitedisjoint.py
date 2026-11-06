"""
Reproducibility of the fitted patterns between halves of the cohort that share no acquisition site.

exp_interpretation_splithalf.py divides the participants of every site between two halves, so both halves
contain the same scanners.  Here the 20 sites are divided instead: each half holds 10 sites, so a pattern that
reproduces must do so across sites.  The plan and the success criterion were written down before this run
(ASD-journal-I/TRUSTWORTHINESS_PLAN.md).

Each half is analysed on its own, from the tangent reference and the scaler to the classifier, with the penalty
fixed at the value selected in every outer fold of the primary analysis.  Two variants:

    raw        the classifier and the structure coefficients exactly as in the primary analysis
    adjusted   site indicators (the half's own sites), age, sex and mean framewise displacement are regressed out
               of every standardised feature, within the half, before the classifier is fitted

The null repeats the procedure with the diagnosis permuted within site.

Usage:
    .venv/bin/python exp_interpretation_splithalf_sitedisjoint.py --workers 6 --splits 40 --n-perm 30
    .venv/bin/python exp_interpretation_splithalf_sitedisjoint.py --analyse
"""

from __future__ import annotations

import exp_nested_outer as eno          # sets thread caps before numpy is imported  # isort: skip

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = eno.ROOT
OUT = ROOT / "results" / "latest_study" / "review"
DIR = OUT / "splithalf_sitedisjoint"
SEED = 42
VARIANTS = ("raw", "adjusted")
LEVELS = ["connection", "region", "region (magnitude)", "system pair", "system pair (magnitude)"]
QUANT = ["coef", "struct"]
PHENO = ROOT / "abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"


def site_halves(y, sites, r):
    """Two halves with 10 sites each, sizes within 42-58 per cent and autistic proportions within 0.08."""
    rng = np.random.default_rng(SEED * 6007 + r)
    uniq = np.unique(sites)
    n = len(y)
    for _ in range(200000):
        pick = rng.permutation(uniq)[: len(uniq) // 2]
        m = np.isin(sites, pick)
        a, b = np.flatnonzero(m), np.flatnonzero(~m)
        if 0.42 <= len(a) / n <= 0.58 and abs(y[a].mean() - y[b].mean()) <= 0.08:
            return a, b
    raise RuntimeError("no admissible division of the sites")


def permuted(y, sites, rng):
    yp = y.copy()
    for s in np.unique(sites):
        idx = np.flatnonzero(sites == s)
        yp[idx] = rng.permutation(y[idx])
    return yp


def nuisance():
    """age, sex, mean framewise displacement for the 871 participants (missing values: cohort median)."""
    lab = np.load(eno.CACHE / "labels.npz", allow_pickle=True)
    ph = pd.read_csv(PHENO)
    ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    sub = ph.set_index("key").reindex([str(s).split("_")[-1].lstrip("0") for s in lab["sids"]])
    cols = []
    for c in ("AGE_AT_SCAN", "SEX", "func_mean_fd"):
        v = pd.to_numeric(sub[c], errors="coerce")
        v = v.mask(v < -9000)
        cols.append(v.fillna(v.median()).to_numpy(float))
    return np.column_stack(cols)


def adjust(Z, N_half, sites_half):
    """Regress the half's site indicators, age, sex and head motion out of every standardised feature."""
    S = pd.get_dummies(pd.Series(sites_half), drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(Z)), (N_half - N_half.mean(0)) / (N_half.std(0) + 1e-9), S])
    beta = np.linalg.lstsq(X, Z.astype(np.float64), rcond=None)[0]
    R = Z - X @ beta
    sd = R.std(0)
    return (R / np.where(sd > 0, sd, 1.0)).astype(np.float32)


def fit_structure(Z, yy, c):
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
    N = nuisance()
    blk = ei.system_blocks()
    canon, to_canon_corr = ei.edge_tables()
    keep = blk["block"] < len(blk["names"]) - 1
    A, B = site_halves(y, sites, r)
    rng = np.random.default_rng(SEED * 104729 + r)
    perms = [permuted(y, sites, rng) for _ in range(n_perm)]
    t0 = time.time()
    nv = len(VARIANTS)
    stats = np.full((nv, 3, len(QUANT), len(LEVELS), 1 + n_perm), np.nan)
    sign = np.full((nv, 3, len(QUANT), 3, 1 + n_perm), np.nan)
    pair_pattern = np.zeros((nv, 3, 2, len(blk["names"]) - 1))
    for v, view in enumerate(eno.VIEWS):
        ZA, _ = eno.view_features(view, A, A[:2], D)
        ZB, _ = eno.view_features(view, B, B[:2], D)
        feats = {"raw": (ZA, ZB), "adjusted": (adjust(ZA, N[A], sites[A]), adjust(ZB, N[B], sites[B]))}
        for vi, var in enumerate(VARIANTS):
            Za, Zb = feats[var]
            for k in range(1 + n_perm):
                lab = y if k == 0 else perms[k - 1]
                wa, sa = fit_structure(Za, lab[A], c)
                wb, sb = fit_structure(Zb, lab[B], c)
                for qi, (xa, xb) in enumerate(((wa, wb), (sa, sb))):
                    M = np.stack([ei.to_canonical(xa.astype(np.float64), v, to_canon_corr),
                                  ei.to_canonical(xb.astype(np.float64), v, to_canon_corr)])
                    Ag = ei.aggregate(M, canon, blk)
                    stats[vi, v, qi, 0, k] = ei.pairwise_rho(M[:, keep])[0]
                    stats[vi, v, qi, 1, k] = ei.pairwise_rho(Ag["region_signed"])[0]
                    stats[vi, v, qi, 2, k] = ei.pairwise_rho(Ag["region_abs"])[0]
                    stats[vi, v, qi, 3, k] = ei.pairwise_rho(Ag["block_signed"])[0]
                    stats[vi, v, qi, 4, k] = ei.pairwise_rho(Ag["block_sq"])[0]
                    sign[vi, v, qi, 0, k] = ei.sign_agreement(M[:, keep])
                    sign[vi, v, qi, 1, k] = ei.sign_agreement(Ag["region_signed"])
                    sign[vi, v, qi, 2, k] = ei.sign_agreement(Ag["block_signed"])
                    if k == 0 and qi == 1:
                        pair_pattern[vi, v] = Ag["block_signed"]
        print(f"  [split {r:03d}] {view:18s} done ({time.time() - t0:.0f}s)", flush=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, stats=stats, sign=sign, pair_pattern=pair_pattern, n_A=len(A), n_B=len(B),
                        sites_A=np.unique(sites[A]), sites_B=np.unique(sites[B]),
                        prev_A=y[A].mean(), prev_B=y[B].mean())
    os.replace(tmp, path)
    return f"{path.name} done in {time.time() - t0:.0f}s"


# =========================================================================================
def analyse() -> None:
    files = [f for f in sorted(DIR.glob("split_*.npz")) if ".tmp" not in f.name]
    if not files:
        raise SystemExit("no split files")
    S = np.stack([np.load(f)["stats"] for f in files])          # (splits, variant, view, quantity, level, 1+P)
    G = np.stack([np.load(f)["sign"] for f in files])
    n_perm = S.shape[-1] - 1
    rows = []
    for vi, var in enumerate(VARIANTS):
        for v, vn in enumerate(eno.VIEWS):
            for qi, q in enumerate(QUANT):
                for li, lev in enumerate(LEVELS):
                    obs = S[:, vi, v, qi, li, 0]
                    null = S[:, vi, v, qi, li, 1:].mean(0)
                    row = dict(variant=var, view=vn, quantity=q, level=lev, n_splits=len(files),
                               observed_rho=float(obs.mean()), observed_sd=float(obs.std(ddof=1)),
                               observed_min=float(obs.min()), null_mean=float(null.mean()),
                               null_sd=float(null.std(ddof=1)), null_p95=float(np.percentile(null, 95)),
                               null_max=float(null.max()), n_perm=n_perm,
                               p_value=float((1 + (null >= obs.mean()).sum()) / (1 + n_perm)))
                    if li in (0, 1, 3):
                        gi = {0: 0, 1: 1, 3: 2}[li]
                        row["sign_agreement"] = float(G[:, vi, v, qi, gi, 0].mean())
                        row["sign_agreement_null"] = float(G[:, vi, v, qi, gi, 1:].mean())
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "interpretation_splithalf_sitedisjoint.csv", index=False)
    crit = {}
    for var in VARIANTS:
        d = df[(df.variant == var) & (df.quantity == "struct") & (df.level == "system pair")]
        passed = int((d.observed_rho > d.null_p95).sum())
        crit[var] = dict(views_exceeding_null_p95=passed, criterion_met=bool(passed >= 2),
                         observed=dict(zip(d.view, d.observed_rho.round(3))), null_p95=dict(zip(d.view, d.null_p95.round(3))))
    sizes = [(int(np.load(f)["n_A"]), int(np.load(f)["n_B"])) for f in files]
    crit["splits"] = len(files)
    crit["half_sizes_mean"] = [float(np.mean([s[0] for s in sizes])), float(np.mean([s[1] for s in sizes]))]
    (OUT / "sitedisjoint_summary.json").write_text(json.dumps(crit, indent=1))
    print(json.dumps(crit, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--splits", type=int, default=40)
    ap.add_argument("--n-perm", type=int, default=30)
    ap.add_argument("--c", type=float, default=0.001)
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.analyse:
        analyse()
        return
    DIR.mkdir(parents=True, exist_ok=True)
    chosen = {float(c) for f in eno.OUT.glob("fold_pooled_*.npz") if ".tmp" not in f.name
              for c in np.load(f)["c_best"]}
    if chosen and chosen != {a.c}:
        raise SystemExit(f"outer folds selected {sorted(chosen)}, not only {a.c}")
    t0 = time.time()
    eno.run_parallel([(r, one_split, (r, a.n_perm, a.c)) for r in range(a.splits)], a.workers,
                     lambda r, msg, mins: print(f"{time.strftime('%H:%M:%S')} split {r:03d}: {msg} "
                                                f"(elapsed {mins:.1f} min)", flush=True))
    (DIR / "run.json").write_text(json.dumps(dict(splits=a.splits, n_perm=a.n_perm, c=a.c, seed=SEED,
                                                   minutes=(time.time() - t0) / 60)))


if __name__ == "__main__":
    main()
