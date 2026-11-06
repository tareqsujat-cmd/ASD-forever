"""
Interpretation of the linear members of the primary ensemble at the level of functional
systems, computed from the outer-fold refits written by exp_nested_outer.py.

Each connectivity view of the ensemble is an l2-logistic model, so its score is the sum of
per-connection terms w_j z_ij plus an intercept, and the score can be split exactly by any
grouping of connections.  The connections are grouped here into the 36 pairs of the eight
functional systems used for the group-difference analyses.  Four things are computed.

1. Held-out contributions.  For every held-out participant the score of the model trained
   without them is split into the contribution of each system pair.  The contribution
   difference between autistic and non-autistic participants sums over pairs to the model's
   mean score difference, so it apportions the model's separation of the groups; the AUROC
   of a single pair's contribution shows how much a pair separates the groups on its own.
2. Reproducibility.  Weights and structure coefficients (the correlation of every connection
   with the model score) are compared across the outer training partitions at the level of
   connections, regions and system pairs.
3. Agreement across the three views of the ensemble.
4. Agreement with the nuisance-adjusted group-difference analysis.

Everything is computed on held-out data or on training-partition fits only.

Usage:
    .venv/bin/python exp_interpretation.py [--reps 2000]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
STUDY = ROOT / "results" / "latest_study"
NESTED = STUDY / "review" / "nested"
OUT = STUDY / "review"
SEED = 42
P = 200
VIEWS = ("correlation", "tangent (static)", "dynamic: mean")
SYSTEM_ORDER = ["visual", "sensorimotor", "dorsal attention", "salience / ventral attention",
                "limbic", "frontoparietal control", "default mode", "subcortical / cerebellar"]


# =========================================================================================
# edge bookkeeping
# =========================================================================================
def edge_tables():
    """Region pair of every feature for each view, and the canonical (upper-triangle) order.

    The tangent views are vectorised over the strict upper triangle in row-major order; the
    correlation view is vectorised by nilearn over the strict lower triangle in row-major
    order, which is a different permutation of the same 19,900 pairs.
    """
    iu = np.triu_indices(P, 1)
    il = np.tril_indices(P, -1)
    canon = np.stack(iu, 1)                                       # (19900, 2), i < j
    lower = np.stack([il[1], il[0]], 1)                           # (i < j) of each lower entry
    pos = {(i, j): n for n, (i, j) in enumerate(canon)}
    to_canon_corr = np.array([pos[(i, j)] for i, j in lower])     # corr feature n -> canonical
    return canon, to_canon_corr


def system_blocks():
    roi = pd.read_csv(ROOT / "atlas" / "roi_table.csv")
    system = roi["system"].to_numpy()
    excluded = roi["excluded"].to_numpy().astype(bool)
    sysidx = np.array([SYSTEM_ORDER.index(s) for s in system])
    pairs = [(a, b) for a in range(8) for b in range(a, 8)]       # 36 unordered pairs
    pid = {p: n for n, p in enumerate(pairs)}
    canon, _ = edge_tables()
    block = np.full(len(canon), len(pairs), int)                  # last block = unassigned
    for n, (i, j) in enumerate(canon):
        if excluded[i] or excluded[j]:
            continue
        a, b = sorted((sysidx[i], sysidx[j]))
        block[n] = pid[(a, b)]
    names = [f"{SYSTEM_ORDER[a]} | {SYSTEM_ORDER[b]}" for a, b in pairs] + ["unassigned"]
    return dict(canon=canon, block=block, pairs=pairs, names=names, sysidx=sysidx,
                excluded=excluded, system=system)


def to_canonical(arr, view_index, to_canon_corr):
    """Reorder the last axis of a per-view array into the canonical edge order."""
    if view_index != 0:
        return arr
    out = np.empty_like(arr)
    out[..., to_canon_corr] = arr
    return out


# =========================================================================================
# data
# =========================================================================================
def load_protocol(protocol):
    files = [f for f in sorted(NESTED.glob(f"fold_{protocol}_*.npz")) if ".tmp" not in f.name]
    if not files:
        raise SystemExit(f"no folds for {protocol}")
    return [np.load(f) for f in files]


def cluster_boot_idx(site, reps, rng):
    sites = np.unique(site)
    by = {s: np.flatnonzero(site == s) for s in sites}
    for _ in range(reps):
        yield np.concatenate([by[s] for s in rng.choice(sites, sites.size, replace=True)])


def auroc_cols(Pm, y):
    """AUROC of every column of Pm against y, by ranks (ties get average ranks)."""
    r = rankdata(Pm, axis=0)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


# =========================================================================================
# 1  held-out contributions
# =========================================================================================
def contributions(protocol, folds, blk, lab, reps):
    y_all, site_all = lab["y"].astype(int), lab["sites"]
    canon, to_canon_corr = edge_tables()
    nb = len(blk["names"])
    onehot = np.zeros((len(canon), nb), np.float64)
    onehot[np.arange(len(canon)), blk["block"]] = 1.0

    parts = [[] for _ in VIEWS]
    order, err = [], 0.0
    for f in folds:
        te = f["te_idx"]
        order.append(te)
        for v in range(3):
            w = to_canonical(f["coef"][v].astype(np.float64), v, to_canon_corr)
            Z = to_canonical(f["z_te"][v].astype(np.float64), v, to_canon_corr)
            parts[v].append(Z @ (onehot * w[:, None]))                    # (n_te, nb)
            s = Z @ w + f["intercept"][v]
            err = max(err, float(np.abs(1 / (1 + np.exp(-s)) - f["p_raw"][v]).max()))
    order = np.concatenate(order)
    y, site = y_all[order], site_all[order]
    C = np.stack([np.concatenate(p) for p in parts])                      # (3, n, nb)
    C = np.concatenate([C, C.mean(0, keepdims=True)])                     # + view average
    names = list(VIEWS) + ["mean of views"]

    # exact-additivity check: the contributions sum to the held-out score minus the intercept
    add_err = 0.0
    off = 0
    for f in folds:
        n = len(f["te_idx"])
        for v in range(3):
            w = to_canonical(f["coef"][v].astype(np.float64), v, to_canon_corr)
            Z = to_canonical(f["z_te"][v].astype(np.float64), v, to_canon_corr)
            add_err = max(add_err, float(np.abs(C[v, off:off + n].sum(1) - Z @ w).max()))
        off += n

    def stats(idx):
        yy, CC, ss = y[idx], C[:, idx, :], site[idx]
        d = CC[:, yy == 1].mean(1) - CC[:, yy == 0].mean(1)               # (4, nb)
        a = np.stack([auroc_cols(CC[v], yy) for v in range(4)])
        # the same difference taken within each site and combined with weights
        # n1 n0 / (n1 + n0), so that differences in prevalence between sites cannot drive it
        num, den = np.zeros_like(d), 0.0
        for u in np.unique(ss):
            m = ss == u
            n1, n0 = int((yy[m] == 1).sum()), int((yy[m] == 0).sum())
            if n1 == 0 or n0 == 0:
                continue
            w = n1 * n0 / (n1 + n0)
            Cm, ym = CC[:, m], yy[m]
            num += w * (Cm[:, ym == 1].mean(1) - Cm[:, ym == 0].mean(1))
            den += w
        return d, a, num / den

    d0, a0, w0 = stats(np.arange(len(y)))
    rng = np.random.default_rng(SEED)
    bd, ba, bw = [], [], []
    for idx in cluster_boot_idx(site, reps, rng):
        if y[idx].min() == y[idx].max():
            continue
        d, a, w_ = stats(idx)
        bd.append(d)
        ba.append(a)
        bw.append(w_)
    bd, ba, bw = np.stack(bd), np.stack(ba), np.stack(bw)
    dlo, dhi = np.percentile(bd, [2.5, 97.5], axis=0)
    alo, ahi = np.percentile(ba, [2.5, 97.5], axis=0)
    wlo, whi = np.percentile(bw, [2.5, 97.5], axis=0)

    rows = []
    for v, vn in enumerate(names):
        total = d0[v].sum()
        for b in range(nb):
            rows.append(dict(protocol=protocol, view=vn, block=blk["names"][b],
                             n_edges=int((blk["block"] == b).sum()),
                             contribution_diff=d0[v, b], diff_lo=dlo[v, b], diff_hi=dhi[v, b],
                             share_of_separation=d0[v, b] / total,
                             total_separation=total,
                             diff_per_connection=d0[v, b] / max(int((blk["block"] == b).sum()), 1),
                             within_site_diff=w0[v, b], within_site_lo=wlo[v, b],
                             within_site_hi=whi[v, b],
                             within_site_share=w0[v, b] / w0[v].sum(),
                             auroc=a0[v, b], auroc_lo=alo[v, b], auroc_hi=ahi[v, b]))
    offs, o = [], 0
    for f in folds:
        offs.append((o, o + len(f["te_idx"])))
        o += len(f["te_idx"])
    extras = dict(C=C, y=y, site=site, offs=offs, onehot=onehot)
    return (pd.DataFrame(rows),
            dict(max_prob_reproduction_error=err, max_additivity_error=add_err, n_boot=len(bd)),
            extras)


# =========================================================================================
# 1b  faithfulness: delete or keep the blocks that the training fit ranks highest
# =========================================================================================
def deletion_test(protocol, folds, blk, ex, n_rand=200):
    """Held-out AUROC after removing (or keeping only) the system pairs ranked highest by
    the training fit, against the same operation on randomly chosen pairs.

    The importance of a pair in a training fit is the share of the variance of the training
    score that its connections carry, sum_j w_j cov(z_j, s) / var(s), which uses the weights
    and the training structure coefficients only.  The held-out score is then recomputed
    from the stored contributions without the removed pairs and without any refitting, so
    this tests what the trained model relies on, not whether any model could do without them.
    The unassigned connections are always retained.
    """
    canon, to_canon_corr = edge_tables()
    nb = len(blk["names"]) - 1
    C, y, site, offs, onehot = ex["C"], ex["y"], ex["site"], ex["offs"], ex["onehot"]

    imp = np.zeros((4, len(folds), nb))                             # per view, fold, pair
    for k, f in enumerate(folds):
        for v in range(3):
            w = to_canonical(f["coef"][v].astype(np.float64), v, to_canon_corr)
            r = to_canonical(f["struct"][v].astype(np.float64), v, to_canon_corr)
            imp[v, k] = (onehot.T @ (w * r))[:nb] / f["score_sd"][v]
    imp[3] = imp[:3].mean(0)

    ms = [0, 1, 2, 3, 4, 6, 8, 12, 18, 24, 30, 35, 36]
    rng = np.random.default_rng(SEED)

    def pooled_auc(mask_by_fold, v):
        # mask_by_fold[k] is a boolean (nb,) vector of the pairs that are retained
        s = np.empty(len(y))
        for k, (a, b) in enumerate(offs):
            keep = np.concatenate([mask_by_fold[k], [True]])
            s[a:b] = C[v, a:b][:, keep].sum(1)
        return float(auroc_cols(s[:, None], y)[0])

    rows = []
    for v, vn in enumerate(list(VIEWS) + ["mean of views"]):
        ranks = [np.argsort(-imp[v, k]) for k in range(len(folds))]
        for m in ms:
            for mode in ("delete top", "keep only top"):
                def mask_for(order_by_fold):
                    out = []
                    for k in range(len(folds)):
                        sel = np.zeros(nb, bool)
                        sel[order_by_fold[k][:m]] = True
                        out.append(~sel if mode == "delete top" else sel)
                    return out
                obs = pooled_auc(mask_for(ranks), v)
                null = np.array([pooled_auc(mask_for([rng.permutation(nb) for _ in folds]), v)
                                 for _ in range(n_rand)])
                rows.append(dict(protocol=protocol, view=vn, mode=mode, n_pairs=m, auroc=obs,
                                 random_mean=float(null.mean()),
                                 random_lo=float(np.percentile(null, 2.5)),
                                 random_hi=float(np.percentile(null, 97.5)),
                                 p_value=float((1 + (null <= obs).sum() if mode == "delete top"
                                                else 1 + (null >= obs).sum()) / (1 + n_rand))))
    return pd.DataFrame(rows)


# =========================================================================================
# 2  reproducibility across training partitions
# =========================================================================================
def pairwise_rho(M):
    """Mean, sd and minimum of pairwise Spearman correlations among the rows of M."""
    R = rankdata(M, axis=1)
    C = np.corrcoef(R)
    iu = np.triu_indices(len(M), 1)
    v = C[iu]
    return float(v.mean()), float(v.std()), float(v.min()), len(v)


def sign_agreement(M):
    s = np.sign(M)
    iu = list(itertools.combinations(range(len(M)), 2))
    return float(np.mean([(s[a] == s[b]).mean() for a, b in iu]))


def aggregate(vec_by_fold, canon, blk):
    """Connection-level values -> region-level and system-pair-level summaries."""
    n = len(vec_by_fold)
    nblk = len(blk["names"]) - 1
    reg_signed = np.zeros((n, P))
    reg_abs = np.zeros((n, P))
    blk_signed = np.zeros((n, nblk))
    blk_sq = np.zeros((n, nblk))
    ok = blk["block"] < nblk                        # connections between retained regions only
    ci, cj, bk = canon[ok, 0], canon[ok, 1], blk["block"][ok]
    deg = (np.bincount(ci, minlength=P) + np.bincount(cj, minlength=P)).astype(float)
    deg[deg == 0] = np.nan                          # the four excluded regions
    nb = np.bincount(bk, minlength=nblk)
    for k, v in enumerate(vec_by_fold):
        vo = v[ok]
        reg_signed[k] = (np.bincount(ci, vo, P) + np.bincount(cj, vo, P)) / deg
        reg_abs[k] = (np.bincount(ci, np.abs(vo), P) + np.bincount(cj, np.abs(vo), P)) / deg
        blk_signed[k] = np.bincount(bk, vo, nblk) / nb
        blk_sq[k] = np.bincount(bk, vo ** 2, nblk) / nb
    keep = ~blk["excluded"]
    return dict(region_signed=reg_signed[:, keep], region_abs=reg_abs[:, keep],
                block_signed=blk_signed, block_sq=blk_sq)


def stability(protocol, folds, blk):
    canon, to_canon_corr = edge_tables()
    rows = []
    agg = {}
    for v, vn in enumerate(VIEWS):
        for q in ("coef", "struct"):
            M = np.stack([to_canonical(f[q][v].astype(np.float64), v, to_canon_corr)
                          for f in folds])
            keep = blk["block"] < len(blk["names"]) - 1            # connections among retained regions
            m, s, lo, npair = pairwise_rho(M[:, keep])
            rows.append(dict(protocol=protocol, view=vn, quantity=q, level="connection",
                             mean_rho=m, sd_rho=s, min_rho=lo, n_pairs=npair,
                             sign_agreement=sign_agreement(M[:, keep])))
            A = aggregate(M, canon, blk)
            agg[(vn, q)] = A
            for lev, key in (("region", "region_signed"), ("region (magnitude)", "region_abs"),
                             ("system pair", "block_signed"), ("system pair (magnitude)", "block_sq")):
                m, s, lo, npair = pairwise_rho(A[key])
                rows.append(dict(protocol=protocol, view=vn, quantity=q, level=lev,
                                 mean_rho=m, sd_rho=s, min_rho=lo, n_pairs=npair,
                                 sign_agreement=sign_agreement(A[key]) if "signed" in key else np.nan))
    return pd.DataFrame(rows), agg


# =========================================================================================
# 3 + 4  agreement across views and with the group-difference analysis
# =========================================================================================
def cross_view(protocol, agg):
    rows = []
    for q in ("coef", "struct"):
        for lev, key in (("region", "region_signed"), ("system pair", "block_signed")):
            mean_by_view = {vn: agg[(vn, q)][key].mean(0) for vn in VIEWS}
            for a, b in itertools.combinations(VIEWS, 2):
                rho = spearmanr(mean_by_view[a], mean_by_view[b]).statistic
                rows.append(dict(protocol=protocol, quantity=q, level=lev, view_a=a, view_b=b,
                                 spearman=float(rho)))
    return pd.DataFrame(rows)


def concordance(protocol, agg, blk):
    rows = []
    for tag, fn in (("all participants", "network_contrast.csv"),
                    ("motion matched", "network_contrast_motion_matched.csv")):
        nc = pd.read_csv(STUDY / "tables" / fn)
        eff = {}
        for _, r in nc.iterrows():
            a, b = sorted((SYSTEM_ORDER.index(r.system_a), SYSTEM_ORDER.index(r.system_b)))
            eff[(a, b)] = r.effect
        idx = [n for n, p in enumerate(blk["pairs"]) if p in eff]
        d = np.array([eff[blk["pairs"][n]] for n in idx])
        for vn in VIEWS:
            for q in ("struct", "coef"):
                cl = agg[(vn, q)]["block_signed"].mean(0)[idx]
                r_s = spearmanr(cl, d)
                r_a = spearmanr(np.abs(cl), np.abs(d))
                rows.append(dict(protocol=protocol, contrast=tag, view=vn, quantity=q,
                                 n_pairs=len(idx), spearman_signed=float(r_s.statistic),
                                 p_signed=float(r_s.pvalue),
                                 spearman_magnitude=float(r_a.statistic),
                                 p_magnitude=float(r_a.pvalue)))
    # regional agreement within the somatomotor system.  The group-difference analysis scores a
    # region by its connections to the other regions of the same system, so the classifier
    # pattern is summarised in the same way (21 regions, 210 connections).
    sm = pd.read_csv(STUDY / "tables" / "sensorimotor_regions.csv")
    canon, to_canon_corr = edge_tables()
    in_sys = np.isin(np.arange(P), np.flatnonzero((blk["system"] == "sensorimotor") & ~blk["excluded"]))
    both = in_sys[canon[:, 0]] & in_sys[canon[:, 1]]
    for vn, v in zip(VIEWS, range(3)):
        M = np.stack([to_canonical(f["struct"][v].astype(np.float64), v, to_canon_corr)
                      for f in agg["_folds"]]).mean(0)
        num = np.bincount(canon[both, 0], M[both], P) + np.bincount(canon[both, 1], M[both], P)
        den = np.bincount(canon[both, 0], minlength=P) + np.bincount(canon[both, 1], minlength=P)
        reg = num / np.maximum(den, 1)
        cl = np.array([reg[int(r) - 1] for r in sm.roi_id])
        r_s = spearmanr(cl, sm.effect)
        rows.append(dict(protocol=protocol, contrast="somatomotor regions, within-system connections",
                         view=vn, quantity="struct", n_pairs=len(sm),
                         spearman_signed=float(r_s.statistic), p_signed=float(r_s.pvalue),
                         spearman_magnitude=np.nan, p_magnitude=np.nan))
    return pd.DataFrame(rows)


# =========================================================================================
def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--protocols", nargs="+", default=["pooled", "loso"])
    ap.add_argument("--out", default=str(OUT), help="directory for the result files")
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    blk = system_blocks()
    lab = np.load(STUDY / "cache" / "nested" / "labels.npz")
    tabs = {k: [] for k in ("blocks", "stability", "crossview", "concordance", "deletion")}
    checks, patterns = {}, {}
    for protocol in args.protocols:
        folds = load_protocol(protocol)
        print(f"{protocol}: {len(folds)} outer folds", flush=True)
        df, chk, ex = contributions(protocol, folds, blk, lab, args.reps)
        checks[protocol] = chk
        print(f"  checks: {chk}", flush=True)
        tabs["deletion"].append(deletion_test(protocol, folds, blk, ex))
        st, agg = stability(protocol, folds, blk)
        agg["_folds"] = folds
        tabs["blocks"].append(df)
        tabs["stability"].append(st)
        tabs["crossview"].append(cross_view(protocol, agg))
        tabs["concordance"].append(concordance(protocol, agg, blk))
        for (vn, q), A in [(k, a) for k, a in agg.items() if k != "_folds"]:
            patterns[f"{protocol}|{vn}|{q}|block_signed"] = A["block_signed"].mean(0)
            patterns[f"{protocol}|{vn}|{q}|region_signed"] = A["region_signed"].mean(0)
    for k, v in tabs.items():
        pd.concat(v).to_csv(OUT / f"interpretation_{k}.csv", index=False)
    np.savez(OUT / "interpretation_patterns.npz", block_names=np.array(blk["names"][:-1]),
             **patterns)
    (OUT / "interpretation_checks.json").write_text(json.dumps(
        dict(checks=checks, reps=args.reps, seed=SEED, systems=SYSTEM_ORDER), indent=2))
    print("wrote interpretation_{blocks,stability,crossview,concordance,deletion}.csv")


if __name__ == "__main__":
    main()
