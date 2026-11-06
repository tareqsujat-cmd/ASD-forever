"""
Permutation reference for the interpretation analyses.

Diagnostic labels are permuted within site, which preserves each site's prevalence, and the
pooled ten-fold procedure is repeated on the permuted labels with the penalty fixed at the
value selected on the real labels.  Under this null there is no relation between the
connectivity of a participant and their diagnosis beyond what site prevalence carries.

Two quantities are referred to it.

* Reproducibility.  The mean pairwise rank correlation of the weights and of the structure
  coefficients across the ten training partitions, at the level of connections, regions and
  system pairs.  A null value shows how much agreement arises when the fits have nothing
  real to reproduce.
* Held-out signal of each system pair.  The AUROC of a pair's held-out contribution, with a
  familywise p-value from the maximum deviation from 0.5 across the 36 pairs in every
  permutation, so that the family of pairs is corrected together.

Stage 1 (heavy) fits the permuted models and stores their weights and held-out
contributions.  Stage 2 (--analyse) compares them with the real fits.

Usage:
    .venv/bin/python exp_interpretation_null.py --workers 4 --n-perm 100
    .venv/bin/python exp_interpretation_null.py --analyse
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
NULL_DIR = OUT / "nested_null"
SEED = 42


def permuted_labels(y, sites, r):
    """Labels permuted within each site (same site prevalence, no connectivity relation)."""
    rng = np.random.default_rng(SEED * 1000 + r)
    yp = y.copy()
    for s in np.unique(sites):
        idx = np.flatnonzero(sites == s)
        yp[idx] = rng.permutation(y[idx])
    return yp


def feature_block_index():
    """Block of every feature in each view's own vectorisation order (36 pairs + unassigned)."""
    import exp_interpretation as ei
    blk = ei.system_blocks()
    _, to_canon_corr = ei.edge_tables()
    return blk["block"][to_canon_corr], blk["block"], len(blk["names"])


def null_fold(k: int, n_perm: int) -> str:
    from sklearn.linear_model import LogisticRegression
    from run_benchmark import _outer_splits

    path = NULL_DIR / f"null_pooled_{k:02d}.npz"
    if path.exists():
        return f"{path.name} exists"
    D = eno.data()
    y, sites = D.y, D.sites
    tr, te = _outer_splits("pooled", y, sites, 10, eno.SEED)[0][k]
    real = np.load(eno.OUT / f"fold_pooled_{k:02d}.npz")
    blk_corr, blk_canon, nb = feature_block_index()
    t0 = time.time()
    n_feat = D.corr.shape[1]

    coef = np.zeros((n_perm, 3, n_feat), np.float32)
    struct = np.zeros((n_perm, 3, n_feat), np.float32)
    partial = np.zeros((n_perm, 3, len(te), nb), np.float32)
    perms = [permuted_labels(y, sites, r) for r in range(n_perm)]
    for v, view in enumerate(eno.VIEWS):
        Ztr, Zte = eno.view_features(view, tr, te, D)
        Z64 = np.asarray(Ztr, np.float64)
        Zc = Z64 - Z64.mean(0)
        bidx = blk_corr if v == 0 else blk_canon
        onehot = np.zeros((n_feat, nb))
        onehot[np.arange(n_feat), bidx] = 1.0
        Zte64 = np.asarray(Zte, np.float64)
        for r in range(n_perm):
            clf = LogisticRegression(C=float(real["c_best"][v]), solver="liblinear",
                                     max_iter=2000).fit(Ztr, perms[r][tr])
            w, b = clf.coef_[0], float(clf.intercept_[0])
            s = Z64 @ w + b
            struct[r, v] = Zc.T @ (s - s.mean()) / (len(tr) * s.std())
            coef[r, v] = w
            partial[r, v] = Zte64 @ (onehot * w[:, None])
        print(f"  [null {k:02d}] {view:18s} done ({time.time() - t0:.0f}s)", flush=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, te_idx=te, coef=coef, struct=struct, partial=partial)
    os.replace(tmp, path)
    return f"{path.name} done in {time.time() - t0:.0f}s"


# =========================================================================================
def analyse(n_perm_expected: int | None = None) -> None:
    import exp_interpretation as ei

    blk = ei.system_blocks()
    canon, to_canon_corr = ei.edge_tables()
    nb = len(blk["names"])
    lab = np.load(eno.CACHE / "labels.npz")
    y_all = lab["y"].astype(int)
    real = [np.load(f) for f in sorted(eno.OUT.glob("fold_pooled_*.npz")) if ".tmp" not in f.name]
    null = [np.load(f) for f in sorted(NULL_DIR.glob("null_pooled_*.npz")) if ".tmp" not in f.name]
    if len(null) != len(real):
        raise SystemExit(f"{len(null)} null folds for {len(real)} real folds")
    R = null[0]["coef"].shape[0]
    keep_conn = blk["block"] < nb - 1
    print(f"{len(real)} folds, {R} permutations", flush=True)

    def stats_from(M):                                     # M: (folds, features) canonical order
        A = ei.aggregate(M, canon, blk)
        out = {"connection": ei.pairwise_rho(M[:, keep_conn])[0]}
        out["region"] = ei.pairwise_rho(A["region_signed"])[0]
        out["region (magnitude)"] = ei.pairwise_rho(A["region_abs"])[0]
        out["system pair"] = ei.pairwise_rho(A["block_signed"])[0]
        out["system pair (magnitude)"] = ei.pairwise_rho(A["block_sq"])[0]
        return out

    rows = []
    for v, vn in enumerate(eno.VIEWS):
        for q in ("coef", "struct"):
            real_M = np.stack([ei.to_canonical(f[q][v].astype(np.float64), v, to_canon_corr)
                               for f in real])
            obs = stats_from(real_M)
            nulls = {k: [] for k in obs}
            for r in range(R):
                M = np.stack([ei.to_canonical(f[q][r, v].astype(np.float64), v, to_canon_corr)
                              for f in null])
                for k, val in stats_from(M).items():
                    nulls[k].append(val)
            for lev in obs:
                nv = np.asarray(nulls[lev])
                rows.append(dict(view=vn, quantity=q, level=lev, observed_rho=obs[lev],
                                 null_mean=float(nv.mean()), null_sd=float(nv.std()),
                                 null_p95=float(np.percentile(nv, 95)),
                                 null_max=float(nv.max()),
                                 p_value=float((1 + (nv >= obs[lev]).sum()) / (1 + R))))
    pd.DataFrame(rows).to_csv(OUT / "interpretation_stability_null.csv", index=False)

    # held-out AUROC of each system pair against the permutation maximum
    order = np.concatenate([f["te_idx"] for f in real])
    y = y_all[order]
    real_blocks = pd.read_csv(OUT / "interpretation_blocks.csv")
    real_blocks = real_blocks[real_blocks.protocol == "pooled"]
    rows = []
    names = list(eno.VIEWS) + ["mean of views"]
    null_part = np.concatenate([f["partial"] for f in null], axis=2)         # (R, 3, n, nb)
    null_part = np.concatenate([null_part, null_part.mean(1, keepdims=True)], axis=1)
    null_auc = np.zeros((R, 4, nb))
    for r in range(R):
        for v in range(4):
            null_auc[r, v] = ei.auroc_cols(null_part[r, v].astype(np.float64), y)
    for v, vn in enumerate(names):
        rb = real_blocks[real_blocks.view == vn].set_index("block")
        maxdev = np.abs(null_auc[:, v, : nb - 1] - 0.5).max(1)               # (R,)
        for b in range(nb - 1):
            obs = rb.loc[blk["names"][b], "auroc"]
            dev = abs(obs - 0.5)
            own = np.abs(null_auc[:, v, b] - 0.5)
            rows.append(dict(view=vn, block=blk["names"][b], auroc=obs,
                             null_mean=float(null_auc[:, v, b].mean()),
                             p_single=float((1 + (own >= dev).sum()) / (1 + R)),
                             p_familywise=float((1 + (maxdev >= dev).sum()) / (1 + R)),
                             null_max_dev_p95=float(np.percentile(maxdev, 95))))
    pd.DataFrame(rows).to_csv(OUT / "interpretation_block_null.csv", index=False)
    print("wrote interpretation_stability_null.csv and interpretation_block_null.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    if args.analyse:
        analyse()
        return
    NULL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eno.run_parallel([(k, null_fold, (k, args.n_perm)) for k in range(10)], args.workers,
                     lambda k, msg, mins: print(f"{time.strftime('%H:%M:%S')} fold {k:02d}: {msg} "
                                                f"(elapsed {mins:.1f} min)", flush=True))
    (NULL_DIR / "run.json").write_text(json.dumps(dict(n_perm=args.n_perm, seed=SEED,
                                                       minutes=(time.time() - t0) / 60)))


if __name__ == "__main__":
    main()
