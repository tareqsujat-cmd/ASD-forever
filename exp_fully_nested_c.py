"""
Fully nested selection of the penalty strength.

In the published pipeline the tangent reference and the feature scaler are fitted once on each
outer training partition, and the penalty strength C is then chosen by three-fold inner
cross-validation on those features.  The inner-validation participants therefore contribute
to the reference and to the scaling statistics that are applied to them, although their
diagnostic labels are never used.  This script removes that dependence.  For every outer
fold and every view, the tangent reference and the scaler are refitted on each inner training
part, applied to the inner validation part, and C is chosen from the resulting AUROCs.  The
inner splits are those of the published search (stratified three-fold, no shuffling), and
the selection rule is the same (mean inner AUROC, ties to the smaller C).

If the fully nested search selects the published C in every case, the published outer
predictions are unchanged.  Where it selects a different C, the outer model of that view is
refitted with it and the ensemble predictions are recomputed, so that the effect on the
headline measures is reported directly.

Because the published search selects the smallest value of its grid, 0.001, the same inner
cross-validation is also run over a grid that extends two decades lower (1e-5 to 1), so that
the effect of the lower boundary on the headline measures is reported as well.

Usage:
    .venv/bin/python exp_fully_nested_c.py --protocol pooled --workers 4
    .venv/bin/python exp_fully_nested_c.py --protocol loso   --workers 4
    .venv/bin/python exp_fully_nested_c.py --summarise
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
DIR = OUT / "fully_nested"


def fold_job(protocol: str, k: int) -> str:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from run_benchmark import _outer_splits

    path = DIR / f"fn_{protocol}_{k:02d}.npz"
    if path.exists():
        return f"{path.name} exists"
    D = eno.data()
    y, sites = D.y, D.sites
    tr, te = _outer_splits(protocol, y, sites, 10, eno.SEED)[0][k]
    ref = np.load(eno.OUT / f"fold_{protocol}_{k:02d}.npz")
    t0 = time.time()
    Cs = [1e-5, 1e-4] + list(eno.C_GRID)                           # extended grid; the published one is its tail
    pub_idx = [Cs.index(c) for c in eno.C_GRID]
    inner = list(StratifiedKFold(3).split(tr, y[tr]))             # the search of GridSearchCV(cv=3)

    mean_auc = np.zeros((3, len(Cs)))
    c_full = np.zeros(3)                                           # fully nested, published grid
    c_ext = np.zeros(3)                                            # fully nested, extended grid
    p_new = ref["p_raw"].copy()
    p_ext = ref["p_raw"].copy()
    changed = np.zeros(3, bool)
    changed_ext = np.zeros(3, bool)
    for v, view in enumerate(eno.VIEWS):
        aucs = np.zeros((len(inner), len(Cs)))
        for j, (a, b) in enumerate(inner):
            Za, Zb = eno.view_features(view, tr[a], tr[b], D)      # reference and scaler: inner-train only
            for ci, c in enumerate(Cs):
                clf = LogisticRegression(C=c, solver="liblinear", max_iter=2000).fit(Za, y[tr[a]])
                aucs[j, ci] = roc_auc_score(y[tr[b]], clf.decision_function(Zb))
        mean_auc[v] = aucs.mean(0)
        c_full[v] = Cs[pub_idx[int(np.argmax(mean_auc[v][pub_idx]))]]   # ties go to the first (smaller) C
        c_ext[v] = Cs[int(np.argmax(mean_auc[v]))]
        changed[v] = c_full[v] != ref["c_best"][v]
        changed_ext[v] = c_ext[v] != ref["c_best"][v]
        if changed[v] or changed_ext[v]:
            Ztr, Zte = eno.view_features(view, tr, te, D)
            for c, flag, dest in ((c_full[v], changed[v], p_new), (c_ext[v], changed_ext[v], p_ext)):
                if flag:
                    dest[v] = LogisticRegression(C=c, solver="liblinear", max_iter=2000) \
                        .fit(Ztr, y[tr]).predict_proba(Zte)[:, 1]
        print(f"  [{protocol} {k:02d}] {view:18s} C_full={c_full[v]:<6g} C_ext={c_ext[v]:<6g} "
              f"published={ref['c_best'][v]:<6g} ({time.time() - t0:.0f}s)", flush=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, te_idx=te, mean_auc=mean_auc, c_grid=np.array(Cs), c_full=c_full,
                        c_ext=c_ext, c_pub=ref["c_best"], changed=changed, changed_ext=changed_ext,
                        p_pub=ref["p_raw"], p_new=p_new, p_ext=p_ext)
    os.replace(tmp, path)
    return (f"{path.name} done in {time.time() - t0:.0f}s ({int(changed.sum())} of 3 views changed on the "
            f"published grid, {int(changed_ext.sum())} on the extended grid)")


def summarise() -> None:
    from sklearn.metrics import roc_auc_score
    lab = np.load(eno.CACHE / "labels.npz")
    y_all = lab["y"].astype(int)
    rows, cs = [], []
    for protocol in ("pooled", "loso"):
        files = [f for f in sorted(DIR.glob(f"fn_{protocol}_*.npz")) if ".tmp" not in f.name]
        if not files:
            continue
        F = [np.load(f) for f in files]
        te = np.concatenate([f["te_idx"] for f in F])
        y = y_all[te]
        pub = np.concatenate([f["p_pub"].mean(0) for f in F])
        new = np.concatenate([f["p_new"].mean(0) for f in F])
        ext = np.concatenate([f["p_ext"].mean(0) for f in F])
        nchg = int(sum(f["changed"].sum() for f in F))
        nchg_ext = int(sum(f["changed_ext"].sum() for f in F))
        for k, f in enumerate(F):
            for v, vn in enumerate(eno.VIEWS):
                cs.append(dict(protocol=protocol, fold=k, view=vn, c_published=float(f["c_pub"][v]),
                               c_fully_nested=float(f["c_full"][v]), changed=bool(f["changed"][v]),
                               c_extended_grid=float(f["c_ext"][v]), changed_extended=bool(f["changed_ext"][v]),
                               **{f"inner_auroc_C={c:g}": float(f["mean_auc"][v, i])
                                  for i, c in enumerate(f["c_grid"])}))
        rows.append(dict(protocol=protocol, n_folds=len(F), cases=3 * len(F), cases_changed=nchg,
                         auroc_published=float(roc_auc_score(y, pub)),
                         auroc_fully_nested=float(roc_auc_score(y, new)),
                         accuracy_published=float(((pub >= .5) == y).mean()),
                         accuracy_fully_nested=float(((new >= .5) == y).mean()),
                         max_abs_probability_change=float(np.abs(pub - new).max()),
                         cases_changed_extended_grid=nchg_ext,
                         auroc_extended_grid=float(roc_auc_score(y, ext)),
                         accuracy_extended_grid=float(((ext >= .5) == y).mean()),
                         max_abs_probability_change_extended=float(np.abs(pub - ext).max())))
    pd.DataFrame(cs).to_csv(OUT / "fully_nested_c_selection.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "fully_nested_c_summary.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--protocol", choices=["pooled", "loso"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--folds", default="")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        summarise()
        return
    if not a.protocol:
        raise SystemExit("--protocol is required")
    DIR.mkdir(parents=True, exist_ok=True)
    n_outer = 10 if a.protocol == "pooled" else len(np.unique(np.load(eno.CACHE / "labels.npz")["sites"]))
    folds = [int(f) for f in a.folds.split(",") if f != ""] or list(range(n_outer))
    t0 = time.time()
    eno.run_parallel([(k, fold_job, (a.protocol, k)) for k in folds], a.workers,
                     lambda k, msg, mins: print(f"{time.strftime('%H:%M:%S')} fold {k:02d}: {msg} "
                                                f"(elapsed {mins:.1f} min)", flush=True))


if __name__ == "__main__":
    main()
