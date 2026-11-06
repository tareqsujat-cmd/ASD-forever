"""
Outer-fold refits that support the calibration and interpretation analyses.

The primary ensemble (three connectivity views, each an l2-logistic model whose penalty is
tuned on its own training partition, probabilities averaged with equal weight) is refitted
on every outer training partition exactly as in the stored predictions.  From the same
refits this script keeps three things.

1. Cross-fitted predictions inside each training partition.  Every outer training partition
   is divided again (stratified 5-fold under pooled validation, grouped-by-site 5-fold under
   leave-one-site-out validation) and the whole ensemble is refitted on each inner training
   part, so that every outer-training participant receives a prediction from a model that
   never saw them.  A recalibration map is fitted on these predictions only, so nothing from
   the outer test fold can reach it.
2. A diagnostic variant of each outer fit in which the intercept is not penalised, used to
   test whether the calibration offset comes from the solver (liblinear penalises the
   intercept together with the weights).
3. For each outer fit, the linear weights, the structure coefficients (correlation between
   every connection and the model score in the training partition), and the standardised
   held-out features, so that held-out contributions of any group of connections can be
   computed afterwards without refitting.

One file is written per outer fold, and existing files are skipped, so the job can be
stopped and resumed.  The analyses themselves are in exp_calibration_nested.py and
exp_interpretation.py.

Usage:
    .venv/bin/python exp_nested_outer.py --protocol pooled --workers 4
    .venv/bin/python exp_nested_outer.py --protocol loso   --workers 4
    .venv/bin/python exp_nested_outer.py --protocol pooled --folds 0 --outer-only   # check
"""

from __future__ import annotations

import os

# Thread caps must precede the numpy / torch imports, and every spawned worker re-runs them.
_THREADS = os.environ.get("NESTED_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _THREADS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SEED = 42
WINDOW, N_WINDOWS = 50, 6                      # as in the stored voting run
C_GRID = (0.001, 0.01, 0.1, 1.0)               # as logreg_factory in run_ablations.py
VIEWS = ("correlation", "tangent (static)", "dynamic: mean")
CACHE = ROOT / "results" / "latest_study" / "cache" / "nested"
OUT = ROOT / "results" / "latest_study" / "review" / "nested"
GS_JOBS = int(os.environ.get("NESTED_GS_JOBS", "1"))     # 1: no joblib children inside the workers


# =========================================================================================
# data and cache
# =========================================================================================
def build_cache() -> None:
    """Covariances, window covariances and correlation vectors are fold-independent
    (Ledoit-Wolf is fitted per participant), so they are computed once and memory-mapped."""
    if all((CACHE / f).exists() for f in ("covs.npy", "wcovs.npy", "corr.npy", "labels.npz")):
        return
    from nilearn.connectome import ConnectivityMeasure
    from dynamic_fc import DynamicTangentFC
    from gpu_tangent import GPUTangentFC
    from run_benchmark import load_dataset

    CACHE.mkdir(parents=True, exist_ok=True)
    ts, y, sites, sids = load_dataset(ROOT / "abide_processed",
                                      ROOT / "abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    print(f"building cache for {len(y)} participants ...", flush=True)
    covs = GPUTangentFC(device="cpu", max_iter=10).compute_covariances(ts)
    np.save(CACHE / "covs.npy", covs)
    wl = DynamicTangentFC(window=WINDOW, n_windows=N_WINDOWS).compute_window_covariances(ts)
    if len({w.shape[0] for w in wl}) != 1:
        raise SystemExit("participants differ in the number of windows; cache layout invalid")
    np.save(CACHE / "wcovs.npy", np.stack(wl))
    cm = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
    np.save(CACHE / "corr.npy", cm.fit_transform(ts))
    np.savez(CACHE / "labels.npz", y=np.asarray(y), sites=np.asarray(sites),
             sids=np.asarray(sids))


class _Rows:
    """List-like view over memory-mapped window covariances that hands out writable copies."""

    def __init__(self, a):
        self.a = a

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, i):
        return np.array(self.a[i])


class Data:
    def __init__(self):
        from dynamic_fc import DynamicTangentFC
        from gpu_tangent import GPUTangentFC
        lab = np.load(CACHE / "labels.npz")
        self.y, self.sites = lab["y"].astype(int), lab["sites"]
        self.covs = np.load(CACHE / "covs.npy", mmap_mode="r")
        self.wcovs = _Rows(np.load(CACHE / "wcovs.npy", mmap_mode="r"))
        self.corr = np.load(CACHE / "corr.npy", mmap_mode="r")
        self.gt = GPUTangentFC(device="cpu", max_iter=10)
        self.dfc = DynamicTangentFC(window=WINDOW, n_windows=N_WINDOWS)


_DATA = None


def data() -> Data:
    global _DATA
    if _DATA is None:
        import torch
        torch.set_num_threads(int(_THREADS))
        _DATA = Data()
    return _DATA


# =========================================================================================
# one member: features fitted on `a`, applied to `b`
# =========================================================================================
def view_features(view: str, a: np.ndarray, b: np.ndarray, D: Data):
    """Standardised features of one view; every learned quantity is fitted on ``a`` only."""
    from sklearn.preprocessing import StandardScaler
    if view == "correlation":
        Xa, Xb = np.asarray(D.corr[a]), np.asarray(D.corr[b])
    elif view == "tangent (static)":
        D.gt.fit(np.asarray(D.covs[a]))
        Xa, Xb = D.gt.transform(np.asarray(D.covs[a])), D.gt.transform(np.asarray(D.covs[b]))
    else:
        D.dfc.fit(D.wcovs, a)
        Xa = D.dfc.transform(D.wcovs, a, blocks=("mean",))["mean"]
        Xb = D.dfc.transform(D.wcovs, b, blocks=("mean",))["mean"]
    sc = StandardScaler().fit(Xa)
    return sc.transform(Xa), sc.transform(Xb)


def fit_member(Za: np.ndarray, ya: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(LogisticRegression(solver="liblinear", max_iter=2000),
                      {"C": list(C_GRID)}, scoring="roc_auc", cv=3, n_jobs=GS_JOBS)
    return gs.fit(Za, ya)


def inner_splits(protocol: str, tr: np.ndarray, y: np.ndarray, sites: np.ndarray, n_inner: int):
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    if protocol == "pooled":
        it = StratifiedKFold(n_inner, shuffle=True, random_state=SEED).split(tr, y[tr])
    else:
        it = GroupKFold(n_splits=n_inner).split(tr, y[tr], groups=sites[tr])
    return [(tr[a], tr[b]) for a, b in it]


# =========================================================================================
# one outer fold
# =========================================================================================
def run_fold(protocol: str, k: int, n_inner: int, outer_only: bool) -> dict:
    from sklearn.linear_model import LogisticRegression
    from run_benchmark import _outer_splits

    D = data()
    y, sites = D.y, D.sites
    tr, te = _outer_splits(protocol, y, sites, 10, SEED)[0][k]
    t0 = time.time()
    n_tr, n_te, n_feat = len(tr), len(te), D.corr.shape[1]

    p_raw = np.zeros((3, n_te))
    p_unpen = np.zeros((3, n_te))
    coef = np.zeros((3, n_feat), np.float32)
    struct = np.zeros((3, n_feat), np.float32)
    z_te = np.zeros((3, n_te, n_feat), np.float32)
    c_best = np.zeros(3)
    icpt = np.zeros(3)
    icpt_unpen = np.zeros(3)
    score_sd = np.zeros(3)

    for v, view in enumerate(VIEWS):
        Ztr, Zte = view_features(view, tr, te, D)
        gs = fit_member(Ztr, y[tr])
        clf = gs.best_estimator_
        p_raw[v] = clf.predict_proba(Zte)[:, 1]
        w, b = clf.coef_[0], float(clf.intercept_[0])
        c_best[v], icpt[v] = gs.best_params_["C"], b
        Z64 = np.asarray(Ztr, np.float64)
        s = Z64 @ w + b
        score_sd[v] = s.std()
        # columns of Z are standardised on the training partition, so this is corr(z_j, s)
        struct[v] = (Z64 - Z64.mean(0)).T @ (s - s.mean()) / (n_tr * s.std())
        coef[v] = w
        z_te[v] = Zte
        un = LogisticRegression(C=c_best[v], solver="lbfgs", max_iter=1000).fit(Ztr, y[tr])
        p_unpen[v] = un.predict_proba(Zte)[:, 1]
        icpt_unpen[v] = float(un.intercept_[0])
        print(f"  [{protocol} {k:02d}] outer {view:18s} C={c_best[v]:<6g} "
              f"({time.time() - t0:.0f}s)", flush=True)

    inner_p = np.full((3, n_tr), np.nan)
    if not outer_only:
        for j, (a, b_) in enumerate(inner_splits(protocol, tr, y, sites, n_inner)):
            pos = np.searchsorted(tr, b_)
            for v, view in enumerate(VIEWS):
                Za, Zb = view_features(view, a, b_, D)
                inner_p[v, pos] = fit_member(Za, y[a]).predict_proba(Zb)[:, 1]
            print(f"  [{protocol} {k:02d}] inner {j + 1}/{n_inner} ({time.time() - t0:.0f}s)",
                  flush=True)

    return dict(tr_idx=tr, te_idx=te, p_raw=p_raw, p_unpen=p_unpen, inner_p=inner_p,
                c_best=c_best, intercept=icpt, intercept_unpen=icpt_unpen,
                score_sd=score_sd, coef=coef, struct=struct, z_te=z_te,
                seconds=np.array(time.time() - t0))


def worker(protocol: str, k: int, n_inner: int, outer_only: bool, out_dir: str) -> str:
    path = Path(out_dir) / f"fold_{protocol}_{k:02d}.npz"
    if path.exists():
        return f"{path.name} exists"
    res = run_fold(protocol, k, n_inner, outer_only)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res)
    os.replace(tmp, path)
    return f"{path.name} done in {float(res['seconds']):.0f}s"


def run_parallel(jobs, workers, report):
    """Run ``jobs`` (a list of (key, function, args)) in spawned worker processes.

    Results are passed to ``report(key, result, elapsed_minutes)`` as they arrive.  On the first
    exception the traceback is printed, every worker is terminated and the process exits with status 1
    at once.  Leaving a ``with ProcessPoolExecutor`` block after a failure waits for the workers to
    exit, which can hang for good when a worker still owns idle child processes, so the pool is shut
    down by hand instead.
    """
    import traceback
    t0 = time.time()
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
    try:
        futs = {ex.submit(fn, *args): key for key, fn, args in jobs}
        for f in as_completed(futs):
            report(futs[f], f.result(), (time.time() - t0) / 60)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        for p in mp.active_children():
            try:
                p.terminate()
            except Exception:
                pass
        os._exit(1)
    ex.shutdown(wait=True)


# =========================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--protocol", choices=["pooled", "loso"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--folds", default="", help="comma-separated outer folds (default: all)")
    ap.add_argument("--outer-only", action="store_true",
                    help="skip the inner cross-fitting (used to check reproduction)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    build_cache()
    lab = np.load(CACHE / "labels.npz")
    n_outer = 10 if args.protocol == "pooled" else len(np.unique(lab["sites"]))
    folds = [int(f) for f in args.folds.split(",") if f != ""] or list(range(n_outer))
    out_dir = Path(args.out + ("_check" if args.outer_only else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.protocol}: {len(folds)} outer folds, {args.workers} workers, "
          f"inner={'none' if args.outer_only else args.inner_folds}", flush=True)

    t0 = time.time()
    run_parallel([(k, worker, (args.protocol, k, args.inner_folds, args.outer_only, str(out_dir)))
                  for k in folds], args.workers,
                 lambda k, msg, mins: print(f"{time.strftime('%H:%M:%S')} fold {k:02d}: {msg}  "
                                            f"(elapsed {mins:.1f} min)", flush=True))
    (out_dir / f"run_{args.protocol}.json").write_text(json.dumps(
        dict(protocol=args.protocol, folds=folds, inner_folds=args.inner_folds,
             outer_only=args.outer_only, c_grid=list(C_GRID), seed=SEED,
             minutes=(time.time() - t0) / 60), indent=2))


if __name__ == "__main__":
    main()
