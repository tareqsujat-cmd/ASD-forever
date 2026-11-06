"""
Ablation suite for the ABIDE-I connectome study.

Runs the experiment ladder that supports the paper's claims and writes every result as
both JSON (full detail, including per-fold values) and CSV (flat, one row per condition)
so that figures can be produced without re-running anything.

Experiments
-----------
representation  correlation -> tangent -> tangent + dynamic blocks          (Novelty N1)
voting          the three representations combined at the decision level
classifier      logistic / low-rank spectral / ridge / SVM / LDA / trees    (Novelty N2)
rank            rank sweep for the low-rank spectral classifier             (Novelty N2)
atlas           each atlas alone, every pair, and all three
fusion          probability averaging vs stacking vs feature concatenation
harmonisation   none vs in-fold ComBat
protocol        pooled vs leave-one-site-out, with conformal abstention     (Novelty N3)
statistics      DeLong and McNemar tests between the headline conditions

Usage:
    python run_ablations.py --experiments representation classifier rank
    python run_ablations.py --all --n_folds 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# torch and the gradient-boosting libraries each ship their own OpenMP runtime.  With
# more than one runtime loaded and several threads each, the two deadlock at the thread
# barrier on macOS and the process stalls at zero CPU.  The ``classifier`` experiment is
# the only one that loads both, and it must therefore be started with
# ``OMP_NUM_THREADS=1`` (the boosting libraries still use their own thread pools).  The
# guard below fails fast rather than allowing a silent stall.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def _check_omp_for_trees(experiments):
    if "classifier" in experiments and os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit(
            "The 'classifier' experiment loads torch and the boosting libraries "
            "together, which deadlocks unless OpenMP is single-threaded.\n"
            "Re-run it as:  OMP_NUM_THREADS=1 python run_ablations.py "
            "--experiments classifier ...")

import warnings
warnings.filterwarnings("ignore")
import numpy as np

from run_benchmark import load_dataset, _outer_splits, _metrics
from train_sota import load_multi_atlas, load_phenotype, ATLAS_INFO
from gpu_tangent import GPUTangentFC

ALL_EXPERIMENTS = ["representation", "voting", "classifier", "rank", "rank_transfer",
                   "learning_curve", "atlas", "fusion", "harmonisation", "protocol",
                   "statistics"]


# --------------------------------------------------------------------------
# shared evaluation
# --------------------------------------------------------------------------

def cv_evaluate(feature_fn, clf_factory, splits, y, sites, tag, protocol="pooled"):
    """Run one condition through the given splits.

    ``feature_fn(tr, te, fold_index) -> (Xtr, Xte)`` must fit everything it learns on ``tr``
    only; the fold index lets a condition reuse a per-fold cache.
    """
    fold_metrics, oof_t, oof_p, oof_s = [], [], [], []
    for k, (tr, te) in enumerate(splits):
        Xtr, Xte = feature_fn(tr, te, k)
        clf = clf_factory()
        clf.fit(Xtr, y[tr])
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(Xte)[:, 1]
        else:
            d = clf.decision_function(Xte)
            prob = 1.0 / (1.0 + np.exp(-d))
        fold_metrics.append(_metrics(y[te], prob))
        oof_t.extend(y[te].tolist()); oof_p.extend(prob.tolist())
        oof_s.extend(np.asarray(sites)[te].tolist())
    keys = list(fold_metrics[0].keys())
    return {
        "condition": tag, "protocol": protocol, "n_folds": len(splits),
        "fold_mean": {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in keys},
        "fold_std": {k: float(np.nanstd([f[k] for f in fold_metrics])) for k in keys},
        "pooled_oof": _metrics(np.array(oof_t), np.array(oof_p)),
        "oof_true": oof_t, "oof_prob": oof_p, "oof_site": oof_s,
    }


def _scaled(Xtr, Xte):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def tangent_features(gt, covs):
    """Return a feature_fn that fits the tangent reference and scaler in-fold."""
    def fn(tr, te, k=None):
        gt.fit(covs[tr])
        return _scaled(gt.transform(covs[tr]), gt.transform(covs[te]))
    return fn


def logreg_factory(C_grid=(0.001, 0.01, 0.1, 1.0)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    return lambda: GridSearchCV(LogisticRegression(solver="liblinear", max_iter=2000),
                                {"C": list(C_grid)}, scoring="roc_auc", cv=3, n_jobs=-1)


def _row(res, experiment):
    fm, fs = res["fold_mean"], res["fold_std"]
    return {
        "experiment": experiment, "condition": res["condition"],
        "protocol": res["protocol"], "n_folds": res["n_folds"],
        "auroc": fm["auroc"], "auroc_std": fs["auroc"],
        "accuracy": fm["accuracy"], "accuracy_std": fs["accuracy"],
        "sensitivity": fm.get("sensitivity"), "specificity": fm.get("specificity"),
        "balanced_accuracy": fm.get("balanced_accuracy"),
        "auroc_pooled_oof": res["pooled_oof"]["auroc"],
    }


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def exp_representation(ctx):
    """Correlation vs tangent vs tangent + dynamic trajectory blocks (N1)."""
    from dynamic_fc import DynamicTangentFC
    from nilearn.connectome import ConnectivityMeasure
    res = []
    ts, y, sites, splits = ctx["ts"], ctx["y"], ctx["sites"], ctx["splits"]
    covs, gt = ctx["covs"], ctx["gt"]

    # correlation
    def corr_fn(tr, te, k=None):
        cm = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
        return _scaled(cm.fit_transform([ts[i] for i in tr]),
                       cm.transform([ts[i] for i in te]))
    res.append(cv_evaluate(corr_fn, logreg_factory(), splits, y, sites, "correlation"))

    # static tangent
    res.append(cv_evaluate(tangent_features(gt, covs), logreg_factory(), splits, y, sites,
                           "tangent (static)"))

    # dynamic trajectory blocks
    print("  computing sliding-window covariances (once) ...", flush=True)
    dfc = DynamicTangentFC(window=ctx["window"], n_windows=ctx["n_windows"])
    wcovs = dfc.compute_window_covariances(ts)
    nwin = np.mean([c.shape[0] for c in wcovs])
    print(f"  mean windows/subject = {nwin:.1f}", flush=True)

    # The tangent projection is the expensive step and is shared by every block
    # combination, so each fold is projected once and all combinations are evaluated
    # before the projection is released.  Holding only one fold at a time keeps the
    # peak memory at a few hundred megabytes instead of several gigabytes.
    all_blocks = ("mean", "dispersion", "velocity")
    # Blocks are combined in two ways.  Concatenation puts every block under a single
    # penalty, so a block with little signal still contributes ~20,000 features and can
    # dilute the informative ones.  Late fusion instead fits one classifier per block and
    # averages the probabilities, which lets each block carry its own regularisation and
    # lets an uninformative block contribute a near-chance vote rather than noise.
    combos = [("mean",), ("mean", "dispersion"), ("mean", "velocity"), all_blocks]
    conds = ([("concat", c) for c in combos] +
             [("fuse", c) for c in combos if len(c) > 1])
    acc = {c: {"m": [], "t": [], "p": [], "s": []} for c in conds}
    make_clf = logreg_factory()

    for k, (tr, te) in enumerate(splits):
        t0 = time.time()
        dfc.fit(wcovs, tr)                                   # reference on TRAIN only
        B_tr = dfc.transform(wcovs, tr, blocks=all_blocks)
        B_te = dfc.transform(wcovs, te, blocks=all_blocks)

        # one classifier per individual block, reused by every fusion condition
        block_prob = {}
        for b in all_blocks:
            Xtr, Xte = _scaled(B_tr[b], B_te[b])
            block_prob[b] = make_clf().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]

        for mode, c in conds:
            if mode == "concat":
                Xtr, Xte = _scaled(np.hstack([B_tr[b] for b in c]),
                                   np.hstack([B_te[b] for b in c]))
                prob = make_clf().fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
            else:
                prob = np.mean([block_prob[b] for b in c], axis=0)
            key = (mode, c)
            acc[key]["m"].append(_metrics(y[te], prob))
            acc[key]["t"].extend(y[te].tolist()); acc[key]["p"].extend(prob.tolist())
            acc[key]["s"].extend(np.asarray(sites)[te].tolist())
        del B_tr, B_te
        print(f"    fold {k + 1}/{len(splits)} all blocks ({time.time() - t0:.0f}s)", flush=True)

    for (mode, c) in conds:
        fm = acc[(mode, c)]["m"]; keys = list(fm[0].keys())
        label = ("dynamic: " + "+".join(c)) if mode == "concat" \
            else ("dynamic (fused): " + "+".join(c))
        res.append({
            "condition": label, "protocol": "pooled", "n_folds": len(splits),
            "fold_mean": {k: float(np.nanmean([f[k] for f in fm])) for k in keys},
            "fold_std": {k: float(np.nanstd([f[k] for f in fm])) for k in keys},
            "pooled_oof": _metrics(np.array(acc[(mode, c)]["t"]),
                                   np.array(acc[(mode, c)]["p"])),
            "oof_true": acc[(mode, c)]["t"], "oof_prob": acc[(mode, c)]["p"],
            "oof_site": acc[(mode, c)]["s"]})
    del wcovs
    return res


def phenotype_matrix(sids, pheno_csv="./abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"):
    """Demographic variables aligned to ``sids`` -> (n, 6), NaN where unrecorded.

    Read from the phenotypic table rather than the processed cache so that the same
    routine serves either quality-control cohort.
    """
    import pandas as pd
    cols = ["AGE_AT_SCAN", "SEX", "FIQ", "VIQ", "PIQ", "HANDEDNESS_SCORES"]
    d = pd.read_csv(pheno_csv)
    d["_sid"] = d["SUB_ID"].astype(int).astype(str)
    d = d.set_index("_sid")
    out = np.full((len(sids), len(cols)), np.nan)
    for i, s_ in enumerate(sids):
        key = str(int(s_))
        if key not in d.index:
            continue
        row = d.loc[key]
        for j, c in enumerate(cols):
            if c in d.columns:
                v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
                if pd.notna(v) and v > -9990:
                    out[i, j] = float(v)
    return out


def _rank_against(reference: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Position of each value of ``x`` within the ``reference`` distribution, in (0, 1).

    The rank of a held-out participant must be taken against the *training* predictions.
    Ranking within the held-out fold instead makes the score depend on the other people
    scored alongside them -- under leave-one-site-out that fold is an entire site, so the
    value would be a rank among patients at the same scanner, uniform by construction and
    impossible to compute for a single incoming patient.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    pos = np.searchsorted(ref, np.asarray(x, dtype=float), side="right")
    return (pos + 0.5) / (len(ref) + 1.0)


def _phenotype_probs(pheno, y, tr, te, seed=42):
    """Probability from the demographic variables alone, fitted inside the fold."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    imp = SimpleImputer(strategy="median").fit(pheno[tr])
    sc = StandardScaler().fit(imp.transform(pheno[tr]))
    Ptr = sc.transform(imp.transform(pheno[tr]))
    Pte = sc.transform(imp.transform(pheno[te]))
    gs = GridSearchCV(LogisticRegression(max_iter=2000),
                      {"C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc",
                      cv=3, n_jobs=-1).fit(Ptr, y[tr])
    return gs.predict_proba(Pte)[:, 1], gs.predict_proba(Ptr)[:, 1]


def exp_voting(ctx):
    """Combining the three connectivity representations at the decision level.

    The classifier comparison shows that models sharing the same features agree almost
    exactly, so averaging them changes little.  The three *representations* — Pearson
    correlation, the static tangent embedding and the windowed tangent mean — describe
    the same scans differently and their predictions agree far less, which is the
    condition under which averaging is useful.

    The combination is fixed in advance as *all three representations, equally weighted*,
    rather than chosen by inspecting results, and both a probability average and a rank
    average are reported.  Every representation, its classifier and the scaling are fitted
    inside the training fold, so the ensemble is evaluated exactly like any single model.
    """
    from dynamic_fc import DynamicTangentFC
    from nilearn.connectome import ConnectivityMeasure
    from scipy.stats import rankdata

    ts, y, sites = ctx["ts"], ctx["y"], ctx["sites"]
    covs, gt = ctx["covs"], ctx["gt"]
    make_clf = logreg_factory()

    print("  computing sliding-window covariances (once) ...", flush=True)
    dfc = DynamicTangentFC(window=ctx["window"], n_windows=ctx["n_windows"])
    wcovs = dfc.compute_window_covariances(ts)

    members = ("correlation", "tangent (static)", "dynamic: mean")
    out = []
    for protocol in ctx.get("voting_protocols", ["pooled"]):
        splits, _ = _outer_splits(protocol, y, sites, ctx["n_folds"], ctx["seed"])
        print(f"  {protocol}: {len(splits)} folds", flush=True)
        out += _voting_pass(splits, protocol, ts, y, sites, covs, gt, dfc, wcovs,
                            members, make_clf, ctx.get("pheno_mat"))
    del wcovs
    return out


def _voting_pass(splits, protocol, ts, y, sites, covs, gt, dfc, wcovs, members, make_clf,
                 pheno=None):
    from nilearn.connectome import ConnectivityMeasure
    from scipy.stats import rankdata
    conds = list(members) + ["ensemble (probability mean)", "ensemble (rank mean)"]
    if pheno is not None:
        conds += ["phenotype only", "ensemble + phenotype (probability mean)",
                  "ensemble + phenotype (rank mean)"]
    acc = {c: {"m": [], "t": [], "p": [], "s": []} for c in conds}

    for k, (tr, te) in enumerate(splits):
        t0 = time.time()
        feats = {}
        cm = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
        feats["correlation"] = _scaled(cm.fit_transform([ts[i] for i in tr]),
                                       cm.transform([ts[i] for i in te]))
        gt.fit(covs[tr])
        feats["tangent (static)"] = _scaled(gt.transform(covs[tr]), gt.transform(covs[te]))
        dfc.fit(wcovs, tr)
        B_tr = dfc.transform(wcovs, tr, blocks=("mean",))
        B_te = dfc.transform(wcovs, te, blocks=("mean",))
        feats["dynamic: mean"] = _scaled(B_tr["mean"], B_te["mean"])

        probs, probs_tr = {}, {}
        for name in members:
            Xtr, Xte = feats[name]
            clf = make_clf().fit(Xtr, y[tr])
            probs[name] = clf.predict_proba(Xte)[:, 1]
            probs_tr[name] = clf.predict_proba(Xtr)[:, 1]

        rk = {m: _rank_against(probs_tr[m], probs[m]) for m in members}
        combined = {
            "ensemble (probability mean)": np.mean([probs[m] for m in members], axis=0),
            "ensemble (rank mean)": np.mean([rk[m] for m in members], axis=0),
        }
        if pheno is not None:
            pp, pp_tr = _phenotype_probs(pheno, y, tr, te)
            combined["phenotype only"] = pp
            combined["ensemble + phenotype (probability mean)"] = np.mean(
                [probs[m] for m in members] + [pp], axis=0)
            combined["ensemble + phenotype (rank mean)"] = np.mean(
                [rk[m] for m in members] + [_rank_against(pp_tr, pp)], axis=0)
        for name, p in {**probs, **combined}.items():
            acc[name]["m"].append(_metrics(y[te], p))
            acc[name]["t"].extend(y[te].tolist()); acc[name]["p"].extend(p.tolist())
            acc[name]["s"].extend(np.asarray(sites)[te].tolist())
        del feats, B_tr, B_te
        print(f"    fold {k + 1}/{len(splits)} ({time.time() - t0:.0f}s)", flush=True)

    out = []
    for c in conds:
        fm = acc[c]["m"]; keys = list(fm[0].keys())
        label = c if protocol == "pooled" else f"{c} [{protocol}]"
        out.append({
            "condition": label, "protocol": protocol, "n_folds": len(splits),
            "fold_mean": {k: float(np.nanmean([f[k] for f in fm])) for k in keys},
            "fold_std": {k: float(np.nanstd([f[k] for f in fm])) for k in keys},
            "pooled_oof": _metrics(np.array(acc[c]["t"]), np.array(acc[c]["p"])),
            "oof_true": acc[c]["t"], "oof_prob": acc[c]["p"], "oof_site": acc[c]["s"]})
    return out


def exp_classifier(ctx):
    """Same tangent features, different classifiers (N2 in context)."""
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from models.lowrank_spectral import LowRankSpectralClassifier

    feats = tangent_features(ctx["gt"], ctx["covs"])
    y, sites, splits = ctx["y"], ctx["sites"], ctx["splits"]
    seed = ctx["seed"]
    factories = {
        "logistic (dense linear)": logreg_factory(),
        "low-rank spectral (r=20)": lambda: LowRankSpectralClassifier(rank=20),
        "ridge": lambda: RidgeClassifier(alpha=100.0),
        "linear SVM": lambda: LinearSVC(C=0.01, max_iter=5000),
        "shrinkage LDA (PCA-200)": lambda: Pipeline([
            ("pca", PCA(n_components=200, random_state=seed)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
        "random forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=1,
                                                        random_state=seed),
    }
    try:
        import xgboost as xgb
        factories["xgboost"] = lambda: xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.3, n_jobs=4, random_state=seed,
            eval_metric="logloss", verbosity=0)
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        factories["catboost"] = lambda: CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.1, rsm=0.3, thread_count=4,
            random_seed=seed, verbose=0, allow_writing_files=False)
    except Exception:
        pass

    # The tangent reference and projection do not depend on which classifier is being
    # evaluated, so each fold is prepared once and every classifier is fitted on it.
    # Preparing it per classifier would repeat the most expensive step of the pipeline
    # once for each of them.
    names = list(factories)
    acc = {n: {"m": [], "t": [], "p": [], "s": []} for n in names}
    for k, (tr, te) in enumerate(splits):
        t0 = time.time()
        Xtr, Xte = feats(tr, te, k)
        for name, make in factories.items():
            clf = make()
            clf.fit(Xtr, y[tr])
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(Xte)[:, 1]
            else:
                prob = 1.0 / (1.0 + np.exp(-clf.decision_function(Xte)))
            acc[name]["m"].append(_metrics(y[te], prob))
            acc[name]["t"].extend(y[te].tolist()); acc[name]["p"].extend(prob.tolist())
            acc[name]["s"].extend(np.asarray(sites)[te].tolist())
        print(f"    fold {k + 1}/{len(splits)} all classifiers ({time.time() - t0:.0f}s)",
              flush=True)

    out = []
    for n in names:
        fm = acc[n]["m"]; keys = list(fm[0].keys())
        out.append({
            "condition": n, "protocol": "pooled", "n_folds": len(splits),
            "fold_mean": {k: float(np.nanmean([f[k] for f in fm])) for k in keys},
            "fold_std": {k: float(np.nanstd([f[k] for f in fm])) for k in keys},
            "pooled_oof": _metrics(np.array(acc[n]["t"]), np.array(acc[n]["p"])),
            "oof_true": acc[n]["t"], "oof_prob": acc[n]["p"], "oof_site": acc[n]["s"]})
    return out


def exp_rank(ctx):
    """Rank sweep for the low-rank spectral classifier (N2).

    The stage-1 dense fit and its eigendecomposition do not depend on r, so they are
    performed once per fold and every rank is obtained by truncating the same spectrum.
    """
    from sklearn.linear_model import LogisticRegression
    from models.lowrank_spectral import vec_to_sym, batch_vec_to_sym, n_rois_from_dim

    y, sites, splits = ctx["y"], ctx["sites"], ctx["splits"]
    gt, covs, ranks = ctx["gt"], ctx["covs"], list(ctx["ranks"])
    acc = {r: {"m": [], "t": [], "p": [], "s": []} for r in ranks}

    for k, (tr, te) in enumerate(splits):
        t0 = time.time()
        gt.fit(covs[tr])
        Xtr, Xte = _scaled(gt.transform(covs[tr]), gt.transform(covs[te]))
        p_rois = n_rois_from_dim(Xtr.shape[1])
        dense = LogisticRegression(C=0.01, solver="liblinear",
                                   max_iter=2000).fit(Xtr, y[tr])
        W = vec_to_sym(dense.coef_.ravel(), p_rois, halve=True)
        w_eig, V = np.linalg.eigh(W)
        order = np.argsort(np.abs(w_eig))[::-1]
        Str, Ste = batch_vec_to_sym(Xtr, p_rois), batch_vec_to_sym(Xte, p_rois)

        for r in ranks:
            U = V[:, order[:r]]
            Ztr = np.einsum("nij,ik,jk->nk", Str, U, U, optimize=True)
            Zte = np.einsum("nij,ik,jk->nk", Ste, U, U, optimize=True)
            head = LogisticRegression(C=1.0, solver="liblinear",
                                      max_iter=2000).fit(Ztr, y[tr])
            prob = head.predict_proba(Zte)[:, 1]
            acc[r]["m"].append(_metrics(y[te], prob))
            acc[r]["t"].extend(y[te].tolist()); acc[r]["p"].extend(prob.tolist())
            acc[r]["s"].extend(np.asarray(sites)[te].tolist())
        print(f"    fold {k + 1}/{len(splits)} all ranks ({time.time() - t0:.0f}s)", flush=True)

    out = []
    for r in ranks:
        fm = acc[r]["m"]; keys = list(fm[0].keys())
        out.append({
            "condition": f"rank={r}", "protocol": "pooled", "n_folds": len(splits),
            "fold_mean": {k: float(np.nanmean([f[k] for f in fm])) for k in keys},
            "fold_std": {k: float(np.nanstd([f[k] for f in fm])) for k in keys},
            "pooled_oof": _metrics(np.array(acc[r]["t"]), np.array(acc[r]["p"])),
            "oof_true": acc[r]["t"], "oof_prob": acc[r]["p"], "oof_site": acc[r]["s"]})
    return out


def _rank_sweep(covs, y, sites, splits, ranks, gt, C_dense=0.01, protocol="pooled",
                train_fraction=1.0, seed=42):
    """Dense model and its rank-truncated versions over the given splits.

    The dense fit and its eigendecomposition are shared across ranks, so the whole
    sweep costs little more than the dense model alone.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from models.lowrank_spectral import vec_to_sym, batch_vec_to_sym, n_rois_from_dim

    conds = ["dense"] + [f"rank={r}" for r in ranks]
    acc = {c: {"m": [], "t": [], "p": [], "s": []} for c in conds}
    rng = np.random.RandomState(seed)

    for k, (tr, te) in enumerate(splits):
        if train_fraction < 1.0:                      # learning-curve subsampling
            n_keep = max(40, int(round(train_fraction * len(tr))))
            tr = rng.permutation(tr)[:n_keep]
            if len(np.unique(y[tr])) < 2:
                continue
        gt.fit(covs[tr])
        Xtr, Xte = _scaled(gt.transform(covs[tr]), gt.transform(covs[te]))
        p_rois = n_rois_from_dim(Xtr.shape[1])

        dense = LogisticRegression(C=C_dense, solver="liblinear",
                                   max_iter=2000).fit(Xtr, y[tr])
        prob = dense.predict_proba(Xte)[:, 1]
        acc["dense"]["m"].append(_metrics(y[te], prob))
        acc["dense"]["t"].extend(y[te].tolist()); acc["dense"]["p"].extend(prob.tolist())
        acc["dense"]["s"].extend(np.asarray(sites)[te].tolist())

        W = vec_to_sym(dense.coef_.ravel(), p_rois, halve=True)
        w_eig, V = np.linalg.eigh(W)
        order = np.argsort(np.abs(w_eig))[::-1]
        Str, Ste = batch_vec_to_sym(Xtr, p_rois), batch_vec_to_sym(Xte, p_rois)
        for r in ranks:
            U = V[:, order[:r]]
            Ztr = np.einsum("nij,ik,jk->nk", Str, U, U, optimize=True)
            Zte = np.einsum("nij,ik,jk->nk", Ste, U, U, optimize=True)
            head = LogisticRegression(C=1.0, solver="liblinear",
                                      max_iter=2000).fit(Ztr, y[tr])
            p = head.predict_proba(Zte)[:, 1]
            c = f"rank={r}"
            acc[c]["m"].append(_metrics(y[te], p))
            acc[c]["t"].extend(y[te].tolist()); acc[c]["p"].extend(p.tolist())
            acc[c]["s"].extend(np.asarray(sites)[te].tolist())

    out = []
    for c in conds:
        fm = acc[c]["m"]
        if not fm:
            continue
        keys = list(fm[0].keys())
        out.append({
            "condition": c, "protocol": protocol, "n_folds": len(fm),
            "train_fraction": train_fraction,
            "fold_mean": {k: float(np.nanmean([f[k] for f in fm])) for k in keys},
            "fold_std": {k: float(np.nanstd([f[k] for f in fm])) for k in keys},
            "pooled_oof": _metrics(np.array(acc[c]["t"]), np.array(acc[c]["p"])),
            "oof_true": acc[c]["t"], "oof_prob": acc[c]["p"], "oof_site": acc[c]["s"]})
    return out


def exp_rank_transfer(ctx):
    """Does constraining the rank help more when the test site is unseen?

    A model with fewer parameters has less freedom to fit variation that is specific to
    the sites it was trained on, so the benefit of the constraint should be larger when
    the evaluation withholds an entire site than when the folds are drawn at random.
    The same rank sweep is therefore run under both partitioning schemes and the two
    curves are compared against the dense model measured in the same scheme.
    """
    y, sites, covs, gt = ctx["y"], ctx["sites"], ctx["covs"], ctx["gt"]
    res = []
    for protocol in ("pooled", "loso"):
        splits, _ = _outer_splits(protocol, y, sites, ctx["n_folds"], ctx["seed"])
        print(f"  {protocol}: {len(splits)} folds", flush=True)
        for r in _rank_sweep(covs, y, sites, splits, ctx["ranks"], gt,
                             protocol=protocol, seed=ctx["seed"]):
            r["condition"] = f"{protocol}: {r['condition']}"
            res.append(r)
    return res


def exp_learning_curve(ctx):
    """How performance depends on the amount of training data.

    The number of free parameters differs by more than an order of magnitude between the
    dense model and its rank-constrained form, so the two are expected to respond
    differently as the training set is reduced.  Each fraction is drawn from the training
    portion of the fold only, and the representation is refitted on that subset.
    """
    y, sites, covs, gt = ctx["y"], ctx["sites"], ctx["covs"], ctx["gt"]
    splits, _ = _outer_splits("pooled", y, sites, ctx["n_folds"], ctx["seed"])
    res = []
    for frac in ctx["fractions"]:
        print(f"  training fraction {frac:.2f}", flush=True)
        for r in _rank_sweep(covs, y, sites, splits, [20, 50], gt,
                             protocol="pooled", train_fraction=frac, seed=ctx["seed"]):
            r["condition"] = f"{r['condition']} @ {int(frac * 100)}% train"
            res.append(r)
    return res


def exp_atlas(ctx):
    """Each atlas alone, every pair, and all three (probability averaging).

    Since fusion happens at the probability level, each atlas member is fit once per
    fold and every combination is formed by averaging the cached member probabilities.
    """
    from itertools import combinations
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    y, sites, splits = ctx["y"], ctx["sites"], ctx["splits"]
    covs_by_atlas, gts = ctx["covs_by_atlas"], ctx["gts"]
    atlases = list(covs_by_atlas)

    fold_probs = []                                   # per fold: {atlas -> test probs}
    for k, (tr, te) in enumerate(splits):
        t0 = time.time()
        d = {}
        for a in atlases:
            gt, covs = gts[a], covs_by_atlas[a]
            gt.fit(covs[tr])
            Xtr, Xte = _scaled(gt.transform(covs[tr]), gt.transform(covs[te]))
            gs = GridSearchCV(LogisticRegression(solver="liblinear", max_iter=2000),
                              {"C": [0.001, 0.01, 0.1, 1.0]}, scoring="roc_auc",
                              cv=3, n_jobs=-1).fit(Xtr, y[tr])
            d[a] = gs.predict_proba(Xte)[:, 1]
        fold_probs.append(d)
        print(f"    fold {k + 1}/{len(splits)} members fit ({time.time() - t0:.0f}s)", flush=True)

    res = []
    for r in (1, 2, 3):
        for combo in combinations(atlases, r):
            fm, ot, op, os_ = [], [], [], []
            for k, (tr, te) in enumerate(splits):
                prob = np.mean([fold_probs[k][a] for a in combo], axis=0)
                fm.append(_metrics(y[te], prob))
                ot.extend(y[te].tolist()); op.extend(prob.tolist())
                os_.extend(np.asarray(sites)[te].tolist())
            keys = list(fm[0].keys())
            res.append({
                "condition": "+".join(combo), "protocol": "pooled", "n_folds": len(splits),
                "fold_mean": {k: float(np.nanmean([f[k] for f in fm])) for k in keys},
                "fold_std": {k: float(np.nanstd([f[k] for f in fm])) for k in keys},
                "pooled_oof": _metrics(np.array(ot), np.array(op)),
                "oof_true": ot, "oof_prob": op, "oof_site": os_})
    return res


def _ensemble_eval(combo, covs_by_atlas, gts, splits, y, sites, tag, mode="mean",
                   pheno=None):
    """Multi-atlas ensemble under a chosen fusion rule."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    fold_metrics, oof_t, oof_p, oof_s = [], [], [], []
    for tr, te in splits:
        member_te, member_tr = [], []
        for a in combo:
            gt, covs = gts[a], covs_by_atlas[a]
            gt.fit(covs[tr])
            Xtr, Xte = _scaled(gt.transform(covs[tr]), gt.transform(covs[te]))
            if mode == "concat":
                member_tr.append(Xtr); member_te.append(Xte); continue
            gs = GridSearchCV(LogisticRegression(solver="liblinear", max_iter=2000),
                              {"C": [0.001, 0.01, 0.1, 1.0]}, scoring="roc_auc",
                              cv=3, n_jobs=-1).fit(Xtr, y[tr])
            member_te.append(gs.predict_proba(Xte)[:, 1])
            if mode == "stack":
                from sklearn.model_selection import cross_val_predict
                member_tr.append(cross_val_predict(
                    LogisticRegression(C=gs.best_params_["C"], solver="liblinear",
                                       max_iter=2000),
                    Xtr, y[tr], cv=3, method="predict_proba")[:, 1])

        if mode == "concat":
            Xtr, Xte = np.hstack(member_tr), np.hstack(member_te)
            gs = GridSearchCV(LogisticRegression(solver="liblinear", max_iter=2000),
                              {"C": [0.001, 0.01, 0.1]}, scoring="roc_auc",
                              cv=3, n_jobs=-1).fit(Xtr, y[tr])
            prob = gs.predict_proba(Xte)[:, 1]
        elif mode == "stack":
            meta = LogisticRegression(max_iter=2000).fit(np.column_stack(member_tr), y[tr])
            prob = meta.predict_proba(np.column_stack(member_te))[:, 1]
        else:
            prob = np.mean(member_te, axis=0)

        fold_metrics.append(_metrics(y[te], prob))
        oof_t.extend(y[te].tolist()); oof_p.extend(prob.tolist())
        oof_s.extend(np.asarray(sites)[te].tolist())

    keys = list(fold_metrics[0].keys())
    return {"condition": tag, "protocol": "pooled", "n_folds": len(splits),
            "fold_mean": {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in keys},
            "fold_std": {k: float(np.nanstd([f[k] for f in fold_metrics])) for k in keys},
            "pooled_oof": _metrics(np.array(oof_t), np.array(oof_p)),
            "oof_true": oof_t, "oof_prob": oof_p, "oof_site": oof_s}


def exp_fusion(ctx):
    """Probability averaging vs stacking vs feature concatenation."""
    combo = tuple(ctx["covs_by_atlas"])
    return [_ensemble_eval(combo, ctx["covs_by_atlas"], ctx["gts"], ctx["splits"],
                           ctx["y"], ctx["sites"], tag=f"fusion: {m}", mode=m)
            for m in ("mean", "stack", "concat")]


def exp_harmonisation(ctx):
    """No harmonisation vs in-fold ComBat."""
    from train_sota import _combat_infold
    y, sites, splits = ctx["y"], ctx["sites"], ctx["splits"]
    gt, covs, pheno = ctx["gt"], ctx["covs"], ctx["pheno"]
    res = [cv_evaluate(tangent_features(gt, covs), logreg_factory(), splits, y, sites,
                       "no harmonisation")]

    def combat_fn(tr, te, k=None):
        gt.fit(covs[tr])
        Xtr, Xte = _scaled(gt.transform(covs[tr]), gt.transform(covs[te]))
        cov_tr = pheno[tr][:, :2] if pheno is not None else None
        cov_te = pheno[te][:, :2] if pheno is not None else None
        return _combat_infold(Xtr, Xte, np.asarray(sites)[tr], np.asarray(sites)[te],
                              cov_tr, cov_te)
    res.append(cv_evaluate(combat_fn, logreg_factory(), splits, y, sites, "in-fold ComBat"))
    return res


def exp_protocol(ctx):
    """Pooled vs leave-one-site-out, plus conformal abstention (N3)."""
    import conformal_selective as CS
    y, sites = ctx["y"], ctx["sites"]
    gt, covs = ctx["gt"], ctx["covs"]
    res = []
    for protocol in ("pooled", "loso"):
        splits, _ = _outer_splits(protocol, y, sites, ctx["n_folds"], ctx["seed"])
        r = cv_evaluate(tangent_features(gt, covs), logreg_factory(), splits, y, sites,
                        f"{protocol}", protocol=protocol)
        # conformal abstention on the out-of-fold predictions
        yv = np.array(r["oof_true"]); pv = np.array(r["oof_prob"]); sv = np.array(r["oof_site"])
        for mode in ("marginal", "mondrian"):
            thr = CS.fit_thresholds(yv, pv, sv, alpha=ctx["alpha"], mode=mode)
            ev = CS.evaluate(yv, pv, sv, thr)
            r[f"conformal_{mode}"] = ev
        res.append(r)
    return res


def exp_statistics(ctx, previous):
    """DeLong and McNemar tests between the headline conditions."""
    from evaluation.stat_compare import delong_roc_test, mcnemar_test
    by_tag = {r["condition"]: r for rs in previous.values() for r in rs
              if isinstance(r, dict) and "oof_prob" in r}
    pairs = [("tangent (static)", "correlation"),
             ("low-rank spectral (r=20)", "logistic (dense linear)"),
             ("dynamic: mean+dispersion+velocity", "tangent (static)"),
             ("logistic (dense linear)", "random forest")]
    rows = []
    for a, b in pairs:
        if a not in by_tag or b not in by_tag:
            continue
        ra, rb = by_tag[a], by_tag[b]
        n = min(len(ra["oof_true"]), len(rb["oof_true"]))
        ya = np.array(ra["oof_true"][:n]); pa = np.array(ra["oof_prob"][:n])
        pb = np.array(rb["oof_prob"][:n])
        try:
            auc_a, auc_b, delong_p = delong_roc_test(ya, pa, pb)
        except Exception as e:
            auc_a = auc_b = float("nan"); delong_p = float("nan")
            print(f"    DeLong failed for {a} vs {b}: {e}")
        try:
            m = mcnemar_test(ya, (pa >= 0.5).astype(int), (pb >= 0.5).astype(int))
        except Exception as e:
            m = {"p_value": float("nan")}
            print(f"    McNemar failed for {a} vs {b}: {e}")
        rows.append({"comparison": f"{a} vs {b}",
                     "auroc_a_oof": auc_a, "auroc_b_oof": auc_b,
                     "delong_p": delong_p, "mcnemar_p": m.get("p_value"),
                     "auroc_a_foldmean": ra["fold_mean"]["auroc"],
                     "auroc_b_foldmean": rb["fold_mean"]["auroc"]})
    return rows


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Ablation suite (ABIDE-I)")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--atlases", nargs="+", default=["cc200", "aal", "ho"])
    ap.add_argument("--experiments", nargs="+", default=["representation", "classifier", "rank"])
    ap.add_argument("--all", action="store_true", help="run every experiment")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ranks", nargs="+", type=int,
                    default=[1, 2, 5, 10, 20, 30, 50, 100])
    ap.add_argument("--window", type=int, default=50, help="dynamic FC window length (volumes)")
    ap.add_argument("--n_windows", type=int, default=6, help="windows per participant (adaptive stride)")
    ap.add_argument("--step", type=int, default=30, help="dynamic FC stride (volumes)")
    ap.add_argument("--phenotype_member", action="store_true",
                    help="add a demographic member to the voting experiment")
    ap.add_argument("--cohort", default="strict", choices=["strict", "func", "all"],
                    help="quality-control cohort; 'func' is the wider 1,035 definition")
    ap.add_argument("--voting_protocols", nargs="+", default=["pooled", "loso"],
                    help="partitioning schemes for the voting experiment")
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.25, 0.5, 0.75, 1.0], help="learning-curve training fractions")
    ap.add_argument("--alpha", type=float, default=0.2, help="conformal target risk")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--run_name", default="ablations")
    args = ap.parse_args()

    import pandas as pd
    from utilities.run_dir import create_run_dir, write_manifest

    experiments = ALL_EXPERIMENTS if args.all else args.experiments
    _check_omp_for_trees(experiments)
    run_dir = create_run_dir(args.out_dir, run_name=args.run_name)
    write_manifest(run_dir, seed=args.seed, device="cpu", args=args, mode="ablations")
    out = run_dir / "ablations"; out.mkdir(parents=True, exist_ok=True)
    print(f"run directory: {run_dir}")

    # -- data (loaded once) --
    needs_multi = any(e in experiments for e in ("atlas", "fusion"))
    if args.cohort == "strict":
        ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    else:
        # the processed metadata lists only the strict cohort, so a wider cohort is
        # assembled from the phenotypic table and the time-series files on disk
        from qc_cohort import load_from_phenotype, cohort_ids
        ts, y, sites, sids = load_from_phenotype(
            "./abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
            Path(args.ts_root))
        keep = cohort_ids(cohort=args.cohort)
        idx = [i for i, s_ in enumerate(sids) if s_ in keep]
        ts = [ts[i] for i in idx]
        y, sites = y[idx], sites[idx]
        sids = [sids[i] for i in idx]
        print(f"cohort '{args.cohort}': {len(y)} participants "
              f"({int((y == 1).sum())} ASD / {int((y == 0).sum())} control)", flush=True)
    pheno = load_phenotype(args.processed_dir, sids) if args.cohort == "strict" else None
    gt = GPUTangentFC(device="cpu", max_iter=10)
    print("computing CC200 covariances ...", flush=True)
    covs = gt.compute_covariances(ts)
    splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)

    covs_by_atlas, gts = {}, {}
    if needs_multi:
        ts_by_atlas, y2, sites2, sids2 = load_multi_atlas(
            Path(args.processed_dir), Path(args.ts_root), args.atlases)
        y, sites = y2, sites2
        splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)
        for a in args.atlases:
            g = GPUTangentFC(device="cpu", max_iter=10)
            print(f"computing {a} covariances ...", flush=True)
            covs_by_atlas[a] = g.compute_covariances(ts_by_atlas[a]); gts[a] = g

    pheno_mat = phenotype_matrix(sids) if args.phenotype_member else None
    if pheno_mat is not None:
        ok = np.isfinite(pheno_mat).all(axis=1).sum()
        print(f"demographic variables: {ok}/{len(sids)} participants complete", flush=True)
    ctx = dict(ts=ts, y=y, sites=sites, splits=splits, covs=covs, gt=gt, pheno=pheno,
               pheno_mat=pheno_mat,
               covs_by_atlas=covs_by_atlas, gts=gts, seed=args.seed, ranks=args.ranks,
               window=args.window, step=args.step, n_windows=args.n_windows,
               fractions=args.fractions, voting_protocols=args.voting_protocols,
               alpha=args.alpha, n_folds=args.n_folds)

    runners = {"representation": exp_representation, "voting": exp_voting,
               "rank_transfer": exp_rank_transfer, "learning_curve": exp_learning_curve,
               "classifier": exp_classifier,
               "rank": exp_rank, "atlas": exp_atlas, "fusion": exp_fusion,
               "harmonisation": exp_harmonisation, "protocol": exp_protocol}

    results, rows = {}, []
    for name in experiments:
        if name == "statistics":
            continue
        t0 = time.time()
        print(f"\n=== experiment: {name} ===", flush=True)
        res = runners[name](ctx)
        results[name] = res
        for r in res:
            rows.append(_row(r, name))
            print(f"  {r['condition']:36s} AUROC={r['fold_mean']['auroc']:.3f} "
                  f"acc={r['fold_mean']['accuracy']:.3f}")
        # per-experiment artefacts
        pd.DataFrame([_row(r, name) for r in res]).to_csv(out / f"{name}.csv", index=False)
        (out / f"{name}.json").write_text(json.dumps(res, indent=2))
        print(f"  ({time.time() - t0:.0f}s) -> {out / (name + '.csv')}")

    if "statistics" in experiments and results:
        stats = exp_statistics(ctx, results)
        pd.DataFrame(stats).to_csv(out / "statistics.csv", index=False)
        (out / "statistics.json").write_text(json.dumps(stats, indent=2))
        print("\nstatistical comparisons ->", out / "statistics.csv")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out / "ablations_all.csv", index=False)
        print("\ncombined ->", out / "ablations_all.csv")
        print(df[["experiment", "condition", "auroc", "accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
