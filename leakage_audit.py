"""
Quantifying how common evaluation shortcuts inflate ABIDE-I accuracy (Experiment 7).

Reported accuracies on ABIDE-I span roughly 0.70 to 0.98, a spread far larger than the
methods themselves plausibly explain.  A large part of it comes from steps that are fit
on the whole dataset before cross-validation, so that information from the held-out
participants reaches the model.  Rather than assert this, we measure it: the identical
model and data are evaluated under one honest protocol and several protocols each
carrying a single, specific shortcut.

Variants
--------
honest              every learned step fit inside the training fold (reference)
tangent_outside     the connectivity representation computed once on all participants
scaler_outside      + feature standardisation also fit on all participants
fs_outside          + top-k edges chosen by an F-test that sees every label
nonnested_tuning    the regularisation constant chosen on the test fold
site_feature        acquisition site added as a one-hot predictor
all_shortcuts       every shortcut above applied together

The first three form a nested sequence: standardisation and feature selection can only
be fitted before the split once the features exist, so a variant that fits either of
them globally also computes the representation globally.  The increment between
consecutive variants therefore isolates the step that was added.  ``nonnested_tuning``
and ``site_feature`` are independent of that sequence and each differ from ``honest``
in one respect only.

Usage:
    python leakage_audit.py --n_folds 10 --out results/ablations
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")

import warnings
warnings.filterwarnings("ignore")
import numpy as np

from run_benchmark import load_dataset, _outer_splits, _metrics
from gpu_tangent import GPUTangentFC

VARIANTS = ["honest", "scaler_outside", "tangent_outside", "fs_outside",
            "nonnested_tuning", "site_feature", "all_shortcuts"]


def _select_k(Xtr, ytr, Xte, k):
    from sklearn.feature_selection import SelectKBest, f_classif
    sel = SelectKBest(f_classif, k=min(k, Xtr.shape[1])).fit(Xtr, ytr)
    return sel.transform(Xtr), sel.transform(Xte)


def run_variant(variant, covs, y, sites, splits, gt, seed, k_features=2000):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import roc_auc_score

    leak_scaler = variant in ("scaler_outside", "all_shortcuts")
    leak_tangent = variant in ("tangent_outside", "all_shortcuts")
    leak_fs = variant in ("fs_outside", "all_shortcuts")
    leak_tune = variant in ("nonnested_tuning", "all_shortcuts")
    use_site = variant in ("site_feature", "all_shortcuts")

    # --- steps fitted on ALL participants when the variant asks for it ---
    # Scaling and feature selection can only be fitted before the split if the features
    # themselves already exist, so any variant that fits one of them globally also
    # computes the representation globally.  The variants are therefore nested, and the
    # difference between consecutive ones isolates the step that was added.
    needs_global = leak_tangent or leak_scaler or leak_fs
    X_all = None
    if needs_global:
        gt.fit(covs)                                  # reference from everyone
        X_all = gt.transform(covs)
    global_scaler = None
    if leak_scaler:
        global_scaler = StandardScaler().fit(X_all)
    global_fs = None
    if leak_fs:
        Z = global_scaler.transform(X_all) if global_scaler is not None else X_all
        # selection uses every label, including those of the held-out participants
        global_fs = SelectKBest(f_classif, k=min(k_features, Z.shape[1])).fit(Z, y)

    site_codes = None
    if use_site:
        uniq = {s: i for i, s in enumerate(np.unique(sites))}
        site_codes = np.eye(len(uniq))[[uniq[s] for s in sites]]

    fold_metrics, oof_true, oof_prob = [], [], []
    for tr, te in splits:
        if needs_global:
            Xtr, Xte = X_all[tr], X_all[te]
        else:
            gt.fit(covs[tr])                          # reference from TRAIN only
            Xtr, Xte = gt.transform(covs[tr]), gt.transform(covs[te])

        if leak_scaler and global_scaler is not None:
            Xtr, Xte = global_scaler.transform(Xtr), global_scaler.transform(Xte)
        else:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        if leak_fs:
            Xtr, Xte = global_fs.transform(Xtr), global_fs.transform(Xte)

        if use_site and site_codes is not None:
            Xtr = np.hstack([Xtr, site_codes[tr]])
            Xte = np.hstack([Xte, site_codes[te]])

        grid = [0.001, 0.01, 0.1, 1.0]
        if leak_tune:                                  # choose C using the TEST fold
            best, best_auc, best_p = None, -np.inf, None
            for C in grid:
                clf = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(Xtr, y[tr])
                p = clf.predict_proba(Xte)[:, 1]
                a = roc_auc_score(y[te], p)
                if a > best_auc:
                    best, best_auc, best_p = C, a, p
            prob = best_p
        else:
            gs = GridSearchCV(LogisticRegression(solver="liblinear", max_iter=2000),
                              {"C": grid}, scoring="roc_auc", cv=3, n_jobs=-1).fit(Xtr, y[tr])
            prob = gs.predict_proba(Xte)[:, 1]

        fold_metrics.append(_metrics(y[te], prob))
        oof_true.extend(y[te].tolist()); oof_prob.extend(prob.tolist())

    keys = list(fold_metrics[0].keys())
    return {
        "variant": variant,
        "fold_mean": {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in keys},
        "fold_std": {k: float(np.nanstd([f[k] for f in fold_metrics])) for k in keys},
        "pooled_oof": _metrics(np.array(oof_true), np.array(oof_prob)),
        "n_folds": len(splits),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Leakage audit on ABIDE-I")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_dir", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_features", type=int, default=2000)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--out", default="results/ablations")
    args = ap.parse_args()

    import pandas as pd
    from tqdm import tqdm

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_dir))
    gt = GPUTangentFC(device="cpu", max_iter=10)
    covs = gt.compute_covariances(ts)
    splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)

    rows = []
    for v in tqdm(args.variants, desc="leakage variants", unit="variant"):
        r = run_variant(v, covs, y, sites, splits, gt, args.seed, args.k_features)
        rows.append(r)
        print(f"  {v:18s} AUROC={r['fold_mean']['auroc']:.3f}  acc={r['fold_mean']['accuracy']:.3f}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "leakage_audit.json").write_text(json.dumps(rows, indent=2))
    base = next((r for r in rows if r["variant"] == "honest"), None)
    flat = []
    for r in rows:
        f = {"variant": r["variant"],
             "auroc": r["fold_mean"]["auroc"], "auroc_std": r["fold_std"]["auroc"],
             "accuracy": r["fold_mean"]["accuracy"], "accuracy_std": r["fold_std"]["accuracy"]}
        if base:
            f["auroc_inflation"] = f["auroc"] - base["fold_mean"]["auroc"]
            f["accuracy_inflation"] = f["accuracy"] - base["fold_mean"]["accuracy"]
        flat.append(f)
    pd.DataFrame(flat).to_csv(out / "leakage_audit.csv", index=False)
    print("\nreport ->", out / "leakage_audit.csv")


if __name__ == "__main__":
    main()
