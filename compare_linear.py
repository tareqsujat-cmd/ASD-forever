"""
Linear-model ablation on tangent-FC: do all linear classifiers cluster together,
and does a linear ensemble improve over the single logistic model?

Same leakage-free pooled CV and the same in-fold tangent-FC features as the main
benchmark; only the linear classifier changes.  Each model's regularization is tuned
by a small inner CV for fairness.  Models:

  logistic (ℓ2) · linear SVM · ridge · elastic-net logistic · shrinkage LDA (PCA)

plus an **ensemble** = mean of the members' standardized scores.

No xgboost/catboost import here, so only torch's OpenMP is loaded (no dual-runtime
deadlock) and we can use multiple threads.

Usage:
    python compare_linear.py --n_folds 10 --out results/latest/benchmarks
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "6")

import warnings
warnings.filterwarnings("ignore")
import numpy as np

from run_benchmark import load_dataset, _outer_splits
from gpu_tangent import GPUTangentFC


def make_models(seed):
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.svm import LinearSVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    inner = 3
    grid_C = {"C": [0.003, 0.01, 0.03, 0.1]}
    models = {
        "logistic (l2)": GridSearchCV(
            LogisticRegression(solver="liblinear", max_iter=2000),
            grid_C, cv=inner, scoring="roc_auc", n_jobs=3),
        "linear SVM": GridSearchCV(
            LinearSVC(max_iter=5000), grid_C, cv=inner, scoring="roc_auc", n_jobs=3),
        "ridge": GridSearchCV(
            RidgeClassifier(), {"alpha": [1.0, 10.0, 100.0, 1000.0]},
            cv=inner, scoring="roc_auc", n_jobs=3),
        "elastic-net logistic": GridSearchCV(
            LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                               max_iter=3000),
            {"C": [0.01, 0.03, 0.1]}, cv=inner, scoring="roc_auc", n_jobs=3),
        # full-D shrinkage LDA needs a 20k x 20k covariance; PCA-reduce first (the
        # signal is rank ~50 so this is lossless) then Ledoit-Wolf shrinkage LDA.
        "shrinkage LDA (PCA)": Pipeline([
            ("pca", PCA(n_components=200, random_state=seed)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))]),
    }
    return models


def score_vec(clf, X):
    """Probability-like ranking score for AUROC (proba if available, else margin)."""
    if hasattr(clf, "predict_proba"):
        try:
            return clf.predict_proba(X)[:, 1]
        except Exception:
            pass
    return clf.decision_function(X)


def main() -> None:
    ap = argparse.ArgumentParser(description="linear-model ablation + ensemble")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_dir", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/latest/benchmarks")
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score
    from tqdm import tqdm

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_dir))
    gt = GPUTangentFC(device="cpu", max_iter=10)
    covs = gt.compute_covariances(ts)
    splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)

    names = list(make_models(args.seed).keys()) + ["ENSEMBLE (mean of linear)"]
    auc = {n: [] for n in names}; acc = {n: [] for n in names}

    for tr, te in tqdm(splits, desc="pooled folds", unit="fold"):
        gt.fit(covs[tr])
        Xtr, Xte = gt.transform(covs[tr]), gt.transform(covs[te])
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        member_scores = []
        for name, clf in make_models(args.seed).items():
            clf.fit(Xtr, y[tr])
            s = score_vec(clf, Xte)
            auc[name].append(roc_auc_score(y[te], s))
            acc[name].append(accuracy_score(y[te], clf.predict(Xte)))
            # standardize each member's scores for a fair ensemble average
            member_scores.append((s - s.mean()) / (s.std() + 1e-9))

        ens = np.mean(member_scores, axis=0)
        auc["ENSEMBLE (mean of linear)"].append(roc_auc_score(y[te], ens))
        acc["ENSEMBLE (mean of linear)"].append(accuracy_score(y[te], (ens > 0).astype(int)))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print("\n=== linear models on tangent-FC, pooled %d-fold CV ===" % args.n_folds)
    rows = []
    for n in names:
        m, s = float(np.mean(auc[n])), float(np.std(auc[n]))
        am, as_ = float(np.mean(acc[n])), float(np.std(acc[n]))
        print(f"  {n:26s}  AUROC {m:.3f} ± {s:.3f}   acc {am:.3f} ± {as_:.3f}")
        rows.append({"classifier": n, "auroc_mean": m, "auroc_std": s,
                     "acc_mean": am, "acc_std": as_})
    (out / "linear_comparison.json").write_text(json.dumps(rows, indent=2))
    print("report ->", out / "linear_comparison.json")

    # figure
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualization.style import ieee_style, COLORS
        with ieee_style():
            order = sorted(rows, key=lambda r: r["auroc_mean"])
            labels = [r["classifier"] for r in order]
            vals = [r["auroc_mean"] for r in order]
            errs = [r["auroc_std"] for r in order]
            colors = [COLORS.get("asd", "#c0392b") if "ENSEMBLE" in l else "#5b8db8"
                      for l in labels]
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            yb = np.arange(len(labels))
            ax.barh(yb, vals, xerr=errs, color=colors, edgecolor="black",
                    linewidth=0.7, capsize=3)
            for yi, v in zip(yb, vals):
                ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.5)
            ax.set_yticks(yb); ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlim(0.5, max(vals) + 0.06)
            ax.axvline(0.5, ls=":", lw=1.0, color="gray")
            ax.set_xlabel("AUROC (pooled %d-fold CV)" % args.n_folds)
            ax.set_title("Linear models cluster on tangent-FC")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out / f"linear_comparison.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
        print("figure ->", out / "linear_comparison.png")
    except Exception as exc:
        print("figure failed:", exc)


if __name__ == "__main__":
    main()
