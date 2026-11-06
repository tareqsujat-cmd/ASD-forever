"""
Classifier comparison on tangent-FC (publication ablation — supports "simple wins").

Runs the SAME leakage-free nested pooled CV on the SAME in-fold tangent-FC features,
swapping only the classifier: ℓ2-logistic vs Random Forest vs XGBoost.  Shows that
tree ensembles do not beat (and typically underperform) the linear model on the
high-dimensional (p >> n), low-rank, linear ABIDE-I connectome signal.

Usage:
    python compare_classifiers.py --n_folds 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# torch (gpu_tangent) and xgboost/catboost each bundle their OWN OpenMP runtime.
# On macOS two libomp copies deadlock at the thread barrier, so we (a) permit the
# duplicate runtime and (b) force single-threaded OpenMP -> no worker barrier, no
# deadlock.  BLAS is likewise capped.  Must be set before numpy/torch import.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

from run_benchmark import load_dataset, _outer_splits
from gpu_tangent import GPUTangentFC


def main() -> None:
    ap = argparse.ArgumentParser(description="logistic vs RF vs XGBoost on tangent-FC")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_dir", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score
    from tqdm import tqdm
    from utilities.run_dir import create_run_dir, write_manifest

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_dir))
    gt = GPUTangentFC(device="cpu", max_iter=10)
    covs = gt.compute_covariances(ts)                       # once
    splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)

    def make():
        clfs = {
            "logistic (linear)": LogisticRegression(C=0.01, solver="liblinear", max_iter=2000),
            "random_forest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                                    n_jobs=1, random_state=args.seed),
        }
        try:
            import xgboost as xgb
            clfs["xgboost"] = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.3, n_jobs=6, random_state=args.seed,
                eval_metric="logloss", verbosity=0)
        except Exception:
            pass
        try:
            from catboost import CatBoostClassifier
            clfs["catboost"] = CatBoostClassifier(
                iterations=300, depth=4, learning_rate=0.1, rsm=0.3,
                thread_count=6, random_seed=args.seed, verbose=0,
                allow_writing_files=False)
        except Exception:
            pass
        return clfs

    names = list(make().keys())
    auc = {n: [] for n in names}; acc = {n: [] for n in names}
    for tr, te in tqdm(splits, desc="pooled folds", unit="fold"):
        gt.fit(covs[tr])
        Xtr, Xte = gt.transform(covs[tr]), gt.transform(covs[te])
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        for name, clf in make().items():
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            auc[name].append(roc_auc_score(y[te], p))
            acc[name].append(accuracy_score(y[te], (p >= 0.5).astype(int)))

    run_dir = create_run_dir(args.out, run_name="classifier_comparison")
    write_manifest(run_dir, seed=args.seed, device="cpu", args=args, mode="compare")
    print("\n=== tangent-FC, pooled %d-fold CV ===" % args.n_folds)
    rows = []
    for n in names:
        m, s = float(np.mean(auc[n])), float(np.std(auc[n]))
        am, as_ = float(np.mean(acc[n])), float(np.std(acc[n]))
        print(f"  {n:20s}  AUROC {m:.3f} ± {s:.3f}   acc {am:.3f} ± {as_:.3f}")
        rows.append({"classifier": n, "auroc_mean": m, "auroc_std": s,
                     "acc_mean": am, "acc_std": as_})
    (run_dir / "classifier_comparison.json").write_text(json.dumps(rows, indent=2))
    print("report ->", run_dir / "classifier_comparison.json")

    _plot(rows, auc, acc, names, run_dir, args.n_folds)


def _plot(rows, auc, acc, names, run_dir, n_folds):
    """Grouped bar chart: AUROC & accuracy per classifier (linear highlighted)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualization.style import ieee_style, COLORS
    except Exception as exc:                                   # pragma: no cover
        print("plot skipped:", exc)
        return

    short = [n.replace(" (linear)", "\n(linear)") for n in names]
    a_m = [np.mean(auc[n]) for n in names]; a_s = [np.std(auc[n]) for n in names]
    c_m = [np.mean(acc[n]) for n in names]; c_s = [np.std(acc[n]) for n in names]
    x = np.arange(len(names)); w = 0.38
    # highlight the linear model; tree ensembles muted
    bar_c = [COLORS.get("asd", "#c0392b") if "linear" in n else "#7f8c8d" for n in names]

    with ieee_style():
        fig, ax = plt.subplots(figsize=(1.5 * len(names) + 1.2, 3.4))
        b1 = ax.bar(x - w / 2, a_m, w, yerr=a_s, capsize=3, color=bar_c,
                    edgecolor="black", linewidth=0.8, label="AUROC")
        b2 = ax.bar(x + w / 2, c_m, w, yerr=c_s, capsize=3, color=bar_c, alpha=0.5,
                    hatch="//", edgecolor="black", linewidth=0.8, label="Accuracy")
        for xi, m in zip(x - w / 2, a_m):
            ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(0.5, ls=":", lw=1.0, color="gray")
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
        ax.set_ylim(0.45, max(a_m) + 0.08)
        ax.set_ylabel("Score (pooled %d-fold CV)" % n_folds)
        ax.set_title("Classifier comparison on tangent-FC")
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(run_dir / f"classifier_comparison.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("figure ->", run_dir / "classifier_comparison.png")


if __name__ == "__main__":
    main()
