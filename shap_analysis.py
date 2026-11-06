"""
SHAP interpretation of the ASD connectome classifier.

For our LINEAR model, SHAP values are exact: phi_ij = coef_ij * (x_ij - E[x_ij]).
So SHAP does not change the *global* edge ranking we already report from coefficients
(we verify this), but it gives two things the coefficient map cannot:

  1. **Per-subject** attributions — *why this person* was called ASD/TC — which we tie
     to selective prediction (a confident-ASD, a confident-TC, and an abstained case).
  2. A standardized, reviewer-recognized attribution, aggregated to the **Yeo-7**
     network level so credit is stable across correlated edges.

We also run **TreeSHAP** on the RF / XGBoost / CatBoost models from the classifier
comparison and show their edge-importance ranking correlates with the linear model's
— i.e. the tree ensembles lean on the *same* signal (supports "simple wins").

All SHAP values are computed **out-of-fold** (explainer background = training fold only),
so nothing leaks.

Usage:
    python shap_analysis.py --n_folds 10 --atlas cc200
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# torch (gpu_tangent) + xgboost/catboost each bundle their own OpenMP runtime; on
# macOS two libomp copies deadlock at the thread barrier.  Permit the duplicate and
# force single-threaded OpenMP.  Must precede numpy/torch import.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import warnings
warnings.filterwarnings("ignore")
import numpy as np

from run_benchmark import load_dataset, _outer_splits
from gpu_tangent import GPUTangentFC

N_ROIS = 200


# --------------------------- out-of-fold linear SHAP ---------------------------

def oof_linear_shap(covs, y, splits, C):
    """Leakage-free per-subject SHAP for the linear model + validation vs coefficients."""
    import shap
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    n = len(y)
    d = covs.shape[1] * (covs.shape[1] - 1) // 2
    gt = GPUTangentFC(device="cpu", max_iter=10)
    sv = np.zeros((n, d), dtype=np.float32)
    proba = np.zeros(n, dtype=np.float32)
    corrs = []

    from tqdm import tqdm
    for tr, te in tqdm(splits, desc="OOF linear SHAP", unit="fold"):
        gt.fit(covs[tr])
        Xtr, Xte = gt.transform(covs[tr]), gt.transform(covs[te])
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(Xtr_s, y[tr])

        expl = shap.LinearExplainer(clf, Xtr_s)          # background = train fold
        vals = expl.shap_values(Xte_s)
        if isinstance(vals, list):                       # some versions return [class0,class1]
            vals = vals[-1]
        sv[te] = vals
        proba[te] = clf.predict_proba(Xte_s)[:, 1]

        # validation: SHAP == coef * (x - background_mean) for a linear model
        expected = clf.coef_.ravel()[None, :] * (Xte_s - Xtr_s.mean(0)[None, :])
        corrs.append(float(np.corrcoef(vals.ravel(), expected.ravel())[0, 1]))

    return sv, proba, float(np.mean(corrs))


# --------------------------- Yeo-7 aggregation ---------------------------

def yeo_aggregate(mean_abs_edge):
    """Aggregate |SHAP| per edge into a Yeo-7 network x network grid (reuses viz_brain)."""
    try:
        from viz_brain import cc200_atlas_img, roi_coords, assign_yeo, YEO7
        atlas_img, _ = cc200_atlas_img()
        coords = roi_coords(atlas_img)
        if len(coords) != N_ROIS:
            return None, None
        yeo = assign_yeo(coords)
    except Exception as exc:
        print("Yeo aggregation skipped:", exc)
        return None, None
    iu = np.triu_indices(N_ROIS, k=1)
    nets = sorted(set(int(v) for v in yeo if v > 0))
    idx = {nn: i for i, nn in enumerate(nets)}
    M = np.zeros((len(nets), len(nets)))
    for e, (a, b) in enumerate(zip(iu[0], iu[1])):
        na, nb = yeo[a], yeo[b]
        if na > 0 and nb > 0:
            M[idx[na], idx[nb]] += mean_abs_edge[e]
            M[idx[nb], idx[na]] += mean_abs_edge[e]
    labels = [YEO7.get(nn, str(nn)) for nn in nets]
    return M, labels


# --------------------------- figures ---------------------------

def _edge_labels(top_idx):
    iu = np.triu_indices(N_ROIS, k=1)
    return [f"R{iu[0][e]}–R{iu[1][e]}" for e in top_idx]


def plot_waterfall(ax, sv_row, proba, title, k=12):
    """Top-k signed SHAP contributions for one subject (custom, style-consistent)."""
    order = np.argsort(np.abs(sv_row))[::-1][:k]
    vals = sv_row[order]
    labels = _edge_labels(order)
    colors = ["#c0392b" if v > 0 else "#2c6fbb" for v in vals]  # red=->ASD, blue=->TC
    ypos = np.arange(len(vals))[::-1]
    ax.barh(ypos, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP (log-odds toward ASD)", fontsize=8)
    ax.set_title(f"{title}\np(ASD)={proba:.2f}", fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser(description="SHAP interpretation (linear + TreeSHAP)")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_dir", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--atlas", default="cc200")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="defaults to results/latest/analyses/shap")
    ap.add_argument("--tree_shap", action="store_true", help="also run TreeSHAP on RF/XGB/CatBoost")
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out) if args.out else Path("results/latest/analyses/shap")
    out.mkdir(parents=True, exist_ok=True)

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_dir))
    gt = GPUTangentFC(device="cpu", max_iter=10)
    covs = gt.compute_covariances(ts)
    splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)

    # 1) OOF linear SHAP + validation
    sv, proba, coef_corr = oof_linear_shap(covs, y, splits, args.C)
    print(f"\nSHAP-vs-coefficient validation: mean r = {coef_corr:.4f} (expect ~1.000)")

    mean_abs = np.abs(sv).mean(0)                        # global importance per edge
    top = np.argsort(mean_abs)[::-1][:20]

    # 2) global Yeo-7 grid
    M, labels = yeo_aggregate(mean_abs)
    if M is not None:
        try:
            from visualization.style import ieee_style
            with ieee_style():
                fig, ax = plt.subplots(figsize=(4.4, 3.8))
                im = ax.imshow(M, cmap="magma")
                ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(labels, fontsize=8)
                ax.set_title("Mean |SHAP| by network (Yeo-7)")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                for ext in ("png", "pdf"):
                    fig.savefig(out / f"shap_yeo_networks.{ext}", dpi=300, bbox_inches="tight")
                plt.close(fig)
            print("Yeo grid ->", out / "shap_yeo_networks.png")
        except Exception as exc:
            print("Yeo figure failed:", exc)

    # 3) per-subject waterfalls: confident ASD, confident TC, abstained
    conf = np.abs(proba - 0.5)
    pred = (proba >= 0.5).astype(int)
    correct = pred == y
    asd_ok = np.where(correct & (y == 1))[0]
    tc_ok = np.where(correct & (y == 0))[0]
    i_asd = asd_ok[np.argmax(conf[asd_ok])]              # most confident correct ASD
    i_tc = tc_ok[np.argmax(conf[tc_ok])]                 # most confident correct TC
    i_abs = int(np.argmin(conf))                         # most uncertain (would abstain)
    try:
        from visualization.style import ieee_style
        with ieee_style():
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
            plot_waterfall(axes[0], sv[i_asd], proba[i_asd], "Confident ASD")
            plot_waterfall(axes[1], sv[i_tc], proba[i_tc], "Confident control")
            plot_waterfall(axes[2], sv[i_abs], proba[i_abs], "Abstained (uncertain)")
            fig.suptitle("Per-subject SHAP explanations (top connections)", fontsize=10)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(out / f"shap_subject_waterfalls.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
        print("waterfalls ->", out / "shap_subject_waterfalls.png")
    except Exception as exc:
        print("waterfall figure failed:", exc)

    report = {
        "shap_vs_coef_corr": coef_corr,
        "n_subjects": int(len(y)), "n_edges": int(sv.shape[1]),
        "top_edges": [{"edge_index": int(e), "roi_a": int(np.triu_indices(N_ROIS, 1)[0][e]),
                       "roi_b": int(np.triu_indices(N_ROIS, 1)[1][e]),
                       "mean_abs_shap": float(mean_abs[e])} for e in top],
        "example_subjects": {"confident_asd": int(i_asd), "confident_tc": int(i_tc),
                             "abstained": int(i_abs)},
    }

    # 4) optional TreeSHAP: do RF/XGB/CatBoost rank the same edges as linear?
    if args.tree_shap:
        report["tree_shap_rank_corr"] = tree_shap_corr(covs, y, mean_abs, args.seed, out)

    (out / "shap_report.json").write_text(json.dumps(report, indent=2))
    print("report ->", out / "shap_report.json")


def tree_shap_corr(covs, y, linear_importance, seed, out):
    """Fit RF/XGB/CatBoost on one split, TreeSHAP global importance, Spearman vs linear."""
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import spearmanr

    gt = GPUTangentFC(device="cpu", max_iter=10)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    gt.fit(covs[tr])
    Xtr, Xte = gt.transform(covs[tr]), gt.transform(covs[te])
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    models = {"random_forest": RandomForestClassifier(n_estimators=200, n_jobs=1,
                                                      random_state=seed)}
    try:
        import xgboost as xgb
        models["xgboost"] = xgb.XGBClassifier(n_estimators=200, max_depth=4, n_jobs=1,
                                              learning_rate=0.1, eval_metric="logloss",
                                              verbosity=0, random_state=seed)
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(iterations=200, depth=4, thread_count=1,
                                                learning_rate=0.1, verbose=0,
                                                allow_writing_files=False, random_seed=seed)
    except Exception:
        pass

    corr = {}
    for name, m in models.items():
        try:
            m.fit(Xtr, y[tr])
            expl = shap.TreeExplainer(m)
            vals = expl.shap_values(Xte)
            if isinstance(vals, list):
                vals = vals[-1]
            imp = np.abs(np.asarray(vals)).mean(0)
            rho = float(spearmanr(imp, linear_importance).correlation)
            corr[name] = rho
            print(f"TreeSHAP {name}: Spearman edge-importance vs linear = {rho:.3f}")
        except Exception as exc:
            print(f"TreeSHAP {name} failed:", exc)
    return corr


if __name__ == "__main__":
    main()
