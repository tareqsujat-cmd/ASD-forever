"""
Explainability: discriminative functional-connectivity edges (publication E6).

Fits an ℓ2-logistic classifier on the FC vectors inside each CV fold and records
the per-edge coefficients.  Edges are ranked by mean |coefficient| across folds,
and their **cross-fold stability** (how consistently an edge lands in the top-K)
is reported — a stable edge is a trustworthy biomarker, not fold noise.

Outputs (under results/run_N/explainability_edges/):
  - top_edges.csv          : ROI_i, ROI_j, mean_coef, sign, stability
  - edge_importance.png    : 200x200 importance heatmap (scienceplots)
  - edge_report.json       : summary

This operates on the saved Fisher-z correlation FC (abide_processed/mri/*.npy),
so it is fast and needs no retraining.  Direction (sign) indicates whether higher
connectivity pushes toward ASD (+) or TC (-).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("edges")

N_ROIS = 200


def _load_fc(processed_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    meta = pd.read_csv(processed_dir / "mri" / "metadata.csv")
    X, y = [], []
    for _, r in meta.iterrows():
        f = processed_dir / "mri" / f"{r['subject_id']}.npy"
        if not f.exists():
            continue
        v = np.load(f).astype(np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        X.append(v); y.append(int(r["label"]))
    return np.asarray(X), np.asarray(y)


def main() -> None:
    p = argparse.ArgumentParser(description="Discriminative FC edge analysis (XAI)")
    p.add_argument("--processed_dir", default="./abide_processed")
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--run_name", default="explainability_edges")
    args = p.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from utilities.reproducibility import seed_everything
    from utilities.run_dir import create_run_dir, attach_file_logger, write_manifest
    seed_everything(args.seed)

    run_dir = create_run_dir(args.out_dir, run_name=args.run_name)
    attach_file_logger(run_dir)
    write_manifest(run_dir, seed=args.seed, device="cpu", args=args, mode="explainability")
    out = run_dir / "edges"
    out.mkdir(parents=True, exist_ok=True)

    X, y = _load_fc(Path(args.processed_dir))
    logger.info("Loaded %d subjects, %d edges", len(y), X.shape[1])

    iu = np.triu_indices(N_ROIS, k=1)     # maps edge index -> (roi_i, roi_j)
    n_edges = X.shape[1]

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    coefs = np.zeros((args.n_folds, n_edges))
    topk_masks = np.zeros((args.n_folds, n_edges), dtype=bool)

    for k, (tr, _) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.01, solver="liblinear", max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        c = clf.coef_.ravel()
        coefs[k] = c
        top_idx = np.argsort(np.abs(c))[-args.top_k:]
        topk_masks[k, top_idx] = True
        logger.info("  fold %d/%d fitted", k + 1, args.n_folds)

    mean_coef = coefs.mean(axis=0)
    stability = topk_masks.mean(axis=0)                 # fraction of folds in top-K
    rank = np.argsort(np.abs(mean_coef))[::-1][:args.top_k]

    rows = []
    for e in rank:
        rows.append({
            "roi_i": int(iu[0][e]), "roi_j": int(iu[1][e]),
            "mean_coef": float(mean_coef[e]),
            "abs_coef": float(abs(mean_coef[e])),
            "direction": "ASD+" if mean_coef[e] > 0 else "TC+",
            "stability": float(stability[e]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "top_edges.csv", index=False)
    logger.info("\nTop-10 discriminative edges:\n%s",
                df.head(10).to_string(index=False))

    # 200x200 importance heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualization.style import apply_ieee_style
        apply_ieee_style()   # scienceplots
        mat = np.zeros((N_ROIS, N_ROIS))
        mat[iu] = np.abs(mean_coef)
        mat = mat + mat.T
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(mat, cmap="magma", aspect="equal")
        ax.set_title("Discriminative FC edge importance (|mean coef|)")
        ax.set_xlabel("ROI"); ax.set_ylabel("ROI")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"edge_importance.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Heatmap → %s", out / "edge_importance.png")
    except Exception as exc:
        logger.warning("Heatmap generation failed: %s", exc)

    summary = {
        "n_subjects": int(len(y)), "n_edges": int(n_edges), "n_folds": args.n_folds,
        "top_k": args.top_k,
        "n_stable_edges(>=0.8 folds)": int(np.sum(stability >= 0.8)),
        "top_edges": rows[:args.top_k],
    }
    (out / "edge_report.json").write_text(json.dumps(summary, indent=2))
    logger.info("Stable edges (in top-%d for >=80%% of folds): %d",
                args.top_k, summary["n_stable_edges(>=0.8 folds)"])
    logger.info("Report → %s", out / "edge_report.json")


if __name__ == "__main__":
    main()
