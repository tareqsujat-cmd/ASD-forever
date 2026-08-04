"""
Calibrated selective prediction for ASD classification (publication contribution — Option A).

A population-level ~0.77 AUROC classifier is not clinically usable on *everyone*.  But if
the model knows *when* it is reliable, it can make a confident call on a subset and abstain
on the rest — a high-precision, deployable tool.  This module turns a run's out-of-fold
predictions into that tool:

  - calibration quality (ECE + reliability curve),
  - a risk–coverage curve (error vs. fraction of subjects the model chooses to answer),
  - AUROC / accuracy **at coverage** (e.g. "AUROC 0.90 on the 50% most-confident subjects"),
  - selective metrics summarised for the paper.

Everything operates on the *out-of-fold* predictions (already held-out, leakage-free), so the
confidence ranking is honest.  Confidence = max(p, 1-p).

Usage:
    python selective_prediction.py --oof results/final/sota/sota_oof.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def confidence(prob: np.ndarray) -> np.ndarray:
    return np.maximum(prob, 1.0 - prob)          # in [0.5, 1.0]


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    ece, xs, ys, ns = 0.0, [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = prob[m].mean(); acc = y[m].mean()
        ece += (m.sum() / len(y)) * abs(acc - conf)
        xs.append(conf); ys.append(acc); ns.append(int(m.sum()))
    return float(ece), xs, ys, ns


def risk_coverage(y: np.ndarray, prob: np.ndarray, thr: float = 0.5,
                  coverages=None) -> list:
    from sklearn.metrics import roc_auc_score, accuracy_score
    coverages = coverages or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    order = np.argsort(-confidence(prob))        # most-confident first
    y_s, p_s = y[order], prob[order]
    n = len(y)
    rows = []
    for cov in coverages:
        k = max(5, int(round(cov * n)))
        yk, pk = y_s[:k], p_s[:k]
        predk = (pk >= thr).astype(int)
        auc = float(roc_auc_score(yk, pk)) if len(set(yk)) > 1 else float("nan")
        rows.append({
            "coverage": round(cov, 2), "n": int(k),
            "accuracy": float(accuracy_score(yk, predk)),
            "auroc": auc,
            "selective_risk": float(1.0 - accuracy_score(yk, predk)),
        })
    return rows


def _figures(y, prob, rc, ece_xs, ece_ys, ece_val, out: Path, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.style import ieee_style, COLORS, SINGLE_COL_W
    out.mkdir(parents=True, exist_ok=True)
    covs = [r["coverage"] for r in rc]
    aucs = [r["auroc"] for r in rc]
    accs = [r["accuracy"] for r in rc]
    risks = [r["selective_risk"] for r in rc]

    # AUROC / accuracy at coverage
    with ieee_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.6, SINGLE_COL_W))
        ax.plot(covs, aucs, "-o", color=COLORS["asd"], lw=2.6, label="AUROC")
        ax.plot(covs, accs, "-s", color=COLORS["model_a"], lw=2.6, label="Accuracy")
        ax.set_xlabel("Coverage (fraction answered)"); ax.set_ylabel("Performance")
        ax.set_title(f"Performance at coverage — {tag}")
        ax.invert_xaxis(); ax.legend(loc="lower left")
        for ext in ("pdf", "png"):
            fig.savefig(out / f"coverage_{tag}.{ext}", bbox_inches="tight")
        plt.close(fig)

    # risk-coverage
    with ieee_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.6, SINGLE_COL_W))
        ax.plot(covs, risks, "-o", color=COLORS["model_b"], lw=2.6)
        ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (error)")
        ax.set_title(f"Risk–coverage — {tag}")
        for ext in ("pdf", "png"):
            fig.savefig(out / f"risk_coverage_{tag}.{ext}", bbox_inches="tight")
        plt.close(fig)

    # reliability
    with ieee_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.4, SINGLE_COL_W))
        ax.plot([0, 1], [0, 1], "--", lw=1.4, color=COLORS["random"], label="Perfect")
        ax.plot(ece_xs, ece_ys, "-o", color=COLORS["model_a"], lw=2.6,
                label=f"ECE = {ece_val:.3f}")
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"Calibration — {tag}"); ax.legend(loc="upper left")
        for ext in ("pdf", "png"):
            fig.savefig(out / f"calibration_{tag}.{ext}", bbox_inches="tight")
        plt.close(fig)


def analyze(y, prob, out: Path, tag: str) -> dict:
    y = np.asarray(y).astype(int); prob = np.asarray(prob, float)
    ece_val, xs, ys, ns = expected_calibration_error(y, prob)
    rc = risk_coverage(y, prob)
    _figures(y, prob, rc, xs, ys, ece_val, out, tag)
    full = next(r for r in rc if r["coverage"] == 1.0)
    half = min(rc, key=lambda r: abs(r["coverage"] - 0.5))
    summary = {
        "tag": tag, "n": int(len(y)), "ece": ece_val,
        "auroc_full": full["auroc"], "acc_full": full["accuracy"],
        "auroc_at_50pct": half["auroc"], "acc_at_50pct": half["accuracy"],
        "risk_coverage": rc,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrated selective prediction (Option A)")
    ap.add_argument("--oof", required=True, help="path to *_oof.json (list of protocol dicts)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = json.load(open(args.oof))
    out = Path(args.out or (Path(args.oof).parent / "selective"))
    results = []
    for entry in data:
        tag = entry.get("protocol", "run")
        y = entry["oof_true"]; p = entry["oof_prob"]
        if len(set(y)) < 2:
            continue
        s = analyze(y, p, out, tag)
        results.append(s)
        print(f"[{tag}]  AUROC full={s['auroc_full']:.3f}  @50%={s['auroc_at_50pct']:.3f}  |  "
              f"acc full={s['acc_full']:.3f}  @50%={s['acc_at_50pct']:.3f}  |  ECE={s['ece']:.3f}")
    (out).mkdir(parents=True, exist_ok=True)
    (out / "selective_report.json").write_text(json.dumps(results, indent=2))
    print("report ->", out / "selective_report.json")


if __name__ == "__main__":
    main()
