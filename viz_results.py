"""
Publication figures from a run's out-of-fold predictions (scienceplots, Q1/ICLR).

Reads a ``*_oof.json`` (written by run_benchmark.py / train_sota.py: a list of
``{"protocol", "oof_true", "oof_prob", "oof_site"}``) and produces, per protocol,
vector + raster figures styled with scienceplots (science+ieee) and the bolder
Q1/ICLR rcParams from visualization.style:

  roc_<proto>          ROC curve (AUROC in legend)
  pr_<proto>           Precision-Recall curve (AUPRC)
  calibration_<proto>  reliability curve + ECE
  confusion_<proto>    confusion matrix at the Youden-J threshold
  per_site_<proto>     per-site AUROC bar chart

Usage
-----
    python viz_results.py --oof results/run_N/sota/sota_oof.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.style import ieee_style, COLORS, PALETTE, SINGLE_COL_W

FORMATS = ("pdf", "png")


def _save(fig, out: Path, name: str) -> None:
    for ext in FORMATS:
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def fig_roc(y, p, proto, out):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y, p)
    a = roc_auc_score(y, p)
    with ieee_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.4, SINGLE_COL_W + 0.1))
        ax.plot(fpr, tpr, color=COLORS["asd"], lw=2.6, label=f"Proposed (AUC = {a:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=COLORS["random"], label="Chance")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC — {proto}")
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.legend(loc="lower right")
        _save(fig, out, f"roc_{proto}")


def fig_pr(y, p, proto, out):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    base = float(np.mean(y))
    with ieee_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.4, SINGLE_COL_W + 0.1))
        ax.plot(rec, prec, color=COLORS["model_c"], lw=2.6, label=f"Proposed (AP = {ap:.3f})")
        ax.axhline(base, ls="--", lw=1.4, color=COLORS["random"], label=f"Prevalence = {base:.2f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall — {proto}")
        ax.set_ylim(0, 1.02); ax.legend(loc="lower left")
        _save(fig, out, f"pr_{proto}")


def fig_calibration(y, p, proto, out, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    xs, ys, ece = [], [], 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = p[m].mean(); acc = y[m].mean()
        xs.append(conf); ys.append(acc)
        ece += (m.sum() / len(y)) * abs(acc - conf)
    with ieee_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 0.4, SINGLE_COL_W + 0.1))
        ax.plot([0, 1], [0, 1], ls="--", lw=1.4, color=COLORS["random"], label="Perfect")
        ax.plot(xs, ys, "-o", color=COLORS["model_a"], lw=2.6, label=f"Proposed (ECE = {ece:.3f})")
        ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"Calibration — {proto}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left")
        _save(fig, out, f"calibration_{proto}")


def fig_confusion(y, p, proto, out):
    from sklearn.metrics import confusion_matrix
    thr = _youden_threshold(y, p)
    cm = confusion_matrix(y, (p >= thr).astype(int))
    with ieee_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["TC", "ASD"]); ax.set_yticklabels(["TC", "ASD"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion (Youden) — {proto}")
        _save(fig, out, f"confusion_{proto}")


def fig_per_site(y, p, sites, proto, out):
    from sklearn.metrics import roc_auc_score
    uniq = sorted(set(sites))
    rows = []
    for s in uniq:
        m = sites == s
        if m.sum() >= 5 and len(set(y[m])) > 1:
            rows.append((str(s), roc_auc_score(y[m], p[m]), int(m.sum())))
    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]; aucs = [r[1] for r in rows]
    with ieee_style():
        fig, ax = plt.subplots(figsize=(SINGLE_COL_W + 1.2, max(2.2, 0.32 * len(rows))))
        ax.barh(labels, aucs, color=COLORS["model_a"], edgecolor="black", linewidth=1.0)
        ax.axvline(0.5, ls="--", lw=1.4, color=COLORS["random"])
        ax.set_xlabel("AUROC"); ax.set_title(f"Per-site AUROC — {proto}")
        ax.set_xlim(0.4, 1.0)
        _save(fig, out, f"per_site_{proto}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Q1/ICLR result figures (scienceplots)")
    ap.add_argument("--oof", required=True, help="path to *_oof.json")
    ap.add_argument("--out", default=None, help="output dir (default: <oof_dir>/figures)")
    args = ap.parse_args()

    data = json.load(open(args.oof))
    out = Path(args.out or (Path(args.oof).parent / "figures"))
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for entry in data:
        proto = entry.get("protocol", "run")
        y = np.asarray(entry["oof_true"]); p = np.asarray(entry["oof_prob"], dtype=float)
        sites = np.asarray(entry.get("oof_site", []))
        if len(y) == 0 or len(set(y)) < 2:
            continue
        fig_roc(y, p, proto, out); fig_pr(y, p, proto, out)
        fig_calibration(y, p, proto, out); fig_confusion(y, p, proto, out)
        if len(sites) == len(y):
            fig_per_site(y, p, sites, proto, out)
        n += 1
    print(f"Wrote figures for {n} protocol(s) -> {out}")


if __name__ == "__main__":
    main()
