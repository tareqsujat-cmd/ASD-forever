"""
Methodology diagram: the analysis pipeline from the released scans to a class decision.

Draws the primary procedure as a single figure — sample definition, preprocessing, the
three connectivity descriptions, the per-description classifiers and their combination,
and the decision at the default threshold — with the evaluation protocol shown alongside
because it applies to every stage.  Secondary and exploratory analyses (probability
calibration, abstention thresholds, subgroup and motion-matched analyses, coefficient and
SHAP summaries) are shown in a separate box because they are not part of the model.

Usage:
    python viz_pipeline.py --out results/latest_study/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# muted, print-safe palette
DATA = "#dbe6f1"      # data and preprocessing
REPR = "#e6dcef"      # connectivity descriptions
MODEL = "#dcece0"     # models and combination
DECIDE = "#f6e2d8"    # decision
INTERP = "#f2eddc"    # secondary and exploratory analyses
PROTO = "#ececec"     # evaluation protocol
EDGE = "#5a5a5a"


def box(ax, x, y, w, h, title, body="", face=DATA, fs_title=9.0, fs_body=7.6,
        bold=True, edge=EDGE, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))
    if body:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=fs_title, fontweight="bold" if bold else "normal", zorder=3)
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center",
                fontsize=fs_body, zorder=3, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs_title, fontweight="bold" if bold else "normal", zorder=3,
                linespacing=1.45)
    return (x + w / 2, y, x + w / 2, y + h)          # bottom-centre, top-centre


def panel(ax, x, y, w, h, title, lines, face=PROTO, fs_title=9.2, fs_body=7.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.0, edgecolor=EDGE, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="center",
            fontsize=fs_title, fontweight="bold", zorder=3)
    body = "\n".join(lines)
    ax.text(x + w / 2, y + (h - 0.44) / 2 + 0.02, body, ha="center", va="center",
            fontsize=fs_body, zorder=3, linespacing=1.5)


def arrow(ax, p0, p1, color=EDGE, lw=1.1, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def build(out: Path, name="fig00_pipeline"):
    fig, ax = plt.subplots(figsize=(10.4, 11.1))
    ax.set_xlim(0, 10); ax.set_ylim(2.20, 13.35); ax.axis("off")

    L, W = 0.35, 6.55            # main column
    px, pw = 7.25, 2.45          # protocol column

    # ---------------- sample ----------------
    b1 = box(ax, L, 12.25, W, 0.72, "ABIDE-I release",
             "1,112 participants · resting-state fMRI · 17 acquisition sites (20 site labels)",
             face=DATA)
    b2 = box(ax, L, 11.25, W, 0.78, "Quality control",
             "every required quality rating must pass\n"
             "241 records lacking a required passing rating are excluded", face=DATA)
    b3 = box(ax, L, 10.30, W, 0.66, "Strict cohort",
             "871 participants — 403 autistic / 468 non-autistic", face=DATA)

    # ---------------- preprocessing ----------------
    b4 = box(ax, L, 9.32, W, 0.72, "Preprocessing (C-PAC derivatives)",
             "band-pass filtered · global signal regression not applied", face=DATA)
    b5 = box(ax, L, 8.34, W, 0.72, "Atlas parcellation",
             "CC200 (200 regions)  →  regional time series", face=DATA)

    for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5)):
        arrow(ax, (a[0], a[1]), (b[0], b[3]))

    # ---------------- three descriptions ----------------
    cw, gap = 2.05, 0.20
    cx = [L, L + cw + gap, L + 2 * (cw + gap)]
    c1 = box(ax, cx[0], 6.62, cw, 1.28, "Correlation",
             "Pearson correlation\nbetween regional\ntime series", face=REPR)
    c2 = box(ax, cx[1], 6.62, cw, 1.28, "Tangent embedding",
             "Ledoit–Wolf covariance\nlog map at the group mean\n19,900 connections",
             face=REPR, fs_body=7.2)
    c3 = box(ax, cx[2], 6.62, cw, 1.28, "Windowed tangent",
             "6 windows × 50 volumes\nstride set per scan so the\ncount is equal for all\nmean over windows",
             face=REPR, fs_body=7.2)
    for c in (c1, c2, c3):
        arrow(ax, (b5[0], b5[1]), (c[0], c[3]), rad=0.0)

    # ---------------- per-branch model ----------------
    m = []
    for i, c in enumerate((c1, c2, c3)):
        mi = box(ax, cx[i], 5.62, cw, 0.72, "standardise → classify",
                 "$\\ell_2$-logistic\npenalty by inner CV", face=MODEL, fs_title=8.2,
                 fs_body=7.0)
        arrow(ax, (c[0], c[1]), (mi[0], mi[3]))
        m.append(mi)

    b6 = box(ax, L, 4.62, W, 0.70, "Combine",
             "average the three probabilities  ·  composition fixed in advance",
             face=MODEL)
    for mi in m:
        arrow(ax, (mi[0], mi[1]), (b6[0], b6[3]), rad=0.0)

    b7 = box(ax, L, 3.66, W, 0.68, "Decision",
             "averaged probability  →  class label at the default threshold of 0.5",
             face=DECIDE)
    arrow(ax, (b6[0], b6[1]), (b7[0], b7[3]))

    # ---------------- secondary and exploratory analyses (not part of the model) -------
    box(ax, L, 2.35, W, 0.98, "Secondary and exploratory analyses",
        "probability calibration · abstention thresholds · IQ and sex subgroups\n"
        "motion-matched system contrasts · coefficient and SHAP summaries",
        face=INTERP, fs_body=7.4)

    # ---------------- protocol column ----------------
    panel(ax, px, 10.25, pw, 2.72, "Evaluation protocol",
          ["Every learned quantity —", "reference, scaling,",
           "harmonisation, classifier —", "is fitted on the training",
           "part of the fold only.", "",
           "Hyperparameters are set", "by a further split inside",
           "that training part."])

    panel(ax, px, 7.95, pw, 2.10, "Two splits",
          ["Pooled 10-fold —", "participants divided", "at random", "",
           "Leave-one-site-out —", "a whole site withheld,", "so the model meets",
           "an unseen scanner"])

    panel(ax, px, 5.95, pw, 1.80, "Reported",
          ["AUROC · accuracy", "sensitivity · specificity", "calibration error", "",
           "differences tested with", "DeLong and McNemar"])

    panel(ax, px, 2.35, pw, 3.40, "Also varied",
          ["connectivity description", "classifier family",
           "number of components", "atlas and fusion rule", "site harmonisation", "",
           "and, for the protocol itself,", "feature-selection scope,",
           "fitting steps before the split,", "and the two quality-control",
           "cohorts"])

    ax.plot([px - 0.28, px - 0.28], [2.35, 12.97], color="#bbbbbb", lw=1.0, ls=(0, (4, 3)))

    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Methodology pipeline diagram")
    ap.add_argument("--out", default="results/latest_study/figures")
    ap.add_argument("--name", default="fig00_pipeline")
    args = ap.parse_args()
    build(Path(args.out), args.name)


if __name__ == "__main__":
    main()
