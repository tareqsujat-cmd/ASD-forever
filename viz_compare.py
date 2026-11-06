"""
Honest comparison figures for ABIDE-I ASD classification (scienceplots, Q1/ICLR).

Two figures, both designed so no claim outruns the evidence:

1. `sota_comparison` — **accuracy vs accuracy** (a single, consistent metric). Published
   full-cohort accuracies, our full-coverage accuracy, AND our selective accuracy on the
   confident 50% (clearly labelled). We do not claim the highest full-coverage accuracy;
   our genuine edge is the selective operating point.

2. `capability_matrix` — which reliability/rigor analyses each method reports
   (calibration, selective prediction, cross-site LOSO, interpretability). This is where
   we legitimately lead. Competitor cells reflect the primary papers to our reading and
   are marked [VERIFY]; '—' = not reported, not necessarily infeasible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from visualization.style import ieee_style, COLORS

# Best *legitimate* full-cohort ABIDE-I accuracies (no subset/leakage, no weak baselines).
# (name, accuracy, protocol note)
LITERATURE = [
    ("Heinsfeld 2018",      0.700, "full · 10-fold"),
    ("ASD-DiagNet 2019",    0.703, "full · 10-fold"),
    ("Parisot 2018 (GCN)",  0.704, "full · 10-fold"),
    ("ASD-SAENet 2021",     0.708, "full · 10-fold"),
    ("MADE-for-ASD 2024",   0.752, "full · 10-fold"),
]
HEINSFELD_BASELINE = 0.70
BEST_HONEST = 0.752


def plot_sota_comparison(our_acc_full: float, out: str | Path,
                         our_acc_selective: Optional[float] = None,
                         coverage: float = 0.5,
                         our_auroc: Optional[float] = None) -> Path:
    """Accuracy-vs-accuracy comparison with an explicit selective operating point."""
    items = [(n, a, note, "pub") for n, a, note in LITERATURE]
    items.append(("Ours", float(our_acc_full), "full · nested", "ours_full"))
    if our_acc_selective is not None:
        items.append((f"Ours (confident {int(coverage*100)}%)",
                      float(our_acc_selective), "selective · nested", "ours_sel"))
    items = sorted(items, key=lambda x: x[1])

    names = [f"{n}\n({note})" for n, _, note, _ in items]
    accs = [a for _, a, _, _ in items]
    kind = [k for _, _, _, k in items]
    cmap = {"pub": COLORS["model_a"], "ours_full": COLORS["asd"], "ours_sel": "#e8a13a"}
    colors = [cmap[k] for k in kind]
    hatch = ["//" if k == "ours_sel" else "" for k in kind]

    with ieee_style():
        fig, ax = plt.subplots(figsize=(6.8, 0.52 * len(items) + 1.3))
        bars = ax.barh(names, accs, color=colors, edgecolor="black", linewidth=1.1)
        for b, h in zip(bars, hatch):
            if h:
                b.set_hatch(h)
        for b, a in zip(bars, accs):
            ax.text(b.get_width() + 0.004, b.get_y() + b.get_height() / 2,
                    f"{a:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.axvline(HEINSFELD_BASELINE, ls="--", lw=1.6, color=COLORS["random"])
        ax.axvline(BEST_HONEST, ls=":", lw=1.6, color=COLORS["model_c"])
        lo = min(min(accs), HEINSFELD_BASELINE) - 0.03
        hi = max(max(accs), BEST_HONEST) + 0.05
        ax.set_xlim(max(0.5, lo), hi)
        ax.set_xlabel("Accuracy (ABIDE-I)")
        ax.set_title("ASD classification on ABIDE-I")
        legend = [
            Patch(facecolor=COLORS["asd"], edgecolor="black", label="Ours (full coverage)"),
            Patch(facecolor="#e8a13a", edgecolor="black", hatch="//",
                  label=f"Ours (confident {int(coverage*100)}%, selective)"),
            Patch(facecolor=COLORS["model_a"], edgecolor="black", label="Published methods"),
            plt.Line2D([], [], ls="--", color=COLORS["random"], label="Heinsfeld 2018 (0.70)"),
            plt.Line2D([], [], ls=":", color=COLORS["model_c"], label="MADE-for-ASD (0.752)"),
        ]
        ax.legend(handles=legend, loc="lower right", fontsize=7.5, framealpha=0.9)
        cap = ("Single metric (accuracy). Published = full-cohort accuracy under each "
               "paper's own CV (N≈871–1035); ours = strict nested CV.")
        if our_auroc is not None:
            cap += f" Ours AUROC = {our_auroc:.3f}."
        fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=6.8, wrap=True)
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        p = out / "sota_comparison"
        for ext in ("pdf", "png"):
            fig.savefig(f"{p}.{ext}", bbox_inches="tight")
        plt.close(fig)
    return Path(f"{p}.png")


# capability rows: (method, calibration, selective, LOSO, interpretability)
# ✓ reported · ✗ not reported · — unclear.  Competitor cells [VERIFY] against sources.
CAPABILITIES = [
    ("Heinsfeld 2018",    "✗", "✗", "✗", "—"),
    ("ASD-DiagNet 2019",  "✗", "✗", "✗", "—"),
    ("Parisot 2018",      "✗", "✗", "✗", "—"),
    ("ASD-SAENet 2021",   "✗", "✗", "✗", "—"),
    ("MADE-for-ASD 2024", "✗", "✗", "—", "✓"),
    ("Ours",              "✓", "✓", "✓", "✓"),
]
CAP_COLS = ["Calibration\n(ECE)", "Selective\nprediction", "Cross-site\n(LOSO)",
            "Interpretable\nbiomarker"]


def plot_capability_matrix(out: str | Path) -> Path:
    """Reliability/rigor breadth per method — where our method legitimately leads."""
    with ieee_style():
        nrow, ncol = len(CAPABILITIES), len(CAP_COLS)
        fig, ax = plt.subplots(figsize=(1.5 * ncol + 2.4, 0.5 * nrow + 1.4))
        ax.set_xlim(0, ncol + 1.6); ax.set_ylim(0, nrow + 1)
        ax.axis("off")
        # header
        for j, c in enumerate(CAP_COLS):
            ax.text(1.6 + j + 0.5, nrow + 0.4, c, ha="center", va="center",
                    fontsize=8, fontweight="bold")
        for i, (name, *cells) in enumerate(CAPABILITIES):
            yy = nrow - 1 - i + 0.5
            ours = name == "Ours"
            if ours:
                ax.add_patch(plt.Rectangle((0.05, yy - 0.45), ncol + 1.55, 0.9,
                                           color=COLORS["asd"], alpha=0.12, zorder=0))
            ax.text(0.15, yy, name, ha="left", va="center", fontsize=8.5,
                    fontweight="bold" if ours else "normal")
            for j, v in enumerate(cells):
                col = {"✓": "#1a8a3a", "✗": "#b23b3b", "—": "#888888"}[v]
                ax.text(1.6 + j + 0.5, yy, v, ha="center", va="center",
                        fontsize=12, color=col, fontweight="bold")
        ax.set_title("Reliability & rigor reported per method", fontsize=10, pad=6)
        fig.text(0.5, 0.005,
                 "✓ reported · ✗ not reported · — unclear.  Competitor cells reflect the "
                 "primary papers to our reading — [VERIFY] before submission.",
                 ha="center", va="bottom", fontsize=6.5)
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        p = out / "capability_matrix"
        for ext in ("pdf", "png"):
            fig.savefig(f"{p}.{ext}", bbox_inches="tight")
        plt.close(fig)
    return Path(f"{p}.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Honest SOTA comparison + capability matrix")
    ap.add_argument("--acc_full", type=float, required=True)
    ap.add_argument("--acc_selective", type=float, default=None)
    ap.add_argument("--coverage", type=float, default=0.5)
    ap.add_argument("--auroc", type=float, default=None)
    ap.add_argument("--out", default="results/latest/model/figures")
    args = ap.parse_args()
    print("wrote", plot_sota_comparison(args.acc_full, args.out,
                                        our_acc_selective=args.acc_selective,
                                        coverage=args.coverage, our_auroc=args.auroc))
    print("wrote", plot_capability_matrix(args.out))
