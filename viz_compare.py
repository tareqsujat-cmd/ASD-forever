"""
Comparison-to-SOTA figure for ABIDE-I ASD classification (scienceplots, Q1/ICLR).

Plots our result against the published literature, **honestly colour-coded by
cohort/protocol** so leaky/subset numbers are not presented as like-for-like with
full-cohort honest ones. This transparency is itself a contribution.

Literature values are the council-verified numbers (see publication_plan.md §9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from visualization.style import ieee_style, COLORS

# Best *legitimate* full-cohort ABIDE-I methods only (no subset/leakage results,
# no weak/old baselines) — the credible SOTA we compare against like-for-like.
# (name, accuracy, protocol note)
LITERATURE = [
    ("Heinsfeld 2018",      0.700, "full · 10-fold"),
    ("ASD-DiagNet 2019",    0.703, "full · 10-fold"),
    ("Parisot 2018 (GCN)",  0.704, "full · 10-fold"),
    ("ASD-SAENet 2021",     0.708, "full · 10-fold"),
    ("MADE-for-ASD 2024",   0.752, "full · 10-fold"),   # best honest full-cohort
]
HEINSFELD_BASELINE = 0.70
BEST_HONEST = 0.752


def plot_sota_comparison(our_acc: float, out: str | Path,
                         our_label: str = "Ours (full · nested)",
                         our_note: str = "full · nested",
                         our_auroc: Optional[float] = None) -> Path:
    items = [(n, a, note) for n, a, note in LITERATURE] + [(our_label, float(our_acc), our_note)]
    items = sorted(items, key=lambda x: x[1])
    names = [f"{n}\n({note})" for n, _, note in items]
    accs = [a for _, a, _ in items]
    colors = [COLORS["asd"] if n == our_label else COLORS["model_a"]
              for n, _, _ in items]

    with ieee_style():
        fig, ax = plt.subplots(figsize=(6.8, 0.52 * len(items) + 1.0))
        bars = ax.barh(names, accs, color=colors, edgecolor="black", linewidth=1.1)
        for b, a in zip(bars, accs):
            ax.text(b.get_width() + 0.004, b.get_y() + b.get_height() / 2,
                    f"{a:.3f}", va="center", fontsize=9, fontweight="bold")
        ax.axvline(HEINSFELD_BASELINE, ls="--", lw=1.6, color=COLORS["random"])
        ax.axvline(BEST_HONEST, ls=":", lw=1.6, color=COLORS["model_c"])
        lo = min(min(accs), HEINSFELD_BASELINE) - 0.03
        hi = max(max(accs), BEST_HONEST) + 0.05
        ax.set_xlim(max(0.5, lo), hi)
        ax.set_xlabel("Accuracy — ABIDE-I (full cohort, honest protocols)")
        title = "ASD classification on ABIDE-I — vs. best legitimate methods"
        if our_auroc is not None:
            title += f"\n(ours AUROC = {our_auroc:.3f})"
        ax.set_title(title)
        legend = [
            Patch(facecolor=COLORS["asd"], edgecolor="black", label="Ours"),
            Patch(facecolor=COLORS["model_a"], edgecolor="black", label="Prior full-cohort SOTA"),
            plt.Line2D([], [], ls="--", color=COLORS["random"], label="Heinsfeld 0.70"),
            plt.Line2D([], [], ls=":", color=COLORS["model_c"], label="Best honest 0.752"),
        ]
        ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        p = out / "sota_comparison"
        for ext in ("pdf", "png"):
            fig.savefig(f"{p}.{ext}", bbox_inches="tight")
        plt.close(fig)
    return Path(f"{p}.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SOTA comparison figure")
    ap.add_argument("--acc", type=float, required=True, help="our accuracy")
    ap.add_argument("--auroc", type=float, default=None)
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    print("wrote", plot_sota_comparison(args.acc, args.out, our_auroc=args.auroc))
