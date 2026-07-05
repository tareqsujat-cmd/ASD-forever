"""
Prediction confidence histograms and error distribution plots.

Two visualisations produced by this module:

1. Confidence histogram (two-panel)
   Left:  Overlapping histograms of P(ASD) for correct vs incorrect
          predictions, separated by true label.  Vertical line at
          decision threshold.  Exposes calibration quality and
          where the model is uncertain.
   Right: Stacked bar chart of error types (TP / TN / FP / FN) showing
          the confusion breakdown and the weighted clinical cost
          (FN weighted 2× by default).

2. Confidence–error scatter panel
   A single panel scatter of confidence (max probability) vs correctness,
   with marginal rug plots.  Highlights hard examples (high confidence,
   wrong label).

Functions
---------
plot_confidence_histogram    — two-panel confidence + error breakdown
plot_confidence_error_scatter— scatter of confidence vs error
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.ticker as mticker

from visualization.style import (
    ieee_style, _ensure_fig_ax,
    COLORS, PALETTE,
    SINGLE_COL_W, DOUBLE_COL_W, FIG_HEIGHT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence histogram (two-panel)
# ---------------------------------------------------------------------------

def plot_confidence_histogram(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float = 0.5,
    n_bins:    int   = 30,
    fn_weight: float = 2.0,
    title:     str   = "Prediction Confidence Distribution",
) -> plt.Figure:
    """
    Two-panel figure: confidence histogram + error breakdown bar chart.

    Parameters
    ----------
    y_true    : binary ground-truth labels (0 = TC, 1 = ASD)
    y_prob    : predicted probability of ASD (positive class)
    threshold : decision threshold (vertical line in left panel)
    n_bins    : number of histogram bins
    fn_weight : clinical weight of FN relative to FP (for cost annotation)
    title     : figure suptitle

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_pred = (y_prob >= threshold).astype(int)
    correct   = (y_pred == y_true)
    incorrect = ~correct

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    clinical_cost = fn * fn_weight + fp

    with ieee_style():
        fig, (ax_hist, ax_bar) = plt.subplots(
            1, 2, figsize=(DOUBLE_COL_W, FIG_HEIGHT)
        )

        # ----- Left: confidence histogram -----
        bins = np.linspace(0, 1, n_bins + 1)

        ax_hist.hist(
            y_prob[correct], bins=bins, density=True,
            color=COLORS["tc"], alpha=0.6, label="Correct", zorder=2
        )
        ax_hist.hist(
            y_prob[incorrect], bins=bins, density=True,
            color=COLORS["asd"], alpha=0.6, label="Incorrect", zorder=3
        )
        ax_hist.axvline(threshold, color="black", lw=1.2, ls="--",
                        label=f"Threshold={threshold:.2f}", zorder=4)

        # Annotate mean confidence for each group
        if correct.any():
            m_corr = y_prob[correct].mean()
            ax_hist.axvline(m_corr, color=COLORS["tc"], lw=1.0, ls=":",
                            alpha=0.8, zorder=3)
        if incorrect.any():
            m_inc = y_prob[incorrect].mean()
            ax_hist.axvline(m_inc, color=COLORS["asd"], lw=1.0, ls=":",
                            alpha=0.8, zorder=3)

        ax_hist.set_xlabel("P(ASD)")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Confidence by Correctness")
        ax_hist.set_xlim(0, 1)
        ax_hist.legend(fontsize=7)

        # ----- Right: error breakdown bar chart -----
        categories = ["TP", "TN", "FP", "FN"]
        counts     = [tp, tn, fp, fn]
        bar_colors = [
            COLORS["tc"],    # TP = correct ASD
            COLORS["model_a"],  # TN = correct TC
            COLORS["model_d"],  # FP = false alarm
            COLORS["asd"],   # FN = missed ASD
        ]

        bars = ax_bar.bar(categories, counts, color=bar_colors,
                          edgecolor="white", linewidth=0.5, zorder=2)

        # Annotate count above each bar
        for bar_obj, cnt in zip(bars, counts):
            ax_bar.text(
                bar_obj.get_x() + bar_obj.get_width() / 2,
                bar_obj.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=7,
            )

        ax_bar.set_xlabel("Prediction category")
        ax_bar.set_ylabel("Count")
        ax_bar.set_title("Error Breakdown")
        ax_bar.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax_bar.text(
            0.98, 0.97,
            f"Clinical cost: {clinical_cost:.1f}\n"
            f"(FN×{fn_weight:.0f} + FP×1)",
            ha="right", va="top",
            transform=ax_bar.transAxes,
            fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.85),
        )

        fig.suptitle(title, fontsize=9)
        fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Confidence–error scatter
# ---------------------------------------------------------------------------

def plot_confidence_error_scatter(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float = 0.5,
    jitter:    float = 0.02,
    seed:      int   = 42,
    title:     str   = "Confidence vs Correctness",
) -> plt.Figure:
    """
    Scatter of per-subject confidence vs ground-truth correctness.

    Confidence = max(P(ASD), 1 − P(ASD)).  Subjects plotted as dots,
    jittered vertically.  High-confidence errors are annotated with their
    index.

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng        = np.random.default_rng(seed)
    y_pred     = (y_prob >= threshold).astype(int)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    correct    = (y_pred == y_true).astype(int)   # 1 = correct, 0 = incorrect

    jitter_y   = rng.uniform(-jitter, jitter, len(y_true))
    y_scatter  = correct + jitter_y

    with ieee_style():
        fig, ax = _ensure_fig_ax(None, figsize=(SINGLE_COL_W + 0.5, FIG_HEIGHT + 0.3))

        # Incorrect predictions
        mask_inc = correct == 0
        ax.scatter(
            confidence[mask_inc], y_scatter[mask_inc],
            c=COLORS["asd"], s=14, alpha=0.7, lw=0,
            label="Incorrect", zorder=3,
        )
        # Correct predictions
        mask_cor = correct == 1
        ax.scatter(
            confidence[mask_cor], y_scatter[mask_cor],
            c=COLORS["tc"], s=14, alpha=0.5, lw=0,
            label="Correct", zorder=2,
        )

        # Highlight hard examples: high confidence AND wrong
        hard_mask = mask_inc & (confidence >= 0.75)
        if hard_mask.any():
            ax.scatter(
                confidence[hard_mask], y_scatter[hard_mask],
                s=40, facecolors="none",
                edgecolors="black", linewidths=0.8,
                label=f"Hard errors (conf≥0.75, n={hard_mask.sum()})",
                zorder=4,
            )

        ax.set_xlabel("Confidence (max probability)")
        ax.set_ylabel("Correctness")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Incorrect", "Correct"])
        ax.set_xlim(0.45, 1.01)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="center left")
        ax.axvline(0.75, color="grey", lw=0.8, ls=":", alpha=0.7)
        ax.text(0.755, 0.5, "hard\nexample\nzone",
                fontsize=6, color="grey", va="center",
                transform=ax.get_xaxis_transform())
        fig.tight_layout()

    return fig
