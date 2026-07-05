"""
Confusion matrix visualization.

Produces a 2×2 heatmap with raw counts (and optional row-normalized
percentages) and derived statistics (sensitivity, specificity, PPV, NPV,
accuracy) in a caption-style annotation block below the matrix.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes

from visualization.style import ieee_style, _ensure_fig_ax

logger = logging.getLogger(__name__)

# Default class labels for ASD detection
_DEFAULT_LABELS = ["TC", "ASD"]


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    normalize: bool = False,
    show_metrics: bool = True,
    cmap: str = "Blues",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    Plot a 2×2 confusion matrix as a colour-coded heatmap.

    Parameters
    ----------
    cm : (2, 2) int array — [[TN, FP], [FN, TP]] (row=actual, col=predicted)
    labels : (2,) class names; defaults to ["TC", "ASD"]
    normalize : if True, normalize each row to fractions summing to 1
    show_metrics : annotate the axes with Sn, Sp, PPV, NPV, Acc
    cmap : matplotlib colormap name

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = np.asarray(cm, dtype=float)
    if cm.shape != (2, 2):
        raise ValueError(f"cm must be (2, 2), got {cm.shape}")
    if labels is None:
        labels = _DEFAULT_LABELS

    cm_display = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    fmt = ".2f" if normalize else ".0f"

    with ieee_style():
        fig, ax = _ensure_fig_ax(ax, figsize=(3.0, 3.0))
        im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap,
                       vmin=0, vmax=cm_display.max())

        # Colour-bar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

        # Tick labels
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")

        # Cell annotations
        thresh = cm_display.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm_display[i, j]
                raw = int(cm[i, j])
                text = f"{val:{fmt}}"
                if not normalize:
                    text = f"{raw}"
                else:
                    text = f"{val:.2f}\n(n={raw})"
                ax.text(j, i, text,
                        ha="center", va="center", fontsize=9,
                        color="white" if val > thresh else "black")

        if show_metrics:
            TN, FP = cm[0, 0], cm[0, 1]
            FN, TP = cm[1, 0], cm[1, 1]
            sn  = TP / (TP + FN + 1e-8)
            sp  = TN / (TN + FP + 1e-8)
            ppv = TP / (TP + FP + 1e-8)
            npv = TN / (TN + FN + 1e-8)
            acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
            stats = (f"Sn={sn:.3f}  Sp={sp:.3f}  PPV={ppv:.3f}"
                     f"  NPV={npv:.3f}  Acc={acc:.3f}")
            ax.set_xlabel(f"Predicted label\n{stats}", fontsize=7)

        fig.tight_layout()
    return fig


def cm_from_report(report) -> np.ndarray:
    """
    Build a (2, 2) confusion matrix from an ``EvaluationReport``.

    Returns [[TN, FP], [FN, TP]].
    """
    return np.array([
        [report.tn, report.fp],
        [report.fn, report.tp],
    ], dtype=float)
