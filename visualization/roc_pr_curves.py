"""
ROC and Precision-Recall curve plotting with optional bootstrap CI bands.

IEEE-ready outputs: single-column (3.5") figures, serif fonts, 300 DPI.

Functions
---------
plot_roc_curve        — single model ROC with optional CI shading
plot_pr_curve         — single model PR with optional CI shading
plot_roc_comparison   — overlay multiple models on one ROC plot
plot_pr_comparison    — overlay multiple models on one PR plot
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from visualization.style import ieee_style, _ensure_fig_ax, COLORS, PALETTE, SINGLE_COL_W

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap CI helpers
# ---------------------------------------------------------------------------

def _bootstrap_roc_band(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    n_grid: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified bootstrap CI band for the ROC curve.

    Returns (fpr_grid, tpr_lo, tpr_hi).
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    fpr_grid = np.linspace(0, 1, n_grid)
    tpr_band = np.zeros((n_bootstrap, n_grid))

    for i in range(n_bootstrap):
        b = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
        ])
        fpr_b, tpr_b, _ = roc_curve(y_true[b], y_prob[b])
        tpr_band[i] = np.interp(fpr_grid, fpr_b, tpr_b)

    lo = float(alpha / 2 * 100)
    hi = float((1 - alpha / 2) * 100)
    return fpr_grid, np.percentile(tpr_band, lo, axis=0), np.percentile(tpr_band, hi, axis=0)


def _bootstrap_pr_band(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    n_grid: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified bootstrap CI band for the PR curve.

    Returns (recall_grid, prec_lo, prec_hi).
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    recall_grid = np.linspace(0, 1, n_grid)
    prec_band = np.zeros((n_bootstrap, n_grid))

    for i in range(n_bootstrap):
        b = np.concatenate([
            rng.choice(pos_idx, len(pos_idx), replace=True),
            rng.choice(neg_idx, len(neg_idx), replace=True),
        ])
        prec_b, rec_b, _ = precision_recall_curve(y_true[b], y_prob[b])
        # precision_recall_curve goes from high recall to low; reverse for interp
        prec_b, rec_b = prec_b[::-1], rec_b[::-1]
        prec_band[i] = np.interp(recall_grid, rec_b, prec_b, left=1.0)

    lo = float(alpha / 2 * 100)
    hi = float((1 - alpha / 2) * 100)
    return recall_grid, np.percentile(prec_band, lo, axis=0), np.percentile(prec_band, hi, axis=0)


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "Model",
    color: Optional[str] = None,
    show_ci: bool = False,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    ax: Optional[matplotlib.axes.Axes] = None,
    seed: int = 42,
) -> plt.Figure:
    """
    Plot an ROC curve with optional bootstrap CI shading.

    Parameters
    ----------
    y_true      : (N,) binary labels
    y_prob      : (N,) predicted probabilities
    label       : legend label (AUC is appended automatically)
    color       : line color; defaults to ``COLORS["model_a"]``
    show_ci     : draw 95% CI band via stratified bootstrap
    n_bootstrap : bootstrap resamples (used if show_ci=True)
    ax          : existing Axes; a new figure is created if None

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if color is None:
        color = COLORS["model_a"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    with ieee_style():
        fig, ax = _ensure_fig_ax(ax)

        if show_ci:
            fpr_g, tpr_lo, tpr_hi = _bootstrap_roc_band(
                y_true, y_prob, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
            )
            ax.fill_between(fpr_g, tpr_lo, tpr_hi,
                            alpha=0.25, color=color, linewidth=0)

        ax.plot(fpr, tpr, color=color, linewidth=1.5,
                label=f"{label}  (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color=COLORS["random"],
                linewidth=0.9, label="Random (AUC=0.500)")

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# PR curve
# ---------------------------------------------------------------------------

def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "Model",
    color: Optional[str] = None,
    show_ci: bool = False,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    ax: Optional[matplotlib.axes.Axes] = None,
    seed: int = 42,
) -> plt.Figure:
    """
    Plot a Precision-Recall curve with optional bootstrap CI shading.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if color is None:
        color = COLORS["model_a"]

    prevalence = float(y_true.mean())
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    with ieee_style():
        fig, ax = _ensure_fig_ax(ax)

        if show_ci:
            rec_g, prec_lo, prec_hi = _bootstrap_pr_band(
                y_true, y_prob, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
            )
            ax.fill_between(rec_g, prec_lo, prec_hi,
                            alpha=0.25, color=color, linewidth=0)

        ax.plot(rec, prec, color=color, linewidth=1.5,
                label=f"{label}  (AP={ap:.3f})")
        ax.axhline(prevalence, linestyle="--", color=COLORS["random"],
                   linewidth=0.9, label=f"Baseline  (prev.={prevalence:.3f})")

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="upper right")
        fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def plot_roc_comparison(
    models: List[Dict],
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    Overlay ROC curves for multiple models.

    Parameters
    ----------
    models : list of dicts, each with keys:
        "name"   : str — legend label
        "y_true" : (N,) array
        "y_prob" : (N,) array
        "color"  : str (optional)

    Returns
    -------
    matplotlib.figure.Figure
    """
    with ieee_style():
        fig, ax = _ensure_fig_ax(ax)

        for i, m in enumerate(models):
            fpr, tpr, _ = roc_curve(m["y_true"], m["y_prob"])
            roc_auc = auc(fpr, tpr)
            color = m.get("color", PALETTE[i % len(PALETTE)])
            ax.plot(fpr, tpr, color=color, linewidth=1.5,
                    label=f"{m['name']}  (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "--", color=COLORS["random"],
                linewidth=0.9, label="Random")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Comparison")
        ax.legend(loc="lower right")
        fig.tight_layout()

    return fig


def plot_pr_comparison(
    models: List[Dict],
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    Overlay PR curves for multiple models.

    Parameters same as ``plot_roc_comparison``.
    """
    with ieee_style():
        fig, ax = _ensure_fig_ax(ax)

        for i, m in enumerate(models):
            prec, rec, _ = precision_recall_curve(m["y_true"], m["y_prob"])
            ap = average_precision_score(m["y_true"], m["y_prob"])
            color = m.get("color", PALETTE[i % len(PALETTE)])
            ax.plot(rec, prec, color=color, linewidth=1.5,
                    label=f"{m['name']}  (AP={ap:.3f})")

        if models:
            prev = float(np.asarray(models[0]["y_true"]).mean())
            ax.axhline(prev, linestyle="--", color=COLORS["random"],
                       linewidth=0.9, label=f"Baseline (prev.={prev:.3f})")

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PR Comparison")
        ax.legend(loc="upper right")
        fig.tight_layout()

    return fig
