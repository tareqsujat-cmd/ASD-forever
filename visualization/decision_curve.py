"""
Decision Curve Analysis (DCA) for ASD classification.

DCA evaluates clinical utility of a model across a range of decision
thresholds (threshold probabilities), computing net benefit:

    NB(t) = TP/N  −  FP/N × t / (1 − t)

where t is the probability threshold at which a clinician would act.
A model is clinically useful at threshold t if its NB exceeds both
"treat all" and "treat none" baselines.

References
----------
Vickers et al. (2006) "Decision curve analysis: a novel method for
evaluating prediction models". Medical Decision Making 26(6):565–574.

Functions
---------
plot_decision_curve        — single or multi-model DCA panel
plot_dca_comparison        — alias for multi-model DCA
compute_net_benefit        — standalone computation helper
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes

from visualization.style import (
    ieee_style, _ensure_fig_ax,
    COLORS, PALETTE,
    SINGLE_COL_W, FIG_HEIGHT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_net_benefit(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Compute net benefit at each decision threshold.

    Parameters
    ----------
    y_true     : binary labels (0/1)
    y_prob     : predicted probability of positive class
    thresholds : array of probability thresholds in [0, 1)

    Returns
    -------
    net_benefit : array of shape (len(thresholds),)
    """
    n = len(y_true)
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        if t >= 1.0:
            nb[i] = np.nan
            continue
        y_hat = (y_prob >= t).astype(int)
        tp = int(np.sum((y_hat == 1) & (y_true == 1)))
        fp = int(np.sum((y_hat == 1) & (y_true == 0)))
        # Harm weight: acting on a false positive costs t/(1-t) "true-positive
        # equivalents" — this is the exchange rate implicit in the threshold
        nb[i] = tp / n - fp / n * (t / (1.0 - t))
    return nb


def compute_treat_all(
    y_true:    np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Net benefit of classifying everyone as positive (treat-all baseline)."""
    prevalence = y_true.mean()
    nb = np.empty(len(thresholds))
    for i, t in enumerate(thresholds):
        if t >= 1.0:
            nb[i] = np.nan
            continue
        nb[i] = prevalence - (1.0 - prevalence) * (t / (1.0 - t))
    return nb


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_decision_curve(
    y_true:         np.ndarray,
    models:         List[Dict],
    ax:             Optional[matplotlib.axes.Axes] = None,
    t_min:          float = 0.01,
    t_max:          float = 0.60,
    n_thresholds:   int   = 200,
    show_treat_all: bool  = True,
    show_treat_none: bool = True,
    clip_negative:  bool  = True,
    title:          str   = "Decision Curve Analysis",
) -> plt.Figure:
    """
    Plot DCA for one or more models on a single axes.

    Parameters
    ----------
    y_true : binary ground-truth labels
    models : list of dicts, each with:
               "name"  : str
               "y_prob": np.ndarray  (predicted probabilities)
               "color" : str  (optional; cycles through PALETTE if omitted)
               "ls"    : str  (optional line style, default "-")
    ax     : existing axes (creates a new figure if None)
    t_min / t_max    : threshold range on x-axis
    n_thresholds     : resolution of the threshold grid
    show_treat_all   : draw the "treat all" reference line
    show_treat_none  : draw the "treat none" (y=0) reference line
    clip_negative    : clamp net benefit to 0 (negative NB = worse than treat-none)
    title            : axes title

    Returns
    -------
    matplotlib.figure.Figure
    """
    thresholds = np.linspace(t_min, t_max, n_thresholds)

    with ieee_style():
        fig, ax = _ensure_fig_ax(ax, figsize=(SINGLE_COL_W, FIG_HEIGHT + 0.3))

        # Baselines
        if show_treat_none:
            ax.axhline(0, color=COLORS["random"], lw=1.0, ls="--",
                       label="Treat none", zorder=1)

        if show_treat_all:
            nb_all = compute_treat_all(y_true, thresholds)
            if clip_negative:
                nb_all = np.clip(nb_all, 0, None)
            ax.plot(thresholds, nb_all,
                    color="dimgray", lw=1.0, ls=":",
                    label="Treat all", zorder=2)

        # Per-model curves
        for idx, m in enumerate(models):
            color = m.get("color", PALETTE[idx % len(PALETTE)])
            ls    = m.get("ls", "-")
            name  = m.get("name", f"Model {idx + 1}")
            y_prob = np.asarray(m["y_prob"])

            nb = compute_net_benefit(y_true, y_prob, thresholds)
            if clip_negative:
                nb = np.clip(nb, 0, None)

            ax.plot(thresholds, nb,
                    color=color, ls=ls, lw=1.6,
                    label=name, zorder=3)

        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(title)
        ax.set_xlim(t_min, t_max)
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()

    return fig


def plot_dca_comparison(
    y_true:  np.ndarray,
    models:  List[Dict],
    **kwargs,
) -> plt.Figure:
    """Convenience alias — same signature as plot_decision_curve."""
    return plot_decision_curve(y_true, models, **kwargs)
