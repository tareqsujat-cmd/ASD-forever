"""
Lift and Gain (cumulative response) curves for ASD classification.

Gain curve
----------
Sort subjects by predicted probability (descending).  At each fraction p of
the population examined, the gain is the fraction of all ASD cases captured:

    Gain(p) = (# positives in top-p fraction) / (total positives)

A perfect model captures all positives in the first `prevalence` fraction.
The random baseline is Gain(p) = p (the diagonal).

Lift curve
----------
Lift(p) = Gain(p) / p — ratio of the model's capture rate to random.
Lift > 1 means the model outperforms random at that fraction.

Functions
---------
plot_gain_curve        — cumulative gain chart
plot_lift_curve        — lift chart
plot_lift_gain_panel   — side-by-side (gain | lift) two-panel figure
compute_gain           — standalone computation helper
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes

from visualization.style import (
    ieee_style, _ensure_fig_ax,
    COLORS, PALETTE,
    SINGLE_COL_W, DOUBLE_COL_W, FIG_HEIGHT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_gain(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute cumulative gain at evenly-spaced population fractions.

    Returns
    -------
    fractions : 1-D array in [0, 1]
    gains     : corresponding gain values in [0, 1]
    """
    order    = np.argsort(y_prob)[::-1]   # descending by score
    y_sorted = y_true[order]
    n_total  = len(y_true)
    n_pos    = y_true.sum()

    if n_pos == 0:
        return np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid)

    # Compute cumulative positives at each rank
    cum_pos = np.cumsum(y_sorted)
    gains_full = cum_pos / n_pos               # shape (n_total,)
    fracs_full = np.arange(1, n_total + 1) / n_total

    # Prepend (0, 0) and subsample to n_grid points
    fracs_full = np.concatenate([[0.0], fracs_full])
    gains_full = np.concatenate([[0.0], gains_full])

    idx       = np.linspace(0, len(fracs_full) - 1, n_grid, dtype=int)
    fractions = fracs_full[idx]
    gains     = gains_full[idx]
    return fractions, gains


# ---------------------------------------------------------------------------
# Gain plot
# ---------------------------------------------------------------------------

def plot_gain_curve(
    y_true:  np.ndarray,
    models:  List[Dict],
    ax:      Optional[matplotlib.axes.Axes] = None,
    title:   str = "Cumulative Gain",
    n_grid:  int = 200,
) -> plt.Figure:
    """
    Plot cumulative gain curves for one or more models.

    Parameters
    ----------
    y_true : binary labels
    models : list of dicts: {"name", "y_prob", "color" (opt), "ls" (opt)}
    """
    with ieee_style():
        fig, ax = _ensure_fig_ax(ax, figsize=(SINGLE_COL_W, FIG_HEIGHT))

        # Random baseline
        ax.plot([0, 1], [0, 1], color=COLORS["random"], lw=1.0, ls="--",
                label="Random", zorder=1)

        # Perfect model
        prevalence = float(y_true.mean())
        ax.plot([0, prevalence, 1.0], [0, 1.0, 1.0],
                color=COLORS["perfect"], lw=1.0, ls=":",
                label="Perfect", zorder=2)

        for idx, m in enumerate(models):
            color  = m.get("color", PALETTE[idx % len(PALETTE)])
            ls     = m.get("ls", "-")
            name   = m.get("name", f"Model {idx + 1}")
            y_prob = np.asarray(m["y_prob"])

            fracs, gains = compute_gain(y_true, y_prob, n_grid=n_grid)
            ax.plot(fracs, gains, color=color, ls=ls, lw=1.6,
                    label=name, zorder=3)

        ax.set_xlabel("Fraction of population")
        ax.set_ylabel("Fraction of positives captured")
        ax.set_title(title)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Lift plot
# ---------------------------------------------------------------------------

def plot_lift_curve(
    y_true:  np.ndarray,
    models:  List[Dict],
    ax:      Optional[matplotlib.axes.Axes] = None,
    title:   str = "Lift Curve",
    n_grid:  int = 200,
) -> plt.Figure:
    """
    Plot lift curves (Gain(p)/p) for one or more models.

    Lift = 1 corresponds to random guessing (shown as a dashed baseline).
    """
    with ieee_style():
        fig, ax = _ensure_fig_ax(ax, figsize=(SINGLE_COL_W, FIG_HEIGHT))

        # Random baseline
        ax.axhline(1.0, color=COLORS["random"], lw=1.0, ls="--",
                   label="Random", zorder=1)

        for idx, m in enumerate(models):
            color  = m.get("color", PALETTE[idx % len(PALETTE)])
            ls     = m.get("ls", "-")
            name   = m.get("name", f"Model {idx + 1}")
            y_prob = np.asarray(m["y_prob"])

            fracs, gains = compute_gain(y_true, y_prob, n_grid=n_grid)
            # Avoid division by zero at fraction=0
            with np.errstate(invalid="ignore", divide="ignore"):
                lift = np.where(fracs > 0, gains / fracs, np.nan)

            ax.plot(fracs, lift, color=color, ls=ls, lw=1.6,
                    label=name, zorder=2)

        ax.set_xlabel("Fraction of population")
        ax.set_ylabel("Lift")
        ax.set_title(title)
        ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Combined two-panel figure
# ---------------------------------------------------------------------------

def plot_lift_gain_panel(
    y_true:  np.ndarray,
    models:  List[Dict],
    title:   str = "Lift and Gain",
    n_grid:  int = 200,
) -> plt.Figure:
    """
    Two-panel figure: left = gain curve, right = lift curve.

    Suitable for a double-column IEEE figure (7.16" wide).
    """
    with ieee_style():
        fig, (ax_gain, ax_lift) = plt.subplots(
            1, 2, figsize=(DOUBLE_COL_W, FIG_HEIGHT)
        )

        # --- Gain panel ---
        ax_gain.plot([0, 1], [0, 1], color=COLORS["random"], lw=1.0, ls="--",
                     label="Random", zorder=1)
        prevalence = float(y_true.mean())
        ax_gain.plot([0, prevalence, 1.0], [0, 1.0, 1.0],
                     color=COLORS["perfect"], lw=1.0, ls=":",
                     label="Perfect", zorder=2)

        # --- Lift panel ---
        ax_lift.axhline(1.0, color=COLORS["random"], lw=1.0, ls="--",
                        label="Random", zorder=1)

        for idx, m in enumerate(models):
            color  = m.get("color", PALETTE[idx % len(PALETTE)])
            ls     = m.get("ls", "-")
            name   = m.get("name", f"Model {idx + 1}")
            y_prob = np.asarray(m["y_prob"])

            fracs, gains = compute_gain(y_true, y_prob, n_grid=n_grid)

            ax_gain.plot(fracs, gains, color=color, ls=ls, lw=1.6,
                         label=name, zorder=3)

            with np.errstate(invalid="ignore", divide="ignore"):
                lift = np.where(fracs > 0, gains / fracs, np.nan)
            ax_lift.plot(fracs, lift, color=color, ls=ls, lw=1.6,
                         label=name, zorder=2)

        ax_gain.set_xlabel("Fraction of population")
        ax_gain.set_ylabel("Fraction of positives captured")
        ax_gain.set_title("Cumulative Gain")
        ax_gain.set_xlim(0, 1); ax_gain.set_ylim(0, 1.05)
        ax_gain.legend(loc="lower right", fontsize=7)

        ax_lift.set_xlabel("Fraction of population")
        ax_lift.set_ylabel("Lift")
        ax_lift.set_title("Lift Curve")
        ax_lift.set_xlim(0, 1); ax_lift.set_ylim(bottom=0)
        ax_lift.legend(loc="upper right", fontsize=7)

        fig.suptitle(title, fontsize=9)
        fig.tight_layout()

    return fig
