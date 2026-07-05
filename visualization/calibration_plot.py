"""
Calibration (reliability diagram) visualization.

A well-calibrated classifier produces predictions P(Y=1|X=x) that equal
the observed fraction of positives in the neighbourhood of x.  The
reliability diagram plots observed fraction vs mean predicted probability
per bin.  Points on the diagonal y=x indicate perfect calibration.

Also plots the Brier score Murphy decomposition as a stacked bar chart.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes

from visualization.style import ieee_style, _ensure_fig_ax, COLORS

logger = logging.getLogger(__name__)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    show_histogram: bool = True,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    Reliability (calibration) diagram.

    Parameters
    ----------
    y_true       : (N,) binary labels
    y_prob       : (N,) predicted probabilities
    n_bins       : number of equally-spaced confidence bins
    strategy     : "uniform" (equal-width) or "quantile" (equal-frequency)
    show_histogram : add a small histogram of prediction counts per bin

    Returns
    -------
    matplotlib.figure.Figure
    """
    from evaluation.calibration import reliability_diagram_data

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    rel = reliability_diagram_data(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    bin_mids  = rel["bin_midpoints"]
    bin_accs  = rel["bin_accuracies"]
    bin_confs = rel["bin_confidences"]
    bin_cnts  = rel["bin_counts"]
    ece       = rel["ece"]
    mce       = rel["mce"]

    with ieee_style():
        if show_histogram:
            fig = plt.figure(figsize=(3.487, 3.2))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax_main = fig.add_subplot(gs[0])
            ax_hist = fig.add_subplot(gs[1], sharex=ax_main)
        else:
            fig, ax_main = _ensure_fig_ax(ax)
            ax_hist = None

        # Perfect calibration diagonal
        ax_main.plot([0, 1], [0, 1], "--", color=COLORS["perfect"],
                     linewidth=1.0, label="Perfect calibration")

        # Observed vs confidence per bin
        w = (bin_mids[1] - bin_mids[0]) * 0.8 if len(bin_mids) > 1 else 0.08
        ax_main.bar(bin_mids, bin_accs, width=w, alpha=0.7,
                    color=COLORS["model_a"], label="Fraction positives")
        ax_main.plot(bin_confs[bin_cnts > 0], bin_accs[bin_cnts > 0],
                     "s-", color=COLORS["asd"], markersize=4, linewidth=1.0,
                     label="Mean confidence")

        ax_main.set_xlim([0, 1])
        ax_main.set_ylim([0, 1.05])
        ax_main.set_ylabel("Fraction of positives")
        ax_main.legend(loc="upper left", fontsize=7)
        ax_main.set_title(
            f"Reliability Diagram  (ECE={ece:.4f}, MCE={mce:.4f})"
        )

        if show_histogram:
            ax_hist.bar(bin_mids, bin_cnts, width=w, alpha=0.7,
                        color=COLORS["model_a"])
            ax_hist.set_xlim([0, 1])
            ax_hist.set_xlabel("Mean predicted probability")
            ax_hist.set_ylabel("Count", fontsize=7)
            plt.setp(ax_main.get_xticklabels(), visible=False)
        else:
            ax_main.set_xlabel("Mean predicted probability")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="tight_layout", category=UserWarning)
            fig.tight_layout()
    return fig


def plot_brier_decomposition(
    brier_dict: dict,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    Stacked bar chart of the Murphy Brier score decomposition.

    Brier = Calibration − Resolution + Uncertainty

    Parameters
    ----------
    brier_dict : dict from ``evaluation.calibration.brier_score_decomposition``

    Returns
    -------
    matplotlib.figure.Figure
    """
    brier = brier_dict["brier_score"]
    calib = brier_dict["calibration"]
    resol = brier_dict["resolution"]
    uncer = brier_dict["uncertainty"]

    components = {
        "Uncertainty": uncer,
        "Calibration": calib,
        "−Resolution": -resol,
    }
    colors = [COLORS["random"], COLORS["asd"], COLORS["model_c"]]
    labels = list(components.keys())
    values = list(components.values())

    with ieee_style():
        fig, ax = _ensure_fig_ax(ax, figsize=(3.0, 2.4))

        bottoms = 0.0
        for i, (lbl, val, clr) in enumerate(zip(labels, values, colors)):
            ax.bar(["Brier"], [abs(val)], bottom=bottoms, color=clr,
                   label=lbl, alpha=0.85)
            mid = bottoms + abs(val) / 2
            ax.text(0, mid, f"{val:+.4f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > 0.02 else "black")
            bottoms += abs(val)

        ax.axhline(brier, linestyle="--", color="black", linewidth=1.0,
                   label=f"Brier = {brier:.4f}")
        ax.set_ylabel("Score")
        ax.set_title("Brier Score Decomposition")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()

    return fig
