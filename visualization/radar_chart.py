"""
Metric Radar (Spider) Chart for ASD model comparison.

Plots up to 4 models as overlapping filled polygons on a polar axes,
with one spoke per metric.  Metrics are independently scaled to [0, 1]
for visual comparison (MCC and Cohen's Kappa are linearly mapped from
[-1, 1] to [0, 1]).

Default metric set (IEEE-standard for clinical AI):
  Sensitivity (TPR), Specificity (TNR), Precision (PPV), NPV,
  F1, AUROC, MCC, Cohen's Kappa

Functions
---------
plot_radar_chart        — single figure with all models
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes

from visualization.style import (
    ieee_style,
    PALETTE,
    SINGLE_COL_W, FIG_HEIGHT,
)

logger = logging.getLogger(__name__)

# Default metrics and their raw value ranges (for normalisation to [0, 1])
_DEFAULT_METRICS: List[Dict] = [
    {"key": "sensitivity", "label": "Sensitivity", "lo": 0.0, "hi": 1.0},
    {"key": "specificity", "label": "Specificity", "lo": 0.0, "hi": 1.0},
    {"key": "ppv",         "label": "Precision",   "lo": 0.0, "hi": 1.0},
    {"key": "npv",         "label": "NPV",          "lo": 0.0, "hi": 1.0},
    {"key": "f1",          "label": "F1",           "lo": 0.0, "hi": 1.0},
    {"key": "auc",         "label": "AUC",          "lo": 0.0, "hi": 1.0},
    {"key": "mcc",         "label": "MCC",          "lo":-1.0, "hi": 1.0},
    {"key": "kappa",       "label": "Kappa",        "lo":-1.0, "hi": 1.0},
]


def _normalise(value: float, lo: float, hi: float) -> float:
    """Map value from [lo, hi] to [0, 1], clamped."""
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def plot_radar_chart(
    models:  List[Dict],
    metrics: Optional[List[Dict]] = None,
    title:   str = "Metric Radar Chart",
    alpha:   float = 0.18,
) -> plt.Figure:
    """
    Plot a multi-model radar chart.

    Parameters
    ----------
    models : list of dicts, each with:
               "name"    : str  — legend label
               "metrics" : dict — {metric_key: float_value, ...}
               "color"   : str  — optional; cycles PALETTE
               "ls"      : str  — optional line style

    metrics : list of metric descriptor dicts:
               {"key": ..., "label": ..., "lo": ..., "hi": ...}
               Defaults to _DEFAULT_METRICS (8 clinical metrics).

    alpha : fill transparency for polygon shading

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS

    n_metrics = len(metrics)
    if n_metrics < 3:
        raise ValueError("Need at least 3 metrics for a meaningful radar chart.")

    # Spoke angles: evenly spaced, first spoke at the top (π/2)
    angles = [
        math.pi / 2 - 2 * math.pi * i / n_metrics
        for i in range(n_metrics)
    ]
    angles_closed = angles + [angles[0]]  # close the polygon

    with ieee_style():
        # Larger figure for radar chart — single column but taller
        fig_size = (SINGLE_COL_W + 0.5, FIG_HEIGHT + 1.0)
        fig = plt.figure(figsize=fig_size)
        ax  = fig.add_subplot(111, polar=True)

        # Grid rings at 0.2, 0.4, 0.6, 0.8, 1.0
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6)
        ax.yaxis.set_tick_params(labelsize=6)

        # Spoke positions and labels
        ax.set_xticks(angles)
        ax.set_xticklabels([m["label"] for m in metrics], fontsize=7)

        # Draw each model
        for idx, m in enumerate(models):
            color  = m.get("color", PALETTE[idx % len(PALETTE)])
            ls     = m.get("ls", "-")
            name   = m.get("name", f"Model {idx + 1}")
            mvals  = m.get("metrics", {})

            # Normalised values for each spoke
            vals = [
                _normalise(
                    float(mvals.get(spec["key"], 0.0)),
                    spec["lo"], spec["hi"],
                )
                for spec in metrics
            ]
            vals_closed = vals + [vals[0]]

            ax.plot(angles_closed, vals_closed,
                    color=color, ls=ls, lw=1.6, label=name, zorder=3)
            ax.fill(angles_closed, vals_closed,
                    color=color, alpha=alpha, zorder=2)

        ax.set_title(title, pad=14, fontsize=9)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=min(len(models), 3),
            fontsize=7,
            framealpha=0.85,
        )
        fig.tight_layout()

    return fig
