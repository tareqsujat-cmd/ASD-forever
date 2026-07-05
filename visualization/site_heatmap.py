"""
Per-site performance heatmap for multi-site ASD studies.

Rows = acquisition sites, columns = metrics.
Cell colour encodes the metric value (sequential colormap);
each cell is annotated with the numeric value and the subject count.

A "grand total" row showing the pooled metric across all sites is
appended at the bottom of the heatmap.

Functions
---------
compute_per_site_metrics  — compute metrics for each site
plot_site_heatmap         — render the heatmap figure
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.colors import Normalize

from visualization.style import (
    ieee_style,
    SINGLE_COL_W, DOUBLE_COL_W, FIG_HEIGHT,
)

logger = logging.getLogger(__name__)

_DEFAULT_METRICS = [
    ("auc",         "AUC"),
    ("sensitivity", "Sens."),
    ("specificity", "Spec."),
    ("f1",          "F1"),
]


# ---------------------------------------------------------------------------
# Per-site computation
# ---------------------------------------------------------------------------

def compute_per_site_metrics(
    y_true:   np.ndarray,
    y_prob:   np.ndarray,
    site_ids: np.ndarray,
    threshold: float = 0.5,
    min_site_n: int  = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute AUC, sensitivity, specificity, and F1 for each site.

    Sites with fewer than ``min_site_n`` subjects are marked with NaN for
    metrics that require both classes (e.g. AUC) but still counted.

    Returns
    -------
    dict mapping site_id → {metric_key: float, "n": int, "n_pos": int}
    """
    from sklearn.metrics import roc_auc_score

    site_ids = np.asarray(site_ids, dtype=str)
    unique_sites = sorted(np.unique(site_ids))
    results: Dict[str, Dict[str, Any]] = {}

    for site in unique_sites:
        mask = site_ids == site
        yt   = y_true[mask]
        yp   = y_prob[mask]
        yhat = (yp >= threshold).astype(int)

        n     = int(mask.sum())
        n_pos = int(yt.sum())
        n_neg = n - n_pos

        metrics: Dict[str, Any] = {"n": n, "n_pos": n_pos}

        if n < min_site_n or n_pos == 0 or n_neg == 0:
            metrics.update({
                "auc":         float("nan"),
                "sensitivity": float("nan"),
                "specificity": float("nan"),
                "f1":          float("nan"),
            })
        else:
            tp = int(np.sum((yhat == 1) & (yt == 1)))
            tn = int(np.sum((yhat == 0) & (yt == 0)))
            fp = int(np.sum((yhat == 1) & (yt == 0)))
            fn = int(np.sum((yhat == 0) & (yt == 1)))

            sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            f1   = (
                2 * prec * sens / (prec + sens)
                if not (np.isnan(prec) or np.isnan(sens) or (prec + sens) == 0)
                else float("nan")
            )
            try:
                auc_val = float(roc_auc_score(yt, yp))
            except ValueError:
                auc_val = float("nan")

            metrics.update({
                "auc":         auc_val,
                "sensitivity": sens,
                "specificity": spec,
                "f1":          f1,
            })

        results[site] = metrics

    return results


def _grand_total_row(
    y_true:   np.ndarray,
    y_prob:   np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute pooled metrics across all subjects."""
    from sklearn.metrics import roc_auc_score

    yhat  = (y_prob >= threshold).astype(int)
    n     = len(y_true)
    n_pos = int(y_true.sum())
    tp = int(np.sum((yhat == 1) & (y_true == 1)))
    tn = int(np.sum((yhat == 0) & (y_true == 0)))
    fp = int(np.sum((yhat == 1) & (y_true == 0)))
    fn = int(np.sum((yhat == 0) & (y_true == 1)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1   = (
        2 * prec * sens / (prec + sens)
        if not (np.isnan(prec) or np.isnan(sens) or (prec + sens) == 0)
        else float("nan")
    )
    try:
        auc_val = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc_val = float("nan")

    return {
        "auc": auc_val, "sensitivity": sens,
        "specificity": spec, "f1": f1,
        "n": n, "n_pos": n_pos,
    }


# ---------------------------------------------------------------------------
# Heatmap figure
# ---------------------------------------------------------------------------

def plot_site_heatmap(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    site_ids:  np.ndarray,
    threshold: float           = 0.5,
    metrics:   Optional[List[Tuple[str, str]]] = None,
    min_site_n: int            = 3,
    cmap:      str             = "YlOrRd",
    vmin:      float           = 0.4,
    vmax:      float           = 1.0,
    title:     str             = "Per-site Performance",
    show_grand_total: bool     = True,
) -> plt.Figure:
    """
    Render a heatmap of per-site classification metrics.

    Parameters
    ----------
    y_true / y_prob / site_ids : arrays of equal length
    threshold     : decision threshold for binary predictions
    metrics       : list of (key, label) pairs; defaults to AUC/Sens./Spec./F1
    min_site_n    : sites with fewer subjects show NaN (greyed out)
    cmap          : matplotlib colormap for cell colour
    vmin / vmax   : colour scale range (set to metric value range of interest)
    show_grand_total : append an "All sites" summary row

    Returns
    -------
    matplotlib.figure.Figure
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS

    site_data  = compute_per_site_metrics(
        y_true, y_prob, site_ids, threshold, min_site_n
    )
    site_names = sorted(site_data.keys())

    if show_grand_total:
        grand = _grand_total_row(y_true, y_prob, threshold)
        site_data["All sites"] = grand
        site_names.append("All sites")

    n_sites   = len(site_names)
    n_metrics = len(metrics)

    # Build data matrix: rows = sites, cols = metrics
    data = np.full((n_sites, n_metrics), np.nan)
    for r, site in enumerate(site_names):
        for c, (key, _) in enumerate(metrics):
            val = site_data[site].get(key, float("nan"))
            data[r, c] = float(val)

    # Auto-size figure: more sites → taller
    fig_h = max(FIG_HEIGHT, 0.5 * n_sites + 0.8)
    fig_w = SINGLE_COL_W + 0.6 * n_metrics

    with ieee_style():
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        # NaN mask for colouring
        masked = np.ma.masked_invalid(data)
        norm   = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        cmap_obj.set_bad(color="whitesmoke")

        im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, norm=norm)

        # Axes labels
        ax.set_xticks(range(n_metrics))
        ax.set_xticklabels([label for _, label in metrics], fontsize=8)
        ax.set_yticks(range(n_sites))
        ax.set_yticklabels(
            [
                f"{site}  (n={site_data[site]['n']})"
                for site in site_names
            ],
            fontsize=7.5,
        )

        # Divider between sites and grand-total row
        if show_grand_total and n_sites > 1:
            ax.axhline(n_sites - 1.5, color="black", lw=1.2, ls="--")

        # Cell annotations
        for r in range(n_sites):
            for c in range(n_metrics):
                val = data[r, c]
                text = f"{val:.3f}" if not np.isnan(val) else "N/A"
                # Dark text on light cells, light text on dark cells
                bg_colour = cmap_obj(norm(val)) if not np.isnan(val) else (1, 1, 1, 1)
                # Luminance ~ 0.299R + 0.587G + 0.114B
                lum = 0.299 * bg_colour[0] + 0.587 * bg_colour[1] + 0.114 * bg_colour[2]
                text_color = "black" if lum > 0.45 else "white"
                ax.text(c, r, text,
                        ha="center", va="center",
                        fontsize=6.5, color=text_color)

        # Colour bar
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Metric value", fontsize=7)

        ax.set_title(title, fontsize=9, pad=6)
        fig.tight_layout()

    return fig
