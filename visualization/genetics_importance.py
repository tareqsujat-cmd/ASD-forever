"""
Gene feature importance and attention heatmap visualization.

Produces horizontal bar charts for per-gene importance scores and 2-D
heatmaps for attention matrices (within the genetics transformer or
cross-modal fusion).
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.colors import Normalize

from visualization.style import ieee_style, _ensure_fig_ax, COLORS, SINGLE_COL_W

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gene importance bar chart
# ---------------------------------------------------------------------------

def plot_gene_importance(
    importances: Union[np.ndarray, Dict[str, float]],
    gene_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
    method_label: str = "Importance",
    color: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    Horizontal bar chart showing the top-k most important genes.

    Parameters
    ----------
    importances  : (n_genes,) array or dict {gene_name: score}
    gene_names   : gene labels; auto-generated ("Gene 0", …) if None
    top_k        : number of genes to display (sorted by importance)
    method_label : x-axis label (e.g. "IG Score", "TabNet Importance")
    color        : bar colour

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Normalise input
    if isinstance(importances, dict):
        gene_names = list(importances.keys())
        imp_arr    = np.array(list(importances.values()), dtype=float)
    else:
        imp_arr = np.asarray(importances, dtype=float)
        if gene_names is None:
            gene_names = [f"Gene {i}" for i in range(len(imp_arr))]

    n_genes = len(imp_arr)
    top_k = min(top_k, n_genes)

    # Select top-k by absolute importance
    order   = np.argsort(np.abs(imp_arr))[::-1][:top_k]
    order   = order[np.argsort(imp_arr[order])]  # sort ascending for horizontal bar
    vals    = imp_arr[order]
    names   = [gene_names[i] for i in order]

    if color is None:
        color = COLORS["model_a"]

    # Signed bars: positive = promoting ASD, negative = opposing
    bar_colors = [
        COLORS["asd"] if v >= 0 else COLORS["tc"]
        for v in vals
    ]

    with ieee_style():
        height = max(2.0, top_k * 0.22)
        fig, ax = _ensure_fig_ax(ax, figsize=(SINGLE_COL_W, height))

        ypos = np.arange(top_k)
        ax.barh(ypos, vals, color=bar_colors, alpha=0.85, height=0.7)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=7)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel(method_label, fontsize=8)
        ax.set_title(f"Top {top_k} Gene Importances")
        ax.grid(True, axis="x", linewidth=0.5, alpha=0.4)

        # Colour legend
        from matplotlib.patches import Patch
        patches = [
            Patch(color=COLORS["asd"], label="Pro-ASD"),
            Patch(color=COLORS["tc"],  label="Pro-TC"),
        ]
        ax.legend(handles=patches, fontsize=7, loc="lower right")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="tight_layout", category=UserWarning)
            fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    attn_matrix: np.ndarray,
    token_labels: Optional[Sequence[str]] = None,
    head_idx: int = 0,
    layer_idx: int = -1,
    title: Optional[str] = None,
    cmap: str = "viridis",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    2-D heatmap of a transformer attention matrix.

    Parameters
    ----------
    attn_matrix  : (n_heads, N, N) or (B, n_heads, N, N) or (N, N)
                   Attention weights.  If batched, the first sample is shown.
    token_labels : labels for each token position
    head_idx     : which attention head to visualise
    layer_idx    : used only in the title for reference
    title        : axes title; auto-generated if None
    cmap         : matplotlib colormap

    Returns
    -------
    matplotlib.figure.Figure
    """
    A = np.asarray(attn_matrix, dtype=float)

    # Shape normalisation
    if A.ndim == 4:          # (B, n_heads, N, N) → first sample
        A = A[0]
    if A.ndim == 3:          # (n_heads, N, N) → select head
        A = A[head_idx % A.shape[0]]
    # Now A is (N, N)
    N = A.shape[0]

    if token_labels is None:
        token_labels = ["CLS"] + [f"G{i}" for i in range(N - 1)] if N > 1 \
                       else [str(i) for i in range(N)]

    max_labels = 30  # avoid cluttered ticks
    show_labels = token_labels if N <= max_labels else None

    with ieee_style():
        size = min(7.0, max(3.0, N * 0.22))
        fig, ax = _ensure_fig_ax(ax, figsize=(size, size * 0.9))

        im = ax.imshow(A, cmap=cmap, vmin=0, vmax=A.max() + 1e-8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

        if show_labels is not None:
            ax.set_xticks(range(N))
            ax.set_yticks(range(N))
            ax.set_xticklabels(show_labels, rotation=90, fontsize=6)
            ax.set_yticklabels(show_labels, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_xlabel("Key token")
        ax.set_ylabel("Query token")
        if title is None:
            title = f"Attention (head {head_idx})"
        ax.set_title(title)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="tight_layout", category=UserWarning)
            fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Multi-head attention summary
# ---------------------------------------------------------------------------

def plot_multihead_attention(
    attn_matrix: np.ndarray,
    token_labels: Optional[Sequence[str]] = None,
    n_cols: int = 4,
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Grid of heatmaps — one per attention head.

    Parameters
    ----------
    attn_matrix : (n_heads, N, N) or (B, n_heads, N, N)

    Returns
    -------
    matplotlib.figure.Figure
    """
    A = np.asarray(attn_matrix, dtype=float)
    if A.ndim == 4:
        A = A[0]
    n_heads = A.shape[0]
    N       = A.shape[1]

    n_rows = int(np.ceil(n_heads / n_cols))
    with ieee_style():
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 1.8, n_rows * 1.8))
        axes = np.array(axes).reshape(-1)

        for hi in range(n_cols * n_rows):
            ax = axes[hi]
            if hi >= n_heads:
                ax.axis("off")
                continue
            ax.imshow(A[hi], cmap=cmap, vmin=0, vmax=A.max())
            ax.set_title(f"h{hi}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle("Multi-head Attention", fontsize=9)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="tight_layout", category=UserWarning)
            fig.tight_layout()

    return fig
