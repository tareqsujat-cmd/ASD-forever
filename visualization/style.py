"""
Shared IEEE-publication style constants and helpers.

Apply with::

    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(...)

or call ``apply_ieee_style()`` to set globally for the process.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# IEEE figure dimensions (inches)
# ---------------------------------------------------------------------------
SINGLE_COL_W = 3.487   # 88.9 mm — one column in double-column IEEE layout
DOUBLE_COL_W = 7.166   # 182.0 mm — full page width
FIG_HEIGHT   = 2.6     # default single-panel height

# ---------------------------------------------------------------------------
# rcParams for IEEE Transactions style
# ---------------------------------------------------------------------------
IEEE_RC: dict = {
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.85,
    "lines.linewidth":   1.4,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.35,
    "axes.grid":         True,
    "figure.dpi":        150,   # screen; overridden at savefig time
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.03,
}

# ---------------------------------------------------------------------------
# Colorblind-safe palette (ColorBrewer Set1 / custom)
# ---------------------------------------------------------------------------
COLORS = {
    "asd":      "#e41a1c",   # red — ASD class
    "tc":       "#377eb8",   # blue — typically-developing class
    "model_a":  "#2c7bb6",   # deep blue
    "model_b":  "#d7191c",   # red-orange
    "model_c":  "#1a9641",   # green
    "model_d":  "#fdae61",   # amber
    "random":   "#aaaaaa",   # grey diagonal / random baseline
    "ci_fill":  "#a8d1e7",   # CI shading
    "perfect":  "#4dac26",   # ideal diagonal
}

# Ordered palette for multi-model comparison
PALETTE = [
    COLORS["model_a"], COLORS["model_b"], COLORS["model_c"],
    COLORS["model_d"], "#984ea3", "#ff7f00",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def ieee_style():
    """Context manager: apply IEEE rcParams, then restore previous settings."""
    with plt.rc_context(IEEE_RC):
        yield


def apply_ieee_style() -> None:
    """Apply IEEE rcParams globally for the current process."""
    mpl.rcParams.update(IEEE_RC)


def _ensure_fig_ax(ax: Optional[mpl.axes.Axes], figsize=None):
    """Return (fig, ax).  Creates a new figure if ax is None."""
    if ax is None:
        w = SINGLE_COL_W
        h = FIG_HEIGHT if figsize is None else figsize[1]
        if figsize is not None:
            w = figsize[0]
        fig, ax = plt.subplots(figsize=(w, h))
    else:
        fig = ax.figure
    return fig, ax
