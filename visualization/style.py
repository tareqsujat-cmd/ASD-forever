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
# Publication rcParams overlaid on the scienceplots base.  Tuned for a bolder,
# high-impact "Q1 / ICLR-style" look: thicker lines, larger readable fonts, and
# crisp axes — while staying vector/PDF and colour-blind-safe.
IEEE_RC: dict = {
    "font.family":        "serif",
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelweight":   "bold",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.3",
    "lines.linewidth":    2.2,     # thicker lines (Q1/ICLR)
    "lines.markersize":   6.0,
    "lines.markeredgewidth": 1.2,
    "patch.linewidth":    1.2,
    "axes.linewidth":     1.1,     # crisper axis spines
    "xtick.major.width":  1.1,
    "ytick.major.width":  1.1,
    "xtick.major.size":   4.5,
    "ytick.major.size":   4.5,
    "grid.linewidth":     0.7,
    "grid.alpha":         0.35,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "figure.dpi":         150,     # screen; overridden at savefig time
    "savefig.dpi":        400,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
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

# ---------------------------------------------------------------------------
# scienceplots integration
# ---------------------------------------------------------------------------
# All publication figures render on top of the scienceplots "science"+"ieee" base
# style, using the "no-latex" variant so figures build on machines without a LaTeX
# install (e.g. cloud/RunPod GPU boxes).  If scienceplots is missing we fall back
# gracefully to plain matplotlib + our IEEE_RC overrides.

_SCIENCEPLOTS_STYLES: Optional[list] = None   # resolved once, cached


def scienceplots_styles() -> list:
    """Return the best available scienceplots style list, or [] if unavailable."""
    global _SCIENCEPLOTS_STYLES
    if _SCIENCEPLOTS_STYLES is not None:
        return _SCIENCEPLOTS_STYLES

    styles: list = []
    try:
        import scienceplots  # noqa: F401  (importing registers the styles)
        for candidate in (["science", "ieee", "no-latex"],
                          ["science", "no-latex"],
                          ["science"]):
            try:
                with plt.style.context(candidate):
                    pass
                styles = candidate
                break
            except Exception:
                continue
    except Exception:
        styles = []

    _SCIENCEPLOTS_STYLES = styles
    return styles


@contextmanager
def ieee_style():
    """Context manager: scienceplots base + IEEE rcParams, restored on exit."""
    styles = scienceplots_styles()
    if styles:
        with plt.style.context(styles), plt.rc_context(IEEE_RC):
            yield
    else:
        with plt.rc_context(IEEE_RC):
            yield


def apply_ieee_style() -> None:
    """Apply scienceplots base + IEEE rcParams globally for the current process."""
    styles = scienceplots_styles()
    if styles:
        try:
            plt.style.use(styles)
        except Exception:
            pass
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
