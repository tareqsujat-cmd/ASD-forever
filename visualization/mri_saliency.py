"""
MRI volume + saliency map overlay visualization.

Produces axial, coronal and sagittal slice panels with a semi-transparent
saliency heatmap overlaid on the anatomical image.

Usage
-----
::

    fig = plot_mri_triplet(volume, saliency, title="GradCAM")
    fig = plot_mri_slice(volume, saliency, dim=0, idx=48)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from visualization.style import ieee_style, DOUBLE_COL_W

logger = logging.getLogger(__name__)


def _prep_volume(volume: np.ndarray) -> np.ndarray:
    """Ensure volume is (D, H, W) float in [0, 1]."""
    v = np.asarray(volume, dtype=float)
    if v.ndim == 4:
        # (C, D, H, W) — take first channel
        v = v[0]
    if v.ndim != 3:
        raise ValueError(f"volume must be (D,H,W) or (C,D,H,W), got {v.shape}")
    vmin, vmax = v.min(), v.max()
    if vmax > vmin:
        v = (v - vmin) / (vmax - vmin)
    return v


def _prep_saliency(saliency: np.ndarray) -> np.ndarray:
    """Ensure saliency is (D, H, W) float in [0, 1]."""
    s = np.asarray(saliency, dtype=float)
    if s.ndim != 3:
        raise ValueError(f"saliency must be (D,H,W), got {s.shape}")
    s = np.clip(s, 0, None)
    smax = s.max()
    if smax > 0:
        s = s / smax
    return s


def _slice(volume: np.ndarray, dim: int, idx: int) -> np.ndarray:
    """Extract a 2D slice from a 3D volume."""
    return np.take(volume, idx, axis=dim)


def plot_mri_slice(
    volume: np.ndarray,
    saliency: Optional[np.ndarray] = None,
    dim: int = 1,
    idx: Optional[int] = None,
    alpha: float = 0.5,
    cmap_vol: str = "gray",
    cmap_sal: str = "hot",
    title: str = "",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    Plot a single 2D slice from a 3D volume with optional saliency overlay.

    Parameters
    ----------
    volume   : (D, H, W) or (C, D, H, W) MRI array
    saliency : (D, H, W) saliency / CAM map; overlaid in colour if given
    dim      : axis along which to slice (0=sagittal,1=coronal,2=axial)
    idx      : slice index; defaults to the midpoint
    alpha    : saliency overlay transparency
    cmap_vol : colormap for MRI grayscale image
    cmap_sal : colormap for saliency (e.g. "hot", "jet", "turbo")
    title    : axes title

    Returns
    -------
    matplotlib.figure.Figure
    """
    vol = _prep_volume(volume)
    D = vol.shape[dim]
    if idx is None:
        idx = D // 2
    idx = int(np.clip(idx, 0, D - 1))

    slice_vol = _slice(vol, dim, idx)

    with ieee_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(2.5, 2.5))
        else:
            fig = ax.figure

        ax.imshow(slice_vol.T, cmap=cmap_vol, origin="lower", aspect="equal")

        if saliency is not None:
            sal = _prep_saliency(saliency)
            slice_sal = _slice(sal, dim, idx)
            ax.imshow(slice_sal.T, cmap=cmap_sal, alpha=alpha,
                      origin="lower", aspect="equal", vmin=0, vmax=1)

        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=8)

    return fig


def plot_mri_triplet(
    volume: np.ndarray,
    saliency: Optional[np.ndarray] = None,
    idx_sagittal: Optional[int] = None,
    idx_coronal:  Optional[int] = None,
    idx_axial:    Optional[int] = None,
    alpha: float = 0.5,
    cmap_sal: str = "hot",
    title: str = "MRI Saliency",
) -> plt.Figure:
    """
    Three-panel figure showing axial, coronal and sagittal slices.

    Parameters
    ----------
    volume       : (D, H, W) or (C, D, H, W) MRI array
    saliency     : (D, H, W) saliency map (e.g. GradCAM output)
    idx_sagittal : slice index along dim=0; defaults to mid
    idx_coronal  : slice index along dim=1; defaults to mid
    idx_axial    : slice index along dim=2; defaults to mid
    alpha        : saliency overlay transparency
    cmap_sal     : colormap for saliency overlay
    title        : figure-level suptitle

    Returns
    -------
    matplotlib.figure.Figure
    """
    vol = _prep_volume(volume)
    D, H, W = vol.shape

    idxs = [
        (D // 2 if idx_sagittal is None else idx_sagittal),
        (H // 2 if idx_coronal  is None else idx_coronal),
        (W // 2 if idx_axial    is None else idx_axial),
    ]
    plane_names = ["Sagittal", "Coronal", "Axial"]

    sal = _prep_saliency(saliency) if saliency is not None else None

    with ieee_style():
        fig, axes = plt.subplots(
            1, 3, figsize=(DOUBLE_COL_W, DOUBLE_COL_W / 3 + 0.4)
        )
        for ax, dim, idx, name in zip(axes, [0, 1, 2], idxs, plane_names):
            slice_vol = _slice(vol, dim, idx)
            ax.imshow(slice_vol.T, cmap="gray", origin="lower", aspect="equal")
            if sal is not None:
                slice_sal = _slice(sal, dim, idx)
                ax.imshow(slice_sal.T, cmap=cmap_sal, alpha=alpha,
                          origin="lower", aspect="equal", vmin=0, vmax=1)
            ax.axis("off")
            ax.set_title(name, fontsize=8)

        # Shared colorbar for saliency
        if sal is not None:
            sm = cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap_sal)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                                fraction=0.015, pad=0.02)
            cbar.set_label("Saliency", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

        fig.suptitle(title, fontsize=9, y=1.01)
        fig.tight_layout()

    return fig


def plot_volume_grid(
    volumes: np.ndarray,
    saliencies: Optional[np.ndarray] = None,
    n_cols: int = 4,
    dim: int = 2,
    alpha: float = 0.5,
    cmap_sal: str = "hot",
    titles: Optional[list] = None,
) -> plt.Figure:
    """
    Grid of axial slices (one per subject in the batch).

    Parameters
    ----------
    volumes    : (B, D, H, W) or (B, C, D, H, W) batch of volumes
    saliencies : (B, D, H, W) matching batch of saliency maps
    n_cols     : columns in the grid
    dim        : slice dimension

    Returns
    -------
    matplotlib.figure.Figure
    """
    volumes = np.asarray(volumes)
    if volumes.ndim == 5:
        volumes = volumes[:, 0]   # take first channel
    B = volumes.shape[0]
    n_rows = int(np.ceil(B / n_cols))

    with ieee_style():
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 2.0, n_rows * 2.0))
        axes = np.array(axes).reshape(-1) if B > 1 else np.array([axes])

        for i in range(n_cols * n_rows):
            ax = axes[i]
            if i >= B:
                ax.axis("off")
                continue
            vol = _prep_volume(volumes[i])
            D = vol.shape[dim]
            idx = D // 2
            ax.imshow(_slice(vol, dim, idx).T,
                      cmap="gray", origin="lower", aspect="equal")
            if saliencies is not None:
                sal = _prep_saliency(saliencies[i])
                ax.imshow(_slice(sal, dim, idx).T,
                          cmap=cmap_sal, alpha=alpha,
                          origin="lower", aspect="equal", vmin=0, vmax=1)
            ax.axis("off")
            if titles is not None and i < len(titles):
                ax.set_title(str(titles[i]), fontsize=7)

        fig.tight_layout()
    return fig
