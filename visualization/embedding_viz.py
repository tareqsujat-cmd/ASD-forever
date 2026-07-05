"""
t-SNE and UMAP visualization of learned feature embeddings.

Usage
-----
::

    from visualization.embedding_viz import compute_embedding, plot_embedding

    emb = compute_embedding(features, method="tsne")   # (N, 2)
    fig = plot_embedding(emb, labels, site_ids=site_ids)

Both t-SNE (sklearn) and UMAP (umap-learn, optional) are supported.
Falls back to PCA-initialized t-SNE if UMAP is not installed.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.patches as mpatches

from visualization.style import ieee_style, _ensure_fig_ax, COLORS, SINGLE_COL_W

logger = logging.getLogger(__name__)

try:
    import umap as _umap_module
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.debug("umap-learn not installed; UMAP will fall back to t-SNE")


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

def compute_embedding(
    features: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    pca_init_dims: int = 50,
) -> np.ndarray:
    """
    Reduce high-dimensional features to 2D for visualization.

    Parameters
    ----------
    features     : (N, D) float array
    method       : "tsne" | "umap" | "pca"
    perplexity   : t-SNE perplexity
    n_neighbors  : UMAP n_neighbors
    min_dist     : UMAP min_dist
    pca_init_dims : number of PCA components used to initialize t-SNE
                    (speeds up t-SNE for D > 50)

    Returns
    -------
    embedding : (N, n_components) float array
    """
    features = np.asarray(features, dtype=float)
    N, D = features.shape

    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components,
                   random_state=random_state).fit_transform(features)

    # PCA pre-processing for t-SNE when D is large
    init_method = "pca"
    pca_features = features
    if D > pca_init_dims and method == "tsne":
        from sklearn.decomposition import PCA
        pca_features = PCA(n_components=min(pca_init_dims, N - 1),
                           random_state=random_state).fit_transform(features)

    if method == "umap":
        if HAS_UMAP:
            reducer = _umap_module.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            return reducer.fit_transform(features)
        else:
            logger.warning("umap-learn not installed; falling back to t-SNE")
            method = "tsne"

    # t-SNE (default)
    from sklearn.manifold import TSNE
    perp = min(perplexity, (N - 1) / 3.0)
    tsne_kwargs = dict(
        n_components=n_components,
        perplexity=perp,
        init=init_method,
        random_state=random_state,
    )
    import sklearn
    _sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if _sk_version >= (1, 4):
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(pca_features)


# ---------------------------------------------------------------------------
# Embedding plot
# ---------------------------------------------------------------------------

def plot_embedding(
    embedding: np.ndarray,
    labels: np.ndarray,
    site_ids: Optional[np.ndarray] = None,
    class_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    Scatter plot of a 2D embedding coloured by class and optionally marked
    by site.

    Parameters
    ----------
    embedding   : (N, 2) float array
    labels      : (N,) binary labels (0=TC, 1=ASD)
    site_ids    : (N,) optional site identifiers for marker shapes
    class_names : ["TC", "ASD"] by default
    title       : figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    embedding = np.asarray(embedding)
    labels    = np.asarray(labels)
    if class_names is None:
        class_names = ["TC", "ASD"]

    class_colors = [COLORS["tc"], COLORS["asd"]]

    # Marker cycle for different sites
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]

    unique_sites = (
        np.unique(site_ids) if site_ids is not None else np.array(["all"])
    )

    with ieee_style():
        fig, ax = _ensure_fig_ax(ax, figsize=(SINGLE_COL_W, SINGLE_COL_W))

        for si, site in enumerate(unique_sites):
            site_mask = (
                site_ids == site if site_ids is not None
                else np.ones(len(labels), dtype=bool)
            )
            marker = markers[si % len(markers)]

            for ci, (cls_name, cls_color) in enumerate(
                zip(class_names, class_colors)
            ):
                mask = site_mask & (labels == ci)
                if mask.sum() == 0:
                    continue
                lbl = cls_name if site == unique_sites[0] else "_nolegend_"
                if site_ids is not None and si == 0:
                    lbl = f"{cls_name}"
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    c=cls_color, marker=marker,
                    s=12, alpha=0.7, linewidths=0,
                    label=lbl,
                    rasterized=True,
                )

        # Class legend
        class_patches = [
            mpatches.Patch(color=c, label=n)
            for n, c in zip(class_names, class_colors)
        ]
        legend1 = ax.legend(handles=class_patches, loc="upper right",
                            fontsize=7, title="Class", title_fontsize=7)
        ax.add_artist(legend1)

        # Site legend (marker shapes)
        if site_ids is not None and len(unique_sites) > 1:
            site_handles = [
                plt.Line2D([0], [0], marker=markers[si % len(markers)],
                           color="grey", markersize=5, linestyle="None",
                           label=str(s))
                for si, s in enumerate(unique_sites)
            ]
            ax.legend(handles=site_handles, loc="lower left",
                      fontsize=7, title="Site", title_fontsize=7)

        if title is None:
            title = "Feature Embedding"
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, linewidth=0.4, alpha=0.3)
        fig.tight_layout()

    return fig
