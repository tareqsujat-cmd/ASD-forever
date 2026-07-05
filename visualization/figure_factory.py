"""
FigureFactory — high-level interface for generating all IEEE-quality figures.

Handles consistent styling, output directory management, and multi-panel
summary figures for paper inclusion.

Usage
-----
::

    ff = FigureFactory(output_dir="results/figures")
    ff.roc_curve(y_true, y_prob, filename="fig1_roc")
    ff.confusion_matrix(report, filename="fig2_cm")
    ff.summary_panel(report, y_true, y_prob, filename="fig_summary")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from visualization.style import ieee_style, DOUBLE_COL_W, SINGLE_COL_W, IEEE_RC

logger = logging.getLogger(__name__)


class FigureFactory:
    """
    Centralised figure generation with consistent IEEE styling.

    Parameters
    ----------
    output_dir : str or Path — base directory for saved figures
    dpi : int — raster DPI (overrides rcParam default of 300)
    formats : list of str — file format(s) for ``save()``
              e.g. ["pdf", "png"] — both saved simultaneously
    """

    def __init__(
        self,
        output_dir: Union[str, Path, None] = None,
        dpi: int = 300,
        formats: Optional[List[str]] = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.formats = formats or ["pdf", "png"]

    # ------------------------------------------------------------------
    # Discriminative metrics
    # ------------------------------------------------------------------

    def roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        label: str = "Model",
        show_ci: bool = True,
        n_bootstrap: int = 500,
        filename: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        from visualization.roc_pr_curves import plot_roc_curve
        fig = plot_roc_curve(
            y_true, y_prob, label=label,
            show_ci=show_ci, n_bootstrap=n_bootstrap, **kwargs
        )
        if filename:
            self.save(fig, filename)
        return fig

    def pr_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        label: str = "Model",
        show_ci: bool = True,
        n_bootstrap: int = 500,
        filename: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        from visualization.roc_pr_curves import plot_pr_curve
        fig = plot_pr_curve(
            y_true, y_prob, label=label,
            show_ci=show_ci, n_bootstrap=n_bootstrap, **kwargs
        )
        if filename:
            self.save(fig, filename)
        return fig

    def roc_comparison(
        self,
        models: List[Dict],
        filename: Optional[str] = None,
    ) -> plt.Figure:
        from visualization.roc_pr_curves import plot_roc_comparison
        fig = plot_roc_comparison(models)
        if filename:
            self.save(fig, filename)
        return fig

    def pr_comparison(
        self,
        models: List[Dict],
        filename: Optional[str] = None,
    ) -> plt.Figure:
        from visualization.roc_pr_curves import plot_pr_comparison
        fig = plot_pr_comparison(models)
        if filename:
            self.save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def confusion_matrix(
        self,
        cm_or_report,
        normalize: bool = False,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        from visualization.confusion_matrix import (
            plot_confusion_matrix, cm_from_report
        )
        if isinstance(cm_or_report, np.ndarray):
            cm = cm_or_report
        else:
            cm = cm_from_report(cm_or_report)
        fig = plot_confusion_matrix(cm, normalize=normalize)
        if filename:
            self.save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        from visualization.calibration_plot import plot_reliability_diagram
        fig = plot_reliability_diagram(y_true, y_prob, n_bins=n_bins)
        if filename:
            self.save(fig, filename)
        return fig

    def brier_decomposition(
        self,
        brier_dict: dict,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        from visualization.calibration_plot import plot_brier_decomposition
        fig = plot_brier_decomposition(brier_dict)
        if filename:
            self.save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embedding(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        site_ids: Optional[np.ndarray] = None,
        method: str = "tsne",
        filename: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        from visualization.embedding_viz import compute_embedding, plot_embedding
        emb = compute_embedding(features, method=method, **kwargs)
        method_label = "UMAP" if method == "umap" else "t-SNE"
        fig = plot_embedding(emb, labels, site_ids=site_ids,
                             title=f"{method_label} Embedding")
        if filename:
            self.save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # MRI saliency
    # ------------------------------------------------------------------

    def mri_saliency(
        self,
        volume: np.ndarray,
        saliency: Optional[np.ndarray] = None,
        title: str = "GradCAM Saliency",
        filename: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        from visualization.mri_saliency import plot_mri_triplet
        fig = plot_mri_triplet(volume, saliency, title=title, **kwargs)
        if filename:
            self.save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Gene importance
    # ------------------------------------------------------------------

    def gene_importance(
        self,
        importances: Union[np.ndarray, Dict],
        gene_names: Optional[Sequence[str]] = None,
        top_k: int = 20,
        method_label: str = "Importance",
        filename: Optional[str] = None,
    ) -> plt.Figure:
        from visualization.genetics_importance import plot_gene_importance
        fig = plot_gene_importance(
            importances, gene_names=gene_names,
            top_k=top_k, method_label=method_label
        )
        if filename:
            self.save(fig, filename)
        return fig

    def attention_heatmap(
        self,
        attn_matrix: np.ndarray,
        token_labels: Optional[Sequence[str]] = None,
        filename: Optional[str] = None,
        **kwargs,
    ) -> plt.Figure:
        from visualization.genetics_importance import plot_attention_heatmap
        fig = plot_attention_heatmap(attn_matrix, token_labels=token_labels, **kwargs)
        if filename:
            self.save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # Summary panel (multi-panel paper figure)
    # ------------------------------------------------------------------

    def summary_panel(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        features: Optional[np.ndarray] = None,
        gene_importances: Optional[np.ndarray] = None,
        gene_names: Optional[Sequence[str]] = None,
        filename: Optional[str] = None,
    ) -> plt.Figure:
        """
        Six-panel summary figure for paper:
          (a) ROC  (b) PR  (c) Calibration
          (d) Confusion matrix  (e) Embedding  (f) Gene importance

        Panels (e) and (f) are skipped if features/gene_importances are None.
        """
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        from visualization.confusion_matrix import plot_confusion_matrix
        from visualization.calibration_plot import plot_reliability_diagram

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        n_panels = 4 + (1 if features is not None else 0) + \
                   (1 if gene_importances is not None else 0)
        n_cols = 3
        n_rows = int(np.ceil(n_panels / n_cols))

        with ieee_style():
            fig = plt.figure(figsize=(DOUBLE_COL_W, n_rows * 2.6))
            gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                    hspace=0.45, wspace=0.38)
            axes = [fig.add_subplot(gs[i // n_cols, i % n_cols])
                    for i in range(n_rows * n_cols)]
            panel_idx = 0

            # --- (a) ROC ---
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            axes[panel_idx].plot(fpr, tpr, linewidth=1.5, color="#2c7bb6",
                                  label=f"AUC={roc_auc:.3f}")
            axes[panel_idx].plot([0, 1], [0, 1], "--", color="#aaaaaa",
                                  linewidth=0.9)
            axes[panel_idx].set_xlabel("FPR"); axes[panel_idx].set_ylabel("TPR")
            axes[panel_idx].set_title("(a) ROC Curve")
            axes[panel_idx].legend(loc="lower right", fontsize=7)
            panel_idx += 1

            # --- (b) PR ---
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(y_true, y_prob)
            axes[panel_idx].plot(rec, prec, linewidth=1.5, color="#d7191c",
                                  label=f"AP={ap:.3f}")
            axes[panel_idx].axhline(y_true.mean(), linestyle="--",
                                     color="#aaaaaa", linewidth=0.9)
            axes[panel_idx].set_xlabel("Recall"); axes[panel_idx].set_ylabel("Precision")
            axes[panel_idx].set_title("(b) Precision-Recall")
            axes[panel_idx].legend(loc="upper right", fontsize=7)
            panel_idx += 1

            # --- (c) Calibration ---
            plot_reliability_diagram(y_true, y_prob, n_bins=10,
                                     show_histogram=False, ax=axes[panel_idx])
            axes[panel_idx].set_title("(c) Calibration")
            panel_idx += 1

            # --- (d) Confusion matrix ---
            # compute from 0.5 threshold
            y_pred = (y_prob >= 0.5).astype(int)
            TP = int(((y_pred == 1) & (y_true == 1)).sum())
            TN = int(((y_pred == 0) & (y_true == 0)).sum())
            FP = int(((y_pred == 1) & (y_true == 0)).sum())
            FN = int(((y_pred == 0) & (y_true == 1)).sum())
            cm = np.array([[TN, FP], [FN, TP]], dtype=float)
            plot_confusion_matrix(cm, ax=axes[panel_idx])
            axes[panel_idx].set_title("(d) Confusion Matrix")
            panel_idx += 1

            # --- (e) Embedding (optional) ---
            if features is not None:
                from visualization.embedding_viz import compute_embedding, plot_embedding
                try:
                    emb = compute_embedding(features, method="tsne",
                                            perplexity=min(30, len(features)//4 + 1))
                    plot_embedding(emb, y_true, ax=axes[panel_idx])
                    axes[panel_idx].set_title("(e) t-SNE Embedding")
                except Exception as e:
                    logger.warning("Embedding panel failed: %s", e)
                    axes[panel_idx].axis("off")
                panel_idx += 1

            # --- (f) Gene importance (optional) ---
            if gene_importances is not None:
                from visualization.genetics_importance import plot_gene_importance
                plot_gene_importance(gene_importances, gene_names=gene_names,
                                     top_k=10, ax=axes[panel_idx])
                axes[panel_idx].set_title("(f) Gene Importance")
                panel_idx += 1

            # Hide unused axes
            for i in range(panel_idx, len(axes)):
                axes[i].axis("off")

            fig.suptitle("ASD Detection — Model Evaluation Summary", fontsize=10)
            fig.tight_layout()

        if filename:
            self.save(fig, filename)
        return fig

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(
        self,
        fig: plt.Figure,
        filename: str,
        formats: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Save a figure to disk.

        Parameters
        ----------
        fig      : matplotlib Figure
        filename : stem (no extension); extension(s) from ``self.formats``
        formats  : override ``self.formats`` for this call

        Returns
        -------
        list of saved Path objects
        """
        fmts = formats or self.formats
        base = self.output_dir / filename if self.output_dir else Path(filename)
        base.parent.mkdir(parents=True, exist_ok=True)
        saved = []
        for fmt in fmts:
            p = base.with_suffix(f".{fmt}")
            fig.savefig(p, dpi=self.dpi, bbox_inches="tight")
            logger.info("Saved %s", p)
            saved.append(p)
        return saved
