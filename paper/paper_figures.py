"""
PaperFigureGenerator — produces all IEEE-ready figures for the ASD paper.

Each ``fig_*`` method corresponds to one entry in ``figure_specs.FIGURE_SPECS``.
Methods accept pre-computed numpy arrays / result objects so they can be
called from a training notebook without re-running experiments.

Usage
-----
::

    gen = PaperFigureGenerator(output_dir="paper/figures")
    gen.fig_roc_pr(models=[
        {"name": "Proposed",  "y_true": y_true, "y_prob": y_prob_proposed},
        {"name": "MRI-only",  "y_true": y_true, "y_prob": y_prob_mri},
    ])
    gen.generate_all(bundle)   # produces all figures from a results bundle

All saved figures are added to an internal manifest.  Call
``gen.latex_manifest()`` to get the full figure-list for the paper appendix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

from visualization.style import (
    ieee_style, COLORS, PALETTE,
    SINGLE_COL_W, DOUBLE_COL_W,
    _ensure_fig_ax,
)
from visualization.roc_pr_curves import (
    plot_roc_comparison, plot_pr_comparison,
    plot_roc_curve, plot_pr_curve,
)
from visualization.calibration_plot import (
    plot_reliability_diagram, plot_brier_decomposition,
)
from visualization.embedding_viz import compute_embedding, plot_embedding
from visualization.mri_saliency import plot_mri_triplet
from visualization.genetics_importance import plot_gene_importance
from visualization.decision_curve import plot_decision_curve
from visualization.lift_gain import plot_lift_gain_panel
from visualization.radar_chart import plot_radar_chart
from visualization.prediction_histogram import plot_confidence_histogram
from visualization.site_heatmap import plot_site_heatmap
from paper.figure_specs import FIGURE_SPECS, FigureSpec, get_spec

logger = logging.getLogger(__name__)

# Panel label font properties
_PANEL_FONT = dict(fontsize=9, fontweight="bold", va="top", ha="left")


# ---------------------------------------------------------------------------
# PaperFigureGenerator
# ---------------------------------------------------------------------------

class PaperFigureGenerator:
    """
    Generates and saves all IEEE paper figures.

    Parameters
    ----------
    output_dir : path-like
        Directory where figure files are written.
    formats : list of str
        File formats to save, e.g. ``["pdf", "png"]``.
    dpi : int
        Raster DPI (only for PNG/TIFF; PDF is always vector).
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "paper/figures",
        formats:    List[str]        = ("pdf", "png"),
        dpi:        int              = 300,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.formats    = list(formats)
        self.dpi        = dpi
        self._manifest: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Saving and manifest
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, spec: FigureSpec) -> List[Path]:
        """Save ``fig`` in all configured formats; update manifest."""
        paths = []
        for fmt in self.formats:
            dest = self.output_dir / f"{spec.filename}.{fmt}"
            kwargs = dict(bbox_inches="tight")
            if fmt != "pdf":
                kwargs["dpi"] = self.dpi
            fig.savefig(dest, **kwargs)
            paths.append(dest)
        self._manifest.append({"spec": spec, "paths": paths})
        logger.info("Saved %s → %s", spec.filename, [p.name for p in paths])
        return paths

    @staticmethod
    def _add_panel_labels(
        axes:   List[plt.Axes],
        labels: List[Tuple[int, str]],
    ) -> None:
        """
        Add bold panel labels "(a)", "(b)", … to specific axes.

        Parameters
        ----------
        axes   : list of Axes (flattened if needed)
        labels : [(ax_index, label_string), …]
        """
        for idx, text in labels:
            if idx < len(axes):
                ax = axes[idx]
                ax.text(
                    -0.12, 1.06, text,
                    transform=ax.transAxes,
                    **_PANEL_FONT,
                )

    def latex_manifest(self) -> str:
        """
        Return a LaTeX ``\\listoffigures``-compatible comment block plus
        full ``figure`` environments for all generated figures.
        """
        lines = [
            "% ============================================================",
            "% Auto-generated figure manifest",
            "% ============================================================",
            "",
        ]
        for entry in self._manifest:
            spec: FigureSpec = entry["spec"]
            lines.append(spec.latex_figure_env(formats=self.formats))
            lines.append("")
        return "\n".join(lines)

    def manifest_dict(self) -> List[dict]:
        """Return manifest as a list of plain dicts (label, paths)."""
        return [
            {
                "label":    e["spec"].label,
                "filename": e["spec"].filename,
                "paths":    [str(p) for p in e["paths"]],
            }
            for e in self._manifest
        ]

    # ------------------------------------------------------------------
    # Fig 1 — Architecture Overview
    # ------------------------------------------------------------------

    def fig_architecture_overview(self) -> plt.Figure:
        """
        Programmatic schematic of the multimodal fusion architecture.

        Uses FancyBboxPatch and annotate arrows — no external assets needed.
        """
        spec = get_spec("fig01_architecture")
        with ieee_style():
            fig, ax = plt.subplots(figsize=(spec.width_inches, 2.8))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 5)
            ax.axis("off")

            def _box(x, y, w, h, color, label, fontsize=7):
                patch = mpatches.FancyBboxPatch(
                    (x, y), w, h,
                    boxstyle="round,pad=0.1",
                    facecolor=color, edgecolor="black", linewidth=0.6,
                )
                ax.add_patch(patch)
                ax.text(x + w / 2, y + h / 2, label,
                        ha="center", va="center",
                        fontsize=fontsize, wrap=True,
                        multialignment="center")

            def _arrow(x1, y1, x2, y2):
                ax.annotate(
                    "", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>", color="black",
                        lw=0.8, mutation_scale=8,
                    ),
                )

            # Input boxes
            _box(0.1, 3.4, 1.5, 0.9, "#cce5ff", "MRI\nVolume\n(3D)")
            _box(0.1, 1.6, 1.5, 0.9, "#d4edda", "Genetics\nSNPs")

            # Encoder boxes
            _box(2.3, 3.4, 1.9, 0.9, "#b8d4f5", "3D Backbone\n(ResNet/Swin3D)")
            _box(2.3, 1.6, 1.9, 0.9, "#b8e4c9", "Transformer\nEncoder")

            # Feature vectors
            _box(4.7, 3.4, 1.1, 0.9, "#aec6cf", "MRI\nFeats\n(256d)")
            _box(4.7, 1.6, 1.1, 0.9, "#aec6cf", "Gen\nFeats\n(256d)")

            # Cross-attention fusion
            _box(6.4, 2.2, 1.6, 1.3, "#f5c6cb", "Cross-Attention\nFusion\n(256d)")

            # Classifier
            _box(8.5, 2.4, 1.3, 0.9, "#ffeeba", "Classifier\nMLP")

            # Output
            ax.text(9.85, 2.85, "ASD", ha="center", va="center",
                    fontsize=7, color=COLORS["asd"], fontweight="bold")
            ax.text(9.85, 2.35, "TC",  ha="center", va="center",
                    fontsize=7, color=COLORS["tc"],  fontweight="bold")

            # Arrows
            _arrow(1.6, 3.85, 2.3, 3.85)
            _arrow(1.6, 2.05, 2.3, 2.05)
            _arrow(4.2, 3.85, 4.7, 3.85)
            _arrow(4.2, 2.05, 4.7, 2.05)
            _arrow(5.8, 3.85, 6.6, 3.5)
            _arrow(5.8, 2.05, 6.6, 2.5)
            _arrow(8.0, 2.85, 8.5, 2.85)
            _arrow(9.8, 2.85, 9.8, 2.90)  # invisible — text labels
            # Output arrow
            ax.annotate("", xy=(9.8, 2.55), xytext=(9.0, 2.85),
                        arrowprops=dict(arrowstyle="-|>", color="black",
                                        lw=0.6, mutation_scale=6))

            ax.set_title("Multimodal ASD Detection Framework", fontsize=8)
            fig.tight_layout(pad=0.3)

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig 2 — Dataset Statistics
    # ------------------------------------------------------------------

    def fig_dataset_statistics(
        self,
        site_counts:  Dict[str, int],
        class_counts: Dict[str, int],
    ) -> plt.Figure:
        """
        (a) Site-level subject counts, (b) ASD / TC class balance.

        Parameters
        ----------
        site_counts  : {"site_name": n_subjects, ...}
        class_counts : {"ASD": n, "TC": n}
        """
        spec = get_spec("fig02_dataset_statistics")
        with ieee_style():
            fig, axes = plt.subplots(1, 2, figsize=(spec.width_inches, 2.0),
                                      gridspec_kw={"width_ratios": [2, 1]})

            # (a) Site bar chart
            ax = axes[0]
            sites  = sorted(site_counts.keys(),
                            key=lambda s: site_counts[s], reverse=True)
            counts = [site_counts[s] for s in sites]
            ypos   = np.arange(len(sites))
            ax.barh(ypos, counts, color=COLORS["model_a"], alpha=0.85, height=0.7)
            ax.set_yticks(ypos)
            ax.set_yticklabels(sites, fontsize=6)
            ax.set_xlabel("Subjects", fontsize=7)
            ax.set_title("Subjects per site", fontsize=7)

            # (b) Pie chart
            ax2 = axes[1]
            pie_labels = list(class_counts.keys())
            pie_vals   = [class_counts[k] for k in pie_labels]
            pie_colors = [COLORS.get(k.lower(), PALETTE[i])
                          for i, k in enumerate(pie_labels)]
            wedges, texts, autotexts = ax2.pie(
                pie_vals,
                labels       = pie_labels,
                colors       = pie_colors,
                autopct      = "%1.0f%%",
                startangle   = 90,
                textprops    = {"fontsize": 6},
            )
            for at in autotexts:
                at.set_fontsize(6)
            ax2.set_title("Class balance", fontsize=7)

            self._add_panel_labels(
                list(axes.flat), spec.panel_labels
            )
            fig.tight_layout()

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig 3 — ROC / PR comparison
    # ------------------------------------------------------------------

    def fig_roc_pr(
        self,
        models:      List[Dict],
        n_bootstrap: int = 1000,
    ) -> plt.Figure:
        """
        Side-by-side ROC and PR curves for multiple models with CI bands.

        Parameters
        ----------
        models : list of dicts with keys
            ``name``, ``y_true`` (array), ``y_prob`` (array),
            optionally ``color`` (hex string).
        n_bootstrap : int
            Number of bootstrap resamples for CI bands on the proposed model.
        """
        spec = get_spec("fig03_roc_pr")
        with ieee_style():
            fig, axes = plt.subplots(1, 2, figsize=(spec.width_inches, 2.3))
            plot_roc_comparison(models, ax=axes[0])
            plot_pr_comparison(models, ax=axes[1])
            axes[0].set_title("ROC", fontsize=7)
            axes[1].set_title("Precision–Recall", fontsize=7)
            self._add_panel_labels(list(axes.flat), spec.panel_labels)
            fig.tight_layout()

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig 4 — Ablation Study
    # ------------------------------------------------------------------

    def fig_ablation(
        self,
        ablation_results,
        primary_metric: str = "val_auc",
        baseline_name:  str = "baseline",
    ) -> plt.Figure:
        """
        Horizontal bar chart of ablation variants vs baseline.

        Parameters
        ----------
        ablation_results : AblationResults
        """
        from ablation.ablation_analyzer import AblationAnalyzer

        spec     = get_spec("fig04_ablation")
        analyzer = AblationAnalyzer(ablation_results, baseline_name=baseline_name)

        with ieee_style():
            fig = analyzer.plot_comparison(
                metric = primary_metric,
                title  = None,
            )
            fig.set_size_inches(spec.width_inches, fig.get_figheight())
            fig.tight_layout()

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig 5 — Explainability Panel
    # ------------------------------------------------------------------

    def fig_explainability(
        self,
        mri_volume:       np.ndarray,
        saliency:         np.ndarray,
        gene_importances: np.ndarray,
        gene_names:       List[str],
        top_k:            int = 20,
    ) -> plt.Figure:
        """
        (a) MRI triplet with GradCAM++ saliency, (b) top-k gene importance bars.

        Parameters
        ----------
        mri_volume       : (D, H, W) float array
        saliency         : (D, H, W) float array, non-negative
        gene_importances : (n_genes,) float array
        gene_names       : list of gene/SNP identifiers
        """
        spec = get_spec("fig05_explainability")
        with ieee_style():
            fig = plt.figure(figsize=(spec.width_inches, 2.5))
            gs  = fig.add_gridspec(
                1, 2, width_ratios=[2.2, 1],
                left=0.04, right=0.97, wspace=0.35,
            )

            # (a) MRI triplet
            gs_mri = gs[0].subgridspec(1, 3, wspace=0.05)
            ax_axial    = fig.add_subplot(gs_mri[0])
            ax_coronal  = fig.add_subplot(gs_mri[1])
            ax_sagittal = fig.add_subplot(gs_mri[2])
            mri_axes    = [ax_axial, ax_coronal, ax_sagittal]

            D, H, W = mri_volume.shape
            slices = [
                (mri_volume[D // 2], saliency[D // 2]),       # axial
                (mri_volume[:, H // 2, :], saliency[:, H // 2, :]),  # coronal
                (mri_volume[:, :, W // 2], saliency[:, :, W // 2]),  # sagittal
            ]
            labels_mri = ["Axial", "Coronal", "Sagittal"]
            for ax, (vol_sl, sal_sl), lbl in zip(mri_axes, slices, labels_mri):
                ax.imshow(vol_sl, cmap="gray", aspect="auto")
                if sal_sl.max() > 0:
                    ax.imshow(sal_sl, cmap="hot", alpha=0.45, aspect="auto",
                              vmin=0, vmax=sal_sl.max())
                ax.set_title(lbl, fontsize=6)
                ax.axis("off")

            # (b) Gene importance
            ax_gene = fig.add_subplot(gs[1])
            plot_gene_importance(
                importances = gene_importances,
                gene_names  = gene_names,
                top_k       = top_k,
                ax          = ax_gene,
            )
            ax_gene.set_title(f"Top-{top_k} SNPs", fontsize=6)

            # Panel labels
            self._add_panel_labels(
                [ax_axial, ax_gene], [(0, "(a)"), (1, "(b)")]
            )
            fig.suptitle("Explainability analysis", fontsize=7, y=1.02)

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig 6 — Cross-Validation Summary
    # ------------------------------------------------------------------

    def fig_cv_summary(
        self,
        cv_results:  List[Dict[str, Any]],
        metrics:     Optional[List[str]] = None,
        model_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Box plots of per-fold metrics across models.

        Parameters
        ----------
        cv_results : list of dicts, each with keys
            ``"model"`` (str), ``"fold"`` (int), and metric name → value.
        metrics : list of metric keys to plot (defaults to val_auc, val_acc, val_f1).
        """
        _HEADERS = {
            "val_auc": "AUC",
            "val_acc": "Accuracy",
            "val_f1":  "F1",
            "val_sensitivity": "Sn",
            "val_specificity": "Sp",
        }
        if metrics is None:
            metrics = ["val_auc", "val_acc", "val_f1"]
        if model_names is None:
            model_names = sorted({r["model"] for r in cv_results})

        spec = get_spec("fig06_cv_summary")
        n_metrics = len(metrics)

        with ieee_style():
            fig, axes = plt.subplots(
                1, n_metrics,
                figsize=(spec.width_inches, 2.0),
                sharey=False,
            )
            if n_metrics == 1:
                axes = [axes]

            for ax, metric in zip(axes, metrics):
                data_per_model = []
                for mn in model_names:
                    fold_vals = [
                        r[metric]
                        for r in cv_results
                        if r["model"] == mn and metric in r
                    ]
                    data_per_model.append(fold_vals)

                valid = [(m, d) for m, d in zip(model_names, data_per_model) if d]
                if not valid:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=6)
                    continue

                names_v, data_v = zip(*valid)
                bp = ax.boxplot(
                    data_v,
                    patch_artist  = True,
                    medianprops   = dict(color="black", lw=1.0),
                    whiskerprops  = dict(lw=0.8),
                    capprops      = dict(lw=0.8),
                    flierprops    = dict(marker=".", markersize=3),
                    widths        = 0.5,
                )
                for patch, color in zip(bp["boxes"],
                                        [PALETTE[i % len(PALETTE)]
                                         for i in range(len(names_v))]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Diamond for mean
                for i, d in enumerate(data_v):
                    ax.scatter(i + 1, np.mean(d), marker="D",
                               color="black", s=10, zorder=5)

                ax.set_xticks(range(1, len(names_v) + 1))
                ax.set_xticklabels(names_v, fontsize=6, rotation=25, ha="right")
                ax.set_ylabel(_HEADERS.get(metric, metric), fontsize=7)
                ax.set_title(_HEADERS.get(metric, metric), fontsize=7)

            fig.suptitle("5-fold cross-validation results", fontsize=7)
            fig.tight_layout()

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig 7 — Calibration Panel
    # ------------------------------------------------------------------

    def fig_calibration(
        self,
        y_true:    np.ndarray,
        y_prob:    np.ndarray,
        n_bins:    int = 10,
        brier_dict: Optional[Dict[str, float]] = None,
    ) -> plt.Figure:
        """
        (a) Reliability diagram, (b) Brier decomposition.

        Parameters
        ----------
        brier_dict : {"brier": float, "calibration": float,
                      "resolution": float, "uncertainty": float}
                     If None, only the reliability diagram is shown.
        """
        spec = get_spec("fig07_calibration")
        with ieee_style():
            if brier_dict is not None:
                fig, axes = plt.subplots(
                    1, 2, figsize=(spec.width_inches, 2.0),
                    gridspec_kw={"width_ratios": [1.4, 1]},
                )
                plot_reliability_diagram(y_true, y_prob, n_bins=n_bins, ax=axes[0])
                plot_brier_decomposition(brier_dict, ax=axes[1])
                self._add_panel_labels(list(axes.flat), spec.panel_labels)
            else:
                fig, ax = plt.subplots(figsize=(spec.width_inches, 2.0))
                plot_reliability_diagram(y_true, y_prob, n_bins=n_bins, ax=ax)
            fig.tight_layout()

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig 8 — Embedding Visualization
    # ------------------------------------------------------------------

    def fig_embedding(
        self,
        features:  np.ndarray,
        labels:    np.ndarray,
        site_ids:  Optional[np.ndarray] = None,
        method:    str                  = "tsne",
    ) -> plt.Figure:
        """
        t-SNE / UMAP of learned representations coloured by diagnosis.

        Parameters
        ----------
        features  : (N, d) array of fused feature vectors
        labels    : (N,) int array  0=TC, 1=ASD
        site_ids  : (N,) int array  acquisition site index (for marker shapes)
        method    : ``"tsne"`` | ``"umap"`` | ``"pca"``
        """
        spec = get_spec("fig08_embedding")
        embedding = compute_embedding(features, method=method, n_components=2)

        with ieee_style():
            fig, ax = plt.subplots(figsize=(spec.width_inches, spec.width_inches))
            plot_embedding(
                embedding  = embedding,
                labels     = labels,
                site_ids   = site_ids,
                class_names = {0: "TC", 1: "ASD"},
                title      = f"{method.upper()} of fused representations",
                ax         = ax,
            )
            fig.tight_layout()

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Supplementary S1 — HPO Summary
    # ------------------------------------------------------------------

    def fig_hpo_summary(self, tuner) -> plt.Figure:
        """
        (a) Optimization history, (b) hyperparameter importance.

        Parameters
        ----------
        tuner : ASDTuner (must have been optimized)
        """
        from hyperparameter_tuning.analysis import TuningAnalyzer

        spec     = get_spec("figS1_hpo")
        analyzer = TuningAnalyzer(tuner)

        with ieee_style():
            fig, axes = plt.subplots(1, 2, figsize=(spec.width_inches, 2.2))
            analyzer.plot_optimization_history(ax=axes[0])
            analyzer.plot_param_importance(ax=axes[1])
            self._add_panel_labels(list(axes.flat), spec.panel_labels)
            fig.suptitle("Bayesian HPO summary", fontsize=7)
            fig.tight_layout()

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig S2 — Decision Curve Analysis
    # ------------------------------------------------------------------

    def fig_decision_curve(
        self,
        y_true: np.ndarray,
        models: List[Dict[str, Any]],
        t_min:  float = 0.01,
        t_max:  float = 0.60,
    ) -> plt.Figure:
        """
        Decision curve analysis — net benefit vs. threshold probability.

        Parameters
        ----------
        y_true : binary labels
        models : list of {"name", "y_prob", "color" (opt)}
        t_min / t_max : threshold range to display
        """
        spec = get_spec("figS2_decision_curve")
        with ieee_style():
            fig = plot_decision_curve(
                y_true  = y_true,
                models  = models,
                t_min   = t_min,
                t_max   = t_max,
                title   = "Decision Curve Analysis",
            )

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig S3 — Lift and Gain Curves
    # ------------------------------------------------------------------

    def fig_lift_gain(
        self,
        y_true: np.ndarray,
        models: List[Dict[str, Any]],
    ) -> plt.Figure:
        """
        Two-panel gain + lift figure.

        Parameters
        ----------
        models : list of {"name", "y_prob", "color" (opt)}
        """
        spec = get_spec("figS3_lift_gain")
        with ieee_style():
            fig = plot_lift_gain_panel(
                y_true = y_true,
                models = models,
                title  = "Lift and Gain",
            )

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig S4 — Metric Radar Chart
    # ------------------------------------------------------------------

    def fig_radar(
        self,
        models: List[Dict[str, Any]],
    ) -> plt.Figure:
        """
        Radar chart comparing models across 8 clinical metrics.

        Parameters
        ----------
        models : list of {"name", "metrics": {key: value}, "color" (opt)}
                 Metric keys: sensitivity, specificity, ppv, npv, f1, auc, mcc, kappa
        """
        spec = get_spec("figS4_radar_chart")
        with ieee_style():
            fig = plot_radar_chart(
                models = models,
                title  = "Metric Radar Chart",
            )

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig S5 — Confidence Histogram
    # ------------------------------------------------------------------

    def fig_confidence_histogram(
        self,
        y_true:    np.ndarray,
        y_prob:    np.ndarray,
        threshold: float = 0.5,
        fn_weight: float = 2.0,
    ) -> plt.Figure:
        """
        Two-panel confidence histogram and error breakdown.

        Parameters
        ----------
        fn_weight : clinical weighting of false negatives vs. false positives
        """
        spec = get_spec("figS5_confidence_histogram")
        with ieee_style():
            fig = plot_confidence_histogram(
                y_true    = y_true,
                y_prob    = y_prob,
                threshold = threshold,
                fn_weight = fn_weight,
                title     = "Prediction Confidence Distribution",
            )

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Fig S6 — Per-site Performance Heatmap
    # ------------------------------------------------------------------

    def fig_site_heatmap(
        self,
        y_true:   np.ndarray,
        y_prob:   np.ndarray,
        site_ids: np.ndarray,
        threshold: float = 0.5,
    ) -> plt.Figure:
        """
        Per-site performance heatmap (AUC, Sensitivity, Specificity, F1).
        """
        spec = get_spec("figS6_site_heatmap")
        with ieee_style():
            fig = plot_site_heatmap(
                y_true    = y_true,
                y_prob    = y_prob,
                site_ids  = site_ids,
                threshold = threshold,
                title     = "Per-site Classification Performance",
            )

        self._save(fig, spec)
        return fig

    # ------------------------------------------------------------------
    # Generate all figures from a results bundle
    # ------------------------------------------------------------------

    def generate_all(self, bundle: Dict[str, Any]) -> Dict[str, List[Path]]:
        """
        Generate every paper figure from a pre-built results bundle.

        The bundle is a plain dict; missing keys cause the corresponding
        figure to be skipped with a warning rather than raising an exception.

        Bundle keys
        -----------
        ``"models"``          — list of dicts for ROC/PR comparison
        ``"ablation_results"``— AblationResults object
        ``"cv_results"``      — list of per-fold dicts
        ``"mri_volume"``      — (D,H,W) numpy array
        ``"saliency"``        — (D,H,W) numpy array
        ``"gene_importances"``— (n_genes,) numpy array
        ``"gene_names"``      — list of str
        ``"features"``        — (N,d) numpy array
        ``"labels"``          — (N,) int array
        ``"site_ids"``        — (N,) int array (optional)
        ``"site_counts"``     — dict {site: count}
        ``"class_counts"``    — dict {label: count}
        ``"y_true"``          — (N,) int array for calibration
        ``"y_prob"``          — (N,) float array for calibration
        ``"brier_dict"``      — dict for Brier decomposition
        ``"tuner"``           — ASDTuner object
        ``"threshold"``       — float, decision threshold (default 0.5)

        Returns
        -------
        dict : figure_filename_stem → list of Path
        """
        results: Dict[str, List[Path]] = {}
        generated = []

        def _try(name, fn, *args, **kwargs):
            try:
                fig = fn(*args, **kwargs)
                plt.close(fig)
                generated.append(name)
            except Exception as e:
                logger.warning("Skipping figure '%s': %s", name, e)

        # Always generate architecture (no data needed)
        _try("architecture", self.fig_architecture_overview)

        if "site_counts" in bundle and "class_counts" in bundle:
            _try("dataset", self.fig_dataset_statistics,
                 bundle["site_counts"], bundle["class_counts"])

        if "models" in bundle:
            _try("roc_pr", self.fig_roc_pr, bundle["models"])

        if "ablation_results" in bundle:
            _try("ablation", self.fig_ablation, bundle["ablation_results"])

        if all(k in bundle for k in ("mri_volume", "saliency",
                                      "gene_importances", "gene_names")):
            _try("explainability", self.fig_explainability,
                 bundle["mri_volume"], bundle["saliency"],
                 bundle["gene_importances"], bundle["gene_names"])

        if "cv_results" in bundle:
            _try("cv_summary", self.fig_cv_summary, bundle["cv_results"])

        if "y_true" in bundle and "y_prob" in bundle:
            _try("calibration", self.fig_calibration,
                 bundle["y_true"], bundle["y_prob"],
                 brier_dict=bundle.get("brier_dict"))

        if "features" in bundle and "labels" in bundle:
            _try("embedding", self.fig_embedding,
                 bundle["features"], bundle["labels"],
                 site_ids=bundle.get("site_ids"))

        if "tuner" in bundle:
            _try("hpo", self.fig_hpo_summary, bundle["tuner"])

        # ---- Supplementary figures ----
        if "models" in bundle and "y_true" in bundle:
            _try("dca", self.fig_decision_curve,
                 bundle["y_true"], bundle["models"])

        if "models" in bundle and "y_true" in bundle:
            _try("lift_gain", self.fig_lift_gain,
                 bundle["y_true"], bundle["models"])

        if "models" in bundle:
            # Build per-model metric dicts from bundle["cv_report"] if present
            radar_models = _build_radar_models(bundle)
            if radar_models:
                _try("radar", self.fig_radar, radar_models)

        if "y_true" in bundle and "y_prob" in bundle:
            _try("confidence_histogram", self.fig_confidence_histogram,
                 bundle["y_true"], bundle["y_prob"],
                 threshold=bundle.get("threshold", 0.5))

        if ("y_true" in bundle and "y_prob" in bundle
                and "site_ids" in bundle):
            _try("site_heatmap", self.fig_site_heatmap,
                 bundle["y_true"], bundle["y_prob"],
                 bundle["site_ids"],
                 threshold=bundle.get("threshold", 0.5))

        # Build output dict from manifest
        for entry in self._manifest:
            fname = entry["spec"].filename
            results[fname] = entry["paths"]

        logger.info(
            "generate_all complete: %d/%d figures produced",
            len(generated), 14,
        )
        return results


# ---------------------------------------------------------------------------
# Helper — assemble radar-chart model dicts from available bundle data
# ---------------------------------------------------------------------------

def _build_radar_models(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the list of per-model metric dicts for the radar chart.

    Tries to use ``bundle["cv_report"]`` (dict of MetricCI objects) for the
    proposed model, then constructs synthetic baselines from ``bundle["models"]``
    for comparison.  Returns an empty list if insufficient data is available.
    """
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        matthews_corrcoef,
    )
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        cohen_kappa_score = None

    radar_models: List[Dict[str, Any]] = []
    y_true_base = bundle.get("y_true")
    if y_true_base is None:
        return []

    models_list = bundle.get("models", [])
    threshold   = bundle.get("threshold", 0.5)

    for idx, m in enumerate(models_list):
        y_prob = np.asarray(m.get("y_prob", []))
        if y_prob.size == 0:
            continue
        y_pred = (y_prob >= threshold).astype(int)
        try:
            tp = int(np.sum((y_pred == 1) & (y_true_base == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true_base == 0)))
            fp = int(np.sum((y_pred == 1) & (y_true_base == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true_base == 1)))
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            f1   = (2 * prec * sens / (prec + sens)
                    if (prec + sens) > 0 else 0.0)
            auc  = float(roc_auc_score(y_true_base, y_prob))
            mcc  = float(matthews_corrcoef(y_true_base, y_pred))
            kap  = (float(cohen_kappa_score(y_true_base, y_pred))
                    if cohen_kappa_score else 0.0)
        except Exception:
            continue

        radar_models.append({
            "name":    m.get("name", f"Model {idx + 1}"),
            "color":   m.get("color", PALETTE[idx % len(PALETTE)]),
            "metrics": {
                "sensitivity": sens,
                "specificity": spec,
                "ppv":         prec,
                "npv":         npv,
                "f1":          f1,
                "auc":         auc,
                "mcc":         mcc,
                "kappa":       kap,
            },
        })

    return radar_models
