from visualization.style import ieee_style, apply_ieee_style, COLORS, PALETTE
from visualization.roc_pr_curves import (
    plot_roc_curve, plot_pr_curve, plot_roc_comparison, plot_pr_comparison
)
from visualization.confusion_matrix import plot_confusion_matrix, cm_from_report
from visualization.calibration_plot import (
    plot_reliability_diagram, plot_brier_decomposition
)
from visualization.embedding_viz import compute_embedding, plot_embedding
from visualization.mri_saliency import plot_mri_slice, plot_mri_triplet, plot_volume_grid
from visualization.genetics_importance import (
    plot_gene_importance, plot_attention_heatmap, plot_multihead_attention
)
from visualization.figure_factory import FigureFactory

__all__ = [
    "ieee_style", "apply_ieee_style", "COLORS", "PALETTE",
    "plot_roc_curve", "plot_pr_curve", "plot_roc_comparison", "plot_pr_comparison",
    "plot_confusion_matrix", "cm_from_report",
    "plot_reliability_diagram", "plot_brier_decomposition",
    "compute_embedding", "plot_embedding",
    "plot_mri_slice", "plot_mri_triplet", "plot_volume_grid",
    "plot_gene_importance", "plot_attention_heatmap", "plot_multihead_attention",
    "FigureFactory",
]
