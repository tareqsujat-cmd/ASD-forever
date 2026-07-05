"""
Figure specifications for the IEEE ASD detection paper.

Each FigureSpec captures everything needed to generate the caption block and
\\includegraphics call in the LaTeX source.  The ``column`` field controls
the matplotlib figure width used during generation.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# FigureSpec dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FigureSpec:
    """Specification for a single paper figure."""

    label:        str                           # LaTeX \\label key, e.g. "fig:roc"
    caption:      str                           # Full IEEE caption text
    filename:     str                           # Base filename (no extension)
    column:       str  = "single"               # "single" | "double" | "half"
    panel_labels: List[Tuple[int, str]] = dataclasses.field(default_factory=list)
    # panel_labels: list of (subplot_linear_index, label_string)
    # e.g. [(0, "(a)"), (1, "(b)"), (2, "(c)")]
    notes:        Optional[str] = None          # Author notes (not in paper)

    @property
    def width_inches(self) -> float:
        from visualization.style import SINGLE_COL_W, DOUBLE_COL_W
        return {
            "single": SINGLE_COL_W,
            "double": DOUBLE_COL_W,
            "half":   SINGLE_COL_W / 2,
        }[self.column]

    def latex_figure_env(
        self,
        formats:   List[str] = ("pdf",),
        fig_path:  str        = "figures",
        placement: str        = "t",
    ) -> str:
        """
        Emit a LaTeX ``figure`` environment string for this spec.

        For single-column figures uses ``figure``; for double uses
        ``figure*``.
        """
        env   = "figure*" if self.column == "double" else "figure"
        width = r"\columnwidth" if self.column == "single" else r"\textwidth"
        fname = f"{fig_path}/{self.filename}.{formats[0]}"

        lines = [
            f"\\begin{{{env}}}[{placement}]",
            r"  \centering",
            f"  \\includegraphics[width={width}]{{{fname}}}",
            f"  \\caption{{{self.caption}}}",
            f"  \\label{{{self.label}}}",
            f"\\end{{{env}}}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry of all paper figures
# ---------------------------------------------------------------------------

FIGURE_SPECS: List[FigureSpec] = [
    FigureSpec(
        label    = "fig:architecture",
        caption  = (
            "Proposed multimodal ASD detection framework. "
            "The MRI branch encodes 3-D structural volumes via a 3-D backbone; "
            "the genetics branch encodes SNP profiles via a Transformer encoder. "
            "Features are fused by cross-attention before a linear classifier."
        ),
        filename = "fig01_architecture",
        column   = "double",
        notes    = "Generated programmatically — no external assets required.",
    ),
    FigureSpec(
        label    = "fig:dataset",
        caption  = (
            "ABIDE I/II dataset statistics. "
            "(a)~Number of subjects per acquisition site. "
            "(b)~Diagnosis distribution (ASD vs.~typically-developing controls)."
        ),
        filename = "fig02_dataset_statistics",
        column   = "single",
        panel_labels = [(0, "(a)"), (1, "(b)")],
    ),
    FigureSpec(
        label    = "fig:roc_pr",
        caption  = (
            "Classification performance on the held-out test folds. "
            "(a)~Receiver operating characteristic (ROC) curves with 95\\% "
            "bootstrap confidence bands. "
            "(b)~Precision--recall (PR) curves. "
            "Shaded regions show stratified bootstrap 95\\% CIs ($B=1000$)."
        ),
        filename = "fig03_roc_pr",
        column   = "double",
        panel_labels = [(0, "(a)"), (1, "(b)")],
    ),
    FigureSpec(
        label    = "fig:ablation",
        caption  = (
            "OFAT ablation study results (AUC $\\pm$ std across five folds). "
            "Asterisk (*) denotes significant improvement over the baseline "
            "(Wilcoxon signed-rank, $\\alpha=0.05$); "
            "dagger (\\dag) denotes significant degradation."
        ),
        filename = "fig04_ablation",
        column   = "single",
    ),
    FigureSpec(
        label    = "fig:explainability",
        caption  = (
            "Explainability analysis for a representative ASD subject. "
            "(a)~GradCAM++ saliency overlaid on axial, coronal, and sagittal "
            "MRI slices. "
            "(b)~Top-20 discriminative SNP features by integrated-gradient "
            "importance. "
            "Red bars indicate pro-ASD evidence; blue bars indicate "
            "pro-control evidence."
        ),
        filename = "fig05_explainability",
        column   = "double",
        panel_labels = [(0, "(a)"), (1, "(b)")],
    ),
    FigureSpec(
        label    = "fig:cv_summary",
        caption  = (
            "Per-fold cross-validation results for the proposed model and "
            "ablation baselines. "
            "Box plots show the distribution over the five site-stratified "
            "folds; diamond markers indicate the mean."
        ),
        filename = "fig06_cv_summary",
        column   = "single",
    ),
    FigureSpec(
        label    = "fig:calibration",
        caption  = (
            "Model calibration analysis. "
            "(a)~Reliability diagram; the diagonal dashed line represents "
            "perfect calibration. "
            "(b)~Brier score decomposition into uncertainty, calibration "
            "loss, and resolution components."
        ),
        filename = "fig07_calibration",
        column   = "single",
        panel_labels = [(0, "(a)"), (1, "(b)")],
    ),
    FigureSpec(
        label    = "fig:embedding",
        caption  = (
            "t-SNE visualisation of the 256-dimensional fused representations "
            "produced by the proposed model on the test set. "
            "Point colour encodes diagnosis (ASD in red, TC in blue); "
            "marker shape encodes acquisition site."
        ),
        filename = "fig08_embedding",
        column   = "single",
    ),
    FigureSpec(
        label    = "fig:hpo",
        caption  = (
            "Bayesian hyperparameter optimisation with TPE sampler. "
            "(a)~Objective (val-AUC) per trial and running best. "
            "(b)~fANOVA hyperparameter importance scores."
        ),
        filename = "figS1_hpo",
        column   = "double",
        panel_labels = [(0, "(a)"), (1, "(b)")],
        notes    = "Supplementary figure.",
    ),
    FigureSpec(
        label    = "fig:dca",
        caption  = (
            "Decision curve analysis. "
            "Net benefit as a function of threshold probability for the proposed "
            "multimodal model and the MRI-only baseline. "
            "The model is clinically useful where its net benefit exceeds both "
            "the ``treat all'' and ``treat none'' strategies."
        ),
        filename = "figS2_decision_curve",
        column   = "single",
        notes    = "Supplementary figure — clinical utility.",
    ),
    FigureSpec(
        label    = "fig:lift_gain",
        caption  = (
            "Cumulative gain and lift curves for the proposed model. "
            "(a)~Gain curve: fraction of ASD cases captured as a function of "
            "the fraction of the population screened. "
            "(b)~Lift curve: gain relative to a random screening strategy. "
            "The dashed diagonal in (a) and the horizontal line at lift=1 in (b) "
            "represent random performance."
        ),
        filename = "figS3_lift_gain",
        column   = "double",
        panel_labels = [(0, "(a)"), (1, "(b)")],
        notes    = "Supplementary figure — screening utility.",
    ),
    FigureSpec(
        label    = "fig:radar",
        caption  = (
            "Metric radar chart comparing the proposed multimodal model with "
            "single-modality baselines across eight clinical performance axes: "
            "sensitivity, specificity, precision (PPV), NPV, F1, AUROC, MCC, "
            "and Cohen's \\kappa. "
            "MCC and \\kappa are linearly normalised from $[-1,1]$ to $[0,1]$."
        ),
        filename = "figS4_radar_chart",
        column   = "single",
        notes    = "Supplementary figure — holistic metric comparison.",
    ),
    FigureSpec(
        label    = "fig:confidence",
        caption  = (
            "Prediction confidence analysis. "
            "(a)~Normalised histograms of predicted ASD probability for correct "
            "and incorrect predictions; the vertical dashed line marks the "
            "decision threshold. "
            "(b)~Breakdown of prediction outcomes (TP/TN/FP/FN) with the "
            "weighted clinical cost annotation (false negatives penalised "
            "$2\\times$ relative to false positives)."
        ),
        filename = "figS5_confidence_histogram",
        column   = "double",
        panel_labels = [(0, "(a)"), (1, "(b)")],
        notes    = "Supplementary figure — confidence calibration.",
    ),
    FigureSpec(
        label    = "fig:site_heatmap",
        caption  = (
            "Per-acquisition-site classification performance heatmap. "
            "Rows correspond to ABIDE sites; columns to AUC, sensitivity, "
            "specificity, and F1. "
            "The final row (``All sites'') reports the pooled metrics. "
            "Sites with fewer than three subjects per class are greyed out."
        ),
        filename = "figS6_site_heatmap",
        column   = "single",
        notes    = "Supplementary figure — cross-site generalisation.",
    ),
]

# Convenience dict: filename stem → FigureSpec
SPEC_BY_FILENAME = {s.filename: s for s in FIGURE_SPECS}
SPEC_BY_LABEL    = {s.label:    s for s in FIGURE_SPECS}


def get_spec(key: str) -> FigureSpec:
    """Look up a FigureSpec by filename stem or LaTeX label."""
    if key in SPEC_BY_FILENAME:
        return SPEC_BY_FILENAME[key]
    if key in SPEC_BY_LABEL:
        return SPEC_BY_LABEL[key]
    raise KeyError(f"No FigureSpec found for '{key}'")
