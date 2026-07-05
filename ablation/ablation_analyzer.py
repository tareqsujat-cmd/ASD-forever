"""
AblationAnalyzer — statistical analysis and table generation for ablation results.

After an ``AblationRunner`` produces an ``AblationResults`` object, this
analyzer provides:

  • Variant ranking by any metric
  • Pairwise significance testing (Wilcoxon signed-rank) vs a baseline
  • Significance matrix for all pairs
  • LaTeX tabular for IEEE papers (best values bolded, sig-markers ✓/✗)
  • Markdown table for reports
  • Bar chart with error bars (via visualization module)

Wilcoxon tests are performed on per-fold metric vectors (K=5 or K=10
folds).  Since K is small, the test is non-parametric and appropriate.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ablation.ablation_results import AblationResults, VariantResult

logger = logging.getLogger(__name__)

# Default metrics shown in summary tables
DEFAULT_METRICS = ["val_auc", "val_acc", "val_f1"]

# LaTeX column headers matching default metrics
DEFAULT_HEADERS = {
    "val_auc": "AUC",
    "val_acc": "Accuracy",
    "val_f1":  "F1",
    "val_sensitivity": "Sn",
    "val_specificity": "Sp",
    "val_mcc": "MCC",
    "val_brier": "Brier",
}


# ---------------------------------------------------------------------------
# AblationAnalyzer
# ---------------------------------------------------------------------------

class AblationAnalyzer:
    """
    Analyses and formats the results of an ablation study.

    Parameters
    ----------
    results : AblationResults
    baseline_name : str, optional
        Which variant is the baseline for pairwise comparisons.
        Defaults to "baseline" if present, else the first variant.
    """

    def __init__(
        self,
        results: AblationResults,
        baseline_name: Optional[str] = None,
    ) -> None:
        self.results = results
        if baseline_name is None:
            if "baseline" in results:
                baseline_name = "baseline"
            else:
                all_v = results.all_variants()
                baseline_name = all_v[0] if all_v else ""
        self.baseline_name = baseline_name

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_variants(
        self,
        metric: str = "val_auc",
        ascending: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        Sort all variants by their mean value for ``metric``.

        Returns
        -------
        list of (variant_name, mean_value) sorted best-first.
        """
        pairs = [
            (name, r.get_metric(metric))
            for name, r in zip(
                self.results.all_variants(), self.results.all_results()
            )
        ]
        return sorted(pairs, key=lambda x: x[1], reverse=not ascending)

    def best_variant(self, metric: str = "val_auc") -> Tuple[str, float]:
        """Return (name, value) of the highest-scoring variant."""
        ranked = self.rank_variants(metric, ascending=False)
        return ranked[0] if ranked else ("", float("nan"))

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def compare_to_baseline(
        self,
        metric: str = "val_auc",
        alpha: float = 0.05,
    ) -> Dict[str, dict]:
        """
        Wilcoxon signed-rank test for each variant vs the baseline.

        Returns
        -------
        dict: variant_name → {
            "mean_diff": float,        — variant_mean - baseline_mean
            "p_value": float,
            "significant": bool,
            "direction": "better" | "worse" | "tie"
        }
        """
        from evaluation.statistical_tests import wilcoxon_cv_test

        baseline = self.results.get(self.baseline_name)
        if baseline is None:
            logger.warning("Baseline '%s' not in results", self.baseline_name)
            return {}

        base_vals = baseline.get_fold_values(metric)
        if not base_vals:
            logger.warning("Baseline has no values for metric '%s'", metric)
            return {}

        comparisons: Dict[str, dict] = {}
        for name in self.results.all_variants():
            if name == self.baseline_name:
                continue
            r = self.results.get(name)
            if r is None:
                continue
            var_vals = r.get_fold_values(metric)
            if not var_vals or len(var_vals) != len(base_vals):
                comparisons[name] = {
                    "mean_diff": float("nan"),
                    "p_value": float("nan"),
                    "significant": False,
                    "direction": "unknown",
                }
                continue

            try:
                wx = wilcoxon_cv_test(var_vals, base_vals)
            except Exception as e:
                logger.debug("Wilcoxon failed for '%s': %s", name, e)
                wx = {"p_value": 1.0, "mean_diff": 0.0}

            mean_diff = float(np.mean(var_vals) - np.mean(base_vals))
            direction = (
                "better" if mean_diff > 0 and ascending_is_better(metric) == False
                else "worse" if mean_diff < 0 and ascending_is_better(metric) == False
                else "tie"
            )
            if abs(mean_diff) < 1e-8:
                direction = "tie"
            elif mean_diff > 0:
                direction = "better"
            else:
                direction = "worse"

            comparisons[name] = {
                "mean_diff": mean_diff,
                "p_value":   float(wx["p_value"]),
                "significant": float(wx["p_value"]) < alpha,
                "direction": direction,
            }

        return comparisons

    def significance_matrix(
        self,
        metric: str = "val_auc",
        alpha: float = 0.05,
    ) -> Dict[str, Dict[str, dict]]:
        """
        Pairwise Wilcoxon tests for all variant pairs.

        Returns
        -------
        dict: variant_a → {variant_b → {"p_value", "significant", "mean_diff"}}
        """
        from evaluation.statistical_tests import wilcoxon_cv_test

        variants = [
            (name, self.results.get(name).get_fold_values(metric))
            for name in self.results.all_variants()
        ]
        variants = [(n, v) for n, v in variants if v]

        matrix: Dict[str, Dict[str, dict]] = {}
        for i, (n_a, v_a) in enumerate(variants):
            matrix[n_a] = {}
            for j, (n_b, v_b) in enumerate(variants):
                if i == j:
                    matrix[n_a][n_b] = {"p_value": 1.0, "significant": False,
                                         "mean_diff": 0.0}
                    continue
                if len(v_a) != len(v_b):
                    matrix[n_a][n_b] = {"p_value": float("nan"),
                                         "significant": False,
                                         "mean_diff": float("nan")}
                    continue
                try:
                    wx = wilcoxon_cv_test(v_a, v_b)
                    matrix[n_a][n_b] = {
                        "p_value":    float(wx["p_value"]),
                        "significant": float(wx["p_value"]) < alpha,
                        "mean_diff":  float(np.mean(v_a) - np.mean(v_b)),
                    }
                except Exception as e:
                    logger.debug("Wilcoxon %s vs %s: %s", n_a, n_b, e)
                    matrix[n_a][n_b] = {"p_value": 1.0, "significant": False,
                                         "mean_diff": float(np.mean(v_a) - np.mean(v_b))}

        return matrix

    # ------------------------------------------------------------------
    # Summary dict
    # ------------------------------------------------------------------

    def summary(
        self,
        metrics: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> Dict[str, dict]:
        """
        Full summary dict: per-variant stats + comparison to baseline.

        Returns
        -------
        dict: variant_name → {
            "mean": {metric: value}, "std": {metric: value},
            "rank": int, "vs_baseline": {...}
        }
        """
        if metrics is None:
            metrics = [m for m in DEFAULT_METRICS
                       if m in self.results.available_metrics()]
            if not metrics:
                metrics = self.results.available_metrics()[:3]

        primary = metrics[0] if metrics else "val_auc"
        ranked = self.rank_variants(primary)
        rank_map = {name: i + 1 for i, (name, _) in enumerate(ranked)}

        cmp = {}
        if self.baseline_name in self.results:
            cmp = self.compare_to_baseline(primary, alpha=alpha)

        result = {}
        for name in self.results.all_variants():
            r = self.results.get(name)
            result[name] = {
                "mean":         {m: r.get_metric(m) for m in metrics},
                "std":          {m: r.std_metrics.get(m, float("nan")) for m in metrics},
                "rank":         rank_map.get(name, -1),
                "vs_baseline":  cmp.get(name, {}),
                "elapsed":      r.elapsed_seconds,
            }
        return result

    # ------------------------------------------------------------------
    # Table generation
    # ------------------------------------------------------------------

    def markdown_table(
        self,
        metrics: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> str:
        """
        Markdown table with mean ± std for each metric column.

        Baseline row marked with (baseline).
        Significant improvements over baseline marked with *.
        """
        if metrics is None:
            metrics = [m for m in DEFAULT_METRICS
                       if m in self.results.available_metrics()]
        if not metrics:
            metrics = self.results.available_metrics()

        primary = metrics[0] if metrics else None
        cmp = self.compare_to_baseline(primary or "", alpha=alpha) if primary else {}
        ranked = self.rank_variants(primary or "", ascending=False) if primary else []
        best_name = ranked[0][0] if ranked else None

        # Header
        col_heads = [DEFAULT_HEADERS.get(m, m) for m in metrics]
        header = "| Variant | " + " | ".join(col_heads) + " |"
        sep    = "|---|" + "---|" * len(metrics)
        rows = [header, sep]

        for name in self.results.all_variants():
            r = self.results.get(name)
            suffix = ""
            if name == self.baseline_name:
                suffix = " *(base)*"
            elif name in cmp and cmp[name].get("significant"):
                suffix = " *"
            cells = [r.summary_str(m) for m in metrics]
            if primary and name == best_name:
                cells[0] = f"**{cells[0]}**"
            rows.append(f"| {name}{suffix} | " + " | ".join(cells) + " |")

        return "\n".join(rows)

    def latex_table(
        self,
        metrics: Optional[List[str]] = None,
        caption: str = "Ablation study results.",
        label: str = "tab:ablation",
        alpha: float = 0.05,
    ) -> str:
        """
        IEEE-ready LaTeX ``tabular`` environment (requires booktabs).

        Best value in each metric column is typeset in ``\\textbf{}``.
        Variants with a significant improvement over baseline receive a
        dagger (†) marker in the Variant column.
        """
        if metrics is None:
            metrics = [m for m in DEFAULT_METRICS
                       if m in self.results.available_metrics()]
        if not metrics:
            metrics = self.results.available_metrics()

        primary = metrics[0] if metrics else None
        cmp = self.compare_to_baseline(primary or "", alpha=alpha) if primary else {}

        # Find best per metric
        best_vals: Dict[str, float] = {}
        for m in metrics:
            vals = [self.results.get(n).get_metric(m)
                    for n in self.results.all_variants()]
            valid = [v for v in vals if math.isfinite(v)]
            best_vals[m] = max(valid) if valid else float("nan")

        col_heads = [DEFAULT_HEADERS.get(m, m.replace("_", "\\_")) for m in metrics]
        n_cols = len(metrics) + 1
        col_fmt = "l" + "c" * len(metrics)

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_fmt}}}",
            r"\toprule",
            "Variant & " + " & ".join(col_heads) + r" \\",
            r"\midrule",
        ]

        for name in self.results.all_variants():
            r = self.results.get(name)
            is_base = (name == self.baseline_name)
            sig_better = (
                name in cmp
                and cmp[name].get("significant")
                and cmp[name].get("direction") == "better"
            )

            display_name = name.replace("_", r"\_")
            if is_base:
                display_name += r" \textsuperscript{base}"
            elif sig_better:
                display_name += r" \textsuperscript{\dag}"

            cells = []
            for m in metrics:
                mean = r.get_metric(m)
                std  = r.std_metrics.get(m, float("nan"))
                cell = (
                    f"{mean:.4f}" if math.isnan(std) or std < 1e-9
                    else f"{mean:.4f} $\\pm$ {std:.4f}"
                )
                if math.isfinite(mean) and math.isfinite(best_vals.get(m, float("nan"))):
                    if abs(mean - best_vals[m]) < 1e-6:
                        cell = f"\\textbf{{{cell}}}"
                cells.append(cell)

            lines.append(display_name + " & " + " & ".join(cells) + r" \\")

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_comparison(
        self,
        metric: str = "val_auc",
        alpha: float = 0.05,
        ax=None,
        title: Optional[str] = None,
    ):
        """
        Horizontal bar chart of mean ± std for all variants.

        Baseline bar is coloured differently; variants significantly better
        than baseline are marked with *.

        Returns matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        from visualization.style import ieee_style, COLORS, SINGLE_COL_W

        ranked = self.rank_variants(metric, ascending=False)
        cmp = self.compare_to_baseline(metric, alpha=alpha)

        names  = [n for n, _ in ranked]
        means  = [self.results.get(n).get_metric(metric) for n in names]
        stds   = [self.results.get(n).std_metrics.get(metric, 0.0) for n in names]
        colors = []
        for n in names:
            if n == self.baseline_name:
                colors.append(COLORS["random"])
            elif n in cmp and cmp[n].get("significant") and cmp[n].get("direction") == "better":
                colors.append(COLORS["model_c"])
            else:
                colors.append(COLORS["model_a"])

        with ieee_style():
            height = max(2.0, len(names) * 0.32)
            if ax is None:
                fig, ax = plt.subplots(figsize=(SINGLE_COL_W, height))
            else:
                fig = ax.figure

            ypos = np.arange(len(names))
            ax.barh(ypos, means, xerr=stds, color=colors, alpha=0.85,
                    height=0.7, capsize=3)

            # Significance markers
            for i, n in enumerate(names):
                if n in cmp and cmp[n].get("significant"):
                    marker = "*" if cmp[n]["direction"] == "better" else "†"
                    ax.text(means[i] + stds[i] + 0.002, i,
                            marker, va="center", fontsize=9, color="black")

            ax.set_yticks(ypos)
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel(DEFAULT_HEADERS.get(metric, metric))
            if title is None:
                title = f"Ablation: {DEFAULT_HEADERS.get(metric, metric)}"
            ax.set_title(title)

            from matplotlib.patches import Patch
            legend_patches = [
                Patch(color=COLORS["model_c"], label="Sig. better than baseline"),
                Patch(color=COLORS["model_a"], label="Not significant"),
                Patch(color=COLORS["random"],  label="Baseline"),
            ]
            ax.legend(handles=legend_patches, fontsize=6, loc="lower right")
            fig.tight_layout()

        return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ascending_is_better(metric: str) -> bool:
    """Return False for metrics where higher = better (AUC, Acc, F1, Sn, Sp)."""
    lower_better = {"brier", "loss", "error", "ece", "mce"}
    return any(kw in metric.lower() for kw in lower_better)
