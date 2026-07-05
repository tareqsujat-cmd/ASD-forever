"""
TuningAnalyzer — post-hoc analysis and IEEE-quality figures for HPO results.

All plots use the project's ``ieee_style()`` context manager so they are
immediately publication-ready.

Analyses provided
-----------------
  ``plot_optimization_history``   — best value at each trial (line + running best)
  ``plot_param_importance``       — fANOVA hyperparameter importance (bar chart)
  ``plot_parallel_coordinates``   — parallel coordinates for top-k trials
  ``plot_contour``                — 2-D contour interaction between two params
  ``plot_trial_duration``         — elapsed time per completed trial
  ``best_config_report``          — markdown / text summary of best config
  ``latex_table``                 — IEEE booktabs table of top-N configurations

All ``plot_*`` methods return a ``matplotlib.figure.Figure``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import optuna
from optuna.trial import FrozenTrial, TrialState

from hyperparameter_tuning.optuna_tuner import ASDTuner, TrialRecord
from visualization.style import ieee_style, COLORS, PALETTE, SINGLE_COL_W, DOUBLE_COL_W

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TuningAnalyzer
# ---------------------------------------------------------------------------

class TuningAnalyzer:
    """
    Post-hoc analyser for a completed ``ASDTuner`` run.

    Parameters
    ----------
    tuner : ASDTuner
        Must have ``tuner.study`` set (i.e. ``optimize()`` was called).
    primary_metric : str
        Metric key used for sorting / best detection (default ``"val_auc"``).
    """

    def __init__(
        self,
        tuner:          ASDTuner,
        primary_metric: str = "val_auc",
    ) -> None:
        if tuner.study is None:
            raise ValueError("tuner.study is None — call tuner.optimize() first")
        self.tuner          = tuner
        self.study          = tuner.study
        self.primary_metric = primary_metric
        self._records       = tuner.records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _complete_trials(self) -> List[FrozenTrial]:
        return [t for t in self.study.trials if t.state == TrialState.COMPLETE]

    def _objective_values(self) -> List[float]:
        return [
            t.value for t in self._complete_trials()
            if t.value is not None
        ]

    def _running_best(self, values: List[float], direction: str = "maximize") -> List[float]:
        best = []
        current = -math.inf if direction == "maximize" else math.inf
        for v in values:
            if direction == "maximize":
                current = max(current, v)
            else:
                current = min(current, v)
            best.append(current)
        return best

    def _record_for(self, trial: FrozenTrial) -> Optional[TrialRecord]:
        return next((r for r in self._records if r.trial_number == trial.number), None)

    # ------------------------------------------------------------------
    # Optimization history
    # ------------------------------------------------------------------

    def plot_optimization_history(
        self,
        ax:            Optional[plt.Axes] = None,
        direction:     str                = "maximize",
        show_all:      bool               = True,
        title:         Optional[str]      = None,
    ) -> plt.Figure:
        """
        Plot objective value per trial with a running best line.

        Parameters
        ----------
        direction : str
            ``"maximize"`` or ``"minimize"`` — controls running best direction.
        show_all : bool
            If True, scatter all trial values (grey); otherwise only best line.
        """
        complete = self._complete_trials()
        trial_nums = [t.number for t in complete if t.value is not None]
        values     = [t.value for t in complete if t.value is not None]
        running    = self._running_best(values, direction)

        with ieee_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.2))
            else:
                fig = ax.figure

            if show_all and values:
                ax.scatter(
                    trial_nums, values,
                    color="lightgrey", s=12, zorder=2, label="All trials",
                )
            if running:
                ax.plot(
                    trial_nums, running,
                    color=COLORS["model_a"], lw=1.5, zorder=3, label="Running best",
                )

            ax.set_xlabel("Trial number")
            ax.set_ylabel(self.primary_metric.replace("val_", ""))
            ax.set_title(title or "Optimization history")
            ax.legend(fontsize=6, loc="lower right")
            fig.tight_layout()

        return fig

    # ------------------------------------------------------------------
    # Hyperparameter importance
    # ------------------------------------------------------------------

    def plot_param_importance(
        self,
        n_top:     int               = 10,
        ax:        Optional[plt.Axes] = None,
        title:     Optional[str]      = None,
    ) -> plt.Figure:
        """
        Horizontal bar chart of hyperparameter importances via fANOVA.

        Requires scikit-learn (already a project dependency).
        """
        try:
            importance = optuna.importance.get_param_importances(self.study)
        except Exception as e:
            logger.warning("Could not compute param importances: %s — using uniform", e)
            params = set()
            for t in self._complete_trials():
                params.update(t.params.keys())
            importance = {p: 1.0 / max(len(params), 1) for p in sorted(params)}

        # Sort and truncate
        items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n_top]
        if not items:
            fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 1.5))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return fig

        names, vals = zip(*items)
        names = [n.replace("_", " ") for n in names]

        with ieee_style():
            height = max(1.5, len(names) * 0.28)
            if ax is None:
                fig, ax = plt.subplots(figsize=(SINGLE_COL_W, height))
            else:
                fig = ax.figure

            ypos = np.arange(len(names))
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
            ax.barh(ypos, vals, color=colors, alpha=0.85, height=0.7)
            ax.set_yticks(ypos)
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel("Importance (fANOVA)")
            ax.set_title(title or "Hyperparameter importance")
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
            fig.tight_layout()

        return fig

    # ------------------------------------------------------------------
    # Parallel coordinates
    # ------------------------------------------------------------------

    def plot_parallel_coordinates(
        self,
        top_k:      int               = 20,
        params:     Optional[List[str]] = None,
        ax:         Optional[plt.Axes] = None,
        title:      Optional[str]      = None,
    ) -> plt.Figure:
        """
        Parallel coordinates plot for the top-k trials by objective value.

        Only numeric hyperparameters are shown.

        Parameters
        ----------
        top_k : int
            Number of trials to include.
        params : list of str, optional
            Subset of parameter names to show; defaults to all numeric params.
        """
        complete = sorted(
            [t for t in self._complete_trials() if t.value is not None],
            key=lambda t: t.value,
            reverse=True,
        )[:top_k]

        if not complete:
            fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 2.0))
            ax.text(0.5, 0.5, "No completed trials", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        # Determine numeric parameters
        all_params: Dict[str, List[float]] = {}
        for t in complete:
            for k, v in t.params.items():
                if isinstance(v, (int, float)):
                    all_params.setdefault(k, []).append(float(v))

        if params is not None:
            all_params = {k: v for k, v in all_params.items() if k in params}

        param_names = list(all_params.keys())
        if not param_names:
            fig, ax = plt.subplots(figsize=(DOUBLE_COL_W, 2.0))
            ax.text(0.5, 0.5, "No numeric parameters", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        # Normalise each axis to [0, 1]
        norms: Dict[str, Tuple[float, float]] = {}
        for p, vals in all_params.items():
            lo, hi = min(vals), max(vals)
            norms[p] = (lo, hi if hi > lo else lo + 1e-9)

        # Objective values for colour mapping
        obj_vals = np.array([t.value for t in complete], dtype=float)
        vmin, vmax = obj_vals.min(), obj_vals.max()
        if vmax == vmin:
            vmax = vmin + 1.0
        cmap = plt.cm.viridis

        with ieee_style():
            n_axes = len(param_names)
            fig_w  = max(SINGLE_COL_W, min(DOUBLE_COL_W, 0.8 * n_axes))
            if ax is None:
                fig, ax = plt.subplots(figsize=(fig_w, 2.5))
            else:
                fig = ax.figure

            x_positions = np.linspace(0, 1, n_axes)

            for t in complete:
                y_vals = []
                for p in param_names:
                    v = t.params.get(p, None)
                    if v is None or not isinstance(v, (int, float)):
                        y_vals.append(float("nan"))
                    else:
                        lo, hi = norms[p]
                        y_vals.append((float(v) - lo) / (hi - lo))

                color = cmap((float(t.value) - vmin) / (vmax - vmin))
                ax.plot(x_positions, y_vals, color=color, alpha=0.6, lw=0.8)

            # Axis labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [p.replace("_", "\n") for p in param_names],
                fontsize=6,
            )
            ax.set_ylabel("Normalised value")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(title or f"Parallel coordinates (top {len(complete)} trials)")

            sm = plt.cm.ScalarMappable(cmap=cmap,
                                        norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=self.primary_metric.replace("val_", ""),
                         fraction=0.04, pad=0.02)
            fig.tight_layout()

        return fig

    # ------------------------------------------------------------------
    # Contour plot
    # ------------------------------------------------------------------

    def plot_contour(
        self,
        param_x:    str,
        param_y:    str,
        n_grid:     int               = 30,
        ax:         Optional[plt.Axes] = None,
        title:      Optional[str]      = None,
    ) -> plt.Figure:
        """
        2-D contour plot of objective value as a function of two parameters.

        Uses scipy's ``griddata`` for scattered interpolation.
        """
        complete = [t for t in self._complete_trials() if t.value is not None]
        xs = [t.params.get(param_x) for t in complete]
        ys = [t.params.get(param_y) for t in complete]
        zs = [t.value for t in complete]

        # Filter to numeric
        valid = [
            (x, y, z) for x, y, z in zip(xs, ys, zs)
            if isinstance(x, (int, float)) and isinstance(y, (int, float))
        ]

        with ieee_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=(SINGLE_COL_W, SINGLE_COL_W))
            else:
                fig = ax.figure

            if len(valid) < 4:
                ax.text(0.5, 0.5, "Insufficient data for contour",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title or f"Contour: {param_x} × {param_y}")
                fig.tight_layout()
                return fig

            from scipy.interpolate import griddata
            xv, yv, zv = map(np.array, zip(*valid))

            xi = np.linspace(xv.min(), xv.max(), n_grid)
            yi = np.linspace(yv.min(), yv.max(), n_grid)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((xv, yv), zv, (Xi, Yi), method="cubic")

            cs = ax.contourf(Xi, Yi, Zi, levels=14, cmap="viridis")
            ax.scatter(xv, yv, c=zv, cmap="viridis", edgecolors="white",
                       linewidths=0.4, s=20, zorder=3)
            fig.colorbar(cs, ax=ax, label=self.primary_metric.replace("val_", ""))
            ax.set_xlabel(param_x.replace("_", " "))
            ax.set_ylabel(param_y.replace("_", " "))
            ax.set_title(title or f"Contour: {param_x} × {param_y}")
            fig.tight_layout()

        return fig

    # ------------------------------------------------------------------
    # Trial duration
    # ------------------------------------------------------------------

    def plot_trial_duration(
        self,
        ax:    Optional[plt.Axes] = None,
        title: Optional[str]      = None,
    ) -> plt.Figure:
        """Bar chart of wall-clock time per completed trial."""
        complete_records = [r for r in self._records if r.state == "complete"]
        if not complete_records:
            fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 1.5))
            ax.text(0.5, 0.5, "No completed trials", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        trial_nums = [r.trial_number for r in complete_records]
        durations  = [r.elapsed_seconds for r in complete_records]

        with ieee_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 1.8))
            else:
                fig = ax.figure
            ax.bar(trial_nums, durations, color=COLORS["model_a"], alpha=0.8, width=0.7)
            ax.axhline(np.mean(durations), color="red", lw=1.0, ls="--",
                       label=f"mean={np.mean(durations):.1f}s")
            ax.set_xlabel("Trial number")
            ax.set_ylabel("Duration (s)")
            ax.set_title(title or "Trial wall-clock time")
            ax.legend(fontsize=6)
            fig.tight_layout()

        return fig

    # ------------------------------------------------------------------
    # Text reports
    # ------------------------------------------------------------------

    def best_config_report(self, n_best: int = 5) -> str:
        """
        Markdown report of the top-n trial configurations.

        Returns
        -------
        str — markdown text suitable for a notebook or README.
        """
        records = self.tuner.get_n_best(n_best, self.primary_metric)
        if not records:
            return "_No completed trials found._"

        lines = [
            f"## HPO Results — top {len(records)} trials",
            "",
            f"**Study:** `{self.study.study_name}`  ",
            f"**Search space:** `{self.tuner.search_space}`  ",
            f"**Total trials:** {len(self.study.trials)}  ",
            "",
        ]

        for rank, r in enumerate(records, 1):
            lines.append(f"### Rank {rank} — Trial {r.trial_number:04d}")
            lines.append("")
            obj = r.get_metric(self.primary_metric)
            lines.append(f"**{self.primary_metric}**: {obj:.5f}  ")
            lines.append("")
            lines.append("| Hyperparameter | Value |")
            lines.append("|---|---|")
            for k, v in sorted(r.overrides.items()):
                if isinstance(v, float):
                    v_str = f"{v:.6g}"
                else:
                    v_str = str(v)
                lines.append(f"| `{k}` | `{v_str}` |")
            lines.append("")

        return "\n".join(lines)

    def latex_table(
        self,
        n_best:    int          = 5,
        metrics:   Optional[List[str]] = None,
        caption:   str          = "Top hyperparameter configurations.",
        label:     str          = "tab:hpo",
    ) -> str:
        """
        IEEE-ready LaTeX ``tabular`` for the top-n trials.

        Shows trial number, objective value (bolded for best), and per-fold
        mean ± std for each metric in ``metrics``.
        """
        records = self.tuner.get_n_best(n_best, self.primary_metric)
        if not records:
            return "% No completed trials"

        if metrics is None:
            metrics = [self.primary_metric]

        _HEADERS = {
            "val_auc": "AUC",
            "val_acc": "Acc",
            "val_f1":  "F1",
            "val_sensitivity": "Sn",
            "val_specificity": "Sp",
            "val_brier": "Brier",
        }
        col_heads = [_HEADERS.get(m, m.replace("_", "\\_")) for m in metrics]
        col_fmt = "c" + "c" * len(metrics)

        # Best value per metric
        best_vals: Dict[str, float] = {}
        for m in metrics:
            vals = [r.get_metric(m) for r in records]
            valid = [v for v in vals if math.isfinite(v)]
            best_vals[m] = max(valid) if valid else float("nan")

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_fmt}}}",
            r"\toprule",
            "Trial & " + " & ".join(col_heads) + r" \\",
            r"\midrule",
        ]

        for r in records:
            cells = []
            for m in metrics:
                mean = r.get_metric(m)
                fold_vals = r.fold_metrics
                if fold_vals:
                    raw = [f.get(m, float("nan")) for f in fold_vals]
                    valid_raw = [v for v in raw if math.isfinite(v)]
                    std = float(np.std(valid_raw, ddof=1)) if len(valid_raw) > 1 else 0.0
                else:
                    std = 0.0
                if math.isnan(std) or std < 1e-9:
                    cell = f"{mean:.4f}"
                else:
                    cell = f"{mean:.4f} $\\pm$ {std:.4f}"
                if math.isfinite(mean) and math.isfinite(best_vals.get(m, float("nan"))):
                    if abs(mean - best_vals[m]) < 1e-6:
                        cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            lines.append(
                f"{r.trial_number:04d} & " + " & ".join(cells) + r" \\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Importance dict
    # ------------------------------------------------------------------

    def param_importance_dict(self) -> Dict[str, float]:
        """
        Return hyperparameter importances as a plain dict.

        Keys are hyperparameter names; values are importance scores summing to 1.
        """
        try:
            return dict(optuna.importance.get_param_importances(self.study))
        except Exception as e:
            logger.warning("param_importance_dict: %s — returning empty dict", e)
            return {}
