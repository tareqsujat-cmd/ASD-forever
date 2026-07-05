"""
ASDEvaluator — orchestrates the full evaluation pipeline.

Usage
-----
    evaluator = ASDEvaluator(cfg)

    # Single test set evaluation
    report = evaluator.evaluate(y_true, y_prob, site_ids=site_ids)
    print(report.summary())

    # K-fold CV aggregation
    cv_report = evaluator.evaluate_cv(fold_results)

    # Two-model statistical comparison
    comparison = evaluator.compare_models(y_true, y_prob_a, y_prob_b)

Output
------
EvaluationReport dataclass with:
  - All metrics with 95% bootstrap CI
  - Calibration analysis
  - Per-site metric breakdown
  - Formatted summary string for paper Tables
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MetricCI:
    """A single metric value with confidence interval."""
    value: float
    ci_lower: float
    ci_upper: float
    std: float = float("nan")

    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class EvaluationReport:
    """Complete evaluation results for one model on one test partition."""

    # Core discriminative metrics
    auroc:             MetricCI = None
    auprc:             MetricCI = None
    accuracy:          MetricCI = None
    balanced_accuracy: MetricCI = None
    sensitivity:       MetricCI = None
    specificity:       MetricCI = None
    ppv:               MetricCI = None
    npv:               MetricCI = None
    f1:                MetricCI = None
    mcc:               MetricCI = None
    brier_score:       MetricCI = None

    # Calibration
    ece:               float = float("nan")
    mce:               float = float("nan")

    # Decision boundary
    threshold:         float = 0.5
    threshold_method:  str   = "youden_j"

    # Confusion matrix (at optimal threshold)
    tp: int = 0; tn: int = 0; fp: int = 0; fn: int = 0
    n_pos: int = 0; n_neg: int = 0

    # Per-site breakdown
    per_site: Dict[str, dict] = dataclasses.field(default_factory=dict)

    # Raw arrays (kept for downstream plotting)
    y_true: Optional[np.ndarray] = dataclasses.field(default=None, repr=False)
    y_prob: Optional[np.ndarray] = dataclasses.field(default=None, repr=False)

    def summary(self) -> str:
        """Formatted summary for paper tables."""
        lines = [
            "=" * 55,
            "ASD Evaluation Report",
            "=" * 55,
            f"  N={self.n_pos + self.n_neg} (ASD={self.n_pos}, TC={self.n_neg})",
            f"  Threshold: {self.threshold:.3f} ({self.threshold_method})",
            "-" * 55,
        ]
        for name in ["auroc", "auprc", "balanced_accuracy", "sensitivity",
                     "specificity", "f1", "mcc", "accuracy", "ppv", "npv",
                     "brier_score"]:
            m = getattr(self, name)
            if m is not None:
                lines.append(f"  {name:<22}: {m}")
        lines += [
            "-" * 55,
            f"  ECE: {self.ece:.4f}    MCE: {self.mce:.4f}",
        ]
        if self.per_site:
            lines.append("-" * 55)
            lines.append(f"  Per-site AUC ({len(self.per_site)} sites):")
            for site, m in sorted(self.per_site.items()):
                auc = m.get("auroc", float("nan"))
                n = m.get("n", "?")
                lines.append(f"    {site:<20} AUC={auc:.4f}  N={n}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, MetricCI):
                d[f.name] = v.to_dict()
            elif isinstance(v, np.ndarray):
                d[f.name] = v.tolist()
            else:
                d[f.name] = v
        return d

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        logger.info("Evaluation report saved to %s", path)


# ---------------------------------------------------------------------------
# ASDEvaluator
# ---------------------------------------------------------------------------

class ASDEvaluator:
    """
    Orchestrates full evaluation: metrics, bootstrap CI, calibration,
    per-site analysis, and statistical significance tests.

    Parameters
    ----------
    cfg : Config
        Loaded configuration.  Only used for reproducibility seed.
    threshold_method : str
        "youden_j" | "f1"
    n_bootstrap : int
        Number of bootstrap resamples for CI computation.
    alpha : float
        Significance level for CIs (0.05 → 95% CI).
    """

    def __init__(
        self,
        cfg=None,
        threshold_method: str = "youden_j",
        n_bootstrap: int = 2000,
        alpha: float = 0.05,
    ) -> None:
        self.threshold_method = threshold_method
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        seed = getattr(cfg, "random_seed", 42) if cfg is not None else 42
        self.seed = seed

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        site_ids: Optional[np.ndarray] = None,
    ) -> EvaluationReport:
        """
        Full evaluation of a single model on a test set.

        Parameters
        ----------
        y_true    : (N,) binary labels, 0=TC, 1=ASD
        y_prob    : (N,) predicted probability of ASD
        site_ids  : (N,) optional site identifiers for per-site analysis

        Returns
        -------
        EvaluationReport
        """
        from evaluation.metrics import (
            compute_all_metrics, optimal_threshold_youden, optimal_threshold_f1
        )
        from evaluation.bootstrap import bootstrap_all_metrics
        from evaluation.calibration import compute_calibration

        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Optimal threshold
        if self.threshold_method == "youden_j":
            threshold = optimal_threshold_youden(y_true, y_prob)
        else:
            threshold = optimal_threshold_f1(y_true, y_prob)

        # Point estimates + bootstrap CI
        logger.info(
            "Computing bootstrap CI (%d resamples)…", self.n_bootstrap
        )
        boot_results = bootstrap_all_metrics(
            y_true, y_prob,
            threshold=threshold,
            n_bootstrap=self.n_bootstrap,
            alpha=self.alpha,
            seed=self.seed,
        )

        # Base metrics for confusion matrix values
        base = compute_all_metrics(y_true, y_prob, threshold=threshold)

        # Calibration
        calib = compute_calibration(y_true, y_prob)

        # Per-site metrics
        per_site = {}
        if site_ids is not None:
            per_site = self._per_site_metrics(y_true, y_prob, site_ids, threshold)

        # Build report
        def _mci(name: str) -> MetricCI:
            r = boot_results.get(name, {})
            return MetricCI(
                value=r.get("value", float("nan")),
                ci_lower=r.get("ci_lower", float("nan")),
                ci_upper=r.get("ci_upper", float("nan")),
            )

        report = EvaluationReport(
            auroc=_mci("auroc"), auprc=_mci("auprc"),
            accuracy=_mci("accuracy"),
            balanced_accuracy=_mci("balanced_accuracy"),
            sensitivity=_mci("sensitivity"), specificity=_mci("specificity"),
            ppv=_mci("ppv"), npv=_mci("npv"),
            f1=_mci("f1"), mcc=_mci("mcc"),
            brier_score=_mci("brier_score"),
            ece=calib["ece"], mce=calib["mce"],
            threshold=threshold, threshold_method=self.threshold_method,
            tp=base["tp"], tn=base["tn"], fp=base["fp"], fn=base["fn"],
            n_pos=base["n_pos"], n_neg=base["n_neg"],
            per_site=per_site,
            y_true=y_true, y_prob=y_prob,
        )

        logger.info(
            "Evaluation: AUC=%.4f [%.4f, %.4f]  F1=%.4f  SEN=%.4f  SPEC=%.4f",
            report.auroc.value, report.auroc.ci_lower, report.auroc.ci_upper,
            report.f1.value, report.sensitivity.value, report.specificity.value,
        )
        return report

    def evaluate_cv(
        self, fold_metrics: List[dict]
    ) -> Dict:
        """
        Aggregate per-fold metric dicts (from ASDTrainer) with t-distribution CIs.

        Parameters
        ----------
        fold_metrics : list of dicts, one per fold (from trainer)

        Returns
        -------
        dict: metric → MetricCI (aggregated across folds)
        """
        from evaluation.bootstrap import aggregate_cv_metrics

        aggregated = aggregate_cv_metrics(fold_metrics, alpha=self.alpha)
        result = {}
        for name, r in aggregated.items():
            result[name] = MetricCI(
                value=r["value"],
                ci_lower=r["ci_lower"],
                ci_upper=r["ci_upper"],
                std=r.get("std", float("nan")),
            )
        return result

    def compare_models(
        self,
        y_true: np.ndarray,
        y_prob_a: np.ndarray,
        y_prob_b: np.ndarray,
        y_pred_a: Optional[np.ndarray] = None,
        y_pred_b: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Full statistical comparison between two models on the same test set.

        Returns
        -------
        dict containing DeLong test and McNemar test results.
        """
        from evaluation.statistical_tests import delong_test, mcnemar_test
        from evaluation.metrics import optimal_threshold_youden

        y_true = np.asarray(y_true)
        y_prob_a = np.asarray(y_prob_a)
        y_prob_b = np.asarray(y_prob_b)

        delong = delong_test(y_true, y_prob_a, y_prob_b)

        if y_pred_a is None:
            thr_a = optimal_threshold_youden(y_true, y_prob_a)
            y_pred_a = (y_prob_a >= thr_a).astype(int)
        if y_pred_b is None:
            thr_b = optimal_threshold_youden(y_true, y_prob_b)
            y_pred_b = (y_prob_b >= thr_b).astype(int)

        mcn = mcnemar_test(y_true, y_pred_a, y_pred_b)

        return {"delong": delong, "mcnemar": mcn}

    # -----------------------------------------------------------------------
    # Per-site analysis
    # -----------------------------------------------------------------------

    def _per_site_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        site_ids: np.ndarray,
        threshold: float,
    ) -> Dict[str, dict]:
        from evaluation.metrics import auroc, sensitivity, specificity, accuracy

        per_site = {}
        for site in np.unique(site_ids):
            mask = site_ids == site
            yt = y_true[mask]
            yp = y_prob[mask]
            yp_bin = (yp >= threshold).astype(int)

            site_dict = {
                "n": int(mask.sum()),
                "n_pos": int(yt.sum()),
                "n_neg": int((1 - yt).sum()),
                "accuracy": accuracy(yt, yp_bin),
                "sensitivity": sensitivity(yt, yp_bin),
                "specificity": specificity(yt, yp_bin),
            }
            if len(np.unique(yt)) == 2:
                site_dict["auroc"] = auroc(yt, yp)
            else:
                site_dict["auroc"] = float("nan")

            per_site[str(site)] = site_dict

        return per_site
