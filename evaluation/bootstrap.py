"""
Bootstrap confidence intervals for evaluation metrics.

Method: percentile bootstrap (Efron & Tibshirani, 1993).
  1. Resample (y_true, y_prob) with replacement N_BOOTSTRAP times.
  2. Compute metric on each resample.
  3. Return [α/2, 1−α/2] percentile quantiles.

Why percentile bootstrap (not BCa)
------------------------------------
BCa (bias-corrected and accelerated) has better coverage for skewed
distributions but requires a jackknife estimate (O(N²) evaluations).
For AUC with N=500 and N_BOOTSTRAP=2000, percentile bootstrap gives
adequate coverage (≥95%) per simulation studies in medical AI.

Stratified resampling
----------------------
We resample separately within the positive (ASD) and negative (TC) groups,
then combine.  This preserves the class ratio in each bootstrap sample,
preventing degenerate all-zero or all-one label samples that would make
AUC undefined.

Reference
---------
Efron B, Tibshirani RJ. (1993). An Introduction to the Bootstrap.
  Chapman & Hall.
Carpenter J, Bithell J. (2000). Bootstrap confidence intervals: when,
  which, what? Statistics in Medicine.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Core bootstrap function
# ---------------------------------------------------------------------------

def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
    stratified: bool = True,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap CI for a single metric.

    Parameters
    ----------
    y_true : (N,) integer labels
    y_prob : (N,) predicted probabilities
    metric_fn : callable(y_true, y_prob) → float
    n_bootstrap : int
    alpha : float  — CI level: 1-alpha (default 0.95)
    seed : int
    stratified : bool
        If True, resample within each class to preserve class balance.

    Returns
    -------
    (point_estimate, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    point_estimate = metric_fn(y_true, y_prob)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    if stratified and len(pos_idx) > 0 and len(neg_idx) > 0:
        boot_values = _stratified_bootstrap(
            y_true, y_prob, metric_fn, pos_idx, neg_idx, n_bootstrap, rng
        )
    else:
        boot_values = _simple_bootstrap(
            y_true, y_prob, metric_fn, n_bootstrap, rng
        )

    # Drop NaN values (degenerate samples)
    boot_values = boot_values[~np.isnan(boot_values)]
    if len(boot_values) == 0:
        return point_estimate, float("nan"), float("nan")

    lower = float(np.percentile(boot_values, 100 * alpha / 2))
    upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))

    return point_estimate, lower, upper


def _stratified_bootstrap(
    y_true, y_prob, metric_fn, pos_idx, neg_idx, n_bootstrap, rng
):
    boot_values = np.empty(n_bootstrap)
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    for i in range(n_bootstrap):
        b_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        b_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        b_idx = np.concatenate([b_pos, b_neg])
        try:
            boot_values[i] = metric_fn(y_true[b_idx], y_prob[b_idx])
        except Exception:
            boot_values[i] = float("nan")
    return boot_values


def _simple_bootstrap(y_true, y_prob, metric_fn, n_bootstrap, rng):
    N = len(y_true)
    boot_values = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        b_idx = rng.choice(N, size=N, replace=True)
        try:
            boot_values[i] = metric_fn(y_true[b_idx], y_prob[b_idx])
        except Exception:
            boot_values[i] = float("nan")
    return boot_values


# ---------------------------------------------------------------------------
# Bootstrap CIs for the full metric suite
# ---------------------------------------------------------------------------

def bootstrap_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap CIs for all standard metrics.

    Parameters
    ----------
    y_true : (N,) labels
    y_prob : (N,) probabilities
    threshold : float, optional — fixed decision threshold
    n_bootstrap : int
    alpha : float — 1-alpha confidence level
    seed : int

    Returns
    -------
    dict: metric_name → {"value": float, "ci_lower": float, "ci_upper": float}
    """
    from evaluation.metrics import (
        auroc, auprc, brier_score,
        accuracy, balanced_accuracy, sensitivity, specificity,
        ppv, npv, f1, mcc,
        optimal_threshold_youden,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if threshold is None:
        threshold = optimal_threshold_youden(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics that depend only on y_prob (threshold-independent)
    prob_metrics = {
        "auroc":       auroc,
        "auprc":       auprc,
        "brier_score": brier_score,
    }

    # Metrics that require thresholded predictions
    def _pred_metric(fn):
        return lambda yt, yp: fn(yt, (yp >= threshold).astype(int))

    pred_metrics = {
        "accuracy":          _pred_metric(accuracy),
        "balanced_accuracy": _pred_metric(balanced_accuracy),
        "sensitivity":       _pred_metric(sensitivity),
        "specificity":       _pred_metric(specificity),
        "ppv":               _pred_metric(ppv),
        "npv":               _pred_metric(npv),
        "f1":                _pred_metric(f1),
        "mcc":               _pred_metric(mcc),
    }

    results = {}
    all_metrics = {**prob_metrics, **pred_metrics}

    for name, fn in all_metrics.items():
        val, lo, hi = bootstrap_metric(
            y_true, y_prob, fn,
            n_bootstrap=n_bootstrap, alpha=alpha,
            seed=seed, stratified=True,
        )
        results[name] = {"value": val, "ci_lower": lo, "ci_upper": hi}

    results["threshold"] = {"value": threshold, "ci_lower": threshold, "ci_upper": threshold}
    return results


# ---------------------------------------------------------------------------
# Cross-fold aggregation
# ---------------------------------------------------------------------------

def aggregate_cv_metrics(
    fold_metrics: list,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-fold metric dicts using mean ± standard deviation.

    For K-fold CV, the standard approach in medical AI is to report the
    mean and SD across folds (not bootstrap CI), since each fold is already
    an independent sample.

    Parameters
    ----------
    fold_metrics : list of dicts from compute_all_metrics()
    alpha : float — used to compute approximate CI from SD

    Returns
    -------
    dict: metric_name → {"value": mean, "ci_lower": mean-k*sd, "ci_upper": mean+k*sd}
    where k ≈ 2 for 95% CI with K≥5 folds (t-distribution approximation).
    """
    from scipy import stats as scipy_stats

    if not fold_metrics:
        return {}

    # t-critical value for 1-alpha/2 with K-1 degrees of freedom
    K = len(fold_metrics)
    t_crit = float(scipy_stats.t.ppf(1 - alpha / 2, df=K - 1))

    scalar_keys = [
        k for k, v in fold_metrics[0].items()
        if isinstance(v, (int, float)) and k not in ("tp", "tn", "fp", "fn", "n_pos", "n_neg")
    ]

    result = {}
    for key in scalar_keys:
        vals = np.array([m[key] for m in fold_metrics if key in m])
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(len(vals)))
        result[key] = {
            "value":    mean,
            "ci_lower": mean - t_crit * sem,
            "ci_upper": mean + t_crit * sem,
            "std":      float(vals.std(ddof=1)),
            "fold_values": vals.tolist(),
        }

    return result
