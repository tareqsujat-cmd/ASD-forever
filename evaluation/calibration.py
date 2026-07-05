"""
Probability calibration analysis.

A well-calibrated model outputs P(Y=1|X=x) that equals the true fraction of
positive cases among samples with predicted probability x.  For clinical use,
calibration matters as much as discrimination (AUC): an AUC=0.90 model with
poor calibration gives misleading confidence scores to clinicians.

Metrics
-------
ECE (Expected Calibration Error):
    Weighted average of |accuracy − confidence| across M equally-spaced bins.
    Standard metric in post-hoc calibration literature.
    Lower is better; ECE=0 is perfect calibration.

MCE (Maximum Calibration Error):
    Worst-case bin calibration gap.  Useful for safety-critical settings
    where the worst bin matters more than the average.

Brier Score:
    Mean squared error of the probability: Σ (y_i − p_i)² / N.
    Decomposed into calibration + resolution + uncertainty components.

Reliability Diagram:
    Plot of (mean confidence per bin, fraction of positives per bin).
    Points on the diagonal y=x → perfect calibration.
    Points above diagonal → underconfident.
    Points below diagonal → overconfident.

Reference
---------
Guo C et al. (2017). On Calibration of Modern Neural Networks.
  ICML 2017.
Niculescu-Mizil A, Caruana R. (2005). Predicting Good Probabilities with
  Supervised Learning. ICML 2005.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict:
    """
    Compute data for a reliability (calibration) diagram.

    Parameters
    ----------
    y_true : (N,) binary labels
    y_prob : (N,) predicted probabilities [0, 1]
    n_bins : int — number of confidence bins
    strategy : str
        "uniform"  — equal-width bins [0, 0.1), [0.1, 0.2), ...
        "quantile" — equal-frequency bins (same number of samples per bin)

    Returns
    -------
    dict:
        "bin_midpoints"  : (n_bins,) centre of each bin
        "bin_accuracies" : (n_bins,) fraction of positives per bin
        "bin_confidences": (n_bins,) mean predicted probability per bin
        "bin_counts"     : (n_bins,) number of samples per bin
        "ece"            : float — Expected Calibration Error
        "mce"            : float — Maximum Calibration Error
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bins[-1] = 1.0 + 1e-8
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'")

    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_mids = 0.5 * (bins[:-1] + bins[1:])

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            bin_accs[i] = 0.0
            bin_confs[i] = bin_mids[i]
        else:
            bin_accs[i] = float(y_true[mask].mean())
            bin_confs[i] = float(y_prob[mask].mean())
            bin_counts[i] = int(mask.sum())

    N = len(y_true)
    ece = float(np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / max(N, 1))
    mce = float(np.max(np.abs(bin_accs[bin_counts > 0] - bin_confs[bin_counts > 0])))

    return {
        "bin_midpoints":   bin_mids,
        "bin_accuracies":  bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts":      bin_counts,
        "ece":             ece,
        "mce":             mce,
        "n_bins":          n_bins,
        "strategy":        strategy,
    }


def brier_score_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Brier score decomposed into calibration, resolution, and uncertainty.

    Brier = Calibration − Resolution + Uncertainty
    where:
        Uncertainty = ō * (1 − ō)  (intrinsic difficulty of the task)
        Resolution  = Σ n_k/N * (ō_k − ō)²  (how spread out predictions are)
        Calibration = Σ n_k/N * (ō_k − f_k)²  (accuracy of probabilities)

    A well-calibrated model has low Calibration term.
    A model with good resolution has high Resolution term (predictions spread).

    Reference: Murphy AH. (1973). A new vector partition of the probability
    score. Journal of Applied Meteorology.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    N = len(y_true)

    # Overall base rate
    o_bar = float(y_true.mean())

    # Bin samples
    bins = np.linspace(0, 1, n_bins + 1)
    bins[-1] += 1e-8

    calibration_term = 0.0
    resolution_term = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        n_k = mask.sum()
        if n_k == 0:
            continue
        o_k = float(y_true[mask].mean())   # observed relative frequency in bin
        f_k = float(y_prob[mask].mean())   # mean forecast in bin

        calibration_term += n_k * (o_k - f_k) ** 2
        resolution_term  += n_k * (o_k - o_bar) ** 2

    calibration_term /= N
    resolution_term  /= N
    uncertainty = o_bar * (1 - o_bar)
    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "brier_score":    brier,
        "calibration":    calibration_term,
        "resolution":     resolution_term,
        "uncertainty":    uncertainty,
        "brier_check":    calibration_term - resolution_term + uncertainty,
    }


def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Full calibration analysis: reliability diagram + Brier decomposition.

    Returns
    -------
    dict containing:
        "ece", "mce", "brier_score", "brier_decomposition",
        "reliability_diagram" (nested dict for plotting)
    """
    rel = reliability_diagram_data(y_true, y_prob, n_bins)
    brier = brier_score_decomposition(y_true, y_prob, n_bins)

    return {
        "ece":                rel["ece"],
        "mce":                rel["mce"],
        "brier_score":        brier["brier_score"],
        "brier_decomposition": brier,
        "reliability_diagram": rel,
    }
