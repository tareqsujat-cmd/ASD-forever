"""
Core evaluation metrics for binary ASD classification.

All functions accept numpy arrays and return Python floats.

Metric selection rationale
---------------------------
AUC-ROC:        Threshold-independent; standard for medical AI benchmarks.
AUC-PR:         More informative when the positive class (ASD) is not
                dominant; highlights precision at high recall.
Sensitivity:    Clinically critical — missing an ASD diagnosis has high cost.
Specificity:    Avoids over-referral of neurotypical subjects.
PPV / NPV:      Operating-point predictive values; prevalence-dependent.
LR+ / LR-:     Prevalence-independent diagnostic strength; used in
                evidence-based medicine meta-analyses.
DOR:            Single threshold-free summary of discriminative power;
                log(DOR) is additive across studies (meta-analysis).
MCC:            Best single scalar for binary imbalanced classification
                (unlike F1, it accounts for all four confusion matrix cells).
Cohen's Kappa:  Chance-corrected agreement; required by IEEE reviewers
                for clinical AI submissions.
Balanced acc:   Arithmetic mean of sensitivity + specificity; often used
                in ABIDE benchmark comparisons.
Log Loss:       Strictly proper scoring rule; penalises confident wrong
                predictions more than Brier score.
ECE:            Calibration quality — important for uncertainty quantification.

References
----------
Chicco D, Jurman G. (2020). The advantages of the Matthews correlation
  coefficient over F1 score and accuracy in binary classification evaluation.
  BMC Genomics.
Saito T, Rehmsmeier M. (2015). The precision-recall plot is more
  informative than the ROC plot when evaluating binary classifiers on
  imbalanced datasets. PLOS ONE.
Glas A et al. (2003). The diagnostic odds ratio: a single indicator of
  test performance. Journal of Clinical Epidemiology.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
    confusion_matrix,
    f1_score,
    brier_score_loss,
    cohen_kappa_score,
    log_loss,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confusion matrix primitives
# ---------------------------------------------------------------------------

def compute_confusion(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """Return (TP, TN, FP, FN) from boolean predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tp), int(tn), int(fp), int(fn)


# ---------------------------------------------------------------------------
# Threshold-dependent metrics — individual functions
# ---------------------------------------------------------------------------

def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True positive rate (Recall for ASD class): TP / (TP + FN)."""
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    return tp / max(tp + fn, 1)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True negative rate (Recall for TC class): TN / (TN + FP)."""
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    return tn / max(tn + fp, 1)


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FPR = FP / (FP + TN) = 1 − specificity."""
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    return fp / max(fp + tn, 1)


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FNR = FN / (FN + TP) = 1 − sensitivity  (Miss Rate)."""
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    return fn / max(fn + tp, 1)


def ppv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Positive predictive value (Precision for ASD class): TP / (TP + FP)."""
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    return tp / max(tp + fp, 1)


def npv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Negative predictive value (Precision for TC class): TN / (TN + FN)."""
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    return tn / max(tn + fn, 1)


def likelihood_ratio_positive(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    LR+ = sensitivity / (1 − specificity) = TPR / FPR.

    Measures how much a positive test result increases the probability of
    disease.  LR+ > 10 is considered strong evidence; LR+ = 1 means no
    discriminative value.

    Returns inf when FPR = 0 (perfect test at this operating point).
    """
    sens = sensitivity(y_true, y_pred)
    fpr  = false_positive_rate(y_true, y_pred)
    if fpr == 0.0:
        return float("inf") if sens > 0 else float("nan")
    return sens / fpr


def likelihood_ratio_negative(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    LR− = (1 − sensitivity) / specificity = FNR / TNR.

    Measures how much a negative test result decreases disease probability.
    LR− < 0.1 is considered strong evidence for ruling out; LR− = 1 → no value.

    Returns 0 when FNR = 0 (no missed positives at this operating point).
    """
    fnr  = false_negative_rate(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    if spec == 0.0:
        return float("nan")
    return fnr / spec


def diagnostic_odds_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    DOR = LR+ / LR−  =  (TP × TN) / (FP × FN).

    Prevalence-independent summary of test performance; used in diagnostic
    meta-analysis.  log(DOR) is normally distributed and additive.

    Returns inf if FP=0 or FN=0 (no errors on one side).
    Returns nan if both FP=0 and FN=0 (perfect classifier; degenerate case).
    """
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    denom = fp * fn
    if denom == 0:
        if fp == 0 and fn == 0:
            return float("nan")   # perfect on both sides → undefined
        return float("inf")
    return (tp * tn) / denom


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Arithmetic mean of sensitivity and specificity."""
    return 0.5 * (sensitivity(y_true, y_pred) + specificity(y_true, y_pred))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall fraction of correctly classified samples."""
    return float((y_true == y_pred).mean())


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews Correlation Coefficient ∈ [-1, +1]."""
    return float(matthews_corrcoef(y_true, y_pred))


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cohen's κ — chance-corrected inter-rater agreement.

    κ = (p_o − p_e) / (1 − p_e)

    Interpretation (Landis & Koch 1977):
      < 0.00  Slight; 0.00–0.20  Slight; 0.21–0.40  Fair;
      0.41–0.60  Moderate; 0.61–0.80  Substantial; > 0.80  Near-perfect.
    """
    return float(cohen_kappa_score(y_true, y_pred))


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 score for the positive (ASD) class."""
    return float(f1_score(y_true, y_pred, zero_division=0))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1: unweighted mean over both classes."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted F1: weighted by support (number of samples per class)."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Per-class precision, recall, and F1 for both TC (class 0) and ASD (class 1).

    Returns
    -------
    dict with keys "TC" and "ASD", each containing:
        precision, recall, f1, support (int)
    """
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)
    return {
        "TC": {
            "precision": float(precision_score(y_true, y_pred,
                                               pos_label=0, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred,
                                            pos_label=0, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred,
                                        pos_label=0, zero_division=0)),
            "support":   int((y_true == 0).sum()),
        },
        "ASD": {
            "precision": float(precision_score(y_true, y_pred,
                                               pos_label=1, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred,
                                            pos_label=1, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred,
                                        pos_label=1, zero_division=0)),
            "support":   int((y_true == 1).sum()),
        },
    }


# ---------------------------------------------------------------------------
# Threshold-independent metrics
# ---------------------------------------------------------------------------

def auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the ROC curve."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the Precision-Recall curve (Average Precision)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score: mean squared error of probability estimates. Lower=better."""
    return float(brier_score_loss(y_true, y_prob))


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Binary cross-entropy (Log Loss). Lower=better.

    Strictly proper scoring rule — penalises confident wrong predictions
    more severely than Brier score.  For a calibrated classifier with p=0.5
    prior, the baseline is log(2) ≈ 0.693.
    """
    # Clip to avoid log(0); sklearn does the same internally
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return float(log_loss(y_true, y_prob))


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def optimal_threshold_youden(
    y_true: np.ndarray, y_prob: np.ndarray
) -> float:
    """
    Optimal decision threshold via Youden's J statistic.
    J = sensitivity + specificity − 1; maximise J over all thresholds.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def optimal_threshold_f1(
    y_true: np.ndarray, y_prob: np.ndarray
) -> float:
    """Threshold that maximises F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-8)
    idx = int(np.argmax(f1s[:-1]))  # last threshold is undefined
    return float(thresholds[idx])


# ---------------------------------------------------------------------------
# Full metric suite
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None,
    threshold_method: str = "youden_j",
) -> dict:
    """
    Compute the complete suite of evaluation metrics for binary ASD classification.

    Covers all metrics required by IEEE medical AI submissions including:
    threshold-independent (AUROC, AUPRC, Brier, Log Loss), threshold-dependent
    (Sensitivity, Specificity, PPV, NPV, FPR, FNR, F1, MCC, κ, Balanced Acc),
    diagnostic (LR+, LR−, DOR), and per-class breakdowns.

    Parameters
    ----------
    y_true : (N,) int array, 0=TC, 1=ASD
    y_prob : (N,) float array, predicted probability of ASD
    threshold : float, optional
        Decision threshold.  If None, computed from threshold_method.
    threshold_method : str
        "youden_j" (maximises Youden's J = TPR − FPR) |
        "f1" (maximises F1 on the training/validation set)

    Returns
    -------
    dict mapping metric name → float (or nested dict for per_class)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if threshold is None:
        if threshold_method == "youden_j":
            threshold = optimal_threshold_youden(y_true, y_prob)
        else:
            threshold = optimal_threshold_f1(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tp, tn, fp, fn = compute_confusion(y_true, y_pred)

    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    fpr_ = fp / max(fp + tn, 1)
    fnr_ = fn / max(fn + tp, 1)

    # Likelihood ratios — compute directly to avoid redundant confusion calls
    lr_pos = (sens / fpr_) if fpr_ > 0 else (float("inf") if sens > 0 else float("nan"))
    lr_neg = (fnr_ / spec) if spec > 0 else float("nan")
    dor    = (lr_pos / lr_neg) if (lr_neg not in (0.0, float("nan"))) else (
        float("nan") if (fp * fn == 0 and tp * tn == 0) else float("inf")
    )

    metrics = {
        # --- Threshold-independent ---
        "auroc":              auroc(y_true, y_prob),
        "auprc":              auprc(y_true, y_prob),
        "brier_score":        brier_score(y_true, y_prob),
        "log_loss":           log_loss_score(y_true, y_prob),

        # --- Overall threshold-dependent ---
        "accuracy":           accuracy(y_true, y_pred),
        "balanced_accuracy":  balanced_accuracy(y_true, y_pred),

        # --- Per-class rates ---
        "sensitivity":        sens,          # TPR / ASD Recall
        "specificity":        spec,          # TNR / TC Recall
        "false_positive_rate": fpr_,         # FPR (1 - specificity)
        "false_negative_rate": fnr_,         # FNR (1 - sensitivity)
        "ppv":                ppv(y_true, y_pred),    # ASD Precision
        "npv":                npv(y_true, y_pred),    # TC Precision

        # --- Composite metrics ---
        "f1":                 f1(y_true, y_pred),
        "macro_f1":           macro_f1(y_true, y_pred),
        "weighted_f1":        weighted_f1(y_true, y_pred),
        "mcc":                mcc(y_true, y_pred),
        "cohen_kappa":        cohen_kappa(y_true, y_pred),

        # --- Diagnostic strength (prevalence-independent) ---
        "lr_positive":        lr_pos,
        "lr_negative":        lr_neg,
        "diagnostic_odds_ratio": dor,

        # --- Decision boundary ---
        "threshold":          threshold,

        # --- Raw confusion matrix cells ---
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,

        # --- Dataset composition ---
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
        "prevalence": float(y_true.mean()),

        # --- Per-class breakdown (nested dict) ---
        "per_class": per_class_metrics(y_true, y_pred),
    }

    return metrics
