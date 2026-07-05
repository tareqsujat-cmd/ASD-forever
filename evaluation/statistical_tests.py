"""
Statistical significance tests for classifier comparison.

Tests implemented:

1. McNemar's Test
   For comparing two classifiers on the same test set.
   Tests whether the classifiers make different errors (not whether they
   differ in accuracy).  The null hypothesis: the error patterns are symmetric.

2. DeLong's Test
   For comparing two ROC AUC values measured on the same subjects.
   More powerful than bootstrap-based AUC comparison because it uses the
   exact covariance structure between the two sets of predictions.
   The method implemented here is the fast O(N log N) version.

3. Wilcoxon Signed-Rank Test
   For comparing paired per-fold metrics across K folds.
   Non-parametric; does not assume normality of fold metric differences.
   Appropriate when K=5 (too few folds for t-test normality assumption).

4. Paired t-Test
   Parametric alternative to Wilcoxon; appropriate for larger K (≥10 folds)
   where the central limit theorem ensures approximate normality of differences.
   Included for completeness and ANOVA post-hoc comparisons.

5. Permutation Test
   Non-parametric test for comparing two classifiers without distributional
   assumptions.  Under H₀ (models are equivalent), the observed metric
   difference is no larger than that produced by random permutation of labels.
   Provides exact p-values even for small sample sizes.

6. One-Way ANOVA (with Kruskal-Wallis fallback)
   For comparing three or more models simultaneously.
   Kruskal-Wallis is preferred (non-parametric; does not assume normality)
   and is used by default.  Parametric one-way ANOVA is offered for cases
   where the normality assumption can be verified.

7. Effect Sizes
   Cohen's d and Hedges' g for continuous metrics; rank-biserial correlation
   for Wilcoxon.  Required by IEEE reviewers for practical significance.

8. Multiple Comparison Correction
   Bonferroni (conservative, controls family-wise error) and Benjamini-Hochberg
   (FDR control; preferred for exploratory comparisons of many ablation models).

References
----------
McNemar Q. (1947). Note on the sampling error of the difference between
  correlated proportions or percentages. Psychometrika.
DeLong ER et al. (1988). Comparing the areas under two or more correlated
  receiver operating characteristic curves: a nonparametric approach.
  Biometrics.
Sun X, Xu W. (2014). Fast implementation of DeLong's algorithm for comparing
  the areas under correlated receiver operating characteristic curves.
  IEEE Signal Processing Letters 21(11):1389–1393.
Wilcoxon F. (1945). Individual comparisons by ranking methods. Biometrics.
Cohen J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.)
Benjamini Y, Hochberg Y. (1995). Controlling the false discovery rate: a
  practical and powerful approach to multiple testing. JRSS-B.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. McNemar's Test
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    continuity_correction: bool = True,
) -> Dict[str, float]:
    """
    McNemar's test for paired classifier comparison.

    Parameters
    ----------
    y_true   : (N,) true binary labels
    y_pred_a : (N,) predicted labels from model A
    y_pred_b : (N,) predicted labels from model B
    continuity_correction : bool
        Apply Edwards' continuity correction (|b-c|-1)² / (b+c).

    Returns
    -------
    dict:
        "b"         : int — A correct, B wrong
        "c"         : int — A wrong, B correct
        "statistic" : chi-squared statistic
        "p_value"   : two-sided p-value
        "significant_0.05" : bool
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)

    b = int(( a_correct & ~b_correct).sum())  # A correct, B wrong
    c = int((~a_correct &  b_correct).sum())  # A wrong, B correct

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0,
                "significant_0.05": False}

    if continuity_correction:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        stat = (b - c) ** 2 / (b + c)

    p_value = float(scipy_stats.chi2.sf(stat, df=1))

    return {
        "b": b,
        "c": c,
        "statistic": float(stat),
        "p_value": p_value,
        "significant_0.05": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# 2. DeLong's Test
# ---------------------------------------------------------------------------

def _structural_components(y_true: np.ndarray, y_score: np.ndarray):
    """
    Compute V10 (for positive cases) and V01 (for negative cases).

    V10_i = P(score_neg < score_pos_i) + 0.5 * P(score_neg == score_pos_i)
    V01_j = P(score_pos > score_neg_j) + 0.5 * P(score_pos == score_neg_j)

    Fast vectorised implementation using sorted arrays.
    """
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return np.array([]), np.array([])

    neg_sorted = np.sort(neg_scores)
    pos_sorted = np.sort(pos_scores)

    # V10: for each positive case, fraction of negatives it beats
    V10 = np.empty(n_pos)
    for i, ps in enumerate(pos_scores):
        less = int(np.searchsorted(neg_sorted, ps, side="left"))
        equal = int(np.searchsorted(neg_sorted, ps, side="right")) - less
        V10[i] = (less + 0.5 * equal) / n_neg

    # V01: for each negative case, fraction of positives that beat it
    V01 = np.empty(n_neg)
    for j, ns in enumerate(neg_scores):
        greater = n_pos - int(np.searchsorted(pos_sorted, ns, side="right"))
        equal = int(np.searchsorted(pos_sorted, ns, side="right")) - \
                int(np.searchsorted(pos_sorted, ns, side="left"))
        V01[j] = (greater + 0.5 * equal) / n_pos

    return V10, V01


def _auc_var(V10: np.ndarray, V01: np.ndarray) -> float:
    """AUC variance from structural components."""
    n_pos, n_neg = len(V10), len(V01)
    s10 = float(np.var(V10, ddof=1)) / n_pos if n_pos > 1 else 0.0
    s01 = float(np.var(V01, ddof=1)) / n_neg if n_neg > 1 else 0.0
    return s10 + s01


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> Dict[str, float]:
    """
    DeLong's test for comparing two correlated ROC AUCs.

    Both models are evaluated on the same subjects (y_true is shared),
    so the AUC estimates are correlated — the DeLong test accounts for
    this correlation, yielding a more powerful test than independent comparison.

    Parameters
    ----------
    y_true    : (N,) true binary labels
    y_score_a : (N,) continuous scores from model A
    y_score_b : (N,) continuous scores from model B

    Returns
    -------
    dict:
        "auc_a", "auc_b" : AUC estimates
        "auc_diff"       : auc_a − auc_b
        "z_statistic"    : z-score
        "p_value"        : two-sided p-value
        "significant_0.05": bool
        "var_a", "var_b", "covar_ab": variance components
    """
    y_true = np.asarray(y_true)

    V10_a, V01_a = _structural_components(y_true, y_score_a)
    V10_b, V01_b = _structural_components(y_true, y_score_b)

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    auc_a = float(np.mean(V10_a)) if len(V10_a) > 0 else float("nan")
    auc_b = float(np.mean(V10_b)) if len(V10_b) > 0 else float("nan")

    var_a = _auc_var(V10_a, V01_a)
    var_b = _auc_var(V10_b, V01_b)

    # Cross-covariance between A and B
    s10_ab = (float(np.cov(V10_a, V10_b, ddof=1)[0, 1]) / n_pos
              if n_pos > 1 else 0.0)
    s01_ab = (float(np.cov(V01_a, V01_b, ddof=1)[0, 1]) / n_neg
              if n_neg > 1 else 0.0)
    covar_ab = s10_ab + s01_ab

    var_diff = var_a + var_b - 2 * covar_ab
    if var_diff <= 0:
        z = 0.0
        p_value = 1.0
    else:
        z = (auc_a - auc_b) / np.sqrt(var_diff)
        p_value = float(2 * scipy_stats.norm.sf(abs(z)))

    return {
        "auc_a":            auc_a,
        "auc_b":            auc_b,
        "auc_diff":         auc_a - auc_b,
        "z_statistic":      float(z),
        "p_value":          p_value,
        "significant_0.05": p_value < 0.05,
        "var_a":            var_a,
        "var_b":            var_b,
        "covar_ab":         covar_ab,
    }


# ---------------------------------------------------------------------------
# 3. Wilcoxon Signed-Rank Test
# ---------------------------------------------------------------------------

def wilcoxon_cv_test(
    fold_metrics_a: List[float],
    fold_metrics_b: List[float],
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test for paired per-fold metrics.

    Use this to compare two models over K-fold CV results.
    The test is non-parametric and does not assume normal differences —
    appropriate for K=5 folds (too few for paired t-test).

    Parameters
    ----------
    fold_metrics_a : list of K metric values (e.g., per-fold AUC) for model A
    fold_metrics_b : list of K metric values for model B

    Returns
    -------
    dict:
        "statistic"        : Wilcoxon test statistic
        "p_value"          : two-sided p-value
        "mean_diff"        : mean(a) − mean(b)
        "significant_0.05" : bool
    """
    a = np.asarray(fold_metrics_a, dtype=float)
    b = np.asarray(fold_metrics_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(
            f"fold_metrics_a and fold_metrics_b must have the same length, "
            f"got {len(a)} and {len(b)}"
        )
    if len(a) < 3:
        logger.warning(
            "Wilcoxon test with K=%d folds has very low power. "
            "Interpret p-value cautiously.", len(a)
        )

    try:
        stat, p = scipy_stats.wilcoxon(a, b, alternative="two-sided",
                                        correction=True)
    except ValueError as e:
        # wilcoxon raises ValueError if all differences are zero
        stat, p = 0.0, 1.0
        logger.debug("Wilcoxon raised ValueError (likely all-zero diffs): %s", e)

    return {
        "statistic":        float(stat),
        "p_value":          float(p),
        "mean_diff":        float(np.mean(a) - np.mean(b)),
        "mean_a":           float(np.mean(a)),
        "mean_b":           float(np.mean(b)),
        "significant_0.05": float(p) < 0.05,
    }


# ---------------------------------------------------------------------------
# 4. Paired t-Test
# ---------------------------------------------------------------------------

def paired_ttest(
    fold_metrics_a: Sequence[float],
    fold_metrics_b: Sequence[float],
) -> Dict[str, float]:
    """
    Paired one-sample t-test for per-fold metric comparison.

    Tests H₀: E[metric_A − metric_B] = 0 against H₁: ≠ 0.

    Parameters
    ----------
    fold_metrics_a : K per-fold metric values for model A
    fold_metrics_b : K per-fold metric values for model B

    Returns
    -------
    dict:
        "t_statistic"       : t-statistic
        "p_value"           : two-sided p-value
        "df"                : degrees of freedom (K-1)
        "mean_diff"         : mean(A) − mean(B)
        "se_diff"           : standard error of the difference
        "ci_95_low"         : lower bound of 95% CI for mean difference
        "ci_95_high"        : upper bound of 95% CI for mean difference
        "significant_0.05"  : bool
    """
    a = np.asarray(fold_metrics_a, dtype=float)
    b = np.asarray(fold_metrics_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(
            f"Sequences must have equal length; got {len(a)} and {len(b)}"
        )
    diffs = a - b
    k = len(diffs)
    mean_diff = float(np.mean(diffs))
    se_diff   = float(np.std(diffs, ddof=1) / np.sqrt(k)) if k > 1 else 0.0

    if se_diff == 0.0:
        t, p = 0.0, 1.0
    else:
        t, p = scipy_stats.ttest_rel(a, b)
        t, p = float(t), float(p)

    t_crit = float(scipy_stats.t.ppf(0.975, df=k - 1)) if k > 1 else float("nan")
    ci_low  = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    return {
        "t_statistic":      t,
        "p_value":          p,
        "df":               k - 1,
        "mean_diff":        mean_diff,
        "se_diff":          se_diff,
        "ci_95_low":        ci_low,
        "ci_95_high":       ci_high,
        "mean_a":           float(np.mean(a)),
        "mean_b":           float(np.mean(b)),
        "significant_0.05": p < 0.05,
    }


# ---------------------------------------------------------------------------
# 5. Permutation Test
# ---------------------------------------------------------------------------

def permutation_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric_fn=None,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    One-sided permutation test for classifier comparison.

    Under H₀, model A and model B are equivalent, so randomly flipping
    which model's prediction is used for each subject should produce
    differences at least as large as the observed one with probability α.

    This yields exact p-values without distributional assumptions and is
    valid even for small test sets.

    Parameters
    ----------
    y_true         : (N,) true binary labels
    y_prob_a       : (N,) predicted probabilities from model A
    y_prob_b       : (N,) predicted probabilities from model B
    metric_fn      : callable(y_true, y_prob) → float
                     Default: AUROC (from sklearn.metrics.roc_auc_score)
    n_permutations : number of random permutations
    seed           : RNG seed for reproducibility

    Returns
    -------
    dict:
        "observed_diff"     : metric_A − metric_B
        "metric_a"          : metric on model A predictions
        "metric_b"          : metric on model B predictions
        "p_value"           : fraction of permutations ≥ observed_diff
        "n_permutations"    : n_permutations
        "significant_0.05"  : bool
    """
    from sklearn.metrics import roc_auc_score

    if metric_fn is None:
        def metric_fn(y_t, y_p):
            if len(np.unique(y_t)) < 2:
                return float("nan")
            return float(roc_auc_score(y_t, y_p))

    y_true   = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a, dtype=float)
    y_prob_b = np.asarray(y_prob_b, dtype=float)
    n        = len(y_true)

    metric_a = metric_fn(y_true, y_prob_a)
    metric_b = metric_fn(y_true, y_prob_b)
    observed = metric_a - metric_b

    rng      = np.random.default_rng(seed)
    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        # For each subject, randomly swap A and B predictions with prob 0.5
        mask = rng.integers(0, 2, size=n, dtype=bool)
        perm_a = np.where(mask, y_prob_a, y_prob_b)
        perm_b = np.where(mask, y_prob_b, y_prob_a)
        null_dist[i] = metric_fn(y_true, perm_a) - metric_fn(y_true, perm_b)

    # One-sided: fraction of permutations with diff ≥ observed |observed_diff|
    p_value = float(np.mean(null_dist >= observed))

    return {
        "observed_diff":    float(observed),
        "metric_a":         float(metric_a),
        "metric_b":         float(metric_b),
        "p_value":          p_value,
        "n_permutations":   n_permutations,
        "null_mean":        float(np.mean(null_dist)),
        "null_std":         float(np.std(null_dist)),
        "significant_0.05": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# 6. Multi-model comparison: Kruskal-Wallis + one-way ANOVA
# ---------------------------------------------------------------------------

def kruskal_wallis_test(
    *groups: Sequence[float],
    model_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Kruskal-Wallis H test for comparing ≥ 3 models simultaneously.

    Non-parametric equivalent of one-way ANOVA; tests H₀: all groups have
    the same distribution of per-fold metrics.  Rejects H₀ when at least
    one group differs from the others.

    Parameters
    ----------
    *groups      : variable-length list of per-fold metric sequences,
                   one per model.  E.g. kruskal_wallis_test(folds_A, folds_B, folds_C)
    model_names  : optional labels for the groups (default "model_0", …)

    Returns
    -------
    dict:
        "h_statistic"       : H statistic
        "p_value"           : asymptotic p-value (chi-squared with k-1 df)
        "df"                : degrees of freedom (number of groups − 1)
        "n_groups"          : number of models compared
        "group_means"       : dict {name: mean} for each group
        "significant_0.05"  : bool
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for Kruskal-Wallis test.")

    arrays = [np.asarray(g, dtype=float) for g in groups]
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(arrays))]

    h, p = scipy_stats.kruskal(*arrays)

    return {
        "h_statistic":      float(h),
        "p_value":          float(p),
        "df":               len(arrays) - 1,
        "n_groups":         len(arrays),
        "group_means":      {name: float(np.mean(arr))
                             for name, arr in zip(model_names, arrays)},
        "significant_0.05": float(p) < 0.05,
    }


def one_way_anova(
    *groups: Sequence[float],
    model_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    One-way ANOVA for comparing ≥ 3 models (parametric; assumes normality).

    Use `kruskal_wallis_test` when normality cannot be verified (K-fold
    with K < 30).  This function is provided for completeness and for cases
    where the Shapiro-Wilk test does not reject normality.

    Returns
    -------
    dict with "f_statistic", "p_value", "df_between", "df_within",
    "group_means", "significant_0.05".
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for ANOVA.")

    arrays = [np.asarray(g, dtype=float) for g in groups]
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(arrays))]

    f, p = scipy_stats.f_oneway(*arrays)
    n_total = sum(len(a) for a in arrays)

    return {
        "f_statistic":      float(f),
        "p_value":          float(p),
        "df_between":       len(arrays) - 1,
        "df_within":        n_total - len(arrays),
        "n_groups":         len(arrays),
        "group_means":      {name: float(np.mean(arr))
                             for name, arr in zip(model_names, arrays)},
        "significant_0.05": float(p) < 0.05,
    }


# ---------------------------------------------------------------------------
# 7. Effect sizes
# ---------------------------------------------------------------------------

def cohen_d(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    paired: bool = True,
) -> Dict[str, float]:
    """
    Cohen's d effect size for continuous metrics.

    For paired samples (e.g., per-fold CV), uses the mean difference divided
    by the standard deviation of differences (paired Cohen's d).  For
    independent samples, uses the pooled standard deviation.

    Interpretation (Cohen 1988):
      |d| < 0.2  Negligible; 0.2–0.5  Small; 0.5–0.8  Medium; > 0.8  Large.

    Returns
    -------
    dict with "cohen_d", "hedges_g" (bias-corrected for small samples),
    "interpretation", "n_a", "n_b".
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    n_a, n_b = len(a), len(b)

    if paired:
        diff = a - b
        d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))
        n = n_a
        # Hedges' g correction factor J(df) ≈ 1 - 3/(4*df - 1)
        df = n - 1
        j  = 1.0 - 3.0 / (4.0 * max(df, 1) - 1.0)
        g  = d * j
    else:
        pooled_var = (
            ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1))
            / max(n_a + n_b - 2, 1)
        )
        d = float((np.mean(a) - np.mean(b)) / (np.sqrt(pooled_var) + 1e-12))
        df = n_a + n_b - 2
        j  = 1.0 - 3.0 / (4.0 * max(df, 1) - 1.0)
        g  = d * j

    abs_d = abs(d)
    if   abs_d < 0.2: interp = "negligible"
    elif abs_d < 0.5: interp = "small"
    elif abs_d < 0.8: interp = "medium"
    else:             interp = "large"

    return {
        "cohen_d":        d,
        "hedges_g":       g,
        "interpretation": interp,
        "n_a":            n_a,
        "n_b":            n_b,
        "paired":         paired,
    }


def rank_biserial_correlation(
    fold_metrics_a: Sequence[float],
    fold_metrics_b: Sequence[float],
) -> float:
    """
    Rank-biserial correlation r = effect size for Wilcoxon signed-rank test.

    r = 1 − (2 × W) / (n × (n+1)/2),  where W is the smaller Wilcoxon stat.

    Interpretation (Cohen 1988):
      |r| < 0.1  Negligible; 0.1–0.3  Small; 0.3–0.5  Medium; > 0.5  Large.
    """
    a = np.asarray(fold_metrics_a, dtype=float)
    b = np.asarray(fold_metrics_b, dtype=float)
    diffs = a - b
    n = len(diffs)
    if n == 0 or np.all(diffs == 0):
        return 0.0
    try:
        stat, _ = scipy_stats.wilcoxon(a, b, alternative="two-sided")
        max_w = n * (n + 1) / 2
        r = 1.0 - (2.0 * float(stat)) / max_w
    except ValueError:
        r = 0.0
    return float(r)


# ---------------------------------------------------------------------------
# 8. Multiple comparison correction
# ---------------------------------------------------------------------------

def bonferroni_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Bonferroni correction: reject H₀_i if p_i < α / m  (m = number of tests).

    Controls the family-wise error rate (FWER) conservatively.  Use when all
    hypotheses are important and a single false positive is unacceptable.

    Parameters
    ----------
    p_values : sequence of raw p-values (one per comparison)
    alpha    : desired FWER significance level (default 0.05)

    Returns
    -------
    dict:
        "corrected_alpha"     : α / m
        "adjusted_p_values"   : min(p_i * m, 1.0) for each p_i
        "reject"              : bool list — True if adjusted_p < alpha
        "n_significant"       : number of rejections
        "n_tests"             : m
    """
    ps = np.asarray(p_values, dtype=float)
    m  = len(ps)
    corrected_alpha = alpha / m
    adjusted = np.minimum(ps * m, 1.0)
    reject   = adjusted < alpha

    return {
        "corrected_alpha":   corrected_alpha,
        "adjusted_p_values": adjusted.tolist(),
        "reject":            reject.tolist(),
        "n_significant":     int(reject.sum()),
        "n_tests":           m,
        "method":            "bonferroni",
    }


def fdr_correction(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Benjamini-Hochberg (BH) False Discovery Rate (FDR) correction.

    Controls the expected proportion of false positives among rejections.
    Less conservative than Bonferroni; preferred for exploratory comparisons
    across many ablation models.

    Procedure: sort p-values p₍₁₎ ≤ … ≤ p₍ₘ₎; reject all H₀₍ᵢ₎ where
    p₍ᵢ₎ ≤ (i/m) × α.

    Returns
    -------
    dict:
        "adjusted_p_values" : BH-adjusted p-values (via Simes formula)
        "reject"            : bool list
        "n_significant"     : number of rejections
        "n_tests"           : m
        "method"            : "benjamini_hochberg"
    """
    ps    = np.asarray(p_values, dtype=float)
    m     = len(ps)
    order = np.argsort(ps)
    ranks = np.argsort(order) + 1  # 1-based ranks in sorted order

    # BH adjusted p-values: p_adj_i = min over j>=rank(i) of p_j * m / j
    ps_sorted    = ps[order]
    adj_sorted   = np.minimum(1.0, ps_sorted * m / np.arange(1, m + 1))
    # Enforce monotonicity from right to left
    for k in range(m - 2, -1, -1):
        adj_sorted[k] = min(adj_sorted[k], adj_sorted[k + 1])

    adjusted = adj_sorted[np.argsort(order)]  # back to original order
    reject   = adjusted < alpha

    return {
        "adjusted_p_values": adjusted.tolist(),
        "reject":            reject.tolist(),
        "n_significant":     int(reject.sum()),
        "n_tests":           m,
        "method":            "benjamini_hochberg",
    }


# ---------------------------------------------------------------------------
# Convenience: run all pairwise comparisons for an ablation study
# ---------------------------------------------------------------------------

def pairwise_comparison_table(
    fold_metrics: Dict[str, List[float]],
    reference_model: Optional[str] = None,
    correction: str = "fdr",
    alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Run all pairwise Wilcoxon tests across ablation models, correct for
    multiple comparisons, and return a structured summary.

    Parameters
    ----------
    fold_metrics     : {model_name: [fold_metric, ...]} for each model
    reference_model  : if given, compare all others against this one only;
                       otherwise compare all unique pairs
    correction       : "bonferroni" | "fdr"
    alpha            : significance level

    Returns
    -------
    dict:
        "pairs"          : list of (model_A, model_B) tuples
        "raw_p_values"   : list of raw p-values
        "adjusted_p_values": list after correction
        "reject"         : list of bool
        "effect_sizes"   : list of Cohen's d
        "mean_diffs"     : list of mean(A) - mean(B)
        "correction"     : correction method used
    """
    names  = list(fold_metrics.keys())
    arrays = {name: np.asarray(v, dtype=float) for name, v in fold_metrics.items()}

    # Build pairs
    pairs: List[Tuple[str, str]] = []
    if reference_model is not None:
        for name in names:
            if name != reference_model:
                pairs.append((reference_model, name))
    else:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((names[i], names[j]))

    raw_p, effect_sizes, mean_diffs = [], [], []
    for (a_name, b_name) in pairs:
        res = wilcoxon_cv_test(arrays[a_name].tolist(),
                               arrays[b_name].tolist())
        raw_p.append(res["p_value"])
        mean_diffs.append(res["mean_diff"])
        eff = cohen_d(arrays[a_name].tolist(), arrays[b_name].tolist(), paired=True)
        effect_sizes.append(eff["cohen_d"])

    if correction == "bonferroni":
        corrected = bonferroni_correction(raw_p, alpha=alpha)
    else:
        corrected = fdr_correction(raw_p, alpha=alpha)

    return {
        "pairs":              [(a, b) for a, b in pairs],
        "raw_p_values":       raw_p,
        "adjusted_p_values":  corrected["adjusted_p_values"],
        "reject":             corrected["reject"],
        "effect_sizes":       effect_sizes,
        "mean_diffs":         mean_diffs,
        "correction":         corrected["method"],
        "alpha":              alpha,
        "n_significant":      corrected["n_significant"],
    }
