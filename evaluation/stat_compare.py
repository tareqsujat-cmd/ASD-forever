"""
Statistical comparison tests for classifier predictions (publication E5).

- ``delong_roc_test`` : DeLong's test for two *correlated* ROC AUCs (same subjects,
  two models) — the standard way to test whether our model's AUROC significantly
  exceeds a baseline's.  Uses the fast Sun & Xu (2014) midrank algorithm.
- ``delong_auc_ci``   : DeLong variance-based 95% CI for a single AUROC.
- ``mcnemar_test``    : McNemar's test for paired binary predictions (accuracy
  comparison), with continuity correction and an exact-binomial fallback.

Validated against sklearn (AUC) and statsmodels (McNemar); see validate_stats().
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Fast DeLong (Sun & Xu 2014)
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted: np.ndarray, m: int):
    """preds_sorted: (k_models, n) with the m positives first. Returns (aucs, cov)."""
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def _order_positives_first(y_true: np.ndarray):
    order = np.argsort(-y_true)          # 1s (positives) first
    m = int(np.sum(y_true == 1))
    return order, m


def delong_roc_test(y_true, prob_a, prob_b) -> Tuple[float, float, float]:
    """
    Return (auc_a, auc_b, p_value) for H0: AUC_a == AUC_b on the same subjects.
    Two-sided p-value from DeLong's covariance.
    """
    from scipy.stats import norm
    y_true = np.asarray(y_true).astype(int)
    order, m = _order_positives_first(y_true)
    preds = np.vstack((np.asarray(prob_a, float), np.asarray(prob_b, float)))[:, order]
    aucs, cov = _fast_delong(preds, m)
    L = np.array([[1.0, -1.0]])
    var = float((L @ cov @ L.T)[0, 0])
    if var <= 0:
        p = 1.0 if aucs[0] == aucs[1] else 0.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2 * norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), p


def delong_auc_ci(y_true, prob, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Single-AUC point estimate + DeLong variance-based (1-alpha) CI."""
    from scipy.stats import norm
    y_true = np.asarray(y_true).astype(int)
    order, m = _order_positives_first(y_true)
    preds = np.asarray(prob, float)[order][None, :]
    aucs, cov = _fast_delong(preds, m)
    se = float(np.sqrt(cov[0, 0]))
    z = norm.ppf(1 - alpha / 2)
    lo, hi = aucs[0] - z * se, aucs[0] + z * se
    return float(aucs[0]), float(max(0.0, lo)), float(min(1.0, hi))


# ---------------------------------------------------------------------------
# McNemar (paired accuracy)
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, pred_a, pred_b, correction: bool = True) -> dict:
    """
    McNemar's test on paired binary predictions.

    b = #(a correct, b wrong), c = #(a wrong, b correct).  For small discordance
    (b+c < 25) an exact binomial test is used; otherwise the chi-square statistic
    (with continuity correction by default).
    """
    from scipy.stats import chi2, binomtest
    y = np.asarray(y_true).astype(int)
    a_ok = (np.asarray(pred_a).astype(int) == y)
    b_ok = (np.asarray(pred_b).astype(int) == y)
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "identical"}
    if n < 25:
        p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": float(p),
                "method": "exact-binomial"}
    stat = (abs(b - c) - (1.0 if correction else 0.0)) ** 2 / n
    p = float(chi2.sf(stat, 1))
    return {"b": b, "c": c, "statistic": float(stat), "p_value": p, "method": "chi2"}


# ---------------------------------------------------------------------------
# Self-validation
# ---------------------------------------------------------------------------

def validate_stats() -> bool:
    """Validate DeLong AUC vs sklearn and McNemar vs statsmodels (if available)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    pa = np.clip(y * 0.5 + rng.normal(0, 0.3, n) + 0.25, 0, 1)   # decent model
    pb = np.clip(y * 0.2 + rng.normal(0, 0.4, n) + 0.4, 0, 1)    # weaker model

    ok = True
    # 1) DeLong AUC matches sklearn
    auc_a, lo, hi = delong_auc_ci(y, pa)
    ref = roc_auc_score(y, pa)
    d = abs(auc_a - ref)
    print(f"DeLong AUC {auc_a:.4f} vs sklearn {ref:.4f}  (|diff|={d:.2e})  CI[{lo:.3f},{hi:.3f}]")
    ok &= d < 1e-6
    # 2) self-comparison -> p ~ 1
    _, _, p_self = delong_roc_test(y, pa, pa)
    print(f"DeLong self-comparison p={p_self:.3f} (expect ~1.0)")
    ok &= p_self > 0.99
    # 3) a-vs-b: a is clearly better -> small p
    aa, ab, p_ab = delong_roc_test(y, pa, pb)
    print(f"DeLong a={aa:.3f} b={ab:.3f} p={p_ab:.2e} (expect small, a>b)")
    ok &= (aa > ab) and (p_ab < 0.05)
    # 4) McNemar vs statsmodels
    preda = (pa >= 0.5).astype(int); predb = (pb >= 0.5).astype(int)
    mc = mcnemar_test(y, preda, predb)
    print(f"McNemar b={mc['b']} c={mc['c']} stat={mc['statistic']:.3f} p={mc['p_value']:.2e} ({mc['method']})")
    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
        table = [[int(np.sum(((pa >= .5) == y) & ((pb >= .5) == y))),
                  mc["b"]], [mc["c"],
                  int(np.sum(((pa >= .5) != y) & ((pb >= .5) != y)))]]
        sm = sm_mcnemar(table, exact=(mc["b"] + mc["c"] < 25), correction=True)
        print(f"  statsmodels McNemar p={sm.pvalue:.2e}")
        ok &= abs(sm.pvalue - mc["p_value"]) < 1e-6
    except Exception as e:
        print(f"  (statsmodels unavailable: {e})")
    print("VALIDATION:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    validate_stats()
