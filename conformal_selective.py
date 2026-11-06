"""
Site-conditional conformal risk control for abstention (Novelty N3).

The selective predictor used so far ranks participants by the heuristic confidence
max(p, 1-p) and answers a fixed fraction of them.  That gives no guarantee on the error
rate among the answered cases, and it treats every acquisition site alike -- which is
precisely where the model is weakest, since a small number of sites transfer poorly.

This module replaces the heuristic with **selective risk control**.  Given a target
error level alpha, a confidence threshold is chosen on labelled calibration data such
that an upper confidence bound (Clopper-Pearson, level delta) on the error rate among
the answered cases is at most alpha.  The threshold is therefore valid in finite
samples rather than only in expectation.

Three regimes are provided:

``marginal``
    One threshold for the whole cohort.

``mondrian``
    One threshold per site, using site as the Mondrian taxonomy.  Sites whose
    calibration errors are high get stricter thresholds and therefore abstain more,
    which is the principled version of "this scanner is hard".

``shift_aware``
    For a site with no labels at all -- the leave-one-site-out case -- a threshold
    cannot be calibrated directly.  The marginal threshold is instead inflated in
    proportion to how far that site's connectivity distribution sits from the training
    reference, measured by the affine-invariant Riemannian distance between the site's
    mean covariance and the reference.  The shift score uses no labels.

Usage:
    thr = fit_thresholds(y_cal, p_cal, sites_cal, alpha=0.2, mode="mondrian")
    report = evaluate(y_test, p_test, sites_test, thr)
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# --------------------------------------------------------------------------
# finite-sample risk bound
# --------------------------------------------------------------------------

def clopper_pearson_upper(n_err: int, n_sel: int, delta: float = 0.1) -> float:
    """Upper (1-delta) confidence bound on a binomial error rate."""
    if n_sel == 0:
        return 1.0
    if n_err == n_sel:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1.0 - delta, n_err + 1, n_sel - n_err))


def selective_threshold(conf: np.ndarray, err: np.ndarray, alpha: float,
                        delta: float = 0.1, min_selected: int = 10) -> Dict[str, float]:
    """Largest-coverage confidence threshold whose risk upper bound is <= alpha.

    Candidate thresholds are the *distinct* confidence values, and each is scored on the
    set it actually selects under the rule ``confidence >= threshold``.  Scoring a cut at
    a fixed count instead would place participants who share the threshold value on the
    other side of it at application time, so that the set evaluated here and the set
    answered later could differ and the guarantee could be missed by a small margin.

    The candidates are walked from the most demanding cut towards the least, and the walk
    stops at the first threshold whose bound exceeds the target.  Continuing past a failure
    and keeping any later threshold that happened to pass would be selecting the most
    favourable of many looks at the same data, which inflates the reported guarantee; the
    fixed-sequence rule used here controls the family-wise error rate over the candidates
    without a multiplicity penalty (Angelopoulos et al., Learn-then-Test).
    """
    conf = np.asarray(conf, dtype=float)
    err = np.asarray(err, dtype=int)
    n = len(conf)

    best = {"threshold": 1.01, "coverage": 0.0, "risk_bound": 1.0, "emp_risk": float("nan")}
    for t in np.unique(conf)[::-1]:                    # most demanding cut first
        sel = conf >= t
        k = int(sel.sum())
        if k < min_selected:
            continue                                   # not yet testable, not yet a failure
        ub = clopper_pearson_upper(int(err[sel].sum()), k, delta)
        if ub > alpha:
            break                                      # first failure ends the sequence
        best = {
            "threshold": float(t),
            "coverage": float(k / n),
            "risk_bound": float(ub),
            "emp_risk": float(err[sel].mean()),
        }
    return best


# --------------------------------------------------------------------------
# threshold fitting
# --------------------------------------------------------------------------

def _conf_err(y: np.ndarray, p: np.ndarray):
    conf = np.maximum(p, 1.0 - p)
    err = ((p >= 0.5).astype(int) != np.asarray(y).astype(int)).astype(int)
    return conf, err


def fit_thresholds(y: np.ndarray, p: np.ndarray, sites: np.ndarray | None = None,
                   alpha: float = 0.2, delta: float = 0.1,
                   mode: str = "marginal", min_selected: int = 10) -> Dict:
    """Calibrate abstention thresholds on labelled data."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    conf, err = _conf_err(y, p)
    marginal = selective_threshold(conf, err, alpha, delta, min_selected)
    out: Dict = {"mode": mode, "alpha": alpha, "delta": delta, "marginal": marginal}

    if mode == "mondrian":
        if sites is None:
            raise ValueError("mondrian mode needs site labels")
        sites = np.asarray(sites)
        per_site = {}
        for s in np.unique(sites):
            m = sites == s
            if m.sum() >= min_selected:
                per_site[str(s)] = selective_threshold(conf[m], err[m], alpha, delta,
                                                       min_selected=min(min_selected, int(m.sum())))
            else:                                   # too small to calibrate on its own
                per_site[str(s)] = dict(marginal)
        out["per_site"] = per_site
    return out


def honest_evaluate(y, p, sites=None, alpha: float = 0.2, delta: float = 0.1,
                    mode: str = "marginal", min_selected: int = 10,
                    n_repeats: int = 200, seed: int = 42) -> Dict:
    """Realised selective risk on data that played no part in choosing the threshold.

    Fitting a threshold and then reporting its bound on the same participants is circular:
    the threshold is chosen so that the bound passes on exactly those cases, so it passes
    by construction and says nothing about a future one.  Here the labelled participants
    are split repeatedly into a calibration half, which chooses the threshold, and an
    evaluation half, which is scored with it and never influenced it.  The reported
    quantity is the distribution of realised risk over those splits, together with how
    often it stayed under the target.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    sites_arr = None if sites is None else np.asarray(sites)
    n = len(y)
    idx = np.arange(n)

    risks, covs, met = [], [], 0
    for _ in range(n_repeats):
        perm = rng.permutation(idx)
        cal, ev = perm[: n // 2], perm[n // 2:]
        try:
            thr = fit_thresholds(y[cal], p[cal],
                                 None if sites_arr is None else sites_arr[cal],
                                 alpha=alpha, delta=delta, mode=mode,
                                 min_selected=min_selected)
            out = evaluate(y[ev], p[ev],
                           None if sites_arr is None else sites_arr[ev], thr)
        except Exception:
            continue
        if not out or not np.isfinite(out.get("selective_risk", np.nan)):
            continue
        risks.append(float(out["selective_risk"]))
        covs.append(float(out["coverage"]))
        met += int(out["selective_risk"] <= alpha)

    if not risks:
        return {"mode": mode, "alpha": alpha, "n_valid": 0}
    risks_a, covs_a = np.asarray(risks), np.asarray(covs)
    return {
        "mode": mode,
        "alpha": float(alpha),
        "n_repeats": int(len(risks)),
        "held_out_risk_mean": float(risks_a.mean()),
        "held_out_risk_median": float(np.median(risks_a)),
        "held_out_risk_p90": float(np.percentile(risks_a, 90)),
        "held_out_coverage_mean": float(covs_a.mean()),
        "fraction_target_met": float(met / len(risks)),
    }


def shift_aware_threshold(marginal_threshold: float, shift_score: float,
                          shift_ref: float, max_inflate: float = 0.25) -> float:
    """Inflate the marginal threshold for an unlabelled, unseen site.

    DEPRECATED and reported only for comparison.  The inflation is heuristic: it carries
    no coverage guarantee of any kind, and estimating the amount of inflation from data
    forfeits the finite-sample guarantee that weighted conformal prediction would
    otherwise provide.  Prefer an exactly valid site-level test of whether the new site
    is in scope at all.

    ``shift_score`` is the site's Riemannian distance to the training reference and
    ``shift_ref`` the median distance across training sites.  A site twice as far as
    typical is required to be correspondingly more confident before it is answered.
    """
    if not np.isfinite(shift_score) or shift_ref <= 0:
        return float(marginal_threshold)
    rel = max(0.0, shift_score / shift_ref - 1.0)
    inflate = min(max_inflate, 0.5 * rel * (1.0 - marginal_threshold))
    return float(min(0.999, marginal_threshold + inflate))


def site_shift_scores(covs: np.ndarray, sites: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """Affine-invariant Riemannian distance from each site's mean covariance to the reference.

    Uses no labels, so it is available for a site that was never seen in training.
    """
    import scipy.linalg as sla
    R = np.asarray(reference, dtype=np.float64)
    w, V = np.linalg.eigh(R)
    R_isqrt = (V * (1.0 / np.sqrt(np.clip(w, 1e-8, None)))[None, :]) @ V.T
    out: Dict[str, float] = {}
    for s in np.unique(sites):
        M = np.asarray(covs[sites == s], dtype=np.float64).mean(0)
        W = R_isqrt @ M @ R_isqrt
        ev = np.linalg.eigvalsh(W)
        out[str(s)] = float(np.sqrt(np.sum(np.log(np.clip(ev, 1e-8, None)) ** 2)))
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(y: np.ndarray, p: np.ndarray, sites: np.ndarray | None,
             thresholds: Dict, per_site_report: bool = True) -> Dict:
    """Apply calibrated thresholds and report coverage and realised risk."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    conf, err = _conf_err(y, p)
    mode = thresholds.get("mode", "marginal")

    if mode == "mondrian" and sites is not None:
        sites = np.asarray(sites)
        thr_vec = np.array([
            thresholds["per_site"].get(str(s), thresholds["marginal"])["threshold"]
            for s in sites])
    else:
        thr_vec = np.full(len(y), thresholds["marginal"]["threshold"])

    sel = conf >= thr_vec
    n_sel = int(sel.sum())
    report = {
        "mode": mode,
        "alpha": thresholds.get("alpha"),
        "coverage": float(n_sel / len(y)) if len(y) else 0.0,
        "selective_risk": float(err[sel].mean()) if n_sel else float("nan"),
        "selective_accuracy": float(1.0 - err[sel].mean()) if n_sel else float("nan"),
        "risk_bound": clopper_pearson_upper(int(err[sel].sum()), n_sel,
                                            thresholds.get("delta", 0.1)),
        "n_selected": n_sel, "n_total": int(len(y)),
        "full_accuracy": float(1.0 - err.mean()) if len(y) else float("nan"),
    }
    if per_site_report and sites is not None:
        rows = {}
        for s in np.unique(sites):
            m = sites == s
            ms = m & sel
            rows[str(s)] = {
                "n": int(m.sum()),
                "coverage": float(ms.sum() / m.sum()) if m.sum() else 0.0,
                "selective_risk": float(err[ms].mean()) if ms.sum() else float("nan"),
                "full_risk": float(err[m].mean()) if m.sum() else float("nan"),
                "target_met": bool(ms.sum() and err[ms].mean() <= thresholds.get("alpha", 1.0)),
            }
        report["per_site"] = rows
    return report
