"""
Error analysis for the ASD detection framework.

After evaluation, a classifier's aggregate metrics (AUC, F1, …) hide the
identity of the subjects it gets wrong.  This module answers:

  1. WHO is misclassified?
     - Subject IDs, sites, demographic groups for every FP and FN
     - Confidence at the time of misclassification

  2. WHY is the model wrong?
     - Hard examples: high-confidence wrong predictions (worst failures)
     - Low-confidence predictions: model is uncertain (borderline cases)
     - Failure modes: systematic patterns (all FPs from one site, etc.)

  3. HOW severe are the errors?
     - Distribution of prediction probabilities per outcome type (TP/TN/FP/FN)
     - Error asymmetry: are FPs or FNs more common?  More confident?
     - Clinical impact: FN (missed ASD) is more costly than FP (over-referral)

  4. WHAT improves next?
     - Generates a ranked list of recommendations based on observed patterns

All outputs are JSON-serialisable for the automated report generator (M8).

Usage
-----
    from evaluation.error_analysis import ErrorAnalyzer

    analyzer = ErrorAnalyzer(clinical_fn_weight=2.0)  # FN twice as costly as FP
    report = analyzer.analyze(
        y_true      = y_true,
        y_prob      = y_prob,
        subject_ids = subject_ids,
        site_ids    = site_ids,
        metadata    = metadata_df,   # optional: age, sex, IQ columns
        out_dir     = Path("results/error_analysis"),
    )
    report.print_summary()
    report.save(out_dir / "error_analysis.json")

References
----------
Varoquaux G, Raamana PR et al. (2017). Assessing and tuning brain decoders:
  cross-validation, caveats, and guidelines. NeuroImage 145:166–179.
Rudin C. (2019). Stop explaining black box machine learning models for
  high-stakes decisions and use interpretable models instead. Nature MI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SubjectError:
    """Record of a single misclassified or uncertain subject."""
    subject_id:  str
    true_label:  int           # 0=TC, 1=ASD
    pred_label:  int
    prob_asd:    float         # P(ASD) output
    error_type:  str           # "FP" | "FN" | "TP" | "TN"
    confidence:  float         # max(prob_asd, 1-prob_asd) ∈ [0.5, 1.0]
    margin:      float         # |prob_asd - threshold| — distance to boundary
    site_id:     str  = "unknown"
    age:         Optional[float] = None
    sex:         Optional[str]   = None
    iq:          Optional[float] = None

    @property
    def is_error(self) -> bool:
        return self.error_type in ("FP", "FN")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ErrorAnalysisReport:
    """Complete error analysis for one model evaluation."""

    # ---- Counts ----
    n_total:    int = 0
    n_tp:       int = 0
    n_tn:       int = 0
    n_fp:       int = 0
    n_fn:       int = 0
    threshold:  float = 0.5

    # ---- Error rate ----
    error_rate:    float = 0.0    # (FP + FN) / N
    fp_rate:       float = 0.0    # FP / (FP + TN)  = 1 − specificity
    fn_rate:       float = 0.0    # FN / (FN + TP)  = 1 − sensitivity

    # ---- Confidence statistics per outcome ----
    conf_tp_mean:  float = float("nan")
    conf_tn_mean:  float = float("nan")
    conf_fp_mean:  float = float("nan")
    conf_fn_mean:  float = float("nan")
    conf_fp_std:   float = float("nan")
    conf_fn_std:   float = float("nan")

    # ---- Hard examples (high-confidence errors) ----
    hard_fp:    List[SubjectError] = field(default_factory=list)
    hard_fn:    List[SubjectError] = field(default_factory=list)

    # ---- Low-confidence predictions (uncertain) ----
    uncertain:  List[SubjectError] = field(default_factory=list)

    # ---- Per-site error breakdown ----
    per_site_errors: Dict[str, dict] = field(default_factory=dict)

    # ---- Failure mode catalogue ----
    failure_modes: List[str] = field(default_factory=list)

    # ---- Ranked recommendations ----
    recommendations: List[str] = field(default_factory=list)

    # ---- All classified subjects ----
    all_subjects: List[SubjectError] = field(default_factory=list)

    # ---- Clinical cost ----
    clinical_fn_weight: float = 2.0
    weighted_error_cost: float = 0.0

    def print_summary(self) -> None:
        """Log a structured summary to the experiment logger."""
        logger.info("=" * 60)
        logger.info("ERROR ANALYSIS REPORT")
        logger.info("=" * 60)
        logger.info("  N=%d  threshold=%.3f", self.n_total, self.threshold)
        logger.info("  TP=%d  TN=%d  FP=%d  FN=%d", self.n_tp, self.n_tn,
                    self.n_fp, self.n_fn)
        logger.info("  Error rate = %.1f%%  |  FP rate = %.1f%%  |  FN rate = %.1f%%",
                    self.error_rate * 100, self.fp_rate * 100, self.fn_rate * 100)
        logger.info("  Weighted clinical cost = %.4f  (FN weight=%.1f×)",
                    self.weighted_error_cost, self.clinical_fn_weight)
        logger.info("-" * 60)
        logger.info("  Confidence (correct) :  TP=%.3f  TN=%.3f",
                    self.conf_tp_mean, self.conf_tn_mean)
        logger.info("  Confidence (errors)  :  FP=%.3f±%.3f  FN=%.3f±%.3f",
                    self.conf_fp_mean, self.conf_fp_std,
                    self.conf_fn_mean, self.conf_fn_std)
        logger.info("-" * 60)
        if self.hard_fp:
            logger.info("  Hard FPs (%d) — high-confidence false positives:", len(self.hard_fp))
            for e in self.hard_fp[:5]:
                logger.info("    %s  prob=%.3f  conf=%.3f  site=%s",
                            e.subject_id, e.prob_asd, e.confidence, e.site_id)
        if self.hard_fn:
            logger.info("  Hard FNs (%d) — high-confidence missed ASD:", len(self.hard_fn))
            for e in self.hard_fn[:5]:
                logger.info("    %s  prob=%.3f  conf=%.3f  site=%s",
                            e.subject_id, e.prob_asd, e.confidence, e.site_id)
        if self.uncertain:
            logger.info("  Uncertain predictions (%d) — near decision boundary:",
                        len(self.uncertain))
        logger.info("-" * 60)
        for mode in self.failure_modes:
            logger.info("  [FAILURE MODE] %s", mode)
        logger.info("-" * 60)
        for i, rec in enumerate(self.recommendations, 1):
            logger.info("  [REC %d] %s", i, rec)
        logger.info("=" * 60)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _serialise(obj):
            if isinstance(obj, float):
                return round(obj, 6) if np.isfinite(obj) else str(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            raise TypeError(f"Cannot serialise {type(obj)}")

        data = {
            "n_total":      self.n_total,
            "n_tp":         self.n_tp,
            "n_tn":         self.n_tn,
            "n_fp":         self.n_fp,
            "n_fn":         self.n_fn,
            "threshold":    self.threshold,
            "error_rate":   self.error_rate,
            "fp_rate":      self.fp_rate,
            "fn_rate":      self.fn_rate,
            "conf_tp_mean": self.conf_tp_mean,
            "conf_tn_mean": self.conf_tn_mean,
            "conf_fp_mean": self.conf_fp_mean,
            "conf_fn_mean": self.conf_fn_mean,
            "conf_fp_std":  self.conf_fp_std,
            "conf_fn_std":  self.conf_fn_std,
            "weighted_error_cost": self.weighted_error_cost,
            "clinical_fn_weight":  self.clinical_fn_weight,
            "n_hard_fp":    len(self.hard_fp),
            "n_hard_fn":    len(self.hard_fn),
            "n_uncertain":  len(self.uncertain),
            "hard_fp":   [e.to_dict() for e in self.hard_fp],
            "hard_fn":   [e.to_dict() for e in self.hard_fn],
            "uncertain": [e.to_dict() for e in self.uncertain[:50]],
            "per_site_errors": self.per_site_errors,
            "failure_modes":   self.failure_modes,
            "recommendations": self.recommendations,
            "all_errors": [
                e.to_dict() for e in self.all_subjects if e.is_error
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_serialise)
        logger.info("Error analysis report saved → %s", path)

    def to_latex_table(self) -> str:
        """
        Generate a LaTeX table summarising per-outcome confidence statistics.
        Suitable for direct inclusion in an IEEE paper supplementary section.
        """
        lines = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \caption{Error Analysis: Confidence Distribution by Outcome}",
            r"  \label{tab:error_analysis}",
            r"  \begin{tabular}{lrrrr}",
            r"    \toprule",
            r"    Outcome & Count & Error Rate & Mean Confidence & Std Confidence \\",
            r"    \midrule",
        ]
        for outcome, count, err_rate, mu, sd in [
            ("TP (correct ASD)", self.n_tp, 0.0,       self.conf_tp_mean, float("nan")),
            ("TN (correct TC)",  self.n_tn, 0.0,       self.conf_tn_mean, float("nan")),
            ("FP (missed TC)",   self.n_fp, self.fp_rate, self.conf_fp_mean, self.conf_fp_std),
            ("FN (missed ASD)",  self.n_fn, self.fn_rate, self.conf_fn_mean, self.conf_fn_std),
        ]:
            mu_s  = f"{mu:.3f}"  if np.isfinite(mu)  else "—"
            sd_s  = f"{sd:.3f}"  if np.isfinite(sd)  else "—"
            er_s  = f"{err_rate:.1\\%}" if err_rate > 0 else "0\\%"
            lines.append(
                f"    {outcome} & {count} & {er_s} & {mu_s} & {sd_s} \\\\"
            )
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class ErrorAnalyzer:
    """
    Classify every test subject by outcome type and diagnose failure patterns.

    Parameters
    ----------
    hard_confidence_threshold : float
        Confidence above which an error is considered a "hard" failure.
        Default 0.75: model is ≥75% confident but still wrong.
    uncertainty_margin : float
        Predictions within this margin of the threshold are "uncertain".
        Default 0.10: |prob - threshold| < 0.10.
    clinical_fn_weight : float
        Relative cost of a false negative (missed ASD) vs. a false positive.
        Literature: FN in ASD screening is 2–3× more harmful than FP
        (delayed diagnosis > unnecessary referral).
    min_site_n : int
        Minimum subjects per site to report per-site error stats.
    """

    def __init__(
        self,
        hard_confidence_threshold: float = 0.75,
        uncertainty_margin:        float = 0.10,
        clinical_fn_weight:        float = 2.0,
        min_site_n:                int   = 5,
    ) -> None:
        self.hard_threshold  = hard_confidence_threshold
        self.unc_margin      = uncertainty_margin
        self.fn_weight       = clinical_fn_weight
        self.min_site_n      = min_site_n

    def analyze(
        self,
        y_true:      np.ndarray,
        y_prob:      np.ndarray,
        threshold:   Optional[float]         = None,
        subject_ids: Optional[Sequence[str]] = None,
        site_ids:    Optional[Sequence]      = None,
        metadata:    Optional[Any]           = None,   # pandas DataFrame
        out_dir:     Optional[Path]          = None,
    ) -> ErrorAnalysisReport:
        """
        Full error analysis pipeline.

        Parameters
        ----------
        y_true      : (N,) binary labels, 0=TC, 1=ASD
        y_prob      : (N,) P(ASD) ∈ [0,1]
        threshold   : decision threshold; if None, uses Youden's J
        subject_ids : (N,) subject identifiers for tracking
        site_ids    : (N,) acquisition site labels
        metadata    : optional DataFrame with columns age / sex / iq
        out_dir     : if given, save report and CSV of all errors
        """
        from evaluation.metrics import optimal_threshold_youden

        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        n      = len(y_true)

        if threshold is None:
            threshold = optimal_threshold_youden(y_true, y_prob)

        y_pred = (y_prob >= threshold).astype(int)

        # Default identifiers
        if subject_ids is None:
            subject_ids = [str(i) for i in range(n)]
        else:
            subject_ids = [str(s) for s in subject_ids]

        if site_ids is None:
            site_ids = ["unknown"] * n
        else:
            site_ids = [str(s) for s in site_ids]

        # Build per-subject records
        all_subjects = self._classify_subjects(
            y_true, y_prob, y_pred, threshold,
            subject_ids, site_ids, metadata,
        )

        report = ErrorAnalysisReport(
            n_total   = n,
            threshold = threshold,
            clinical_fn_weight = self.fn_weight,
            all_subjects = all_subjects,
        )

        self._fill_counts(report, all_subjects)
        self._fill_confidence_stats(report, all_subjects)
        self._identify_hard_examples(report, all_subjects)
        self._identify_uncertain(report, all_subjects)
        self._per_site_breakdown(report, all_subjects)
        self._detect_failure_modes(report, all_subjects, metadata)
        self._generate_recommendations(report, metadata)

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            report.save(out_dir / "error_analysis.json")
            self._save_error_csv(all_subjects, out_dir / "misclassified_subjects.csv")

        return report

    # ------------------------------------------------------------------
    # Step 1 — classify every subject
    # ------------------------------------------------------------------

    def _classify_subjects(
        self,
        y_true:      np.ndarray,
        y_prob:      np.ndarray,
        y_pred:      np.ndarray,
        threshold:   float,
        subject_ids: List[str],
        site_ids:    List[str],
        metadata,
    ) -> List[SubjectError]:
        records = []
        for i in range(len(y_true)):
            p    = float(y_prob[i])
            conf = max(p, 1.0 - p)          # ∈ [0.5, 1.0]
            margin = abs(p - threshold)

            true_lbl = int(y_true[i])
            pred_lbl = int(y_pred[i])

            if   true_lbl == 1 and pred_lbl == 1:  etype = "TP"
            elif true_lbl == 0 and pred_lbl == 0:  etype = "TN"
            elif true_lbl == 0 and pred_lbl == 1:  etype = "FP"
            else:                                   etype = "FN"  # true=ASD, pred=TC

            rec = SubjectError(
                subject_id = subject_ids[i],
                true_label = true_lbl,
                pred_label = pred_lbl,
                prob_asd   = p,
                error_type = etype,
                confidence = conf,
                margin     = margin,
                site_id    = site_ids[i],
            )

            # Attach metadata if available
            if metadata is not None:
                rec = self._attach_metadata(rec, i, metadata)

            records.append(rec)

        return records

    @staticmethod
    def _attach_metadata(rec: SubjectError, idx: int, metadata) -> SubjectError:
        """Extract age / sex / IQ from a pandas DataFrame row."""
        try:
            row = metadata.iloc[idx] if hasattr(metadata, "iloc") else {}
            if "age" in row:
                rec.age = float(row["age"]) if row["age"] == row["age"] else None
            if "sex" in row:
                rec.sex = str(row["sex"])
            for iq_col in ("full_2", "viq_1", "piq_1", "iq", "full_iq"):
                if iq_col in row and row[iq_col] == row[iq_col]:
                    rec.iq = float(row[iq_col])
                    break
        except Exception:
            pass
        return rec

    # ------------------------------------------------------------------
    # Step 2 — aggregate counts
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_counts(report: ErrorAnalysisReport,
                     subjects: List[SubjectError]) -> None:
        report.n_tp = sum(1 for s in subjects if s.error_type == "TP")
        report.n_tn = sum(1 for s in subjects if s.error_type == "TN")
        report.n_fp = sum(1 for s in subjects if s.error_type == "FP")
        report.n_fn = sum(1 for s in subjects if s.error_type == "FN")
        n = report.n_total

        report.error_rate = (report.n_fp + report.n_fn) / max(n, 1)

        # Clinical weighted cost: FN penalised more than FP
        report.weighted_error_cost = (
            report.n_fp * 1.0 + report.n_fn * report.clinical_fn_weight
        ) / max(n, 1)

        denom_fp = report.n_fp + report.n_tn
        denom_fn = report.n_fn + report.n_tp
        report.fp_rate = report.n_fp / max(denom_fp, 1)
        report.fn_rate = report.n_fn / max(denom_fn, 1)

    # ------------------------------------------------------------------
    # Step 3 — confidence statistics per outcome type
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_confidence_stats(report: ErrorAnalysisReport,
                               subjects: List[SubjectError]) -> None:
        def _stats(group: str):
            confs = [s.confidence for s in subjects if s.error_type == group]
            if not confs:
                return float("nan"), float("nan")
            return float(np.mean(confs)), float(np.std(confs, ddof=1) if len(confs) > 1 else 0.0)

        report.conf_tp_mean, _ = _stats("TP")
        report.conf_tn_mean, _ = _stats("TN")
        report.conf_fp_mean, report.conf_fp_std = _stats("FP")
        report.conf_fn_mean, report.conf_fn_std = _stats("FN")

    # ------------------------------------------------------------------
    # Step 4 — hard examples (high-confidence errors)
    # ------------------------------------------------------------------

    def _identify_hard_examples(self, report: ErrorAnalysisReport,
                                subjects: List[SubjectError]) -> None:
        """
        Hard FPs: TC subjects predicted as ASD with high confidence.
        Hard FNs: ASD subjects predicted as TC with high confidence.

        Clinical significance:
        - Hard FNs are the most dangerous: the model is confident the child
          is neurotypical when they actually have ASD — delaying diagnosis.
        - Hard FPs cause unnecessary referrals (less harmful but resource-intensive).
        """
        fps = [s for s in subjects if s.error_type == "FP"
               and s.confidence >= self.hard_threshold]
        fns = [s for s in subjects if s.error_type == "FN"
               and s.confidence >= self.hard_threshold]

        # Sort by confidence descending — most confident errors first
        report.hard_fp = sorted(fps, key=lambda s: s.confidence, reverse=True)
        report.hard_fn = sorted(fns, key=lambda s: s.confidence, reverse=True)

        if report.hard_fn:
            logger.warning(
                "Hard FNs: %d ASD subjects confidently misclassified as TC "
                "(confidence ≥ %.0f%%) — clinical risk.",
                len(report.hard_fn), self.hard_threshold * 100,
            )
        if report.hard_fp:
            logger.warning(
                "Hard FPs: %d TC subjects confidently misclassified as ASD "
                "(confidence ≥ %.0f%%).",
                len(report.hard_fp), self.hard_threshold * 100,
            )

    # ------------------------------------------------------------------
    # Step 5 — uncertain predictions (near decision boundary)
    # ------------------------------------------------------------------

    def _identify_uncertain(self, report: ErrorAnalysisReport,
                            subjects: List[SubjectError]) -> None:
        """
        Uncertain predictions: |P(ASD) − threshold| < unc_margin.
        These are the subjects where the model is close to flipping its
        decision; small changes in calibration or threshold would change them.
        """
        uncertain = [s for s in subjects if s.margin < self.unc_margin]
        report.uncertain = sorted(uncertain, key=lambda s: s.margin)

        n_err_unc = sum(1 for s in uncertain if s.is_error)
        logger.info(
            "Uncertain predictions: %d total, %d are errors (%.0f%%)",
            len(uncertain), n_err_unc,
            100 * n_err_unc / max(len(uncertain), 1),
        )

    # ------------------------------------------------------------------
    # Step 6 — per-site breakdown
    # ------------------------------------------------------------------

    def _per_site_breakdown(self, report: ErrorAnalysisReport,
                            subjects: List[SubjectError]) -> None:
        from collections import defaultdict

        site_groups: Dict[str, List[SubjectError]] = defaultdict(list)
        for s in subjects:
            site_groups[s.site_id].append(s)

        for site, group in site_groups.items():
            n      = len(group)
            if n < self.min_site_n:
                continue

            n_fp = sum(1 for s in group if s.error_type == "FP")
            n_fn = sum(1 for s in group if s.error_type == "FN")
            n_tp = sum(1 for s in group if s.error_type == "TP")
            n_tn = sum(1 for s in group if s.error_type == "TN")
            err_rate = (n_fp + n_fn) / max(n, 1)
            conf_err = [s.confidence for s in group if s.is_error]

            report.per_site_errors[site] = {
                "n":          n,
                "n_tp":       n_tp, "n_tn": n_tn,
                "n_fp":       n_fp, "n_fn": n_fn,
                "error_rate": round(err_rate, 4),
                "conf_error_mean": round(float(np.mean(conf_err)), 4)
                                   if conf_err else float("nan"),
            }

        # Flag sites with notably high error rates
        if report.per_site_errors:
            mean_err = np.mean([v["error_rate"]
                                for v in report.per_site_errors.values()])
            for site, stats in report.per_site_errors.items():
                if stats["error_rate"] > mean_err + 0.15:
                    logger.warning(
                        "Site '%s' has unusually high error rate: %.0f%% "
                        "(mean across sites: %.0f%%)",
                        site, stats["error_rate"] * 100, mean_err * 100,
                    )

    # ------------------------------------------------------------------
    # Step 7 — detect failure modes
    # ------------------------------------------------------------------

    def _detect_failure_modes(
        self,
        report:   ErrorAnalysisReport,
        subjects: List[SubjectError],
        metadata,
    ) -> None:
        """
        Automated pattern-matching for common ASD classifier failure modes.
        Each detected pattern is added as a human-readable string.
        """
        modes = []
        errors = [s for s in subjects if s.is_error]

        if not errors:
            report.failure_modes = ["No misclassifications — model performance is perfect on this set."]
            return

        total_err = len(errors)

        # --- 1. FN dominance (missed ASD) ---
        fn_ratio = report.n_fn / max(total_err, 1)
        if fn_ratio > 0.65:
            modes.append(
                f"FN dominance: {report.n_fn}/{total_err} errors ({fn_ratio:.0%}) are false "
                f"negatives (missed ASD diagnoses).  Model is biased toward predicting TC. "
                f"Causes: class imbalance, low ASD signal in features, threshold too high."
            )

        # --- 2. FP dominance (over-prediction) ---
        fp_ratio = report.n_fp / max(total_err, 1)
        if fp_ratio > 0.65:
            modes.append(
                f"FP dominance: {report.n_fp}/{total_err} errors ({fp_ratio:.0%}) are false "
                f"positives (TC predicted as ASD).  Model over-detects ASD. "
                f"Causes: Focal Loss γ too high, threshold too low."
            )

        # --- 3. High-confidence errors (hard failures) ---
        n_hard = len(report.hard_fp) + len(report.hard_fn)
        if n_hard > 0:
            modes.append(
                f"High-confidence errors: {n_hard} predictions are ≥{self.hard_threshold:.0%} "
                f"confident but wrong ({len(report.hard_fn)} FN, {len(report.hard_fp)} FP). "
                f"These subjects may have atypical presentations not captured in training."
            )

        # --- 4. Site-specific failure ---
        if report.per_site_errors:
            site_errs = {k: v["error_rate"] for k, v in report.per_site_errors.items()}
            worst_site = max(site_errs, key=site_errs.get)
            worst_rate = site_errs[worst_site]
            best_rate  = min(site_errs.values())
            if worst_rate - best_rate > 0.20:
                modes.append(
                    f"Site-specific failure: site '{worst_site}' has {worst_rate:.0%} error rate "
                    f"vs best-site {best_rate:.0%}. "
                    f"Causes: scanner differences, acquisition protocol variation, "
                    f"inadequate ComBat harmonisation."
                )

        # --- 5. Uncertain boundary errors ---
        n_unc_err = sum(1 for s in subjects if s.is_error and s.margin < self.unc_margin)
        n_unc_all = len(report.uncertain)
        if n_unc_all > 0 and n_unc_err / max(n_unc_all, 1) > 0.5:
            modes.append(
                f"Decision boundary instability: {n_unc_err}/{n_unc_all} subjects near the "
                f"threshold (|P − τ| < {self.unc_margin:.2f}) are misclassified. "
                f"Temperature scaling or threshold calibration may improve performance."
            )

        # --- 6. Age / IQ bias (if metadata available) ---
        age_err, age_ok, iq_err, iq_ok = [], [], [], []
        for s in subjects:
            if s.age is not None:
                (age_err if s.is_error else age_ok).append(s.age)
            if s.iq is not None:
                (iq_err if s.is_error else iq_ok).append(s.iq)

        if age_err and age_ok and len(age_err) >= 5:
            from scipy import stats as scipy_stats
            t_stat, p_age = scipy_stats.ttest_ind(age_err, age_ok)
            if p_age < 0.05:
                modes.append(
                    f"Age bias: misclassified subjects are on average "
                    f"{np.mean(age_err):.1f} years old vs. {np.mean(age_ok):.1f} for correct "
                    f"predictions (t={t_stat:.2f}, p={p_age:.3f}).  "
                    f"Model may generalise poorly to a certain age range."
                )

        if iq_err and iq_ok and len(iq_err) >= 5:
            from scipy import stats as scipy_stats
            t_stat, p_iq = scipy_stats.ttest_ind(iq_err, iq_ok)
            if p_iq < 0.05:
                modes.append(
                    f"IQ bias: misclassified subjects differ significantly in IQ from correct "
                    f"predictions ({np.mean(iq_err):.0f} vs {np.mean(iq_ok):.0f}, "
                    f"t={t_stat:.2f}, p={p_iq:.3f})."
                )

        # --- 7. Sex imbalance in errors ---
        sex_err = [s.sex for s in errors if s.sex is not None]
        if sex_err:
            female_err = sex_err.count("F") / max(len(sex_err), 1)
            all_sex = [s.sex for s in subjects if s.sex is not None]
            female_all = all_sex.count("F") / max(len(all_sex), 1) if all_sex else 0.5
            if abs(female_err - female_all) > 0.15:
                modes.append(
                    f"Sex bias in errors: {female_err:.0%} of errors involve female subjects "
                    f"vs {female_all:.0%} overall (>{15:.0f}pp difference).  "
                    f"ASD presents differently in females — consider sex-stratified evaluation."
                )

        if not modes:
            modes.append(
                "No dominant failure mode detected.  Errors appear uniformly distributed "
                "across sites, confidence levels, and demographics."
            )

        report.failure_modes = modes

    # ------------------------------------------------------------------
    # Step 8 — ranked recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self, report: ErrorAnalysisReport, metadata
    ) -> None:
        recs: List[str] = []

        # Priority 1 — high clinical risk (hard FNs)
        if report.hard_fn:
            recs.append(
                f"[P1 — Clinical Risk] Investigate {len(report.hard_fn)} hard FNs: "
                f"ASD subjects confidently misclassified as TC.  "
                f"Review their MRI and genetic profiles for unusual presentations.  "
                f"Consider data augmentation targeted at these subjects."
            )

        # Priority 2 — model calibration (high confidence in errors overall)
        all_err_conf = [s.confidence for s in report.all_subjects if s.is_error]
        if all_err_conf and np.mean(all_err_conf) > 0.70:
            recs.append(
                f"[P2 — Calibration] Mean confidence in errors = {np.mean(all_err_conf):.3f}. "
                f"Apply temperature scaling or Platt scaling post-training to reduce overconfidence.  "
                f"Ensemble methods (MC Dropout, Deep Ensembles) also reduce miscalibration."
            )

        # Priority 3 — site failure
        if report.per_site_errors:
            site_errs = {k: v["error_rate"] for k, v in report.per_site_errors.items()}
            worst_site = max(site_errs, key=site_errs.get)
            if site_errs[worst_site] > 0.3:
                recs.append(
                    f"[P3 — Site Generalisation] Site '{worst_site}' has {site_errs[worst_site]:.0%} "
                    f"error rate.  Apply site-specific ComBat harmonisation, or add site as a "
                    f"conditioning variable in the model (site-adaptive normalisation)."
                )

        # Priority 4 — threshold adjustment
        fn_ratio = report.n_fn / max(report.n_fn + report.n_fp, 1)
        if fn_ratio > 0.60:
            recs.append(
                f"[P4 — Threshold] {fn_ratio:.0%} of errors are FNs.  "
                f"Lower the decision threshold (currently {report.threshold:.3f}) to improve "
                f"sensitivity at the cost of specificity — appropriate for a screening tool "
                f"where missed ASD diagnoses are more costly than false alarms."
            )
        elif fn_ratio < 0.40 and report.n_fp > 0:
            recs.append(
                f"[P4 — Threshold] {1-fn_ratio:.0%} of errors are FPs.  "
                f"Raise the decision threshold to reduce over-referral."
            )

        # Priority 5 — uncertain predictions
        if len(report.uncertain) > report.n_total * 0.15:
            recs.append(
                f"[P5 — Uncertainty] {len(report.uncertain)}/{report.n_total} "
                f"({len(report.uncertain)/report.n_total:.0%}) predictions are near the "
                f"decision boundary.  Consider adding uncertainty quantification "
                f"(MC Dropout, evidential deep learning) and routing uncertain cases "
                f"for expert review."
            )

        if not recs:
            recs.append(
                "No high-priority interventions identified.  "
                "Consider ablation studies to understand which modality drives performance."
            )

        report.recommendations = recs

    # ------------------------------------------------------------------
    # Utility: save misclassified subjects to CSV
    # ------------------------------------------------------------------

    @staticmethod
    def _save_error_csv(subjects: List[SubjectError], path: Path) -> None:
        """Save all misclassified subjects to a CSV for manual inspection."""
        try:
            import pandas as pd
            rows = [s.to_dict() for s in subjects if s.is_error]
            if rows:
                df = pd.DataFrame(rows)
                df = df.sort_values("confidence", ascending=False)
                df.to_csv(path, index=False)
                logger.info("Misclassified subjects CSV → %s (%d rows)", path, len(df))
            else:
                logger.info("No misclassifications — CSV not written.")
        except ImportError:
            logger.debug("pandas not available — skipping CSV export.")
        except Exception as exc:
            logger.warning("Failed to save error CSV: %s", exc)
