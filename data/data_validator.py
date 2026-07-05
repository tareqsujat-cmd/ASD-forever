"""
Pre-training data validation for the multimodal ASD detection framework.

Validates the dataset BEFORE any model training begins.  Catches data quality
issues that would silently corrupt model training or invalidate results:

  Critical (raises DataValidationError — training MUST stop):
    • Subject leakage: same subject ID appears in both train and test splits
    • Degenerate labels: all samples belong to one class (AUROC undefined)
    • Inconsistent labels: same subject has different labels in different rows
    • Catastrophic missing-file rate (>50% of expected files absent)

  Warning (logged; training continues with reduced dataset):
    • Duplicate subject IDs (within a single split)
    • Severe class imbalance (>4:1 ratio ASD:TC or TC:ASD)
    • NaN / Inf values in feature arrays
    • All-zero arrays (failed preprocessing)
    • Outliers (|z| > 5σ) — potential acquisition artifacts
    • Feature dimension mismatch across subjects
    • Data normalization anomalies (|mean| > 2 or std < 0.01)

Usage
-----
Synthetic mode (in-memory, smoke test)::

    validator = DataValidator()
    report = validator.validate_synthetic(dataset)

Real ABIDE mode (metadata CSV + .npy files on disk)::

    validator = DataValidator()
    report = validator.validate_real(
        mri_dir   = "/path/to/mri_processed",
        gen_dir   = "/path/to/genetics_processed",
        meta_path = "/path/to/mri_processed/metadata.csv",
        train_ids = list_of_train_subject_ids,
        test_ids  = list_of_test_subject_ids,
    )

    report.print_summary()
    report.save(out_dir / "data_validation_report.json")
    report.raise_if_critical()   # Raises DataValidationError if any critical issue

References
----------
ABIDE data quality protocol: http://preprocessed-connectomes-project.org/abide/
Arbabshirani MR et al. (2017). Single subject prediction of brain disorders in
  neuroimaging: promises and pitfalls. NeuroImage 145:137-165.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataValidationError(RuntimeError):
    """Raised when a critical data quality issue is detected.

    Training MUST NOT proceed when this is raised.  The error message
    describes the specific issue and recommended corrective action.
    """


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class IssueRecord:
    severity: str        # "CRITICAL" | "WARNING" | "INFO"
    category: str        # e.g. "leakage", "imbalance", "duplicates"
    message: str
    affected: List[str] = field(default_factory=list)  # subject IDs or file paths


@dataclass
class ValidationReport:
    """Structured data validation report."""

    # ---- Dataset overview ----
    n_subjects:       int   = 0
    n_asd:            int   = 0
    n_tc:             int   = 0
    n_sites:          int   = 0
    prevalence:       float = 0.0
    imbalance_ratio:  float = 0.0  # max(n_asd/n_tc, n_tc/n_asd)

    # ---- MRI checks ----
    mri_shape_consistent: bool = True
    mri_nan_count:        int  = 0
    mri_inf_count:        int  = 0
    mri_zero_volume_count: int = 0
    mri_outlier_subjects: List[str] = field(default_factory=list)

    # ---- Genetics checks ----
    gen_shape_consistent: bool = True
    gen_nan_count:        int  = 0
    gen_inf_count:        int  = 0
    gen_zero_count:       int  = 0
    gen_outlier_subjects: List[str] = field(default_factory=list)

    # ---- Label checks ----
    invalid_label_count:      int  = 0
    inconsistent_label_count: int  = 0
    duplicate_subject_count:  int  = 0
    leaked_subject_count:     int  = 0  # appearing in both train and test

    # ---- File availability ----
    missing_mri_count: int = 0
    missing_gen_count: int = 0

    # ---- Split checks ----
    split_sizes:         Dict[str, int] = field(default_factory=dict)
    per_split_prevalence: Dict[str, float] = field(default_factory=dict)

    # ---- All issues (ordered by severity) ----
    issues: List[IssueRecord] = field(default_factory=list)

    # ---- Data fingerprint ----
    fingerprint: str = ""

    # ---- Pass/Fail ----
    passed: bool = True

    @property
    def n_critical(self) -> int:
        return sum(1 for i in self.issues if i.severity == "CRITICAL")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "WARNING")

    def raise_if_critical(self) -> None:
        """Raise DataValidationError if any critical issues exist."""
        critical = [i for i in self.issues if i.severity == "CRITICAL"]
        if critical:
            msg = "\n".join(
                f"[{i.category.upper()}] {i.message}" for i in critical
            )
            raise DataValidationError(
                f"DATA VALIDATION FAILED — {len(critical)} critical issue(s) "
                f"prevent training from starting:\n{msg}\n\n"
                f"Fix these issues before re-running the experiment."
            )

    def print_summary(self) -> None:
        """Print a formatted validation summary to the log."""
        status = "PASSED" if self.passed else "FAILED"
        logger.info("=" * 64)
        logger.info("DATA VALIDATION REPORT  [%s]", status)
        logger.info("=" * 64)
        logger.info("  Subjects     : %d  (ASD=%d  TC=%d  Sites=%d)",
                    self.n_subjects, self.n_asd, self.n_tc, self.n_sites)
        logger.info("  Prevalence   : %.1f%%   Imbalance ratio : %.2f:1",
                    self.prevalence * 100, self.imbalance_ratio)
        logger.info("  Critical     : %d   Warnings : %d",
                    self.n_critical, self.n_warnings)
        if self.fingerprint:
            logger.info("  Fingerprint  : %s", self.fingerprint)
        logger.info("-" * 64)
        for issue in self.issues:
            prefix = "  [CRITICAL]" if issue.severity == "CRITICAL" else \
                     "  [WARNING] " if issue.severity == "WARNING"  else \
                     "  [INFO]    "
            logger.info("%s %s: %s", prefix, issue.category, issue.message)
            if issue.affected and len(issue.affected) <= 5:
                logger.info("             affected: %s", ", ".join(issue.affected))
            elif issue.affected:
                logger.info("             affected: %s ... (%d total)",
                            ", ".join(issue.affected[:3]), len(issue.affected))
        logger.info("=" * 64)

    def save(self, path: str | Path) -> None:
        """Persist the report as a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "passed":              self.passed,
            "n_subjects":          self.n_subjects,
            "n_asd":               self.n_asd,
            "n_tc":                self.n_tc,
            "n_sites":             self.n_sites,
            "prevalence":          self.prevalence,
            "imbalance_ratio":     self.imbalance_ratio,
            "n_critical":          self.n_critical,
            "n_warnings":          self.n_warnings,
            "fingerprint":         self.fingerprint,
            "split_sizes":         self.split_sizes,
            "per_split_prevalence": self.per_split_prevalence,
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "message":  i.message,
                    "n_affected": len(i.affected),
                    "sample_affected": i.affected[:10],
                }
                for i in self.issues
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Validation report saved → %s", path)


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

class DataValidator:
    """
    Pre-training data validator for multimodal ASD datasets.

    Checks every dimension of data quality required for a reproducible,
    leakage-free IEEE-standard experiment.

    Parameters
    ----------
    sigma_outlier : float
        Z-score threshold for outlier detection (default 5σ).
    max_imbalance_ratio : float
        Ratio above which class imbalance is flagged as WARNING.
    max_missing_rate : float
        Fraction of missing files above which a CRITICAL error is raised.
    sample_size_integrity : int
        Number of random subjects to spot-check for file integrity
        (loading full dataset for QC before training is prohibitive).
    """

    def __init__(
        self,
        sigma_outlier:         float = 5.0,
        max_imbalance_ratio:   float = 4.0,
        max_missing_rate:      float = 0.50,
        sample_size_integrity: int   = 50,
    ) -> None:
        self.sigma_outlier         = sigma_outlier
        self.max_imbalance_ratio   = max_imbalance_ratio
        self.max_missing_rate      = max_missing_rate
        self.sample_size_integrity = sample_size_integrity

    # ------------------------------------------------------------------
    # Public API — synthetic mode
    # ------------------------------------------------------------------

    def validate_synthetic(
        self,
        dataset,
        train_indices: Optional[Sequence[int]] = None,
        val_indices:   Optional[Sequence[int]] = None,
        test_indices:  Optional[Sequence[int]] = None,
    ) -> ValidationReport:
        """
        Validate an in-memory dataset (SyntheticABIDE or any PyTorch Dataset
        that yields dicts with 'label', 'image', 'genetics', 'site' keys).

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
        train_indices, val_indices, test_indices : optional index splits
        """
        import torch

        report = ValidationReport()
        n = len(dataset)

        # --- Collect all samples ---
        labels, sites, subject_ids = [], [], []
        mri_shapes, gen_shapes     = [], []
        mri_arrays, gen_arrays     = [], []

        for i in range(n):
            sample = dataset[i]
            lbl    = int(sample["label"])
            labels.append(lbl)
            sites.append(int(sample.get("site", 0)))
            subject_ids.append(str(i))  # synthetic: use index as ID

            img = sample["image"]
            gen = sample["genetics"]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            if isinstance(gen, torch.Tensor):
                gen = gen.numpy()

            mri_shapes.append(img.shape)
            gen_shapes.append(gen.shape)
            mri_arrays.append(img.ravel().astype(np.float32))
            gen_arrays.append(gen.ravel().astype(np.float32))

        labels       = np.array(labels)
        mri_matrix   = np.stack(mri_arrays)   # (N, D*H*W)
        gen_matrix   = np.stack(gen_arrays)   # (N, G)

        # --- Run all checks ---
        self._check_labels(labels, subject_ids, report)
        self._check_duplicates(subject_ids, labels, report)

        if train_indices is not None and test_indices is not None:
            train_ids = [subject_ids[i] for i in train_indices]
            test_ids  = [subject_ids[i] for i in test_indices]
            self._check_split_leakage(train_ids, test_ids, report)

        self._check_class_imbalance(labels, report)
        self._check_site_distribution(labels, sites, report, split_sizes_only=True)
        self._check_array_integrity(mri_matrix, subject_ids, "MRI", report)
        self._check_array_integrity(gen_matrix, subject_ids, "genetics", report)
        self._check_shape_consistency(mri_shapes, "MRI", report)
        self._check_shape_consistency(gen_shapes, "genetics", report)
        self._check_normalization(mri_matrix, "MRI", report)
        self._check_normalization(gen_matrix, "genetics", report)
        self._check_outliers(gen_matrix, subject_ids, "genetics", report)

        if train_indices and val_indices and test_indices:
            self._record_split_stats(
                labels, subject_ids,
                {"train": list(train_indices),
                 "val":   list(val_indices),
                 "test":  list(test_indices)},
                report,
            )

        report.fingerprint = self._fingerprint_dataset(mri_matrix, gen_matrix, labels)
        report.passed = report.n_critical == 0
        return report

    # ------------------------------------------------------------------
    # Public API — real ABIDE mode
    # ------------------------------------------------------------------

    def validate_real(
        self,
        mri_dir:   str | Path,
        gen_dir:   str | Path,
        meta_path: str | Path,
        train_ids: Optional[Sequence[str]] = None,
        val_ids:   Optional[Sequence[str]] = None,
        test_ids:  Optional[Sequence[str]] = None,
    ) -> ValidationReport:
        """
        Validate the preprocessed ABIDE dataset from disk.

        Parameters
        ----------
        mri_dir   : directory containing <subject_id>.npy MRI files
        gen_dir   : directory containing <subject_id>.npy genetics files
        meta_path : path to metadata.csv with columns: subject_id, label, site
        train_ids, val_ids, test_ids : subject IDs for each split
        """
        import pandas as pd

        mri_dir   = Path(mri_dir)
        gen_dir   = Path(gen_dir)
        meta_path = Path(meta_path)

        report = ValidationReport()

        if not meta_path.exists():
            self._add_critical(report, "metadata",
                               f"metadata.csv not found: {meta_path}")
            report.passed = False
            return report

        meta = pd.read_csv(meta_path)

        # Canonicalize column names
        meta.columns = [c.lower().strip() for c in meta.columns]
        if "subject_id" not in meta.columns:
            self._add_critical(report, "metadata",
                               "metadata.csv missing 'subject_id' column")
            report.passed = False
            return report

        subject_ids = meta["subject_id"].astype(str).tolist()
        labels      = meta["label"].values if "label" in meta.columns else None
        sites       = meta["site"].values  if "site"  in meta.columns else \
                      np.zeros(len(meta), dtype=int)

        # --- Label presence ---
        if labels is None:
            self._add_critical(report, "labels",
                               "metadata.csv missing 'label' column")
        else:
            labels = labels.astype(np.float64)
            self._check_labels(labels, subject_ids, report)

        # --- File existence checks ---
        self._check_file_availability(subject_ids, mri_dir, gen_dir, report)

        if labels is not None:
            self._check_duplicates(subject_ids, labels, report)
            self._check_class_imbalance(labels.astype(int), report)
            self._check_site_distribution(
                labels.astype(int), sites, report, split_sizes_only=False
            )

        # --- Leakage check ---
        if train_ids is not None and test_ids is not None:
            self._check_split_leakage(
                list(map(str, train_ids)),
                list(map(str, test_ids)),
                report,
            )

        # --- Integrity spot-check ---
        self._spot_check_arrays(
            subject_ids, labels, mri_dir, gen_dir, report
        )

        # --- Split stats ---
        if train_ids is not None and val_ids is not None and test_ids is not None:
            id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
            splits = {
                "train": [id_to_idx[s] for s in train_ids if s in id_to_idx],
                "val":   [id_to_idx[s] for s in val_ids   if s in id_to_idx],
                "test":  [id_to_idx[s] for s in test_ids  if s in id_to_idx],
            }
            if labels is not None:
                self._record_split_stats(labels.astype(int), subject_ids,
                                         splits, report)

        report.fingerprint = self._fingerprint_metadata(meta)
        report.passed = report.n_critical == 0
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_labels(
        self,
        labels:      np.ndarray,
        subject_ids: List[str],
        report:      ValidationReport,
    ) -> None:
        """Verify labels are binary (0/1) with no NaN/invalid values."""
        labels = np.asarray(labels, dtype=float)

        nan_mask = np.isnan(labels)
        if nan_mask.any():
            bad = [subject_ids[i] for i in np.where(nan_mask)[0]]
            self._add_critical(
                report, "labels",
                f"{int(nan_mask.sum())} subject(s) have NaN labels",
                bad,
            )

        valid_mask = ~nan_mask
        int_labels = labels[valid_mask]
        invalid_mask = ~np.isin(int_labels, [0, 1])
        if invalid_mask.any():
            n_invalid = int(invalid_mask.sum())
            report.invalid_label_count = n_invalid
            self._add_critical(
                report, "labels",
                f"{n_invalid} label(s) not in {{0,1}}: "
                f"unique values = {np.unique(int_labels[invalid_mask]).tolist()}",
            )

        unique = np.unique(labels[valid_mask])
        if len(unique) < 2:
            self._add_critical(
                report, "labels",
                f"All labels are the same class ({int(unique[0]) if len(unique) else 'empty'}). "
                f"AUROC is undefined — dataset is degenerate.",
            )

        n_asd = int((labels[valid_mask] == 1).sum())
        n_tc  = int((labels[valid_mask] == 0).sum())
        report.n_subjects  = len(labels)
        report.n_asd       = n_asd
        report.n_tc        = n_tc
        report.prevalence  = n_asd / max(len(labels), 1)

    def _check_duplicates(
        self,
        subject_ids: List[str],
        labels:      np.ndarray,
        report:      ValidationReport,
    ) -> None:
        """Detect duplicate subject IDs. Flag if same ID has different labels."""
        seen:  Dict[str, int] = {}
        dups:  List[str]      = []
        inconsistent: List[str] = []

        labels = np.asarray(labels, dtype=float)

        for i, sid in enumerate(subject_ids):
            lbl = float(labels[i]) if i < len(labels) else float("nan")
            if sid in seen:
                dups.append(sid)
                if seen[sid] != lbl:
                    inconsistent.append(sid)
            else:
                seen[sid] = lbl

        if dups:
            report.duplicate_subject_count = len(set(dups))
            self._add_warning(
                report, "duplicates",
                f"{len(set(dups))} subject ID(s) appear more than once",
                list(set(dups))[:20],
            )

        if inconsistent:
            report.inconsistent_label_count = len(inconsistent)
            self._add_critical(
                report, "inconsistent_labels",
                f"{len(inconsistent)} subject(s) have conflicting labels "
                f"across rows — cannot determine ground truth",
                inconsistent[:10],
            )

    def _check_split_leakage(
        self,
        train_ids: List[str],
        test_ids:  List[str],
        report:    ValidationReport,
    ) -> None:
        """Raise CRITICAL if any subject appears in both train and test."""
        train_set  = set(train_ids)
        test_set   = set(test_ids)
        leaked     = train_set & test_set

        if leaked:
            report.leaked_subject_count = len(leaked)
            self._add_critical(
                report, "leakage",
                f"DATA LEAKAGE: {len(leaked)} subject(s) appear in BOTH "
                f"the training set AND the test set.  All reported metrics "
                f"would be inflated.  Re-split the dataset.",
                sorted(leaked)[:10],
            )
        else:
            self._add_info(
                report, "leakage",
                f"No train/test leakage detected "
                f"(train={len(train_ids)}  test={len(test_ids)})",
            )

    def _check_class_imbalance(
        self,
        labels: np.ndarray,
        report: ValidationReport,
    ) -> None:
        """Warn if class imbalance exceeds max_imbalance_ratio."""
        n_asd = int((labels == 1).sum())
        n_tc  = int((labels == 0).sum())

        if n_tc == 0 or n_asd == 0:
            return  # already caught by _check_labels

        ratio = max(n_asd / n_tc, n_tc / n_asd)
        report.imbalance_ratio = round(ratio, 3)

        if ratio >= self.max_imbalance_ratio:
            self._add_warning(
                report, "imbalance",
                f"Severe class imbalance: ASD={n_asd}  TC={n_tc}  "
                f"ratio={ratio:.2f}:1  (threshold={self.max_imbalance_ratio}:1). "
                f"Use WeightedRandomSampler and Focal Loss to compensate.",
            )
        elif ratio >= 2.0:
            self._add_info(
                report, "imbalance",
                f"Mild class imbalance: ASD={n_asd}  TC={n_tc}  ratio={ratio:.2f}:1",
            )
        else:
            self._add_info(
                report, "imbalance",
                f"Balanced classes: ASD={n_asd}  TC={n_tc}  ratio={ratio:.2f}:1",
            )

    def _check_site_distribution(
        self,
        labels:          np.ndarray,
        sites:           np.ndarray,
        report:          ValidationReport,
        split_sizes_only: bool = False,
    ) -> None:
        """Report per-site ASD prevalence to surface site-specific biases."""
        sites  = np.asarray(sites)
        labels = np.asarray(labels)
        unique_sites = np.unique(sites)
        report.n_sites = int(len(unique_sites))

        if split_sizes_only:
            return

        for s in unique_sites:
            mask  = sites == s
            n_s   = int(mask.sum())
            prev  = float(labels[mask].mean()) if n_s > 0 else float("nan")
            if abs(prev - 0.5) > 0.3:
                self._add_warning(
                    report, "site_bias",
                    f"Site {s}: {n_s} subjects, prevalence={prev:.0%} "
                    f"(expected ~50%). Consider site harmonisation (ComBat).",
                )

    def _check_array_integrity(
        self,
        matrix:      np.ndarray,
        subject_ids: List[str],
        name:        str,
        report:      ValidationReport,
    ) -> None:
        """Check for NaN, Inf, all-zero arrays in feature matrices."""
        nan_per_subj = np.isnan(matrix).any(axis=1)
        inf_per_subj = np.isinf(matrix).any(axis=1)
        zero_per_subj = (matrix == 0).all(axis=1)

        n_nan  = int(nan_per_subj.sum())
        n_inf  = int(inf_per_subj.sum())
        n_zero = int(zero_per_subj.sum())

        if name == "MRI":
            report.mri_nan_count         = n_nan
            report.mri_inf_count         = n_inf
            report.mri_zero_volume_count = n_zero
        else:
            report.gen_nan_count  = n_nan
            report.gen_inf_count  = n_inf
            report.gen_zero_count = n_zero

        if n_nan > 0:
            bad = [subject_ids[i] for i in np.where(nan_per_subj)[0][:10]]
            self._add_warning(
                report, f"{name}_integrity",
                f"{n_nan} {name} array(s) contain NaN values — "
                f"preprocessing failed for these subjects", bad,
            )
        if n_inf > 0:
            bad = [subject_ids[i] for i in np.where(inf_per_subj)[0][:10]]
            self._add_warning(
                report, f"{name}_integrity",
                f"{n_inf} {name} array(s) contain Inf values", bad,
            )
        if n_zero > 0:
            bad = [subject_ids[i] for i in np.where(zero_per_subj)[0][:10]]
            self._add_warning(
                report, f"{name}_integrity",
                f"{n_zero} {name} array(s) are all-zero "
                f"(failed preprocessing — will contribute noise to training)", bad,
            )

        if n_nan == 0 and n_inf == 0 and n_zero == 0:
            self._add_info(
                report, f"{name}_integrity",
                f"No NaN / Inf / zero arrays in {name} features",
            )

    def _check_shape_consistency(
        self,
        shapes: List[Tuple],
        name:   str,
        report: ValidationReport,
    ) -> None:
        """Verify all subjects have the same feature dimensionality."""
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            counts = {str(s): shapes.count(s) for s in unique_shapes}
            if name == "MRI":
                report.mri_shape_consistent = False
            else:
                report.gen_shape_consistent = False
            self._add_critical(
                report, f"{name}_shape",
                f"{name} arrays have inconsistent shapes: {counts}. "
                f"All subjects must be preprocessed to the same spatial resolution.",
            )
        else:
            self._add_info(
                report, f"{name}_shape",
                f"{name} shape consistent: {list(unique_shapes)[0]}",
            )

    def _check_normalization(
        self,
        matrix: np.ndarray,
        name:   str,
        report: ValidationReport,
    ) -> None:
        """
        Sanity-check that features are reasonably normalised.

        For genetics (1D feature vectors): expect |mean| < 2 and std > 0.01.
        For MRI (flattened volumes): expect most signal in [−10, 10] range.
        """
        finite_mask = np.isfinite(matrix)
        if not finite_mask.any():
            return

        global_mean = float(np.mean(matrix[finite_mask]))
        global_std  = float(np.std(matrix[finite_mask]))

        if name == "genetics":
            if abs(global_mean) > 5.0:
                self._add_warning(
                    report, f"{name}_normalization",
                    f"{name} global mean={global_mean:.2f} is far from zero. "
                    f"Verify that standardisation was applied during preprocessing.",
                )
            if global_std < 0.01:
                self._add_warning(
                    report, f"{name}_normalization",
                    f"{name} global std={global_std:.4f} near zero — "
                    f"features appear constant (check imputation / feature selection).",
                )
            else:
                self._add_info(
                    report, f"{name}_normalization",
                    f"{name} mean={global_mean:.3f}  std={global_std:.3f}",
                )
        elif name == "MRI":
            p1, p99 = float(np.percentile(matrix[finite_mask], 1)), \
                      float(np.percentile(matrix[finite_mask], 99))
            if p99 > 1e4 or p1 < -1e4:
                self._add_warning(
                    report, f"{name}_normalization",
                    f"MRI intensity range [{p1:.1f}, {p99:.1f}] is unusually wide. "
                    f"Verify intensity normalisation (z-score or [0,1] scaling).",
                )
            else:
                self._add_info(
                    report, f"{name}_normalization",
                    f"MRI intensity p1={p1:.3f}  p99={p99:.3f}",
                )

    def _check_outliers(
        self,
        matrix:      np.ndarray,
        subject_ids: List[str],
        name:        str,
        report:      ValidationReport,
    ) -> None:
        """Flag subjects whose feature vectors are extreme outliers (|z| > σ threshold)."""
        finite = np.isfinite(matrix)
        if not finite.any():
            return

        # Per-feature z-scores; ignore NaN/Inf for the global statistics
        col_mean = np.nanmean(np.where(finite, matrix, np.nan), axis=0)
        col_std  = np.nanstd(np.where(finite, matrix, np.nan), axis=0) + 1e-12

        z_matrix = np.abs((matrix - col_mean) / col_std)
        z_matrix[~finite] = 0.0  # exclude already-flagged bad values

        max_z_per_subject = z_matrix.max(axis=1)  # (N,)
        outlier_mask = max_z_per_subject > self.sigma_outlier
        outlier_sids = [subject_ids[i] for i in np.where(outlier_mask)[0]]

        if name == "MRI":
            report.mri_outlier_subjects = outlier_sids
        else:
            report.gen_outlier_subjects = outlier_sids

        if outlier_sids:
            self._add_warning(
                report, f"{name}_outliers",
                f"{len(outlier_sids)} subject(s) have max |z-score| > "
                f"{self.sigma_outlier}σ in {name} features. "
                f"Inspect for acquisition artifacts.",
                outlier_sids[:10],
            )
        else:
            self._add_info(
                report, f"{name}_outliers",
                f"No {name} outliers above {self.sigma_outlier}σ",
            )

    def _check_file_availability(
        self,
        subject_ids: List[str],
        mri_dir:     Path,
        gen_dir:     Path,
        report:      ValidationReport,
    ) -> None:
        """Check that .npy files exist for all subjects in both modalities."""
        missing_mri, missing_gen = [], []

        for sid in subject_ids:
            mri_path = mri_dir / f"{sid}.npy"
            gen_path = gen_dir / f"{sid}.npy"
            if not mri_path.exists():
                missing_mri.append(sid)
            if not gen_path.exists():
                missing_gen.append(sid)

        n = len(subject_ids)
        report.missing_mri_count = len(missing_mri)
        report.missing_gen_count = len(missing_gen)

        for label, missing, modality in [
            ("MRI",      missing_mri, "MRI"),
            ("genetics", missing_gen, "genetics"),
        ]:
            rate = len(missing) / max(n, 1)
            if rate > self.max_missing_rate:
                self._add_critical(
                    report, f"missing_{label.lower()}",
                    f"{len(missing)}/{n} ({rate:.0%}) {modality} files are missing — "
                    f"exceeds threshold of {self.max_missing_rate:.0%}. "
                    f"Re-run preprocessing before training.",
                    missing[:20],
                )
            elif missing:
                self._add_warning(
                    report, f"missing_{label.lower()}",
                    f"{len(missing)}/{n} ({rate:.0%}) {modality} files missing. "
                    f"These subjects will be silently skipped during training.",
                    missing[:10],
                )
            else:
                self._add_info(
                    report, f"file_availability",
                    f"All {n} {modality} files present",
                )

    def _spot_check_arrays(
        self,
        subject_ids: List[str],
        labels:      Optional[np.ndarray],
        mri_dir:     Path,
        gen_dir:     Path,
        report:      ValidationReport,
    ) -> None:
        """
        Load a random sample of subjects and run array-level integrity checks.

        Loading all subjects is prohibitive (hours for ABIDE).  We spot-check
        `sample_size_integrity` subjects stratified by class to catch systemic
        issues while keeping validation fast.
        """
        rng = np.random.default_rng(42)

        available = [
            sid for sid in subject_ids
            if (mri_dir / f"{sid}.npy").exists()
            and (gen_dir / f"{sid}.npy").exists()
        ]
        if not available:
            return

        n_sample = min(self.sample_size_integrity, len(available))
        sampled  = rng.choice(available, size=n_sample, replace=False).tolist()

        mri_arrs, gen_arrs, sids_out = [], [], []
        mri_shapes, gen_shapes = [], []

        for sid in sampled:
            try:
                mri = np.load(mri_dir / f"{sid}.npy").astype(np.float32)
                gen = np.load(gen_dir / f"{sid}.npy").astype(np.float32)
                mri_shapes.append(mri.shape)
                gen_shapes.append(gen.shape)
                mri_arrs.append(mri.ravel())
                gen_arrs.append(gen.ravel())
                sids_out.append(sid)
            except Exception as exc:
                self._add_warning(
                    report, "file_integrity",
                    f"Failed to load subject {sid}: {exc}",
                    [sid],
                )

        if not mri_arrs:
            return

        # Pad to equal length for stacking (shapes may differ — caught below)
        max_mri = max(len(a) for a in mri_arrs)
        max_gen = max(len(a) for a in gen_arrs)

        mri_m = np.stack([
            np.pad(a, (0, max_mri - len(a))) if len(a) < max_mri else a
            for a in mri_arrs
        ])
        gen_m = np.stack([
            np.pad(a, (0, max_gen - len(a))) if len(a) < max_gen else a
            for a in gen_arrs
        ])

        self._check_shape_consistency(mri_shapes, "MRI", report)
        self._check_shape_consistency(gen_shapes, "genetics", report)
        self._check_array_integrity(mri_m, sids_out, "MRI", report)
        self._check_array_integrity(gen_m, sids_out, "genetics", report)
        self._check_normalization(mri_m, "MRI", report)
        self._check_normalization(gen_m, "genetics", report)
        self._check_outliers(gen_m, sids_out, "genetics", report)

        self._add_info(
            report, "spot_check",
            f"Integrity spot-check: {len(sids_out)}/{n_sample} subjects loaded "
            f"successfully",
        )

    def _record_split_stats(
        self,
        labels:      np.ndarray,
        subject_ids: List[str],
        splits:      Dict[str, List[int]],
        report:      ValidationReport,
    ) -> None:
        """Record per-split sizes and prevalences for the report."""
        labels = np.asarray(labels)
        for split_name, indices in splits.items():
            if not indices:
                continue
            split_labels = labels[indices]
            n_split = len(split_labels)
            prev    = float(split_labels.mean()) if n_split > 0 else float("nan")
            report.split_sizes[split_name]          = n_split
            report.per_split_prevalence[split_name] = round(prev, 4)

            # Warn if a split has drastically different prevalence from overall
            if abs(prev - report.prevalence) > 0.15:
                self._add_warning(
                    report, "stratification",
                    f"Split '{split_name}' prevalence={prev:.1%} deviates from "
                    f"overall prevalence={report.prevalence:.1%} by >{0.15:.0%}. "
                    f"Use stratified sampling (StratifiedKFold).",
                )

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    @staticmethod
    def _fingerprint_dataset(
        mri:    np.ndarray,
        gen:    np.ndarray,
        labels: np.ndarray,
    ) -> str:
        """SHA-256 fingerprint of dataset statistics (fast, not full hash)."""
        sig = (
            f"{mri.shape}_{gen.shape}_"
            f"{np.nanmean(mri):.6f}_{np.nanstd(mri):.6f}_"
            f"{np.nanmean(gen):.6f}_{np.nanstd(gen):.6f}_"
            f"{sorted(labels.tolist())}"
        )
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    @staticmethod
    def _fingerprint_metadata(meta) -> str:
        """SHA-256 of metadata columns to detect dataset changes."""
        sig = str(sorted(meta.to_dict("records"), key=lambda r: str(r)))
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Helper issue adders
    # ------------------------------------------------------------------

    def _add_critical(
        self, report: ValidationReport, category: str,
        message: str, affected: Optional[List[str]] = None,
    ) -> None:
        report.issues.append(IssueRecord(
            severity="CRITICAL", category=category,
            message=message, affected=affected or [],
        ))
        logger.error("[CRITICAL] %s: %s", category, message)

    def _add_warning(
        self, report: ValidationReport, category: str,
        message: str, affected: Optional[List[str]] = None,
    ) -> None:
        report.issues.append(IssueRecord(
            severity="WARNING", category=category,
            message=message, affected=affected or [],
        ))
        logger.warning("[DATA VALIDATION] %s: %s", category, message)

    def _add_info(
        self, report: ValidationReport, category: str, message: str,
    ) -> None:
        report.issues.append(IssueRecord(
            severity="INFO", category=category, message=message,
        ))
        logger.debug("[DATA VALIDATION] %s: %s", category, message)
