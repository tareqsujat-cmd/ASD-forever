"""
MRI Quality Control (QC) scoring and subject exclusion.

Why QC is mandatory for publication
-------------------------------------
ABIDE contains scans with motion artifacts, field dropout, and acquisition
failures.  Including poor-quality scans as training data hurts performance
AND confounds interpretation: a model that learns "low-SNR scans tend to be
from ASD subjects" is finding acquisition artifacts, not biology.

MRIQC (Esteban et al., 2017) defines the standard IQMs (Image Quality
Metrics).  We implement the most relevant structural IQMs:
  - SNR          : signal-to-noise ratio (brain / background noise)
  - CNR          : contrast-to-noise ratio (WM vs. GM contrast)
  - FBER         : foreground-background energy ratio
  - EFC          : entropy focus criterion (sharpness)
  - WM2MAX       : white matter to max intensity ratio

Subjects below a configurable threshold are flagged and excluded.

Reference
---------
Esteban O, et al. (2017). MRIQC: Advancing the automatic prediction of image
quality in MRI from unseen sites. PLOS ONE 12(9):e0184661.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QCReport:
    """Structured QC report for a single subject."""
    subject_id: str
    snr: float = 0.0
    cnr: float = 0.0
    fber: float = 0.0
    efc: float = 0.0
    wm2max: float = 0.0
    brain_volume_cm3: float = 0.0
    passed: bool = True
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "subject_id": self.subject_id,
            "snr": round(self.snr, 4),
            "cnr": round(self.cnr, 4),
            "fber": round(self.fber, 4),
            "efc": round(self.efc, 6),
            "wm2max": round(self.wm2max, 4),
            "brain_volume_cm3": round(self.brain_volume_cm3, 1),
            "passed": self.passed,
            "failure_reasons": "; ".join(self.failure_reasons),
        }


class MRIQualityChecker:
    """
    Compute structural MRI quality metrics and filter out failed scans.

    Parameters
    ----------
    snr_threshold : float
        Minimum acceptable SNR. Literature suggests >10 for usable scans.
    cnr_threshold : float
        Minimum WM/GM contrast-to-noise ratio.
    fber_threshold : float
        Minimum foreground-background energy ratio.
    efc_threshold : float
        Maximum entropy focus criterion (higher = blurrier).
    brain_volume_min_cm3 : float
        Minimum plausible brain volume in cm³.
    brain_volume_max_cm3 : float
        Maximum plausible brain volume in cm³.
    """

    def __init__(
        self,
        snr_threshold: float = 8.0,
        cnr_threshold: float = 1.5,
        fber_threshold: float = 100.0,
        efc_threshold: float = 0.6,
        brain_volume_min_cm3: float = 750.0,
        brain_volume_max_cm3: float = 2000.0,
    ) -> None:
        self.snr_threshold = snr_threshold
        self.cnr_threshold = cnr_threshold
        self.fber_threshold = fber_threshold
        self.efc_threshold = efc_threshold
        self.brain_volume_min_cm3 = brain_volume_min_cm3
        self.brain_volume_max_cm3 = brain_volume_max_cm3

    def evaluate(
        self,
        data: np.ndarray,
        brain_mask: np.ndarray,
        subject_id: str = "unknown",
        voxel_size_mm: float = 2.0,
    ) -> QCReport:
        """
        Compute all IQMs for a single subject.

        Parameters
        ----------
        data : np.ndarray, shape (D, H, W)
            Preprocessed MRI volume (after bias correction, skull stripping).
        brain_mask : np.ndarray
            Binary brain mask.
        subject_id : str
            Subject identifier for the report.
        voxel_size_mm : float
            Isotropic voxel size (for volume computation).

        Returns
        -------
        QCReport
        """
        brain_mask = (brain_mask > 0)
        bg_mask = ~brain_mask

        brain_voxels = data[brain_mask]
        bg_voxels = data[bg_mask]

        # Compute metrics
        snr = self._compute_snr(brain_voxels, bg_voxels)
        cnr = self._compute_cnr(data, brain_mask)
        fber = self._compute_fber(brain_voxels, bg_voxels)
        efc = self._compute_efc(data)
        wm2max = self._compute_wm2max(brain_voxels)

        # Brain volume
        from preprocessing.mri.nifti_utils import compute_brain_volume
        volume_cm3 = compute_brain_volume(brain_mask, voxel_size_mm)

        report = QCReport(
            subject_id=subject_id,
            snr=snr,
            cnr=cnr,
            fber=fber,
            efc=efc,
            wm2max=wm2max,
            brain_volume_cm3=volume_cm3,
        )

        # Apply thresholds
        self._apply_thresholds(report)

        if report.passed:
            logger.debug(f"QC PASS: {subject_id} | SNR={snr:.1f} CNR={cnr:.2f} "
                         f"FBER={fber:.0f} Vol={volume_cm3:.0f}cm³")
        else:
            logger.warning(f"QC FAIL: {subject_id} | {report.failure_reasons}")

        return report

    def evaluate_batch(
        self,
        volumes: List[np.ndarray],
        masks: List[np.ndarray],
        subject_ids: List[str],
        voxel_size_mm: float = 2.0,
    ) -> Tuple[List[QCReport], List[int]]:
        """
        Evaluate a batch and return reports + indices of passing subjects.

        Returns
        -------
        reports : list of QCReport
        passing_indices : list of int
        """
        reports = []
        passing = []
        for i, (vol, mask, sid) in enumerate(zip(volumes, masks, subject_ids)):
            report = self.evaluate(vol, mask, sid, voxel_size_mm)
            reports.append(report)
            if report.passed:
                passing.append(i)

        n_total = len(reports)
        n_pass = len(passing)
        logger.info(f"QC complete: {n_pass}/{n_total} passed "
                    f"({100 * n_pass / n_total:.1f}%)")
        return reports, passing

    def save_report(
        self,
        reports: List[QCReport],
        output_path: str,
    ) -> None:
        """Save QC reports to a CSV file."""
        import pandas as pd
        df = pd.DataFrame([r.to_dict() for r in reports])
        df.to_csv(output_path, index=False)
        logger.info(f"QC report saved: {output_path} ({len(df)} subjects)")

    # ------------------------------------------------------------------
    # IQM implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_snr(brain_voxels: np.ndarray, bg_voxels: np.ndarray) -> float:
        """
        SNR = mean(brain) / std(background).

        Background noise std is the best noise estimator for 2D/3D MRI
        because it avoids the bias introduced by computing std within the
        brain (where signal variation includes true anatomical variation).
        """
        bg_std = bg_voxels.std()
        if bg_std < 1e-8:
            return 0.0
        return float(brain_voxels.mean() / bg_std)

    @staticmethod
    def _compute_cnr(data: np.ndarray, brain_mask: np.ndarray) -> float:
        """
        CNR = |mu_WM - mu_GM| / sqrt(sigma_WM^2 + sigma_GM^2).

        We approximate WM/GM segmentation using k-means (k=3) on brain
        intensities: the highest-intensity cluster = WM, middle = GM.
        This is a simplified Otsu-based approach that avoids FSL FAST.
        """
        brain_vals = data[brain_mask]
        if len(brain_vals) < 100:
            return 0.0

        # Simple 3-class k-means approximation
        try:
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=3, n_init=3, random_state=42)
            labels = kmeans.fit_predict(brain_vals.reshape(-1, 1))
            centers = kmeans.cluster_centers_.flatten()
            sorted_idx = np.argsort(centers)
            # sorted_idx[2] = WM (highest), sorted_idx[1] = GM
            wm_label = sorted_idx[2]
            gm_label = sorted_idx[1]
            wm_vals = brain_vals[labels == wm_label]
            gm_vals = brain_vals[labels == gm_label]

            if len(wm_vals) < 10 or len(gm_vals) < 10:
                return 0.0

            mu_diff = abs(wm_vals.mean() - gm_vals.mean())
            noise = np.sqrt(wm_vals.var() + gm_vals.var() + 1e-8)
            return float(mu_diff / noise)
        except Exception:
            return 0.0

    @staticmethod
    def _compute_fber(brain_voxels: np.ndarray, bg_voxels: np.ndarray) -> float:
        """
        FBER = var(brain) / var(background).

        High FBER means the foreground has much more energy than background,
        indicating good contrast and little noise contamination.
        """
        bg_var = bg_voxels.var()
        if bg_var < 1e-8:
            return 0.0
        return float(brain_voxels.var() / bg_var)

    @staticmethod
    def _compute_efc(data: np.ndarray) -> float:
        """
        EFC (Entropy Focus Criterion) — Atkinson et al. (1997).

        Measures image sharpness in the frequency domain.  A sharp image
        has energy concentrated in fewer frequency components (low entropy).
        Lower EFC = sharper image = better quality.

        EFC = H(F) / H_max
        where H(F) is the Shannon entropy of the k-space magnitude.
        """
        try:
            F = np.fft.fftn(data)
            mag = np.abs(F).flatten()
            mag = mag[mag > 0]
            mag_norm = mag / (mag.sum() + 1e-12)
            entropy = -np.sum(mag_norm * np.log(mag_norm + 1e-12))
            n = len(mag_norm)
            max_entropy = np.log(n) if n > 0 else 1.0
            return float(entropy / (max_entropy + 1e-12))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_wm2max(brain_voxels: np.ndarray) -> float:
        """
        WM2MAX = p95(brain_intensities) / max(brain_intensities).

        Values close to 1 indicate good contrast.  Very low values
        suggest clipping or gain saturation artifacts.
        """
        if len(brain_voxels) == 0:
            return 0.0
        p95 = np.percentile(brain_voxels, 95)
        max_val = brain_voxels.max()
        if max_val < 1e-8:
            return 0.0
        return float(p95 / max_val)

    def _apply_thresholds(self, report: QCReport) -> None:
        """Flag QC failures with descriptive reasons."""
        if report.snr < self.snr_threshold:
            report.failure_reasons.append(
                f"SNR={report.snr:.1f} < {self.snr_threshold}"
            )
        if report.cnr < self.cnr_threshold:
            report.failure_reasons.append(
                f"CNR={report.cnr:.2f} < {self.cnr_threshold}"
            )
        if report.fber < self.fber_threshold:
            report.failure_reasons.append(
                f"FBER={report.fber:.0f} < {self.fber_threshold}"
            )
        if report.efc > self.efc_threshold:
            report.failure_reasons.append(
                f"EFC={report.efc:.3f} > {self.efc_threshold}"
            )
        if report.brain_volume_cm3 < self.brain_volume_min_cm3:
            report.failure_reasons.append(
                f"Volume={report.brain_volume_cm3:.0f}cm³ < "
                f"{self.brain_volume_min_cm3:.0f}cm³"
            )
        if report.brain_volume_cm3 > self.brain_volume_max_cm3:
            report.failure_reasons.append(
                f"Volume={report.brain_volume_cm3:.0f}cm³ > "
                f"{self.brain_volume_max_cm3:.0f}cm³"
            )
        report.passed = len(report.failure_reasons) == 0
