"""
Intensity normalization for structural MRI.

Why normalization matters for publication
-----------------------------------------
ABIDE data comes from 20+ sites with different field strengths (1.5T, 3T),
manufacturers (Siemens, GE, Philips), and acquisition parameters.  Raw
T1w intensities are NOT comparable across sites:  a voxel value of 1000
in a Siemens 3T scan means something completely different from 1000 in a
GE 1.5T scan.  Normalization is therefore a critical pre-processing step,
not optional.

Site-aware normalization rule
------------------------------
CRITICAL: Never compute normalization statistics on the full dataset.
This leaks test-set information into training.  Instead:
  - Compute stats (mean, std, percentiles) on TRAINING folds only
  - Apply those same stats to validation and test sets
  - Optionally: compute per-site stats on training data to remove site bias

Methods implemented
-------------------
1. z_score        : (x - mu) / sigma  [per-volume or per-site]
2. min_max        : (x - min) / (max - min)  -> [0, 1]
3. percentile     : clip at p2/p98, then min-max
4. nyul_udupa     : histogram matching to a learned template (gold standard)
   Nyul LG, Udupa JK (1999). On standardizing the MR image intensity scale.
   MRM 42:1072-1081.

Usage
-----
    from preprocessing.mri.normalization import IntensityNormalizer
    norm = IntensityNormalizer(method="z_score")
    norm.fit(train_volumes)          # compute stats from training data
    normalized = norm.transform(volume, site_id="NYU")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class IntensityNormalizer:
    """
    Intensity normalizer with fit/transform API (sklearn-style).

    This design ensures that normalization statistics are always computed
    from training data and applied to val/test, preventing data leakage.

    Parameters
    ----------
    method : str
        "z_score" | "min_max" | "percentile" | "nyul_udupa"
    site_aware : bool
        If True, compute and store per-site statistics during fit().
        Apply site-specific stats during transform().
        This removes inter-site intensity bias without requiring
        explicit ComBat harmonization (useful as a fast alternative).
    low_percentile : float
        Lower percentile for clipping (used in "percentile" method).
    high_percentile : float
        Upper percentile for clipping.
    n_histogram_bins : int
        Bins for Nyul-Udupa histogram matching.
    """

    def __init__(
        self,
        method: str = "z_score",
        site_aware: bool = True,
        low_percentile: float = 2.0,
        high_percentile: float = 98.0,
        n_histogram_bins: int = 1000,
    ) -> None:
        self.method = method.lower()
        self.site_aware = site_aware
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.n_histogram_bins = n_histogram_bins

        # Statistics populated by fit()
        self._global_stats: Dict = {}
        self._site_stats: Dict[str, Dict] = {}
        self._nyul_template: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        volumes: List[np.ndarray],
        brain_masks: Optional[List[np.ndarray]] = None,
        site_ids: Optional[List[str]] = None,
    ) -> "IntensityNormalizer":
        """
        Compute normalization statistics from training volumes.

        Parameters
        ----------
        volumes : list of np.ndarray
            Training MRI volumes (brain-masked, bias-corrected).
        brain_masks : list of np.ndarray, optional
            Binary brain masks.  If provided, statistics are computed
            on brain voxels only (more accurate).
        site_ids : list of str, optional
            Site identifiers for each volume.  Required if site_aware=True.

        Returns
        -------
        self
        """
        logger.info(f"Fitting normalizer ({self.method}, site_aware={self.site_aware}) "
                    f"on {len(volumes)} volumes")

        brain_intensities_all = []
        for i, vol in enumerate(volumes):
            mask = brain_masks[i] if brain_masks else (vol > 0)
            brain_intensities_all.append(vol[mask > 0])

        # Global statistics
        all_vals = np.concatenate(brain_intensities_all)
        self._global_stats = self._compute_stats(all_vals)
        logger.debug(f"Global stats: {self._global_stats}")

        # Per-site statistics
        if self.site_aware and site_ids is not None:
            unique_sites = set(site_ids)
            for site in unique_sites:
                site_mask = [s == site for s in site_ids]
                site_vals = np.concatenate([
                    brain_intensities_all[i] for i, m in enumerate(site_mask) if m
                ])
                self._site_stats[site] = self._compute_stats(site_vals)
            logger.info(f"Per-site stats computed for {len(unique_sites)} sites")

        # Nyul-Udupa template
        if self.method == "nyul_udupa":
            self._nyul_template = self._build_nyul_template(brain_intensities_all)

        self._fitted = True
        return self

    def transform(
        self,
        volume: np.ndarray,
        brain_mask: Optional[np.ndarray] = None,
        site_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Normalize a single volume.

        Parameters
        ----------
        volume : np.ndarray
            MRI volume to normalize.
        brain_mask : np.ndarray, optional
            Binary brain mask.
        site_id : str, optional
            Site identifier (used for site-aware normalization).

        Returns
        -------
        np.ndarray
            Normalized volume, same shape.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        mask = brain_mask if brain_mask is not None else (volume > 0)
        mask = mask > 0

        # Select statistics
        if self.site_aware and site_id is not None and site_id in self._site_stats:
            stats = self._site_stats[site_id]
        else:
            stats = self._global_stats

        result = volume.copy()

        if self.method == "z_score":
            result[mask] = (volume[mask] - stats["mean"]) / (stats["std"] + 1e-8)

        elif self.method == "min_max":
            lo, hi = stats["min"], stats["max"]
            result[mask] = (volume[mask] - lo) / (hi - lo + 1e-8)

        elif self.method == "percentile":
            lo, hi = stats["p_low"], stats["p_high"]
            clipped = np.clip(volume[mask], lo, hi)
            result[mask] = (clipped - lo) / (hi - lo + 1e-8)

        elif self.method == "nyul_udupa":
            if self._nyul_template is None:
                raise RuntimeError("Nyul template not built. Call fit() first.")
            result[mask] = self._apply_nyul(volume[mask])

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        # Zero out background explicitly (mask=False regions stay 0)
        result[~mask] = 0.0
        return result.astype(np.float32)

    def fit_transform(
        self,
        volumes: List[np.ndarray],
        brain_masks: Optional[List[np.ndarray]] = None,
        site_ids: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """Fit on all volumes and return transformed versions (for use on training data)."""
        self.fit(volumes, brain_masks, site_ids)
        return [
            self.transform(
                v,
                brain_mask=brain_masks[i] if brain_masks else None,
                site_id=site_ids[i] if site_ids else None,
            )
            for i, v in enumerate(volumes)
        ]

    def save(self, path: Union[str, Path]) -> None:
        """Persist fitted normalizer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Normalizer saved: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "IntensityNormalizer":
        """Load a previously fitted normalizer."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected IntensityNormalizer")
        return obj

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_stats(self, values: np.ndarray) -> Dict:
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p_low": float(np.percentile(values, self.low_percentile)),
            "p_high": float(np.percentile(values, self.high_percentile)),
            "median": float(np.median(values)),
        }

    def _build_nyul_template(self, brain_intensities_list: List[np.ndarray]) -> np.ndarray:
        """
        Build the reference histogram for Nyul-Udupa standardization.

        Computes the mean CDF across all training subjects.  New subjects
        are mapped to match this reference CDF.
        """
        bins = np.linspace(0, 1, self.n_histogram_bins)
        cdfs = []
        for vals in brain_intensities_list:
            plow = np.percentile(vals, self.low_percentile)
            phigh = np.percentile(vals, self.high_percentile)
            clipped = np.clip(vals, plow, phigh)
            normalized = (clipped - plow) / (phigh - plow + 1e-8)
            hist, _ = np.histogram(normalized, bins=self.n_histogram_bins, range=(0, 1))
            cdf = np.cumsum(hist).astype(float)
            cdf /= cdf[-1] + 1e-8
            cdfs.append(cdf)

        return np.stack(cdfs, axis=0).mean(axis=0)

    def _apply_nyul(self, brain_vals: np.ndarray) -> np.ndarray:
        """Map subject histogram to match the learned template CDF."""
        plow = np.percentile(brain_vals, self.low_percentile)
        phigh = np.percentile(brain_vals, self.high_percentile)
        clipped = np.clip(brain_vals, plow, phigh)
        normalized = (clipped - plow) / (phigh - plow + 1e-8)

        # Compute subject CDF
        hist, edges = np.histogram(normalized, bins=self.n_histogram_bins, range=(0, 1))
        subj_cdf = np.cumsum(hist).astype(float)
        subj_cdf /= subj_cdf[-1] + 1e-8
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Map subject CDF to template CDF via inverse lookup
        template_bins = np.linspace(0, 1, len(self._nyul_template))
        mapped = np.interp(
            np.interp(normalized, bin_centers, subj_cdf),
            self._nyul_template,
            template_bins,
        )
        return mapped.astype(np.float32)
