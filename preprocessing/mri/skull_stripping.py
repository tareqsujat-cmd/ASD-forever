"""
Brain extraction (skull stripping).

Skull stripping removes the non-brain tissue (skull, scalp, dura) from T1w
MRI volumes.  Non-brain voxels carry no signal of interest and, if included,
corrupt z-score normalization and bias feature extraction toward anatomy that
is identical across subjects.

Methods supported
-----------------
1. ANTs BrainExtraction (antspyx) — gold standard, morphological approach
2. HD-BET (deep learning, GPU) — fastest and most accurate on 1.5T/3T T1w
3. SimpleITK Otsu fallback — no extra dependency, good enough for QC preview

Strategy: try HD-BET first (GPU available), fall back to ANTs, then Otsu.

References
----------
Isensee F, et al. (2019). Automated brain extraction of MRI using deep
learning. bioRxiv. https://github.com/MIC-DKFZ/HD-BET

Usage
-----
    from preprocessing.mri.skull_stripping import SkullStripper
    stripper = SkullStripper(method="auto")
    masked_data, brain_mask = stripper.strip(data, affine)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SkullStripper:
    """
    Multi-method skull stripper with automatic fallback.

    Parameters
    ----------
    method : str
        "auto" | "hdbet" | "ants" | "otsu"
        "auto" tries hdbet -> ants -> otsu in order.
    ants_template_dir : str, optional
        Path to ANTs brain extraction template directory.
        If None, uses the default ANTs template bundled with antspyx.
    device : str
        "cuda" | "cpu" for HD-BET.
    """

    def __init__(
        self,
        method: str = "auto",
        ants_template_dir: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.method = method.lower()
        self.ants_template_dir = ants_template_dir
        self.device = device

        self._available_methods = self._detect_available_methods()
        logger.info(f"SkullStripper initialized. Available methods: "
                    f"{self._available_methods}")

    def strip(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        return_mask_only: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove skull and non-brain tissue.

        Parameters
        ----------
        data : np.ndarray, shape (D, H, W)
            T1w MRI volume (after bias correction).
        affine : np.ndarray, shape (4, 4)
            Voxel-to-world affine.
        return_mask_only : bool
            If True, return (masked_data, brain_mask).
            If False (default), return (masked_data, brain_mask).

        Returns
        -------
        masked_data : np.ndarray
            Volume with non-brain voxels zeroed.
        brain_mask : np.ndarray
            Binary mask (1=brain, 0=background), same shape as data.
        """
        method = self._resolve_method()
        logger.debug(f"Skull stripping with method: {method}")

        if method == "hdbet":
            brain_mask = self._strip_hdbet(data, affine)
        elif method == "ants":
            brain_mask = self._strip_ants(data, affine)
        else:
            brain_mask = self._strip_otsu(data)

        brain_mask = brain_mask.astype(np.float32)
        masked_data = data * brain_mask

        # QC: warn if brain volume is implausibly small or large
        from preprocessing.mri.nifti_utils import compute_brain_volume, get_voxel_size
        vox = get_voxel_size(affine).mean()
        vol_cm3 = compute_brain_volume(brain_mask, voxel_size_mm=vox)
        if vol_cm3 < 800 or vol_cm3 > 1800:
            logger.warning(
                f"Brain volume {vol_cm3:.0f} cm³ is outside normal range "
                f"[800, 1800] cm³ — skull stripping may have failed."
            )
        else:
            logger.debug(f"Brain volume: {vol_cm3:.0f} cm³ (normal)")

        return masked_data, brain_mask

    def strip_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        mask_output_path: Optional[str | Path] = None,
    ) -> None:
        """Strip skull from a NIfTI file and save result."""
        from preprocessing.mri.nifti_utils import load_nifti, save_nifti
        data, affine = load_nifti(input_path)
        masked, mask = self.strip(data, affine)
        save_nifti(masked, affine, output_path)
        if mask_output_path is not None:
            save_nifti(mask, affine, mask_output_path)
        logger.info(f"Skull-stripped saved: {output_path}")

    # ------------------------------------------------------------------
    # Method implementations
    # ------------------------------------------------------------------

    def _strip_hdbet(self, data: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """
        HD-BET: deep learning brain extraction.

        Saves to a temp file, runs HD-BET CLI, reads back the mask.
        """
        try:
            import hd_bet  # noqa: F401
        except ImportError:
            logger.warning("HD-BET not installed, falling back to ANTs")
            return self._strip_ants(data, affine)

        from preprocessing.mri.nifti_utils import load_nifti, save_nifti

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            in_path = tmpdir / "input.nii.gz"
            out_path = tmpdir / "output"

            save_nifti(data, affine, in_path)

            import subprocess
            cmd = [
                "hd-bet",
                "-i", str(in_path),
                "-o", str(out_path),
                "-device", self.device,
                "-mode", "fast",
                "-tta", "0",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"HD-BET failed: {result.stderr}. Falling back to ANTs.")
                return self._strip_ants(data, affine)

            mask_path = out_path.parent / (out_path.name + "_mask.nii.gz")
            if not mask_path.exists():
                mask_path = tmpdir / "output_mask.nii.gz"

            mask_data, _ = load_nifti(mask_path)
            return (mask_data > 0.5).astype(np.float32)

    def _strip_ants(self, data: np.ndarray, affine: np.ndarray) -> np.ndarray:
        """ANTs BrainExtraction — morphological approach using atlas registration."""
        try:
            import ants
        except ImportError:
            logger.warning("antspyx not installed, falling back to Otsu")
            return self._strip_otsu(data)

        # Convert to ANTsImage
        spacing = tuple(
            float(np.sqrt((affine[:3, i] ** 2).sum())) for i in range(3)
        )
        ants_img = ants.from_numpy(data, spacing=spacing)

        try:
            result = ants.get_mask(ants_img, low_thresh=None, cleanup=3)
            mask = result.numpy()
        except Exception as exc:
            logger.warning(f"ANTs mask failed: {exc}. Falling back to Otsu.")
            return self._strip_otsu(data)

        return (mask > 0.5).astype(np.float32)

    def _strip_otsu(self, data: np.ndarray) -> np.ndarray:
        """
        SimpleITK Otsu-threshold + morphological operations fallback.

        Less accurate than ANTs/HD-BET but requires no external dependencies.
        Suitable for QC previews and CPU-only environments.
        """
        try:
            import SimpleITK as sitk

            sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
            otsu = sitk.OtsuThresholdImageFilter()
            otsu.SetInsideValue(0)
            otsu.SetOutsideValue(1)
            mask = otsu.Execute(sitk_img)

            # Fill holes and erode to clean up
            mask = sitk.BinaryFillhole(mask)
            mask = sitk.BinaryErode(mask, [2, 2, 2])
            mask = sitk.BinaryDilate(mask, [3, 3, 3])

            # Keep largest connected component (brain)
            mask = sitk.ConnectedComponent(mask)
            label_stats = sitk.LabelShapeStatisticsImageFilter()
            label_stats.Execute(mask)
            labels = label_stats.GetLabels()
            if labels:
                sizes = {lbl: label_stats.GetNumberOfPixels(lbl) for lbl in labels}
                largest = max(sizes, key=sizes.get)
                mask = sitk.Equal(mask, largest)

            return sitk.GetArrayFromImage(mask).astype(np.float32)

        except ImportError:
            logger.warning("SimpleITK not available; using intensity threshold fallback")
            # Last resort: threshold at 10% of max
            threshold = np.percentile(data[data > 0], 10) if data.max() > 0 else 0
            return (data > threshold).astype(np.float32)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _resolve_method(self) -> str:
        if self.method != "auto":
            return self.method
        priority = ["hdbet", "ants", "otsu"]
        for m in priority:
            if m in self._available_methods:
                return m
        return "otsu"

    @staticmethod
    def _detect_available_methods() -> list:
        available = ["otsu"]  # always available (SimpleITK fallback)
        try:
            import ants  # noqa: F401
            available.append("ants")
        except ImportError:
            pass
        try:
            import hd_bet  # noqa: F401
            available.append("hdbet")
        except ImportError:
            pass
        return available
