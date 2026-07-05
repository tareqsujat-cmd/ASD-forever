"""
N4ITK Bias Field Correction.

MRI scanners produce images with a slowly-varying intensity inhomogeneity
called the bias field (or gain field). This arises from RF coil non-uniformity
and degrades both visual interpretation and quantitative analysis.

N4ITK (Tustison et al., 2010) is the gold-standard algorithm:
  - B-spline approximation of the bias field
  - Iterative histogram sharpening
  - Convergence-monitored iterative refinement

Reference
---------
Tustison NJ, et al. (2010). N4ITK: Improved N3 bias correction.
IEEE Trans Med Imaging, 29(6):1310-20. doi:10.1109/TMI.2010.2046908

Usage
-----
    from preprocessing.mri.bias_correction import N4BiasCorrector
    corrector = N4BiasCorrector()
    corrected_data, bias_field = corrector.correct(data, affine)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class N4BiasCorrector:
    """
    N4ITK bias field corrector via SimpleITK.

    Parameters
    ----------
    n_fitting_levels : int
        Number of B-spline fitting levels. 4 is standard; more = finer
        correction but slower.
    n_iterations : list of int
        Max iterations per fitting level. [50, 50, 50, 50] is standard.
    convergence_threshold : float
        Stop if change in bias field energy drops below this. Default 1e-6.
    mask_image : bool
        If True, create an Otsu-thresholded mask before correction.
        Using a brain mask gives more accurate bias estimation than
        including background voxels.
    shrink_factor : int
        Downsample image by this factor during fitting (speeds up N4).
        Final correction is applied at full resolution. 4 is typical.
    """

    def __init__(
        self,
        n_fitting_levels: int = 4,
        n_iterations: Optional[list] = None,
        convergence_threshold: float = 1e-6,
        mask_image: bool = True,
        shrink_factor: int = 4,
    ) -> None:
        if n_iterations is None:
            n_iterations = [50] * n_fitting_levels

        self.n_fitting_levels = n_fitting_levels
        self.n_iterations = n_iterations
        self.convergence_threshold = convergence_threshold
        self.mask_image = mask_image
        self.shrink_factor = shrink_factor

        self._check_sitk()

    def _check_sitk(self) -> None:
        try:
            import SimpleITK as sitk  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SimpleITK is required for N4 bias correction. "
                "Install with: pip install SimpleITK"
            ) from e

    def correct(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        brain_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply N4ITK bias field correction.

        Parameters
        ----------
        data : np.ndarray, shape (D, H, W)
            Input MRI volume (float32, any intensity range).
        affine : np.ndarray, shape (4, 4)
            Voxel-to-world affine (used to preserve spacing/direction).
        brain_mask : np.ndarray, optional
            Binary mask (1=brain). If provided, used for bias estimation.
            If None and self.mask_image=True, auto-generates Otsu mask.

        Returns
        -------
        corrected_data : np.ndarray
            Bias-corrected volume (same shape as input).
        bias_field : np.ndarray
            Estimated bias field (for logging/QC).

        Notes
        -----
        N4 operates on log-domain intensities, so the input must be
        strictly positive. We add a small offset if needed.
        """
        import SimpleITK as sitk

        # N4 requires positive intensities
        data_shifted = data.copy()
        min_val = data_shifted.min()
        if min_val <= 0:
            data_shifted = data_shifted - min_val + 1e-6

        # Convert numpy -> SimpleITK image (preserving spacing from affine)
        sitk_img = self._numpy_to_sitk(data_shifted, affine)

        # Create mask
        if brain_mask is not None:
            sitk_mask = sitk.GetImageFromArray(brain_mask.astype(np.uint8))
            sitk_mask.CopyInformation(sitk_img)
        elif self.mask_image:
            sitk_mask = self._otsu_mask(sitk_img)
        else:
            sitk_mask = None

        # Downsample for faster fitting
        if self.shrink_factor > 1:
            sitk_img_shrunk = sitk.Shrink(
                sitk_img, [self.shrink_factor] * sitk_img.GetDimension()
            )
            if sitk_mask is not None:
                sitk_mask_shrunk = sitk.Shrink(
                    sitk_mask, [self.shrink_factor] * sitk_mask.GetDimension()
                )
            else:
                sitk_mask_shrunk = None
        else:
            sitk_img_shrunk = sitk_img
            sitk_mask_shrunk = sitk_mask

        # Run N4
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(self.n_iterations)
        corrector.SetConvergenceThreshold(self.convergence_threshold)
        corrector.SetNumberOfFittingLevels(self.n_fitting_levels)

        if sitk_mask_shrunk is not None:
            corrector.Execute(sitk_img_shrunk, sitk_mask_shrunk)
        else:
            corrector.Execute(sitk_img_shrunk)

        # Extract bias field and apply at full resolution
        log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_img)
        bias_field = sitk.Exp(log_bias_field)

        # Apply correction: corrected = original / bias_field
        corrected_sitk = sitk.Divide(sitk_img, bias_field)

        corrected_np = sitk.GetArrayFromImage(corrected_sitk).astype(np.float32)
        bias_np = sitk.GetArrayFromImage(bias_field).astype(np.float32)

        # Restore original intensity offset
        if min_val <= 0:
            corrected_np = corrected_np + min_val - 1e-6

        logger.debug(
            f"N4 correction complete. Bias field range: "
            f"[{bias_np.min():.3f}, {bias_np.max():.3f}], "
            f"std={bias_np.std():.4f}"
        )
        return corrected_np, bias_np

    def correct_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        mask_path: Optional[str | Path] = None,
    ) -> None:
        """
        Correct a NIfTI file and save the result.

        Parameters
        ----------
        input_path : str or Path
            Input NIfTI path.
        output_path : str or Path
            Output NIfTI path.
        mask_path : str or Path, optional
            Optional brain mask NIfTI.
        """
        from preprocessing.mri.nifti_utils import load_nifti, save_nifti

        data, affine = load_nifti(input_path)

        brain_mask = None
        if mask_path is not None:
            brain_mask, _ = load_nifti(mask_path)
            brain_mask = (brain_mask > 0).astype(np.uint8)

        corrected, _ = self.correct(data, affine, brain_mask)
        save_nifti(corrected, affine, output_path)
        logger.info(f"Bias-corrected saved: {output_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _numpy_to_sitk(data: np.ndarray, affine: np.ndarray):
        """
        Convert a numpy array + affine to a SimpleITK Image.

        SimpleITK uses (x, y, z) = (W, H, D) convention (opposite of numpy).
        We handle the transpose here explicitly.
        """
        import SimpleITK as sitk

        # SimpleITK expects (z, y, x) -> numpy (D, H, W) matches this
        sitk_img = sitk.GetImageFromArray(data)

        # Extract spacing (voxel size in mm) from affine
        spacing = tuple(float(np.sqrt((affine[:3, i] ** 2).sum())) for i in range(3))
        sitk_img.SetSpacing(spacing)

        # Extract origin
        origin = tuple(float(affine[i, 3]) for i in range(3))
        sitk_img.SetOrigin(origin)

        # Direction cosines from affine
        direction = []
        for i in range(3):
            col = affine[:3, i] / (np.linalg.norm(affine[:3, i]) + 1e-12)
            direction.extend(col.tolist())
        sitk_img.SetDirection(direction)

        return sitk_img

    @staticmethod
    def _otsu_mask(sitk_img):
        """Generate a binary mask using Otsu thresholding."""
        import SimpleITK as sitk

        otsu = sitk.OtsuThresholdImageFilter()
        otsu.SetInsideValue(0)
        otsu.SetOutsideValue(1)
        mask = otsu.Execute(sitk_img)
        # Dilate slightly to avoid clipping brain edges
        mask = sitk.BinaryDilate(mask, [3, 3, 3])
        return mask
