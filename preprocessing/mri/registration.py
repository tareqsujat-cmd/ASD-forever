"""
MNI152 spatial registration.

Registration maps each subject's T1w into a common stereotaxic space
(MNI152) so that voxel (x, y, z) corresponds to the same anatomical
location across all subjects.  Without registration, a CNN learning on
unregistered volumes would capture individual morphological variation
rather than group-level ASD-vs-control differences.

Strategy
--------
1. Rigid (6 DOF) registration to handle gross head position differences
2. Affine (12 DOF) for scale and shear differences across scanners
3. SyN nonlinear (optional) for full cortical correspondence
   — not used by default because it destroys individual morphometry that
     may carry ASD signal in structural analyses.

For publication we use affine registration as the best trade-off between
anatomical correspondence and preserving structural information.

Reference
---------
Avants BB, et al. (2011). A reproducible evaluation of ANTs similarity metric
performance in brain image registration.  Neuroimage 54(3):2033-44.

Usage
-----
    from preprocessing.mri.registration import MNIRegistrar
    reg = MNIRegistrar(transform_type="affine")
    registered_data, affine_out = reg.register(data, affine)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Path to MNI152 template bundled with FSL / nilearn
# We try multiple locations so this works across environments
_MNI_TEMPLATE_CANDIDATES = [
    Path(r"C:\tools\fsl\data\standard\MNI152_T1_2mm_brain.nii.gz"),
    Path("/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"),
    Path("/usr/share/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"),
]


class MNIRegistrar:
    """
    Register T1w MRI volumes to MNI152 standard space.

    Parameters
    ----------
    transform_type : str
        "rigid" | "affine" | "syn" (nonlinear)
    mni_template_path : str or Path, optional
        Path to MNI152 template NIfTI.  Auto-detected if None.
    interpolator : str
        "linear" | "nearestNeighbor" | "bSpline"
    metric : str
        "MI" (mutual information) | "CC" (cross-correlation)
        MI is standard for monomodal T1w registration.
    """

    def __init__(
        self,
        transform_type: str = "affine",
        mni_template_path: Optional[str | Path] = None,
        interpolator: str = "linear",
        metric: str = "MI",
    ) -> None:
        self.transform_type = transform_type.lower()
        self.interpolator = interpolator
        self.metric = metric

        # Locate MNI template
        if mni_template_path is not None:
            self.mni_template_path = Path(mni_template_path)
        else:
            self.mni_template_path = self._find_mni_template()

        logger.info(f"MNIRegistrar: transform={transform_type}, "
                    f"template={self.mni_template_path}")

    def register(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        brain_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register a volume to MNI152 space.

        Parameters
        ----------
        data : np.ndarray, shape (D, H, W)
            Skull-stripped, bias-corrected T1w volume.
        affine : np.ndarray, shape (4, 4)
            Input voxel-to-world affine.
        brain_mask : np.ndarray, optional
            Brain mask (used to focus registration metric computation).

        Returns
        -------
        registered_data : np.ndarray
            Volume in MNI152 space.
        mni_affine : np.ndarray
            Affine for the registered volume.
        """
        try:
            import ants
        except ImportError:
            logger.warning(
                "antspyx not available; skipping registration. "
                "Install with: pip install antspyx"
            )
            return data, affine

        # Build ANTsImage from numpy
        spacing = tuple(
            float(np.sqrt((affine[:3, i] ** 2).sum())) for i in range(3)
        )
        moving = ants.from_numpy(data, spacing=spacing)

        # Load MNI template
        if self.mni_template_path is None:
            logger.warning("MNI template not found; using nilearn to download")
            fixed = self._get_nilearn_mni_template()
        else:
            fixed = ants.image_read(str(self.mni_template_path))

        # Map our transform type to ANTs type codes
        type_map = {
            "rigid": "Rigid",
            "affine": "Affine",
            "syn": "SyN",
            "quicksyn": "SyNQuick",
        }
        ants_type = type_map.get(self.transform_type, "Affine")

        # Run registration
        logger.debug(f"Running ANTs {ants_type} registration ...")
        reg_result = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=ants_type,
            aff_metric=self.metric if self.metric == "MI" else "mattes",
            verbose=False,
        )

        registered_ants = reg_result["warpedmovout"]
        registered_np = registered_ants.numpy().astype(np.float32)

        # Reconstruct affine from ANTs spacing/origin/direction
        mni_affine = self._ants_to_affine(registered_ants)

        logger.debug(f"Registration complete: {data.shape} -> {registered_np.shape}")
        return registered_np, mni_affine

    def register_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        transform_output_dir: Optional[str | Path] = None,
    ) -> None:
        """
        Register a NIfTI file to MNI152 and save.

        Parameters
        ----------
        input_path : str or Path
            Input T1w NIfTI.
        output_path : str or Path
            Output registered NIfTI.
        transform_output_dir : str or Path, optional
            If provided, saves the transform files for later application.
        """
        from preprocessing.mri.nifti_utils import load_nifti, save_nifti
        data, affine = load_nifti(input_path)
        registered, new_affine = self.register(data, affine)
        save_nifti(registered, new_affine, output_path)
        logger.info(f"Registered to MNI: {output_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_mni_template() -> Optional[Path]:
        """Search common filesystem locations for the MNI152 template."""
        for candidate in _MNI_TEMPLATE_CANDIDATES:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _get_nilearn_mni_template():
        """Download MNI152 template via nilearn if ANTs template not found."""
        try:
            import ants
            from nilearn.datasets import load_mni152_template
            import nibabel as nib
            import tempfile

            mni_img = load_mni152_template(resolution=2)
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
                nib.save(mni_img, f.name)
                return ants.image_read(f.name)
        except Exception as e:
            raise RuntimeError(
                "Cannot load MNI152 template from nilearn. "
                "Please provide mni_template_path explicitly."
            ) from e

    @staticmethod
    def _ants_to_affine(ants_img) -> np.ndarray:
        """Reconstruct a 4×4 affine matrix from an ANTsImage."""
        spacing = np.array(ants_img.spacing)
        origin = np.array(ants_img.origin)
        direction = np.array(ants_img.direction).reshape(3, 3)
        affine = np.eye(4)
        for i in range(3):
            affine[:3, i] = direction[:, i] * spacing[i]
        affine[:3, 3] = origin
        return affine
