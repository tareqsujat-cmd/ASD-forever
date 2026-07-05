"""
3D MRI data augmentation transforms.

Augmentation rationale for ASD/MRI research
--------------------------------------------
ABIDE I has ~1112 subjects — small by deep learning standards.  Augmentation
artificially expands the training set while staying biologically plausible.

Constraint: Augmentations MUST preserve neuroanatomical structure.  We avoid:
  - Large rotations (>15°): distort sulcal geometry used by the model
  - Heavy elastic deformation: could move cortical features between regions
  - Color/brightness changes that simulate scanner differences: those are
    handled by normalization, not augmentation

Augmentations we do use:
  - Random flip along L-R axis (ASD shows subtle asymmetry, so flip carefully)
  - Small random rotations (±10°)
  - Small random scaling (±10%)
  - Gaussian noise (SNR simulation)
  - Random intensity shift/scale (simulate brightness variation)
  - Random cutout (regularization)

All transforms work on 3D numpy arrays (D, H, W) and are implemented
without external dependencies where possible, falling back to torchio/monai.

Usage
-----
    from preprocessing.mri.augmentation import MRIAugmentor
    aug = MRIAugmentor(config)
    augmented = aug(volume)  # volume: np.ndarray (D, H, W)
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MRIAugmentor:
    """
    Configurable 3D MRI augmentation pipeline.

    Each transform is applied independently with its own probability,
    so the effective augmentation space is the combinatorial product.

    Parameters
    ----------
    enabled : bool
        Master switch.
    flip_prob : float
        Probability of left-right flip.
    rotation_degrees : float
        Max rotation in degrees (±).
    scale_range : tuple
        (min_scale, max_scale) for random scaling.
    noise_std : float
        Std of Gaussian noise added (in normalized intensity units).
    intensity_shift_range : tuple
        Random intensity offset range (min, max).
    intensity_scale_range : tuple
        Random intensity scale factor range.
    cutout_prob : float
        Probability of random 3D cutout (regularization).
    cutout_size_fraction : float
        Cutout cube size as fraction of volume size.
    seed : int
        Random seed for this augmentor instance.
    """

    def __init__(
        self,
        enabled: bool = True,
        flip_prob: float = 0.5,
        rotation_degrees: float = 10.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.01,
        intensity_shift_range: Tuple[float, float] = (-0.05, 0.05),
        intensity_scale_range: Tuple[float, float] = (0.95, 1.05),
        cutout_prob: float = 0.2,
        cutout_size_fraction: float = 0.15,
        seed: Optional[int] = None,
    ) -> None:
        self.enabled = enabled
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range
        self.cutout_prob = cutout_prob
        self.cutout_size_fraction = cutout_size_fraction
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """Apply the augmentation pipeline to a single 3D volume."""
        if not self.enabled:
            return volume

        vol = volume.copy()
        vol = self._random_flip(vol)
        vol = self._random_rotate(vol)
        vol = self._random_scale(vol)
        vol = self._random_intensity_shift(vol)
        vol = self._random_intensity_scale(vol)
        vol = self._add_gaussian_noise(vol)
        vol = self._random_cutout(vol)
        return vol

    # ------------------------------------------------------------------
    # Individual transforms
    # ------------------------------------------------------------------

    def _random_flip(self, vol: np.ndarray) -> np.ndarray:
        """Flip along the left-right (first) axis."""
        if self._rng.random() < self.flip_prob:
            vol = np.flip(vol, axis=0).copy()
        return vol

    def _random_rotate(self, vol: np.ndarray) -> np.ndarray:
        """
        Apply a small 3D rotation using scipy's affine_transform.

        We use the full 3D rotation (not slice-wise) to avoid artifacts
        at slice boundaries that would appear in volumetric CNN feature maps.
        """
        if self.rotation_degrees <= 0:
            return vol

        try:
            from scipy.ndimage import affine_transform
            angle = self._rng.uniform(-self.rotation_degrees, self.rotation_degrees)
            angle_rad = np.deg2rad(angle)

            # Rotation matrix around z-axis (axial plane)
            axis = self._rng.integers(0, 3)  # random axis
            R = self._rotation_matrix_3d(angle_rad, axis)

            center = np.array(vol.shape) / 2.0
            offset = center - R @ center

            rotated = affine_transform(
                vol, R, offset=offset, mode="constant", cval=0.0, order=1
            )
            return rotated.astype(vol.dtype)
        except ImportError:
            return vol

    def _random_scale(self, vol: np.ndarray) -> np.ndarray:
        """
        Scale volume by a random factor via zoom.

        Scaling simulates anatomical size variation across subjects.
        The volume is zoomed and then center-padded/cropped back to
        its original shape.
        """
        min_s, max_s = self.scale_range
        if min_s == max_s == 1.0:
            return vol

        scale = self._rng.uniform(min_s, max_s)
        try:
            from scipy.ndimage import zoom
            target_shape = vol.shape
            zoomed = zoom(vol, scale, order=1, mode="constant", cval=0.0)

            from preprocessing.mri.nifti_utils import pad_or_crop_to_shape
            return pad_or_crop_to_shape(zoomed, target_shape).astype(vol.dtype)
        except ImportError:
            return vol

    def _random_intensity_shift(self, vol: np.ndarray) -> np.ndarray:
        """Add a random global intensity offset."""
        lo, hi = self.intensity_shift_range
        shift = self._rng.uniform(lo, hi)
        return (vol + shift).astype(vol.dtype)

    def _random_intensity_scale(self, vol: np.ndarray) -> np.ndarray:
        """Multiply intensities by a random factor."""
        lo, hi = self.intensity_scale_range
        scale = self._rng.uniform(lo, hi)
        return (vol * scale).astype(vol.dtype)

    def _add_gaussian_noise(self, vol: np.ndarray) -> np.ndarray:
        """Add zero-mean Gaussian noise."""
        if self.noise_std <= 0:
            return vol
        noise = self._rng.normal(0, self.noise_std, vol.shape).astype(vol.dtype)
        return vol + noise

    def _random_cutout(self, vol: np.ndarray) -> np.ndarray:
        """
        Random 3D cubic cutout (zeroed region).

        Cutout forces the model to use spatially distributed features rather
        than relying on a single region.  Particularly important for ASD
        where no single brain region is fully diagnostic.
        """
        if self._rng.random() >= self.cutout_prob:
            return vol

        D, H, W = vol.shape
        size_d = max(1, int(D * self.cutout_size_fraction))
        size_h = max(1, int(H * self.cutout_size_fraction))
        size_w = max(1, int(W * self.cutout_size_fraction))

        d0 = self._rng.integers(0, D - size_d + 1)
        h0 = self._rng.integers(0, H - size_h + 1)
        w0 = self._rng.integers(0, W - size_w + 1)

        result = vol.copy()
        result[d0:d0 + size_d, h0:h0 + size_h, w0:w0 + size_w] = 0.0
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _rotation_matrix_3d(angle_rad: float, axis: int) -> np.ndarray:
        """Generate a 3×3 rotation matrix around the specified axis (0=x,1=y,2=z)."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        if axis == 0:
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == 1:
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_augmentor_from_config(aug_config) -> MRIAugmentor:
    """
    Construct an MRIAugmentor from an AugmentationConfig dataclass.

    Parameters
    ----------
    aug_config : AugmentationConfig
        From configs.config_schema.

    Returns
    -------
    MRIAugmentor
    """
    return MRIAugmentor(
        enabled=aug_config.enabled,
        flip_prob=aug_config.flip_prob,
        rotation_degrees=float(aug_config.rotation_degrees),
        scale_range=tuple(aug_config.scale_range),
        noise_std=aug_config.noise_std,
    )
