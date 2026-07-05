"""
Core NIfTI I/O and spatial manipulation utilities.

All spatial operations preserve affine transforms so that output coordinates
remain interpretable in MNI space — required for neuroimaging publication.

Key operations
--------------
- load_nifti          : load with header validation
- resample_to_target  : resample to isotropic voxel size
- crop_to_brain       : tight crop around nonzero voxels
- pad_to_shape        : zero-pad to target shape
- apply_brain_mask    : zero out non-brain voxels
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output

logger = logging.getLogger(__name__)

# Nibabel resampling interpolation codes
_INTERP_MAP = {
    "linear": 1,
    "nearest": 0,
    "cubic": 3,
    "quintic": 5,
}


def load_nifti(
    path: Union[str, Path],
    dtype: np.dtype = np.float32,
    validate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file, returning (data_array, affine_matrix).

    Parameters
    ----------
    path : str or Path
        Path to a .nii or .nii.gz file.
    dtype : numpy dtype
        Cast the volume to this dtype (float32 saves memory vs float64).
    validate : bool
        If True, check that the file is 3D and non-empty.

    Returns
    -------
    data : np.ndarray, shape (D, H, W)
        Volume data in RAS+ orientation.
    affine : np.ndarray, shape (4, 4)
        Voxel-to-world affine transform.

    Raises
    ------
    FileNotFoundError, ValueError
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)  # reorient to RAS+

    data = img.get_fdata(dtype=np.float32).astype(dtype)
    affine = img.affine.copy()

    if validate:
        if data.ndim not in (3, 4):
            raise ValueError(f"Expected 3D or 4D NIfTI, got shape {data.shape}: {path}")
        if data.ndim == 4:
            # For 4D (fMRI), take mean across time dimension
            logger.debug(f"4D NIfTI detected {data.shape}, computing temporal mean")
            data = data.mean(axis=-1)
        if data.max() == data.min():
            raise ValueError(f"NIfTI appears empty (uniform intensity): {path}")

    return data, affine


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    path: Union[str, Path],
    header: Optional[nib.Nifti1Header] = None,
) -> None:
    """
    Save a numpy array as a compressed NIfTI file.

    Parameters
    ----------
    data : np.ndarray
        3D volume array.
    affine : np.ndarray
        4×4 voxel-to-world affine.
    path : str or Path
        Output path (should end in .nii.gz for compression).
    header : Nifti1Header, optional
        Header to copy metadata from (pixdim, description, etc.).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = nib.Nifti1Image(data.astype(np.float32), affine, header=header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(path))
    logger.debug(f"Saved NIfTI: {path} shape={data.shape}")


def get_voxel_size(affine: np.ndarray) -> np.ndarray:
    """Extract voxel size in mm from an affine matrix."""
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def resample_volume(
    data: np.ndarray,
    affine: np.ndarray,
    target_voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    interpolation: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a volume to target isotropic voxel size.

    We use nibabel's resample_to_output which correctly handles the affine
    and preserves MNI coordinate correspondence.

    Parameters
    ----------
    data : np.ndarray, shape (D, H, W)
        Input volume.
    affine : np.ndarray, shape (4, 4)
        Input affine.
    target_voxel_size : tuple of float
        Target voxel size in mm, e.g. (2.0, 2.0, 2.0).
    interpolation : str
        "linear" (default) | "nearest" | "cubic"

    Returns
    -------
    resampled_data : np.ndarray
    new_affine : np.ndarray
    """
    order = _INTERP_MAP.get(interpolation, 1)
    img = nib.Nifti1Image(data.astype(np.float32), affine)

    resampled = resample_to_output(
        img,
        voxel_sizes=target_voxel_size,
        order=order,
        mode="constant",
        cval=0.0,
    )

    new_data = resampled.get_fdata(dtype=np.float32)
    new_affine = resampled.affine
    logger.debug(f"Resampled: {data.shape} -> {new_data.shape} "
                 f"(voxel size: {get_voxel_size(new_affine).round(2)} mm)")
    return new_data, new_affine


def crop_to_nonzero(
    data: np.ndarray,
    margin: int = 5,
) -> Tuple[np.ndarray, Tuple[slice, ...]]:
    """
    Crop a volume to the tight bounding box of nonzero voxels.

    Parameters
    ----------
    data : np.ndarray
        Input volume (brain mask or intensity volume).
    margin : int
        Extra voxels to include around the bounding box.

    Returns
    -------
    cropped_data : np.ndarray
    slices : tuple of slices
        The slice indices used; apply to other volumes (e.g. mask) for
        consistency.
    """
    nonzero = np.argwhere(data > 0)
    if len(nonzero) == 0:
        return data, tuple(slice(None) for _ in range(data.ndim))

    mins = nonzero.min(axis=0)
    maxs = nonzero.max(axis=0)

    slices = tuple(
        slice(max(0, mn - margin), min(sz, mx + margin + 1))
        for mn, mx, sz in zip(mins, maxs, data.shape)
    )
    return data[slices], slices


def pad_or_crop_to_shape(
    data: np.ndarray,
    target_shape: Tuple[int, int, int],
    mode: str = "constant",
    constant_value: float = 0.0,
) -> np.ndarray:
    """
    Pad or center-crop a volume to exactly target_shape.

    All preprocessing pipelines must output the same spatial shape for
    batching.  This function handles the general case where the input may
    be larger or smaller than the target in any dimension.

    Parameters
    ----------
    data : np.ndarray, shape (D, H, W)
        Input volume.
    target_shape : tuple of int
        Desired output shape (D, H, W).
    mode : str
        numpy.pad mode for padding: "constant" | "reflect" | "edge"
    constant_value : float
        Padding value when mode="constant".

    Returns
    -------
    np.ndarray, shape == target_shape
    """
    result = data.copy()

    # Step 1: Crop dimensions that are too large (center crop)
    for dim in range(3):
        if result.shape[dim] > target_shape[dim]:
            excess = result.shape[dim] - target_shape[dim]
            start = excess // 2
            slc = [slice(None)] * 3
            slc[dim] = slice(start, start + target_shape[dim])
            result = result[tuple(slc)]

    # Step 2: Pad dimensions that are too small
    pad_width = []
    for dim in range(3):
        deficit = target_shape[dim] - result.shape[dim]
        if deficit > 0:
            before = deficit // 2
            after = deficit - before
            pad_width.append((before, after))
        else:
            pad_width.append((0, 0))

    if any(p[0] > 0 or p[1] > 0 for p in pad_width):
        kwargs = {"constant_values": constant_value} if mode == "constant" else {}
        result = np.pad(result, pad_width, mode=mode, **kwargs)

    assert result.shape == tuple(target_shape), (
        f"Shape mismatch after pad/crop: got {result.shape}, expected {target_shape}"
    )
    return result


def apply_brain_mask(
    data: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Zero out voxels outside the brain mask.

    Parameters
    ----------
    data : np.ndarray
        Intensity volume.
    mask : np.ndarray
        Binary brain mask (1=brain, 0=background). Must match data.shape.

    Returns
    -------
    np.ndarray
        Masked volume.
    """
    if data.shape != mask.shape:
        raise ValueError(f"Data shape {data.shape} != mask shape {mask.shape}")
    return data * (mask > 0).astype(data.dtype)


def compute_brain_volume(mask: np.ndarray, voxel_size_mm: float = 2.0) -> float:
    """
    Compute brain volume in cm³ from a binary mask.

    Used as a QC metric — extreme values indicate failed skull stripping.

    Parameters
    ----------
    mask : np.ndarray
        Binary brain mask.
    voxel_size_mm : float
        Isotropic voxel size in mm.

    Returns
    -------
    float
        Brain volume in cm³.
    """
    n_voxels = (mask > 0).sum()
    voxel_vol_cm3 = (voxel_size_mm / 10.0) ** 3  # mm³ -> cm³
    return n_voxels * voxel_vol_cm3


def nifti_to_numpy_batch(
    paths: list,
    target_shape: Tuple[int, int, int],
    target_voxel_size: Tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> np.ndarray:
    """
    Load and preprocess a list of NIfTI paths into a batch array.

    Convenience function for quick batch loading in notebooks.
    For the training pipeline, use the Dataset class instead.

    Parameters
    ----------
    paths : list of str/Path
        NIfTI file paths.
    target_shape : tuple
        Output spatial shape.
    target_voxel_size : tuple
        Resample to this voxel size first.

    Returns
    -------
    np.ndarray, shape (N, D, H, W)
    """
    batch = []
    for path in paths:
        data, affine = load_nifti(path)
        data, _ = resample_volume(data, affine, target_voxel_size)
        data = pad_or_crop_to_shape(data, target_shape)
        batch.append(data)
    return np.stack(batch, axis=0)
