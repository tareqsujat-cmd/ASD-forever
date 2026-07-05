"""
PyTorch Dataset and DataLoader factory for preprocessed MRI volumes.

Design choices
--------------
1. Lazy loading: volumes are loaded from disk on demand, not pre-loaded into
   RAM.  A single ABIDE subject at 96^3 float32 = 3.5 MB; 1112 subjects =
   3.9 GB.  GPU training would compete with RAM for this data.
   Solution: cache to a memory-mapped HDF5 file on first access.

2. Caching strategy: first epoch loads from .nii.gz (slow), subsequent
   epochs load from an HDF5 cache (fast).  Cache invalidated when
   preprocessing parameters change (detected via config hash).

3. Channel dimension: outputs shape (1, D, H, W) — 3D CNNs expect an
   explicit channel dimension.

4. Balanced sampling: for class-imbalanced ABIDE subsets, we use a
   WeightedRandomSampler so each batch has ~equal ASD/TC samples without
   oversampling (which would duplicate subjects and inflate performance).

Usage
-----
    from preprocessing.mri.dataset import MRIDataset, build_dataloaders
    train_ds = MRIDataset(metadata_df, processed_dir, augment=True, cfg=cfg)
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, cfg)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class MRIDataset(Dataset):
    """
    PyTorch Dataset for preprocessed 3D MRI volumes.

    Parameters
    ----------
    metadata : pd.DataFrame
        Must have columns: 'subject_id', 'label', 'processed_path' (or
        we construct paths from processed_dir + subject_id).
    processed_dir : str or Path
        Directory containing preprocessed .nii.gz files (one per subject).
    augmentor : MRIAugmentor, optional
        Augmentation object.  Pass None for val/test.
    target_shape : tuple
        Expected spatial shape (D, H, W) after preprocessing.
    cache_dir : str or Path, optional
        Directory to store HDF5 cache.  Skips caching if None.
    transform : callable, optional
        Additional transform applied to the tensor (e.g., normalization).
    dtype : torch.dtype
        Output tensor dtype.  float32 is standard.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        processed_dir: str | Path,
        augmentor=None,
        target_shape: Tuple[int, int, int] = (96, 96, 96),
        cache_dir: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.processed_dir = Path(processed_dir)
        self.augmentor = augmentor
        self.target_shape = target_shape
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform
        self.dtype = dtype

        # Build subject-path index
        self._paths = self._resolve_paths()
        self._labels = metadata["label"].values.astype(np.int64)

        # HDF5 cache handle (opened on first use)
        self._cache = None
        self._cache_key = self._compute_cache_key()

        logger.info(
            f"MRIDataset: {len(self)} subjects | "
            f"ASD={self._labels.sum()} TC={(self._labels==0).sum()} | "
            f"shape={target_shape} | augment={augmentor is not None}"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a single subject's data.

        Returns
        -------
        dict with keys:
            "image"   : torch.Tensor, shape (1, D, H, W)
            "label"   : torch.Tensor, scalar int64
            "subject" : str, subject ID
            "site"    : str, site name (for grouped analysis)
        """
        label = int(self._labels[idx])
        subject_id = str(self.metadata.loc[idx, "subject_id"])
        site = str(self.metadata.loc[idx, "site_name"]) \
            if "site_name" in self.metadata.columns else "unknown"

        # Try cache first
        volume = self._load_from_cache(idx)
        if volume is None:
            volume = self._load_from_disk(idx)
            self._write_to_cache(idx, volume)

        # Augmentation (training only)
        if self.augmentor is not None:
            volume = self.augmentor(volume)

        # Ensure correct shape
        from preprocessing.mri.nifti_utils import pad_or_crop_to_shape
        volume = pad_or_crop_to_shape(volume, self.target_shape)

        # Convert to tensor: add channel dim -> (1, D, H, W)
        tensor = torch.from_numpy(volume).unsqueeze(0).to(self.dtype)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "subject": subject_id,
            "site": site,
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute per-class weights for WeightedRandomSampler.

        Returns weight for each sample (len = N), where rare-class samples
        have higher weight, balancing the effective class distribution
        within each batch.
        """
        n = len(self._labels)
        counts = np.bincount(self._labels, minlength=2)
        weights_per_class = 1.0 / (counts + 1e-6)
        sample_weights = torch.tensor(
            [weights_per_class[lbl] for lbl in self._labels],
            dtype=torch.float32,
        )
        return sample_weights

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_paths(self) -> list:
        """
        Build the list of NIfTI file paths for all subjects.

        Checks for a 'processed_path' column first; falls back to
        constructing paths as processed_dir / subject_id_preprocessed.nii.gz
        """
        paths = []
        for idx, row in self.metadata.iterrows():
            if "processed_path" in self.metadata.columns and pd.notna(row.get("processed_path")):
                p = Path(str(row["processed_path"]))
            else:
                sid = str(row["subject_id"])
                p = self.processed_dir / f"{sid}_preprocessed.nii.gz"

            if not p.exists():
                logger.warning(f"Missing preprocessed file: {p}")
            paths.append(p)
        return paths

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self, idx: int) -> np.ndarray:
        """Load a volume from its NIfTI file."""
        from preprocessing.mri.nifti_utils import load_nifti, resample_volume, pad_or_crop_to_shape
        path = self._paths[idx]
        try:
            data, affine = load_nifti(path)
            # Defensive resample in case preprocessing was done at different resolution
            current_vox = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).mean()
            if abs(current_vox - 2.0) > 0.1:
                data, _ = resample_volume(data, affine, (2.0, 2.0, 2.0))
            data = pad_or_crop_to_shape(data, self.target_shape)
            return data.astype(np.float32)
        except Exception as exc:
            logger.error(f"Failed to load {path}: {exc}")
            # Return zeros so training does not crash; QC should have caught this
            return np.zeros(self.target_shape, dtype=np.float32)

    def _load_from_cache(self, idx: int) -> Optional[np.ndarray]:
        """Try to load a preprocessed volume from the HDF5 cache."""
        if self.cache_dir is None:
            return None
        cache_path = self.cache_dir / f"mri_cache_{self._cache_key}.h5"
        if not cache_path.exists():
            return None

        try:
            import h5py
            if self._cache is None:
                self._cache = h5py.File(cache_path, "r")
            key = str(idx)
            if key in self._cache:
                return self._cache[key][:]
        except Exception as exc:
            logger.debug(f"Cache read failed for idx={idx}: {exc}")
        return None

    def _write_to_cache(self, idx: int, data: np.ndarray) -> None:
        """Write a volume to the HDF5 cache."""
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"mri_cache_{self._cache_key}.h5"
        try:
            import h5py
            with h5py.File(cache_path, "a") as f:
                key = str(idx)
                if key not in f:
                    f.create_dataset(key, data=data, compression="lzf")
        except Exception as exc:
            logger.debug(f"Cache write failed for idx={idx}: {exc}")

    def _compute_cache_key(self) -> str:
        """Hash the dataset parameters to detect cache invalidation."""
        sig = f"{self.target_shape}_{len(self.metadata)}_{sorted(self.metadata['subject_id'].tolist())}"
        return hashlib.md5(sig.encode()).hexdigest()[:8]

    def __del__(self):
        if self._cache is not None:
            try:
                self._cache.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    train_dataset: MRIDataset,
    val_dataset: MRIDataset,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders.

    Parameters
    ----------
    train_dataset : MRIDataset
    val_dataset : MRIDataset
    batch_size : int
    num_workers : int
        Number of parallel data loading workers. 4 is a good default;
        increase to 8 on machines with fast NVMe storage.
    pin_memory : bool
        Pin memory for faster CPU-to-GPU transfers.
    use_weighted_sampler : bool
        Balance class frequencies per batch.
    seed : int
        Worker seed for reproducibility.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    """
    from utilities.reproducibility import get_worker_init_fn

    worker_init = get_worker_init_fn(seed)

    # Training: optionally balanced sampling
    train_sampler = None
    if use_weighted_sampler:
        sample_weights = train_dataset.get_class_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        drop_last=True,   # avoid partial batches in BN layers
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    logger.info(
        f"DataLoaders built: train={len(train_loader)} batches "
        f"(bs={batch_size}), val={len(val_loader)} batches"
    )
    return train_loader, val_loader
