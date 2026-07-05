"""
PyTorch Dataset and DataLoader for preprocessed ABIDE I data.

Reads from the output of data/preprocess_abide.py:
    abide_processed/
        mri/metadata.csv   — subject_id, label, site, split
        mri/<id>.npy       — (19900,) FC upper-triangle vector (float32)
        gen/<id>.npy       — (6,) phenotypic features (float32)

MRI branch reshape
------------------
The existing 3D ResNet backbone expects input (B, 1, D, H, W).
We reshape the flat (19900,) FC vector by zero-padding to 21952 = 28³
and reshaping to (1, 28, 28, 28).  The network's AdaptiveAvgPool3d(1)
makes it spatial-size agnostic, so this works without any model changes.

For future work, swapping to a 2D CNN on (1, 200, 200) or a 1D CNN
on (1, 19900) would be more principled, but requires backbone changes.

Usage
-----
    from data.abide_dataset import make_loaders
    loaders = make_loaders(cfg)
    for mri, gen, labels in loaders["train"]:
        # mri:    (B, 1, 28, 28, 28)  float32 on device
        # gen:    (B, 6)               float32 on device
        # labels: (B,)                 long on device
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Reshape constants
# ---------------------------------------------------------------------------

FC_DIM      = 19900             # upper-triangle of 200×200 matrix
CUBE_SIDE   = 28                # 28³ = 21952  ≥  FC_DIM
CUBE_VOL    = CUBE_SIDE ** 3   # 21952
PHENO_DIM   = 6


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ABIDEDataset(Dataset):
    """
    ABIDE I dataset for the ASD multimodal framework.

    Parameters
    ----------
    mri_dir   : Path to abide_processed/mri/
    gen_dir   : Path to abide_processed/gen/
    split     : "train" | "val" | "test" | "all"
    augment   : Whether to apply training augmentation to the FC vector
    """

    def __init__(
        self,
        mri_dir:  Path,
        gen_dir:  Path,
        split:    str  = "train",
        augment:  bool = False,
    ) -> None:
        self.mri_dir = Path(mri_dir)
        self.gen_dir = Path(gen_dir)
        self.augment = augment

        meta = pd.read_csv(self.mri_dir / "metadata.csv")
        if split != "all":
            meta = meta[meta["split"] == split].reset_index(drop=True)

        if len(meta) == 0:
            raise ValueError(
                f"No subjects found for split='{split}'. "
                f"Available splits: {pd.read_csv(self.mri_dir / 'metadata.csv')['split'].unique().tolist()}"
            )

        self.subject_ids: list = meta["subject_id"].astype(str).tolist()
        self.labels:      list = meta["label"].tolist()
        self.sites:       list = meta["site"].tolist()

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sid = self.subject_ids[idx]

        # ---- Load FC vector (19900,) ----
        fc = np.load(str(self.mri_dir / f"{sid}.npy"))  # (19900,)

        if self.augment:
            fc = self._augment_fc(fc)

        # Pad to 28³ = 21952, reshape to (1, 28, 28, 28)
        padded = np.zeros(CUBE_VOL, dtype=np.float32)
        padded[:FC_DIM] = fc
        mri_tensor = torch.from_numpy(padded.reshape(1, CUBE_SIDE, CUBE_SIDE, CUBE_SIDE))

        # ---- Load phenotypic features (6,) ----
        gen = np.load(str(self.gen_dir / f"{sid}.npy"))  # (6,)
        gen_tensor = torch.from_numpy(gen.astype(np.float32))

        label = int(self.labels[idx])
        return mri_tensor, gen_tensor, label

    @staticmethod
    def _augment_fc(fc: np.ndarray) -> np.ndarray:
        """
        Light augmentation for FC vectors.

        - Gaussian noise (σ = 0.01 × std of vector) — simulates scan noise
        - Random sign flip of a small fraction of edges (5%) — mimics
          anti-correlated network variability

        These are conservative; stronger augmentation can distort connectivity.
        """
        rng  = np.random.default_rng()
        std  = float(fc.std()) or 1.0
        fc   = fc + rng.normal(0, 0.01 * std, size=fc.shape).astype(np.float32)
        if rng.random() < 0.5:
            n_flip  = max(1, int(0.05 * len(fc)))
            flip_idx = rng.choice(len(fc), n_flip, replace=False)
            fc[flip_idx] *= -1
        return fc


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(cfg: dict) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from config.

    Parameters
    ----------
    cfg : parsed config.yaml dict (top-level keys: paths, training, project)

    Returns
    -------
    dict with keys "train", "val", "test"
    """
    root     = Path(cfg["paths"]["root"])
    mri_dir  = root / cfg["paths"]["data_processed_mri"]
    gen_dir  = root / cfg["paths"]["data_processed_genetics"]
    batch    = cfg["training"]["batch_size"]
    seed     = cfg["project"]["random_seed"]
    # Windows multiprocessing with DataLoader requires num_workers=0 unless
    # the script uses if __name__ == '__main__'. Safe default is 0.
    n_workers = 0

    def _loader(split: str, augment: bool, shuffle: bool) -> DataLoader:
        ds = ABIDEDataset(mri_dir, gen_dir, split=split, augment=augment)
        g  = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size  = batch,
            shuffle     = shuffle,
            num_workers = n_workers,
            pin_memory  = cfg["project"].get("pin_memory", False) and torch.cuda.is_available(),
            drop_last   = (split == "train"),
            generator   = g if shuffle else None,
        )

    return {
        "train": _loader("train", augment=True,  shuffle=True),
        "val":   _loader("val",   augment=False, shuffle=False),
        "test":  _loader("test",  augment=False, shuffle=False),
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

def _smoke_test(cfg: dict) -> None:
    import time
    loaders = make_loaders(cfg)
    for split, loader in loaders.items():
        t0    = time.time()
        batch = next(iter(loader))
        mri, gen, labels = batch
        elapsed = time.time() - t0
        print(
            f"  {split:5s}: n={len(loader.dataset):4d} | "
            f"mri={tuple(mri.shape)} gen={tuple(gen.shape)} "
            f"labels={tuple(labels.shape)} | "
            f"first_batch={elapsed:.2f}s"
        )
        assert mri.shape[1:] == (1, CUBE_SIDE, CUBE_SIDE, CUBE_SIDE), \
            f"Unexpected MRI shape: {mri.shape}"
        assert gen.shape[1] == PHENO_DIM, \
            f"Unexpected genetics shape: {gen.shape}"
        assert set(labels.tolist()).issubset({0, 1}), \
            f"Unexpected labels: {set(labels.tolist())}"
    print("  Smoke test passed.")


if __name__ == "__main__":
    import yaml
    cfg_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print("DataLoader smoke test:")
    _smoke_test(cfg)
