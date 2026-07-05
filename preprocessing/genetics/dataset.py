"""
PyTorch Dataset for preprocessed genetic features.

Multi-modal alignment challenge
---------------------------------
The genetic Dataset must align with the MRI Dataset by subject ID so that
`train_ds[i]` always returns the same subject across both modalities.

When MRI and genetics come from different cohorts (the common case), we
support two alignment modes:
  1. "paired"    — only subjects with BOTH modalities; used for end-to-end fusion
  2. "unpaired"  — each modality trains independently; fusion via learned alignment
  3. "metadata"  — use ABIDE phenotypic features (age, sex, IQ, site) as the
                    "genetics" branch; common shortcut in the literature

The metadata proxy (mode 3) is scientifically valid when:
  - The fusion model learns complementary representations from MRI and covariates
  - Reported as such in the Methods section (not presented as true genetics)
  - Compared against a genetics-only baseline

Usage
-----
    from preprocessing.genetics.dataset import GeneticsDataset, MetadataProxyDataset

    # Real genetics data
    gen_ds = GeneticsDataset(features_df, metadata_df)

    # Phenotypic features as genetics proxy
    proxy_ds = MetadataProxyDataset(abide_metadata_df, feature_cols=["age", "sex_encoded"])
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class GeneticsDataset(Dataset):
    """
    PyTorch Dataset for gene expression features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Shape (n_genes, n_samples). Columns are subject IDs.
    metadata_df : pd.DataFrame
        Must have 'subject_id' and 'label' columns.
    scaler : fitted sklearn-style scaler, optional
        For feature-level scaling (RobustScaler recommended for genes).
    subject_order : list of str, optional
        If provided, reindexes data to match this subject order.
        Critical for paired multi-modal datasets.
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        scaler=None,
        subject_order: Optional[List[str]] = None,
    ) -> None:
        meta = metadata_df.copy()

        # Accept either 'subject_id' (ABIDE) or 'sample_id' (GEO)
        id_col = None
        for candidate in ("subject_id", "sample_id"):
            if candidate in meta.columns:
                id_col = candidate
                break
        if id_col is None:
            raise KeyError(
                "metadata_df must have a 'subject_id' or 'sample_id' column"
            )
        if id_col != "subject_id":
            meta = meta.rename(columns={id_col: "subject_id"})

        # Align subjects between features and metadata
        available_subjects = [
            s for s in meta["subject_id"].astype(str)
            if s in features_df.columns
        ]

        if len(available_subjects) == 0:
            raise ValueError(
                "No subject IDs match between features_df columns "
                f"and metadata_df['{id_col}']. Check ID formatting."
            )

        if subject_order is not None:
            # Respect external ordering for paired multi-modal alignment
            available_subjects = [s for s in subject_order if s in available_subjects]

        meta = meta[meta["subject_id"].astype(str).isin(available_subjects)]
        meta = meta.reset_index(drop=True)

        # Extract features in subject order
        feat_sub = features_df[available_subjects]  # (n_genes, n_subjects)
        features_array = feat_sub.values.T.astype(np.float32)  # (n_subjects, n_genes)

        # Optional scaling
        if scaler is not None:
            features_array = scaler.transform(features_array).astype(np.float32)

        self._features = features_array  # (n_subjects, n_genes)
        self._labels = meta["label"].values.astype(np.int64)
        self._subject_ids = available_subjects
        self._metadata = meta

        n_missing = len(metadata_df) - len(available_subjects)
        if n_missing > 0:
            logger.warning(
                f"{n_missing} subjects in metadata have no genetic data"
            )
        logger.info(
            f"GeneticsDataset: {len(self)} subjects, "
            f"{self._features.shape[1]} features, "
            f"ASD={self._labels.sum()} TC={(self._labels==0).sum()}"
        )

    def __len__(self) -> int:
        return len(self._subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            "genetics"  : torch.Tensor, shape (n_genes,)
            "label"     : torch.Tensor, scalar int64
            "subject"   : str
        """
        return {
            "genetics": torch.from_numpy(self._features[idx]),
            "label": torch.tensor(self._labels[idx], dtype=torch.long),
            "subject": self._subject_ids[idx],
        }

    def get_feature_matrix(self) -> np.ndarray:
        """Return full feature matrix (n_subjects, n_genes) for batch operations."""
        return self._features.copy()

    def get_class_weights(self) -> torch.Tensor:
        counts = np.bincount(self._labels, minlength=2)
        weights_per_class = 1.0 / (counts + 1e-6)
        return torch.tensor(
            [weights_per_class[lbl] for lbl in self._labels],
            dtype=torch.float32,
        )


class MetadataProxyDataset(Dataset):
    """
    Uses ABIDE phenotypic covariates as a genetics proxy.

    This is a documented, methodologically sound approach when true
    genetics data is not available for ABIDE subjects.  The fusion model
    learns to combine imaging features with demographic/clinical features.

    Features included (configurable):
      - age_at_scan (continuous)
      - sex_encoded (0=male, 1=female)
      - full_iq (continuous, normalized)
      - site_encoded (one-hot or label-encoded)
      - handedness (categorical)

    Parameters
    ----------
    metadata_df : pd.DataFrame
        ABIDE metadata with phenotypic columns.
    feature_cols : list of str
        Columns to use as features. If None, uses default set.
    normalize : bool
        Apply standard normalization to continuous features.
    """

    _DEFAULT_COLS = [
        "age", "sex_encoded", "full_iq", "verbal_iq", "performance_iq"
    ]

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
        subject_order: Optional[List[str]] = None,
    ) -> None:
        if feature_cols is None:
            feature_cols = [c for c in self._DEFAULT_COLS if c in metadata_df.columns]

        meta = metadata_df.copy().reset_index(drop=True)

        # Accept either 'subject_id' (ABIDE) or 'sample_id' (GEO)
        for candidate in ("subject_id", "sample_id"):
            if candidate in meta.columns and candidate != "subject_id":
                meta = meta.rename(columns={candidate: "subject_id"})
                break

        if subject_order is not None:
            sid_to_idx = {str(sid): i for i, sid in enumerate(meta["subject_id"])}
            ordered_idx = [sid_to_idx[s] for s in subject_order if s in sid_to_idx]
            meta = meta.iloc[ordered_idx].reset_index(drop=True)

        # Extract and encode features
        X = self._encode_features(meta, feature_cols, normalize)

        self._features = X.astype(np.float32)
        self._labels = meta["label"].values.astype(np.int64)
        self._subject_ids = meta["subject_id"].astype(str).tolist()
        self.feature_dim = X.shape[1]

        logger.info(
            f"MetadataProxyDataset: {len(self)} subjects, "
            f"{self.feature_dim} phenotypic features: {feature_cols}"
        )

    def __len__(self) -> int:
        return len(self._subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "genetics": torch.from_numpy(self._features[idx]),
            "label": torch.tensor(self._labels[idx], dtype=torch.long),
            "subject": self._subject_ids[idx],
        }

    @staticmethod
    def _encode_features(
        meta: pd.DataFrame,
        feature_cols: List[str],
        normalize: bool,
    ) -> np.ndarray:
        """Extract and normalize the feature columns."""
        parts = []
        for col in feature_cols:
            if col not in meta.columns:
                continue
            vals = pd.to_numeric(meta[col], errors="coerce")
            vals = vals.fillna(vals.median())
            parts.append(vals.values.reshape(-1, 1))

        # One-hot encode site if available
        if "site_name" in meta.columns:
            site_dummies = pd.get_dummies(
                meta["site_name"].astype(str), prefix="site"
            ).astype(float)
            parts.append(site_dummies.values)

        if not parts:
            raise ValueError("No valid feature columns found in metadata")

        X = np.hstack(parts).astype(np.float64)

        if normalize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X = scaler.fit_transform(X)

        return X


# ---------------------------------------------------------------------------
# Paired multi-modal dataset
# ---------------------------------------------------------------------------

class PairedMultiModalDataset(Dataset):
    """
    Returns aligned (MRI, genetics, label) triplets from paired datasets.

    This is the dataset used by the fusion model.  It aligns the MRI and
    genetics datasets by subject ID, keeping only subjects present in both.

    Parameters
    ----------
    mri_dataset : MRIDataset
    genetics_dataset : GeneticsDataset or MetadataProxyDataset
    """

    def __init__(self, mri_dataset, genetics_dataset) -> None:
        from preprocessing.mri.dataset import MRIDataset

        # Build subject_id -> index maps
        mri_ids = [
            str(mri_dataset.metadata.loc[i, "subject_id"])
            for i in range(len(mri_dataset))
        ]
        gen_ids = genetics_dataset._subject_ids

        # Find common subjects
        common = [sid for sid in mri_ids if sid in set(gen_ids)]

        if len(common) == 0:
            logger.warning(
                "No common subjects between MRI and genetics datasets. "
                "Check subject ID formatting. Falling back to index alignment."
            )
            common_len = min(len(mri_dataset), len(genetics_dataset))
            self._mri_indices = list(range(common_len))
            self._gen_indices = list(range(common_len))
        else:
            mri_id_to_idx = {sid: i for i, sid in enumerate(mri_ids)}
            gen_id_to_idx = {sid: i for i, sid in enumerate(gen_ids)}
            self._mri_indices = [mri_id_to_idx[s] for s in common]
            self._gen_indices = [gen_id_to_idx[s] for s in common]

        self._mri_dataset = mri_dataset
        self._gen_dataset = genetics_dataset
        self._common_subjects = common

        logger.info(
            f"PairedMultiModalDataset: {len(self)} paired subjects "
            f"(MRI={len(mri_dataset)}, genetics={len(genetics_dataset)})"
        )

    def __len__(self) -> int:
        return len(self._common_subjects)

    def __getitem__(self, idx: int) -> Dict:
        mri_item = self._mri_dataset[self._mri_indices[idx]]
        gen_item = self._gen_dataset[self._gen_indices[idx]]

        return {
            "image": mri_item["image"],
            "genetics": gen_item["genetics"],
            "label": mri_item["label"],
            "subject": mri_item["subject"],
            "site": mri_item.get("site", "unknown"),
        }
