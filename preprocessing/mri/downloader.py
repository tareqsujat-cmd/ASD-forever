"""
ABIDE I/II dataset downloader.

Wraps nilearn's fetch_abide_pcp with:
  - Automatic retry on network failure
  - Metadata extraction and CSV generation
  - Subject-level file validation
  - Support for multiple preprocessing pipelines

Scientific note
---------------
ABIDE distributes two kinds of preprocessed data:
  1. Functional (fMRI): resting-state timeseries, ROI connectivity matrices.
     Available via nilearn directly.
  2. Structural (T1w): raw NIfTI that must be preprocessed locally.

For our structural MRI branch we download raw T1w and run our own pipeline.
This gives us full methodological control, which reviewers require.

Usage
-----
    from preprocessing.mri.downloader import ABIDEDownloader
    dl = ABIDEDownloader(data_dir="datasets/raw/mri", version="ABIDE_I")
    metadata = dl.download(pipeline="cpac", strategy="filt_global")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ABIDE phenotypic columns we care about
_PHENO_COLS = [
    "subject",
    "site_id",
    "site_name",
    "dx_group",        # 1=ASD, 2=TC
    "sex",             # 1=Male, 2=Female
    "age",
    "handedness_category",
    "eye_status_at_scan",
    "full_iq",
    "verbal_iq",
    "performance_iq",
]


class ABIDEDownloader:
    """
    Downloads and validates ABIDE I or ABIDE II data.

    Parameters
    ----------
    data_dir : str or Path
        Root directory for downloaded files.
    version : str
        "ABIDE_I" or "ABIDE_II"
    """

    def __init__(self, data_dir: str | Path, version: str = "ABIDE_I") -> None:
        self.data_dir = Path(data_dir)
        self.version = version.upper()
        if self.version not in ("ABIDE_I", "ABIDE_II"):
            raise ValueError(f"version must be ABIDE_I or ABIDE_II, got {version}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ABIDEDownloader initialized: {self.data_dir} ({self.version})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(
        self,
        pipeline: str = "cpac",
        strategy: str = "filt_global",
        derivatives: Optional[List[str]] = None,
        band_pass_filtering: bool = True,
        global_signal_regression: bool = True,
        quality_checked: bool = True,
        n_subjects: Optional[int] = None,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Download ABIDE preprocessed data and return phenotypic metadata.

        Parameters
        ----------
        pipeline : str
            Preprocessing pipeline: "cpac" | "dpabi" | "css" | "niak"
        strategy : str
            Denoising strategy: "filt_global" | "nofilt_global" | etc.
        derivatives : list of str, optional
            Which derivatives to download. Defaults to ["func_preproc"].
        band_pass_filtering : bool
            Whether to use band-pass filtered data.
        global_signal_regression : bool
            Whether to use global signal regression.
        quality_checked : bool
            If True, only download subjects that passed ABIDE QC.
        n_subjects : int, optional
            Limit for testing (None = all subjects).
        max_retries : int
            Number of download retry attempts.

        Returns
        -------
        pd.DataFrame
            Phenotypic metadata with file paths for downloaded subjects.
        """
        if derivatives is None:
            derivatives = ["func_preproc"]

        logger.info(f"Downloading {self.version} | pipeline={pipeline} | "
                    f"strategy={strategy} | n_subjects={n_subjects}")

        pheno_df = self._download_with_retry(
            pipeline=pipeline,
            strategy=strategy,
            derivatives=derivatives,
            band_pass_filtering=band_pass_filtering,
            global_signal_regression=global_signal_regression,
            quality_checked=quality_checked,
            n_subjects=n_subjects,
            max_retries=max_retries,
        )

        # Standardize the metadata
        pheno_df = self._standardize_metadata(pheno_df)

        # Validate downloaded files exist
        pheno_df = self._validate_files(pheno_df, derivatives)

        # Save metadata to CSV
        out_csv = self.data_dir / f"{self.version}_metadata.csv"
        pheno_df.to_csv(out_csv, index=False)
        logger.info(f"Metadata saved: {out_csv} ({len(pheno_df)} subjects)")

        return pheno_df

    def load_metadata(self) -> pd.DataFrame:
        """Load previously downloaded metadata CSV."""
        csv_path = self.data_dir / f"{self.version}_metadata.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {csv_path}. Run download() first."
            )
        return pd.read_csv(csv_path)

    def get_class_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Return per-site and overall class distribution."""
        overall = df["label"].value_counts().to_dict()
        per_site = {}
        for site, grp in df.groupby("site_name"):
            per_site[site] = grp["label"].value_counts().to_dict()
        logger.info(f"Overall distribution: {overall}")
        return {"overall": overall, "per_site": per_site}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_with_retry(
        self,
        pipeline: str,
        strategy: str,
        derivatives: List[str],
        band_pass_filtering: bool,
        global_signal_regression: bool,
        quality_checked: bool,
        n_subjects: Optional[int],
        max_retries: int,
    ) -> pd.DataFrame:
        """Attempt download with exponential back-off retry."""
        try:
            from nilearn.datasets import fetch_abide_pcp
        except ImportError as e:
            raise ImportError(
                "nilearn is required for ABIDE download. "
                "Install with: pip install nilearn"
            ) from e

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Download attempt {attempt}/{max_retries}")
                dataset = fetch_abide_pcp(
                    data_dir=str(self.data_dir),
                    pipeline=pipeline,
                    band_pass_filtering=band_pass_filtering,
                    global_signal_regression=global_signal_regression,
                    derivatives=derivatives,
                    quality_checked=quality_checked,
                    n_subjects=n_subjects,
                    verbose=1,
                )
                pheno = dataset["phenotypic"]
                # Attach file paths
                for deriv in derivatives:
                    if deriv in dataset:
                        paths = dataset[deriv]
                        if len(paths) == len(pheno):
                            pheno[f"{deriv}_path"] = paths
                return pheno

            except Exception as exc:
                logger.warning(f"Download attempt {attempt} failed: {exc}")
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.info(f"Retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Download failed after {max_retries} attempts"
                    ) from exc

        # Should never reach here
        raise RuntimeError("Unexpected download failure")

    def _standardize_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and encode labels.

        ABIDE uses 1=ASD, 2=TC. We convert to 0=TC, 1=ASD for
        standard binary classification convention.
        """
        df = df.copy()

        # Rename columns to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Standardize the label column
        # DX_GROUP: 1=Autism, 2=Control
        label_col = None
        for candidate in ["dx_group", "diagnosis", "label"]:
            if candidate in df.columns:
                label_col = candidate
                break

        if label_col is None:
            raise KeyError("No diagnosis column found in ABIDE metadata. "
                           "Expected 'dx_group', 'diagnosis', or 'label'.")

        # 1=ASD -> 1, 2=TC -> 0
        df["label"] = (df[label_col] == 1).astype(int)
        df["dx_label"] = df["label"].map({1: "ASD", 0: "TC"})

        # Standardize site column
        for candidate in ["site_id", "site", "scan_site_id"]:
            if candidate in df.columns:
                df["site_name"] = df[candidate].astype(str).str.strip()
                break

        # Ensure subject ID column
        for candidate in ["subject", "sub_id", "subject_id", "participant_id"]:
            if candidate in df.columns:
                df["subject_id"] = df[candidate].astype(str).str.strip()
                break

        # Encode sex: 1=Male->0, 2=Female->1
        if "sex" in df.columns:
            df["sex_encoded"] = (df["sex"] == 2).astype(int)

        # Age as float
        if "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")

        logger.info(f"Metadata standardized: {len(df)} subjects, "
                    f"ASD={df['label'].sum()}, TC={(df['label']==0).sum()}")
        return df

    def _validate_files(self, df: pd.DataFrame, derivatives: List[str]) -> pd.DataFrame:
        """
        Remove subjects whose downloaded files are missing or corrupted.

        A missing file is a silent failure mode that causes confusing errors
        later in the pipeline, so we catch it here explicitly.
        """
        initial_count = len(df)
        valid_mask = pd.Series(True, index=df.index)

        for deriv in derivatives:
            path_col = f"{deriv}_path"
            if path_col not in df.columns:
                logger.warning(f"No path column for derivative '{deriv}'")
                continue

            for idx, path in df[path_col].items():
                if path is None or not Path(str(path)).exists():
                    logger.warning(f"Missing file for subject {df.loc[idx, 'subject_id']}: {path}")
                    valid_mask[idx] = False
                elif Path(str(path)).stat().st_size < 1024:
                    # Suspiciously small NIfTI (< 1 KB) indicates failed download
                    logger.warning(f"Suspect file size for {path}: "
                                   f"{Path(str(path)).stat().st_size} bytes")
                    valid_mask[idx] = False

        df = df[valid_mask].reset_index(drop=True)
        n_removed = initial_count - len(df)
        if n_removed > 0:
            logger.warning(f"Removed {n_removed} subjects with missing/corrupt files")
        logger.info(f"Valid subjects after file validation: {len(df)}")
        return df


def create_subject_splits(
    metadata: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    n_folds: int = 5,
    group_by_site: bool = True,
    random_seed: int = 42,
    save_dir: Optional[str | Path] = None,
) -> Dict:
    """
    Create stratified train/val/test splits with optional site grouping.

    This is the most scientifically important splitting function in the
    pipeline.  group_by_site=True ensures no site appears in both train
    and test, preventing the model from learning site-specific MRI artifacts
    as ASD signal — the most common methodological flaw in ABIDE papers.

    Parameters
    ----------
    metadata : pd.DataFrame
        Standardized ABIDE metadata with 'label' and 'site_name' columns.
    test_size : float
        Fraction of subjects for the final held-out test set.
    val_size : float
        Fraction of non-test subjects for validation.
    n_folds : int
        Number of cross-validation folds on the training set.
    group_by_site : bool
        If True, all subjects from a site go to the same fold.
    random_seed : int
        Master seed (derived per operation internally).
    save_dir : str or Path, optional
        Directory to save split CSVs.

    Returns
    -------
    dict with keys: "test", "folds" (list of {"train", "val"} dicts)
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    labels = metadata["label"].values
    sites = metadata["site_name"].values if "site_name" in metadata.columns else None

    # ------------------------------------------------------------------
    # Step 1: Hold out test set
    # ------------------------------------------------------------------
    n = len(metadata)
    n_test = int(n * test_size)

    rng = np.random.default_rng(random_seed)

    if group_by_site and sites is not None:
        unique_sites = np.unique(sites)
        rng.shuffle(unique_sites)

        # Pick sites until we have enough test subjects
        test_sites = []
        test_count = 0
        for site in unique_sites:
            site_mask = sites == site
            if test_count < n_test:
                test_sites.append(site)
                test_count += site_mask.sum()

        test_mask = np.isin(sites, test_sites)
        test_idx = np.where(test_mask)[0]
        trainval_idx = np.where(~test_mask)[0]

        logger.info(f"Test set: {len(test_idx)} subjects from sites {test_sites}")
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_seed
        )
        trainval_idx, test_idx = next(sss.split(np.zeros(n), labels))

    # ------------------------------------------------------------------
    # Step 2: Cross-validation on train+val
    # ------------------------------------------------------------------
    trainval_labels = labels[trainval_idx]
    trainval_sites = sites[trainval_idx] if sites is not None else None

    if group_by_site and trainval_sites is not None:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        fold_splits = list(cv.split(
            np.zeros(len(trainval_idx)), trainval_labels, groups=trainval_sites
        ))
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        fold_splits = list(cv.split(np.zeros(len(trainval_idx)), trainval_labels))

    folds = []
    for fold_i, (train_local, val_local) in enumerate(fold_splits):
        train_global = trainval_idx[train_local]
        val_global = trainval_idx[val_local]

        fold_asd = labels[train_global].sum()
        fold_tc = (labels[train_global] == 0).sum()
        logger.info(f"Fold {fold_i + 1}: train={len(train_global)} "
                    f"(ASD={fold_asd}, TC={fold_tc}), val={len(val_global)}")

        folds.append({
            "fold": fold_i,
            "train_idx": train_global.tolist(),
            "val_idx": val_global.tolist(),
        })

    splits = {"test_idx": test_idx.tolist(), "folds": folds}

    # ------------------------------------------------------------------
    # Step 3: Persist splits
    # ------------------------------------------------------------------
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        import json
        out_json = save_dir / "splits.json"
        with open(out_json, "w") as f:
            json.dump(splits, f, indent=2)

        # Also save readable per-fold CSVs
        test_df = metadata.iloc[test_idx][["subject_id", "site_name", "label"]].copy()
        test_df.to_csv(save_dir / "test_subjects.csv", index=False)

        for fold_info in folds:
            fi = fold_info["fold"]
            train_df = metadata.iloc[fold_info["train_idx"]][
                ["subject_id", "site_name", "label"]
            ].copy()
            val_df = metadata.iloc[fold_info["val_idx"]][
                ["subject_id", "site_name", "label"]
            ].copy()
            train_df.to_csv(save_dir / f"fold{fi}_train.csv", index=False)
            val_df.to_csv(save_dir / f"fold{fi}_val.csv", index=False)

        logger.info(f"Splits saved to {save_dir}")

    return splits
