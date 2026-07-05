"""
Missing value imputation for gene expression data.

Missing value sources in genomics
-----------------------------------
1. Microarray: probe hybridization failures (< 5% typical)
2. RNA-seq: zero-count genes (dropout; can be 40–60% for scRNA-seq,
   but typically < 10% for bulk RNA-seq at >5M reads)
3. SNP arrays: genotyping failures at specific loci

Strategy selection guide
------------------------
  - < 5% missing, MCAR: median imputation (fast, unbiased)
  - 5–20% missing, MAR: KNN imputation (uses gene correlation structure)
  - > 20% missing: drop the feature (set missing_threshold in config)
  - Structured missingness (all missing in one site): ComBat batch correction

Reference
---------
Troyanskaya O, et al. (2001). Missing value estimation methods for DNA
microarrays. Bioinformatics 17(6):520–525.

Usage
-----
    from preprocessing.genetics.imputation import GeneExpressionImputer
    imp = GeneExpressionImputer(method="knn", k=10)
    imp.fit(train_df)
    imputed_train = imp.transform(train_df)
    imputed_test  = imp.transform(test_df)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GeneExpressionImputer:
    """
    Fit-transform imputer for gene expression matrices.

    The fit-on-train-only API ensures imputation statistics (column medians,
    KNN neighbor graph) are never computed from test data.

    Parameters
    ----------
    method : str
        "median"      — simple per-gene median (fast, good for low missingness)
        "knn"         — k-nearest neighbors in gene correlation space
        "iterative"   — IterativeImputer (MICE-style; slow but most accurate)
        "zero"        — Replace with 0 (appropriate for RNA-seq counts)
    k : int
        Number of neighbors for KNN imputation.
    max_iter : int
        Maximum iterations for iterative imputer.
    missing_threshold : float
        Drop genes with missing rate above this threshold BEFORE imputing.
        Prevents imputing genes that are essentially missing.
    """

    def __init__(
        self,
        method: str = "knn",
        k: int = 10,
        max_iter: int = 10,
        missing_threshold: float = 0.2,
    ) -> None:
        self.method = method.lower()
        self.k = k
        self.max_iter = max_iter
        self.missing_threshold = missing_threshold
        self._imputer = None
        self._drop_mask: list = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "GeneExpressionImputer":
        """
        Fit imputer on training expression matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Shape (n_genes, n_samples) — genes as rows, samples as columns.
            NaN marks missing values.

        Returns
        -------
        self
        """
        # Drop features with too many missing values
        missing_rate = df.isna().mean(axis=1)
        self._drop_mask = missing_rate[missing_rate > self.missing_threshold].index.tolist()
        df_clean = df.drop(index=self._drop_mask)

        n_dropped = len(self._drop_mask)
        if n_dropped > 0:
            logger.info(f"Dropped {n_dropped} genes with >{self.missing_threshold*100:.0f}% "
                        f"missing values")

        # We operate on samples-as-rows for sklearn compatibility
        X = df_clean.T.values  # shape (n_samples, n_genes)

        if self.method == "median":
            # Column medians of training data
            self._medians = np.nanmedian(X, axis=0)
            logger.info(f"Median imputer fitted on {X.shape[1]} genes")

        elif self.method == "knn":
            from sklearn.impute import KNNImputer
            self._imputer = KNNImputer(n_neighbors=self.k, weights="distance")
            self._imputer.fit(X)
            logger.info(f"KNN imputer (k={self.k}) fitted on {X.shape[1]} genes, "
                        f"{X.shape[0]} samples")

        elif self.method == "iterative":
            from sklearn.experimental import enable_iterative_imputer  # noqa
            from sklearn.impute import IterativeImputer
            self._imputer = IterativeImputer(max_iter=self.max_iter, random_state=42)
            self._imputer.fit(X)
            logger.info(f"Iterative imputer fitted ({self.max_iter} max iters)")

        elif self.method == "zero":
            logger.info("Zero imputer: no fitting required")

        else:
            raise ValueError(f"Unknown imputation method: {self.method}")

        self._fitted = True
        self._fitted_genes = df_clean.index.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using fitted statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Shape (n_genes, n_samples).

        Returns
        -------
        pd.DataFrame
            Imputed matrix, same shape (minus dropped genes).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        # Drop the same genes that were dropped during fit
        df_clean = df.drop(index=[g for g in self._drop_mask if g in df.index])

        # Align to fitted gene set (handle new test genes not seen in train)
        common_genes = [g for g in self._fitted_genes if g in df_clean.index]
        df_aligned = df_clean.loc[common_genes]

        X = df_aligned.T.values  # (n_samples, n_genes)
        missing_frac = np.isnan(X).mean()
        if missing_frac > 0:
            logger.debug(f"Imputing {missing_frac*100:.1f}% missing values")

        if self.method == "median":
            # Use training medians; clip to aligned gene set
            medians = self._medians[:X.shape[1]] if len(self._medians) > X.shape[1] \
                else self._medians
            nan_mask = np.isnan(X)
            result = X.copy()
            for j in range(X.shape[1]):
                result[nan_mask[:, j], j] = medians[j]

        elif self.method in ("knn", "iterative"):
            result = self._imputer.transform(X)

        elif self.method == "zero":
            result = np.nan_to_num(X, nan=0.0)

        else:
            result = X

        out_df = pd.DataFrame(
            result.T,  # back to (n_genes, n_samples)
            index=df_aligned.index,
            columns=df_aligned.columns,
        )
        return out_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Imputer saved: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GeneExpressionImputer":
        with open(path, "rb") as f:
            return pickle.load(f)


def report_missing_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a per-gene missing value report.

    Parameters
    ----------
    df : pd.DataFrame
        Shape (n_genes, n_samples).

    Returns
    -------
    pd.DataFrame with columns: gene, missing_count, missing_rate
    """
    missing = df.isna().sum(axis=1)
    rate = missing / df.shape[1]
    report = pd.DataFrame({
        "gene": df.index,
        "missing_count": missing.values,
        "missing_rate": rate.values,
    })
    report = report.sort_values("missing_rate", ascending=False).reset_index(drop=True)
    logger.info(f"Missing value summary: "
                f"total genes={len(df)}, "
                f"genes with any missing={( rate > 0).sum()}, "
                f"max missing rate={rate.max():.2%}")
    return report
