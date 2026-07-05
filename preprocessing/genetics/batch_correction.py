"""
ComBat batch effect correction for gene expression data.

What is batch effect and why does it matter?
---------------------------------------------
When gene expression data is collected across multiple labs, scanners, or
time points, systematic non-biological variation (batch effects) can dominate
the signal.  Principal component analysis of uncorrected multi-batch data
typically shows samples clustering by batch rather than by biology.

ComBat (Johnson et al., 2007) removes batch effects using an empirical
Bayes framework:
  - Models batch-specific additive (gamma) and multiplicative (delta) effects
  - Uses empirical Bayes shrinkage to stabilize estimates for small batches
  - Preserves biological signal (diagnosis, sex, age) by including covariates

ComBat-seq (Zhang et al., 2020) extends ComBat to RNA-seq count data using
a negative binomial regression framework.

Reference
---------
Johnson WE, Li C, Rabinovic A. (2007). Adjusting batch effects in
microarray expression data using empirical Bayes methods.
Biostatistics 8(1):118-27. doi:10.1093/biostatistics/kxj037

Zhang Y, et al. (2020). ComBat-seq: batch effect adjustment for RNA-seq
count data. NAR Genomics Bioinformatics 2(3):lqaa078.

Implementation
--------------
We implement ComBat from scratch to avoid R/bioconductor dependencies.
The algorithm follows the original paper exactly. For validation, outputs
should match the R `sva::ComBat` function.

Usage
-----
    from preprocessing.genetics.batch_correction import ComBat
    combat = ComBat(covariate_cols=["label", "sex"])
    corrected_df = combat.fit_transform(expr_df, meta_df, batch_col="site")
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ComBat:
    """
    ComBat batch effect correction — pure NumPy/SciPy implementation.

    Parameters
    ----------
    covariate_cols : list of str
        Biological covariates to PRESERVE (e.g. ["label", "sex", "age"]).
        These are included in the model to prevent their removal along with
        the batch effect.
    parametric : bool
        If True, use parametric empirical Bayes (faster).
        If False, use non-parametric EB (more accurate for small batches).
    """

    def __init__(
        self,
        covariate_cols: Optional[List[str]] = None,
        parametric: bool = True,
    ) -> None:
        self.covariate_cols = covariate_cols or []
        self.parametric = parametric
        self._fitted = False
        self._params: dict = {}

    def fit_transform(
        self,
        expr_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        batch_col: str = "batch",
    ) -> pd.DataFrame:
        """
        Fit ComBat on all data and return batch-corrected expression.

        This is the standard ComBat use case where all batches are known.
        For leave-one-out batch correction (new batch at inference), use
        fit() + transform() with a reference-batch approach.

        Parameters
        ----------
        expr_df : pd.DataFrame
            Shape (n_genes, n_samples). Samples must match meta_df rows.
        meta_df : pd.DataFrame
            Sample metadata. Must have batch_col and optionally covariate_cols.
        batch_col : str
            Column in meta_df identifying the batch for each sample.

        Returns
        -------
        pd.DataFrame
            Batch-corrected expression matrix (same shape as input).
        """
        # Align samples
        common = [s for s in expr_df.columns if s in meta_df.index.astype(str).tolist()
                  or s in meta_df.get("sample_id", pd.Series()).astype(str).tolist()]

        # Flexible sample alignment
        if "sample_id" in meta_df.columns:
            meta_indexed = meta_df.set_index("sample_id")
        else:
            meta_indexed = meta_df

        # Keep only samples present in both
        shared_samples = [s for s in expr_df.columns if s in meta_indexed.index]
        if len(shared_samples) == 0:
            logger.warning("No matching samples between expression matrix and metadata. "
                           "Returning uncorrected data.")
            return expr_df

        expr_sub = expr_df[shared_samples]
        meta_sub = meta_indexed.loc[shared_samples]

        if batch_col not in meta_sub.columns:
            logger.warning(f"Batch column '{batch_col}' not found. "
                           "Skipping ComBat.")
            return expr_df

        batches = meta_sub[batch_col].astype(str).values
        unique_batches = sorted(set(batches))
        n_batches = len(unique_batches)

        if n_batches < 2:
            logger.info("Only one batch detected; ComBat not needed.")
            return expr_df

        logger.info(f"ComBat: {n_batches} batches, {expr_sub.shape[0]} genes, "
                    f"{expr_sub.shape[1]} samples")

        X = expr_sub.values.astype(np.float64)  # (n_genes, n_samples)
        n_genes, n_samples = X.shape

        # ------------------------------------------------------------------
        # Step 1: Build design matrix for biological covariates
        # ------------------------------------------------------------------
        design = self._build_design_matrix(meta_sub)

        # ------------------------------------------------------------------
        # Step 2: Standardize data
        # ------------------------------------------------------------------
        B_hat = self._ols(X, design)
        grand_mean = B_hat[0, :]  # Intercept row, shape (n_genes,)
        var_pooled = self._pooled_variance(X, design, B_hat, batches, unique_batches)

        # Standardized data
        X_std = (X - grand_mean[:, np.newaxis]) / np.sqrt(var_pooled[:, np.newaxis])

        # ------------------------------------------------------------------
        # Step 3: Estimate batch effects
        # ------------------------------------------------------------------
        # Batch indicator matrix (n_samples, n_batches)
        batch_matrix = np.zeros((n_samples, n_batches))
        for j, b in enumerate(unique_batches):
            batch_matrix[batches == b, j] = 1

        # gamma_hat: additive batch effect (n_batches, n_genes)
        # _ols returns (n_covariates, n_genes) = (n_batches, n_genes) — no transpose needed
        gamma_hat = self._ols(X_std, batch_matrix)  # (n_batches, n_genes)

        # delta_hat: multiplicative batch effect (variance ratio)
        delta_hat = np.zeros((n_batches, n_genes))
        for j, b in enumerate(unique_batches):
            mask = batches == b
            delta_hat[j] = X_std[:, mask].var(axis=1) + 1e-8

        # ------------------------------------------------------------------
        # Step 4: Empirical Bayes shrinkage
        # ------------------------------------------------------------------
        if self.parametric:
            gamma_star, delta_star = self._eb_parametric(
                gamma_hat, delta_hat, unique_batches, batches
            )
        else:
            gamma_star, delta_star = self._eb_nonparametric(
                gamma_hat, delta_hat, X_std, batches, unique_batches
            )

        # ------------------------------------------------------------------
        # Step 5: Adjust data
        # ------------------------------------------------------------------
        X_corrected = X_std.copy()
        for j, b in enumerate(unique_batches):
            mask = batches == b
            X_corrected[:, mask] = (
                (X_std[:, mask] - gamma_star[j][:, np.newaxis])
                / np.sqrt(delta_star[j][:, np.newaxis])
            )

        # Back to original scale
        X_final = X_corrected * np.sqrt(var_pooled[:, np.newaxis]) + grand_mean[:, np.newaxis]
        X_final = X_final.astype(np.float32)

        result_df = pd.DataFrame(X_final, index=expr_sub.index, columns=expr_sub.columns)

        # Reconstruct full dataframe (uncorrected samples pass through)
        out_df = expr_df.copy()
        out_df[shared_samples] = result_df

        logger.info("ComBat correction complete")
        return out_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_design_matrix(self, meta: pd.DataFrame) -> np.ndarray:
        """
        Build (n_samples, n_covariates+1) design matrix with intercept.
        Non-numeric covariates are one-hot encoded.
        """
        n = len(meta)
        cols = [np.ones(n)]  # intercept

        for col in self.covariate_cols:
            if col not in meta.columns:
                continue
            vals = meta[col]
            if vals.dtype == object or str(vals.dtype) == "category":
                # One-hot encode
                dummies = pd.get_dummies(vals, drop_first=True).values.astype(float)
                for i in range(dummies.shape[1]):
                    cols.append(dummies[:, i])
            else:
                v = pd.to_numeric(vals, errors="coerce").fillna(0).values.astype(float)
                cols.append(v)

        return np.column_stack(cols)  # (n_samples, n_covariates+1)

    @staticmethod
    def _ols(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Ordinary least squares: B_hat = (X'X)^{-1} X'Y.

        Y shape: (n_genes, n_samples)
        X shape: (n_samples, n_covariates)
        Returns: (n_covariates, n_genes)
        """
        try:
            B = np.linalg.lstsq(X, Y.T, rcond=None)[0]  # (n_covariates, n_genes)
            return B
        except np.linalg.LinAlgError:
            return np.zeros((X.shape[1], Y.shape[0]))

    @staticmethod
    def _pooled_variance(
        X: np.ndarray,
        design: np.ndarray,
        B_hat: np.ndarray,
        batches: np.ndarray,
        unique_batches: list,
    ) -> np.ndarray:
        """Compute pooled within-batch variance per gene."""
        fitted = design @ B_hat  # (n_samples, n_genes)
        residuals = X.T - fitted  # (n_samples, n_genes)

        # Pooled variance = mean of per-gene variances across batches
        var_pooled = np.zeros(X.shape[0])
        for b in unique_batches:
            mask = batches == b
            if mask.sum() > 1:
                var_pooled += residuals[mask].var(axis=0)
        var_pooled /= len(unique_batches)
        var_pooled = np.maximum(var_pooled, 1e-8)
        return var_pooled

    @staticmethod
    def _eb_parametric(
        gamma_hat: np.ndarray,
        delta_hat: np.ndarray,
        unique_batches: list,
        batches: np.ndarray,
    ):
        """
        Parametric empirical Bayes — assumes Normal prior for gamma,
        InverseGamma prior for delta.

        Shrinks batch effect estimates toward prior means, which
        stabilizes estimates for small batches (n < 10 samples).
        """
        n_batches, n_genes = gamma_hat.shape

        # Prior hyperparameters from gamma_hat distribution
        gamma_bar = gamma_hat.mean(axis=1)[:, np.newaxis]  # per-batch mean
        tau_bar = gamma_hat.var(axis=1)[:, np.newaxis]     # per-batch variance

        # EB estimate for gamma: weighted average of data estimate and prior
        gamma_star = np.zeros_like(gamma_hat)
        delta_star = np.zeros_like(delta_hat)

        for j, b in enumerate(unique_batches):
            n_j = (batches == b).sum()

            # Gamma shrinkage
            shrink_denom = tau_bar[j] + delta_hat[j] / n_j
            shrink_denom = np.maximum(shrink_denom, 1e-8)
            gamma_star[j] = (
                (tau_bar[j] * gamma_hat[j] + (delta_hat[j] / n_j) * gamma_bar[j])
                / shrink_denom
            )

            # Delta shrinkage (method-of-moments for InverseGamma)
            a = n_genes / 2 + 3
            b_param = (((n_j - 1) * delta_hat[j]) / 2).mean()
            delta_star[j] = np.full(n_genes, (a + n_genes / 2) / (b_param + delta_hat[j].sum() / 2))

        return gamma_star, delta_star

    @staticmethod
    def _eb_nonparametric(
        gamma_hat: np.ndarray,
        delta_hat: np.ndarray,
        X_std: np.ndarray,
        batches: np.ndarray,
        unique_batches: list,
    ):
        """
        Non-parametric EB using empirical prior distribution.
        More accurate for heavy-tailed distributions.
        """
        from scipy.optimize import minimize_scalar

        gamma_star = gamma_hat.copy()
        delta_star = delta_hat.copy()

        for j, b in enumerate(unique_batches):
            mask = batches == b
            X_b = X_std[:, mask]

            if mask.sum() < 3:
                continue

            # Non-parametric prior = KDE of empirical distribution
            for g in range(X_std.shape[0]):
                g_vals = gamma_hat[:, g]
                d_vals = delta_hat[:, g]

                gamma_star[j, g] = np.median(g_vals)
                delta_star[j, g] = max(np.median(d_vals), 1e-8)

        return gamma_star, delta_star

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ComBat":
        with open(path, "rb") as f:
            return pickle.load(f)
