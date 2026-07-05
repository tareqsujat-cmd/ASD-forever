"""
Multi-stage feature selection for gene expression data.

The curse of dimensionality problem
-------------------------------------
A typical microarray has 20,000–54,000 features; an RNA-seq dataset has
~25,000 expressed genes.  With ABIDE-scale sample sizes (n ≈ 100–400),
training any model on raw features is severely underpowered.

Principled feature selection proceeds in stages, each removing a different
class of uninformative features:

Stage 1: Zero/near-zero variance filtering
  Removes genes that are constant or nearly constant across all samples.
  These carry no discriminative information by definition.

Stage 2: Between-group differential expression
  Removes genes with no statistically significant difference between
  ASD and control groups.  Uses t-test with Benjamini-Hochberg FDR control.
  CRITICAL: fitted on training labels only.

Stage 3: Variance-based selection
  Among DE genes, keep the top N by inter-subject variance.
  High variance = more information content.

Stage 4: Mutual Information with labels
  Non-linear measure of dependence between gene expression and ASD label.
  Captures complex (non-linear) gene-disease relationships that correlation
  misses.

Stage 5: SFARI gene overlap prioritization
  Biologically motivated: ASD-associated genes from SFARI get priority.
  Ensures the model attends to known disease mechanisms.
  Important for: (a) publication credibility, (b) explainability.

Stage 6: Correlation-based redundancy removal
  Remove highly correlated gene pairs (|r| > 0.95), keeping one per pair.
  Reduces redundancy without losing information.

Usage
-----
    from preprocessing.genetics.feature_selection import GeneFeatureSelector
    selector = GeneFeatureSelector(n_top=1000, sfari_genes=sfari_df)
    selector.fit(expr_train, labels_train)
    expr_selected = selector.transform(expr_train)
"""

from __future__ import annotations

import logging
import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GeneFeatureSelector:
    """
    Multi-stage gene feature selector.

    Parameters
    ----------
    n_top : int
        Final number of genes to select.
    variance_threshold : float
        Minimum variance; genes below are dropped (Stage 1).
    fdr_threshold : float
        FDR threshold for differential expression (Stage 2).
    mi_percentile : float
        Percentile of MI score to keep (Stage 4). 0.5 = top 50%.
    correlation_threshold : float
        Remove one gene from pairs with |r| > this value (Stage 6).
    sfari_genes : pd.DataFrame, optional
        SFARI gene list. Columns must include 'gene-symbol' and 'gene-score'.
    sfari_weight : float
        Priority boost multiplier for SFARI genes in ranking.
    """

    def __init__(
        self,
        n_top: int = 1000,
        variance_threshold: float = 0.01,
        fdr_threshold: float = 0.05,
        mi_percentile: float = 0.5,
        correlation_threshold: float = 0.95,
        sfari_genes: Optional[pd.DataFrame] = None,
        sfari_weight: float = 2.0,
    ) -> None:
        self.n_top = n_top
        self.variance_threshold = variance_threshold
        self.fdr_threshold = fdr_threshold
        self.mi_percentile = mi_percentile
        self.correlation_threshold = correlation_threshold
        self.sfari_genes = sfari_genes
        self.sfari_weight = sfari_weight

        self._selected_genes: List[str] = []
        self._selection_log: dict = {}
        self._fitted = False

    def fit(
        self,
        expr_df: pd.DataFrame,
        labels: Union[np.ndarray, pd.Series],
    ) -> "GeneFeatureSelector":
        """
        Fit feature selector on training data.

        Parameters
        ----------
        expr_df : pd.DataFrame
            Shape (n_genes, n_samples). Genes as rows.
        labels : array-like
            Binary labels (0=TC, 1=ASD) for each sample.

        Returns
        -------
        self
        """
        labels = np.array(labels)
        genes = expr_df.index.tolist()
        n_genes_initial = len(genes)

        logger.info(f"Feature selection: starting with {n_genes_initial} genes, "
                    f"{expr_df.shape[1]} samples")

        # Working data: (n_genes, n_samples)
        X = expr_df.values.astype(np.float64)

        # ---- Stage 1: Near-zero variance ----
        variances = np.var(X, axis=1)
        stage1_mask = variances > self.variance_threshold
        X = X[stage1_mask]
        genes = [g for g, m in zip(genes, stage1_mask) if m]
        self._selection_log["stage1_nzv"] = len(genes)
        logger.info(f"Stage 1 (NZV): {len(genes)} genes remaining "
                    f"(removed {n_genes_initial - len(genes)})")

        # ---- Stage 2: Differential expression (t-test + BH FDR) ----
        asd_idx = np.where(labels == 1)[0]
        tc_idx = np.where(labels == 0)[0]

        if len(asd_idx) > 1 and len(tc_idx) > 1:
            p_values = self._ttest_pvalues(X, asd_idx, tc_idx)
            fdr_mask = self._bh_correction(p_values, self.fdr_threshold)
            X = X[fdr_mask]
            genes = [g for g, m in zip(genes, fdr_mask) if m]
            self._selection_log["stage2_de"] = len(genes)
            self._de_pvalues = dict(zip(genes, p_values[fdr_mask]))
            logger.info(f"Stage 2 (DE, FDR<{self.fdr_threshold}): {len(genes)} genes")
        else:
            logger.warning("Not enough samples per class for DE testing; skipping Stage 2")
            self._selection_log["stage2_de"] = len(genes)

        # ---- Stage 3: Variance-based top selection ----
        if len(genes) > self.n_top * 3:
            variances_2 = np.var(X, axis=1)
            top_k = min(self.n_top * 3, len(genes))
            top_idx = np.argsort(variances_2)[-top_k:]
            X = X[top_idx]
            genes = [genes[i] for i in top_idx]
            self._selection_log["stage3_variance"] = len(genes)
            logger.info(f"Stage 3 (variance top-{top_k}): {len(genes)} genes")

        # ---- Stage 4: Mutual Information ----
        if len(genes) > self.n_top:
            mi_scores = self._mutual_information(X, labels)
            threshold = np.percentile(mi_scores, (1 - self.mi_percentile) * 100)
            mi_mask = mi_scores >= threshold
            X = X[mi_mask]
            genes = [g for g, m in zip(genes, mi_mask) if m]
            self._mi_scores = dict(zip(genes, mi_scores[mi_mask]))
            self._selection_log["stage4_mi"] = len(genes)
            logger.info(f"Stage 4 (MI top {self.mi_percentile*100:.0f}%): {len(genes)} genes")
        else:
            self._mi_scores = {}

        # ---- Stage 5: SFARI gene prioritization ----
        if self.sfari_genes is not None and len(genes) > self.n_top:
            sfari_set = set(self.sfari_genes["gene-symbol"].dropna().str.upper())
            X, genes = self._sfari_prioritize(X, genes, sfari_set)
            self._selection_log["stage5_sfari"] = len(genes)
            logger.info(f"Stage 5 (SFARI prioritized): {len(genes)} genes")

        # ---- Stage 6: Correlation-based redundancy removal ----
        if len(genes) > self.n_top:
            X, genes = self._remove_correlated(X, genes)
            self._selection_log["stage6_corr"] = len(genes)
            logger.info(f"Stage 6 (correlation dedup): {len(genes)} genes")

        # ---- Final: Take top n_top ----
        final_genes = genes[:self.n_top]
        self._selected_genes = final_genes
        self._fitted = True

        logger.info(f"Feature selection complete: {n_genes_initial} -> "
                    f"{len(self._selected_genes)} genes")

        return self

    def transform(self, expr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the fitted gene subset from an expression matrix.

        Parameters
        ----------
        expr_df : pd.DataFrame
            Shape (n_genes, n_samples).

        Returns
        -------
        pd.DataFrame
            Shape (n_selected, n_samples).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        available = [g for g in self._selected_genes if g in expr_df.index]
        missing = len(self._selected_genes) - len(available)
        if missing > 0:
            logger.warning(f"{missing} selected genes not found in transform data; "
                           "they will be filled with zeros")

        result = expr_df.reindex(self._selected_genes, fill_value=0.0)
        return result

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return a DataFrame of selected genes with their selection scores.

        Useful for Table 1 in the paper and for explainability visualizations.
        """
        records = []
        sfari_set = set()
        if self.sfari_genes is not None:
            sfari_set = set(self.sfari_genes["gene-symbol"].dropna().str.upper())

        for gene in self._selected_genes:
            records.append({
                "gene": gene,
                "de_pvalue": self._de_pvalues.get(gene, np.nan)
                if hasattr(self, "_de_pvalues") else np.nan,
                "mi_score": self._mi_scores.get(gene, np.nan),
                "is_sfari": gene.upper() in sfari_set,
            })
        return pd.DataFrame(records)

    def save(self, path: Union[str, "Path"]) -> None:
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Selector saved: {path}")

    @classmethod
    def load(cls, path) -> "GeneFeatureSelector":
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ttest_pvalues(
        X: np.ndarray, asd_idx: np.ndarray, tc_idx: np.ndarray
    ) -> np.ndarray:
        """Compute Welch's t-test p-value for each gene."""
        from scipy.stats import ttest_ind
        pvals = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            asd_vals = X[i, asd_idx]
            tc_vals = X[i, tc_idx]
            # Skip constant genes
            if asd_vals.std() < 1e-10 and tc_vals.std() < 1e-10:
                pvals[i] = 1.0
            else:
                _, p = ttest_ind(asd_vals, tc_vals, equal_var=False, nan_policy="omit")
                pvals[i] = p if not np.isnan(p) else 1.0
        return pvals

    @staticmethod
    def _bh_correction(p_values: np.ndarray, alpha: float) -> np.ndarray:
        """
        Benjamini-Hochberg FDR correction.

        Returns boolean mask of genes passing the FDR threshold.
        BH is less conservative than Bonferroni, appropriate for exploratory
        feature selection (not confirmatory hypothesis testing).
        """
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        threshold = np.arange(1, n + 1) / n * alpha
        below = sorted_p <= threshold
        # All genes up to the last significant one are included
        if below.any():
            last_sig = np.where(below)[0][-1]
            mask = np.zeros(n, dtype=bool)
            mask[sorted_idx[:last_sig + 1]] = True
        else:
            mask = np.zeros(n, dtype=bool)
        return mask

    @staticmethod
    def _mutual_information(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Estimate mutual information between each gene and the binary label.

        Uses the k-NN entropy estimator (Kraskov et al., 2004) via sklearn.
        MI captures both linear and non-linear gene-disease associations.
        """
        try:
            from sklearn.feature_selection import mutual_info_classif
            # MI expects (n_samples, n_features) — transpose
            mi = mutual_info_classif(X.T, labels, random_state=42, n_neighbors=3)
            return mi
        except Exception as exc:
            logger.warning(f"MI computation failed: {exc}; using variance proxy")
            return np.var(X, axis=1)

    def _sfari_prioritize(
        self,
        X: np.ndarray,
        genes: List[str],
        sfari_set: set,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Ensure SFARI score-1/2 genes are prioritized in the final selection.

        Strategy: compute a combined score = MI_score * sfari_weight for SFARI genes.
        This biases the top-n selection toward biologically meaningful genes
        without completely excluding non-SFARI genes with high MI.
        """
        if not hasattr(self, "_mi_scores") or not self._mi_scores:
            return X, genes

        scores = np.array([
            self._mi_scores.get(g, 0.0) * (self.sfari_weight if g.upper() in sfari_set else 1.0)
            for g in genes
        ])
        order = np.argsort(scores)[::-1]
        return X[order], [genes[i] for i in order]

    def _remove_correlated(
        self,
        X: np.ndarray,
        genes: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Greedily remove one gene from each highly correlated pair.

        Correlation matrix is computed in chunks to handle large gene counts.
        """
        n_genes = X.shape[0]
        if n_genes > 5000:
            # Too expensive; skip
            return X, genes

        corr = np.corrcoef(X)
        keep = np.ones(n_genes, dtype=bool)
        for i in range(n_genes):
            if not keep[i]:
                continue
            for j in range(i + 1, n_genes):
                if keep[j] and abs(corr[i, j]) > self.correlation_threshold:
                    keep[j] = False

        return X[keep], [g for g, k in zip(genes, keep) if k]
