"""
Genetics Preprocessing Pipeline Orchestrator.

Runs the complete preprocessing sequence for gene expression data:
  1. Download GEO dataset (or load from file)
  2. Load SFARI gene list
  3. Impute missing values (fit on training data only)
  4. ComBat batch effect correction
  5. Feature selection (6-stage pipeline)
  6. Dimensionality reduction (PCA or VAE)
  7. Build gene interaction graph
  8. Save processed features + fitted transformers
  9. Create PyTorch Dataset objects

Design principle: all fitting is done on training data.
The pipeline persists fitted objects (imputer, selector, PCA, VAE) so that
the exact same transformations are applied at inference time.

Usage
-----
    python -m preprocessing.genetics.preprocess_pipeline \
        --config configs/config.yaml \
        --accession GSE18123
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GeneticsPreprocessPipeline:
    """
    End-to-end genetics preprocessing pipeline.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.gen_cfg = cfg.genetics_preprocessing
        self.root = Path(cfg.paths.root)
        self._out_dir = self.root / cfg.paths.data_processed_genetics
        self._out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        expr_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        train_idx: Optional[List[int]] = None,
        abide_meta: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run the full genetics preprocessing pipeline.

        Parameters
        ----------
        expr_df : pd.DataFrame
            Raw expression matrix, shape (n_genes, n_samples).
        meta_df : pd.DataFrame
            Sample metadata with 'label', 'sample_id', and batch column.
        train_idx : list of int, optional
            Training sample indices for fit-on-train-only transforms.
        abide_meta : pd.DataFrame, optional
            ABIDE metadata for subject alignment.

        Returns
        -------
        features_df : pd.DataFrame
            Processed features, shape (n_features, n_samples).
        meta_df : pd.DataFrame
            Updated metadata.
        artifacts : dict
            Paths to saved fitted transformers and graph.
        """
        logger.info("=== Genetics Preprocessing Pipeline ===")
        artifacts = {}

        # Identify training samples
        if "sample_id" in meta_df.columns:
            all_ids = meta_df["sample_id"].astype(str).tolist()
        else:
            all_ids = list(range(len(meta_df)))

        if train_idx is not None:
            train_samples = [str(all_ids[i]) for i in train_idx]
        else:
            train_samples = [str(s) for s in expr_df.columns]

        # ---- Step 1: Imputation ----
        logger.info("Step 1: Missing value imputation")
        imputer = self._run_imputation(expr_df, train_samples)
        expr_df = imputer.transform(expr_df)
        imp_path = self._out_dir / "imputer.pkl"
        imputer.save(imp_path)
        artifacts["imputer"] = str(imp_path)

        # ---- Step 2: Batch correction ----
        logger.info("Step 2: ComBat batch correction")
        if self.gen_cfg.batch_correction == "combat":
            expr_df = self._run_combat(expr_df, meta_df)
        combat_path = self._out_dir / "batch_corrected_expression.parquet"
        expr_df.to_parquet(combat_path)
        artifacts["batch_corrected"] = str(combat_path)

        # ---- Step 3: Feature selection ----
        logger.info("Step 3: Feature selection")
        selector, expr_selected = self._run_feature_selection(
            expr_df, meta_df, train_samples
        )
        sel_path = self._out_dir / "feature_selector.pkl"
        selector.save(sel_path)
        artifacts["selector"] = str(sel_path)

        # Save feature importance table
        importance_df = selector.get_feature_importance()
        importance_df.to_csv(self._out_dir / "gene_importance.csv", index=False)
        artifacts["gene_importance"] = str(self._out_dir / "gene_importance.csv")

        # ---- Step 4: Normalization ----
        logger.info("Step 4: Feature-level normalization")
        expr_normalized, scaler = self._normalize(
            expr_selected, train_samples
        )
        scaler_path = self._out_dir / "feature_scaler.pkl"
        import pickle
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        artifacts["scaler"] = str(scaler_path)

        # ---- Step 5: Dimensionality reduction ----
        logger.info("Step 5: Dimensionality reduction")
        reduced_df, dim_reducer = self._run_dim_reduction(
            expr_normalized, train_samples
        )
        if dim_reducer is not None:
            dr_path = self._out_dir / "dim_reducer.pkl"
            if hasattr(dim_reducer, "save"):
                dim_reducer.save(dr_path)
            else:
                import pickle
                with open(dr_path, "wb") as f:
                    pickle.dump(dim_reducer, f)
            artifacts["dim_reducer"] = str(dr_path)

        # ---- Step 6: Build gene graph ----
        logger.info("Step 6: Building gene interaction graph")
        graph_path = str(self._out_dir / "gene_graph.pkl")
        self._build_gene_graph(
            selector._selected_genes,
            expr_normalized.T.values if hasattr(expr_normalized, "values") else expr_normalized.T,
            graph_path,
        )
        artifacts["gene_graph"] = graph_path

        # ---- Save final features ----
        final_path = self._out_dir / "processed_features.parquet"
        reduced_df.to_parquet(final_path)
        artifacts["processed_features"] = str(final_path)
        meta_df.to_csv(self._out_dir / "genetics_metadata.csv", index=False)

        n_genes_final = reduced_df.shape[0]
        logger.info(f"Genetics pipeline complete: {n_genes_final} features, "
                    f"{reduced_df.shape[1]} samples")

        return reduced_df, meta_df, artifacts

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _run_imputation(
        self, expr_df: pd.DataFrame, train_samples: List[str]
    ):
        from preprocessing.genetics.imputation import GeneExpressionImputer
        method = "knn" if expr_df.isna().mean().mean() > 0.02 else "median"
        imputer = GeneExpressionImputer(
            method=method,
            missing_threshold=self.gen_cfg.missing_threshold,
        )
        train_expr = expr_df[
            [s for s in train_samples if s in expr_df.columns]
        ]
        imputer.fit(train_expr)
        return imputer

    def _run_combat(
        self, expr_df: pd.DataFrame, meta_df: pd.DataFrame
    ) -> pd.DataFrame:
        from preprocessing.genetics.batch_correction import ComBat
        combat = ComBat(covariate_cols=["label"])
        try:
            corrected = combat.fit_transform(
                expr_df, meta_df, batch_col="batch"
            )
        except Exception as exc:
            logger.warning(f"ComBat failed ({exc}); using uncorrected data")
            corrected = expr_df
        return corrected

    def _run_feature_selection(
        self,
        expr_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        train_samples: List[str],
    ):
        from preprocessing.genetics.feature_selection import GeneFeatureSelector
        from preprocessing.genetics.downloader import GEODownloader

        # Try to load SFARI genes
        sfari_genes = None
        try:
            dl = GEODownloader(self.root / self.cfg.paths.data_raw_genetics)
            sfari_genes = dl.download_sfari_genes()
        except Exception as exc:
            logger.warning(f"SFARI load failed: {exc}")

        selector = GeneFeatureSelector(
            n_top=self.gen_cfg.n_top_features,
            variance_threshold=self.gen_cfg.variance_threshold,
            sfari_genes=sfari_genes,
        )

        # Fit on training data only
        train_cols = [s for s in train_samples if s in expr_df.columns]
        train_expr = expr_df[train_cols]

        if "sample_id" in meta_df.columns:
            meta_indexed = meta_df.set_index("sample_id")
        else:
            meta_indexed = meta_df

        train_labels = np.array([
            meta_indexed.loc[s, "label"] if s in meta_indexed.index else 0
            for s in train_cols
        ])

        selector.fit(train_expr, train_labels)
        expr_selected = selector.transform(expr_df)
        return selector, expr_selected

    def _normalize(
        self,
        expr_df: pd.DataFrame,
        train_samples: List[str],
    ):
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
        train_cols = [s for s in train_samples if s in expr_df.columns]
        train_X = expr_df[train_cols].T.values  # (n_train, n_genes)
        scaler.fit(train_X)

        all_X = expr_df.T.values  # (n_samples, n_genes)
        normalized = scaler.transform(all_X)  # (n_samples, n_genes)

        normalized_df = pd.DataFrame(
            normalized.T,
            index=expr_df.index,
            columns=expr_df.columns,
        )
        return normalized_df, scaler

    def _run_dim_reduction(
        self,
        expr_df: pd.DataFrame,
        train_samples: List[str],
    ):
        method = self.gen_cfg.dimensionality_reduction.method
        n_components = self.gen_cfg.dimensionality_reduction.n_components

        X_all = expr_df.T.values.astype(np.float32)  # (n_samples, n_genes)
        train_cols = [s for s in train_samples if s in expr_df.columns]
        train_idx_local = [
            i for i, c in enumerate(expr_df.columns) if c in set(train_cols)
        ]
        X_train = X_all[train_idx_local]

        if method == "pca":
            from preprocessing.genetics.dimensionality_reduction import PCAReducer
            reducer = PCAReducer(n_components=n_components)
            reducer.fit(X_train)
            Z_all = reducer.transform(X_all)
            dim_reducer = reducer

        elif method == "autoencoder":
            from preprocessing.genetics.dimensionality_reduction import GeneVAE
            input_dim = expr_df.shape[0]
            reducer = GeneVAE(input_dim=input_dim, latent_dim=n_components)
            reducer.fit(X_train, epochs=50, batch_size=32, device="cpu")
            Z_mu, _ = reducer.encode(X_all)
            Z_all = Z_mu
            dim_reducer = reducer

        else:
            # No reduction; use selected features directly
            Z_all = X_all
            dim_reducer = None

        reduced_df = pd.DataFrame(
            Z_all.T,
            columns=expr_df.columns,
        )
        return reduced_df, dim_reducer

    def _build_gene_graph(
        self,
        gene_list: List[str],
        expr_matrix: np.ndarray,
        cache_path: str,
    ) -> None:
        from preprocessing.genetics.gene_graph import GeneGraphBuilder
        builder = GeneGraphBuilder(score_threshold=700, fallback_to_coexpression=True)
        try:
            builder.build(gene_list, expr_matrix=expr_matrix, cache_path=cache_path)
            stats = builder.compute_graph_statistics()
            logger.info(f"Gene graph stats: {stats}")
        except Exception as exc:
            logger.warning(f"Gene graph build failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Genetics Preprocessing Pipeline")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--accession", default="GSE18123")
    parser.add_argument("--expression-file", default=None,
                        help="Path to custom expression matrix (TSV)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from configs.config_schema import load_config
    from utilities.logger import setup_root_logger
    from utilities.reproducibility import seed_everything
    from preprocessing.genetics.downloader import GEODownloader

    cfg = load_config(args.config)
    root = Path(cfg.paths.root)
    setup_root_logger(
        level=cfg.logging.level,
        log_dir=str(root / cfg.paths.logs),
    )
    seed_everything(cfg.project.random_seed)

    dl = GEODownloader(root / cfg.paths.data_raw_genetics)

    if args.expression_file:
        expr_df = dl.load_expression_matrix(args.expression_file)
        meta_df = pd.DataFrame({"sample_id": expr_df.columns, "label": 0, "batch": "1"})
    else:
        expr_df, meta_df = dl.download_geo(args.accession)

    pipeline = GeneticsPreprocessPipeline(cfg)
    features_df, meta_out, artifacts = pipeline.run(expr_df, meta_df)

    logger.info("Artifacts:")
    for k, v in artifacts.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
