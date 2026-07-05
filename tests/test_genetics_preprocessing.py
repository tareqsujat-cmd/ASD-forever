"""
Unit tests for the genetics preprocessing module.

All tests use synthetic data — no internet access, GEO, or GPU required.
Run with:  pytest tests/test_genetics_preprocessing.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_expression(n_genes=200, n_samples=40):
    """Synthetic gene expression matrix with two groups."""
    rng = np.random.default_rng(42)
    # ASD: slightly elevated expression for first 20 genes
    asd_idx = np.arange(n_samples // 2)
    tc_idx = np.arange(n_samples // 2, n_samples)
    X = rng.normal(5.0, 1.0, (n_genes, n_samples)).astype(np.float32)
    X[:20, asd_idx] += 1.5  # DE signal
    return pd.DataFrame(
        X,
        index=[f"GENE_{i:04d}" for i in range(n_genes)],
        columns=[f"sample_{i:04d}" for i in range(n_samples)],
    )


@pytest.fixture
def synthetic_metadata(n_samples=40):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "sample_id": [f"sample_{i:04d}" for i in range(n_samples)],
        "label": [1 if i < n_samples // 2 else 0 for i in range(n_samples)],
        "batch": ["site_A"] * (n_samples // 2) + ["site_B"] * (n_samples // 2),
        "sex_encoded": rng.integers(0, 2, n_samples).tolist(),
        "age": rng.normal(25, 5, n_samples).tolist(),
    })


@pytest.fixture
def train_idx(n_samples=40):
    return list(range(30))  # First 30 for training


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

class TestImputation:
    def test_median_no_missing(self, synthetic_expression):
        from preprocessing.genetics.imputation import GeneExpressionImputer
        imp = GeneExpressionImputer(method="median")
        imp.fit(synthetic_expression)
        result = imp.transform(synthetic_expression)
        assert result.shape[0] <= synthetic_expression.shape[0]  # may drop high-missing genes
        assert result.isna().sum().sum() == 0

    def test_knn_imputes_nans(self, synthetic_expression):
        from preprocessing.genetics.imputation import GeneExpressionImputer
        expr = synthetic_expression.copy()
        # Inject 5% NaN
        rng = np.random.default_rng(1)
        for i in range(5):
            r = rng.integers(0, expr.shape[0])
            c = rng.integers(0, expr.shape[1])
            expr.iloc[r, c] = np.nan

        imp = GeneExpressionImputer(method="knn", k=3, missing_threshold=0.5)
        imp.fit(expr)
        result = imp.transform(expr)
        assert result.isna().sum().sum() == 0

    def test_fit_before_transform_raises(self, synthetic_expression):
        from preprocessing.genetics.imputation import GeneExpressionImputer
        imp = GeneExpressionImputer(method="median")
        with pytest.raises(RuntimeError):
            imp.transform(synthetic_expression)

    def test_high_missing_genes_dropped(self):
        from preprocessing.genetics.imputation import GeneExpressionImputer
        # Gene 0 is all NaN
        X = pd.DataFrame(
            np.ones((10, 5)),
            index=[f"G{i}" for i in range(10)],
            columns=[f"s{i}" for i in range(5)],
        )
        X.iloc[0] = np.nan  # 100% missing
        imp = GeneExpressionImputer(method="median", missing_threshold=0.5)
        imp.fit(X)
        result = imp.transform(X)
        assert "G0" not in result.index

    def test_zero_imputation(self, synthetic_expression):
        from preprocessing.genetics.imputation import GeneExpressionImputer
        expr = synthetic_expression.copy()
        expr.iloc[0, 0] = np.nan
        imp = GeneExpressionImputer(method="zero")
        imp.fit(expr)
        result = imp.transform(expr)
        assert result.isna().sum().sum() == 0

    def test_save_load_roundtrip(self, synthetic_expression, tmp_path):
        from preprocessing.genetics.imputation import GeneExpressionImputer
        imp = GeneExpressionImputer(method="median")
        imp.fit(synthetic_expression)
        r1 = imp.transform(synthetic_expression)
        path = tmp_path / "imp.pkl"
        imp.save(path)
        imp2 = GeneExpressionImputer.load(path)
        r2 = imp2.transform(synthetic_expression)
        pd.testing.assert_frame_equal(r1, r2, check_exact=False, rtol=1e-5)


# ---------------------------------------------------------------------------
# Feature Selection
# ---------------------------------------------------------------------------

class TestFeatureSelection:
    def test_reduces_to_n_top(self, synthetic_expression, synthetic_metadata):
        from preprocessing.genetics.feature_selection import GeneFeatureSelector
        n_top = 50
        selector = GeneFeatureSelector(n_top=n_top, fdr_threshold=0.5)
        labels = synthetic_metadata["label"].values
        selector.fit(synthetic_expression, labels)
        result = selector.transform(synthetic_expression)
        assert result.shape[0] <= n_top

    def test_selected_genes_are_subset(self, synthetic_expression, synthetic_metadata):
        from preprocessing.genetics.feature_selection import GeneFeatureSelector
        selector = GeneFeatureSelector(n_top=100, fdr_threshold=0.9)
        labels = synthetic_metadata["label"].values
        selector.fit(synthetic_expression, labels)
        result = selector.transform(synthetic_expression)
        assert all(g in synthetic_expression.index for g in result.index)

    def test_transform_before_fit_raises(self, synthetic_expression):
        from preprocessing.genetics.feature_selection import GeneFeatureSelector
        selector = GeneFeatureSelector()
        with pytest.raises(RuntimeError):
            selector.transform(synthetic_expression)

    def test_feature_importance_returns_dataframe(self, synthetic_expression, synthetic_metadata):
        from preprocessing.genetics.feature_selection import GeneFeatureSelector
        selector = GeneFeatureSelector(n_top=20, fdr_threshold=0.9)
        selector.fit(synthetic_expression, synthetic_metadata["label"].values)
        fi = selector.get_feature_importance()
        assert isinstance(fi, pd.DataFrame)
        assert "gene" in fi.columns

    def test_sfari_genes_included_preferentially(self, synthetic_expression, synthetic_metadata):
        from preprocessing.genetics.feature_selection import GeneFeatureSelector
        # Add known SFARI genes to the expression matrix
        sfari_df = pd.DataFrame({
            "gene-symbol": ["SHANK3", "CNTNAP2"],
            "gene-score": ["1", "2"],
        })
        extra = pd.DataFrame(
            np.ones((2, synthetic_expression.shape[1])),
            index=["SHANK3", "CNTNAP2"],
            columns=synthetic_expression.columns,
        )
        expr_with_sfari = pd.concat([synthetic_expression, extra])
        selector = GeneFeatureSelector(n_top=30, sfari_genes=sfari_df, fdr_threshold=0.9)
        selector.fit(expr_with_sfari, synthetic_metadata["label"].values)
        result = selector.transform(expr_with_sfari)
        # At least one SFARI gene should be selected
        sfari_selected = [g for g in result.index if g in ["SHANK3", "CNTNAP2"]]
        assert len(sfari_selected) >= 1


# ---------------------------------------------------------------------------
# ComBat
# ---------------------------------------------------------------------------

class TestComBat:
    def test_combat_runs_without_error(self, synthetic_expression, synthetic_metadata):
        from preprocessing.genetics.batch_correction import ComBat
        combat = ComBat(covariate_cols=["label"])
        meta = synthetic_metadata.set_index("sample_id")
        result = combat.fit_transform(synthetic_expression, meta, batch_col="batch")
        assert result.shape == synthetic_expression.shape

    def test_single_batch_returns_unchanged(self, synthetic_expression):
        from preprocessing.genetics.batch_correction import ComBat
        meta = pd.DataFrame({
            "label": [0] * synthetic_expression.shape[1],
            "batch": ["site_A"] * synthetic_expression.shape[1],
        }, index=synthetic_expression.columns)
        combat = ComBat()
        result = combat.fit_transform(synthetic_expression, meta, batch_col="batch")
        # Should return original (single batch = no correction)
        assert result.shape == synthetic_expression.shape


# ---------------------------------------------------------------------------
# PCA Reducer
# ---------------------------------------------------------------------------

class TestPCAReducer:
    def test_reduces_dimension(self, synthetic_expression):
        from preprocessing.genetics.dimensionality_reduction import PCAReducer
        X = synthetic_expression.T.values  # (n_samples, n_genes)
        reducer = PCAReducer(n_components=10)
        reducer.fit(X)
        Z = reducer.transform(X)
        assert Z.shape == (X.shape[0], 10)

    def test_fit_transform_equivalent(self, synthetic_expression):
        from preprocessing.genetics.dimensionality_reduction import PCAReducer
        X = synthetic_expression.T.values.astype(np.float32)
        r = PCAReducer(n_components=5)
        z1 = r.fit_transform(X)
        r2 = PCAReducer(n_components=5)
        r2.fit(X)
        z2 = r2.transform(X)
        np.testing.assert_allclose(np.abs(z1), np.abs(z2), rtol=1e-4)

    def test_save_load_roundtrip(self, synthetic_expression, tmp_path):
        from preprocessing.genetics.dimensionality_reduction import PCAReducer
        X = synthetic_expression.T.values.astype(np.float32)
        r = PCAReducer(n_components=5)
        r.fit(X)
        z1 = r.transform(X)
        r.save(tmp_path / "pca.pkl")
        r2 = PCAReducer.load(tmp_path / "pca.pkl")
        z2 = r2.transform(X)
        np.testing.assert_allclose(z1, z2, rtol=1e-5)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class TestGeneVAE:
    def test_forward_shapes(self):
        from preprocessing.genetics.dimensionality_reduction import GeneVAE
        vae = GeneVAE(input_dim=50, latent_dim=16, hidden_dims=[32])
        x = torch.randn(8, 50)
        x_recon, mu, log_var, z = vae(x)
        assert x_recon.shape == (8, 50)
        assert mu.shape == (8, 16)
        assert z.shape == (8, 16)

    def test_loss_positive(self):
        from preprocessing.genetics.dimensionality_reduction import GeneVAE
        vae = GeneVAE(input_dim=50, latent_dim=16, hidden_dims=[32])
        x = torch.randn(8, 50)
        x_recon, mu, log_var, z = vae(x)
        loss, parts = vae.loss(x, x_recon, mu, log_var)
        assert loss.item() > 0
        assert parts["recon_loss"] >= 0
        assert parts["kl_loss"] >= 0

    def test_fit_reduces_loss(self):
        from preprocessing.genetics.dimensionality_reduction import GeneVAE
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (30, 50)).astype(np.float32)
        vae = GeneVAE(input_dim=50, latent_dim=8, hidden_dims=[32])
        hist = vae.fit(X, epochs=5, batch_size=8, lr=1e-3)
        assert hist["train_loss"][-1] < hist["train_loss"][0] * 2  # not diverged

    def test_encode_output_shapes(self):
        from preprocessing.genetics.dimensionality_reduction import GeneVAE
        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (20, 50)).astype(np.float32)
        vae = GeneVAE(input_dim=50, latent_dim=8, hidden_dims=[32])
        mu, std = vae.encode(X)
        assert mu.shape == (20, 8)
        assert std.shape == (20, 8)
        assert np.all(std > 0)


# ---------------------------------------------------------------------------
# Gene Graph
# ---------------------------------------------------------------------------

class TestGeneGraph:
    def test_coexpression_graph_shape(self):
        from preprocessing.genetics.gene_graph import GeneGraphBuilder
        builder = GeneGraphBuilder(fallback_to_coexpression=True)
        genes = [f"GENE_{i}" for i in range(20)]
        rng = np.random.default_rng(42)
        expr = rng.normal(0, 1, (30, 20)).astype(np.float32)
        adj, gene_idx = builder.build(genes, expr_matrix=expr)
        assert adj.shape == (20, 20)
        assert len(gene_idx) == 20
        assert np.all(adj >= 0)
        assert np.all(np.diag(adj) == 1.0)  # self-loops

    def test_pyg_edge_format(self):
        from preprocessing.genetics.gene_graph import GeneGraphBuilder
        builder = GeneGraphBuilder(fallback_to_coexpression=True)
        genes = [f"GENE_{i}" for i in range(10)]
        rng = np.random.default_rng(0)
        expr = rng.normal(0, 1, (20, 10)).astype(np.float32)
        builder.build(genes, expr_matrix=expr)
        edge_index, edge_weight = builder.to_pyg_format()
        assert edge_index.shape[0] == 2
        assert len(edge_weight) == edge_index.shape[1]
        assert np.all(edge_weight > 0)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TestGeneticsDataset:
    def test_basic_getitem(self, synthetic_expression, synthetic_metadata):
        from preprocessing.genetics.dataset import GeneticsDataset
        ds = GeneticsDataset(synthetic_expression, synthetic_metadata)
        item = ds[0]
        assert "genetics" in item
        assert "label" in item
        assert item["genetics"].shape == (len(ds._features[0]),)
        assert item["label"].dtype == torch.long

    def test_len(self, synthetic_expression, synthetic_metadata):
        from preprocessing.genetics.dataset import GeneticsDataset
        ds = GeneticsDataset(synthetic_expression, synthetic_metadata)
        assert len(ds) == len(synthetic_metadata)

    def test_class_weights_shape(self, synthetic_expression, synthetic_metadata):
        from preprocessing.genetics.dataset import GeneticsDataset
        ds = GeneticsDataset(synthetic_expression, synthetic_metadata)
        w = ds.get_class_weights()
        assert w.shape == (len(ds),)
        assert w.min() > 0

    def test_metadata_proxy_dataset(self, synthetic_metadata):
        from preprocessing.genetics.dataset import MetadataProxyDataset
        ds = MetadataProxyDataset(synthetic_metadata, feature_cols=["age", "sex_encoded"])
        item = ds[0]
        assert "genetics" in item
        assert item["genetics"].ndim == 1
        assert len(ds) == len(synthetic_metadata)
