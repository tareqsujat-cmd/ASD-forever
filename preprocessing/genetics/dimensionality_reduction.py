"""
Dimensionality reduction for gene expression features.

After feature selection we have ~1000 genes.  The feature extraction models
(Transformer, TabNet) can work at this scale, but we also support:
1. PCA — fast, interpretable, principled
2. Variational Autoencoder (VAE) — learns a non-linear latent manifold;
   the probabilistic latent space is better for downstream classification
   than a deterministic AE because it encourages disentanglement

Why VAE over regular Autoencoder?
----------------------------------
A regular AE can learn a degenerate encoding where all points collapse to a
single region of the latent space (posterior collapse).  The VAE's KL
divergence term regularizes the latent space to be approximately Gaussian,
producing a smooth, interpolable manifold.  This is important for ASD because:
  - The genetic basis is continuous (polygenicity), not binary
  - The latent space is used as input to the fusion model — smooth gradients
    improve cross-attention training stability

Reference
---------
Kingma DP, Welling M. (2014). Auto-Encoding Variational Bayes. ICLR 2014.
arXiv:1312.6114

Usage
-----
    from preprocessing.genetics.dimensionality_reduction import (
        PCAReducer, GeneVAE
    )

    # PCA
    pca = PCAReducer(n_components=256)
    pca.fit(expr_train_T)    # (n_samples, n_genes)
    latent = pca.transform(expr_test_T)

    # VAE
    vae = GeneVAE(input_dim=1000, latent_dim=256)
    vae.fit(expr_train_T)
    z_mean, z_std = vae.encode(expr_test_T)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PCA Reducer
# ---------------------------------------------------------------------------

class PCAReducer:
    """
    PCA-based dimensionality reduction with sklearn-style fit/transform API.

    Whitens the data (unit variance components) by default since gene expression
    features have very different scales across genes.

    Parameters
    ----------
    n_components : int or float
        If int: exact number of components.
        If float (0 < x < 1): number of components to explain that fraction
        of variance.
    whiten : bool
        Normalize component scales to unit variance.
    random_state : int
    """

    def __init__(
        self,
        n_components: Union[int, float] = 256,
        whiten: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self._pca = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "PCAReducer":
        """
        Fit PCA on training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_genes)
        """
        from sklearn.decomposition import PCA
        self._pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
            svd_solver="full",
        )
        self._pca.fit(X)
        explained = self._pca.explained_variance_ratio_.cumsum()[-1]
        logger.info(f"PCA: {self._pca.n_components_} components, "
                    f"explained variance = {explained:.3f}")
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        return self._pca.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_component_genes(
        self, gene_names: list, top_k: int = 10
    ) -> dict:
        """
        Return the top-k genes contributing to each principal component.

        Used for biological interpretation and paper figures.

        Parameters
        ----------
        gene_names : list
            Gene names corresponding to input features.
        top_k : int
            Number of top genes per component.

        Returns
        -------
        dict: {component_idx: [(gene, loading), ...]}
        """
        components = {}
        for i, comp in enumerate(self._pca.components_):
            top_idx = np.argsort(np.abs(comp))[-top_k:][::-1]
            components[i] = [(gene_names[j], float(comp[j])) for j in top_idx]
        return components

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "PCAReducer":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Variational Autoencoder
# ---------------------------------------------------------------------------

class _VAEEncoder(nn.Module):
    """Encoder q(z|x): maps gene expression to (mu, log_var)."""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        self.network = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, latent_dim)
        self.log_var_head = nn.Linear(prev, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.network(x)
        return self.mu_head(h), self.log_var_head(h)


class _VAEDecoder(nn.Module):
    """Decoder p(x|z): maps latent code back to gene expression."""

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int) -> None:
        super().__init__()
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class GeneVAE(nn.Module):
    """
    Variational Autoencoder for gene expression dimensionality reduction.

    Architecture
    ------------
    Encoder: input_dim -> [512, 256] -> (mu, log_var) ∈ R^latent_dim
    Reparameterization: z = mu + eps * exp(0.5 * log_var)
    Decoder: latent_dim -> [256, 512] -> input_dim

    Loss
    ----
    ELBO = E[log p(x|z)] - beta * KL[q(z|x) || p(z)]
    where p(z) = N(0,I)

    beta > 1 (beta-VAE) encourages more disentangled latent factors.

    Parameters
    ----------
    input_dim : int
        Number of genes (after feature selection).
    latent_dim : int
        Latent space dimensionality.
    hidden_dims : list of int
        Encoder hidden layer sizes.
    beta : float
        KL weight for beta-VAE (1.0 = standard VAE).
    """

    def __init__(
        self,
        input_dim: int = 1000,
        latent_dim: int = 256,
        hidden_dims: Optional[list] = None,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = _VAEEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = _VAEDecoder(latent_dim, hidden_dims, input_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + eps * sigma using the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var, z

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        ELBO loss = reconstruction loss + beta * KL divergence.

        We use MSE reconstruction loss (appropriate for normalized continuous
        gene expression values).  For raw count data, use Poisson NLL.
        """
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        total = recon_loss + self.beta * kl_loss
        return total, {"recon_loss": recon_loss.item(), "kl_loss": kl_loss.item()}

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "cpu",
        val_X: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Train the VAE on gene expression data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_genes)
            Training data (normalized expression).
        epochs : int
        batch_size : int
        lr : float
            Learning rate.
        device : str
        val_X : np.ndarray, optional
            Validation data for early stopping.

        Returns
        -------
        dict
            Training history: {"train_loss": [...], "val_loss": [...]}
        """
        self.to(device)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=False)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        patience_count = 0
        patience = 20

        logger.info(f"Training VAE: input_dim={self.input_dim}, "
                    f"latent_dim={self.latent_dim}, epochs={epochs}")

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0

            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                optimizer.zero_grad()
                x_recon, mu, log_var, _ = self(batch_x)
                loss, loss_parts = self.loss(batch_x, x_recon, mu, log_var)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            epoch_loss /= len(X)
            history["train_loss"].append(epoch_loss)
            scheduler.step()

            # Validation
            if val_X is not None:
                val_loss = self._eval_loss(val_X, device, batch_size)
                history["val_loss"].append(val_loss)
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    patience_count = 0
                else:
                    patience_count += 1
                if patience_count >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                val_str = f", val={history['val_loss'][-1]:.4f}" if val_X is not None else ""
                logger.info(f"VAE epoch {epoch + 1}/{epochs}: "
                            f"train={epoch_loss:.4f}{val_str}")

        return history

    @torch.no_grad()
    def encode(
        self, X: np.ndarray, device: str = "cpu"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode expression matrix to latent space.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_genes)

        Returns
        -------
        mu : np.ndarray, shape (n_samples, latent_dim)
        std : np.ndarray, shape (n_samples, latent_dim)
        """
        self.eval()
        self.to(device)
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        mu, log_var = self.encoder(X_t)
        std = torch.exp(0.5 * log_var)
        return mu.cpu().numpy(), std.cpu().numpy()

    @torch.no_grad()
    def _eval_loss(
        self, X: np.ndarray, device: str, batch_size: int
    ) -> float:
        self.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
        total = 0.0
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            x_recon, mu, log_var, _ = self(batch_x)
            loss, _ = self.loss(batch_x, x_recon, mu, log_var)
            total += loss.item() * len(batch_x)
        return total / len(X)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "beta": self.beta,
            },
        }, path)
        logger.info(f"VAE saved: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GeneVAE":
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
