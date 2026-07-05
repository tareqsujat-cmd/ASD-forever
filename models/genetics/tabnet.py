"""
TabNet encoder for gene expression data.

TabNet uses sequential, attention-based feature selection: each step learns
to attend to a *different* sparse subset of genes, building an interpretable
representation across N_steps iterations.

Key properties for genetics data
---------------------------------
1. Instance-wise sparsity: different samples may rely on different gene subsets,
   matching the biological reality that ASD subtypes have distinct genetic profiles.
2. Built-in explainability: the accumulated attention masks directly identify
   which genes drove each prediction (no separate SHAP/IG computation needed).
3. Strong implicit regularisation via sparsity penalty prevents overfitting on
   small datasets (~200 samples after train/val split per fold).
4. Ghost Batch Normalisation decouples performance from batch size.

Architecture (N_steps=3, N_d=N_a=64):
  Input: (B, n_genes) → Ghost BN
  For t in 0..N_steps-1:
    Attentive transformer: M[t] = Sparsemax(prior[t] ⊙ h[t-1] → linear → BN)
    Update prior: P[t] = P[t-1] ⊙ (γ - M[t])
    Feature transformer: h[t] = FeatureTransformer(M[t] ⊙ x_BN)
    Decision output: d[t] = ReLU(h[t][:n_d])
  Output = (Σ d[t]) / √N_steps → projection head

Reference
---------
Arik SO, Pfister T. (2021). TabNet: Attentive Interpretable Tabular Learning.
AAAI-21. arXiv:1908.07442.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sparsemax
# ---------------------------------------------------------------------------

def _sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax activation (Martins & Astudillo, 2016).

    Unlike softmax (always dense), sparsemax projects onto the probability
    simplex via an optimal threshold, setting many entries to exactly 0.
    Critical for TabNet's interpretability: features with weight 0 are
    definitively ruled out for that sample at that step.

    Algorithm: sort descending, find threshold via cumulative sum,
    clamp negatives to 0.
    """
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)

    # k_shape: all 1s except along `dim`
    k_shape = [1] * z.dim()
    k_shape[dim] = z.shape[dim]
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    k = k.view(k_shape)

    valid = (1 + k * z_sorted > z_cumsum)               # (…, n_genes)
    k_z = valid.sum(dim=dim, keepdim=True).clamp(min=1)  # number of non-zero entries
    tau = (z_cumsum.gather(dim, k_z.long() - 1) - 1.0) / k_z.float()
    return torch.clamp(z - tau, min=0.0)


# ---------------------------------------------------------------------------
# Ghost Batch Normalisation
# ---------------------------------------------------------------------------

class _GhostBN(nn.Module):
    """
    Ghost Batch Normalisation (Hoffer et al., 2017).

    Splits the real batch into virtual mini-batches of `virtual_batch_size`
    and applies BN to each independently.  Gives stable statistics when the
    real batch size is small (forced here by 3-D MRI memory constraints).
    """

    def __init__(
        self, num_features: int, virtual_batch_size: int = 8, momentum: float = 0.02
    ) -> None:
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.training and B > self.virtual_batch_size:
            chunks = x.split(self.virtual_batch_size, dim=0)
            return torch.cat([self.bn(c) for c in chunks], dim=0)
        return self.bn(x)


# ---------------------------------------------------------------------------
# Feature Transformer layer (shared + step-specific FC + GLU)
# ---------------------------------------------------------------------------

class _FTLayer(nn.Module):
    """FC(in → out*2) → GhostBN → GLU → (B, out)."""

    def __init__(self, in_dim: int, out_dim: int, vbs: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2, bias=False)
        self.bn = _GhostBN(out_dim * 2, vbs)
        self._scale = math.sqrt(2.0)  # compensate for GLU halving

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn(self.fc(x))
        h1, h2 = h.chunk(2, dim=-1)
        return h1 * torch.sigmoid(h2) * self._scale


class _FeatureTransformer(nn.Module):
    """
    TabNet Feature Transformer with shared + step-specific layers.

    Shared layers are applied identically across all steps (parameter
    efficiency).  Step layers add step-specific capacity.  A residual
    connection scaled by √0.5 connects shared → step output.
    """

    def __init__(
        self,
        input_dim: int,
        n_d: int,
        n_a: int,
        n_shared: int,
        n_steps: int,
        vbs: int,
    ) -> None:
        super().__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.out_dim = n_d + n_a

        # Shared layers
        if n_shared > 0:
            shared = []
            in_d = input_dim
            for _ in range(n_shared):
                shared.append(_FTLayer(in_d, self.out_dim, vbs))
                in_d = self.out_dim
            self.shared = nn.ModuleList(shared)
        else:
            self.shared = None

        # Step-specific layers
        step_in = self.out_dim if n_shared > 0 else input_dim
        self.step_layers = nn.ModuleList([
            _FTLayer(step_in, self.out_dim, vbs) for _ in range(n_steps)
        ])
        self._sqrt05 = math.sqrt(0.5)

    def forward(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Returns (B, n_d + n_a)."""
        h = x
        if self.shared is not None:
            for layer in self.shared:
                h = layer(h)

        h_step = self.step_layers[step](h if self.shared is not None else x)

        if self.shared is not None:
            return (h + h_step) * self._sqrt05
        return h_step


# ---------------------------------------------------------------------------
# TabNet Encoder
# ---------------------------------------------------------------------------

class TabNetEncoder(nn.Module):
    """
    TabNet encoder for gene expression data.

    Parameters
    ----------
    n_genes : int
        Number of input genes.
    n_d : int
        Width of the decision step output (contributes to final representation).
    n_a : int
        Width of the attention step (feeds the attentive transformer).
        Typically n_d == n_a.
    n_steps : int
        Number of sequential attention steps.  3–5 steps for tabular data.
    gamma : float
        Feature re-use coefficient.  Higher γ → more re-use allowed.
        γ=1.0: strict (each feature used once), γ=1.5: moderate.
    n_shared : int
        Number of shared Feature Transformer layers (parameter efficiency).
    epsilon : float
        Numerical stability constant for entropy regularisation.
    virtual_batch_size : int
        Ghost BN virtual mini-batch size.
    feature_dim : int
        Final output dimensionality (projection from n_d).
    dropout : float
        Dropout on projection head.
    """

    def __init__(
        self,
        n_genes: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.5,
        n_shared: int = 2,
        epsilon: float = 1e-15,
        virtual_batch_size: int = 8,
        feature_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.feature_dim = feature_dim

        self.initial_bn = nn.BatchNorm1d(n_genes, momentum=0.02)

        self.feature_transformer = _FeatureTransformer(
            input_dim=n_genes,
            n_d=n_d, n_a=n_a,
            n_shared=n_shared,
            n_steps=n_steps,
            vbs=virtual_batch_size,
        )

        # Attentive transformers: one per step, each maps n_a → n_genes
        self.att_transformers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_a, n_genes, bias=False),
                _GhostBN(n_genes, virtual_batch_size),
            )
            for _ in range(n_steps)
        ])

        self.projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_d, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        self._attention_masks: List[torch.Tensor] = []

        logger.info(
            "TabNetEncoder: %d genes → n_steps=%d, n_d=%d, n_a=%d → feature_dim=%d",
            n_genes, n_steps, n_d, n_a, feature_dim,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, n_genes)

        Returns
        -------
        features : (B, feature_dim)
        sparsity_loss : scalar  — entropy regularisation, add λ * this to training loss.
            Encourages sparse feature selection.
        """
        B = x.shape[0]
        x_bn = self.initial_bn(x)

        # All genes eligible at step 0
        prior_scales = torch.ones(B, self.n_genes, device=x.device)

        # Accumulate decision outputs across steps
        total_d = torch.zeros(B, self.n_d, device=x.device)
        total_entropy = torch.zeros(1, device=x.device)
        self._attention_masks = []

        # Initial hidden state from first application of feature transformer
        h = self.feature_transformer(x_bn, step=0)
        h_a = h[:, self.n_d:]  # attention portion: (B, n_a)

        for step in range(self.n_steps):
            # ----- Attentive transformer -----
            att_logits = self.att_transformers[step](h_a)     # (B, n_genes)
            att_logits = att_logits * prior_scales             # penalise re-used features
            M = _sparsemax(att_logits, dim=-1)                 # (B, n_genes)

            # Update prior: reduce weight for features already selected
            prior_scales = prior_scales * (self.gamma - M)

            # ----- Feature transformer on masked features -----
            masked_x = M * x_bn
            h = self.feature_transformer(masked_x, step=step)
            h_a = h[:, self.n_d:]                             # next step's attention input

            # ----- Accumulate decision output -----
            d = torch.relu(h[:, : self.n_d])                  # (B, n_d)
            total_d = total_d + d

            # ----- Entropy regularisation -----
            entropy = (-M * torch.log(M + self.epsilon)).sum(dim=-1).mean()
            total_entropy = total_entropy + entropy

            self._attention_masks.append(M.detach())

        # Scale by √N_steps (TabNet paper §3.1)
        total_d = total_d / math.sqrt(self.n_steps)
        sparsity_loss = total_entropy / self.n_steps

        return self.projector(total_d), sparsity_loss

    def get_feature_importances(self) -> Optional[torch.Tensor]:
        """
        Aggregate attention masks across steps: per-gene importance.

        Call immediately after forward() to obtain masks from that pass.

        Returns
        -------
        (B, n_genes) importance scores in [0, 1], or None if not yet run.
        """
        if not self._attention_masks:
            return None
        return torch.stack(self._attention_masks, dim=0).mean(dim=0)
