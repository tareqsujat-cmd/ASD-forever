"""
Unified Genetics Encoder wrapper.

Role in the system
------------------
This class wraps any genetics backbone (Transformer, TabNet, GNN, MLP) behind
a single interface used by all downstream modules:
  - Fusion module: receives `features` vector (B, feature_dim)
  - Training engine: reads `last_aux_loss` attribute for regularisation
  - Ablation runner: swaps backbone via config string, zero code changes

Auxiliary loss contract
-----------------------
TabNet produces a sparsity-entropy regularisation term.
GNNEncoder returns a zero scalar (API symmetry).
GeneTransformerEncoder and MLPEncoder return a zero scalar.

The training engine adds:  total_loss += genetics_lambda * encoder.last_aux_loss
No special casing required — every backbone adheres to the same contract.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _MLPEncoder(nn.Module):
    """
    Simple MLP baseline encoder.

    Used in ablation studies to quantify how much of the genetics contribution
    comes from architecture (Transformer / GAT) vs. raw feature information.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dims: list,
        feature_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        layers = []
        in_d = n_genes
        for h in hidden_dims:
            layers += [
                nn.Linear(in_d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_d = h
        layers += [
            nn.Linear(in_d, feature_dim),
            nn.LayerNorm(feature_dim),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeneticsEncoder(nn.Module):
    """
    Unified wrapper around any genetics backbone.

    All downstream modules import GeneticsEncoder only, never the backbone
    classes directly.  This is critical for the ablation study.

    Parameters
    ----------
    backbone : nn.Module
        Any backbone with a compatible forward() signature (see below).
    feature_dim : int
        Expected output dimensionality from backbone.
    dropout : float
        Additional output dropout applied after the backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(feature_dim)

        # Detect whether backbone returns (features, aux_loss) or just features
        from models.genetics.tabnet import TabNetEncoder
        from models.genetics.gnn_encoder import GNNEncoder
        self._returns_aux = isinstance(backbone, (TabNetEncoder, GNNEncoder))

        # last_aux_loss is read by the training engine after every forward call
        self.last_aux_loss: torch.Tensor = torch.zeros(1)

    def forward(
        self, x: torch.Tensor, adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract feature vector from gene expression inputs.

        Parameters
        ----------
        x   : (B, n_genes) — normalised gene expression values
        adj : (n_genes, n_genes), optional
            Only used when backbone is GNNEncoder.  If the backbone
            has a built-in adjacency buffer this parameter is ignored.

        Returns
        -------
        (B, feature_dim)
        """
        from models.genetics.gnn_encoder import GNNEncoder

        if isinstance(self.backbone, GNNEncoder):
            # GNNEncoder has its adjacency registered as a buffer;
            # the adj parameter is accepted for API symmetry but ignored here.
            features, aux = self.backbone(x)
        elif self._returns_aux:
            features, aux = self.backbone(x)
        else:
            features = self.backbone(x)
            aux = torch.zeros(1, device=x.device)

        self.last_aux_loss = aux
        features = self.output_norm(self.output_dropout(features))
        return features

    def get_feature_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return attention weights or feature-importance masks for explainability.

        Returns
        -------
        For Transformer: list of per-layer attention tensors (from get_attention_weights)
        For TabNet: (B, n_genes) importance mask (from get_feature_importances)
        For others: None
        """
        from models.genetics.transformer_encoder import GeneTransformerEncoder
        from models.genetics.tabnet import TabNetEncoder

        if isinstance(self.backbone, GeneTransformerEncoder):
            return self.backbone.get_attention_weights(x)
        if isinstance(self.backbone, TabNetEncoder):
            _ = self.backbone(x)  # populate masks
            return self.backbone.get_feature_importances()
        return None


class GeneticsClassifier(nn.Module):
    """
    Genetics-only classifier for single-modality ablation baseline.

    Used in:
    1. Ablation study (genetics-only vs fusion)
    2. Pre-training the genetics encoder before fusion fine-tuning
    3. Publication baseline
    """

    def __init__(
        self,
        encoder: GeneticsEncoder,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.feature_dim, encoder.feature_dim // 2),
            nn.LayerNorm(encoder.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(encoder.feature_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
            "logits"   : (B, num_classes)
            "features" : (B, feature_dim)
        """
        features = self.encoder(x, adj)
        return {"logits": self.classifier(features), "features": features}
