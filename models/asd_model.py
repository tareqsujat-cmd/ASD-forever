"""
ASDModel — full end-to-end ASD detection model.

Combines:
  MRIEncoder (Module 3) → (B, mri_feature_dim)
  GeneticsEncoder (Module 4) → (B, gen_feature_dim)
  MultiModalFusion (Module 5) → {logits, fused_features}

This wrapper is the single object passed to the Trainer.  Downstream code
(explainability, ablation runner) only imports ASDModel.

Operating modes
---------------
"multimodal"     : full MRI + genetics pipeline (default)
"mri_only"       : MRI encoder + linear classifier (no genetics)
"genetics_only"  : genetics encoder + linear classifier (no MRI)

Modes are used by the ablation runner (Module 10); the Trainer always
uses "multimodal".
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ASDModel(nn.Module):
    """
    Full ASD detection model.

    Parameters
    ----------
    mri_encoder : MRIEncoder
    gen_encoder : GeneticsEncoder
    fusion : MultiModalFusion
    """

    def __init__(
        self,
        mri_encoder: nn.Module,
        gen_encoder: nn.Module,
        fusion: nn.Module,
    ) -> None:
        super().__init__()
        self.mri_encoder = mri_encoder
        self.gen_encoder = gen_encoder
        self.fusion = fusion

        from utilities.hardware import count_parameters
        _, trainable = count_parameters(self)
        logger.info("ASDModel: %s trainable parameters", f"{trainable:,}")

    def forward(
        self,
        mri: torch.Tensor,
        genetics: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Parameters
        ----------
        mri      : (B, 1, D, H, W)
        genetics : (B, n_genes)
        adj      : (n_genes, n_genes), optional — for GNN genetics backend

        Returns
        -------
        dict:
          "logits"         : (B, num_classes)
          "fused_features" : (B, fusion_dim)
          "mri_features"   : (B, mri_feature_dim)   — for explainability
          "gen_features"   : (B, gen_feature_dim)   — for explainability
        """
        mri_features = self.mri_encoder(mri)
        gen_features = self.gen_encoder(genetics, adj)

        fusion_out = self.fusion(mri_features, gen_features)

        return {
            **fusion_out,
            "mri_features": mri_features,
            "gen_features": gen_features,
        }

    def forward_mri_only(self, mri: torch.Tensor) -> torch.Tensor:
        """Return MRI feature vector (used by GradCAM / MRI-only ablation)."""
        return self.mri_encoder(mri)

    def forward_gen_only(self, genetics: torch.Tensor) -> torch.Tensor:
        """Return genetics feature vector (used by genetics-only ablation)."""
        return self.gen_encoder(genetics)
