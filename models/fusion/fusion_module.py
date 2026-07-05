"""
MultiModalFusion — unified fusion wrapper.

All downstream modules (training engine, evaluation, explainability) import
MultiModalFusion only, never the concrete fusion classes directly.  Swapping
fusion methods is a one-line config change, zero code change.

Interface contract
------------------
    output = fusion(mri_features, gen_features)
    # output is always a dict containing at minimum:
    #   "logits"         : (B, num_classes)
    #   "fused_features" : (B, fusion_dim)
    # Some fusion methods add extra keys (gate_weights, modality_weights, etc.)

Fusion dim for downstream modules
----------------------------------
fusion.feature_dim is the dimension of "fused_features".  This is what the
explainability module (Module 8) uses for SHAP / Integrated Gradients.

For LateFusion, fusion_dim = 2 * num_classes (probability concatenation),
not the usual fusion_dim.  The factory handles this automatically.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiModalFusion(nn.Module):
    """
    Thin wrapper around a concrete fusion backend.

    Parameters
    ----------
    backend : nn.Module
        One of: CrossAttentionFusion, GatedFusion, IntermediateFusion,
        LateFusion, DynamicFusion.
    fusion_dim : int
        Output feature dimensionality.  Stored for downstream modules.
    num_classes : int
    method : str
        Name of the fusion method (for logging / serialisation).
    """

    def __init__(
        self,
        backend: nn.Module,
        fusion_dim: int,
        num_classes: int,
        method: str,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        self.method = method

    def forward(
        self,
        mri_features: torch.Tensor,
        gen_features: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        mri_features : (B, mri_feature_dim)
        gen_features : (B, gen_feature_dim)

        Returns
        -------
        dict containing at minimum:
            "logits"         : (B, num_classes)
            "fused_features" : (B, fusion_dim)
        """
        return self.backend(mri_features, gen_features)

    def get_attention_weights(
        self,
        mri_features: torch.Tensor,
        gen_features: torch.Tensor,
    ) -> Optional[dict]:
        """
        Return cross-attention weights if the backend supports it.
        Used by Module 8 (explainability) for attention visualisation.
        """
        from models.fusion.cross_attention import CrossAttentionFusion
        if isinstance(self.backend, CrossAttentionFusion):
            return self.backend.get_attention_weights(mri_features, gen_features)
        return None
