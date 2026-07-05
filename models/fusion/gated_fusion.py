"""
Alternative fusion strategies used in ablation study.

Four methods are implemented here, ordered by complexity:

1. IntermediateFusion — concatenate projected features, classify
   - Simplest non-trivial fusion; strong baseline in practice
   - Expressiveness: linear combination in the projection space

2. GatedFusion — input-conditioned scalar gate per modality
   - The gate learns when to trust MRI vs genetics for a given sample
   - Interpretable: gate values indicate modality reliability

3. LateFusion — separate per-modality classifiers, ensemble at logit level
   - Each modality classified independently, then averaged/weighted
   - Best when modalities are conditionally independent given the label

4. DynamicFusion — sample-wise softmax attention over modalities
   - Similar to GatedFusion but normalised via softmax (sum-to-1 constraint)
   - Interpretable attention weights are a natural measure of modality quality
     (e.g., low-quality genetics data → attention collapses to MRI)

References
----------
Ramachandram D, Taylor GW. (2017). Deep Multimodal Learning: A survey on
  recent advances and trends. IEEE Signal Processing Magazine.
Zadeh A et al. (2018). Memory Fusion Network for Multi-view Sequential
  Learning. AAAI-18. — gated fusion for sentiment analysis.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_classifier(in_dim: int, hidden_dim: int, num_classes: int, dropout: float):
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout / 2),
        nn.Linear(hidden_dim, num_classes),
    )


# ---------------------------------------------------------------------------
# 1. Intermediate (Concatenation) Fusion
# ---------------------------------------------------------------------------

class IntermediateFusion(nn.Module):
    """
    Project-then-concatenate fusion.

    mri → Linear → mri_proj
    gen → Linear → gen_proj
    fused = MLP(concat([mri_proj, gen_proj]))
    """

    def __init__(
        self,
        mri_dim: int,
        gen_dim: int,
        fusion_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim

        self.mri_proj = nn.Sequential(
            nn.Linear(mri_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gen_proj = nn.Sequential(
            nn.Linear(gen_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = _make_classifier(fusion_dim, fusion_dim // 2, num_classes, dropout)

        logger.info(
            "IntermediateFusion: mri_dim=%d, gen_dim=%d → fusion_dim=%d",
            mri_dim, gen_dim, fusion_dim,
        )

    def forward(
        self, mri_features: torch.Tensor, gen_features: torch.Tensor
    ) -> dict:
        m = self.mri_proj(mri_features)
        g = self.gen_proj(gen_features)
        fused = self.fusion_net(torch.cat([m, g], dim=-1))
        return {"logits": self.classifier(fused), "fused_features": fused}


# ---------------------------------------------------------------------------
# 2. Gated Fusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    """
    Input-conditioned per-modality scalar gates.

    Gate network: Linear(mri_dim + gen_dim → 2) → gate_activation → (α_mri, α_gen)
    Fused = α_mri * W_mri(mri)  +  α_gen * W_gen(gen)

    When gate_activation="sigmoid", gates are independent and can both be high.
    When gate_activation="softmax", gates sum to 1 (exclusive weighting).
    """

    def __init__(
        self,
        mri_dim: int,
        gen_dim: int,
        fusion_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
        gate_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.gate_activation = gate_activation

        self.mri_proj = nn.Sequential(
            nn.Linear(mri_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.gen_proj = nn.Sequential(
            nn.Linear(gen_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        # Gate network: maps concatenated inputs to 2 scalar gates
        self.gate_net = nn.Sequential(
            nn.Linear(mri_dim + gen_dim, (mri_dim + gen_dim) // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear((mri_dim + gen_dim) // 2, 2),
        )

        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = _make_classifier(fusion_dim, fusion_dim // 2, num_classes, dropout)

        logger.info(
            "GatedFusion: mri_dim=%d, gen_dim=%d, gate=%s → fusion_dim=%d",
            mri_dim, gen_dim, gate_activation, fusion_dim,
        )

    def forward(
        self, mri_features: torch.Tensor, gen_features: torch.Tensor
    ) -> dict:
        m = self.mri_proj(mri_features)   # (B, fusion_dim)
        g = self.gen_proj(gen_features)   # (B, fusion_dim)

        raw_gates = self.gate_net(torch.cat([mri_features, gen_features], dim=-1))  # (B, 2)

        if self.gate_activation == "sigmoid":
            gates = torch.sigmoid(raw_gates)
        elif self.gate_activation == "softmax":
            gates = torch.softmax(raw_gates, dim=-1)
        else:
            raise ValueError(f"Unknown gate_activation: {self.gate_activation}")

        alpha_m = gates[:, 0:1]  # (B, 1)
        alpha_g = gates[:, 1:2]  # (B, 1)

        fused = self.fusion_norm(alpha_m * m + alpha_g * g)  # (B, fusion_dim)
        return {
            "logits": self.classifier(fused),
            "fused_features": fused,
            "gate_weights": gates.detach(),  # (B, 2) — useful for analysis
        }


# ---------------------------------------------------------------------------
# 3. Late Fusion
# ---------------------------------------------------------------------------

class LateFusion(nn.Module):
    """
    Separate per-modality classifiers, combined at the logit level.

    ensemble_logits = w_mri * mri_logits  +  w_gen * gen_logits
    where w_mri, w_gen are learned scalars initialised to 0.5.

    This corresponds to a linear ensemble of two independent classifiers.
    Best when modalities are conditionally independent given the label, or
    when one modality has poor data quality for some subjects.
    """

    def __init__(
        self,
        mri_dim: int,
        gen_dim: int,
        fusion_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim

        self.mri_classifier = _make_classifier(mri_dim, fusion_dim, num_classes, dropout)
        self.gen_classifier = _make_classifier(gen_dim, fusion_dim, num_classes, dropout)

        # Learnable per-modality ensemble weights (before softmax normalisation)
        self.ensemble_logit_w = nn.Parameter(torch.zeros(2))

        logger.info(
            "LateFusion: mri_dim=%d, gen_dim=%d → %d classes via logit ensemble",
            mri_dim, gen_dim, num_classes,
        )

    def forward(
        self, mri_features: torch.Tensor, gen_features: torch.Tensor
    ) -> dict:
        mri_logits = self.mri_classifier(mri_features)  # (B, num_classes)
        gen_logits = self.gen_classifier(gen_features)   # (B, num_classes)

        # Softmax-normalised ensemble weights: always sum to 1
        w = torch.softmax(self.ensemble_logit_w, dim=0)  # (2,)
        ensemble_logits = w[0] * mri_logits + w[1] * gen_logits

        # fused_features: concatenate the probability-space representations
        fused = torch.cat([
            F.softmax(mri_logits, dim=-1),
            F.softmax(gen_logits, dim=-1),
        ], dim=-1)  # (B, 2 * num_classes) — useful for analysis

        return {
            "logits": ensemble_logits,
            "fused_features": fused,
            "mri_logits": mri_logits.detach(),
            "gen_logits": gen_logits.detach(),
            "ensemble_weights": w.detach(),
        }


# ---------------------------------------------------------------------------
# 4. Dynamic (Attention-based) Fusion
# ---------------------------------------------------------------------------

class DynamicFusion(nn.Module):
    """
    Per-sample softmax attention over modalities.

    Computes a 2-way softmax attention weight from the concatenated features,
    then takes a weighted sum of the projected modalities.

    Unlike GatedFusion (sigmoid, independent gates), the softmax constraint
    makes this a strict trade-off: if MRI is high, genetics is low.
    Particularly useful when genetics data is missing for some subjects (the
    model can learn to ignore it gracefully).

    The attention weights are directly interpretable as per-sample modality
    confidence, and can be correlated with data quality metrics at analysis.
    """

    def __init__(
        self,
        mri_dim: int,
        gen_dim: int,
        fusion_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim

        self.mri_proj = nn.Sequential(
            nn.Linear(mri_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.gen_proj = nn.Sequential(
            nn.Linear(gen_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # Attention score network: (mri_dim + gen_dim) → 2 scores → softmax
        hidden = (mri_dim + gen_dim) // 4
        self.attn_net = nn.Sequential(
            nn.Linear(mri_dim + gen_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.classifier = _make_classifier(fusion_dim, fusion_dim // 2, num_classes, dropout)

        logger.info(
            "DynamicFusion: mri_dim=%d, gen_dim=%d → fusion_dim=%d",
            mri_dim, gen_dim, fusion_dim,
        )

    def forward(
        self, mri_features: torch.Tensor, gen_features: torch.Tensor
    ) -> dict:
        m = self.mri_proj(mri_features)
        g = self.gen_proj(gen_features)

        # Per-sample modality attention weights: (B, 2)
        scores = self.attn_net(torch.cat([mri_features, gen_features], dim=-1))
        weights = torch.softmax(scores, dim=-1)  # sums to 1 per sample

        fused = self.fusion_norm(
            weights[:, 0:1] * m + weights[:, 1:2] * g
        )

        return {
            "logits": self.classifier(fused),
            "fused_features": fused,
            "modality_weights": weights.detach(),  # (B, 2) — [mri_weight, gen_weight]
        }
