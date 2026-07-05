"""
Bidirectional Cross-Attention Fusion — primary novel contribution.

Motivation
----------
Standard fusion methods (concatenation, gating) treat the two modalities
symmetrically but independently: they never let each modality explicitly
query specific aspects of the other.  Cross-attention breaks this limitation:

  MRI attends to genetics:    "Which genetic variants are associated with
                               the structural anomalies I detect in this scan?"
  Genetics attends to MRI:    "Given these gene expression levels, which
                               cortical thickness changes should I expect?"

This bidirectional exchange aligns complementary information across the
biological levels represented by each modality.

Multi-token expansion
---------------------
With single-vector inputs (B, dim), cross-attention degenerates:
softmax of a single Q×K product is always 1.0 — no selection happens.
To make attention non-trivial, each modality vector is projected to
n_tokens=4 tokens.  Now the 4 MRI tokens each attend to the 4 genetics
tokens, learning different "aspects" of the genetics relevant to imaging
(e.g., one token may specialise in synaptic genes, another in myelination).

Publication context
-------------------
No published ASD paper uses bidirectional cross-attention between 3-D MRI
features and genetics features.  This is our primary architectural novelty.

References
----------
Lu J et al. (2019). ViLBERT: Pretraining task-agnostic visiolinguistic
  representations for vision-and-language tasks. NeurIPS.
Chen YC et al. (2020). UNITER: Universal Image-Text Representation Learning.
  ECCV. — bidirectional cross-attention in multimodal BERT.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _BidirectionalCrossAttentionBlock(nn.Module):
    """
    One layer of bidirectional cross-attention.

    Each pass:
      mri'  = LayerNorm(mri  + CrossAttn(Q=mri,  KV=gen))
      gen'  = LayerNorm(gen  + CrossAttn(Q=gen,  KV=mri'))
      mri'' = LayerNorm(mri' + FFN(mri'))
      gen'' = LayerNorm(gen' + FFN(gen'))

    Pre-LN ordering (norm before attention) for training stability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # MRI attends to genetics
        self.mri_to_gen_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Genetics attends to MRI
        self.gen_to_mri_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward networks
        self.mri_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.gen_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        self.mri_norm1 = nn.LayerNorm(d_model)
        self.mri_norm2 = nn.LayerNorm(d_model)
        self.gen_norm1 = nn.LayerNorm(d_model)
        self.gen_norm2 = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)

        # Set True by CrossAttentionFusion.get_attention_weights()
        self._save_attn: bool = False
        self._mri_attn_w: Optional[torch.Tensor] = None
        self._gen_attn_w: Optional[torch.Tensor] = None

    def forward(
        self,
        mri: torch.Tensor,
        gen: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mri : (B, n_mri_tokens, d_model)
        gen : (B, n_gen_tokens, d_model)
        Returns enriched (mri, gen) of the same shapes.
        """
        # --- MRI attends to genetics (pre-LN) ---
        mri_q = self.mri_norm1(mri)
        attn_out, mri_w = self.mri_to_gen_attn(
            mri_q, gen, gen,
            need_weights=self._save_attn,
            average_attn_weights=False,
        )
        if self._save_attn:
            self._mri_attn_w = mri_w.detach()
        mri = mri + self.drop(attn_out)

        # --- Genetics attends to updated MRI ---
        gen_q = self.gen_norm1(gen)
        attn_out, gen_w = self.gen_to_mri_attn(
            gen_q, mri, mri,
            need_weights=self._save_attn,
            average_attn_weights=False,
        )
        if self._save_attn:
            self._gen_attn_w = gen_w.detach()
        gen = gen + self.drop(attn_out)

        # --- Feed-forward (pre-LN) ---
        mri = mri + self.mri_ffn(self.mri_norm2(mri))
        gen = gen + self.gen_ffn(self.gen_norm2(gen))

        return mri, gen


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention multimodal fusion.

    Parameters
    ----------
    mri_dim : int
        Input MRI feature dimensionality.
    gen_dim : int
        Input genetics feature dimensionality.
    fusion_dim : int
        Internal token dimension and final fused feature size.
    n_heads : int
        Number of attention heads (must divide fusion_dim).
    n_layers : int
        Number of stacked bidirectional cross-attention blocks.
    n_tokens : int
        Number of tokens each modality vector is expanded to.
        n_tokens=1 is valid but degenerates to a weighted linear combination.
        n_tokens=4 (default) yields non-trivial 4×4 attention matrices.
    ffn_dim : int
        Feed-forward hidden dimension.
    num_classes : int
    dropout : float
    """

    def __init__(
        self,
        mri_dim: int,
        gen_dim: int,
        fusion_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        n_tokens: int = 4,
        ffn_dim: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if fusion_dim % n_heads != 0:
            raise ValueError(
                f"fusion_dim={fusion_dim} must be divisible by n_heads={n_heads}"
            )
        self.fusion_dim = fusion_dim
        self.n_tokens = n_tokens

        # Expand each modality vector to n_tokens tokens in fusion_dim space
        self.mri_to_tokens = nn.Sequential(
            nn.Linear(mri_dim, n_tokens * fusion_dim),
            nn.Unflatten(-1, (n_tokens, fusion_dim)),
            nn.LayerNorm(fusion_dim),
        )
        self.gen_to_tokens = nn.Sequential(
            nn.Linear(gen_dim, n_tokens * fusion_dim),
            nn.Unflatten(-1, (n_tokens, fusion_dim)),
            nn.LayerNorm(fusion_dim),
        )

        # Stacked cross-attention blocks
        self.ca_blocks = nn.ModuleList([
            _BidirectionalCrossAttentionBlock(fusion_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        # Learned token-level pooling: (B, n_tokens, fusion_dim) → (B, fusion_dim)
        self.mri_pool_w = nn.Parameter(torch.ones(1, n_tokens, 1) / n_tokens)
        self.gen_pool_w = nn.Parameter(torch.ones(1, n_tokens, 1) / n_tokens)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

        # Projection to export a fusion_dim fused vector for downstream use
        self.fused_proj = nn.Sequential(
            nn.Linear(2 * fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        self._init_weights()
        logger.info(
            "CrossAttentionFusion: mri_dim=%d, gen_dim=%d → fusion_dim=%d "
            "(%d tokens × %d layers × %d heads)",
            mri_dim, gen_dim, fusion_dim, n_tokens, n_layers, n_heads,
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        mri_features: torch.Tensor,
        gen_features: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        mri_features : (B, mri_dim)
        gen_features : (B, gen_dim)

        Returns
        -------
        dict:
          "logits"          : (B, num_classes)
          "fused_features"  : (B, fusion_dim)
        """
        # Expand to token sequences
        mri = self.mri_to_tokens(mri_features)  # (B, n_tokens, fusion_dim)
        gen = self.gen_to_tokens(gen_features)   # (B, n_tokens, fusion_dim)

        # Stacked bidirectional cross-attention
        for block in self.ca_blocks:
            mri, gen = block(mri, gen)

        # Learned weighted pooling over tokens → (B, fusion_dim)
        mri_w = torch.softmax(self.mri_pool_w, dim=1)   # (1, n_tokens, 1)
        gen_w = torch.softmax(self.gen_pool_w, dim=1)
        mri_pooled = (mri_w * mri).sum(dim=1)            # (B, fusion_dim)
        gen_pooled = (gen_w * gen).sum(dim=1)

        # Concatenate enriched representations from both modalities
        concat = torch.cat([mri_pooled, gen_pooled], dim=-1)  # (B, 2*fusion_dim)

        logits = self.classifier(concat)
        fused = self.fused_proj(concat)

        return {"logits": logits, "fused_features": fused}

    def get_attention_weights(
        self, mri_features: torch.Tensor, gen_features: torch.Tensor
    ) -> dict:
        """
        Extract cross-attention weight matrices for visualisation.

        Returns
        -------
        dict:
          "mri_to_gen": list of (B, n_heads, n_tokens, n_tokens) per layer
          "gen_to_mri": list of (B, n_heads, n_tokens, n_tokens) per layer
        """
        for block in self.ca_blocks:
            block._save_attn = True
        try:
            with torch.no_grad():
                _ = self.forward(mri_features, gen_features)
            return {
                "mri_to_gen": [b._mri_attn_w for b in self.ca_blocks],
                "gen_to_mri": [b._gen_attn_w for b in self.ca_blocks],
            }
        finally:
            for block in self.ca_blocks:
                block._save_attn = False
                block._mri_attn_w = None
                block._gen_attn_w = None
