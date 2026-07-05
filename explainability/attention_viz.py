"""
Attention weight extraction and visualization utilities.

Three attention sources are supported:

1. Genetics transformer attention
   The GeneTransformerEncoder exposes per-layer attention matrices via
   ``get_attention_weights()``.  Each matrix is (B, n_heads, N+1, N+1) where
   position 0 is the CLS token and positions 1..N are gene tokens.

2. Fusion cross-attention
   CrossAttentionFusion exposes bidirectional attention via
   ``get_attention_weights()``.  Returns {"mri_to_gen": [...], "gen_to_mri": [...]}.

3. Attention rollout (Abnar & Zuidema, 2020)
   Propagates attention through all layers accounting for residual connections.
   Gives a single (B, N, N) matrix summarising where each token "looks" across
   the full depth of the transformer.

Reference
---------
Abnar S, Zuidema W. (2020). Quantifying Attention Flow in Transformers.
  ACL 2020. arXiv:2005.00928
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------

def attention_rollout(
    attn_matrices: List[torch.Tensor],
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """
    Attention rollout across transformer layers.

    For each layer:
      1. Average heads: Ā = mean_h(A_h)
      2. Add residual: Â = 0.5 * Ā + 0.5 * I
      3. Normalize rows
      4. Accumulate by matrix multiplication: R = Â^L @ ... @ Â^1

    Parameters
    ----------
    attn_matrices : list of (B, n_heads, N, N) tensors, one per layer
    discard_ratio : float in [0, 1)
        Fraction of lowest-attention positions to zero out per head before
        averaging, reducing noise from near-uniform attention.

    Returns
    -------
    rollout : (B, N, N) tensor — values sum to 1 per row
    """
    if not attn_matrices:
        raise ValueError("attn_matrices is empty")

    result: Optional[torch.Tensor] = None

    for attn in attn_matrices:
        # attn: (B, n_heads, N, N)
        B, n_heads, N, _ = attn.shape

        if discard_ratio > 0.0:
            flat = attn.view(B, n_heads, -1)                          # (B, H, N*N)
            threshold = flat.quantile(discard_ratio, dim=-1, keepdim=True)  # (B, H, 1)
            mask = (flat >= threshold).view(B, n_heads, N, N)
            attn = attn * mask.float()

        avg = attn.mean(dim=1)                     # (B, N, N)
        I = torch.eye(N, device=avg.device, dtype=avg.dtype).unsqueeze(0)
        aug = 0.5 * avg + 0.5 * I                 # residual mix
        aug = aug / (aug.sum(dim=-1, keepdim=True) + 1e-8)  # row norm

        if result is None:
            result = aug
        else:
            result = torch.bmm(aug, result)

    return result  # (B, N, N)


def rollout_cls_to_tokens(rollout: torch.Tensor) -> torch.Tensor:
    """
    Extract the CLS token's attention to all other tokens.

    Parameters
    ----------
    rollout : (B, N+1, N+1) — position 0 is CLS

    Returns
    -------
    cls_attn : (B, N) — normalized attention from CLS to gene tokens
    """
    cls_attn = rollout[:, 0, 1:]  # (B, N)
    cls_attn = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-8)
    return cls_attn


# ---------------------------------------------------------------------------
# AttentionExtractor
# ---------------------------------------------------------------------------

class AttentionExtractor:
    """
    Extracts attention weights from any component of the ASD model that
    exposes a ``get_attention_weights()`` method.

    Parameters
    ----------
    genetics_encoder : nn.Module, optional
        A ``GeneTransformerEncoder`` (or compatible) that has
        ``get_attention_weights(x)`` returning a list of ``(B, H, N, N)`` tensors.
    fusion_module : nn.Module, optional
        A ``MultiModalFusion`` (or compatible) that has
        ``get_attention_weights(mri_feat, gen_feat)`` returning a dict with
        ``"mri_to_gen"`` and ``"gen_to_mri"`` keys.
    mri_encoder : nn.Module, optional
        MRI encoder used to extract features before passing to fusion.
    gen_encoder : nn.Module, optional
        Genetics encoder (wrapping genetics_encoder) used to extract features.
    device : torch.device, optional
    """

    def __init__(
        self,
        genetics_encoder: Optional[nn.Module] = None,
        fusion_module: Optional[nn.Module] = None,
        mri_encoder: Optional[nn.Module] = None,
        gen_encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.genetics_encoder = genetics_encoder
        self.fusion_module = fusion_module
        self.mri_encoder = mri_encoder
        self.gen_encoder = gen_encoder
        self.device = device or torch.device("cpu")

    # ------------------------------------------------------------------

    def get_genetics_attention(
        self,
        genetics: torch.Tensor,
    ) -> Optional[List[torch.Tensor]]:
        """
        Per-layer attention from the genetics transformer.

        Returns
        -------
        list of (B, n_heads, N+1, N+1) tensors, one per layer, or None if
        the encoder is not a transformer / does not expose attention.
        """
        enc = self.genetics_encoder
        if enc is None or not hasattr(enc, "get_attention_weights"):
            return None

        genetics = genetics.to(self.device)
        enc.eval()
        with torch.no_grad():
            return enc.get_attention_weights(genetics)

    def get_genetics_rollout(
        self,
        genetics: torch.Tensor,
        discard_ratio: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """
        Attention rollout through all genetics transformer layers.

        Returns
        -------
        rollout : (B, N+1, N+1) or None
        """
        attn_list = self.get_genetics_attention(genetics)
        if attn_list is None:
            return None
        return attention_rollout(attn_list, discard_ratio=discard_ratio)

    def get_gene_importance_from_attention(
        self,
        genetics: torch.Tensor,
        discard_ratio: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """
        Gene-level importance from CLS-token attention rollout.

        Returns
        -------
        importance : (B, n_genes) normalized float tensor, or None
        """
        rollout = self.get_genetics_rollout(genetics, discard_ratio)
        if rollout is None:
            return None
        return rollout_cls_to_tokens(rollout)

    def get_fusion_attention(
        self,
        mri_features: torch.Tensor,
        gen_features: torch.Tensor,
    ) -> Optional[Dict[str, List[torch.Tensor]]]:
        """
        Cross-modal attention from the fusion module.

        Parameters
        ----------
        mri_features : (B, mri_dim) pre-computed MRI encoder output
        gen_features : (B, gen_dim) pre-computed genetics encoder output

        Returns
        -------
        dict with keys ``"mri_to_gen"`` and ``"gen_to_mri"``, each a list of
        (B, n_heads, n_tokens, n_tokens) tensors, or None if not applicable.
        """
        fm = self.fusion_module
        if fm is None or not hasattr(fm, "get_attention_weights"):
            return None

        mri_features = mri_features.to(self.device)
        gen_features = gen_features.to(self.device)

        fm_inner = getattr(fm, "fusion", fm)
        if not hasattr(fm_inner, "get_attention_weights"):
            return None

        with torch.no_grad():
            return fm_inner.get_attention_weights(mri_features, gen_features)

    def get_head_importance(
        self,
        genetics: torch.Tensor,
        layer_idx: int = -1,
    ) -> Optional[torch.Tensor]:
        """
        Rank attention heads by their mean attention entropy at a given layer.

        Lower entropy → more focused attention → higher importance.

        Returns
        -------
        (n_heads,) tensor of importance scores (higher = more focused), or None
        """
        attn_list = self.get_genetics_attention(genetics)
        if attn_list is None:
            return None

        attn = attn_list[layer_idx]  # (B, n_heads, N, N)
        attn = attn + 1e-9
        attn = attn / attn.sum(dim=-1, keepdim=True)  # row-normalize
        entropy = -(attn * attn.log()).sum(dim=-1)     # (B, n_heads, N)
        mean_entropy = entropy.mean(dim=(0, 2))        # (n_heads,)
        importance = -mean_entropy                     # lower entropy → higher importance
        return importance.detach()
