"""
Connectome Transformer — a proper architecture for functional-connectivity input.

Replaces the "reshape the 19,900-d FC vector into a fake 28^3 volume and feed a
3D-CNN" workaround with a model designed for connectomes (BrainNetTF / METAFormer
lineage):

  FC matrix (B, R, R)  ── each ROI is a token whose features are its full
                          connectivity profile (row of the FC matrix)
      │  linear embed → (B, R, d_model)
      │  Transformer encoder (self-attention over ROI tokens)
      │  attention-pooled readout → (B, d_model)
      └  MLP head → logits

Self-attention over ROIs is interpretable (attention weights → salient regions),
which feeds the explainability analysis (publication experiment E6).

The model accepts either:
  - a full FC matrix ``(B, R, R)``, or
  - a flat upper-triangle vector ``(B, R*(R-1)/2)`` (auto-reconstructed to (B,R,R)).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _upper_to_matrix(vec: torch.Tensor, n_rois: int) -> torch.Tensor:
    """(B, R*(R-1)/2) upper-triangle -> symmetric (B, R, R) with unit diagonal."""
    B = vec.shape[0]
    iu = torch.triu_indices(n_rois, n_rois, offset=1, device=vec.device)
    mat = torch.zeros(B, n_rois, n_rois, dtype=vec.dtype, device=vec.device)
    mat[:, iu[0], iu[1]] = vec
    mat = mat + mat.transpose(1, 2)
    idx = torch.arange(n_rois, device=vec.device)
    mat[:, idx, idx] = 1.0
    return mat


class _AttentionReadout(nn.Module):
    """Learned attention pooling over the R ROI tokens → (B, d_model)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, R, d)
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)   # (B, R)
        pooled = torch.einsum("br,brd->bd", w, x)             # (B, d)
        return pooled, w


class ConnectomeTransformer(nn.Module):
    """
    Transformer over ROI connectivity-profile tokens.

    Parameters
    ----------
    n_rois : int          number of ROIs (200 for CC200)
    d_model : int         token embedding dim
    n_heads : int         attention heads
    n_layers : int        transformer encoder layers
    ffn_mult : int        feed-forward expansion
    dropout : float
    n_classes : int
    readout : str         "attention" | "mean" | "cls"
    """

    def __init__(
        self,
        n_rois: int = 200,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_mult: int = 4,
        dropout: float = 0.3,
        n_classes: int = 2,
        readout: str = "attention",
    ) -> None:
        super().__init__()
        self.n_rois = n_rois
        self.readout_kind = readout

        self.embed = nn.Linear(n_rois, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_rois, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        if readout == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        if readout == "attention":
            self.readout = _AttentionReadout(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        self._last_attn: Optional[torch.Tensor] = None

    def forward(self, fc: torch.Tensor) -> torch.Tensor:
        """fc: (B, R, R) matrix or (B, R*(R-1)/2) upper-triangle vector."""
        if fc.dim() == 2:
            fc = _upper_to_matrix(fc, self.n_rois)
        elif fc.dim() == 4:                       # (B,1,R,R) — squeeze channel
            fc = fc.squeeze(1)

        x = self.embed(fc) + self.pos             # (B, R, d)
        if self.readout_kind == "cls":
            cls = self.cls.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = self.encoder(x)
        x = self.norm(x)

        if self.readout_kind == "attention":
            pooled, attn = self.readout(x)
            self._last_attn = attn.detach()
        elif self.readout_kind == "cls":
            pooled = x[:, 0]
        else:                                     # mean
            pooled = x.mean(dim=1)

        return self.head(pooled)

    def last_attention(self) -> Optional[torch.Tensor]:
        """Return the last forward pass's ROI attention weights (B, R) for XAI."""
        return self._last_attn


class MaskedConnectomeAutoencoder(nn.Module):
    """
    Self-supervised pretext model (publication experiment E2.2).

    Randomly masks a fraction of ROI tokens and reconstructs their connectivity
    profiles from context, so the encoder learns connectome structure from
    *unlabeled* scans (leakage-safe).  After pretraining, transfer ``encoder``
    (embed + transformer) weights into a ConnectomeTransformer classifier.
    """

    def __init__(self, n_rois: int = 200, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1, mask_ratio: float = 0.25) -> None:
        super().__init__()
        self.n_rois = n_rois
        self.mask_ratio = mask_ratio
        self.embed = nn.Linear(n_rois, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_rois, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.decoder = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_rois))

    def forward(self, fc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if fc.dim() == 2:
            fc = _upper_to_matrix(fc, self.n_rois)
        B, R, _ = fc.shape
        tokens = self.embed(fc) + self.pos
        n_mask = max(1, int(self.mask_ratio * R))
        # per-sample random mask
        noise = torch.rand(B, R, device=fc.device)
        mask_idx = noise.argsort(dim=1)[:, :n_mask]           # (B, n_mask)
        mask = torch.zeros(B, R, dtype=torch.bool, device=fc.device)
        mask.scatter_(1, mask_idx, True)
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        h = self.encoder(tokens)
        recon = self.decoder(h)                               # (B, R, R)
        loss = ((recon - fc) ** 2)[mask].mean()               # reconstruct masked rows
        return recon, loss


def build_connectome_transformer(cfg=None, **kw) -> ConnectomeTransformer:
    """Factory: build from a config object's fields or explicit kwargs."""
    if cfg is not None:
        mc = getattr(cfg, "connectome_transformer", None)
        if mc is not None:
            kw.setdefault("d_model", getattr(mc, "d_model", 128))
            kw.setdefault("n_heads", getattr(mc, "n_heads", 4))
            kw.setdefault("n_layers", getattr(mc, "n_layers", 2))
            kw.setdefault("dropout", getattr(mc, "dropout", 0.3))
    return ConnectomeTransformer(**kw)
