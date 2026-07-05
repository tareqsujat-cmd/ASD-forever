"""
Swin Transformer 3D — Primary MRI backbone.

Why Swin3D is the publication-worthy choice for ASD
------------------------------------------------------
1. Long-range dependencies: Standard CNNs have limited receptive fields.
   ASD structural differences are bilateral and span hemispheres.
   Swin3D's self-attention captures these global correlations natively.

2. Hierarchical representations: Brain structure is inherently hierarchical
   (voxels → cortical columns → regions → networks). Swin3D's patch merging
   mirrors this hierarchy through its 4 stages.

3. Shifted windows eliminate the quadratic attention cost of vanilla ViT,
   making 3D volumetric attention computationally feasible.

4. Publication gap: No published ASD paper uses Swin3D as of 2024.
   This is our primary novelty claim on the imaging side.

Architecture (Swin3D-Tiny)
--------------------------
Input: (B, 1, 96, 96, 96)
Patch partition: 4×4×4 patches → (B, 24³, 96) token sequence
Stage 1: 2 SwinTransformer3D blocks, 96 channels, window=(4,4,4)
Stage 2: 2 blocks, 192 channels
Stage 3: 6 blocks, 384 channels
Stage 4: 2 blocks, 768 channels
Global pool → 768 → feature_dim projection

Reference
---------
Liu Z, et al. (2022). Video Swin Transformer. CVPR 2022.
Liu Z, et al. (2021). Swin Transformer: Hierarchical Vision Transformer
using Shifted Windows. ICCV 2021.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


def window_partition_3d(
    x: torch.Tensor, window_size: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Partition 3D feature maps into non-overlapping windows.

    Parameters
    ----------
    x : torch.Tensor, shape (B, D, H, W, C)
    window_size : (wd, wh, ww)

    Returns
    -------
    torch.Tensor, shape (num_windows*B, wd*wh*ww, C)
    """
    B, D, H, W, C = x.shape
    wd, wh, ww = window_size
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return windows.view(-1, wd * wh * ww, C)


def window_reverse_3d(
    windows: torch.Tensor,
    window_size: Tuple[int, int, int],
    D: int, H: int, W: int,
) -> torch.Tensor:
    """Reverse window partition back to full feature map."""
    wd, wh, ww = window_size
    B = int(windows.shape[0] / (D * H * W / wd / wh / ww))
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)


class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-head Self-Attention with relative position bias.

    Relative position bias encodes the 3D positional relationship between
    tokens within the same window.  Absolute positional encodings are
    problematic for medical images because the same anatomical structure
    can appear at slightly different positions after registration.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    window_size : tuple of int
        (Wd, Wh, Ww) — spatial size of each attention window.
    num_heads : int
        Number of attention heads.
    qkv_bias : bool
    attn_drop : float
    proj_drop : float
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table: (2Wd-1) × (2Wh-1) × (2Ww-1) entries
        wd, wh, ww = window_size
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # Precompute relative position index
        self.register_buffer("rel_pos_index", self._compute_rel_pos_index(window_size))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _compute_rel_pos_index(
        window_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """Precompute relative position index for all token pairs in window."""
        wd, wh, ww = window_size
        coords_d = torch.arange(wd)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        # shape (3, Wd, Wh, Ww)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w,
                                             indexing="ij"))
        coords_flat = coords.flatten(1)  # (3, Wd*Wh*Ww)
        # Pairwise relative positions
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, N, N)
        rel = rel.permute(1, 2, 0).contiguous()  # (N, N, 3)
        # Shift to start from 0
        rel[:, :, 0] += wd - 1
        rel[:, :, 1] += wh - 1
        rel[:, :, 2] += ww - 1
        rel[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        rel[:, :, 1] *= (2 * ww - 1)
        return rel.sum(-1)  # (N, N)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (num_windows*B, N, C)
            N = window tokens = Wd*Wh*Ww
        mask : torch.Tensor, optional
            Attention mask for shifted windows.
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)]
        bias = bias.view(*self.rel_pos_index.shape, self.num_heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock3D(nn.Module):
    """
    One Swin Transformer block: W-MSA or SW-MSA + MLP.

    Alternating between regular window attention (W-MSA) and shifted-window
    attention (SW-MSA) enables cross-window information exchange without
    the quadratic cost of global attention.

    Parameters
    ----------
    dim : int
    num_heads : int
    window_size : tuple of int
    shift_size : tuple of int
        (0,0,0) for W-MSA, (wd//2, wh//2, ww//2) for SW-MSA.
    mlp_ratio : float
    drop_path : float
        Stochastic depth rate (regularization for deep networks).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int],
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size, num_heads,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(proj_drop),
        )

        # Stochastic depth (drop entire residual path)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _compute_attn_mask(
        self, D: int, H: int, W: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Compute attention mask for shifted window attention."""
        if any(s > 0 for s in self.shift_size):
            img_mask = torch.zeros(1, D, H, W, 1, device=device)
            wd, wh, ww = self.window_size
            sd, sh, sw = self.shift_size
            slices_d = [slice(0, -wd), slice(-wd, -sd), slice(-sd, None)]
            slices_h = [slice(0, -wh), slice(-wh, -sh), slice(-sh, None)]
            slices_w = [slice(0, -ww), slice(-ww, -sw), slice(-sw, None)]
            cnt = 0
            for d in slices_d:
                for h in slices_h:
                    for w in slices_w:
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1

            mask_windows = window_partition_3d(img_mask, self.window_size)
            mask_windows = mask_windows.squeeze(-1)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
            return attn_mask
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, H, W, C)"""
        B, D, H, W, C = x.shape

        # Pad to be divisible by window size
        wd, wh, ww = self.window_size
        pad_d = (wd - D % wd) % wd
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        x_pad = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dp, Hp, Wp, _ = x_pad.shape

        # Cyclic shift for SW-MSA
        if any(s > 0 for s in self.shift_size):
            sd, sh, sw = self.shift_size
            x_shifted = torch.roll(x_pad, shifts=(-sd, -sh, -sw), dims=(1, 2, 3))
        else:
            x_shifted = x_pad

        attn_mask = self._compute_attn_mask(Dp, Hp, Wp, x.device)

        # Partition into windows
        x_windows = window_partition_3d(x_shifted, self.window_size)  # (nW*B, N, C)

        # Self-attention
        attn_out = self.attn(x_windows, mask=attn_mask)

        # Reverse windows
        attn_out = window_reverse_3d(attn_out, self.window_size, Dp, Hp, Wp)

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            sd, sh, sw = self.shift_size
            attn_out = torch.roll(attn_out, shifts=(sd, sh, sw), dims=(1, 2, 3))

        # Remove padding
        if pad_d or pad_h or pad_w:
            attn_out = attn_out[:, :D, :H, :W, :].contiguous()

        # Residual + MLP
        x = x + self.drop_path(self.norm1(attn_out))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    """
    Patch merging layer: 2× spatial downsampling + channel doubling.
    Merges 2×2×2 neighboring patches and projects to 2C channels.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, H, W, C)"""
        B, D, H, W, C = x.shape
        # Pad odd dimensions
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        return self.reduction(self.norm(x))


class DropPath(nn.Module):
    """Stochastic depth — drops entire residual paths during training."""

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        rand = torch.floor(rand + keep)
        return x * rand / keep


class SwinTransformer3D(nn.Module):
    """
    3D Swin Transformer for volumetric MRI.

    Parameters
    ----------
    in_channels : int
        Input image channels (1 for single-channel MRI).
    patch_size : tuple of int
        Initial patch partition size. (4,4,4) → 4× downsampling.
    embed_dim : int
        Embedding dimension after patch embedding.
    depths : list of int
        Number of Swin blocks per stage.
    num_heads : list of int
        Attention heads per stage (must match len(depths)).
    window_size : tuple of int
        Attention window size. (4,4,4) for 96³ input at stage 1.
    mlp_ratio : float
    drop_path_rate : float
        Maximum stochastic depth rate (linearly scaled per block).
    feature_dim : int
        Final output feature vector dimensionality.
    dropout : float
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        embed_dim: int = 96,
        depths: List[int] = (2, 2, 6, 2),
        num_heads: List[int] = (3, 6, 12, 24),
        window_size: Tuple[int, int, int] = (4, 4, 4),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        feature_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.depths = depths
        self.num_stages = len(depths)

        # Patch embedding: partition into tokens
        pd, ph, pw = patch_size
        self.patch_embed = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size,
                      stride=patch_size, bias=False),
            # Rearrange to (B, D, H, W, C) in forward
        )
        self.patch_norm = nn.LayerNorm(embed_dim)

        # Stochastic depth rate schedule (linearly increasing)
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Build stages
        self.stages = nn.ModuleList()
        self.patch_merging = nn.ModuleList()
        block_idx = 0

        for stage_i in range(self.num_stages):
            stage_dim = embed_dim * (2 ** stage_i)
            stage_heads = num_heads[stage_i]
            stage_blocks = nn.ModuleList()

            for j in range(depths[stage_i]):
                # Alternate between W-MSA and SW-MSA
                shift = tuple(s // 2 for s in window_size) \
                    if j % 2 == 1 else (0, 0, 0)
                stage_blocks.append(
                    SwinTransformerBlock3D(
                        dim=stage_dim,
                        num_heads=stage_heads,
                        window_size=window_size,
                        shift_size=shift,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[block_idx],
                        proj_drop=dropout,
                    )
                )
                block_idx += 1

            self.stages.append(stage_blocks)

            if stage_i < self.num_stages - 1:
                self.patch_merging.append(PatchMerging3D(stage_dim))

        final_dim = embed_dim * (2 ** (self.num_stages - 1))
        self.final_norm = nn.LayerNorm(final_dim)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, D', H', W', C) feature tensor for attention visualization."""
        # Patch embedding
        x = self.patch_embed(x)  # (B, C, D/p, H/p, W/p)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        x = self.patch_norm(x)

        for stage_i, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)
            if stage_i < self.num_stages - 1:
                x = self.patch_merging[stage_i](x)

        return self.final_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, D, H, W)

        Returns
        -------
        torch.Tensor, shape (B, feature_dim)
        """
        feat = self.forward_features(x)
        # Global spatial average pooling over (D, H, W) dimensions
        feat = feat.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)
        pooled = self.global_pool(feat).flatten(1)         # (B, C)
        return self.projector(pooled)


def swin3d_tiny(feature_dim: int = 512, dropout: float = 0.3) -> SwinTransformer3D:
    """Swin3D-Tiny: 28M parameters, best speed/accuracy for ABIDE scale."""
    return SwinTransformer3D(
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=(4, 4, 4), feature_dim=feature_dim, dropout=dropout,
    )

def swin3d_small(feature_dim: int = 512, dropout: float = 0.3) -> SwinTransformer3D:
    """Swin3D-Small: 50M parameters."""
    return SwinTransformer3D(
        embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
        window_size=(4, 4, 4), feature_dim=feature_dim, dropout=dropout,
    )
