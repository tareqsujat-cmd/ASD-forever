"""
3D ConvNeXt for volumetric MRI.

ConvNeXt (Liu et al., 2022) "modernizes" ResNet by incorporating design
principles from Vision Transformers while keeping pure convolutions:
  - Depthwise 7×7×7 convolution (large receptive field, few parameters)
  - Inverted bottleneck (expand then compress channels)
  - LayerNorm instead of BatchNorm (more stable for small-batch 3D training)
  - GELU activation
  - Fewer activation functions and normalization layers than ResNet

For ASD / small-dataset regime, ConvNeXt's LayerNorm is especially valuable:
  - BatchNorm requires large batches for stable statistics
  - 3D MRI forces small batches (batch_size=4-8) due to memory
  - LayerNorm is batch-size independent → stable training at batch_size=4

Reference
---------
Liu Z, et al. (2022). A ConvNet for the 2020s. CVPR 2022. arXiv:2201.03545
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConvNeXtBlock3D(nn.Module):
    """
    ConvNeXt block in 3D.

    Structure:
      depthwise 7×7×7 conv → LayerNorm → pointwise 1→4× expand → GELU
      → pointwise 4→1× compress → LayerScale → StochasticDepth residual

    LayerScale (learnable per-channel scaling) prevents instability in
    deep networks when using small random initializations.
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()

        # Large depthwise conv: captures local 3D structure efficiently
        self.dwconv = nn.Conv3d(
            dim, dim, kernel_size=7, padding=3, groups=dim, bias=True
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # LayerScale: learned per-channel scaling initialized near 0
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim), requires_grad=True
        )

        from models.mri.swin3d import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W)"""
        identity = x
        # Depthwise conv operates on (B, C, D, H, W)
        out = self.dwconv(x)
        # LayerNorm and pointwise ops need channels last
        out = out.permute(0, 2, 3, 4, 1).contiguous()  # (B, D, H, W, C)
        out = self.norm(out)
        out = self.pwconv2(self.act(self.pwconv1(out)))
        out = self.gamma * out
        out = out.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, D, H, W)
        return identity + self.drop_path(out)


class ConvNeXt3D(nn.Module):
    """
    3D ConvNeXt backbone.

    Parameters
    ----------
    in_channels : int
        Input image channels.
    depths : list of int
        Number of ConvNeXt blocks per stage.
    dims : list of int
        Channel dimensions per stage.
    drop_path_rate : float
        Maximum stochastic depth rate.
    feature_dim : int
        Output projection dimension.
    dropout : float
    """

    def __init__(
        self,
        in_channels: int = 1,
        depths: List[int] = (3, 3, 9, 3),
        dims: List[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.2,
        feature_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Stem: aggressive 4×4×4 downsampling to reduce spatial resolution quickly
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=4, stride=4, bias=True),
            nn.LayerNorm(dims[0], elementwise_affine=True),  # channel-last
        )

        # Downsampling layers between stages (2× spatial reduction)
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.LayerNorm(dims[i], elementwise_affine=True),  # channel-last
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=True),
            ))

        # Stochastic depth schedule
        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # ConvNeXt stages
        self.stages = nn.ModuleList()
        block_idx = 0
        for i, (n_blocks, dim) in enumerate(zip(depths, dims)):
            stage = nn.Sequential(*[
                ConvNeXtBlock3D(dim, drop_path=dpr[block_idx + j])
                for j in range(n_blocks)
            ])
            self.stages.append(stage)
            block_idx += n_blocks

        self.final_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dims[-1], feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _stem_norm_channels_last(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stem with channels-last LayerNorm."""
        # stem conv: (B, C_in, D, H, W) -> (B, C_out, D', H', W')
        x = self.stem[0](x)
        # LayerNorm on channels-last
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.stem[1](x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature map before final pooling (for GradCAM)."""
        x = self._stem_norm_channels_last(x)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < 3:
                # Downsample: channels-last norm then conv
                x = x.permute(0, 2, 3, 4, 1).contiguous()
                x = self.downsample_layers[i][0](x)
                x = x.permute(0, 4, 1, 2, 3).contiguous()
                x = self.downsample_layers[i][1](x)

        # Final norm (channels last)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.final_norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

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
        pooled = self.global_pool(feat).flatten(1)
        return self.projector(pooled)


def convnext3d_tiny(feature_dim: int = 512, dropout: float = 0.3) -> ConvNeXt3D:
    """ConvNeXt-Tiny-3D: 28M parameters."""
    return ConvNeXt3D(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
        feature_dim=feature_dim, dropout=dropout,
    )

def convnext3d_small(feature_dim: int = 512, dropout: float = 0.3) -> ConvNeXt3D:
    """ConvNeXt-Small-3D: 50M parameters."""
    return ConvNeXt3D(
        depths=[3, 3, 27, 3], dims=[96, 192, 384, 768],
        feature_dim=feature_dim, dropout=dropout,
    )
