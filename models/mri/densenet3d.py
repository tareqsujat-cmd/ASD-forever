"""
3D DenseNet for volumetric MRI feature extraction.

Why DenseNet for small medical datasets?
-----------------------------------------
DenseNet (Huang et al., 2017) connects each layer to every subsequent layer
within a dense block.  For ABIDE-scale data (~1000 subjects):
  - Dense connections act as strong regularization (each layer receives
    direct gradient signal from the loss via every subsequent layer)
  - Parameter efficiency: fewer parameters than ResNet for same accuracy
  - Feature reuse: low-level edge/texture features propagate to deep layers
    without degradation — important for subtle ASD structural differences

Published evidence: Heinsfeld et al. (2018) showed DenseNet outperforms
plain CNN on ABIDE functional connectivity.  Our structural MRI variant
extends this to volumetric T1w data.

Architecture
------------
DenseNet-121 configuration: growth_rate=32, block_config=(6,12,24,16)
  - Block 1: 6 dense layers
  - Block 2: 12 dense layers
  - Block 3: 24 dense layers (most ASD-relevant structural features)
  - Block 4: 16 dense layers

Reference
---------
Huang G, et al. (2017). Densely connected convolutional networks. CVPR.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _DenseLayer3D(nn.Module):
    """
    Single dense layer: BN → ReLU → 1×1 Conv → BN → ReLU → 3×3×3 Conv.
    Bottleneck design reduces computation before the 3D convolution.
    """

    def __init__(self, in_features: int, growth_rate: int,
                 bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        inter = bn_size * growth_rate
        self.norm1 = nn.BatchNorm3d(in_features)
        self.conv1 = nn.Conv3d(in_features, inter, 1, bias=False)
        self.norm2 = nn.BatchNorm3d(inter)
        self.conv2 = nn.Conv3d(inter, growth_rate, 3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        if self.drop_rate > 0 and self.training:
            out = F.dropout3d(out, p=self.drop_rate)
        return torch.cat([x, out], dim=1)  # Dense connection


class _DenseBlock3D(nn.Module):
    """Stack of dense layers within one dense block."""

    def __init__(self, num_layers: int, in_features: int,
                 growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                _DenseLayer3D(in_features + i * growth_rate,
                              growth_rate, bn_size, drop_rate)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def out_features(self, in_features: int, growth_rate: int) -> int:
        return in_features + len(self.layers) * growth_rate


class _Transition3D(nn.Sequential):
    """
    Transition layer between dense blocks: BN → 1×1 Conv → AvgPool.
    Reduces spatial dimensions and halves channel count (compression=0.5).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.BatchNorm3d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_features, out_features, 1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2),
        )


class DenseNet3D(nn.Module):
    """
    3D DenseNet for volumetric MRI.

    Parameters
    ----------
    growth_rate : int
        Number of new feature maps per dense layer.
    block_config : tuple of int
        Number of dense layers in each block. DenseNet-121: (6,12,24,16).
    num_init_features : int
        Feature maps after the initial convolution.
    bn_size : int
        Bottleneck size multiplier for dense layers.
    drop_rate : float
        Dropout within dense layers.
    in_channels : int
        Input channels (1 for single-channel MRI).
    feature_dim : int
        Output feature vector dimensionality.
    compression : float
        Channel compression ratio at transitions (0.5 = halve channels).
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, ...] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.2,
        in_channels: int = 1,
        feature_dim: int = 512,
        compression: float = 0.5,
    ) -> None:
        super().__init__()

        # Initial convolution + pooling (stride-2 twice → 4× spatial downsampling)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks + transitions
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        in_feat = num_init_features

        for i, n_layers in enumerate(block_config):
            block = _DenseBlock3D(n_layers, in_feat, growth_rate, bn_size, drop_rate)
            self.dense_blocks.append(block)
            in_feat = in_feat + n_layers * growth_rate

            if i < len(block_config) - 1:  # no transition after last block
                out_feat = int(in_feat * compression)
                self.transitions.append(_Transition3D(in_feat, out_feat))
                in_feat = out_feat

        # Final batch norm
        self.final_bn = nn.BatchNorm3d(in_feat)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Projection head
        self.projector = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_feat, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        self._backbone_out_features = in_feat
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return final feature map before global pooling (for GradCAM)."""
        x = self.stem(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        return F.relu(self.final_bn(x), inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, D, H, W)

        Returns
        -------
        torch.Tensor, shape (B, feature_dim)
        """
        feat_map = self.forward_features(x)
        pooled = self.global_pool(feat_map).flatten(1)
        return self.projector(pooled)


def densenet121_3d(feature_dim: int = 512, dropout: float = 0.2) -> DenseNet3D:
    """DenseNet-121 configuration."""
    return DenseNet3D(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=dropout,
        feature_dim=feature_dim,
    )

def densenet169_3d(feature_dim: int = 512, dropout: float = 0.2) -> DenseNet3D:
    """DenseNet-169: deeper, more capacity for larger datasets."""
    return DenseNet3D(
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        num_init_features=64,
        drop_rate=dropout,
        feature_dim=feature_dim,
    )
