"""
3D ResNet with MedicalNet pretrained weight support.

MedicalNet (Chen et al., 2019) pretrains ResNet variants on 8 medical
segmentation datasets (>1.4M slices), giving a vastly better initialization
than ImageNet weights for 3D medical imaging tasks.

For ASD detection on ABIDE (~1000 subjects), pretraining is critical:
  - Random init 3D ResNet on ABIDE alone → high overfitting risk
  - MedicalNet init → learned low-level medical texture features,
    faster convergence, better generalization across sites

Reference
---------
Chen S, et al. (2019). Med3D: Transfer learning for 3D medical image analysis.
arXiv:1904.00625. https://github.com/Tencent/MedicalNet

Architecture
------------
ResNet variants available: ResNet-10, 18, 34, 50 (increasing capacity).
ResNet-50 is the default: best accuracy / compute trade-off for our task.

The final global average pooling + projection head produces a fixed-length
feature vector used by the fusion module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Conv3dBnRelu(nn.Sequential):
    """3D Conv → BatchNorm → ReLU block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


class BasicBlock3D(nn.Module):
    """
    ResNet basic block for ResNet-10/18/34.
    Two 3×3×3 convolutions with residual connection.
    """
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck3D(nn.Module):
    """
    ResNet bottleneck block for ResNet-50/101/152.
    1×1×1 → 3×3×3 → 1×1×1 with 4× channel expansion.

    The bottleneck design reduces parameters while maintaining receptive field,
    critical for 3D volumes where parameter count grows cubically.
    """
    expansion = 4

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        width = out_ch
        self.conv1 = nn.Conv3d(in_ch, width, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = nn.Conv3d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = nn.Conv3d(width, out_ch * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_ch * self.expansion)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# ResNet3D
# ---------------------------------------------------------------------------

class ResNet3D(nn.Module):
    """
    3D ResNet backbone for volumetric MRI feature extraction.

    Parameters
    ----------
    block : type
        BasicBlock3D or Bottleneck3D.
    layers : list of int
        Number of blocks per stage, e.g. [3, 4, 6, 3] for ResNet-50.
    in_channels : int
        Input channels (1 for single-channel MRI).
    feature_dim : int
        Output feature vector dimensionality after global pooling + projection.
    dropout : float
        Dropout rate in the projection head.
    """

    def __init__(
        self,
        block,
        layers: List[int],
        in_channels: int = 1,
        feature_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.in_ch = 64

        # Stem: large 7×7×7 conv to handle initial 3D structure
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Global spatial aggregation
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Projection head: backbone_out → feature_dim
        backbone_out = 512 * block.expansion
        self.projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_out, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _make_layer(self, block, planes: int, n_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        out_ch = planes * block.expansion
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
        layers = [block(self.in_ch, planes, stride, downsample)]
        self.in_ch = out_ch
        for _ in range(1, n_blocks):
            layers.append(block(self.in_ch, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature maps before global pooling (for GradCAM)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
        feat_map = self.forward_features(x)
        pooled = self.global_pool(feat_map).flatten(1)
        return self.projector(pooled)


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------

def resnet10_3d(feature_dim: int = 512, dropout: float = 0.3, **kwargs) -> ResNet3D:
    return ResNet3D(BasicBlock3D, [1, 1, 1, 1], feature_dim=feature_dim,
                   dropout=dropout, **kwargs)

def resnet18_3d(feature_dim: int = 512, dropout: float = 0.3, **kwargs) -> ResNet3D:
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], feature_dim=feature_dim,
                   dropout=dropout, **kwargs)

def resnet34_3d(feature_dim: int = 512, dropout: float = 0.3, **kwargs) -> ResNet3D:
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], feature_dim=feature_dim,
                   dropout=dropout, **kwargs)

def resnet50_3d(feature_dim: int = 512, dropout: float = 0.3, **kwargs) -> ResNet3D:
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], feature_dim=feature_dim,
                   dropout=dropout, **kwargs)


def load_medicalnet_weights(
    model: ResNet3D,
    weights_path: Optional[str],
    strict: bool = False,
) -> ResNet3D:
    """
    Load MedicalNet pretrained weights into a ResNet3D model.

    MedicalNet weights are available at:
    https://github.com/Tencent/MedicalNet (requires manual download)

    We use strict=False because MedicalNet's final FC layer differs from
    our projection head — we only transfer the backbone weights.

    Parameters
    ----------
    model : ResNet3D
    weights_path : str or None
        Path to MedicalNet .pth weight file.
        If None, falls back to random initialization with a warning.
    strict : bool
        If False, only load keys that match (skip mismatched final layers).

    Returns
    -------
    ResNet3D with pretrained weights.
    """
    if weights_path is None or not Path(weights_path).exists():
        logger.warning(
            "MedicalNet weights not found. Using random initialization. "
            "Download from: https://github.com/Tencent/MedicalNet\n"
            "Expected path: " + str(weights_path)
        )
        return model

    checkpoint = torch.load(weights_path, map_location="cpu")

    # MedicalNet checkpoints may be wrapped in 'state_dict' or 'net'
    state_dict = checkpoint
    for key in ("state_dict", "net", "model"):
        if key in checkpoint:
            state_dict = checkpoint[key]
            break

    # Strip 'module.' prefix from DataParallel training
    state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    logger.info(
        f"MedicalNet weights loaded from {weights_path}\n"
        f"  Missing keys:    {len(missing)}\n"
        f"  Unexpected keys: {len(unexpected)}"
    )
    return model
