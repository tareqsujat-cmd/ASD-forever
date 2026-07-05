"""
Unified MRI Encoder wrapper.

This class wraps any backbone (ResNet3D, DenseNet3D, Swin3D, ConvNeXt3D)
behind a common interface used by all downstream components:
  - Fusion module (takes `features` vector)
  - Explainability module (takes `feature_maps` tensor for GradCAM)
  - Ablation runner (swaps backbone via config string)

Design principle: downstream modules should never import backbone classes
directly. They import MRIEncoder and receive the same interface regardless
of which backbone is active. This is critical for the ablation study.

Additional components:
  - SE (Squeeze-and-Excitation) attention recalibration on feature maps
  - Multi-scale feature aggregation (for Swin3D: concatenate stage outputs)
  - Classification head for MRI-only baseline
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation channel attention for 3D feature maps.

    SE recalibrates channel-wise feature responses by modeling inter-channel
    dependencies.  For brain MRI: selectively emphasizes anatomically
    relevant channels (e.g., cortical thickness channels) over noise channels.

    Reference: Hu et al. (2018). Squeeze-and-Excitation Networks. CVPR.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W)"""
        scale = self.se(x).view(x.shape[0], x.shape[1], 1, 1, 1)
        return x * scale


class MRIEncoder(nn.Module):
    """
    Unified MRI encoder: backbone + SE attention + projection.

    Parameters
    ----------
    backbone : nn.Module
        Any backbone with a .forward(x) → (B, feature_dim) interface
        and optionally a .forward_features(x) → (B, C, D, H, W) method.
    feature_dim : int
        Expected output dimensionality from backbone.
    use_se : bool
        Apply SE channel attention on backbone feature maps.
    freeze_backbone : bool
        If True, backbone weights are frozen (useful for linear probing).
    dropout : float
        Additional dropout on the final feature vector.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int = 512,
        use_se: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.use_se = use_se

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Backbone weights frozen (linear probe mode)")

        # SE block on backbone feature maps (if backbone exposes them)
        self._has_feature_maps = hasattr(backbone, "forward_features")
        if use_se and self._has_feature_maps:
            # Detect feature map channel count by a dry run
            self._se_channels = self._detect_feature_map_channels()
            if self._se_channels is not None:
                self.se = SEBlock3D(self._se_channels)
                logger.info(f"SE attention: {self._se_channels} channels")
            else:
                self.use_se = False
        else:
            self.use_se = False

        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(feature_dim)

    def _detect_feature_map_channels(self) -> Optional[int]:
        """Detect channel count of backbone feature maps via a tiny dry run."""
        try:
            dummy = torch.zeros(1, 1, 32, 32, 32)
            with torch.no_grad():
                feat = self.backbone.forward_features(dummy)
            if feat.dim() == 5:  # (B, C, D, H, W)
                return feat.shape[1]
            elif feat.dim() == 5 and feat.shape[-1] > 1:  # channels-last (Swin)
                return feat.shape[-1]
        except Exception:
            pass
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature vector from a 3D MRI volume.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, D, H, W)

        Returns
        -------
        torch.Tensor, shape (B, feature_dim)
        """
        if self.use_se and self._has_feature_maps:
            feat_maps = self.backbone.forward_features(x)

            # Handle channels-last (Swin) vs channels-first (ResNet/DenseNet)
            if feat_maps.dim() == 5 and feat_maps.shape[-1] != feat_maps.shape[1]:
                # Likely channels-last: (B, D, H, W, C)
                feat_maps = feat_maps.permute(0, 4, 1, 2, 3).contiguous()

            feat_maps = self.se(feat_maps)

            # Now we need to pass these through the backbone's projection head
            # Most backbones have a global_pool + projector after forward_features
            if hasattr(self.backbone, "global_pool") and \
               hasattr(self.backbone, "projector"):
                pooled = self.backbone.global_pool(feat_maps).flatten(1)
                features = self.backbone.projector(pooled)
            else:
                pooled = F.adaptive_avg_pool3d(feat_maps, 1).flatten(1)
                features = pooled
        else:
            features = self.backbone(x)

        features = self.output_norm(self.output_dropout(features))
        return features

    def get_feature_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return intermediate feature maps for GradCAM / attention visualization.

        Returns None if the backbone does not expose feature maps.
        """
        if not self._has_feature_maps:
            return None
        with torch.enable_grad():
            feat = self.backbone.forward_features(x)
            if feat.dim() == 5 and feat.shape[-1] != feat.shape[1]:
                feat = feat.permute(0, 4, 1, 2, 3).contiguous()
            return feat

    def forward_with_maps(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (features, feature_maps) simultaneously."""
        return self.forward(x), self.get_feature_maps(x)


class MRIClassifier(nn.Module):
    """
    MRI-only classifier (single-modality baseline).

    Used in:
    1. Ablation study (MRI-only vs fusion)
    2. Pretraining the MRI encoder before fusion fine-tuning
    3. Establishing the publication baseline

    Parameters
    ----------
    encoder : MRIEncoder
    num_classes : int
    dropout : float
    """

    def __init__(
        self,
        encoder: MRIEncoder,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.feature_dim, encoder.feature_dim // 2),
            nn.LayerNorm(encoder.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(encoder.feature_dim // 2, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, D, H, W)

        Returns
        -------
        dict with keys:
            "logits"   : (B, num_classes)
            "features" : (B, feature_dim)  — for fusion module
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return {"logits": logits, "features": features}
