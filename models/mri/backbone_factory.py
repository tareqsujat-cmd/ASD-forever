"""
Backbone factory: constructs any MRI backbone from a config string.

This is the single import point for all downstream modules.
Adding a new backbone requires only:
  1. Implementing the backbone class
  2. Adding one entry to _BACKBONE_REGISTRY

Usage
-----
    from models.mri.backbone_factory import build_mri_encoder
    encoder = build_mri_encoder(cfg)
    # encoder is always an MRIEncoder regardless of backbone choice
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def build_mri_encoder(cfg, weights_path: Optional[str] = None):
    """
    Build an MRIEncoder from configuration.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object.
    weights_path : str, optional
        Override path for pretrained weights (used by ablation runner).

    Returns
    -------
    MRIEncoder
    """
    from models.mri.mri_encoder import MRIEncoder

    mri_cfg = cfg.mri_model
    backbone = _build_backbone(mri_cfg, weights_path)

    encoder = MRIEncoder(
        backbone=backbone,
        feature_dim=mri_cfg.feature_dim,
        use_se=True,
        freeze_backbone=mri_cfg.freeze_backbone,
        dropout=mri_cfg.dropout,
    )

    from utilities.hardware import count_parameters
    count_parameters(encoder)
    return encoder


def _build_backbone(mri_cfg, weights_path: Optional[str]) -> nn.Module:
    """Construct the backbone neural network from config."""
    name = mri_cfg.backbone.lower()

    if name in ("resnet10_3d", "resnet10"):
        from models.mri.resnet3d import resnet10_3d
        return resnet10_3d(feature_dim=mri_cfg.feature_dim, dropout=mri_cfg.dropout)

    elif name in ("resnet18_3d", "resnet18"):
        from models.mri.resnet3d import resnet18_3d
        return resnet18_3d(feature_dim=mri_cfg.feature_dim, dropout=mri_cfg.dropout)

    elif name in ("resnet34_3d", "resnet34"):
        from models.mri.resnet3d import resnet34_3d
        return resnet34_3d(feature_dim=mri_cfg.feature_dim, dropout=mri_cfg.dropout)

    elif name in ("resnet50_3d", "resnet50", "medicalnet_resnet50"):
        from models.mri.resnet3d import resnet50_3d, load_medicalnet_weights
        backbone = resnet50_3d(
            feature_dim=mri_cfg.feature_dim, dropout=mri_cfg.dropout
        )
        if mri_cfg.pretrained:
            backbone = load_medicalnet_weights(backbone, weights_path)
        return backbone

    elif name in ("densenet121_3d", "densenet121"):
        from models.mri.densenet3d import densenet121_3d
        return densenet121_3d(feature_dim=mri_cfg.feature_dim, dropout=mri_cfg.dropout)

    elif name in ("densenet169_3d", "densenet169"):
        from models.mri.densenet3d import densenet169_3d
        return densenet169_3d(feature_dim=mri_cfg.feature_dim, dropout=mri_cfg.dropout)

    elif name in ("swin3d_tiny", "swin3d", "swin_tiny"):
        from models.mri.swin3d import swin3d_tiny
        return swin3d_tiny(feature_dim=mri_cfg.feature_dim, dropout=mri_cfg.dropout)

    elif name in ("swin3d_small", "swin_small"):
        from models.mri.swin3d import swin3d_small
        return swin3d_small(feature_dim=mri_cfg.feature_dim, dropout=mri_cfg.dropout)

    elif name in ("convnext3d_tiny", "convnext3d", "convnext_tiny"):
        from models.mri.convnext3d import convnext3d_tiny
        return convnext3d_tiny(feature_dim=mri_cfg.feature_dim, dropout=mri_cfg.dropout)

    elif name in ("convnext3d_small", "convnext_small"):
        from models.mri.convnext3d import convnext3d_small
        return convnext3d_small(feature_dim=mri_cfg.feature_dim, dropout=mri_cfg.dropout)

    else:
        raise ValueError(
            f"Unknown MRI backbone: '{mri_cfg.backbone}'. "
            f"Available: resnet10/18/34/50_3d, medicalnet_resnet50, "
            f"densenet121/169_3d, swin3d_tiny/small, convnext3d_tiny/small"
        )
