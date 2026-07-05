"""
Fusion module factory.

Usage
-----
    from models.fusion.fusion_factory import build_fusion_module
    fusion = build_fusion_module(cfg)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_fusion_module(cfg) -> "MultiModalFusion":
    """
    Build a MultiModalFusion from configuration.

    Parameters
    ----------
    cfg : Config
        Loaded configuration (config_schema.py).

    Returns
    -------
    MultiModalFusion
    """
    from models.fusion.fusion_module import MultiModalFusion

    fc = cfg.fusion
    method = fc.method.lower()
    backend, fusion_dim = _build_backend(method, fc)

    module = MultiModalFusion(
        backend=backend,
        fusion_dim=fusion_dim,
        num_classes=fc.num_classes,
        method=method,
    )

    from utilities.hardware import count_parameters
    total, trainable = count_parameters(module)
    logger.info(
        "MultiModalFusion [%s]: %s total params, %s trainable",
        method, f"{total:,}", f"{trainable:,}",
    )
    return module


def _build_backend(method: str, fc):
    """Returns (backend_module, fusion_dim)."""

    if method == "cross_attention":
        from models.fusion.cross_attention import CrossAttentionFusion
        ca = fc.cross_attention
        backend = CrossAttentionFusion(
            mri_dim=fc.mri_feature_dim,
            gen_dim=fc.genetics_feature_dim,
            fusion_dim=fc.fusion_dim,
            n_heads=fc.num_heads,
            n_layers=ca.num_layers,
            n_tokens=4,
            ffn_dim=ca.ffn_dim,
            num_classes=fc.num_classes,
            dropout=fc.dropout,
        )
        return backend, fc.fusion_dim

    elif method == "gated":
        from models.fusion.gated_fusion import GatedFusion
        gate_act = fc.gated.gate_activation
        backend = GatedFusion(
            mri_dim=fc.mri_feature_dim,
            gen_dim=fc.genetics_feature_dim,
            fusion_dim=fc.fusion_dim,
            num_classes=fc.num_classes,
            dropout=fc.dropout,
            gate_activation=gate_act,
        )
        return backend, fc.fusion_dim

    elif method in ("intermediate", "concat"):
        from models.fusion.gated_fusion import IntermediateFusion
        backend = IntermediateFusion(
            mri_dim=fc.mri_feature_dim,
            gen_dim=fc.genetics_feature_dim,
            fusion_dim=fc.fusion_dim,
            num_classes=fc.num_classes,
            dropout=fc.dropout,
        )
        return backend, fc.fusion_dim

    elif method == "late":
        from models.fusion.gated_fusion import LateFusion
        backend = LateFusion(
            mri_dim=fc.mri_feature_dim,
            gen_dim=fc.genetics_feature_dim,
            fusion_dim=fc.fusion_dim,
            num_classes=fc.num_classes,
            dropout=fc.dropout,
        )
        # Late fusion exports 2*num_classes as fused_features
        return backend, 2 * fc.num_classes

    elif method == "dynamic":
        from models.fusion.gated_fusion import DynamicFusion
        backend = DynamicFusion(
            mri_dim=fc.mri_feature_dim,
            gen_dim=fc.genetics_feature_dim,
            fusion_dim=fc.fusion_dim,
            num_classes=fc.num_classes,
            dropout=fc.dropout,
        )
        return backend, fc.fusion_dim

    else:
        raise ValueError(
            f"Unknown fusion method: '{method}'. "
            f"Available: cross_attention, gated, intermediate, concat, late, dynamic"
        )
