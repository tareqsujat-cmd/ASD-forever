"""
Genetics backbone factory: constructs any genetics encoder from a config string.

Adding a new backbone requires only:
  1. Implementing the backbone class in its own module
  2. Adding one case to build_genetics_encoder

Usage
-----
    from models.genetics.backbone_factory import build_genetics_encoder
    encoder = build_genetics_encoder(cfg, n_genes=256)
    # Always returns GeneticsEncoder regardless of backbone choice
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def build_genetics_encoder(
    cfg,
    n_genes: int,
    adj: Optional[torch.Tensor] = None,
) -> "GeneticsEncoder":
    """
    Build a GeneticsEncoder from configuration.

    Parameters
    ----------
    cfg : Config
        Loaded configuration object (from config_schema.py).
    n_genes : int
        Actual number of genes after feature selection / dimensionality reduction.
        Must be provided at runtime (not known at config-write time).
    adj : torch.Tensor, optional
        (n_genes, n_genes) binary adjacency matrix.
        Required only for architecture="gnn" or "graph_transformer".

    Returns
    -------
    GeneticsEncoder
    """
    from models.genetics.genetics_encoder import GeneticsEncoder

    gcfg = cfg.genetics_model
    arch = gcfg.architecture.lower()

    backbone = _build_backbone(arch, gcfg, n_genes, adj)

    encoder = GeneticsEncoder(
        backbone=backbone,
        feature_dim=gcfg.feature_dim,
        dropout=gcfg.dropout,
    )

    from utilities.hardware import count_parameters
    total, trainable = count_parameters(encoder)
    logger.info(
        "GeneticsEncoder [%s]: %s total params, %s trainable",
        arch, f"{total:,}", f"{trainable:,}",
    )
    return encoder


def _build_backbone(arch: str, gcfg, n_genes: int, adj):
    if arch == "transformer":
        from models.genetics.transformer_encoder import GeneTransformerEncoder
        return GeneTransformerEncoder(
            n_genes=n_genes,
            d_model=gcfg.feature_dim,           # reuse feature_dim as d_model
            n_heads=gcfg.num_heads,
            n_layers=gcfg.num_layers,
            dim_feedforward=gcfg.feature_dim * 4,
            feature_dim=gcfg.feature_dim,
            dropout=gcfg.dropout,
        )

    elif arch == "tabnet":
        from models.genetics.tabnet import TabNetEncoder
        n_d = getattr(gcfg, "tabnet_n_d", 64)
        n_a = getattr(gcfg, "tabnet_n_a", 64)
        n_steps = getattr(gcfg, "tabnet_n_steps", 3)
        gamma = getattr(gcfg, "tabnet_gamma", 1.5)
        return TabNetEncoder(
            n_genes=n_genes,
            n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
            feature_dim=gcfg.feature_dim,
            dropout=gcfg.dropout,
        )

    elif arch in ("gnn", "graph_transformer"):
        from models.genetics.gnn_encoder import GNNEncoder
        if adj is None:
            raise ValueError(
                f"architecture='{arch}' requires an adjacency matrix. "
                "Pass adj= to build_genetics_encoder() or use a different architecture."
            )
        gat_heads = getattr(gcfg, "gnn_heads", 4)
        gat_hidden = getattr(gcfg, "gnn_hidden_dim", 64)
        return GNNEncoder(
            n_genes=n_genes,
            adj=adj,
            emb_dim=gcfg.feature_dim // 4,
            gat_hidden=gat_hidden,
            gat_out=gat_hidden,
            n_heads=gat_heads,
            feature_dim=gcfg.feature_dim,
            dropout=gcfg.dropout,
        )

    elif arch == "mlp":
        from models.genetics.genetics_encoder import _MLPEncoder
        hidden = list(gcfg.hidden_dims)
        return _MLPEncoder(
            n_genes=n_genes,
            hidden_dims=hidden,
            feature_dim=gcfg.feature_dim,
            dropout=gcfg.dropout,
        )

    else:
        raise ValueError(
            f"Unknown genetics architecture: '{gcfg.architecture}'. "
            f"Available: transformer, tabnet, gnn, graph_transformer, mlp"
        )
