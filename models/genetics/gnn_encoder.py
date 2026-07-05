"""
Graph Attention Network (GAT) encoder for gene expression data.

The STRING PPI graph (built in Module 2) encodes known protein-protein
interactions.  Treating genes as nodes and PPI edges as connections lets
the model propagate information through biological pathways rather than
treating genes as fully independent features.

Why GAT over GCN
----------------
GCN (Kipf & Welling, 2017) aggregates neighbours with fixed, normalised
weights.  GAT (Veličković et al., 2018) learns per-edge attention weights,
letting the model up-weight functionally important connections and down-weight
noisy ones.  For ASD genetics, only a subset of interactions in the STRING
graph are causally relevant — GAT's learned masking handles this naturally.

Implementation
--------------
No PyTorch Geometric dependency: the GATLayer uses dense-matrix message
passing (O(N²) memory), which is efficient for the ~200-500 gene graphs
produced after feature selection.  With N=500, the e_{ij} matrix is
500² * B * n_heads * 4 bytes ≈ 32 MB at B=8, n_heads=4 — well within budget.

Architecture
------------
Input: (B, n_nodes)   — gene expression values
Node embedding: (B, n_nodes, emb_dim)   — shared linear lift
GAT layer 1: emb_dim → hidden_dim, n_heads heads, concatenate → hidden_dim*n_heads
GAT layer 2: hidden_dim*n_heads → out_dim, n_heads heads, mean → out_dim
Graph readout: learned attention pooling over nodes → (B, out_dim)
Projection: → (B, feature_dim)

References
----------
Veličković P et al. (2018). Graph Attention Networks. ICLR 2018.
Kipf TN, Welling M. (2017). Semi-Supervised Classification with GCN. ICLR 2017.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GATLayer(nn.Module):
    """
    Single Graph Attention Network layer (dense adjacency variant).

    Multi-head attention with decomposed bilinear scoring:
        e_{ij} = LeakyReLU(a_src^T W h_i  +  a_dst^T W h_j)
        α_{ij} = softmax_j(e_{ij})  subject to adj_{ij}=1
        h'_i   = ELU( Σ_j α_{ij} W h_j )

    Parameters
    ----------
    in_features : int
    out_features : int   — per-head output dimension
    n_heads : int
    dropout : float
    concat : bool
        True → concatenate heads  → out_features * n_heads
        False → mean across heads → out_features
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat

        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)

        # Decomposed attention vectors (one per head)
        self.a_src = nn.Parameter(torch.empty(n_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_features))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU(inplace=False)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, N, in_features) or (N, in_features)
        adj : (N, N) binary adjacency (self-loops must be included)

        Returns
        -------
        (B, N, out_features * n_heads)  if concat
        (B, N, out_features)            if not concat
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        B, N, _ = x.shape

        # Linear projection then reshape to (B, N, n_heads, out_features)
        h = self.W(x).view(B, N, self.n_heads, self.out_features)

        # Decomposed attention scores (efficient: O(B*N*H) instead of O(B*N²*H))
        # a_src/dst: (n_heads, out_features)
        e_src = (h * self.a_src).sum(-1)           # (B, N, n_heads)
        e_dst = (h * self.a_dst).sum(-1)           # (B, N, n_heads)

        # e_{ij} = e_src_i + e_dst_j: broadcast → (B, N, N, n_heads)
        e = self.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1))

        # Mask non-edges: (N, N) → (1, N, N, 1), broadcast
        not_edge = (adj == 0).unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(not_edge, float("-inf"))

        # Normalised attention: softmax over source-node dimension (dim=2)
        alpha = torch.softmax(e, dim=2)           # (B, N, N, n_heads)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # isolated nodes → 0
        alpha = self.dropout(alpha)

        # Aggregation via batch matmul — O(B * n_heads * N²), memory-efficient
        # alpha_p: (B, n_heads, N_dest, N_src)
        # h_p:     (B, n_heads, N_src,  out_features)
        alpha_p = alpha.permute(0, 3, 1, 2)           # (B, n_heads, N, N)
        h_p = h.permute(0, 2, 1, 3)                   # (B, n_heads, N, out_features)
        out = torch.matmul(alpha_p, h_p)               # (B, n_heads, N, out_features)
        out = self.elu(out)
        out = out.permute(0, 2, 1, 3)                  # (B, N, n_heads, out_features)

        if self.concat:
            out = out.reshape(B, N, self.n_heads * self.out_features)
        else:
            out = out.mean(dim=2)                      # mean across heads

        if squeeze:
            out = out.squeeze(0)
        return out


class _AttentionReadout(nn.Module):
    """
    Learned attention pooling: compress (B, N, d) → (B, d).

    Each node gets a scalar attention weight (softmax-normalised over N nodes).
    More expressive than mean pooling: can focus on ASD-relevant genes.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, d) → (B, d)."""
        scores = torch.softmax(self.gate(h), dim=1)  # (B, N, 1)
        return (scores * h).sum(dim=1)               # (B, d)


class GNNEncoder(nn.Module):
    """
    2-layer GAT encoder over the STRING PPI gene interaction graph.

    Parameters
    ----------
    n_genes : int
        Number of input genes (graph nodes).
    adj : torch.Tensor, shape (n_genes, n_genes)
        Binary adjacency matrix from GeneGraphBuilder.
        Registered as a buffer (moves to GPU automatically).
    emb_dim : int
        Node embedding dimension (linear lift from scalar expression).
    gat_hidden : int
        Per-head hidden dimension in GAT layer 1.
    gat_out : int
        Per-head output dimension in GAT layer 2.
    n_heads : int
        Number of GAT attention heads.
    feature_dim : int
        Final projection output dimension.
    dropout : float
    """

    def __init__(
        self,
        n_genes: int,
        adj: torch.Tensor,
        emb_dim: int = 64,
        gat_hidden: int = 64,
        gat_out: int = 64,
        n_heads: int = 4,
        feature_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.feature_dim = feature_dim

        # Register adjacency as a buffer (no gradients, moved to device automatically)
        self.register_buffer("adj", adj.float())

        # Lift scalar expression value to embedding dimension
        self.input_embed = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
        )

        # Layer 1: emb_dim → gat_hidden per head (concat → gat_hidden * n_heads)
        self.gat1 = GATLayer(
            emb_dim, gat_hidden, n_heads=n_heads, dropout=dropout, concat=True
        )
        self.norm1 = nn.LayerNorm(gat_hidden * n_heads)

        # Layer 2: gat_hidden*n_heads → gat_out per head (mean → gat_out)
        self.gat2 = GATLayer(
            gat_hidden * n_heads, gat_out, n_heads=n_heads, dropout=dropout, concat=False
        )
        self.norm2 = nn.LayerNorm(gat_out)

        self.readout = _AttentionReadout(gat_out)

        self.projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gat_out, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        self._init_weights()
        logger.info(
            "GNNEncoder: %d genes (nodes) → emb=%d, GAT(%d heads, %d→%d) → feature_dim=%d",
            n_genes, emb_dim, n_heads, gat_hidden, gat_out, feature_dim,
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, n_genes)

        Returns
        -------
        features : (B, feature_dim)
        aux_loss : scalar zero (GNN has no auxiliary loss; returned for API symmetry)
        """
        B, N = x.shape

        # Lift each scalar expression value to an embedding: (B, N, emb_dim)
        h = self.input_embed(x.unsqueeze(-1))  # (B, N, 1) → (B, N, emb_dim)

        # GAT layer 1 with residual-style norm
        h = self.gat1(h, self.adj)             # (B, N, gat_hidden * n_heads)
        h = self.norm1(h)

        # GAT layer 2
        h = self.gat2(h, self.adj)             # (B, N, gat_out)
        h = self.norm2(h)

        # Graph-level readout: (B, gat_out)
        h = self.readout(h)

        aux_loss = torch.zeros(1, device=x.device)
        return self.projector(h), aux_loss
