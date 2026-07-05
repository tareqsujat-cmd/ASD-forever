"""
Gene Expression Transformer Encoder.

Design rationale
----------------
Gene expression data is an unordered set: each feature carries a fixed
biological identity (gene 42 is always CNTNAP2 after feature selection),
unlike language tokens whose meaning shifts with position.  We therefore
use IDENTITY embeddings (keyed by gene index) rather than positional
sinusoidal encodings.  A CLS token summarises the full profile.

Why Transformer over MLP
------------------------
1. Self-attention captures gene co-expression patterns natively in the model
   rather than forcing the user to pre-compute a co-expression graph.
2. Attention maps = interpretable per-gene importance weights (Module 8).
3. Published evidence: Transformers outperform MLPs on expression-based
   classification (scBERT 2022; Geneformer Nature 2023; Chen et al. ICLR 2020).
4. Pre-LN architecture is more stable than post-LN at small batch sizes,
   which are forced here by 3-D MRI memory constraints.

References
----------
Vaswani A et al. (2017). Attention Is All You Need. NeurIPS.
Xiong R et al. (2020). On Layer Normalization in the Transformer Encoder. ICML.
Yang H et al. (2022). scBERT as a large-scale pretrained deep language model
  for cell type annotation of single-cell RNA-seq data. Nature Machine Intelligence.
Theodoris CV et al. (2023). Transfer learning enables predictions in network
  biology. Nature.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GeneTokenizer(nn.Module):
    """
    Map (B, n_genes) expression vector → (B, n_genes+1, d_model) token sequence.

    Token construction for gene i:
        token_i = identity_embedding_i + value_projection(x_i)

    identity_embedding_i ∈ R^d_model encodes WHICH gene this is.
    value_projection(x_i) ∈ R^d_model encodes HOW STRONGLY it is expressed.

    The additive decomposition lets the transformer disentangle identity from
    magnitude, attending to interactions between specific genes at any
    expression level.
    """

    def __init__(self, n_genes: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_genes = n_genes

        # One learned embedding vector per gene (identity, not position)
        self.gene_embeddings = nn.Embedding(n_genes, d_model)

        # Small MLP projects scalar expression value to d_model
        self.value_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

        # CLS token: aggregates the full expression profile after attention
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Precomputed gene-index buffer, not a learnable parameter
        self.register_buffer("gene_ids", torch.arange(n_genes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_genes) — normalised gene expression values

        Returns
        -------
        (B, n_genes+1, d_model)  first token is CLS
        """
        B = x.shape[0]

        # (n_genes, d_model) → (B, n_genes, d_model)
        id_emb = self.gene_embeddings(self.gene_ids).unsqueeze(0).expand(B, -1, -1)

        # (B, n_genes) → (B, n_genes, 1) → (B, n_genes, d_model)
        val_emb = self.value_proj(x.unsqueeze(-1))

        tokens = id_emb + val_emb

        cls = self.cls_token.expand(B, -1, -1)
        return self.dropout(torch.cat([cls, tokens], dim=1))  # (B, n_genes+1, d_model)


class _PreLNTransformerLayer(nn.Module):
    """
    Single Pre-LN transformer encoder layer with optional attention caching.

    Pre-LN (normalise before attention) is numerically more stable than
    post-LN for small-batch training and deep networks.

    The _save_attn flag is toggled by GeneTransformerEncoder.get_attention_weights
    to extract attention maps without a second forward pass.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self._save_attn: bool = False
        self._attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, attn_w = self.attn(
            normed, normed, normed,
            need_weights=self._save_attn,
            average_attn_weights=False,
        )
        if self._save_attn:
            self._attn_weights = attn_w.detach()
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


class GeneTransformerEncoder(nn.Module):
    """
    Transformer encoder for gene expression feature vectors.

    Parameters
    ----------
    n_genes : int
        Number of input genes (post feature-selection).
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads. Must divide d_model evenly.
    n_layers : int
        Number of stacked transformer layers.
    dim_feedforward : int
        Feed-forward hidden dimension (default 4×d_model).
    feature_dim : int
        Output projection dimension (fed to fusion module).
    dropout : float
    """

    def __init__(
        self,
        n_genes: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        feature_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.n_genes = n_genes
        self.d_model = d_model
        self.feature_dim = feature_dim

        self.tokenizer = GeneTokenizer(n_genes, d_model, dropout)

        self.layers = nn.ModuleList([
            _PreLNTransformerLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.projector = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )

        self._init_weights()
        logger.info(
            "GeneTransformerEncoder: %d genes → d_model=%d, "
            "n_layers=%d, n_heads=%d → feature_dim=%d",
            n_genes, d_model, n_layers, n_heads, feature_dim,
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, n_genes)

        Returns
        -------
        (B, feature_dim)
        """
        tokens = self.tokenizer(x)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.final_norm(tokens)
        cls_out = tokens[:, 0, :]          # CLS token aggregates full profile
        return self.projector(cls_out)

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract per-layer attention matrices in a single forward pass.

        Used by Module 8 (explainability) for gene-importance visualisation.

        Returns
        -------
        list of (B, n_heads, n_genes+1, n_genes+1) tensors, one per layer.
        The [:, :, 0, 1:] slice gives CLS→gene attention (gene importance).
        """
        for layer in self.layers:
            layer._save_attn = True
        try:
            with torch.no_grad():
                _ = self.forward(x)
            return [layer._attn_weights for layer in self.layers]
        finally:
            for layer in self.layers:
                layer._save_attn = False
                layer._attn_weights = None
