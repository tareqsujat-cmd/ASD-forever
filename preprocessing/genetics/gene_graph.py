"""
Gene interaction graph construction for Graph Neural Networks.

Why model genes as a graph?
-----------------------------
Genes do not act independently: they form regulatory networks, signaling
pathways, and protein-protein interaction (PPI) networks.  A flat vector
of gene expression values ignores this structure.

A Graph Neural Network (GNN) operating on the gene graph can:
1. Propagate information between interacting genes (message passing)
2. Weight genes by their network centrality / pathway membership
3. Capture epistatic (gene-gene interaction) effects on ASD risk

Graph source: STRING database (v12.0)
--------------------------------------
STRING (Szklarczyk et al., 2023) provides PPI scores based on:
  - Co-expression evidence
  - Experimental binding assays
  - Text-mining
  - Orthology-based transfer from model organisms

We use combined_score >= 700 (high-confidence interactions only).

Reference
---------
Szklarczyk D, et al. (2023). The STRING database in 2023: protein–protein
association networks and functional enrichment analyses for any sequenced
genome of interest. Nucleic Acids Research 51(D1):D638-D646.

Usage
-----
    from preprocessing.genetics.gene_graph import GeneGraphBuilder
    builder = GeneGraphBuilder(score_threshold=700)
    adj_matrix, gene_index = builder.build(selected_genes)
    edge_index, edge_weight = builder.to_pyg_format()
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# STRING API endpoint (programmatic access)
_STRING_API_URL = "https://string-db.org/api/tsv/network"
_STRING_SPECIES_HUMAN = 9606


class GeneGraphBuilder:
    """
    Build gene interaction graphs from STRING PPI network.

    Parameters
    ----------
    score_threshold : int
        Minimum STRING combined score (0-1000). 700 = high confidence.
    add_self_loops : bool
        Add self-loop edges (required by many GNN architectures).
    normalize_weights : bool
        Normalize edge weights to [0, 1].
    fallback_to_coexpression : bool
        If STRING download fails, compute co-expression graph from the data.
    """

    def __init__(
        self,
        score_threshold: int = 700,
        add_self_loops: bool = True,
        normalize_weights: bool = True,
        fallback_to_coexpression: bool = True,
    ) -> None:
        self.score_threshold = score_threshold
        self.add_self_loops = add_self_loops
        self.normalize_weights = normalize_weights
        self.fallback_to_coexpression = fallback_to_coexpression
        self._ppi_df: Optional[pd.DataFrame] = None
        self._gene_index: Dict[str, int] = {}
        self._adj_matrix: Optional[np.ndarray] = None

    def build(
        self,
        gene_list: List[str],
        expr_matrix: Optional[np.ndarray] = None,
        cache_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Build adjacency matrix for the given gene list.

        Parameters
        ----------
        gene_list : list of str
            Gene symbols to include in the graph.
        expr_matrix : np.ndarray, optional
            Shape (n_samples, n_genes) — used for co-expression fallback.
        cache_path : str, optional
            Save/load the PPI network cache.

        Returns
        -------
        adj_matrix : np.ndarray, shape (n_genes, n_genes)
            Weighted adjacency matrix.
        gene_index : dict
            Maps gene symbol -> row/col index in adj_matrix.
        """
        gene_list = [g.upper() for g in gene_list]
        self._gene_index = {g: i for i, g in enumerate(gene_list)}
        n = len(gene_list)

        # Check cache
        if cache_path and Path(cache_path).exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"Loaded gene graph from cache: {cache_path}")
            self._adj_matrix = cached["adj"]
            return self._adj_matrix, self._gene_index

        # Download STRING interactions
        ppi_df = self._download_string(gene_list)

        if ppi_df is None or len(ppi_df) == 0:
            if self.fallback_to_coexpression and expr_matrix is not None:
                logger.info("STRING unavailable; building co-expression graph")
                self._adj_matrix = self._coexpression_graph(expr_matrix, gene_list)
            else:
                logger.warning("No interactions found; using identity (no-graph) adjacency")
                self._adj_matrix = np.eye(n, dtype=np.float32)
        else:
            self._adj_matrix = self._build_adj_from_ppi(ppi_df, gene_list)

        # Add self-loops
        if self.add_self_loops:
            np.fill_diagonal(self._adj_matrix, 1.0)

        # Normalize
        if self.normalize_weights:
            max_w = self._adj_matrix.max()
            if max_w > 0:
                self._adj_matrix /= max_w

        # Cache
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"adj": self._adj_matrix, "gene_index": self._gene_index}, f)

        n_edges = (self._adj_matrix > 0).sum() - n  # exclude self-loops
        logger.info(f"Gene graph: {n} genes, {n_edges} PPI edges")
        return self._adj_matrix, self._gene_index

    def to_pyg_format(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert adjacency matrix to PyTorch Geometric edge_index format.

        Returns
        -------
        edge_index : np.ndarray, shape (2, n_edges)
            Source and target node indices.
        edge_weight : np.ndarray, shape (n_edges,)
            Edge weights.
        """
        if self._adj_matrix is None:
            raise RuntimeError("Call build() first")

        rows, cols = np.where(self._adj_matrix > 0)
        edge_index = np.stack([rows, cols], axis=0)
        edge_weight = self._adj_matrix[rows, cols]
        return edge_index, edge_weight

    def compute_graph_statistics(self) -> Dict:
        """Compute graph-theoretic statistics for the paper methods section."""
        if self._adj_matrix is None:
            return {}

        A = (self._adj_matrix > 0).astype(float)
        n = A.shape[0]
        degree = A.sum(axis=1)

        return {
            "n_nodes": n,
            "n_edges": int(A.sum()) // 2,
            "mean_degree": float(degree.mean()),
            "max_degree": float(degree.max()),
            "density": float(A.sum()) / (n * (n - 1)),
            "isolated_nodes": int((degree == 0).sum()),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_string(
        self, gene_list: List[str]
    ) -> Optional[pd.DataFrame]:
        """Download STRING interactions for the gene list via REST API."""
        try:
            import requests

            # STRING API accepts up to 2000 identifiers per request
            chunk_size = 500
            all_interactions = []

            for start in range(0, len(gene_list), chunk_size):
                chunk = gene_list[start:start + chunk_size]
                identifiers = "\r".join(chunk)

                params = {
                    "identifiers": identifiers,
                    "species": _STRING_SPECIES_HUMAN,
                    "required_score": self.score_threshold,
                    "caller_identity": "asd_multimodal_framework",
                }
                resp = requests.post(
                    _STRING_API_URL, data=params, timeout=60
                )
                if resp.status_code != 200:
                    logger.warning(f"STRING API error {resp.status_code}")
                    return None

                lines = resp.text.strip().split("\n")
                if len(lines) < 2:
                    continue

                headers = lines[0].split("\t")
                for line in lines[1:]:
                    parts = line.split("\t")
                    if len(parts) >= len(headers):
                        all_interactions.append(
                            dict(zip(headers, parts))
                        )

            if not all_interactions:
                return None

            df = pd.DataFrame(all_interactions)
            # Standard STRING column names
            if "preferredName_A" not in df.columns:
                # Try alternate column names
                for col_a, col_b, col_w in [
                    ("protein1", "protein2", "combined_score"),
                    ("node1_string_id", "node2_string_id", "combined_score"),
                ]:
                    if col_a in df.columns:
                        df = df.rename(columns={
                            col_a: "preferredName_A",
                            col_b: "preferredName_B",
                            col_w: "combined_score",
                        })
                        break

            if "combined_score" in df.columns:
                df["combined_score"] = pd.to_numeric(df["combined_score"], errors="coerce")
                df = df[df["combined_score"] >= self.score_threshold]

            logger.info(f"STRING: {len(df)} interactions downloaded")
            return df

        except Exception as exc:
            logger.warning(f"STRING download failed: {exc}")
            return None

    def _build_adj_from_ppi(
        self,
        ppi_df: pd.DataFrame,
        gene_list: List[str],
    ) -> np.ndarray:
        """Build adjacency matrix from PPI edge list."""
        n = len(gene_list)
        adj = np.zeros((n, n), dtype=np.float32)
        gene_set = set(gene_list)

        for _, row in ppi_df.iterrows():
            g1 = str(row.get("preferredName_A", "")).upper()
            g2 = str(row.get("preferredName_B", "")).upper()
            w = float(row.get("combined_score", 700)) / 1000.0

            if g1 in gene_set and g2 in gene_set:
                i = self._gene_index.get(g1)
                j = self._gene_index.get(g2)
                if i is not None and j is not None:
                    adj[i, j] = w
                    adj[j, i] = w  # undirected

        return adj

    @staticmethod
    def _coexpression_graph(
        expr_matrix: np.ndarray,
        gene_list: List[str],
        percentile: float = 90.0,
    ) -> np.ndarray:
        """
        Build co-expression graph from Pearson correlations.

        Only keep edges where |r| > p90 threshold to keep the graph sparse.
        """
        # expr_matrix: (n_samples, n_genes)
        corr = np.corrcoef(expr_matrix.T)
        threshold = np.percentile(np.abs(corr[corr != 1]), percentile)
        adj = np.where(np.abs(corr) >= threshold, np.abs(corr), 0.0).astype(np.float32)
        np.fill_diagonal(adj, 0)  # self-loops added separately
        logger.info(f"Co-expression graph: threshold={threshold:.3f}, "
                    f"edges={(adj > 0).sum() // 2}")
        return adj
