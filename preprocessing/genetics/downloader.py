"""
Genetics dataset downloader and parser.

Supported sources
-----------------
1. NCBI GEO (Gene Expression Omnibus) — microarray and RNA-seq
   - GSE18123: Lymphoblastoid cell lines, ASD vs control, Affymetrix HG-U133 Plus 2.0
   - GSE28521: Post-mortem brain tissue, ASD vs control, Illumina HumanRef-8 v3.0
   - GSE102741: Blood, ASD vs control, RNA-seq
2. SFARI Gene database — curated list of ASD-associated genes
3. Custom CSV/TSV — user-provided expression matrix

Data alignment
--------------
The critical challenge is matching GEO samples to ABIDE subjects.
GEO and ABIDE use different participant ID systems.
Where direct matching is not possible, we treat the datasets independently
and use multi-task learning or domain adaptation (documented as a limitation).

For the fusion framework to be fully supervised, we use:
  - Option A: Datasets that have BOTH imaging and genetics (few exist)
  - Option B: ABIDE metadata features (site, sex, age, IQ) as the genetic proxy
    for subjects without real genetic data (documented; commonly done in literature)
  - Option C: Transfer a genetics encoder pre-trained on GEO, fine-tune on ABIDE

We implement all three options. Option B is the default for the initial
fusion experiment (justified by Heinsfeld et al. 2018 approach).

Usage
-----
    from preprocessing.genetics.downloader import GEODownloader
    dl = GEODownloader(data_dir="datasets/raw/genetics")
    expr_df, meta_df = dl.download_geo(accession="GSE18123")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Curated list of GEO accessions relevant to ASD
ASD_GEO_ACCESSIONS = {
    "GSE18123": {
        "description": "Lymphoblastoid cell lines, ASD vs control, Affymetrix HG-U133 Plus 2.0",
        "platform": "GPL570",
        "n_asd": 146,
        "tissue": "lymphoblastoid",
    },
    "GSE28521": {
        "description": "Post-mortem brain (frontal/temporal cortex), ASD vs control",
        "platform": "GPL6947",
        "n_asd": 29,
        "tissue": "brain",
    },
    "GSE102741": {
        "description": "Peripheral blood mononuclear cells, ASD vs neurotypical",
        "platform": "RNA-seq",
        "n_asd": 69,
        "tissue": "blood",
    },
    "GSE6575": {
        "description": "Lymphoblastoid cell lines, ASD vs control",
        "platform": "GPL570",
        "n_asd": 35,
        "tissue": "lymphoblastoid",
    },
}


class GEODownloader:
    """
    Download and parse gene expression datasets from NCBI GEO.

    Uses GEOparse for SOFT file parsing. Falls back to manual FTP download
    if GEOparse is unavailable.

    Parameters
    ----------
    data_dir : str or Path
        Directory to store downloaded files.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"GEODownloader initialized: {self.data_dir}")

    def download_geo(
        self,
        accession: str,
        max_retries: int = 3,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download a GEO dataset and return (expression_matrix, metadata).

        Parameters
        ----------
        accession : str
            GEO accession number, e.g. "GSE18123".
        max_retries : int
            Download retry attempts.

        Returns
        -------
        expression_df : pd.DataFrame
            Shape (n_genes, n_samples). Index = gene symbols / probe IDs.
        metadata_df : pd.DataFrame
            Sample metadata with diagnosis labels.
        """
        accession = accession.upper()
        logger.info(f"Downloading GEO accession: {accession}")

        # Check cache
        cache_expr = self.data_dir / f"{accession}_expression.parquet"
        cache_meta = self.data_dir / f"{accession}_metadata.csv"

        if cache_expr.exists() and cache_meta.exists():
            logger.info(f"Loading from cache: {cache_expr}")
            expr_df = pd.read_parquet(cache_expr)
            meta_df = pd.read_csv(cache_meta)
            return expr_df, meta_df

        # Download via GEOparse
        expr_df, meta_df = self._download_geoparse(accession, max_retries)

        # Standardize
        expr_df, meta_df = self._standardize_geo(expr_df, meta_df, accession)

        # Cache
        expr_df.to_parquet(cache_expr)
        meta_df.to_csv(cache_meta, index=False)
        logger.info(f"Cached: {cache_expr} ({expr_df.shape[0]} genes, "
                    f"{expr_df.shape[1]} samples)")

        return expr_df, meta_df

    def download_sfari_genes(self) -> pd.DataFrame:
        """
        Load SFARI Gene ASD-associated gene list.

        SFARI Gene categorizes genes by evidence level:
          Score 1: High confidence (de novo mutations in multiple studies)
          Score 2: Strong candidate
          Score 3: Suggestive evidence
          Score S: Syndromic

        Returns
        -------
        pd.DataFrame with columns: gene_symbol, gene_id, sfari_score, ...
        """
        sfari_path = self.data_dir / "SFARI_genes.csv"
        if sfari_path.exists():
            return pd.read_csv(sfari_path)

        # Attempt download from SFARI API
        sfari_url = (
            "https://gene.sfari.org/wp-content/themes/sfari-gene/assets/data/"
            "SFARI-Gene_genes_01-01-2024release_01-01-2024export.csv"
        )
        try:
            import requests
            logger.info("Downloading SFARI gene list ...")
            for attempt in range(3):
                try:
                    resp = requests.get(sfari_url, timeout=30)
                    if resp.status_code == 200:
                        sfari_path.write_bytes(resp.content)
                        df = pd.read_csv(sfari_path)
                        logger.info(f"SFARI genes: {len(df)} entries")
                        return df
                except Exception as e:
                    logger.warning(f"SFARI download attempt {attempt+1} failed: {e}")
                    time.sleep(2 ** attempt)
        except ImportError:
            pass

        # Provide minimal built-in list as fallback
        logger.warning("SFARI download failed; using built-in high-confidence gene list")
        return self._builtin_sfari_genes()

    def load_expression_matrix(
        self,
        path: str | Path,
        sep: str = "\t",
        index_col: int = 0,
    ) -> pd.DataFrame:
        """
        Load a custom expression matrix from a TSV/CSV file.

        Format: genes as rows, samples as columns.
        First column must be gene identifiers (Entrez ID or gene symbol).

        Parameters
        ----------
        path : str or Path
        sep : str
            Separator character.
        index_col : int
            Column to use as row index (gene IDs).

        Returns
        -------
        pd.DataFrame, shape (n_genes, n_samples)
        """
        path = Path(path)
        logger.info(f"Loading expression matrix: {path}")
        df = pd.read_csv(path, sep=sep, index_col=index_col)
        logger.info(f"Loaded: {df.shape[0]} genes × {df.shape[1]} samples")
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_geoparse(
        self, accession: str, max_retries: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download and parse via GEOparse."""
        try:
            import GEOparse
        except ImportError:
            raise ImportError(
                "GEOparse is required for GEO download. "
                "Install with: pip install GEOparse"
            )

        for attempt in range(1, max_retries + 1):
            try:
                gse = GEOparse.get_GEO(
                    geo=accession,
                    destdir=str(self.data_dir),
                    silent=True,
                )
                break
            except Exception as exc:
                logger.warning(f"GEOparse attempt {attempt}/{max_retries}: {exc}")
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to download {accession} after {max_retries} attempts"
                    ) from exc
                time.sleep(2 ** attempt)

        # Extract expression matrix and metadata
        expr_tables = []
        meta_rows = []

        for gsm_id, gsm in gse.gsms.items():
            # Metadata
            chars = gsm.metadata.get("characteristics_ch1", [])
            row = {"sample_id": gsm_id}
            for ch in chars:
                if ":" in str(ch):
                    k, v = str(ch).split(":", 1)
                    row[k.strip().lower().replace(" ", "_")] = v.strip()
            # Diagnosis
            row["source_title"] = gsm.metadata.get("title", [""])[0]
            meta_rows.append(row)

            # Expression data
            tbl = gsm.table[["ID_REF", "VALUE"]].copy()
            tbl.columns = ["probe_id", gsm_id]
            tbl = tbl.set_index("probe_id")
            expr_tables.append(tbl)

        if not expr_tables:
            raise ValueError(f"No expression data found in {accession}")

        expr_df = pd.concat(expr_tables, axis=1)
        meta_df = pd.DataFrame(meta_rows)

        return expr_df, meta_df

    def _standardize_geo(
        self,
        expr_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        accession: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize expression matrix and metadata.

        - Convert probe IDs to gene symbols where possible
        - Standardize diagnosis labels to 0/1
        - Log2-transform raw counts if needed
        """
        meta_df = meta_df.copy()
        expr_df = expr_df.copy()

        # Numeric conversion
        expr_df = expr_df.apply(pd.to_numeric, errors="coerce")

        # Drop probes with >50% missing
        missing_rate = expr_df.isna().mean(axis=1)
        expr_df = expr_df[missing_rate < 0.5]

        # Log2-transform if values look like raw counts (max > 1000)
        max_val = expr_df.max().max()
        if max_val > 1000:
            expr_df = np.log2(expr_df + 1)
            logger.info(f"Applied log2(x+1) transform (max was {max_val:.0f})")

        # Standardize diagnosis column
        diag_candidates = [
            "diagnosis", "disease_state", "phenotype", "asd", "group",
            "condition", "status", "dx"
        ]
        label_col = None
        for col in diag_candidates:
            if col in meta_df.columns:
                label_col = col
                break

        if label_col is None:
            # Try to infer from title
            meta_df["label"] = meta_df["source_title"].str.contains(
                r"(?i)autism|asd", regex=True
            ).astype(int)
            logger.warning("Diagnosis column not found; inferred from sample title")
        else:
            asd_patterns = r"(?i)autism|asd|affected|case"
            meta_df["label"] = meta_df[label_col].str.contains(
                asd_patterns, regex=True, na=False
            ).astype(int)

        n_asd = meta_df["label"].sum()
        n_tc = (meta_df["label"] == 0).sum()
        logger.info(f"GEO {accession}: {n_asd} ASD, {n_tc} control "
                    f"| {expr_df.shape[0]} genes × {expr_df.shape[1]} samples")

        return expr_df, meta_df

    @staticmethod
    def _builtin_sfari_genes() -> pd.DataFrame:
        """Minimal built-in SFARI score-1/2 gene list (subset)."""
        high_confidence = [
            "SHANK3", "PTEN", "TSC1", "TSC2", "FMR1", "MECP2", "CNTNAP2",
            "NRXN1", "SHANK2", "NLGN3", "NLGN4X", "ADNP", "ARID1B",
            "CHD8", "DYRK1A", "FOXP1", "GRIN2B", "KDM6A", "KMT2A",
            "MED13L", "POGZ", "RAI1", "SCN1A", "SCN2A", "SETD5",
            "SHANK1", "SYNGAP1", "TBR1", "TBCK", "ANKRD11", "ASXL3",
            "CACNA1A", "CTNNB1", "DSCAM", "EHMT1", "FOXP2", "GIGYF1",
            "KCNQ3", "KDM5C", "MBD5", "NAA15", "PHF21A", "PTCHD1",
            "RELN", "SETBP1", "SLC6A1", "STXBP1", "TNRC6B", "WAC",
        ]
        return pd.DataFrame({
            "gene-symbol": high_confidence,
            "gene-score": ["1"] * len(high_confidence),
            "syndromic": [0] * len(high_confidence),
        })


def align_genetics_to_abide(
    genetics_meta: pd.DataFrame,
    abide_meta: pd.DataFrame,
    match_on: str = "subject_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align genetics and ABIDE metadata by subject ID.

    In practice, matched GEO-ABIDE cohorts are rare.  This function
    handles three cases:
      1. Exact match by subject_id
      2. No match → return disjoint datasets for pre-training / transfer
      3. Partial match → use matched subset for multi-modal training

    Parameters
    ----------
    genetics_meta : pd.DataFrame
        Genetics metadata (must have 'sample_id' and 'label').
    abide_meta : pd.DataFrame
        ABIDE metadata (must have 'subject_id' and 'label').
    match_on : str
        Column to join on.

    Returns
    -------
    matched_genetics : pd.DataFrame
    matched_abide : pd.DataFrame
    """
    if match_on not in genetics_meta.columns or match_on not in abide_meta.columns:
        logger.warning(
            "Cannot align genetics and ABIDE by subject_id — no common key. "
            "Datasets will be used separately for pre-training."
        )
        return genetics_meta, abide_meta

    merged = pd.merge(
        genetics_meta, abide_meta[[match_on, "label"]],
        on=match_on, how="inner", suffixes=("_genetics", "_abide")
    )
    logger.info(f"Aligned {len(merged)} subjects between genetics and ABIDE")
    return merged, merged
