"""
ABIDE I / II data preparation pipeline.

Downloads ABIDE I or ABIDE II preprocessed fMRI data via nilearn, runs the
project's MRI and genetics preprocessing pipelines (Modules 1–2), and writes
the directory layout that ``run_experiment.py --real_data`` expects.

Typical two-step workflow
-------------------------
Step 1 — Prepare ABIDE I as the training + internal-validation corpus::

    python prepare_abide.py \\
        --dataset abide1 \\
        --out_dir datasets/abide1

Step 2 — Prepare ABIDE II as the held-out external test set, reusing ABIDE I's
fitted intensity normalizer to prevent data leakage::

    python prepare_abide.py \\
        --dataset abide2 \\
        --role held_out \\
        --abide1_dir datasets/abide1 \\
        --out_dir datasets/abide2

Then train + evaluate::

    python run_experiment.py --real_data \\
        --mri_dir datasets/abide1/mri \\
        --gen_dir datasets/abide1/genetics \\
        --held_out_mri_dir datasets/abide2/mri \\
        --held_out_gen_dir datasets/abide2/genetics \\
        --n_folds 5 --max_epochs 100

Output layout (same for both datasets)
---------------------------------------
<out_dir>/
  mri/
    metadata.csv               — subject_id, label, site, split
    intensity_normalizer.pkl   — fitted IntensityNormalizer (train role only)
    <subject_id>.npy           — (1, D, H, W) float32 preprocessed MRI volume
  genetics/
    metadata.csv               — same rows as mri/metadata.csv
    <subject_id>.npy           — (n_components,) float32 genetics feature vector

Splits
------
  train role   : 70 % train / 15 % val / 15 % test  (stratified per site)
  held_out role: all rows set to "test"  (ABIDE II is never used for training)

Genetics modes
--------------
  ``--genetics_mode phenotypic``  (default)
      Uses ABIDE's phenotypic variables (age, sex, FIQ, VIQ, PIQ, …) as a proxy
      genetics vector — available immediately without external data.

  ``--genetics_mode geo``
      Loads a GEO gene-expression matrix (CSV, genes × subjects) and runs the
      full Module-2 pipeline: imputation → ComBat → feature selection → PCA.
      Requires ``--geo_csv <path>``.  For held-out role also supply
      ``--abide1_geo_selector <path>`` and ``--abide1_geo_pca <path>`` so the
      same feature selector and PCA transform learned on ABIDE I are applied to
      ABIDE II (no leakage).

Other options
-------------
  --n_subjects N     Process only the first N subjects (useful for smoke tests)
  --resume           Skip subjects whose .npy files already exist
  --qc_strict        Exclude subjects failing MRIQualityChecker thresholds
  --pipeline         ABIDE preprocessing pipeline: cpac | dpabi | niak | ccs
  --target_shape     MRI spatial shape after resampling (default 96 96 96)
  --voxel_size       Isotropic voxel size in mm (default 2.0 2.0 2.0)
  --n_pca_components PCA output dimension for genetics vector (default 256)
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_abide")
warnings.filterwarnings("ignore", category=FutureWarning)


# ===========================================================================
# Phenotypic column schemas
# ===========================================================================

# ABIDE I: fetch_abide_pcp() column names
_ABIDE1_COLS = dict(
    sub_id   = "SUB_ID",
    dx_group = "DX_GROUP",   # 1 = ASD, 2 = TC
    site_id  = "SITE_ID",
    age      = "AGE_AT_SCAN",
    sex      = "SEX",
    fiq      = "FIQ",
    viq      = "VIQ",
    piq      = "PIQ",
    eye      = "EYE_STATUS_AT_SCAN",
    hand     = "HANDEDNESS_CATEGORY",
    adi_verb = "ADI_R_VERBAL_TOTAL_BV",
    adi_soc  = "ADI_R_SOCIAL_TOTAL_A",
    ados_mod = "ADOS_MODULE",
    ados_tot = "ADOS_TOTAL",
)

# ABIDE II: fetch_abide2() column names (mostly identical, a few differ)
_ABIDE2_COLS = dict(
    sub_id   = "SUB_ID",
    dx_group = "DX_GROUP",   # 1 = ASD, 2 = TC  (same encoding)
    site_id  = "SITE_ID",
    age      = "AGE_AT_SCAN",
    sex      = "SEX",
    fiq      = "FIQ",
    viq      = "VIQ",
    piq      = "PIQ",
    eye      = "EYE_STATUS_AT_SCAN",
    hand     = "HANDEDNESS_CATEGORY",
    adi_verb = "ADI_R_VERBAL_TOTAL_BV",
    adi_soc  = "ADI_R_SOCIAL_TOTAL_A",
    ados_mod = "ADOS_MODULE",
    ados_tot = "ADOS_TOTAL",
)

_SCHEMA = {"abide1": _ABIDE1_COLS, "abide2": _ABIDE2_COLS}


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and preprocess ABIDE I or ABIDE II data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Dataset identity ---
    p.add_argument("--dataset",       type=str, default="abide1",
                   choices=["abide1", "abide2"],
                   help="Which ABIDE dataset to prepare")
    p.add_argument("--role",          type=str, default=None,
                   choices=["train", "held_out"],
                   help="Dataset role. Default: 'train' for abide1, 'held_out' for abide2")
    p.add_argument("--out_dir",       type=str, default=None,
                   help="Root output directory (default: datasets/<dataset>)")

    # --- Download ---
    p.add_argument("--pipeline",      type=str, default="cpac",
                   choices=["cpac", "dpabi", "niak", "ccs"],
                   help="ABIDE preprocessing pipeline")
    p.add_argument("--strategy",      type=str, default="filt_global",
                   help="Noise-correction strategy (ABIDE I only)")
    p.add_argument("--n_subjects",    type=int, default=None,
                   help="Limit to first N subjects (None = all)")
    p.add_argument("--data_dir",      type=str, default=None,
                   help="nilearn cache dir (uses default if None)")

    # --- MRI preprocessing ---
    p.add_argument("--target_shape",  type=int, nargs=3, default=[96, 96, 96],
                   metavar=("D", "H", "W"),
                   help="Target MRI volume shape after resampling")
    p.add_argument("--voxel_size",    type=float, nargs=3, default=[2.0, 2.0, 2.0],
                   metavar=("X", "Y", "Z"),
                   help="Target isotropic voxel size (mm)")
    p.add_argument("--qc_strict",     action="store_true",
                   help="Exclude subjects failing MRIQualityChecker thresholds")

    # --- Leakage-free held_out processing ---
    p.add_argument("--abide1_dir",    type=str, default=None,
                   help="Path to prepared ABIDE I output dir. Required when "
                        "--role held_out to load fitted normalizer.")

    # --- Genetics ---
    p.add_argument("--genetics_mode", type=str, default="phenotypic",
                   choices=["phenotypic", "geo"],
                   help="Genetics data source")
    p.add_argument("--geo_csv",       type=str, default=None,
                   help="Path to GEO expression CSV (genes × subjects)")
    p.add_argument("--abide1_geo_selector", type=str, default=None,
                   help="Saved GeneFeatureSelector from ABIDE I (held_out + geo only)")
    p.add_argument("--abide1_geo_pca",      type=str, default=None,
                   help="Saved PCAReducer from ABIDE I (held_out + geo only)")
    p.add_argument("--n_pca_components", type=int, default=256,
                   help="PCA output dimension for genetics feature vector")

    # --- Misc ---
    p.add_argument("--resume",        action="store_true",
                   help="Skip subjects whose .npy files already exist")
    p.add_argument("--seed",          type=int, default=42)

    args = p.parse_args()

    # Fill in defaults that depend on --dataset / --role
    if args.role is None:
        args.role = "train" if args.dataset == "abide1" else "held_out"
    if args.out_dir is None:
        args.out_dir = f"datasets/{args.dataset}"

    return args


# ===========================================================================
# Step 1 — Download
# ===========================================================================

def _normalise_pheno_columns(
    phenotypic: "pd.DataFrame",
    schema:     Dict[str, str],
) -> "pd.DataFrame":
    """
    Normalise column names (nilearn may return bytes in older versions) and
    add canonical ``label`` and ``site`` columns using the dataset's schema.
    """
    phenotypic = phenotypic.copy()
    phenotypic.columns = [
        c.decode() if isinstance(c, bytes) else c
        for c in phenotypic.columns
    ]

    dx_col   = schema["dx_group"]
    site_col = schema["site_id"]

    if dx_col not in phenotypic.columns:
        candidates = [c for c in phenotypic.columns
                      if "dx" in c.lower() or "group" in c.lower()]
        if candidates:
            dx_col = candidates[0]
            logger.warning("DX column not found as '%s'; using '%s'",
                           schema["dx_group"], dx_col)
        else:
            raise KeyError(
                f"Cannot find diagnosis column in phenotypic. "
                f"Available: {list(phenotypic.columns)}"
            )

    # DX_GROUP: 1=ASD, 2=TC → binary label
    phenotypic["label"] = (phenotypic[dx_col] == 1).astype(int)
    phenotypic["site"]  = phenotypic[site_col].astype(str)

    # Ensure SUB_ID is a string column named consistently
    sub_col = schema["sub_id"]
    if sub_col not in phenotypic.columns:
        candidates = [c for c in phenotypic.columns if "sub" in c.lower()]
        sub_col = candidates[0] if candidates else phenotypic.columns[0]
        logger.warning("SUB_ID column not found; using '%s'", sub_col)
    phenotypic["SUB_ID"] = phenotypic[sub_col].astype(str)

    return phenotypic


def download_abide1(
    pipeline:   str,
    strategy:   str,
    n_subjects: Optional[int],
    data_dir:   Optional[str],
) -> Tuple["pd.DataFrame", List[str]]:
    """
    Fetch ABIDE I preprocessed fMRI via nilearn.fetch_abide_pcp().

    Returns (phenotypic_df, func_file_paths).
    """
    try:
        from nilearn.datasets import fetch_abide_pcp
    except ImportError:
        logger.error("nilearn is required: pip install nilearn")
        sys.exit(1)

    import pandas as pd

    logger.info("Fetching ABIDE I (pipeline=%s, strategy=%s) …", pipeline, strategy)
    logger.info("First run may download ~8 GB of data — this can take 20-60 min.")

    kwargs: Dict = dict(
        pipeline                 = pipeline,
        band_pass_filtering      = True,
        global_signal_regression = (strategy == "filt_global"),
        derivatives              = ["func_preproc"],
        verbose                  = 0,
    )
    if data_dir:
        kwargs["data_dir"] = data_dir
    if n_subjects:
        kwargs["n_subjects"] = n_subjects

    dataset    = fetch_abide_pcp(**kwargs)
    func_files = dataset.func_preproc
    phenotypic = _normalise_pheno_columns(
        pd.DataFrame(dataset.phenotypic), _ABIDE1_COLS
    )

    logger.info(
        "ABIDE I: %d subjects  ASD=%d  TC=%d  sites=%d",
        len(phenotypic),
        int((phenotypic["label"] == 1).sum()),
        int((phenotypic["label"] == 0).sum()),
        phenotypic["site"].nunique(),
    )
    return phenotypic, func_files


def download_abide2(
    pipeline:   str,
    n_subjects: Optional[int],
    data_dir:   Optional[str],
) -> Tuple["pd.DataFrame", List[str]]:
    """
    Fetch ABIDE II preprocessed fMRI via nilearn.fetch_abide2().

    ABIDE II (Martino et al. 2017) extends ABIDE I with ~1,114 subjects from
    27 international sites.  The nilearn API mirrors fetch_abide_pcp() but
    omits the noise-correction strategy parameter.

    Returns (phenotypic_df, func_file_paths).
    """
    try:
        from nilearn.datasets import fetch_abide2
    except ImportError:
        logger.error(
            "nilearn >= 0.9.0 with ABIDE II support is required.\n"
            "  pip install --upgrade nilearn"
        )
        sys.exit(1)

    import pandas as pd

    logger.info("Fetching ABIDE II (pipeline=%s) …", pipeline)
    logger.info("First run may download ~9 GB of data — this can take 20-60 min.")

    kwargs: Dict = dict(
        pipeline    = pipeline,
        derivatives = ["func_preproc"],
        verbose     = 0,
    )
    if data_dir:
        kwargs["data_dir"] = data_dir
    if n_subjects:
        kwargs["n_subjects"] = n_subjects

    dataset    = fetch_abide2(**kwargs)
    func_files = dataset.func_preproc
    phenotypic = _normalise_pheno_columns(
        pd.DataFrame(dataset.phenotypic), _ABIDE2_COLS
    )

    logger.info(
        "ABIDE II: %d subjects  ASD=%d  TC=%d  sites=%d",
        len(phenotypic),
        int((phenotypic["label"] == 1).sum()),
        int((phenotypic["label"] == 0).sum()),
        phenotypic["site"].nunique(),
    )
    return phenotypic, func_files


# ===========================================================================
# Step 2 — MRI preprocessing (Module 1)
# ===========================================================================

def preprocess_mri_subject(
    func_path:    str,
    subject_id:   str,
    target_shape: Tuple[int, int, int],
    voxel_size:   Tuple[float, float, float],
    qc_checker,
    strict_qc:    bool,
) -> Optional[np.ndarray]:
    """
    Load one subject's functional NIfTI, compute the temporal-mean 3D volume,
    resample to target voxel size, pad/crop to target_shape, and QC.

    Returns (D, H, W) float32 array or None if loading or strict QC fails.
    """
    from preprocessing.mri import (
        load_nifti, resample_volume, pad_or_crop_to_shape,
    )

    try:
        data, affine = load_nifti(func_path, validate=False)
    except Exception as exc:
        logger.warning("  %s — load failed: %s", subject_id, exc)
        return None

    # 4-D fMRI → temporal mean 3-D volume
    if data.ndim == 4:
        data = data.mean(axis=-1)
    data = data.astype(np.float32)

    data, affine = resample_volume(
        data, affine, target_voxel_size=voxel_size
    )
    data = pad_or_crop_to_shape(data, target_shape)

    brain_mask = (data > data.mean()).astype(np.uint8)
    try:
        qc_report = qc_checker.evaluate(
            data, brain_mask,
            subject_id    = subject_id,
            voxel_size_mm = float(voxel_size[0]),
        )
        if strict_qc and not qc_report.passed:
            logger.warning(
                "  %s — QC FAIL (snr=%.1f)",
                subject_id,
                qc_report.snr if hasattr(qc_report, "snr") else float("nan"),
            )
            return None
    except Exception as exc:
        logger.debug("  %s — QC warning (non-fatal): %s", subject_id, exc)

    return data.astype(np.float32)


def run_mri_pipeline(
    phenotypic:           "pd.DataFrame",
    func_files:           List[str],
    out_dir:              Path,
    target_shape:         Tuple[int, int, int],
    voxel_size:           Tuple[float, float, float],
    strict_qc:            bool,
    resume:               bool,
    pretrained_normalizer: Optional[Path] = None,
) -> "pd.DataFrame":
    """
    Preprocess all subjects, apply intensity normalisation, save .npy files.

    Parameters
    ----------
    pretrained_normalizer
        If provided, load a previously fitted IntensityNormalizer and call
        ``transform()`` only (no re-fitting).  Required for the held-out
        ABIDE II set to avoid leaking ABIDE II statistics into the normalizer.

    Returns filtered phenotypic DataFrame (QC-passed subjects only).
    """
    import pandas as pd
    from preprocessing.mri import IntensityNormalizer, MRIQualityChecker

    out_dir.mkdir(parents=True, exist_ok=True)
    qc_checker = MRIQualityChecker()

    # Load or create normalizer
    if pretrained_normalizer is not None:
        logger.info("Loading pretrained IntensityNormalizer from %s",
                    pretrained_normalizer)
        with open(pretrained_normalizer, "rb") as fh:
            normalizer = pickle.load(fh)
        fit_normalizer = False
    else:
        normalizer     = IntensityNormalizer(method="z_score", site_aware=True)
        fit_normalizer = True

    # ------------------------------------------------------------------ #
    # Pass 1 — load, resample, QC every subject
    # ------------------------------------------------------------------ #
    volumes:   List[np.ndarray] = []
    site_ids:  List[str]        = []
    kept_rows: List[int]        = []

    for idx, (_, row) in enumerate(phenotypic.iterrows()):
        sub_id   = str(row["SUB_ID"])
        npy_path = out_dir / f"{sub_id}.npy"

        if resume and npy_path.exists():
            logger.info("  [%d/%d] %s — skip (exists)",
                        idx + 1, len(phenotypic), sub_id)
            vol = np.load(npy_path)
            if vol.ndim == 4:
                vol = vol[0]          # strip channel dim loaded from disk
            volumes.append(vol)
            site_ids.append(str(row["site"]))
            kept_rows.append(idx)
            continue

        logger.info("  [%d/%d] %s", idx + 1, len(phenotypic), sub_id)
        if idx >= len(func_files):
            logger.warning("  No func file at index %d — skipping subject", idx)
            continue

        vol = preprocess_mri_subject(
            func_path    = func_files[idx],
            subject_id   = sub_id,
            target_shape = target_shape,
            voxel_size   = voxel_size,
            qc_checker   = qc_checker,
            strict_qc    = strict_qc,
        )
        if vol is None:
            continue

        volumes.append(vol)
        site_ids.append(str(row["site"]))
        kept_rows.append(idx)

    if not volumes:
        logger.error("No subjects survived preprocessing — aborting.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Pass 2 — intensity normalisation
    # ------------------------------------------------------------------ #
    if fit_normalizer:
        logger.info(
            "Fitting site-aware IntensityNormalizer on %d volumes …", len(volumes)
        )
        normed = normalizer.fit_transform(volumes, site_ids=site_ids)
        normalizer.save(out_dir / "intensity_normalizer.pkl")
        logger.info("Normalizer saved → %s", out_dir / "intensity_normalizer.pkl")
    else:
        logger.info(
            "Applying pretrained normalizer to %d volumes (no re-fitting) …",
            len(volumes),
        )
        normed = [
            normalizer.transform(v, site_id=s)
            for v, s in zip(volumes, site_ids)
        ]

    # ------------------------------------------------------------------ #
    # Pass 3 — save .npy with leading channel dim (1, D, H, W)
    # ------------------------------------------------------------------ #
    kept_pheno = phenotypic.iloc[kept_rows].copy().reset_index(drop=True)
    for i, (normed_vol, orig_idx) in enumerate(zip(normed, kept_rows)):
        sub_id   = str(phenotypic.iloc[orig_idx]["SUB_ID"])
        npy_path = out_dir / f"{sub_id}.npy"
        np.save(npy_path, normed_vol[np.newaxis].astype(np.float32))

    logger.info(
        "MRI pipeline complete: %d / %d subjects kept",
        len(kept_pheno), len(phenotypic),
    )
    return kept_pheno


# ===========================================================================
# Step 3a — Genetics: phenotypic proxy (Module 2 pipeline)
# ===========================================================================

def _phenotypic_feature_cols(schema: Dict[str, str]) -> List[str]:
    """Return the list of phenotypic column names to use as genetics proxy."""
    return [
        schema.get("age",      "AGE_AT_SCAN"),
        schema.get("sex",      "SEX"),
        schema.get("fiq",      "FIQ"),
        schema.get("viq",      "VIQ"),
        schema.get("piq",      "PIQ"),
        schema.get("eye",      "EYE_STATUS_AT_SCAN"),
        schema.get("hand",     "HANDEDNESS_CATEGORY"),
        schema.get("adi_verb", "ADI_R_VERBAL_TOTAL_BV"),
        schema.get("adi_soc",  "ADI_R_SOCIAL_TOTAL_A"),
        schema.get("ados_mod", "ADOS_MODULE"),
        schema.get("ados_tot", "ADOS_TOTAL"),
    ]


def _phenotypic_genetics(
    pheno:        "pd.DataFrame",
    schema:       Dict[str, str],
    n_components: int,
    seed:         int,
) -> "np.ndarray":
    """
    Build (N, n_components) from ABIDE phenotypic variables.

    Handles missing columns gracefully, imputes with median, standardises,
    random-projects if fewer features than n_components, then PCA-reduces.
    """
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from preprocessing.genetics import PCAReducer

    feature_cols = _phenotypic_feature_cols(schema)
    avail        = [c for c in feature_cols if c in pheno.columns]
    if not avail:
        logger.warning(
            "No phenotypic feature columns found — returning zero vectors."
        )
        return np.zeros((len(pheno), n_components), dtype=np.float32)

    X = pheno[avail].copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.Categorical(X[col]).codes.astype(float)

    X = X.values.astype(np.float64)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    if X.shape[1] < n_components:
        rng  = np.random.default_rng(seed)
        proj = rng.standard_normal((X.shape[1], n_components)).astype(np.float64)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-9
        X    = X @ proj

    reducer = PCAReducer(n_components=n_components, random_state=seed)
    return reducer.fit_transform(X.astype(np.float32)).astype(np.float32)


# ===========================================================================
# Step 3b — Genetics: GEO gene expression (Module 2 pipeline)
# ===========================================================================

def _geo_genetics_fit(
    pheno:        "pd.DataFrame",
    geo_csv:      str,
    n_components: int,
    seed:         int,
    save_dir:     Path,
) -> "np.ndarray":
    """
    Full Module-2 pipeline on ABIDE I: impute → ComBat → select → PCA.

    Saves fitted GeneFeatureSelector and PCAReducer to ``save_dir`` for
    later application to the held-out ABIDE II set.
    """
    import pandas as pd
    from preprocessing.genetics import (
        GeneExpressionImputer, ComBat, GeneFeatureSelector, PCAReducer,
    )

    logger.info("Loading GEO expression matrix from %s …", geo_csv)
    expr     = pd.read_csv(geo_csv, index_col=0)   # (n_genes, n_subjects)
    sub_ids  = pheno["SUB_ID"].astype(str).tolist()
    common   = [s for s in sub_ids if s in expr.columns]
    missing  = len(sub_ids) - len(common)
    if missing:
        logger.warning(
            "%d subjects have no GEO match — their genetics vectors will be zero.",
            missing,
        )

    expr_sub = expr[common].T   # (n_common, n_genes)

    logger.info("Imputing missing gene values …")
    imputer      = GeneExpressionImputer(method="knn")
    expr_imp     = imputer.fit_transform(expr_sub)

    pheno_common = pheno[pheno["SUB_ID"].astype(str).isin(common)].copy()
    meta_df      = pheno_common[["SUB_ID", "site"]].rename(
        columns={"site": "batch"}
    ).reset_index(drop=True)
    logger.info(
        "Running ComBat site correction across %d batches …",
        meta_df["batch"].nunique(),
    )
    combat        = ComBat(parametric=True)
    expr_corrected = combat.fit_transform(
        pd.DataFrame(
            expr_imp,
            columns = expr_sub.columns,
            index   = pheno_common["SUB_ID"].astype(str),
        ),
        meta_df, batch_col="batch",
    )

    logger.info("Selecting informative genes …")
    labels   = pheno_common["label"].values
    selector = GeneFeatureSelector(n_top=1000)
    selector.fit(expr_corrected, labels)
    expr_sel = selector.transform(expr_corrected)
    selector.save(save_dir / "genetics_selector.pkl")

    reducer  = PCAReducer(n_components=n_components, random_state=seed)
    X_pca    = reducer.fit_transform(expr_sel.values.astype(np.float32))
    with open(save_dir / "genetics_pca.pkl", "wb") as fh:
        pickle.dump(reducer, fh)
    logger.info(
        "Selector saved → %s  |  PCA saved → %s",
        save_dir / "genetics_selector.pkl",
        save_dir / "genetics_pca.pkl",
    )

    X_full     = np.zeros((len(sub_ids), n_components), dtype=np.float32)
    common_pos = {s: i for i, s in enumerate(common)}
    for j, s in enumerate(sub_ids):
        if s in common_pos:
            X_full[j] = X_pca[common_pos[s]]
    return X_full


def _geo_genetics_apply(
    pheno:            "pd.DataFrame",
    geo_csv:          str,
    selector_path:    str,
    pca_path:         str,
    n_components:     int,
) -> "np.ndarray":
    """
    Apply pre-fitted selector + PCA to held-out ABIDE II GEO expression.

    No re-fitting — statistics from ABIDE I are reused verbatim.
    """
    import pandas as pd
    from preprocessing.genetics import GeneExpressionImputer

    logger.info("Loading GEO expression matrix from %s …", geo_csv)
    expr    = pd.read_csv(geo_csv, index_col=0)
    sub_ids = pheno["SUB_ID"].astype(str).tolist()
    common  = [s for s in sub_ids if s in expr.columns]
    missing = len(sub_ids) - len(common)
    if missing:
        logger.warning(
            "%d subjects have no GEO match — zero vectors used.", missing
        )

    expr_sub = expr[common].T

    imputer  = GeneExpressionImputer(method="knn")
    expr_imp = imputer.fit_transform(expr_sub)

    with open(selector_path, "rb") as fh:
        selector = pickle.load(fh)
    with open(pca_path, "rb") as fh:
        reducer  = pickle.load(fh)

    expr_corrected = pd.DataFrame(
        expr_imp,
        columns = expr_sub.columns,
        index   = pd.Index(common),
    )
    expr_sel = selector.transform(expr_corrected)
    X_pca    = reducer.transform(expr_sel.values.astype(np.float32))

    X_full     = np.zeros((len(sub_ids), n_components), dtype=np.float32)
    common_pos = {s: i for i, s in enumerate(common)}
    for j, s in enumerate(sub_ids):
        if s in common_pos:
            X_full[j] = X_pca[common_pos[s]]
    return X_full


# ===========================================================================
# Step 3 — Genetics pipeline (dispatcher)
# ===========================================================================

def run_genetics_pipeline(
    pheno:         "pd.DataFrame",
    schema:        Dict[str, str],
    out_dir:       Path,
    mode:          str,
    n_components:  int,
    seed:          int,
    role:          str,
    resume:        bool,
    geo_csv:       Optional[str]  = None,
    abide1_selector: Optional[str] = None,
    abide1_pca:    Optional[str]  = None,
) -> None:
    """Save one .npy per subject (genetics feature vector) to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        sub_ids = pheno["SUB_ID"].astype(str).tolist()
        missing = [s for s in sub_ids if not (out_dir / f"{s}.npy").exists()]
        if not missing:
            logger.info("All genetics .npy files exist — skipping.")
            return

    if mode == "phenotypic":
        logger.info("Building phenotypic proxy genetics features …")
        X = _phenotypic_genetics(pheno, schema, n_components, seed)

    else:  # geo
        if not geo_csv:
            logger.error("--geo_csv is required when --genetics_mode geo")
            sys.exit(1)

        if role == "held_out":
            if not abide1_selector or not abide1_pca:
                logger.error(
                    "For held-out GEO genetics, supply both "
                    "--abide1_geo_selector and --abide1_geo_pca "
                    "(from the ABIDE I run)."
                )
                sys.exit(1)
            logger.info("Applying pretrained GEO genetics pipeline (no re-fitting) …")
            X = _geo_genetics_apply(
                pheno          = pheno,
                geo_csv        = geo_csv,
                selector_path  = abide1_selector,
                pca_path       = abide1_pca,
                n_components   = n_components,
            )
        else:
            logger.info("Fitting GEO genetics pipeline on training data …")
            X = _geo_genetics_fit(
                pheno        = pheno,
                geo_csv      = geo_csv,
                n_components = n_components,
                seed         = seed,
                save_dir     = out_dir,
            )

    for i, row in enumerate(pheno.itertuples(index=False)):
        sub_id   = str(row.SUB_ID)
        npy_path = out_dir / f"{sub_id}.npy"
        np.save(npy_path, X[i].astype(np.float32))

    logger.info(
        "Genetics pipeline complete: %d subjects, %d components",
        len(pheno), n_components,
    )


# ===========================================================================
# Step 4 — Write metadata.csv
# ===========================================================================

def write_metadata(
    pheno:      "pd.DataFrame",
    mri_dir:    Path,
    gen_dir:    Path,
    role:       str,
    seed:       int,
) -> None:
    """
    Write ``metadata.csv`` to both mri_dir and gen_dir.

    For the *train* role:   70 % train / 15 % val / 15 % test per site.
    For the *held_out* role: every subject is assigned split = "test".
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    meta = pheno[["SUB_ID", "label", "site"]].copy()
    meta["subject_id"] = meta["SUB_ID"].astype(str)
    meta = meta.drop(columns=["SUB_ID"])

    if role == "held_out":
        meta["split"] = "test"
    else:
        split_map: Dict[int, str] = {}
        for site, grp in meta.groupby("site"):
            idx   = grp.index.tolist()
            strat = grp["label"].values
            if len(idx) < 6:
                for i in idx:
                    split_map[i] = "train"
                continue
            train_idx, rest_idx = train_test_split(
                idx, test_size=0.30, stratify=strat, random_state=seed
            )
            rest_strat = grp.loc[rest_idx, "label"].values
            try:
                val_idx, test_idx = train_test_split(
                    rest_idx, test_size=0.50,
                    stratify=rest_strat, random_state=seed,
                )
            except ValueError:
                half = len(rest_idx) // 2
                val_idx, test_idx = rest_idx[:half], rest_idx[half:]
            for i in train_idx: split_map[i] = "train"
            for i in val_idx:   split_map[i] = "val"
            for i in test_idx:  split_map[i] = "test"

        meta["split"] = meta.index.map(split_map)

    meta = meta[["subject_id", "label", "site", "split"]]
    mri_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(mri_dir / "metadata.csv", index=False)
    meta.to_csv(gen_dir / "metadata.csv", index=False)

    logger.info(
        "metadata.csv: total=%d  ASD=%d  TC=%d  sites=%d",
        len(meta),
        int((meta["label"] == 1).sum()),
        int((meta["label"] == 0).sum()),
        meta["site"].nunique(),
    )
    if role == "train":
        logger.info(
            "  splits → train=%d  val=%d  test=%d",
            int((meta["split"] == "train").sum()),
            int((meta["split"] == "val").sum()),
            int((meta["split"] == "test").sum()),
        )
    else:
        logger.info("  all %d subjects marked split=test (held-out)", len(meta))


# ===========================================================================
# Step 5 — Summary
# ===========================================================================

def print_summary(
    mri_dir:  Path,
    dataset:  str,
    role:     str,
    gen_dir:  Path,
) -> None:
    import pandas as pd
    meta_path = mri_dir / "metadata.csv"
    if not meta_path.exists():
        return
    meta = pd.read_csv(meta_path)

    label = dataset.upper()
    logger.info("\n%s", "=" * 60)
    logger.info("%s Dataset Summary  [role: %s]", label, role)
    logger.info("=" * 60)
    logger.info("Total   : %d", len(meta))
    logger.info("ASD     : %d  (%.1f %%)",
                int((meta["label"] == 1).sum()),
                100 * (meta["label"] == 1).mean())
    logger.info("TC      : %d  (%.1f %%)",
                int((meta["label"] == 0).sum()),
                100 * (meta["label"] == 0).mean())
    logger.info("Sites   : %d", meta["site"].nunique())

    if role == "train":
        logger.info("Train / Val / Test : %d / %d / %d",
                    int((meta["split"] == "train").sum()),
                    int((meta["split"] == "val").sum()),
                    int((meta["split"] == "test").sum()))
    logger.info("-" * 60)
    site_counts = meta.groupby("site").size().sort_values(ascending=False)
    for site, n in site_counts.items():
        asd_n = int((meta.loc[meta["site"] == site, "label"] == 1).sum())
        logger.info("  %-24s  %3d  (ASD=%d  TC=%d)", site, n, asd_n, n - asd_n)
    logger.info("=" * 60)

    if role == "train":
        logger.info(
            "\nPrepare ABIDE II as held-out test set:\n"
            "  python prepare_abide.py \\\n"
            "    --dataset abide2 \\\n"
            "    --role held_out \\\n"
            "    --abide1_dir %s \\\n"
            "    --out_dir datasets/abide2",
            mri_dir.parent.resolve(),
        )
    else:
        logger.info(
            "\nRun experiment with external validation:\n"
            "  python run_experiment.py --real_data \\\n"
            "    --mri_dir %s \\\n"
            "    --gen_dir %s \\\n"
            "    --held_out_mri_dir %s \\\n"
            "    --held_out_gen_dir %s \\\n"
            "    --n_folds 5 --max_epochs 100",
            mri_dir.parent.parent / "abide1" / "mri",
            mri_dir.parent.parent / "abide1" / "genetics",
            mri_dir.resolve(),
            gen_dir.resolve(),
        )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.out_dir)
    mri_dir = out_dir / "mri"
    gen_dir = out_dir / "genetics"
    schema  = _SCHEMA[args.dataset]

    logger.info("=" * 60)
    logger.info("ABIDE Data Preparation  dataset=%s  role=%s",
                args.dataset.upper(), args.role)
    logger.info("Output : %s", out_dir.resolve())
    logger.info("Pipeline: %s  |  genetics: %s", args.pipeline, args.genetics_mode)
    if args.n_subjects:
        logger.info("Subjects: first %d", args.n_subjects)
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # Resolve held-out normalizer path before processing starts
    # ------------------------------------------------------------------ #
    pretrained_normalizer: Optional[Path] = None
    if args.role == "held_out":
        if args.abide1_dir is None:
            logger.warning(
                "--abide1_dir not provided for held_out role. "
                "A fresh normalizer will be fitted on ABIDE II data, "
                "which may introduce leakage if ABIDE II is used for evaluation. "
                "Pass --abide1_dir <path> to suppress this warning."
            )
        else:
            candidate = Path(args.abide1_dir) / "mri" / "intensity_normalizer.pkl"
            if candidate.exists():
                pretrained_normalizer = candidate
                logger.info("Pretrained normalizer: %s", candidate)
            else:
                logger.warning(
                    "intensity_normalizer.pkl not found at %s. "
                    "Fitting a new normalizer on ABIDE II instead.",
                    candidate,
                )

    # ------------------------------------------------------------------ #
    # Step 1 — Download
    # ------------------------------------------------------------------ #
    logger.info("\n--- Downloading ---")
    if args.dataset == "abide1":
        phenotypic, func_files = download_abide1(
            pipeline   = args.pipeline,
            strategy   = args.strategy,
            n_subjects = args.n_subjects,
            data_dir   = args.data_dir,
        )
    else:
        phenotypic, func_files = download_abide2(
            pipeline   = args.pipeline,
            n_subjects = args.n_subjects,
            data_dir   = args.data_dir,
        )

    # ------------------------------------------------------------------ #
    # Step 2 — MRI preprocessing
    # ------------------------------------------------------------------ #
    logger.info("\n--- MRI preprocessing (%d subjects) ---", len(phenotypic))
    pheno_kept = run_mri_pipeline(
        phenotypic            = phenotypic,
        func_files            = func_files,
        out_dir               = mri_dir,
        target_shape          = tuple(args.target_shape),
        voxel_size            = tuple(args.voxel_size),
        strict_qc             = args.qc_strict,
        resume                = args.resume,
        pretrained_normalizer = pretrained_normalizer,
    )

    # ------------------------------------------------------------------ #
    # Step 3 — Genetics
    # ------------------------------------------------------------------ #
    logger.info("\n--- Genetics pipeline (mode=%s) ---", args.genetics_mode)
    run_genetics_pipeline(
        pheno           = pheno_kept,
        schema          = schema,
        out_dir         = gen_dir,
        mode            = args.genetics_mode,
        n_components    = args.n_pca_components,
        seed            = args.seed,
        role            = args.role,
        resume          = args.resume,
        geo_csv         = args.geo_csv,
        abide1_selector = args.abide1_geo_selector,
        abide1_pca      = args.abide1_geo_pca,
    )

    # ------------------------------------------------------------------ #
    # Step 4 — Metadata
    # ------------------------------------------------------------------ #
    logger.info("\n--- Writing metadata ---")
    write_metadata(
        pheno   = pheno_kept,
        mri_dir = mri_dir,
        gen_dir = gen_dir,
        role    = args.role,
        seed    = args.seed,
    )

    # ------------------------------------------------------------------ #
    # Step 5 — Summary + next-step hint
    # ------------------------------------------------------------------ #
    print_summary(mri_dir, args.dataset, args.role, gen_dir)


if __name__ == "__main__":
    main()
