"""
Preprocess ABIDE I ROI timeseries → .npy files for the ASD framework.

Input (from download_abide.py, CC200 atlas):
    abide_raw/ABIDE_pcp/cpac/filt_global/rois_cc200/
        <file_id>_rois_cc200.1D    # T timepoints × 200 ROIs

Output:
    abide_processed/
        mri/
            metadata.csv           # subject_id, label, site, split
            50001.npy              # (19900,) FC upper-triangle vector
            50002.npy
            ...
        gen/
            metadata.csv           # same columns, same rows
            50001.npy              # (6,) phenotypic vector
        scaler.pkl                 # RobustScaler fitted on train set only
        preprocessing_log.json     # summary + per-subject QC

Branch mapping
--------------
MRI branch  (framework "image" input):
    200×200 Pearson correlation matrix
    → Fisher z-transform (arctanh)
    → upper triangle: 200×199//2 = 19,900 values
    → saved as float32 (19900,) .npy

    NOTE: The 3D ResNet branch expects (B, 1, D, H, W).
    Options for the DataLoader:
      a) Reshape to (1, 141, 141, 1) padded — nearest 2D square approach
      b) Reshape to (1, 1, 100, 199, 1) — keep interpretable ROI structure
      c) Replace MRI branch with an MLP/Transformer — best accuracy
    We save the flat (19900,) vector so any option is available later.

Genetics branch (framework "genetics" input):
    [age, sex_bin, FIQ, VIQ, PIQ, handedness] → (6,) float32

Label convention:
    DX_GROUP=1 → ASD → label=1
    DX_GROUP=2 → TC  → label=0

Usage
-----
    # Full preprocessing (~3 GB download required first):
    python data/preprocess_abide.py \\
        --raw_dir ./abide_raw \\
        --out_dir ./abide_processed \\
        --atlas rois_cc200 \\
        --n_jobs 4

    # Verify already-saved files:
    python data/preprocess_abide.py --verify_only --out_dir ./abide_processed

    # Re-run only missing subjects (resume after interruption):
    python data/preprocess_abide.py \\
        --raw_dir ./abide_raw --out_dir ./abide_processed --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ROIS        = 200                             # Craddock CC200 atlas
N_UPPER       = N_ROIS * (N_ROIS - 1) // 2     # 19900 upper-triangle elements
MIN_TIMEPOINTS = 50                             # discard very short scans
ABIDE_MISSING  = -9999.0                        # ABIDE sentinel for missing data


# ---------------------------------------------------------------------------
# Functional connectivity helpers
# ---------------------------------------------------------------------------

def _load_timeseries(path: Path) -> np.ndarray:
    """Load .1D file → float32 array of shape (T, N_ROIS)."""
    ts = np.loadtxt(str(path), dtype=np.float32)
    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)
    return ts


def _fc_vector(ts: np.ndarray) -> np.ndarray:
    """
    Compute Fisher z-transformed Pearson correlation, return upper triangle.

    Steps:
      1. Mean-centre each ROI timeseries column
      2. Pearson correlation → (N_ROIS, N_ROIS) in [-1, 1]
      3. Clip to [-0.999, 0.999] (avoids arctanh = ±∞ on diagonal/perfect corr)
      4. Fisher z-transform: z = arctanh(r) — stabilises variance across sites
      5. Extract upper triangle (k=1 skips diagonal): 19,900 values

    Returns
    -------
    vec : np.ndarray, shape (19900,), float32
    """
    ts = ts - ts.mean(axis=0, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = np.corrcoef(ts.T).astype(np.float32)
    r = np.clip(r, -0.999, 0.999)
    z = np.arctanh(r)
    idx = np.triu_indices(N_ROIS, k=1)
    return z[idx]                               # (19900,)


# ---------------------------------------------------------------------------
# Phenotypic feature extraction
# ---------------------------------------------------------------------------

_HANDEDNESS_MAP: Dict[str, float] = {
    "R": 1.0, "Right": 1.0,
    "L": -1.0, "Left": -1.0,
    "Ambi": 0.0, "Ambidextrous": 0.0,
    "Mixed": 0.0,
}


def _extract_pheno(row: pd.Series) -> np.ndarray:
    """
    Extract 6 phenotypic features from one phenotypic CSV row.

    Layout:
      [0] age         — AGE_AT_SCAN (years, float)
      [1] sex         — 0=Male, 1=Female  (remapped from ABIDE 1=M/2=F)
      [2] FIQ         — Full-scale IQ (NaN if -9999)
      [3] VIQ         — Verbal IQ (NaN if -9999)
      [4] PIQ         — Performance IQ (NaN if -9999)
      [5] handedness  — 1=R, -1=L, 0=Ambi/missing (from HANDEDNESS_CATEGORY)
    """
    def _val(col: str) -> float:
        v = row.get(col, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return np.nan
        return np.nan if v == ABIDE_MISSING else v

    age = _val("AGE_AT_SCAN")
    sex_raw = _val("SEX")
    sex = np.nan if np.isnan(sex_raw) else float(sex_raw == 2)  # 1=M→0, 2=F→1
    fiq = _val("FIQ")
    viq = _val("VIQ")
    piq = _val("PIQ")

    # Handedness: prefer HANDEDNESS_CATEGORY string, fall back to numeric score
    hand_cat = str(row.get("HANDEDNESS_CATEGORY", "")).strip()
    if hand_cat in _HANDEDNESS_MAP:
        hand = _HANDEDNESS_MAP[hand_cat]
    else:
        hand_score = _val("HANDEDNESS_SCORES")
        if not np.isnan(hand_score):
            # ABIDE scores: positive = right, negative = left
            hand = np.sign(hand_score)
        else:
            hand = np.nan

    return np.array([age, sex, fiq, viq, piq, hand], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-subject processing worker (runs in child process)
# ---------------------------------------------------------------------------

def _process_one(args: Tuple[str, Path, pd.Series]) -> Dict:
    """
    Compute FC vector and phenotypic features for one subject.

    Returns a result dict; caller checks result["error"] is None before saving.
    """
    subject_id, ts_path, pheno_row = args
    out: Dict = {
        "subject_id": subject_id,
        "n_timepoints": None,
        "n_rois": None,
        "fc": None,
        "pheno": None,
        "error": None,
    }
    try:
        ts = _load_timeseries(ts_path)
        out["n_timepoints"] = ts.shape[0]
        out["n_rois"]       = ts.shape[1]

        if ts.shape[0] < MIN_TIMEPOINTS:
            out["error"] = f"only {ts.shape[0]} timepoints (< {MIN_TIMEPOINTS})"
            return out

        if ts.shape[1] != N_ROIS:
            out["error"] = f"expected {N_ROIS} ROIs, got {ts.shape[1]}"
            return out

        # Warn if many ROIs are scrubbed to zero
        zero_rois = int(np.all(ts == 0, axis=0).sum())
        if zero_rois > N_ROIS // 5:
            out["error"] = f"{zero_rois} ROIs all-zero (excessive scrubbing)"
            return out

        out["fc"]    = _fc_vector(ts)
        out["pheno"] = _extract_pheno(pheno_row)

    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"

    return out


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def _stratified_site_split(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac:  float = 0.15,
    seed:      int   = 42,
) -> pd.DataFrame:
    """
    Assign train/val/test splits using StratifiedGroupKFold with groups=site.

    This guarantees:
    - Each site contributes subjects to all three splits (no site-exclusive split)
    - Class ratio (ASD/TC) is approximately balanced across splits
    """
    df = df.copy()
    df["split"] = "train"
    labels = df["label"].values
    sites  = df["site"].values
    idx    = np.arange(len(df))

    n_test_folds = max(2, round(1.0 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True,
                                 random_state=seed)
    _, test_idx = next(sgkf.split(idx, labels, groups=sites))
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    # Val from remaining train pool
    train_df = df[df["split"] == "train"]
    n_val_folds = max(2, round((1 - test_frac) / val_frac))
    sgkf2 = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True,
                                  random_state=seed + 1)
    tp_idx    = train_df.index.values
    tp_labels = train_df["label"].values
    tp_sites  = train_df["site"].values
    _, val_rel = next(sgkf2.split(tp_idx, tp_labels, groups=tp_sites))
    df.loc[tp_idx[val_rel], "split"] = "val"

    return df


# ---------------------------------------------------------------------------
# Phenotypic imputation and scaling
# ---------------------------------------------------------------------------

def _impute_site_mean(
    arr: np.ndarray,
    sites: np.ndarray,
    train_mask: np.ndarray,
    n_scalar: int = 6,
) -> np.ndarray:
    """
    Replace NaN in scalar columns with per-site mean (computed on train only).
    Falls back to global train mean if a site has no valid values for a column.
    """
    arr = arr.copy()
    for col in range(n_scalar):
        col_vals = arr[:, col]
        global_mean = float(np.nanmean(col_vals[train_mask]))
        if np.isnan(global_mean):
            global_mean = 0.0

        unique_sites = np.unique(sites[train_mask])
        site_means: Dict[str, float] = {}
        for s in unique_sites:
            mask_s = train_mask & (sites == s)
            m = float(np.nanmean(col_vals[mask_s]))
            site_means[s] = m if not np.isnan(m) else global_mean

        nan_rows = np.where(np.isnan(arr[:, col]))[0]
        for i in nan_rows:
            arr[i, col] = site_means.get(sites[i], global_mean)

    return arr


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _load_pheno_and_ts_manual(
    pheno_csv: Path,
    ts_dir: Path,
    atlas: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load phenotypic CSV and match .1D timeseries files for manual downloads.

    Supports the layout produced by:
        aws s3 sync s3://fcp-indi/.../rois_cc200/ ./abide_raw/timeseries/

    File naming convention: <SITE>_<SUB_ID>_rois_cc200.1D
    """
    pheno = pd.read_csv(str(pheno_csv))
    suffix = f"_{atlas}.1D"
    ts_files: List[str] = []
    missing: List[str] = []

    for _, row in pheno.iterrows():
        sub_id = str(int(row["SUB_ID"]))
        site   = str(row["SITE_ID"]).strip()

        # Filenames use site-specific capitalisation, e.g. NYU_0050952_rois_cc200.1D
        # Try exact match first, then case-insensitive glob
        candidate = ts_dir / f"{site}_{sub_id}{suffix}"
        if candidate.exists():
            ts_files.append(str(candidate))
            continue

        # Glob fallback: site name in filename may differ slightly
        matches = list(ts_dir.glob(f"*{sub_id}*{suffix}"))
        if matches:
            ts_files.append(str(matches[0]))
        else:
            ts_files.append("")   # placeholder; filtered out in work-list step
            missing.append(sub_id)

    if missing:
        logger.warning(
            "%d subjects have no matching .1D file: %s%s",
            len(missing), missing[:5], " …" if len(missing) > 5 else "",
        )

    return pheno, ts_files


def preprocess(
    raw_dir:   Path,
    out_dir:   Path,
    pipeline:  str  = "cpac",
    atlas:     str  = "rois_cc200",
    n_jobs:    int  = 4,
    resume:    bool = False,
    pheno_csv: Optional[Path] = None,
    ts_dir:    Optional[Path] = None,
) -> None:
    """
    End-to-end preprocessing: load ABIDE → FC vectors + phenotypics → splits.

    Two modes:
      nilearn mode (default): raw_dir points to nilearn cache root.
      manual mode: pass pheno_csv + ts_dir for AWS CLI downloads.
    """
    out_dir  = Path(out_dir)
    mri_dir  = out_dir / "mri"
    gen_dir  = out_dir / "gen"
    mri_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load phenotypic data and timeseries file list
    # ------------------------------------------------------------------
    if pheno_csv is not None and ts_dir is not None:
        # Manual download mode (AWS CLI layout)
        logger.info("Manual mode: reading %s", pheno_csv)
        logger.info("             timeseries from %s", ts_dir)
        pheno, ts_files = _load_pheno_and_ts_manual(
            Path(pheno_csv), Path(ts_dir), atlas
        )
    else:
        # nilearn mode
        logger.info("nilearn mode: loading ABIDE I from %s …", raw_dir)
        from nilearn import datasets

        abide = datasets.fetch_abide_pcp(
            data_dir              = str(raw_dir),
            pipeline              = pipeline,
            band_pass_filtering   = True,
            global_signal_regression = False,
            derivatives           = [atlas],
            verbose               = 0,
        )
        _pheno_raw = abide.phenotypic
        pheno: pd.DataFrame = (
            _pheno_raw if isinstance(_pheno_raw, pd.DataFrame)
            else pd.DataFrame(_pheno_raw)
        ).copy()
        ts_files: List[str] = abide[atlas]

    logger.info("  Subjects in phenotypic CSV : %d", len(pheno))
    logger.info("  Timeseries files matched   : %d",
                sum(1 for f in ts_files if f))

    # ------------------------------------------------------------------
    # 2. Build work list — skip subjects already saved (if --resume)
    # ------------------------------------------------------------------
    subject_ids = [str(int(r["SUB_ID"])) for _, r in pheno.iterrows()]

    work: List[Tuple[str, Path, pd.Series]] = []
    for sid, ts_path, (_, row) in zip(subject_ids, ts_files, pheno.iterrows()):
        if not ts_path:
            continue                            # no file found for this subject
        fc_out   = mri_dir / f"{sid}.npy"
        gen_out  = gen_dir / f"{sid}_raw.npy"
        if resume and fc_out.exists() and gen_out.exists():
            continue
        work.append((sid, Path(ts_path), row))

    logger.info("Subjects to process: %d (already on disk: %d)",
                len(work), len(subject_ids) - len(work))

    # ------------------------------------------------------------------
    # 3. Parallel FC computation
    # ------------------------------------------------------------------
    qc: List[Dict] = []
    ok_ids: List[str] = []

    if work:
        if n_jobs == 1:
            results = [_process_one(a) for a in work]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = {pool.submit(_process_one, a): a[0] for a in work}
                results = [f.result() for f in as_completed(futures)]

        n_failed = 0
        for res in results:
            sid = res["subject_id"]
            if res["error"] is not None:
                logger.warning("  SKIP %s: %s", sid, res["error"])
                qc.append({"subject_id": sid, "status": "failed",
                           "reason": res["error"]})
                n_failed += 1
                continue
            np.save(str(mri_dir / f"{sid}.npy"),      res["fc"])    # (19900,)
            np.save(str(gen_dir  / f"{sid}_raw.npy"), res["pheno"]) # (6,) — pre-scale
            qc.append({"subject_id": sid, "status": "ok",
                       "n_timepoints": res["n_timepoints"]})
            ok_ids.append(sid)

        logger.info("Passed: %d  |  Failed: %d", len(ok_ids), n_failed)

    # All usable subjects on disk (including from prior resume runs)
    all_ok = sorted(
        p.stem for p in mri_dir.glob("*.npy")
        if (gen_dir / f"{p.stem}_raw.npy").exists()
    )
    logger.info("Total usable subjects on disk: %d", len(all_ok))
    if len(all_ok) < 50:
        raise RuntimeError(
            f"Only {len(all_ok)} subjects — check preprocessing_log.json."
        )

    # ------------------------------------------------------------------
    # 4. Build metadata
    # ------------------------------------------------------------------
    pheno_idx = {str(int(r["SUB_ID"])): r for _, r in pheno.iterrows()}
    rows = []
    for sid in all_ok:
        row = pheno_idx.get(sid)
        if row is None:
            logger.warning("  %s: no phenotypic entry — skipping", sid)
            continue
        dx    = int(row["DX_GROUP"])
        label = 1 if dx == 1 else 0
        site  = str(row["SITE_ID"]).strip()
        rows.append({"subject_id": sid, "label": label, "site": site})

    meta = pd.DataFrame(rows)
    logger.info("ASD: %d  |  TC: %d  |  Sites: %d",
                (meta["label"] == 1).sum(),
                (meta["label"] == 0).sum(),
                meta["site"].nunique())

    # ------------------------------------------------------------------
    # 5. Site-stratified train/val/test split
    # ------------------------------------------------------------------
    meta = _stratified_site_split(meta, test_frac=0.15, val_frac=0.15, seed=42)
    for spl in ["train", "val", "test"]:
        m     = meta[meta["split"] == spl]
        n_asd = (m["label"] == 1).sum()
        n_tc  = (m["label"] == 0).sum()
        logger.info("  %s : n=%d  ASD=%d  TC=%d", spl, len(m), n_asd, n_tc)
        if n_asd == 0 or n_tc == 0:
            raise RuntimeError(f"Split '{spl}' has only one class — adjust split fractions.")

    # ------------------------------------------------------------------
    # 6. Phenotypic imputation + scaling (fit on train only)
    # ------------------------------------------------------------------
    logger.info("Fitting RobustScaler on training phenotypics …")
    ordered_ids  = meta["subject_id"].values
    site_arr     = meta["site"].values
    train_mask   = (meta["split"] == "train").values

    raw_pheno = np.stack([
        np.load(str(gen_dir / f"{sid}_raw.npy")) for sid in ordered_ids
    ])  # (N, 6)

    # Site-mean imputation (NaN → per-site train mean)
    raw_pheno = _impute_site_mean(raw_pheno, site_arr, train_mask, n_scalar=6)

    # RobustScaler: robust to outliers (IQ scores have long tails)
    scaler = RobustScaler()
    raw_pheno = scaler.fit(raw_pheno[train_mask]).transform(raw_pheno)

    # Save scaler for inference
    scaler_path = out_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Scaler saved → %s", scaler_path)

    # Save final scaled .npy per subject (overwrites nothing; _raw stays)
    for i, sid in enumerate(ordered_ids):
        np.save(str(gen_dir / f"{sid}.npy"), raw_pheno[i])  # (6,) float32

    # ------------------------------------------------------------------
    # 7. Write metadata.csv into both mri/ and gen/
    # ------------------------------------------------------------------
    for folder in (mri_dir, gen_dir):
        meta.to_csv(folder / "metadata.csv", index=False)
    logger.info("metadata.csv written to mri/ and gen/")

    # ------------------------------------------------------------------
    # 8. Quick validation sample
    # ------------------------------------------------------------------
    _validate_sample(meta, mri_dir, gen_dir, n=min(30, len(meta)))

    # ------------------------------------------------------------------
    # 9. Save preprocessing log
    # ------------------------------------------------------------------
    log = {
        "n_pheno_rows": len(pheno),
        "n_usable":     len(all_ok),
        "n_train":      int(train_mask.sum()),
        "n_val":        int((meta["split"] == "val").sum()),
        "n_test":       int((meta["split"] == "test").sum()),
        "n_asd":        int((meta["label"] == 1).sum()),
        "n_tc":         int((meta["label"] == 0).sum()),
        "n_sites":      int(meta["site"].nunique()),
        "fc_vector_dim": N_UPPER,           # 19900
        "pheno_dim":     6,
        "atlas":         atlas,
        "pipeline":      pipeline,
        "per_subject":   qc,
    }
    log_path = out_dir / "preprocessing_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    logger.info("Log → %s", log_path)
    logger.info("Phase 2 preprocessing complete.")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Update configs/config.yaml (see instructions below)")
    logger.info("  2. Implement a DataLoader that reshapes (19900,) → model input")
    logger.info("  3. Run: python run_experiment.py --config configs/config.yaml")


# ---------------------------------------------------------------------------
# Verification pass
# ---------------------------------------------------------------------------

def verify(out_dir: Path) -> None:
    out_dir  = Path(out_dir)
    mri_dir  = out_dir / "mri"
    gen_dir  = out_dir / "gen"
    meta     = pd.read_csv(mri_dir / "metadata.csv")

    logger.info("Verifying %d subjects …", len(meta))
    errors: List[str] = []
    for _, row in meta.iterrows():
        sid    = str(row["subject_id"])
        fc_p   = mri_dir / f"{sid}.npy"
        gen_p  = gen_dir  / f"{sid}.npy"

        if not fc_p.exists():
            errors.append(f"{sid}: missing mri/{sid}.npy")
            continue
        if not gen_p.exists():
            errors.append(f"{sid}: missing gen/{sid}.npy")
            continue

        fc  = np.load(str(fc_p))
        gen = np.load(str(gen_p))

        if fc.shape != (N_UPPER,):
            errors.append(f"{sid}: FC shape {fc.shape}, expected ({N_UPPER},)")
        if not np.isfinite(fc).all():
            errors.append(f"{sid}: FC has non-finite values")
        if gen.shape != (6,):
            errors.append(f"{sid}: pheno shape {gen.shape}, expected (6,)")
        if not np.isfinite(gen).all():
            errors.append(f"{sid}: pheno has non-finite values")

    if errors:
        for e in errors[:25]:
            logger.error("  %s", e)
        if len(errors) > 25:
            logger.error("  … and %d more", len(errors) - 25)
        sys.exit(f"Verification failed: {len(errors)} errors.")

    logger.info("All %d subjects OK.", len(meta))
    split_counts = meta["split"].value_counts().to_dict()
    logger.info("  FC dim   : %d", N_UPPER)
    logger.info("  Pheno dim: 6")
    logger.info("  Splits   : %s", split_counts)


def _validate_sample(
    meta: pd.DataFrame,
    mri_dir: Path,
    gen_dir: Path,
    n: int = 30,
) -> None:
    sample = meta.sample(n=n, random_state=0)
    bad = 0
    for _, row in sample.iterrows():
        sid = str(row["subject_id"])
        try:
            fc  = np.load(str(mri_dir / f"{sid}.npy"))
            gen = np.load(str(gen_dir  / f"{sid}.npy"))
            assert fc.shape  == (N_UPPER,), f"FC shape {fc.shape}"
            assert gen.shape == (6,),       f"pheno shape {gen.shape}"
            assert np.isfinite(fc).all(),   "non-finite FC"
            assert np.isfinite(gen).all(),  "non-finite pheno"
        except Exception as e:
            logger.error("Validation failed for %s: %s", sid, e)
            bad += 1
    if bad:
        raise RuntimeError(f"{bad}/{n} sample subjects failed validation.")
    logger.info("Sample validation passed (%d subjects).", n)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Preprocess ABIDE I ROI timeseries → FC vectors + phenotypics"
    )
    p.add_argument("--raw_dir",  default="./abide_raw",
                   help="Folder where download_abide.py saved the data")
    p.add_argument("--out_dir",  default="./abide_processed",
                   help="Output folder for processed .npy files")
    p.add_argument("--pipeline", default="cpac", choices=["cpac", "dparsf"])
    p.add_argument("--atlas",    default="rois_cc200",
                   choices=["rois_cc200", "rois_ho", "rois_aal"])
    p.add_argument("--n_jobs",   type=int, default=4,
                   help="Parallel workers for FC computation")
    p.add_argument("--verify_only", action="store_true",
                   help="Check saved files without reprocessing")
    p.add_argument("--resume", action="store_true",
                   help="Skip subjects already saved to disk")
    # Manual download mode (AWS CLI layout)
    p.add_argument("--pheno_csv", default=None,
                   help="Path to Phenotypic_V1_0b_preprocessed1.csv "
                        "(manual download mode)")
    p.add_argument("--ts_dir", default=None,
                   help="Folder containing *_rois_cc200.1D files "
                        "(manual download mode)")
    args = p.parse_args()

    if args.verify_only:
        verify(Path(args.out_dir))
    else:
        preprocess(
            raw_dir   = Path(args.raw_dir),
            out_dir   = Path(args.out_dir),
            pipeline  = args.pipeline,
            atlas     = args.atlas,
            n_jobs    = args.n_jobs,
            resume    = args.resume,
            pheno_csv = Path(args.pheno_csv) if args.pheno_csv else None,
            ts_dir    = Path(args.ts_dir)    if args.ts_dir    else None,
        )


if __name__ == "__main__":
    main()
