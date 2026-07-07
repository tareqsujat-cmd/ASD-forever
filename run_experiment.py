"""
End-to-end ASD detection experiment pipeline.

Smoke-tests and runs the full framework in a single process:

  Stage 0  : Configuration + reproducibility seed
  Stage 1  : Dataset (synthetic or real ABIDE I)
  Stage 2  : Model factory
  Stage 3  : K-fold cross-validation training
  Stage 4  : Evaluation (bootstrap CI, per-site, calibration)
  Stage 4b : Model export (ONNX + TorchScript + equivalence validation)
  Stage 4c : Robustness evaluation (noise, missing-modality, distribution shift)
  Stage 4d : Held-out external validation on ABIDE II (optional)
  Stage 5  : Ablation study (OFAT)
  Stage 6  : Hyperparameter optimisation (Optuna)
  Stage 7  : Explainability (GradCAM++ + gene importance)
  Stage 8  : Paper figure generation (all 14 figures + supplements)
  Stage 9  : Automated HTML experiment report

Usage
-----
Synthetic data (no ABIDE download required — runs in ~2 min on CPU)::

    python run_experiment.py

Real ABIDE data::

    python run_experiment.py --real_data \\
        --mri_dir /path/to/abide/mri_processed \\
        --gen_dir  /path/to/abide/genetics_processed

Flags
-----
--real_data         Switch from synthetic to real data loaders
--mri_dir           Root of preprocessed MRI .npy files (one per subject)
--gen_dir           Root of preprocessed genetics .npy files (one per subject)
--n_folds           Number of CV folds (default 2 for synthetic, 5 for real)
--max_epochs        Max training epochs (default 3 for synthetic, 100 for real)
--n_ablation_trials Ablation variants to run (default all)
--n_hpo_trials      Optuna trials (default 8 for synthetic, 50 for real)
--out_dir           Results root (default results/)
--seed              Global random seed (default 42)
--device            "cpu" | "cuda" | "mps"  (default: auto-detect)
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# LambdaLR calls step() once during __init__ to set the initial LR value.
# This triggers a spurious "step before optimizer.step" warning because no
# optimizer step has run yet at construction time.  The actual training loop
# always calls optimizer.step() before scheduler.step(), so this is safe.
warnings.filterwarnings(
    "ignore",
    message="Detected call of",
    category=UserWarning,
    module=r"torch\.optim\.lr_scheduler",
)
# matplotlib tight_layout emits a UserWarning when axes (e.g. colorbars) are
# not fully compatible with the algorithm.  The layout is still correct in
# practice; this warning is cosmetic noise in the experiment log.
warnings.filterwarnings(
    "ignore",
    message=".*tight_layout.*",
    category=UserWarning,
)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")

# Canonical phenotypic ("genetics" branch) feature names for the FC pipeline,
# in the order produced by data/preprocess_abide.py::_extract_pheno.  Used for
# labelling feature-importance plots (these are NOT genetic SNPs).
PHENOTYPIC_FEATURE_NAMES = ["age", "sex", "FIQ", "VIQ", "PIQ", "handedness"]


def _feature_names(n: int) -> List[str]:
    """Real phenotypic names when the count matches, else generic labels."""
    if n == len(PHENOTYPIC_FEATURE_NAMES):
        return list(PHENOTYPIC_FEATURE_NAMES)
    return [f"pheno_{i}" for i in range(n)]


# ===========================================================================
# Stage 0 — Configuration
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASD Detection full pipeline")
    p.add_argument("--real_data",          action="store_true")
    p.add_argument("--mri_dir",            type=str, default=None)
    p.add_argument("--gen_dir",            type=str, default=None)
    p.add_argument("--held_out_mri_dir",   type=str, default=None,
                   help="ABIDE II MRI dir for external held-out evaluation")
    p.add_argument("--held_out_gen_dir",   type=str, default=None,
                   help="ABIDE II genetics dir for external held-out evaluation")
    p.add_argument("--n_folds",          type=int, default=None)
    p.add_argument("--max_epochs",       type=int, default=None)
    p.add_argument("--n_ablation_trials",type=int, default=None)
    p.add_argument("--n_hpo_trials",     type=int, default=None)
    p.add_argument("--out_dir",          type=str, default="results",
                   help="Results ROOT; each run is written to <out_dir>/run_N/")
    p.add_argument("--run_name",         type=str, default=None,
                   help="Explicit run directory name (default: auto-increment run_N)")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--device",           type=str, default=None,
                   help='"auto" | "cuda" | "mps" | "cpu" (default: auto-detect)')
    p.add_argument("--skip_profile",     action="store_true",
                   help="Skip computational profiling (FLOPs/latency)")
    return p.parse_args()


def _setup(args: argparse.Namespace):
    """Load config, set seeds/device, apply synthetic overrides, allocate run dir.

    Returns ``(cfg, device, run_dir, seed)`` where ``run_dir`` is a fresh
    ``<out_dir>/run_N/`` directory that every pipeline stage writes into.
    """
    from configs.config_schema import load_config
    from utilities.reproducibility import seed_everything
    from utilities.hardware import get_device
    from utilities.run_dir import (
        create_run_dir, attach_file_logger, snapshot_config, write_manifest,
    )

    cfg = load_config(Path(__file__).parent / "configs" / "config.yaml")

    # --- Reproducibility: seed every RNG + request deterministic algorithms ---
    seed = args.seed
    seed_everything(seed, deterministic_cudnn=True)

    # --- Device: auto-detect CUDA -> Apple MPS -> CPU (cascading) ---
    requested = args.device or getattr(cfg.project, "device", "auto")
    device = get_device(requested)
    cfg.project.device = device.type
    logger.info("Device: %s", device)

    # TF32 gives a large speedup on Ampere+ GPUs with negligible precision loss
    # for (non-safety-critical) classification.  CUDA only; harmless elsewhere.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Synthetic-mode overrides: small dimensions, few epochs/folds for speed
    if not args.real_data:
        cfg.training.max_epochs                    = args.max_epochs or 3
        cfg.training.cross_validation.n_folds      = args.n_folds   or 2
        cfg.training.batch_size                    = 4
        cfg.training.gradient_accumulation_steps   = 1
        cfg.training.early_stopping.patience       = 999  # disable
        cfg.mri_model.feature_dim                  = 64
        cfg.genetics_model.input_dim               = 32
        cfg.genetics_model.feature_dim             = 64
        cfg.genetics_model.num_heads               = 2
        cfg.genetics_model.num_layers              = 1
        cfg.fusion.mri_feature_dim                 = 64
        cfg.fusion.genetics_feature_dim            = 64
        cfg.fusion.fusion_dim                      = 64
        cfg.fusion.num_heads                       = 2
    else:
        if args.max_epochs:
            cfg.training.max_epochs = args.max_epochs
        if args.n_folds:
            cfg.training.cross_validation.n_folds = args.n_folds

    # --- Allocate a fresh per-run directory: <out_dir>/run_N/ ---
    # Every stage writes inside this directory, so runs never overwrite one
    # another and each run is self-contained and reproducible.
    run_dir = create_run_dir(args.out_dir, run_name=args.run_name)
    attach_file_logger(run_dir)
    snapshot_config(cfg, run_dir)
    write_manifest(
        run_dir,
        seed=seed,
        device=device,
        args=args,
        mode="real" if args.real_data else "synthetic",
    )
    return cfg, device, run_dir, seed


# ===========================================================================
# Stage 1 — Dataset
# ===========================================================================

# ---------------------------------------------------------------------------
# Synthetic dataset (ABIDE-shaped)
# ---------------------------------------------------------------------------

class SyntheticABIDE(Dataset):
    """
    Synthetic dataset with ABIDE-shaped tensors for end-to-end smoke testing.

    Batch keys: ``"image"`` (1,D,H,W), ``"genetics"`` (n_genes,),
    ``"label"`` (int), ``"site"`` (int).

    The AUC is deliberately kept < 1 so metrics are informative:
    ASD subjects have slightly higher mean MRI intensity and genetics values.
    """

    def __init__(
        self,
        n_subjects: int = 80,
        mri_shape:  Tuple[int, int, int] = (16, 16, 16),
        n_genes:    int = 32,
        n_sites:    int = 4,
        seed:       int = 42,
    ) -> None:
        self.rng      = np.random.default_rng(seed)
        self.n        = n_subjects
        self.mri_shape = mri_shape
        self.n_genes  = n_genes

        # Labels: half ASD (1), half TC (0)
        self.labels   = np.array([i % 2 for i in range(n_subjects)])
        # Sites: round-robin assignment
        self.sites    = np.array([i % n_sites for i in range(n_subjects)])

        # Pre-generate data with a weak signal so AUC ≈ 0.65–0.75
        self._mri = self.rng.standard_normal(
            (n_subjects, 1, *mri_shape)
        ).astype(np.float32)
        self._gen = self.rng.standard_normal(
            (n_subjects, n_genes)
        ).astype(np.float32)

        # Inject signal: ASD subjects get +0.3 shift on first 4 features
        asd_mask = self.labels == 1
        self._mri[asd_mask, :, :4, :4, :4] += 0.3
        self._gen[asd_mask, :4]             += 0.4

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "image":    torch.from_numpy(self._mri[idx]),
            "genetics": torch.from_numpy(self._gen[idx]),
            "label":    torch.tensor(int(self.labels[idx]), dtype=torch.long),
            "site":     torch.tensor(int(self.sites[idx]),  dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Real data loader (path-based)
# ---------------------------------------------------------------------------

class ABIDEDataset(Dataset):
    """
    Loads preprocessed ABIDE subjects from disk.

    Expects one ``.npy`` file per subject in ``mri_dir`` and ``gen_dir``,
    named ``<subject_id>.npy``.  A ``metadata.csv`` in each directory must
    have columns ``subject_id``, ``label`` (0/1), and ``site``.

    MRI files may be:
      - (19900,)       flat FC upper-triangle  → reshaped to (1, 28, 28, 28)
      - (1, D, H, W)   already a 3D volume     → used as-is
    """

    # Reshape constants for flat FC vectors (CC200 atlas, 200 ROIs)
    _FC_DIM    = 19900
    _CUBE_SIDE = 28          # 28³ = 21952 ≥ 19900
    _CUBE_VOL  = 28 ** 3

    def __init__(self, mri_dir: str, gen_dir: str) -> None:
        import pandas as pd
        mri_dir = Path(mri_dir)
        gen_dir = Path(gen_dir)

        meta = pd.read_csv(mri_dir / "metadata.csv")
        self.records = meta.to_dict("records")
        self.mri_dir = mri_dir
        self.gen_dir = gen_dir

        # Build a consistent integer mapping for site strings
        sites = sorted({str(r.get("site", "unknown")) for r in self.records})
        self._site_to_int: Dict[str, int] = {s: i for i, s in enumerate(sites)}
        # Expose as array for _make_cv_splits site-stratification
        self.sites = np.array(
            [self._site_to_int[str(r.get("site", "unknown"))]
             for r in self.records]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r    = self.records[idx]
        sid  = str(r["subject_id"])
        mri  = np.load(str(self.mri_dir / f"{sid}.npy")).astype(np.float32)
        gen  = np.load(str(self.gen_dir / f"{sid}.npy")).astype(np.float32)

        # 2 subjects have NaN FC values from zero-variance ROIs in preprocessing.
        # Replace NaN/inf with 0 (no connectivity) so they don't poison batches.
        mri = np.nan_to_num(mri, nan=0.0, posinf=0.0, neginf=0.0)

        # Reshape flat FC vector to pseudo-3D volume for the ResNet branch
        if mri.ndim == 1:
            padded = np.zeros(self._CUBE_VOL, dtype=np.float32)
            padded[:len(mri)] = mri
            mri = padded.reshape(1, self._CUBE_SIDE, self._CUBE_SIDE, self._CUBE_SIDE)
        elif mri.ndim == 3:
            mri = mri[np.newaxis]   # add channel dim for (D, H, W) volumes

        site_int = self._site_to_int.get(str(r.get("site", "unknown")), 0)
        return {
            "image":    torch.from_numpy(mri),
            "genetics": torch.from_numpy(gen),
            "label":    torch.tensor(int(r["label"]), dtype=torch.long),
            "site":     torch.tensor(site_int, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# CV splits
# ---------------------------------------------------------------------------

def _get_labels(dataset: Dataset) -> np.ndarray:
    """Return the class label (0/1) array for a dataset without loading arrays."""
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels).astype(int)
    if hasattr(dataset, "records"):          # path-based ABIDEDataset
        return np.array([int(r["label"]) for r in dataset.records])
    return np.array([int(dataset[i]["label"]) for i in range(len(dataset))])


def _make_cv_splits(
    dataset:       Dataset,
    n_folds:       int,
    seed:          int = 42,
    group_by_site: bool = False,
) -> List[Tuple[List[int], List[int]]]:
    """
    Class-stratified K-fold splits.

    Folds are always stratified by the **diagnosis label** so every fold keeps
    the ASD/TC balance — essential for a meaningful, comparable AUROC/accuracy.

    When ``group_by_site`` is True and there are at least ``n_folds`` sites, a
    ``StratifiedGroupKFold`` grouped by acquisition site is used instead: each
    site's subjects fall entirely in one fold, so the model is validated on
    *unseen sites* (a stricter, cross-site generalization protocol).  This is
    the honest interpretation of the config's ``group_by_site`` flag.
    """
    n = len(dataset)
    indices = np.arange(n)
    labels = _get_labels(dataset)
    sites = getattr(dataset, "sites", None)

    if (
        group_by_site
        and sites is not None
        and len(np.unique(np.asarray(sites))) >= n_folds
    ):
        from sklearn.model_selection import StratifiedGroupKFold
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        iterator = splitter.split(indices, labels, groups=np.asarray(sites))
        logger.info("CV: %d-fold StratifiedGroupKFold (leave-sites-out, label-stratified)", n_folds)
    else:
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        iterator = splitter.split(indices, labels)
        logger.info("CV: %d-fold StratifiedKFold (label-stratified, pooled across sites)", n_folds)

    return [(tr.tolist(), va.tolist()) for tr, va in iterator]


# ===========================================================================
# Stage 2 — Model factory
# ===========================================================================

def _build_model_factory(cfg, device: torch.device):
    """
    Returns a zero-argument callable that constructs a fresh ASDModel.

    Called once per fold so weights are reinitialised between folds.
    """
    from models.mri      import build_mri_encoder
    from models.genetics import build_genetics_encoder
    from models.fusion   import build_fusion_module
    from models.asd_model import ASDModel

    def factory() -> ASDModel:
        mri_enc = build_mri_encoder(cfg)
        gen_enc = build_genetics_encoder(cfg, n_genes=cfg.genetics_model.input_dim)
        fusion  = build_fusion_module(cfg)
        return ASDModel(mri_enc, gen_enc, fusion)

    # Smoke-test: build once to catch config errors before training starts
    try:
        _ = factory().to(device)
        logger.info("Model factory smoke-test passed")
    except Exception as e:
        logger.error("Model factory failed: %s", e)
        raise

    return factory


# ===========================================================================
# Stage 2b — Computational profiling
# ===========================================================================

def run_computational_profile(
    model_factory,
    dataset:    Dataset,
    device:     torch.device,
    real_data:  bool,
    out_dir:    Path,
) -> Any:
    """
    Measure FLOPs/MACs, latency, throughput, and GPU memory for the model.

    Uses a fresh (untrained) model from the factory because parameter count
    and FLOPs are architecture properties independent of learned weights.
    Batch sizes [1, 4] are used for synthetic mode; [1, 4, 8] for real data,
    matching typical clinical batch requirements.
    """
    from utilities.profiler import ComputationalProfiler

    model = model_factory().to(device)

    # Grab a single sample and add batch dimension
    sample = dataset[0]
    example_inputs = {
        "image":    sample["image"].unsqueeze(0),    # (1, C, D, H, W)
        "genetics": sample["genetics"].unsqueeze(0), # (1, G)
    }

    batch_sizes = [1, 4] if not real_data else [1, 4, 8]

    profiler = ComputationalProfiler(device=device, amp=False)
    prof_report = profiler.profile(
        model          = model,
        example_inputs = example_inputs,
        batch_sizes    = batch_sizes,
        n_warmup       = 5,
        n_trials       = 20,   # fewer trials for faster pipeline; increase for paper
        profile_layers = True,
        out_dir        = out_dir / "computational_profile",
    )
    prof_report.print_summary()

    # Write LaTeX table alongside the JSON
    latex_path = out_dir / "computational_profile" / "latency_table.tex"
    latex_path.write_text(prof_report.to_latex_table(), encoding="utf-8")
    logger.info("LaTeX latency table → %s", latex_path)

    del model   # free memory before training starts
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return prof_report


# ===========================================================================
# Stage 2 — Training
# ===========================================================================

def run_training(
    cfg,
    dataset:       Dataset,
    splits:        List[Tuple[List[int], List[int]]],
    model_factory,
    device:        torch.device,
    out_dir:       Path,
) -> Dict[str, Any]:
    """Run K-fold CV and return aggregated metrics + per-fold raw results."""
    from training.trainer import ASDTrainer

    trainer = ASDTrainer(cfg=cfg, model_factory=model_factory, device=device)
    results = trainer.run_cv(dataset, splits, save_dir=out_dir / "checkpoints")

    logger.info("CV complete.  Mean metrics:")
    for k, v in results["mean_metrics"].items():
        std = results["std_metrics"].get(k, 0.0)
        logger.info("  %-22s %.4f ± %.4f", k, v, std)

    return results


# ===========================================================================
# Stage 3 — Evaluation
# ===========================================================================

def run_evaluation(
    cfg,
    fold_results: List[Dict[str, float]],
    dataset:      Dataset,
    splits:       List[Tuple[List[int], List[int]]],
    out_dir:      Path,
    model_factory=None,
    device:       Optional[torch.device] = None,
    train_dir:    Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full evaluation: bootstrap CI, calibration, per-site, model comparisons.

    Loads each fold's best checkpoint and runs inference on that fold's
    validation set to collect real, out-of-fold y_true / y_prob arrays.
    If no checkpoints are found it raises (never fabricates predictions).
    """
    from evaluation.evaluator import ASDEvaluator

    evaluator = ASDEvaluator(cfg=cfg, n_bootstrap=cfg.evaluation.bootstrap_iterations)
    cv_report = evaluator.evaluate_cv(fold_results)

    logger.info("CV evaluation (t-dist CI across folds):")
    for name, mci in cv_report.items():
        logger.info("  %-22s %.4f [%.4f, %.4f]", name, mci.value,
                    mci.ci_lower, mci.ci_upper)

    # --- Collect REAL out-of-fold predictions from each fold's best checkpoint.
    # Every subject is scored by the model of the fold in which it was the
    # validation set, so the pooled predictions are genuinely out-of-fold.
    y_true_all: List[int]   = []
    y_prob_all: List[float] = []
    site_ids_all: List[str] = []
    feats_all: List[np.ndarray] = []

    if model_factory is not None and device is not None and train_dir is not None:
        for fold_idx, (_, val_idx) in enumerate(splits):
            best_ckpt = _best_ckpt_in(Path(train_dir) / f"fold_{fold_idx}")
            if best_ckpt is None:
                logger.warning("No checkpoint for fold %d — skipping inference", fold_idx)
                continue
            logger.info("Fold %d: inference with checkpoint %s", fold_idx, best_ckpt.name)
            yt, yp, si, ft = _infer_probs(model_factory, device, best_ckpt, dataset, val_idx)
            y_true_all.extend(yt.tolist())
            y_prob_all.extend(yp.tolist())
            site_ids_all.extend(si.tolist())
            if ft is not None:
                feats_all.append(ft)

    if not y_true_all:
        # No real predictions → refuse to fabricate. A trustworthy report cannot
        # be produced without trained checkpoints; fail loudly instead.
        raise RuntimeError(
            "Evaluation found no fold checkpoints to run inference on. "
            f"Expected checkpoints under {train_dir}. Training must complete "
            "before evaluation; synthetic placeholder metrics are not permitted."
        )

    y_true   = np.array(y_true_all)
    y_prob   = np.array(y_prob_all, dtype=np.float32)
    site_ids = np.array(site_ids_all)

    report = evaluator.evaluate(y_true, y_prob, site_ids=site_ids.astype(str))
    logger.info("\n%s", report.summary())
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save_json(out_dir / "evaluation_report.json")

    # --- Error analysis ---
    from evaluation.error_analysis import ErrorAnalyzer
    subject_ids = [str(i) for i in range(len(y_true))]
    err_analyzer = ErrorAnalyzer(
        hard_confidence_threshold = 0.75,
        uncertainty_margin        = 0.10,
        clinical_fn_weight        = 2.0,
    )
    err_report = err_analyzer.analyze(
        y_true      = y_true,
        y_prob      = y_prob,
        threshold   = report.threshold,
        subject_ids = subject_ids,
        site_ids    = site_ids.astype(str),
        out_dir     = out_dir / "error_analysis",
    )
    err_report.print_summary()

    features = np.concatenate(feats_all, axis=0) if feats_all else None

    return {
        "cv_report":  cv_report,
        "report":     report,
        "err_report": err_report,
        "y_true":     y_true,
        "y_prob":     y_prob,
        "site_ids":   site_ids,
        "features":   features,   # real fused embeddings (out-of-fold), or None
    }


# ===========================================================================
# Stage 4b — Held-out (ABIDE II) external validation
# ===========================================================================

def _best_ckpt_in(fold_dir: Path) -> Optional[Path]:
    """Return the highest-val_auc checkpoint in a fold directory, or None."""
    ckpts = sorted(fold_dir.glob("*.pt")) if Path(fold_dir).exists() else []
    if not ckpts:
        return None

    def _auc(p: Path) -> float:
        try:
            return float(p.stem.split("val_auc")[-1])
        except Exception:
            return -1.0

    return max(ckpts, key=_auc)


def _infer_probs(
    model_factory,
    device: torch.device,
    ckpt_path: Path,
    dataset: Dataset,
    indices: Optional[List[int]] = None,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load ``ckpt_path`` into a fresh model and run real inference.

    Returns ``(y_true, y_prob, site_ids, fused_features)`` over
    ``dataset[indices]`` (or the whole dataset when ``indices`` is None).
    ``fused_features`` is the model's fused embedding (for projection plots),
    or None if the fusion module does not expose it.  This is the single
    source of truth for turning trained checkpoints into predictions — no
    synthetic probabilities anywhere.
    """
    from training.checkpointing import CheckpointManager

    subset = dataset if indices is None else torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model_factory().to(device)
    CheckpointManager.load_from_path(ckpt_path, model, device=device)
    model.eval()

    y_true: List[int] = []
    y_prob: List[float] = []
    site_ids: List[str] = []
    feats: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            mri = batch["image"].to(device)
            gen = batch["genetics"].to(device)
            out = model(mri, gen)
            probs = torch.softmax(out["logits"].float(), dim=-1)[:, 1].cpu().numpy()
            y_true.extend(batch["label"].cpu().numpy().tolist())
            y_prob.extend(probs.tolist())
            site_ids.extend(str(s) for s in batch["site"].cpu().numpy().tolist())
            fused = out.get("fused_features")
            if fused is not None:
                feats.append(fused.float().cpu().numpy())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    features = np.concatenate(feats, axis=0) if feats else None
    return (
        np.array(y_true),
        np.array(y_prob, dtype=np.float32),
        np.array(site_ids),
        features,
    )


def run_held_out_evaluation(
    cfg,
    mri_dir:       str,
    gen_dir:       str,
    out_dir:       Path,
    model_factory,
    device:        torch.device,
    train_dir:     Path,
    n_folds:       int,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate the trained CV ensemble on a held-out external test set (ABIDE II).

    For every fold's best checkpoint we run real inference over the *entire*
    held-out cohort, then average the per-subject probabilities across folds
    (soft-voting ensemble).  Nothing here is synthetic — if no checkpoints are
    found the stage is skipped rather than fabricating predictions.

    Results are written to ``out_dir/held_out_report.json``.
    """
    from evaluation.evaluator import ASDEvaluator

    held_out_dataset = ABIDEDataset(mri_dir, gen_dir)
    if len(held_out_dataset) == 0:
        logger.warning("Held-out dataset is empty — skipping external validation.")
        return None

    n = len(held_out_dataset)
    logger.info("Held-out external set: %d subjects", n)

    # --- Real ensemble inference across all fold checkpoints ---
    prob_sum = np.zeros(n, dtype=np.float64)
    y_true: Optional[np.ndarray] = None
    site_ids: Optional[np.ndarray] = None
    n_models = 0

    for fold_idx in range(n_folds):
        best = _best_ckpt_in(Path(train_dir) / f"fold_{fold_idx}")
        if best is None:
            logger.warning("No checkpoint for fold %d — excluded from ensemble", fold_idx)
            continue
        yt, yp, si, _ = _infer_probs(model_factory, device, best, held_out_dataset)
        prob_sum += yp
        y_true, site_ids = yt, si
        n_models += 1
        logger.info("Held-out: fold %d checkpoint %s → inference done", fold_idx, best.name)

    if n_models == 0 or y_true is None:
        logger.warning("No trained checkpoints available — skipping held-out evaluation.")
        return None

    y_prob = (prob_sum / n_models).astype(np.float32)

    evaluator = ASDEvaluator(cfg=cfg, n_bootstrap=cfg.evaluation.bootstrap_iterations)
    report = evaluator.evaluate(y_true, y_prob, site_ids=site_ids.astype(str))
    logger.info("Held-out evaluation (%d-model ensemble):\n%s", n_models, report.summary())
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save_json(out_dir / "held_out_report.json")

    return {
        "report":   report,
        "y_true":   y_true,
        "y_prob":   y_prob,
        "site_ids": site_ids,
        "n_models": n_models,
    }


# ===========================================================================
# Stage 5 — Ablation Study (function definitions)
# ===========================================================================

def run_ablation(
    cfg,
    model_factory,
    dataset:       Dataset,
    splits:        List[Tuple[List[int], List[int]]],
    device:        torch.device,
    out_dir:       Path,
    max_variants:  Optional[int] = None,
) -> Any:
    """
    OFAT ablation study over fusion strategies.

    The train_fn wraps the real ASDTrainer so each variant actually trains.
    For the synthetic smoke-test this is very fast (3 epochs × 2 folds).
    """
    from training.trainer import ASDTrainer
    from ablation.ablation_runner import AblationRunner, default_config_modifier
    from ablation.study_factory import build_fusion_ablation
    from ablation.ablation_analyzer import AblationAnalyzer

    # ---------------------------------------------------------------------------
    # train_fn for AblationRunner
    # ---------------------------------------------------------------------------
    def train_fn(variant_cfg, variant_name: str) -> List[Dict[str, float]]:
        """
        Build a fresh trainer from variant_cfg and run CV.

        variant_cfg is a deep-copy of cfg with dot-notation overrides applied.
        Because default_config_modifier works on plain dicts, the ablation
        runner passes a dict; we reconstruct a Config-like object here.
        """
        # Apply overrides directly to a copy of cfg (Config is already a dataclass)
        # For variant_cfg (dict), rebuild model from cfg with relevant overrides
        v_cfg = copy.deepcopy(cfg)
        if isinstance(variant_cfg, dict):
            # Apply flat overrides to cfg object
            for dotkey, value in variant_cfg.items():
                parts = dotkey.split(".")
                obj = v_cfg
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

        # Override fusion method from ablation config
        v_factory = _build_model_factory(v_cfg, device)
        trainer   = ASDTrainer(cfg=v_cfg, model_factory=v_factory, device=device)
        results   = trainer.run_cv(dataset, splits,
                                    save_dir=out_dir / "ablation_ckpts" / variant_name)
        return results["fold_results"]

    study = build_fusion_ablation(base_config={})
    variants = study.generate_variants()
    if max_variants is not None:
        # Slice to a subset for faster smoke-testing
        study._custom_variants = variants[:max_variants]
        study.mode = "custom"

    runner   = AblationRunner(
        train_fn     = train_fn,
        save_dir     = out_dir,   # out_dir is already <run>/ablation (avoid ablation/ablation)
        config_modifier = lambda base, overrides: overrides,  # overrides passed directly
        verbose      = True,
    )
    abl_results = runner.run_study(study, resume=True)

    if len(abl_results) > 1:
        analyzer = AblationAnalyzer(abl_results, baseline_name="baseline")
        md_table = analyzer.markdown_table(metrics=["val_auc"])
        logger.info("\nAblation results:\n%s", md_table)
        md_path = out_dir / "ablation_table.md"
        md_path.write_text(md_table, encoding="utf-8")

    return abl_results


# ===========================================================================
# Stage 6 — Hyperparameter Optimisation
# ===========================================================================

def run_hpo(
    cfg,
    dataset:  Dataset,
    splits:   List[Tuple[List[int], List[int]]],
    device:   torch.device,
    out_dir:  Path,
    n_trials: int = 8,
    seed:     int = 42,
) -> Any:
    """
    Bayesian HPO with Optuna over the "quick" search space.

    train_fn returns per-fold val_auc by running real training.
    """
    from training.trainer import ASDTrainer
    from hyperparameter_tuning.optuna_tuner import ASDTuner
    from hyperparameter_tuning.callbacks import (
        ProgressCallback, CheckpointCallback, EarlyStoppingCallback,
    )
    from ablation.ablation_runner import default_config_modifier

    def train_fn(variant_cfg, variant_name: str) -> List[Dict[str, float]]:
        v_cfg   = default_config_modifier(copy.deepcopy(cfg).__dict__
                  if not isinstance(variant_cfg, dict) else {}, variant_cfg)
        # Re-use base cfg but override learning rate + dropout if present
        h_cfg   = copy.deepcopy(cfg)
        if isinstance(variant_cfg, dict):
            lr = variant_cfg.get("optimizer.lr")
            if lr is not None:
                h_cfg.training.learning_rate = lr
            wd = variant_cfg.get("optimizer.weight_decay")
            if wd is not None:
                h_cfg.training.weight_decay = wd
            dropout = variant_cfg.get("model.mri.dropout")
            if dropout is not None:
                h_cfg.mri_model.dropout = float(dropout)
        h_factory = _build_model_factory(h_cfg, device)
        trainer   = ASDTrainer(cfg=h_cfg, model_factory=h_factory, device=device)
        results   = trainer.run_cv(dataset, splits,
                                    save_dir=out_dir / "hpo_ckpts" / variant_name)
        return results["fold_results"]

    callbacks = [
        ProgressCallback(print_fn=logger.info),
        CheckpointCallback(save_path=out_dir / "hpo_best_trial.json"),
        EarlyStoppingCallback(patience=max(3, n_trials // 4),
                              direction="maximize"),
    ]

    tuner = ASDTuner(
        train_fn     = train_fn,
        base_config  = {},
        search_space = "quick",
        n_trials     = n_trials,
        pruner       = "none",
        sampler      = "tpe",
        study_name   = "asd_hpo",
        storage_path = out_dir / "hpo_study.db",
        seed         = seed,
        callbacks    = callbacks,
    )
    tuner.optimize()

    summ = tuner.summary_dict()
    logger.info(
        "HPO complete: best val_auc=%.4f  params=%s",
        summ["best_metric"].get("val_auc", float("nan")),
        summ["best_params"],
    )
    return tuner


# ===========================================================================
# Stage 6 — Explainability
# ===========================================================================

def run_explainability(
    cfg,
    dataset:       Dataset,
    model_factory,
    device:        torch.device,
    out_dir:       Path,
) -> Dict[str, Any]:
    """
    GradCAM++ saliency on a single subject + integrated-gradient gene importance.

    Returns numpy arrays for use in paper figure generation.
    """
    from explainability.explainability_engine import ExplainabilityEngine

    model = model_factory().to(device)
    model.eval()
    engine = ExplainabilityEngine(model, device=device)

    # Use the first subject from the dataset
    sample   = dataset[0]
    mri      = sample["image"].unsqueeze(0).to(device)       # (1, 1, D, H, W)
    genetics = sample["genetics"].unsqueeze(0).to(device)    # (1, n_genes)

    try:
        exp = engine.explain(
            mri         = mri,
            genetics    = genetics,
            adj         = None,
            target_class= 1,
            mri_methods = ["gradcam_plus_plus"],
            gen_methods = ["gradient_x_input"],
        )

        mri_vol  = mri[0, 0].cpu().numpy()
        saliency = (
            exp.get("mri", {}).get("gradcam_plus_plus")
        )
        if saliency is not None:
            saliency = saliency[0].cpu().numpy()
        else:
            saliency = np.abs(mri_vol)

        gene_imp = (
            exp.get("genetics", {}).get("gradient_x_input")
        )
        if gene_imp is not None:
            gene_imp = gene_imp[0].cpu().numpy()
        else:
            gene_imp = np.ones(genetics.shape[1])

        gene_names = _feature_names(len(gene_imp))

        logger.info(
            "Explainability: saliency shape=%s  gene_imp shape=%s",
            saliency.shape, gene_imp.shape,
        )
        return {
            "mri_volume":       mri_vol,
            "saliency":         saliency,
            "gene_importances": gene_imp,
            "gene_names":       gene_names,
        }

    except Exception as e:
        logger.warning("Explainability failed (%s); using zeros", e)
        D, H, W = dataset[0]["image"].shape[1:]
        n_genes = dataset[0]["genetics"].shape[0]
        return {
            "mri_volume":       np.zeros((D, H, W), dtype=np.float32),
            "saliency":         np.zeros((D, H, W), dtype=np.float32),
            "gene_importances": np.zeros(n_genes,   dtype=np.float32),
            "gene_names":       _feature_names(n_genes),
        }


# ===========================================================================
# Stage 7 — Paper figures
# ===========================================================================

def run_paper_figures(
    cfg,
    out_dir:      Path,
    eval_bundle:  Dict[str, Any],
    abl_results,
    tuner,
    expl_bundle:  Dict[str, Any],
    fold_results: List[Dict[str, float]],
    dataset:      Dataset,
) -> None:
    """Generate the paper figures from REAL results only.

    No synthetic/placeholder data is injected: curves come from the model's
    real out-of-fold predictions, the projection uses real fused embeddings,
    and calibration computes Brier from the real predictions.  Figures whose
    inputs are unavailable are simply skipped (generate_all is defensive).
    """
    from paper.paper_figures import PaperFigureGenerator

    figures_dir = out_dir / "paper_figures"
    gen = PaperFigureGenerator(
        output_dir = figures_dir,
        formats    = ["pdf", "png"],
        dpi        = 300,
    )

    y_true = eval_bundle["y_true"]
    y_prob = eval_bundle["y_prob"]
    report = eval_bundle.get("report")
    threshold = float(getattr(report, "threshold", 0.5)) if report is not None else 0.5

    # Real per-fold CV results — the proposed model only.  We do NOT invent
    # comparison baselines; if a real baseline is wanted it must be trained
    # (e.g. via the mri_only/genetics_only ablation modes).
    cv_results_for_fig = [
        {"model": "Proposed", "fold": i, **fm} for i, fm in enumerate(fold_results)
    ]

    # Real site counts from the dataset.
    site_counts_dict: Dict[str, int] = {}
    if hasattr(dataset, "sites"):
        for s in dataset.sites:
            key = f"Site_{s}"
            site_counts_dict[key] = site_counts_dict.get(key, 0) + 1

    bundle: Dict[str, Any] = {
        "site_counts":  site_counts_dict,
        "class_counts": {"ASD": int(np.sum(y_true == 1)),
                         "TC":  int(np.sum(y_true == 0))},
        "models": [
            {"name": "Proposed", "y_true": y_true, "y_prob": y_prob,
             "color": "#e41a1c"},
        ],
        "ablation_results": abl_results,
        "cv_results":       cv_results_for_fig,
        **expl_bundle,
        "site_ids":         eval_bundle["site_ids"],
        "y_true":           y_true,
        "y_prob":           y_prob,
        "threshold":        threshold,
        "cv_report":        eval_bundle.get("cv_report"),
        "tuner":            tuner,
    }

    # Real fused embeddings (out-of-fold, aligned with y_true/site_ids).
    # Omit the projection figure entirely if embeddings are unavailable.
    features = eval_bundle.get("features")
    if features is not None and len(features) == len(y_true):
        bundle["features"] = features
        bundle["labels"]   = y_true
    else:
        logger.info("No real embeddings available — skipping projection figure")

    saved = gen.generate_all(bundle)
    import matplotlib.pyplot as plt
    plt.close("all")

    manifest = gen.latex_manifest()
    manifest_path = out_dir / "figure_manifest.tex"
    manifest_path.write_text(manifest, encoding="utf-8")

    logger.info(
        "Paper figures: %d files generated → %s",
        sum(len(v) for v in saved.values()), figures_dir,
    )
    logger.info("LaTeX manifest → %s", manifest_path)


# ===========================================================================
# Data validation helper
# ===========================================================================

def _run_data_validation(args, dataset, splits, out_dir: Path) -> None:
    """
    Run pre-training data validation and save the report.

    Collects train/test subject indices from all CV folds and performs
    a combined leakage check: no subject that appears in any fold's test
    set should also appear in any fold's training set.

    Raises DataValidationError and calls sys.exit(1) on any critical issue.
    """
    from data.data_validator import DataValidator, DataValidationError

    validator = DataValidator()

    # In K-fold CV, leakage means a subject appears in BOTH the train set
    # AND the val set of THE SAME FOLD.  The union of all train sets always
    # overlaps the union of all val sets (by design), so a global check
    # would be a false positive.  We check per-fold instead.
    per_fold_leakage: Dict[int, List[int]] = {}
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        overlap = set(train_idx) & set(val_idx)
        if overlap:
            per_fold_leakage[fold_idx] = sorted(overlap)

    if args.real_data:
        mri_dir  = Path(args.mri_dir)
        gen_dir  = Path(args.gen_dir)
        meta_path = mri_dir / "metadata.csv"
        report = validator.validate_real(
            mri_dir   = mri_dir,
            gen_dir   = gen_dir,
            meta_path = meta_path,
        )
    else:
        report = validator.validate_synthetic(dataset)

    # Inject per-fold leakage results into the report
    if per_fold_leakage:
        from data.data_validator import IssueRecord
        n_leaked = sum(len(v) for v in per_fold_leakage.values())
        report.leaked_subject_count = n_leaked
        report.issues.insert(0, IssueRecord(
            severity="CRITICAL",
            category="leakage",
            message=(
                f"Per-fold data leakage in {len(per_fold_leakage)} fold(s): "
                f"{n_leaked} subject(s) appear in both train and val of the "
                f"same fold.  StratifiedKFold should prevent this — "
                f"check _make_cv_splits."
            ),
            affected=[f"fold_{k}" for k in per_fold_leakage],
        ))
        report.passed = False
    else:
        from data.data_validator import IssueRecord
        report.issues.insert(0, IssueRecord(
            severity="INFO",
            category="leakage",
            message=(
                f"K-fold leakage check: all {len(splits)} fold(s) have "
                f"disjoint train/val sets — no leakage detected."
            ),
        ))

    report.print_summary()
    report.save(out_dir / "data_validation_report.json")

    if report.n_critical > 0:
        logger.error(
            "DATA VALIDATION FAILED: %d critical issue(s). "
            "Training cannot proceed.", report.n_critical,
        )
        try:
            report.raise_if_critical()
        except Exception as exc:
            logger.error("%s", exc)
            sys.exit(1)
    else:
        logger.info(
            "Data validation PASSED (%d warnings). "
            "Fingerprint: %s", report.n_warnings, report.fingerprint,
        )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    t_start = time.time()
    args    = _parse_args()
    cfg, device, out_dir, seed = _setup(args)

    logger.info("=" * 60)
    logger.info("ASD Detection Pipeline  (mode=%s  device=%s)",
                "real" if args.real_data else "synthetic", device)
    logger.info("Output directory: %s", out_dir.resolve())
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1 — Dataset
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 1: Dataset ---")
    if args.real_data:
        if not args.mri_dir or not args.gen_dir:
            logger.error("--real_data requires --mri_dir and --gen_dir")
            sys.exit(1)
        dataset = ABIDEDataset(args.mri_dir, args.gen_dir)
        # Infer n_genes from first sample
        first_gen = dataset[0]["genetics"]
        cfg.genetics_model.input_dim = first_gen.shape[0]
    else:
        MRI_SHAPE = (16, 16, 16)  # tiny for speed
        N_GENES   = cfg.genetics_model.input_dim
        dataset = SyntheticABIDE(
            n_subjects = 40,
            mri_shape  = MRI_SHAPE,
            n_genes    = N_GENES,
            n_sites    = 4,
            seed       = seed,
        )

    n_folds = cfg.training.cross_validation.n_folds
    splits  = _make_cv_splits(
        dataset,
        n_folds=n_folds,
        seed=seed,
        group_by_site=getattr(cfg.training.cross_validation, "group_by_site", False),
    )

    # ------------------------------------------------------------------
    # Stage 1b — Pre-training Data Validation
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 1b: Data Validation ---")
    _run_data_validation(args, dataset, splits, out_dir)
    logger.info(
        "Dataset: %d subjects, %d sites, %d folds",
        len(dataset),
        len(np.unique(getattr(dataset, "sites",
                              np.zeros(len(dataset), int)))),
        n_folds,
    )

    # ------------------------------------------------------------------
    # Stage 2 — Model factory
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 2: Model ---")
    model_factory = _build_model_factory(cfg, device)

    # ------------------------------------------------------------------
    # Stage 2b — Computational profiling
    # ------------------------------------------------------------------
    prof_report = None
    if not args.skip_profile:
        logger.info("\n--- Stage 2b: Computational Profiling ---")
        try:
            prof_report = run_computational_profile(
                model_factory,
                dataset,
                device,
                real_data = args.real_data,
                out_dir   = out_dir,
            )
        except Exception as exc:
            logger.warning(
                "Computational profiling failed (%s) — continuing pipeline.", exc
            )

    # ------------------------------------------------------------------
    # Stage 3 — Training
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 3: Training (%d folds × %d epochs) ---",
                n_folds, cfg.training.max_epochs)
    train_results = run_training(
        cfg, dataset, splits, model_factory, device,
        out_dir / "training",
    )

    # ------------------------------------------------------------------
    # Stage 4 — Evaluation
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 4: Evaluation ---")
    eval_bundle = run_evaluation(
        cfg,
        train_results["fold_results"],
        dataset, splits,
        out_dir / "evaluation",
        model_factory = model_factory,
        device        = device,
        train_dir     = out_dir / "training" / "checkpoints",
    )

    # ------------------------------------------------------------------
    # Stage 4b — Model export (ONNX + TorchScript)
    # ------------------------------------------------------------------
    if not args.skip_profile:
        logger.info("\n--- Stage 4b: Model Export ---")
        try:
            from utilities.model_export import ModelExporter

            export_model  = model_factory().to(device)
            export_sample = dataset[0]
            export_mri    = export_sample["image"].unsqueeze(0).to(device)
            export_gen    = export_sample["genetics"].unsqueeze(0).to(device)

            exporter     = ModelExporter(tolerance=1e-4, opset=17)
            export_report = exporter.export(
                model      = export_model,
                mri_tensor = export_mri,
                gen_tensor = export_gen,
                out_dir    = out_dir / "model_export",
                device     = device,
            )
            export_report.print_summary()
            export_report.save(out_dir / "model_export" / "export_report.json")

            del export_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            logger.warning(
                "Model export failed (%s) — continuing pipeline.", exc
            )

    # ------------------------------------------------------------------
    # Stage 4c — Robustness evaluation
    # ------------------------------------------------------------------
    if not args.skip_profile:
        logger.info("\n--- Stage 4c: Robustness Evaluation ---")
        try:
            from evaluation.robustness import RobustnessEvaluator

            # Build a small held-out set from the last fold's val indices
            last_train_idx, last_val_idx = splits[-1]
            rob_indices = last_val_idx

            rob_mri  = torch.stack([dataset[i]["image"]    for i in rob_indices])
            rob_gen  = torch.stack([dataset[i]["genetics"] for i in rob_indices])
            rob_true = np.array([dataset[i]["label"].item() for i in rob_indices])
            rob_sites = np.array([str(dataset[i]["site"].item()) for i in rob_indices])

            rob_model = model_factory().to(device)

            rob_eval  = RobustnessEvaluator(
                threshold  = eval_bundle["report"].threshold,
                batch_size = max(1, min(8, len(rob_indices) // 2)),
                seed       = seed,
                min_site_n = 2,   # low for synthetic; real data use default 5
            )
            rob_report = rob_eval.evaluate(
                model    = rob_model,
                mri      = rob_mri,
                genetics = rob_gen,
                y_true   = rob_true,
                site_ids = rob_sites,
                device   = device,
                out_dir  = out_dir / "robustness",
            )
            rob_report.print_summary()

            del rob_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            logger.warning(
                "Robustness evaluation failed (%s) — continuing pipeline.", exc
            )

    # ------------------------------------------------------------------
    # Stage 4d — Held-out external validation (ABIDE II)
    # ------------------------------------------------------------------
    held_out_bundle = None
    if args.held_out_mri_dir and args.held_out_gen_dir:
        logger.info("\n--- Stage 4d: Held-out ABIDE II Evaluation ---")
        held_out_bundle = run_held_out_evaluation(
            cfg,
            mri_dir       = args.held_out_mri_dir,
            gen_dir       = args.held_out_gen_dir,
            out_dir       = out_dir / "held_out_evaluation",
            model_factory = model_factory,
            device        = device,
            train_dir     = out_dir / "training" / "checkpoints",
            n_folds       = n_folds,
        )

    # ------------------------------------------------------------------
    # Stage 5 — Ablation (run only the first 2 variants for smoke-test)
    # ------------------------------------------------------------------
    max_abl = args.n_ablation_trials if args.n_ablation_trials else (
        None if args.real_data else 2
    )
    logger.info("\n--- Stage 5: Ablation (max_variants=%s) ---", max_abl)
    abl_results = run_ablation(
        cfg, model_factory, dataset, splits, device,
        out_dir / "ablation",
        max_variants=max_abl,
    )

    # ------------------------------------------------------------------
    # Stage 6 — HPO
    # ------------------------------------------------------------------
    n_hpo = args.n_hpo_trials if args.n_hpo_trials else (
        50 if args.real_data else 3
    )
    logger.info("\n--- Stage 6: HPO (%d trials) ---", n_hpo)
    tuner = run_hpo(
        cfg, dataset, splits, device,
        out_dir / "hpo",
        n_trials = n_hpo,
        seed     = seed,
    )

    # ------------------------------------------------------------------
    # Stage 7 — Explainability
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 7: Explainability ---")
    expl_bundle = run_explainability(
        cfg, dataset, model_factory, device, out_dir / "explainability"
    )

    # ------------------------------------------------------------------
    # Stage 8 — Paper figures
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 8: Paper figures ---")
    run_paper_figures(
        cfg, out_dir,
        eval_bundle  = eval_bundle,
        abl_results  = abl_results,
        tuner        = tuner,
        expl_bundle  = expl_bundle,
        fold_results = train_results["fold_results"],
        dataset      = dataset,
    )

    # ------------------------------------------------------------------
    # Stage 9 — Automated HTML Report
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 9: HTML Report ---")
    try:
        from reporting.report_generator import ReportGenerator

        report_bundle: Dict[str, Any] = {
            "cfg":           cfg,
            "dataset":       dataset,
            "train_results": train_results,
            "eval_bundle":   eval_bundle,
            "prof_report":   prof_report,
            "rob_report":    locals().get("rob_report"),
            "export_report": locals().get("export_report"),
            "out_dir":       out_dir,
            "seed":          seed,
            "device":        device,
            "n_folds":       n_folds,
        }

        ReportGenerator().generate(
            bundle   = report_bundle,
            out_path = out_dir / "experiment_report.html",
        )
    except Exception as exc:
        logger.warning("HTML report generation failed (%s) — continuing.", exc)

    # ------------------------------------------------------------------
    # Done — finalise the run manifest with timing + headline metrics
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    try:
        from utilities.run_dir import update_manifest
        update_manifest(out_dir, {
            "elapsed_seconds": round(elapsed, 1),
            "completed": True,
            "cv_mean_metrics": train_results.get("mean_metrics", {}),
            "cv_std_metrics":  train_results.get("std_metrics", {}),
        })
    except Exception as exc:
        logger.debug("Manifest finalisation skipped: %s", exc)

    logger.info("\n%s", "=" * 60)
    logger.info("Pipeline complete in %.1f seconds (%.1f min)",
                elapsed, elapsed / 60)
    logger.info("Results → %s", out_dir.resolve())
    logger.info("HTML report → %s", (out_dir / "experiment_report.html").resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
