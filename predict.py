"""
Single-subject ASD inference from a trained checkpoint.

Loads a saved ASDModel checkpoint, preprocesses one subject's MRI NIfTI and
genetics vector, runs forward inference, and writes a JSON prediction report.

Usage
-----
Minimal (MRI only, zero genetics vector)::

    python predict.py \\
        --checkpoint results/full_run/training/checkpoints/best_model.pt \\
        --mri_nifti  path/to/subject.nii.gz

With genetics::

    python predict.py \\
        --checkpoint results/full_run/training/checkpoints/best_model.pt \\
        --mri_nifti  path/to/subject.nii.gz \\
        --genetics   path/to/subject_genetics.npy \\
        --out_json   output/prediction.json

Explain (GradCAM++ saliency + gene importance)::

    python predict.py \\
        --checkpoint results/full_run/training/checkpoints/best_model.pt \\
        --mri_nifti  path/to/subject.nii.gz \\
        --genetics   path/to/subject_genetics.npy \\
        --explain \\
        --saliency_npy output/saliency.npy \\
        --out_json     output/prediction.json

Output JSON schema
------------------
{
    "subject_id":      "optional string",
    "prediction":      "ASD" | "TC",
    "asd_probability": float  (0.0–1.0),
    "tc_probability":  float  (0.0–1.0),
    "logits":          [float, float],
    "threshold":       float,
    "checkpoint":      "path/to/checkpoint.pt",
    "config": {
        "mri_backbone": "...",
        "genetics_backbone": "...",
        "fusion_type": "...",
        "target_shape": [D, H, W],
        "n_genetics_features": int
    },
    "explainability": {           # only present when --explain
        "saliency_saved_to":  "path/to/saliency.npy",
        "top_genes": [            # only when genetics provided
            {"rank": 1, "feature_index": int, "importance": float},
            ...
        ]
    }
}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-subject ASD inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Path to .pt checkpoint saved by CheckpointManager")
    p.add_argument("--mri_nifti",     required=True,
                   help="Path to subject's preprocessed fMRI NIfTI file (.nii / .nii.gz)")
    p.add_argument("--genetics",      default=None,
                   help="Path to .npy genetics feature vector (n_components,). "
                        "If omitted, a zero vector of the model's expected length is used.")
    p.add_argument("--config",        default=None,
                   help="Path to config.yaml (default: configs/config.yaml)")
    p.add_argument("--subject_id",    default=None,
                   help="Subject identifier written to output JSON")
    p.add_argument("--threshold",     type=float, default=0.5,
                   help="Decision threshold for ASD / TC label")
    p.add_argument("--device",        default=None,
                   help="Compute device: cuda / cpu / mps (auto-detected if omitted)")
    p.add_argument("--explain",       action="store_true",
                   help="Compute GradCAM++ saliency and gene importances")
    p.add_argument("--saliency_npy",  default=None,
                   help="Where to save GradCAM++ saliency map as .npy (requires --explain)")
    p.add_argument("--out_json",      default=None,
                   help="Output prediction JSON path (prints to stdout if omitted)")
    p.add_argument("--top_k_genes",   type=int, default=20,
                   help="Number of top-importance genes to include in explainability output")
    return p.parse_args()


# ===========================================================================
# Device resolution
# ===========================================================================

def _resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ===========================================================================
# Checkpoint loading
# ===========================================================================

def load_checkpoint(
    ckpt_path: str,
    device:    torch.device,
) -> Dict:
    """
    Load a checkpoint saved by CheckpointManager.

    Expected keys: ``"model_state_dict"``, ``"config"`` or ``"cfg_dict"``.
    Falls back gracefully when optional keys are absent.
    """
    path = Path(ckpt_path)
    if not path.exists():
        logger.error("Checkpoint not found: %s", path)
        sys.exit(1)

    logger.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    required = "model_state_dict"
    if required not in ckpt:
        logger.error(
            "Checkpoint missing key '%s'. "
            "Available keys: %s", required, list(ckpt.keys())
        )
        sys.exit(1)

    logger.info(
        "Checkpoint loaded  epoch=%s  val_auc=%s",
        ckpt.get("epoch", "?"),
        f"{ckpt.get('val_auc', '?'):.4f}" if isinstance(ckpt.get("val_auc"), float) else "?",
    )
    return ckpt


# ===========================================================================
# Config loading
# ===========================================================================

def _load_config(config_path: Optional[str], ckpt: Dict):
    """
    Load config from (in priority order):
      1. --config CLI argument
      2. Embedded config dict in checkpoint
      3. Default configs/config.yaml
    """
    from configs import load_config

    if config_path:
        cfg = load_config(config_path)
        logger.info("Config loaded from: %s", config_path)
        return cfg

    if "cfg_dict" in ckpt:
        import yaml, io
        cfg = load_config(io.StringIO(yaml.dump(ckpt["cfg_dict"])))
        logger.info("Config loaded from checkpoint embedding")
        return cfg

    default = Path(__file__).parent / "configs" / "config.yaml"
    if default.exists():
        cfg = load_config(str(default))
        logger.info("Config loaded from default: %s", default)
        return cfg

    logger.error(
        "No config found. Pass --config path/to/config.yaml or ensure "
        "configs/config.yaml exists."
    )
    sys.exit(1)


# ===========================================================================
# Model construction
# ===========================================================================

def build_model(cfg, n_genetics_features: int, device: torch.device):
    """Rebuild the model architecture from config and load checkpoint weights."""
    from models.mri      import build_mri_encoder
    from models.genetics import build_genetics_encoder
    from models.fusion   import build_fusion_module
    from models.asd_model import ASDModel

    cfg.genetics_model.input_dim = n_genetics_features
    mri_enc = build_mri_encoder(cfg)
    gen_enc = build_genetics_encoder(cfg, n_genes=n_genetics_features)
    fusion  = build_fusion_module(cfg)
    model   = ASDModel(mri_enc, gen_enc, fusion).to(device)
    return model


# ===========================================================================
# MRI preprocessing
# ===========================================================================

def preprocess_mri(
    nifti_path:   str,
    target_shape: tuple,
    voxel_size:   tuple = (2.0, 2.0, 2.0),
) -> torch.Tensor:
    """
    Load NIfTI, compute temporal mean (if 4-D), resample, pad/crop, and
    return a (1, 1, D, H, W) batch tensor ready for the model.
    """
    from preprocessing.mri import (
        load_nifti, resample_volume, pad_or_crop_to_shape,
    )

    logger.info("Preprocessing MRI: %s", nifti_path)
    data, affine = load_nifti(nifti_path, validate=False)

    if data.ndim == 4:
        logger.info("  4-D fMRI detected — computing temporal mean")
        data = data.mean(axis=-1)
    data = data.astype(np.float32)

    data, _ = resample_volume(data, affine, target_voxel_size=voxel_size)
    data     = pad_or_crop_to_shape(data, target_shape)

    # Simple z-score normalisation (site-level normaliser not available at
    # inference; brain-masked statistics approximate training normalisation)
    mask = data > data.mean()
    if mask.sum() > 0:
        mu  = data[mask].mean()
        sig = data[mask].std() + 1e-6
        data = (data - mu) / sig

    logger.info("  MRI shape after preprocessing: %s", data.shape)
    # (1, 1, D, H, W)
    tensor = torch.from_numpy(data[np.newaxis, np.newaxis]).float()
    return tensor


# ===========================================================================
# Inference
# ===========================================================================

@torch.no_grad()
def run_inference(
    model:     torch.nn.Module,
    mri:       torch.Tensor,
    genetics:  torch.Tensor,
    device:    torch.device,
) -> Dict[str, float]:
    """
    Run one forward pass and return logits + probabilities.

    Returns
    -------
    dict with keys: ``logit_tc``, ``logit_asd``, ``prob_tc``, ``prob_asd``
    """
    model.eval()
    mri      = mri.to(device)
    genetics = genetics.to(device)

    batch = {
        "image":    mri,
        "genetics": genetics,
        "label":    torch.zeros(1, dtype=torch.long, device=device),
        "site":     torch.zeros(1, dtype=torch.long, device=device),
    }

    logits = model(batch)          # (1, 2)
    probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return {
        "logit_tc":   float(logits[0, 0].cpu()),
        "logit_asd":  float(logits[0, 1].cpu()),
        "prob_tc":    float(probs[0]),
        "prob_asd":   float(probs[1]),
    }


# ===========================================================================
# Explainability
# ===========================================================================

def run_explainability(
    model:        torch.nn.Module,
    mri:          torch.Tensor,
    genetics:     torch.Tensor,
    device:       torch.device,
    saliency_out: Optional[str],
    top_k_genes:  int,
    has_genetics: bool,
) -> Dict:
    """
    Run GradCAM++ on the MRI branch and feature-importance on the genetics
    branch.  Returns a dict suitable for embedding in the output JSON.
    """
    from explainability import GradCAMPlusPlus3D, GeneticsFeatureImportance

    expl_dict: Dict = {}

    # --- GradCAM++ on MRI ---
    try:
        mri_for_grad = mri.to(device).requires_grad_(True)
        batch = {
            "image":    mri_for_grad,
            "genetics": genetics.to(device),
            "label":    torch.zeros(1, dtype=torch.long, device=device),
            "site":     torch.zeros(1, dtype=torch.long, device=device),
        }

        gradcam = GradCAMPlusPlus3D(model)
        saliency = gradcam.explain(batch, target_class=1)  # ASD = class 1
        saliency_np = saliency.squeeze().cpu().numpy()

        if saliency_out:
            saliency_path = Path(saliency_out)
            saliency_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(saliency_path, saliency_np.astype(np.float32))
            logger.info("Saliency map saved: %s  shape=%s",
                        saliency_path, saliency_np.shape)
            expl_dict["saliency_saved_to"] = str(saliency_path.resolve())
        else:
            expl_dict["saliency_shape"] = list(saliency_np.shape)
            expl_dict["saliency_max"]   = float(saliency_np.max())
            expl_dict["saliency_mean"]  = float(saliency_np.mean())

    except Exception as exc:
        logger.warning("GradCAM++ failed (non-fatal): %s", exc)
        expl_dict["saliency_error"] = str(exc)

    # --- Genetics feature importance ---
    if has_genetics:
        try:
            imp_engine  = GeneticsFeatureImportance(model)
            batch_plain = {
                "image":    mri.to(device),
                "genetics": genetics.to(device),
                "label":    torch.zeros(1, dtype=torch.long, device=device),
                "site":     torch.zeros(1, dtype=torch.long, device=device),
            }
            importances = imp_engine.compute(batch_plain)  # (n_genes,)
            importances_np = importances.cpu().numpy()

            top_idx  = np.argsort(np.abs(importances_np))[::-1][:top_k_genes]
            top_genes = [
                {
                    "rank":          int(rank + 1),
                    "feature_index": int(idx),
                    "importance":    float(importances_np[idx]),
                }
                for rank, idx in enumerate(top_idx)
            ]
            expl_dict["top_genes"] = top_genes
        except Exception as exc:
            logger.warning("Gene importance failed (non-fatal): %s", exc)
            expl_dict["gene_importance_error"] = str(exc)

    return expl_dict


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args   = _parse_args()
    device = _resolve_device(args.device)
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------ #
    # Load checkpoint
    # ------------------------------------------------------------------ #
    ckpt = load_checkpoint(args.checkpoint, device)

    # ------------------------------------------------------------------ #
    # Determine genetics dimensionality before building model
    # ------------------------------------------------------------------ #
    if args.genetics:
        genetics_np = np.load(args.genetics).astype(np.float32)
        if genetics_np.ndim != 1:
            logger.error(
                "Genetics file must be a 1-D array, got shape %s", genetics_np.shape
            )
            sys.exit(1)
        n_genetics = genetics_np.shape[0]
        has_genetics = True
        logger.info("Genetics features: %d", n_genetics)
    else:
        # Infer from checkpoint config or fall back to config.yaml
        n_genetics   = ckpt.get("n_genetics_features", None)
        has_genetics = False

    # ------------------------------------------------------------------ #
    # Load config and build model
    # ------------------------------------------------------------------ #
    cfg = _load_config(args.config, ckpt)

    if n_genetics is None:
        n_genetics = cfg.genetics_model.input_dim
        logger.info(
            "Genetics dim inferred from config: %d (using zero vector)", n_genetics
        )

    model = build_model(cfg, n_genetics_features=n_genetics, device=device)

    # Load weights
    missing, unexpected = model.load_state_dict(
        ckpt["model_state_dict"], strict=False
    )
    if missing:
        logger.warning("Missing keys in checkpoint: %s", missing[:5])
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected[:5])
    model.eval()
    logger.info("Model weights loaded")

    # ------------------------------------------------------------------ #
    # Preprocess MRI
    # ------------------------------------------------------------------ #
    target_shape = tuple(cfg.mri_model.target_shape) \
        if hasattr(cfg.mri_model, "target_shape") else (96, 96, 96)
    mri_tensor = preprocess_mri(
        nifti_path   = args.mri_nifti,
        target_shape = target_shape,
    )

    # ------------------------------------------------------------------ #
    # Prepare genetics tensor
    # ------------------------------------------------------------------ #
    if has_genetics:
        gen_tensor = torch.from_numpy(genetics_np).unsqueeze(0).float()
    else:
        logger.info(
            "No genetics file provided — using zero vector (%d features)", n_genetics
        )
        gen_tensor = torch.zeros(1, n_genetics)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    scores = run_inference(model, mri_tensor, gen_tensor, device)
    label  = "ASD" if scores["prob_asd"] >= args.threshold else "TC"

    logger.info(
        "Prediction: %s  (P(ASD)=%.4f  P(TC)=%.4f  threshold=%.2f)",
        label, scores["prob_asd"], scores["prob_tc"], args.threshold,
    )

    # ------------------------------------------------------------------ #
    # Explainability (optional)
    # ------------------------------------------------------------------ #
    expl_dict: Optional[Dict] = None
    if args.explain:
        logger.info("Running explainability …")
        expl_dict = run_explainability(
            model        = model,
            mri          = mri_tensor,
            genetics     = gen_tensor,
            device       = device,
            saliency_out = args.saliency_npy,
            top_k_genes  = args.top_k_genes,
            has_genetics = has_genetics,
        )

    # ------------------------------------------------------------------ #
    # Build output JSON
    # ------------------------------------------------------------------ #
    output = {
        "subject_id":      args.subject_id,
        "prediction":      label,
        "asd_probability": round(scores["prob_asd"], 6),
        "tc_probability":  round(scores["prob_tc"],  6),
        "logits":          [round(scores["logit_tc"], 6),
                            round(scores["logit_asd"], 6)],
        "threshold":       args.threshold,
        "checkpoint":      str(Path(args.checkpoint).resolve()),
        "config": {
            "mri_backbone":       cfg.mri_model.backbone,
            "genetics_backbone":  cfg.genetics_model.backbone,
            "fusion_type":        cfg.fusion.fusion_type,
            "target_shape":       list(target_shape),
            "n_genetics_features": n_genetics,
        },
    }
    if expl_dict is not None:
        output["explainability"] = expl_dict

    # ------------------------------------------------------------------ #
    # Write / print result
    # ------------------------------------------------------------------ #
    json_str = json.dumps(output, indent=2)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        logger.info("Prediction saved: %s", out_path.resolve())
    else:
        print(json_str)


if __name__ == "__main__":
    main()
