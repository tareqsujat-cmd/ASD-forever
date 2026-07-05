"""
Hyperparameter search spaces for the ASD detection framework.

Each search space is a function that takes an Optuna ``Trial`` object and
returns a flat dict of dot-notation overrides compatible with
``default_config_modifier``.

Search space registry
---------------------
  ``"full"``         — all hyperparameters (~20 dims)
  ``"optimizer"``    — optimizer params only (5 dims)
  ``"architecture"`` — backbone/genetics architecture (8 dims)
  ``"fusion"``       — fusion module (4 dims)
  ``"training"``     — loss / regularisation (8 dims)
  ``"quick"``        — 6-dim space for rapid prototyping / unit tests
"""

from __future__ import annotations

import enum
from typing import Any, Callable, Dict, List


class SearchSpaceType(str, enum.Enum):
    FULL         = "full"
    OPTIMIZER    = "optimizer"
    ARCHITECTURE = "architecture"
    FUSION       = "fusion"
    TRAINING     = "training"
    QUICK        = "quick"


# ---------------------------------------------------------------------------
# Individual sub-space suggester functions
# ---------------------------------------------------------------------------

def suggest_optimizer_params(trial) -> Dict[str, Any]:
    return {
        "optimizer.lr":           trial.suggest_float("lr",           1e-5, 1e-2, log=True),
        "optimizer.weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "optimizer.beta1":        trial.suggest_float("beta1",        0.85, 0.99),
        "optimizer.beta2":        trial.suggest_float("beta2",        0.90, 0.999),
        "optimizer.warmup_steps": trial.suggest_int("warmup_steps",   100, 1000, step=100),
    }


def suggest_architecture_params(trial) -> Dict[str, Any]:
    return {
        "model.mri.architecture":      trial.suggest_categorical(
            "mri_arch", ["resnet10", "resnet50", "densenet121", "swin3d"]),
        "model.mri.dropout":           trial.suggest_float("mri_dropout", 0.0, 0.5),
        "model.mri.feature_dim":       trial.suggest_categorical(
            "mri_feature_dim", [128, 256, 512]),
        "model.genetics.architecture": trial.suggest_categorical(
            "gen_arch", ["transformer", "tabnet", "gnn"]),
        "model.genetics.n_heads":      trial.suggest_categorical(
            "gen_n_heads", [2, 4, 8]),
        "model.genetics.n_layers":     trial.suggest_int("gen_n_layers", 2, 6),
        "model.genetics.hidden_dim":   trial.suggest_categorical(
            "gen_hidden_dim", [64, 128, 256]),
        "model.genetics.dropout":      trial.suggest_float("gen_dropout", 0.0, 0.5),
    }


def suggest_fusion_params(trial) -> Dict[str, Any]:
    return {
        "fusion.architecture": trial.suggest_categorical(
            "fusion_arch", ["cross_attention", "gated", "late", "intermediate", "dynamic"]),
        "fusion.n_heads":      trial.suggest_categorical("fusion_n_heads", [2, 4, 8]),
        "fusion.dropout":      trial.suggest_float("fusion_dropout", 0.0, 0.4),
        "fusion.output_dim":   trial.suggest_categorical("fusion_output_dim", [128, 256, 512]),
    }


def suggest_training_params(trial) -> Dict[str, Any]:
    return {
        "training.batch_size":       trial.suggest_categorical("batch_size", [8, 16, 32]),
        "training.grad_accum_steps": trial.suggest_categorical("grad_accum", [1, 2, 4]),
        "training.ema_decay":        trial.suggest_float("ema_decay", 0.990, 0.9999),
        "training.loss":             trial.suggest_categorical(
            "loss", ["focal", "cross_entropy", "balanced"]),
        "training.focal_gamma":      trial.suggest_float("focal_gamma", 1.0, 3.0),
        "training.focal_alpha":      trial.suggest_float("focal_alpha", 0.25, 0.75),
        "training.label_smoothing":  trial.suggest_float("label_smoothing", 0.0, 0.15),
        "training.mixup_alpha":      trial.suggest_float("mixup_alpha", 0.0, 0.4),
    }


def suggest_full_params(trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    params.update(suggest_optimizer_params(trial))
    params.update(suggest_architecture_params(trial))
    params.update(suggest_fusion_params(trial))
    params.update(suggest_training_params(trial))
    return params


def suggest_quick_params(trial) -> Dict[str, Any]:
    """6-dimensional space — rapid prototyping and unit tests."""
    return {
        "optimizer.lr":           trial.suggest_float("lr",         1e-4, 1e-2, log=True),
        "optimizer.weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "model.mri.dropout":      trial.suggest_float("mri_dropout", 0.0,  0.4),
        "fusion.architecture":    trial.suggest_categorical(
            "fusion_arch", ["cross_attention", "gated", "late"]),
        "training.batch_size":    trial.suggest_categorical("batch_size", [16, 32]),
        "training.focal_gamma":   trial.suggest_float("focal_gamma", 1.0, 3.0),
    }


# ---------------------------------------------------------------------------
# Registry and public API
# ---------------------------------------------------------------------------

_SPACE_REGISTRY: Dict[SearchSpaceType, Callable] = {
    SearchSpaceType.FULL:         suggest_full_params,
    SearchSpaceType.OPTIMIZER:    suggest_optimizer_params,
    SearchSpaceType.ARCHITECTURE: suggest_architecture_params,
    SearchSpaceType.FUSION:       suggest_fusion_params,
    SearchSpaceType.TRAINING:     suggest_training_params,
    SearchSpaceType.QUICK:        suggest_quick_params,
}


def suggest_params(trial, space: str = "full") -> Dict[str, Any]:
    """
    Suggest hyperparameters for an Optuna trial.

    Parameters
    ----------
    trial : optuna.Trial
    space : str
        One of "full", "optimizer", "architecture", "fusion", "training", "quick".

    Returns
    -------
    dict
        Flat dot-notation override dict compatible with ``default_config_modifier``.
    """
    space_enum = SearchSpaceType(space)
    return _SPACE_REGISTRY[space_enum](trial)


def get_space_names() -> List[str]:
    """Return all registered search space names."""
    return [s.value for s in SearchSpaceType]


def get_space_dim(space: str) -> int:
    """Return number of hyperparameter axes in the given search space."""
    _DIMS = {
        SearchSpaceType.FULL:         21,
        SearchSpaceType.OPTIMIZER:    5,
        SearchSpaceType.ARCHITECTURE: 8,
        SearchSpaceType.FUSION:       4,
        SearchSpaceType.TRAINING:     8,
        SearchSpaceType.QUICK:        6,
    }
    return _DIMS[SearchSpaceType(space)]
