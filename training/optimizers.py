"""
Optimizer and learning-rate scheduler builders.

Optimizer
---------
AdamW with separate parameter groups:
  - Weight parameters:   weight_decay = cfg value (1e-5)
  - Bias / norm params:  weight_decay = 0   (standard best practice)

Applying weight decay to biases and LayerNorm/BN scale+shift parameters is
incorrect: it shrinks them toward 0, which is not regularisation but bias.

Schedulers
----------
cosine_warmup (default):
  Linear warmup for warmup_epochs, then cosine annealing to min_lr.
  Standard for vision transformers; prevents instability in early epochs.

step:
  Multiply LR by gamma every step_size epochs.

plateau:
  Reduce LR on validation metric plateau.

onecycle:
  1-cycle policy: fast warmup then steep cosine decay.
  Works well for short fine-tuning runs.
"""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    LambdaLR, StepLR, ReduceLROnPlateau, OneCycleLR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter grouping
# ---------------------------------------------------------------------------

def _no_decay_names(model: nn.Module) -> List[str]:
    """Return parameter names that should NOT have weight decay applied."""
    no_decay = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d,
                                nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.GroupNorm)):
            # weight and bias of norm layers
            for pname, _ in module.named_parameters(recurse=False):
                no_decay.append(f"{name}.{pname}" if name else pname)
    # Also include all "bias" parameters across the model
    for pname, _ in model.named_parameters():
        if pname.endswith(".bias"):
            no_decay.append(pname)
    return list(set(no_decay))


def _build_param_groups(
    model: nn.Module, weight_decay: float, lr: float
) -> List[Dict[str, Any]]:
    """Split model parameters into decay / no-decay groups."""
    no_decay = set(_no_decay_names(model))

    decay_params = [
        p for n, p in model.named_parameters()
        if n not in no_decay and p.requires_grad
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if n in no_decay and p.requires_grad
    ]

    logger.debug(
        "Param groups: %d decay params, %d no-decay params",
        len(decay_params), len(no_decay_params),
    )

    return [
        {"params": decay_params,    "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay_params, "weight_decay": 0.0,          "lr": lr},
    ]


# ---------------------------------------------------------------------------
# Optimizer builder
# ---------------------------------------------------------------------------

def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    """
    Build optimizer from configuration.

    Parameters
    ----------
    cfg : Config
    model : nn.Module

    Returns
    -------
    torch.optim.Optimizer
    """
    tcfg = cfg.training
    param_groups = _build_param_groups(model, tcfg.weight_decay, tcfg.learning_rate)

    opt_name = tcfg.optimizer.lower()
    if opt_name == "adamw":
        optimizer = AdamW(param_groups, lr=tcfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    elif opt_name == "adam":
        optimizer = Adam(param_groups, lr=tcfg.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    elif opt_name == "sgd":
        optimizer = SGD(param_groups, lr=tcfg.learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: '{tcfg.optimizer}'")

    logger.info(
        "Optimizer: %s (lr=%.2e, wd=%.2e)", opt_name, tcfg.learning_rate, tcfg.weight_decay
    )
    return optimizer


# ---------------------------------------------------------------------------
# Scheduler builder
# ---------------------------------------------------------------------------

def build_scheduler(
    cfg,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> tuple[Any, str]:
    """
    Build LR scheduler from configuration.

    Parameters
    ----------
    cfg : Config
    optimizer : torch.optim.Optimizer
    steps_per_epoch : int
        Number of optimizer steps per training epoch.
        (= ceil(len(train_loader) / grad_accum_steps))

    Returns
    -------
    (scheduler, step_frequency)
        step_frequency: "epoch" → scheduler.step() once per epoch
                        "step"  → scheduler.step() after every optimizer step
                        "metric" → scheduler.step(metric) at epoch end (ReduceLROnPlateau)
    """
    tcfg = cfg.training
    sched_name = tcfg.scheduler.lower()

    if sched_name == "cosine_warmup":
        total_steps = tcfg.max_epochs * steps_per_epoch
        warmup_steps = tcfg.warmup_epochs * steps_per_epoch
        min_lr_factor = tcfg.min_lr / tcfg.learning_rate

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay: 1 → min_lr_factor
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cos_decay

        # LambdaLR calls step() once during __init__ to set initial LR values.
        # This triggers a spurious "step before optimizer.step" warning from
        # PyTorch because no optimizer step has run yet.  The warning is
        # benign — suppress it for this one construction call only.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Detected call of `lr_scheduler.step\\(\\)` before",
                category=UserWarning,
            )
            scheduler = LambdaLR(optimizer, lr_lambda)
        freq = "step"
        logger.info(
            "Scheduler: cosine_warmup (warmup=%d steps, total=%d steps, min_lr=%.2e)",
            warmup_steps, total_steps, tcfg.min_lr,
        )

    elif sched_name == "step":
        step_size = getattr(tcfg, "scheduler_step_size", 30)
        gamma = getattr(tcfg, "scheduler_gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        freq = "epoch"
        logger.info("Scheduler: StepLR(step_size=%d, gamma=%.2f)", step_size, gamma)

    elif sched_name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=tcfg.early_stopping.mode,
            patience=tcfg.early_stopping.patience // 2,
            factor=0.5,
            min_lr=tcfg.min_lr,
        )
        freq = "metric"
        logger.info("Scheduler: ReduceLROnPlateau")

    elif sched_name == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=tcfg.learning_rate,
            total_steps=tcfg.max_epochs * steps_per_epoch,
            pct_start=tcfg.warmup_epochs / tcfg.max_epochs,
            anneal_strategy="cos",
            final_div_factor=tcfg.learning_rate / tcfg.min_lr,
        )
        freq = "step"
        logger.info("Scheduler: OneCycleLR")

    else:
        raise ValueError(f"Unknown scheduler: '{tcfg.scheduler}'")

    return scheduler, freq
