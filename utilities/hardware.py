"""
Hardware detection and GPU memory utilities.

Medical imaging models are memory-intensive (a single 96^3 float32 MRI volume
= 3.5 MB; batch of 8 = 28 MB for data alone, before activations).
This module provides helpers to:
  - Select the best available device
  - Estimate available GPU memory
  - Recommend batch size and gradient accumulation steps
  - Monitor GPU utilization during training
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def get_device(preference: str = "auto") -> torch.device:
    """
    Return the best available device, cascading to the next-best when the
    requested one is missing.

    Parameters
    ----------
    preference : str
        "auto" | "cuda" | "mps" | "cpu".  ``"auto"`` (and the historical default
        of ``"cuda"``) try CUDA → Apple MPS → CPU in order, so the same config
        runs GPU-accelerated on both an NVIDIA box and an Apple-silicon Mac.
        Pass an explicit ``"cpu"`` to force CPU.

    Returns
    -------
    torch.device
    """
    pref = (preference or "auto").lower()

    if pref == "cpu":
        logger.info("Using CPU (explicitly requested)")
        return torch.device("cpu")

    # CUDA first for "auto"/"cuda".
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        _log_gpu_memory()
        return device

    # Apple MPS next.  "cuda" cascades to MPS too (common on Macs where the
    # config default is "cuda" but no NVIDIA GPU exists).
    if pref in ("auto", "cuda", "mps") and _mps_available():
        # Route any op without an MPS kernel to the CPU instead of crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        logger.info("Using Apple MPS backend (Metal GPU)")
        return torch.device("mps")

    if pref == "cuda":
        logger.warning("CUDA/MPS requested but unavailable — falling back to CPU")
    elif pref == "mps":
        logger.warning("MPS requested but unavailable — falling back to CPU")
    else:
        logger.info("No GPU available — using CPU")
    return torch.device("cpu")


def get_gpu_memory_info() -> Tuple[float, float]:
    """
    Return (free_GB, total_GB) for the first CUDA device.

    Returns (0, 0) if no CUDA device is present.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info(0)
    return free / 1e9, total / 1e9


def _log_gpu_memory() -> None:
    free_gb, total_gb = get_gpu_memory_info()
    used_gb = total_gb - free_gb
    logger.info(f"GPU memory: {used_gb:.1f} GB used / {total_gb:.1f} GB total "
                f"({free_gb:.1f} GB free)")


def recommend_batch_config(
    volume_shape: Tuple[int, int, int] = (96, 96, 96),
    dtype_bytes: int = 4,
    target_batch_size: int = 8,
    safety_factor: float = 0.6,
) -> Tuple[int, int]:
    """
    Recommend (batch_size, gradient_accumulation_steps) given GPU memory.

    Uses a conservative memory estimate:
      - Forward pass activations ≈ 4× data size
      - Backward pass ≈ 2× forward
      - Safety factor to leave room for model weights and optimizer states

    Parameters
    ----------
    volume_shape : tuple
        (D, H, W) of the input volume.
    dtype_bytes : int
        Bytes per element (4 for float32, 2 for float16).
    target_batch_size : int
        Desired effective batch size.
    safety_factor : float
        Fraction of free GPU memory to use.

    Returns
    -------
    (actual_batch_size, grad_accum_steps)
    """
    free_gb, _ = get_gpu_memory_info()
    if free_gb == 0:
        return 1, target_batch_size  # CPU fallback

    D, H, W = volume_shape
    volume_bytes = D * H * W * dtype_bytes
    # Empirical: 6x overhead for activations + gradients in a 3D CNN
    per_sample_gb = (volume_bytes * 6) / 1e9
    max_batch = int((free_gb * safety_factor) / per_sample_gb)
    max_batch = max(1, min(max_batch, target_batch_size))
    grad_accum = max(1, target_batch_size // max_batch)

    logger.info(f"Recommended: batch_size={max_batch}, "
                f"grad_accum={grad_accum} "
                f"(effective={max_batch * grad_accum})")
    return max_batch, grad_accum


def setup_mixed_precision(enabled: bool = True) -> Optional[torch.cuda.amp.GradScaler]:
    """
    Initialize AMP GradScaler if mixed precision is enabled and CUDA is available.

    Mixed precision (FP16 for forward, FP32 for optimizer states) reduces
    memory by ~40% and speeds up conv ops on Tensor Core GPUs by 2-3×.

    Parameters
    ----------
    enabled : bool
        Whether to enable mixed precision.

    Returns
    -------
    GradScaler or None
    """
    if enabled and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision (AMP) enabled")
        return scaler
    logger.info("Mixed precision disabled (CPU or explicitly off)")
    return None


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total:,} total, {trainable:,} trainable "
                f"({trainable / total * 100:.1f}%)")
    return total, trainable
