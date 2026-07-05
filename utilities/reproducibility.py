"""
Reproducibility utilities.

ABIDE is a multi-site dataset with high variance.  A single global seed is
insufficient: two calls to the same function on different machines can diverge
if the PyTorch PRNG state was consumed by an earlier call.  This module
provides per-scope seeds derived deterministically from the master seed so
that each experiment component is independently reproducible.

Usage
-----
    from utilities.reproducibility import seed_everything, ScopedSeed

    seed_everything(42)            # global seed at startup

    with ScopedSeed("data_split", base_seed=42):
        # data split logic here
        ...
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Set all random seeds for full reproducibility.

    Parameters
    ----------
    seed : int
        Master random seed.
    deterministic_cudnn : bool
        If True, sets cuDNN to deterministic mode.  This trades a small
        performance hit for reproducible convolution results across runs.
        Always True for published experiments.

    Notes
    -----
    torch.use_deterministic_algorithms(True) requires CUBLAS_WORKSPACE_CONFIG
    to be set when using CUDA >= 10.2.  This is handled automatically here.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for deterministic CUDA ops in PyTorch >= 1.8
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass  # older torch version

    logger.info(f"Global seed set to {seed} (deterministic_cudnn={deterministic_cudnn})")


def derive_seed(scope: str, base_seed: int = 42) -> int:
    """
    Deterministically derive a seed for a named scope from the base seed.

    Uses SHA-256 so the derived seeds are uniform and collision-resistant
    even for very similar scope names.

    Parameters
    ----------
    scope : str
        Unique name for this scope (e.g. "fold_3_train", "augmentation").
    base_seed : int
        Master seed to derive from.

    Returns
    -------
    int
        A 32-bit unsigned integer seed.
    """
    digest = hashlib.sha256(f"{base_seed}:{scope}".encode()).hexdigest()
    return int(digest[:8], 16)  # Take first 4 bytes -> 32-bit uint


@contextmanager
def ScopedSeed(scope: str, base_seed: int = 42):
    """
    Context manager that applies a scoped seed and restores RNG state on exit.

    This allows deterministic sub-operations without permanently altering the
    global RNG state, enabling independent reproducibility per component.

    Parameters
    ----------
    scope : str
        Descriptive name for this seed scope.
    base_seed : int
        Master seed to derive from.

    Example
    -------
        with ScopedSeed("kfold_split", base_seed=42):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)
    """
    scoped = derive_seed(scope, base_seed)

    # Save current RNG states
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]

    # Apply scoped seed
    random.seed(scoped)
    np.random.seed(scoped)
    torch.manual_seed(scoped)
    torch.cuda.manual_seed_all(scoped)

    logger.debug(f"ScopedSeed '{scope}' -> {scoped}")

    try:
        yield scoped
    finally:
        # Restore RNG states
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        for i, state in enumerate(cuda_states):
            torch.cuda.set_rng_state(state, i)


class _WorkerInitFn:
    """
    Picklable worker init callable for DataLoader multiprocessing.

    A plain closure cannot be pickled on Windows (spawn start method),
    so we use a class with __call__ which pickle handles correctly.
    """

    def __init__(self, base_seed: int) -> None:
        self.base_seed = base_seed

    def __call__(self, worker_id: int) -> None:
        worker_seed = derive_seed(f"worker_{worker_id}", self.base_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


def get_worker_init_fn(base_seed: int = 42) -> _WorkerInitFn:
    """
    Return a DataLoader worker_init_fn that seeds each worker deterministically.

    Without this, multiple DataLoader workers produce non-deterministic results
    because they inherit the same RNG state and diverge unpredictably.

    Returns a picklable callable (required for Windows spawn multiprocessing).
    """
    return _WorkerInitFn(base_seed)
