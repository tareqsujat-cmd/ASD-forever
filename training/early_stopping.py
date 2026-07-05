"""
Early stopping callback.

Monitors a validation metric and signals when training should stop
(no improvement beyond min_delta for patience consecutive epochs).

Why early stopping is critical for ABIDE
-----------------------------------------
ABIDE has ~500 training subjects per fold.  Without early stopping,
models overfit after ~30 epochs and validation AUC degrades.  With
patience=15 and monitor='val_auc', we capture the AUC peak reliably.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Parameters
    ----------
    patience : int
        Epochs to wait after last improvement before stopping.
    mode : str
        "max" → higher is better (AUC, accuracy).
        "min" → lower is better (loss).
    min_delta : float
        Minimum improvement to count as an improvement.
    monitor : str
        Metric name (for logging).
    """

    def __init__(
        self,
        patience: int = 15,
        mode: str = "max",
        min_delta: float = 1e-4,
        monitor: str = "val_auc",
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.monitor = monitor

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False

    def __call__(self, metric: float, epoch: int) -> bool:
        """
        Parameters
        ----------
        metric : float — current epoch's monitored metric
        epoch : int

        Returns
        -------
        bool — True if training should stop
        """
        improved = self._is_improved(metric)

        if improved:
            self.best_score = metric
            self.best_epoch = epoch
            self.counter = 0
            logger.info(
                "EarlyStopping: %s improved to %.4f at epoch %d",
                self.monitor, metric, epoch,
            )
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping: no improvement (%d/%d). Best=%.4f at epoch %d",
                self.counter, self.patience, self.best_score or 0.0, self.best_epoch,
            )

        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(
                "EarlyStopping triggered: no %s improvement for %d epochs. "
                "Best was %.4f at epoch %d.",
                self.monitor, self.patience,
                self.best_score or 0.0, self.best_epoch,
            )

        return self.should_stop

    def _is_improved(self, metric: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return metric > self.best_score + self.min_delta
        return metric < self.best_score - self.min_delta

    def reset(self) -> None:
        """Reset for a new fold."""
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    @property
    def best(self) -> Optional[float]:
        return self.best_score
