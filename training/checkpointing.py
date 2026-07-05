"""
Checkpoint manager: saves and loads model checkpoints, keeping only top-k.

Saved checkpoint schema
------------------------
{
    "epoch": int,
    "fold": int,
    "model_state": OrderedDict,
    "ema_state": dict | None,
    "optimizer_state": dict,
    "scheduler_state": dict | None,
    "metrics": dict,       # {"val_auc": 0.85, "val_acc": 0.78, ...}
    "cfg_dict": dict,      # serialised config for reproducibility
}
"""

from __future__ import annotations

import heapq
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Parameters
    ----------
    save_dir : str or Path
        Directory where checkpoints are written.
    top_k : int
        Maximum number of checkpoints to retain.
    metric_name : str
        Key in the metrics dict to rank checkpoints by.
    mode : str
        "max" → higher is better.  "min" → lower is better.
    fold : int
        Current cross-validation fold (used in checkpoint filenames).
    """

    def __init__(
        self,
        save_dir: str | Path,
        top_k: int = 3,
        metric_name: str = "val_auc",
        mode: str = "max",
        fold: int = 0,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.metric_name = metric_name
        self.mode = mode
        self.fold = fold

        # Min-heap of (score, path) for tracking top-k
        # For "max" mode: negate scores so the smallest = worst
        self._heap: list = []  # heap of (heap_score, path_str)

    def _heap_score(self, metric: float) -> float:
        return -metric if self.mode == "max" else metric

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float],
        scheduler=None,
        ema=None,
        cfg=None,
    ) -> Optional[Path]:
        """
        Save a checkpoint if it improves upon the current top-k.

        Returns
        -------
        Path of saved checkpoint, or None if not in top-k.
        """
        score = metrics.get(self.metric_name, 0.0)
        h_score = self._heap_score(score)

        # Check if this beats the worst checkpoint in our top-k.
        # heap[0] is the MIN h_score (= best checkpoint in max mode), so we must
        # search for the MAX h_score to find the actual worst.
        if len(self._heap) >= self.top_k:
            worst_h_score = max(e[0] for e in self._heap)
            if h_score >= worst_h_score:  # not better than current worst → skip
                return None

        filename = (
            f"fold{self.fold}_epoch{epoch:04d}_"
            f"{self.metric_name}{score:.4f}.pt"
        )
        ckpt_path = self.save_dir / filename

        # Omit optimizer/scheduler state to cut checkpoint size ~50%.
        # Top-k checkpoints are for inference/evaluation, not training resumption.
        state = {
            "epoch": epoch,
            "fold": self.fold,
            "model_state": model.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "metrics": metrics,
        }
        if cfg is not None:
            # Serialise config to dict for portability
            try:
                import dataclasses
                state["cfg_dict"] = dataclasses.asdict(cfg)
            except Exception:
                pass

        torch.save(state, ckpt_path)
        logger.info("Checkpoint saved: %s (%s=%.4f)", filename, self.metric_name, score)

        # Maintain top-k heap — evict the WORST (max h_score), not the best.
        heapq.heappush(self._heap, (h_score, str(ckpt_path)))

        if len(self._heap) > self.top_k:
            worst_idx = max(range(len(self._heap)), key=lambda i: self._heap[i][0])
            _, old_path = self._heap[worst_idx]
            self._heap[worst_idx] = self._heap[-1]
            self._heap.pop()
            heapq.heapify(self._heap)
            Path(old_path).unlink(missing_ok=True)
            logger.debug("Evicted worst checkpoint: %s", old_path)

        return ckpt_path

    def best_checkpoint_path(self) -> Optional[Path]:
        """Return path of the best checkpoint (by metric)."""
        if not self._heap:
            return None
        if self.mode == "max":
            # Minimum heap_score = most negative = highest metric
            best_entry = min(self._heap, key=lambda x: x[0])
        else:
            best_entry = min(self._heap, key=lambda x: x[0])
        return Path(best_entry[1])

    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        ema=None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Load best checkpoint weights into model (and optionally optimizer/scheduler).

        Returns
        -------
        dict with "epoch", "metrics", and other checkpoint metadata.
        """
        path = self.best_checkpoint_path()
        if path is None or not path.exists():
            logger.warning("No checkpoint found in %s", self.save_dir)
            return {}

        state = torch.load(path, map_location=device or "cpu", weights_only=False)
        model.load_state_dict(state["model_state"])

        if ema is not None and state.get("ema_state") is not None:
            ema.load_state_dict(state["ema_state"])
        if optimizer is not None and state.get("optimizer_state"):
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])

        logger.info(
            "Loaded best checkpoint: %s (epoch %d, %s=%.4f)",
            path.name, state["epoch"],
            self.metric_name, state["metrics"].get(self.metric_name, 0.0),
        )
        return state

    @staticmethod
    def load_from_path(
        path: str | Path,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load weights from an explicit path (for inference / evaluation)."""
        state = torch.load(path, map_location=device or "cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        return state
