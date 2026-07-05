"""
ASD Trainer — full training loop with AMP, gradient accumulation, EMA, and K-fold CV.

Training loop details
---------------------
Per batch:
  1. Forward pass under autocast (AMP)
  2. Compute loss (focal/CE + genetics auxiliary loss)
  3. loss.backward() — scaled by GradScaler
  4. Accumulate for grad_accum_steps batches
  5. Unscale gradients → clip norm → optimizer.step() → scaler.update()
  6. EMA update
  7. Step-level scheduler update (cosine_warmup, onecycle)

Per epoch:
  1. Run training loop (above)
  2. Run validation loop (EMA model if available)
  3. Epoch-level scheduler update (step, plateau)
  4. Early stopping check
  5. Checkpoint if top-k improvement

K-fold CV:
  1. Per fold: load train/val splits, build fresh DataLoaders
  2. Reinitialise model weights (via model_factory callable)
  3. Run training loop
  4. Aggregate fold metrics

Mixed precision notes
---------------------
AMP is used on CUDA only.  On CPU the autocast is a no-op (no perf gain).
GradScaler is also CUDA-only; on CPU we use a dummy scaler that does nothing.
"""

from __future__ import annotations

import json
import logging
import math
import time
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dummy GradScaler for CPU training (avoids if/else in the training loop)
# ---------------------------------------------------------------------------

class _DummyScaler:
    def scale(self, loss): return loss
    def unscale_(self, optimizer): pass
    def step(self, optimizer): optimizer.step()
    def update(self): pass
    def state_dict(self): return {}
    def load_state_dict(self, _): pass


# ---------------------------------------------------------------------------
# Metric tracker
# ---------------------------------------------------------------------------

class _MetricTracker:
    """Accumulates per-batch metrics and computes epoch-level averages."""

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, key: str, value: float, n: int = 1) -> None:
        self._sums[key] = self._sums.get(key, 0.0) + value * n
        self._counts[key] = self._counts.get(key, 0) + n

    def mean(self, key: str) -> float:
        return self._sums[key] / max(self._counts[key], 1)

    def means(self) -> Dict[str, float]:
        return {k: self.mean(k) for k in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# ASDTrainer
# ---------------------------------------------------------------------------

class ASDTrainer:
    """
    Full training engine for the ASD detection framework.

    Parameters
    ----------
    cfg : Config
        Loaded configuration (config_schema.py).
    model_factory : Callable[[], ASDModel]
        Zero-argument callable that returns a freshly initialised ASDModel.
        Called once per fold.  Enables proper weight reinitialisation between
        folds (sharing weights across folds would violate independence).
    device : torch.device
    exp_logger : ExperimentLogger, optional
        Dual MLflow + WandB logger from utilities/logger.py.
    """

    def __init__(
        self,
        cfg,
        model_factory: Callable[[], nn.Module],
        device: torch.device,
        exp_logger=None,
    ) -> None:
        self.cfg = cfg
        self.model_factory = model_factory
        self.device = device
        self.exp_logger = exp_logger
        self.use_amp = device.type == "cuda"

        tcfg = cfg.training
        self.grad_accum = tcfg.gradient_accumulation_steps
        self.max_epochs = tcfg.max_epochs
        self.clip_norm = tcfg.gradient_clip_norm
        self.ema_decay = tcfg.ema_decay
        self.tabnet_lambda = getattr(tcfg, "tabnet_lambda", 1e-3)

        # Build loss criterion (shared across folds)
        from training.losses import build_criterion
        self.criterion = build_criterion(cfg).to(device)

    # -----------------------------------------------------------------------
    # K-fold cross-validation entry point
    # -----------------------------------------------------------------------

    def run_cv(
        self,
        dataset,
        splits: List[Tuple[List[int], List[int]]],
        save_dir: str | Path = "saved_models",
    ) -> Dict[str, Any]:
        """
        Run site-stratified K-fold cross-validation.

        Parameters
        ----------
        dataset : Dataset
            Full dataset (PairedMultiModalDataset or similar).
            Must support integer indexing.
        splits : list of (train_indices, val_indices) tuples
            Pre-computed site-stratified splits from create_subject_splits().
        save_dir : path
            Root directory for per-fold checkpoints.

        Returns
        -------
        dict:
          "fold_results" : list of per-fold metric dicts
          "mean_metrics" : averaged metrics across folds
          "std_metrics"  : std across folds
        """
        fold_results = []
        save_dir = Path(save_dir)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(
                "=== Fold %d / %d  (train=%d, val=%d) ===",
                fold_idx + 1, len(splits), len(train_idx), len(val_idx),
            )

            # Release GPU memory from the previous fold before allocating next
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Fresh model for each fold (no weight leakage across folds)
            model = self.model_factory().to(self.device)

            # torch.compile speeds up repeated forward passes via kernel fusion.
            # Requires PyTorch >= 2.0 + Triton (Linux only; not available on
            # Windows with the standard PyTorch wheel).
            if (
                self.device.type == "cuda"
                and hasattr(torch, "compile")
                and __import__("sys").platform != "win32"
            ):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("torch.compile enabled (mode=reduce-overhead)")
                except Exception as exc:
                    logger.debug("torch.compile skipped: %s", exc)
            elif self.device.type == "cuda":
                logger.debug(
                    "torch.compile skipped "
                    "(Windows: install triton via WSL2 for this optimisation)"
                )

            train_ds = Subset(dataset, train_idx)
            val_ds = Subset(dataset, val_idx)

            fold_save_dir = save_dir / f"fold_{fold_idx}"
            metrics = self.fit(
                train_dataset=train_ds,
                val_dataset=val_ds,
                model=model,
                fold_idx=fold_idx,
                save_dir=fold_save_dir,
            )
            fold_results.append(metrics)

            if self.exp_logger:
                self.exp_logger.log_metrics(
                    {f"fold{fold_idx}/{k}": v for k, v in metrics.items()},
                    step=fold_idx,
                )

        return self._aggregate_fold_results(fold_results)

    # -----------------------------------------------------------------------
    # Single fold training
    # -----------------------------------------------------------------------

    def fit(
        self,
        train_dataset,
        val_dataset,
        model: nn.Module,
        fold_idx: int = 0,
        save_dir: str | Path = "saved_models",
    ) -> Dict[str, float]:
        """
        Train for one fold.

        Returns
        -------
        dict of best validation metrics for this fold.
        """
        from training.early_stopping import EarlyStopping
        from training.checkpointing import CheckpointManager
        from training.ema import ModelEMA
        from training.optimizers import build_optimizer, build_scheduler

        cfg = self.cfg
        tcfg = cfg.training
        save_dir = Path(save_dir)

        # DataLoaders
        train_loader = self._make_loader(train_dataset, shuffle=True)
        val_loader = self._make_loader(val_dataset, shuffle=False)

        # Optimizer + scheduler
        optimizer = build_optimizer(cfg, model)
        steps_per_epoch = math.ceil(len(train_loader) / self.grad_accum)
        scheduler, sched_freq = build_scheduler(cfg, optimizer, steps_per_epoch)

        # AMP — use the device-agnostic API (torch.cuda.amp deprecated in 2.x)
        scaler = (
            torch.amp.GradScaler("cuda") if self.use_amp else _DummyScaler()
        )

        # EMA
        ema = ModelEMA(model, decay=self.ema_decay, device=self.device)

        # Callbacks
        es = EarlyStopping(
            patience=tcfg.early_stopping.patience,
            mode=tcfg.early_stopping.mode,
            min_delta=tcfg.early_stopping.min_delta,
            monitor=tcfg.early_stopping.monitor,
        )
        ckpt = CheckpointManager(
            save_dir=save_dir,
            top_k=tcfg.save_top_k,
            metric_name=tcfg.checkpoint_metric,
            mode=tcfg.early_stopping.mode,
            fold=fold_idx,
        )

        # Update balanced CE weights from training labels (if used)
        from training.losses import BalancedCrossEntropyLoss
        if isinstance(self.criterion, BalancedCrossEntropyLoss):
            all_labels = torch.tensor([
                train_dataset[i]["label"] for i in range(len(train_dataset))
            ])
            self.criterion.update_weights(all_labels)

        logger.info(
            "Fold %d: max_epochs=%d, steps_per_epoch=%d, "
            "effective_batch=%d (accum=%d)",
            fold_idx, self.max_epochs, steps_per_epoch,
            tcfg.batch_size * self.grad_accum, self.grad_accum,
        )

        history: List[Dict] = []

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            # ---- Training ----
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, scaler, ema,
                scheduler if sched_freq == "step" else None,
                epoch,
            )

            # ---- Validation (EMA model) ----
            val_metrics = self._validate(ema.shadow, val_loader)

            # ---- Epoch-level scheduler ----
            if sched_freq == "epoch":
                scheduler.step()
            elif sched_freq == "metric":
                scheduler.step(val_metrics[tcfg.early_stopping.monitor])

            # ---- Logging ----
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics["lr"] = optimizer.param_groups[0]["lr"]
            epoch_metrics["epoch"] = epoch
            history.append(epoch_metrics)

            elapsed = time.time() - t0
            logger.info(
                "Fold %d Epoch %3d/%d  loss=%.4f  val_auc=%.4f  val_acc=%.4f  %.1fs",
                fold_idx, epoch, self.max_epochs,
                train_metrics.get("train_loss", 0),
                val_metrics.get("val_auc", 0),
                val_metrics.get("val_acc", 0),
                elapsed,
            )

            if self.exp_logger:
                self.exp_logger.log_metrics(
                    {f"fold{fold_idx}/{k}": v for k, v in epoch_metrics.items()},
                    step=epoch,
                )

            # ---- Checkpointing ----
            ckpt.save(
                epoch=epoch, model=ema.shadow, optimizer=optimizer,
                metrics=val_metrics, scheduler=scheduler, ema=None,
                cfg=self.cfg,
            )

            # ---- Early stopping ----
            monitor_val = val_metrics.get(tcfg.early_stopping.monitor, 0.0)
            if es(monitor_val, epoch):
                break

        # Load best weights into model for final evaluation
        best_state = ckpt.load_best(model, device=self.device)

        return best_state.get("metrics", val_metrics)

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler,
        ema,
        step_scheduler,  # scheduler stepped per optimizer step (or None)
        epoch: int,
    ) -> Dict[str, float]:
        model.train()
        tracker = _MetricTracker()
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.amp.autocast("cuda") if self.use_amp else nullcontext()
        )

        for batch_idx, batch in enumerate(loader):
            mri = batch["image"].to(self.device, non_blocking=True)
            gen = batch["genetics"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            B = labels.shape[0]

            with autocast_ctx:
                output = model(mri, gen)
                logits = output["logits"]                # (B, num_classes)
                cls_loss = self.criterion(logits, labels)

                # Genetics auxiliary loss (TabNet sparsity; 0 for other backends)
                aux = getattr(model.gen_encoder, "last_aux_loss",
                              torch.zeros(1, device=self.device))
                loss = cls_loss + self.tabnet_lambda * aux

                # Scale for gradient accumulation
                loss = loss / self.grad_accum

            scaler.scale(loss).backward()

            # Optimizer step every grad_accum batches, or at epoch end
            is_accum_step = (batch_idx + 1) % self.grad_accum == 0
            is_last_batch = (batch_idx + 1) == len(loader)

            if is_accum_step or is_last_batch:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                if step_scheduler is not None:
                    step_scheduler.step()

            # Track metrics
            tracker.update("train_loss", cls_loss.item(), B)
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
            tracker.update("train_acc", acc, B)

        return {f"train_{k}" if not k.startswith("train_") else k: v
                for k, v in tracker.means().items()}

    # -----------------------------------------------------------------------
    # Validation step
    # -----------------------------------------------------------------------

    def _validate(
        self, model: nn.Module, loader: DataLoader
    ) -> Dict[str, float]:
        model.eval()
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        total_loss = 0.0
        n_samples = 0

        autocast_ctx = (
            torch.amp.autocast("cuda") if self.use_amp else nullcontext()
        )

        with torch.no_grad():
            for batch in loader:
                mri = batch["image"].to(self.device, non_blocking=True)
                gen = batch["genetics"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                B = labels.shape[0]

                with autocast_ctx:
                    output = model(mri, gen)
                    logits = output["logits"]
                    loss = self.criterion(logits, labels)

                all_logits.append(logits.float().cpu())
                all_labels.append(labels.cpu())
                total_loss += loss.item() * B
                n_samples += B

        all_logits = torch.cat(all_logits, dim=0)   # (N, num_classes)
        all_labels = torch.cat(all_labels, dim=0)   # (N,)

        return self._compute_val_metrics(all_logits, all_labels, total_loss / n_samples)

    # -----------------------------------------------------------------------
    # Metric computation
    # -----------------------------------------------------------------------

    def _compute_val_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        avg_loss: float,
    ) -> Dict[str, float]:
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        preds = logits.argmax(dim=-1).numpy()
        y = labels.numpy()

        acc = float((preds == y).mean())

        # AUC requires sklearn
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, probs))
        except Exception:
            auc = 0.0

        return {
            "val_loss": avg_loss,
            "val_acc": acc,
            "val_auc": auc,
        }

    # -----------------------------------------------------------------------
    # DataLoader builder
    # -----------------------------------------------------------------------

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        from utilities.reproducibility import get_worker_init_fn

        tcfg       = self.cfg.training
        pin        = self.device.type == "cuda"
        n_workers  = getattr(self.cfg.project, "num_workers", 4)

        kwargs: Dict[str, Any] = dict(
            batch_size  = tcfg.batch_size,
            shuffle     = shuffle,
            num_workers = n_workers,
            pin_memory  = pin,
            drop_last   = shuffle,
            worker_init_fn = get_worker_init_fn(getattr(self.cfg.project, "random_seed", 42)),
        )
        if n_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"]    = 2

        return DataLoader(dataset, **kwargs)

    # -----------------------------------------------------------------------
    # Fold aggregation
    # -----------------------------------------------------------------------

    @staticmethod
    def _aggregate_fold_results(
        fold_results: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        if not fold_results:
            return {}

        metric_keys = [k for k in fold_results[0] if isinstance(fold_results[0][k], float)]
        means, stds = {}, {}
        for k in metric_keys:
            vals = [r[k] for r in fold_results if k in r]
            means[k] = float(np.mean(vals))
            stds[k] = float(np.std(vals))

        logger.info("CV summary (mean ± std):")
        for k in metric_keys:
            logger.info("  %s: %.4f ± %.4f", k, means[k], stds[k])

        return {
            "fold_results": fold_results,
            "mean_metrics": means,
            "std_metrics": stds,
        }
