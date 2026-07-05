"""
Optuna callbacks for the ASD hyperparameter tuning pipeline.

All callbacks follow the Optuna signature::

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None

Available callbacks
-------------------
  ``ProgressCallback``    — prints per-trial progress to stdout
  ``CheckpointCallback``  — saves the best config as JSON after each improvement
  ``MLflowCallback``      — logs params + metrics to an MLflow experiment
  ``WandBCallback``       — logs params + metrics to Weights & Biases
  ``EarlyStoppingCallback`` — stops the study once n consecutive trials show
                              no improvement beyond ``min_delta``
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import optuna
from optuna.trial import FrozenTrial, TrialState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ProgressCallback
# ---------------------------------------------------------------------------

class ProgressCallback:
    """
    Prints a one-line summary after each completed trial.

    Parameters
    ----------
    metric : str
        Metric name to display (for informational purposes; Optuna's objective
        value is always shown).
    print_fn : callable, optional
        Defaults to ``print``.  Can be replaced with a logger.
    """

    def __init__(
        self,
        metric:   str      = "val_auc",
        print_fn: Callable = print,
    ) -> None:
        self.metric   = metric
        self.print_fn = print_fn
        self._start   = time.time()

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        elapsed = time.time() - self._start
        obj_str = (
            f"{trial.value:.5f}"
            if trial.value is not None
            else str(trial.values)
        )
        try:
            best = study.best_value
            best_str = f"  best={best:.5f}"
        except ValueError:
            best_str = ""

        self.print_fn(
            f"[{elapsed:7.1f}s] Trial {trial.number:04d}  "
            f"obj={obj_str}{best_str}  "
            f"params={trial.params}"
        )


# ---------------------------------------------------------------------------
# CheckpointCallback
# ---------------------------------------------------------------------------

class CheckpointCallback:
    """
    Saves the best trial's parameters as JSON whenever a new best is found.

    Parameters
    ----------
    save_path : path-like
        File to write (overwritten on each improvement).
    metric : str
        Metric key to track for best detection; used only for display —
        the actual best is determined by Optuna's objective value.
    """

    def __init__(
        self,
        save_path: Path | str,
        metric:    str = "val_auc",
    ) -> None:
        self.save_path    = Path(save_path)
        self.metric       = metric
        self._best_value: float = -math.inf

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        # Single-objective only
        try:
            best = study.best_trial
        except ValueError:
            return
        if best.number != trial.number:
            return

        # New best found
        obj_val = trial.value if trial.value is not None else float("nan")
        if obj_val <= self._best_value:
            return
        self._best_value = float(obj_val)

        payload = {
            "trial_number":  trial.number,
            "objective":     obj_val,
            "params":        trial.params,
            "datetime_utc":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "CheckpointCallback: new best trial %d (obj=%.5f) saved to %s",
            trial.number, obj_val, self.save_path,
        )


# ---------------------------------------------------------------------------
# EarlyStoppingCallback
# ---------------------------------------------------------------------------

class EarlyStoppingCallback:
    """
    Stops the study once ``patience`` consecutive trials show no improvement
    of more than ``min_delta`` in the best objective value.

    Only active for single-objective studies.

    Parameters
    ----------
    patience : int
        Number of consecutive non-improving trials before stopping.
    min_delta : float
        Minimum absolute change in the best value to count as improvement.
    direction : str
        ``"maximize"`` or ``"minimize"`` (must match the study direction).
    """

    def __init__(
        self,
        patience:  int   = 15,
        min_delta: float = 1e-4,
        direction: str   = "maximize",
    ) -> None:
        if direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")
        self.patience         = patience
        self.min_delta        = min_delta
        self._best: float     = -math.inf if direction == "maximize" else math.inf
        self._no_improve: int = 0
        self._direction       = direction

    def _improved(self, value: float) -> bool:
        if self._direction == "maximize":
            return value > self._best + self.min_delta
        return value < self._best - self.min_delta

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            return
        val = float(trial.value)
        if self._improved(val):
            self._best      = val
            self._no_improve = 0
        else:
            self._no_improve += 1

        if self._no_improve >= self.patience:
            logger.info(
                "EarlyStoppingCallback: no improvement for %d consecutive trials "
                "(patience=%d); stopping study.",
                self._no_improve, self.patience,
            )
            study.stop()

    @property
    def no_improve_count(self) -> int:
        return self._no_improve


# ---------------------------------------------------------------------------
# MLflowCallback
# ---------------------------------------------------------------------------

class MLflowCallback:
    """
    Logs each completed trial as an MLflow run.

    Requires ``mlflow`` to be installed.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment to log into.
    tracking_uri : str, optional
        MLflow tracking server URI.  Defaults to the local ``mlruns/`` folder.
    metric_prefix : str
        Prefix added to all logged metric names.
    """

    def __init__(
        self,
        experiment_name: str          = "asd_hpo",
        tracking_uri:    Optional[str] = None,
        metric_prefix:   str          = "",
    ) -> None:
        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            raise ImportError("mlflow is required for MLflowCallback: pip install mlflow")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.metric_prefix = metric_prefix
        self.experiment_name = experiment_name

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return
        with self._mlflow.start_run(run_name=f"trial_{trial.number:04d}"):
            # Log hyperparameters
            self._mlflow.log_params(trial.params)
            # Log objective value(s)
            if trial.value is not None:
                self._mlflow.log_metric(
                    f"{self.metric_prefix}objective", float(trial.value)
                )
            elif trial.values:
                for i, v in enumerate(trial.values):
                    self._mlflow.log_metric(
                        f"{self.metric_prefix}objective_{i}", float(v)
                    )
            # Log trial metadata
            self._mlflow.set_tag("trial_number", str(trial.number))
            self._mlflow.set_tag("study_name", study.study_name)


# ---------------------------------------------------------------------------
# WandBCallback
# ---------------------------------------------------------------------------

class WandBCallback:
    """
    Logs each completed trial to Weights & Biases.

    Requires ``wandb`` to be installed and ``wandb.init()`` to have been
    called beforehand (or pass ``project``/``entity`` to auto-init here).

    Parameters
    ----------
    project : str, optional
        W&B project name.  If provided, a new run is started per trial.
    entity : str, optional
        W&B entity (team/username).
    reinit : bool
        Whether to reinitialise wandb for each trial.
    metric_prefix : str
    """

    def __init__(
        self,
        project:       Optional[str] = None,
        entity:        Optional[str] = None,
        reinit:        bool          = True,
        metric_prefix: str           = "",
    ) -> None:
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError("wandb is required for WandBCallback: pip install wandb")
        self.project       = project
        self.entity        = entity
        self.reinit        = reinit
        self.metric_prefix = metric_prefix

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        if trial.state != TrialState.COMPLETE:
            return

        init_kwargs: Dict[str, Any] = {
            "reinit": self.reinit,
            "config": trial.params,
        }
        if self.project:
            init_kwargs["project"] = self.project
        if self.entity:
            init_kwargs["entity"]  = self.entity
        init_kwargs["name"] = f"trial_{trial.number:04d}"

        run = self._wandb.init(**init_kwargs)
        log_data: Dict[str, float] = {}
        if trial.value is not None:
            log_data[f"{self.metric_prefix}objective"] = float(trial.value)
        elif trial.values:
            for i, v in enumerate(trial.values):
                log_data[f"{self.metric_prefix}objective_{i}"] = float(v)
        self._wandb.log(log_data)
        run.finish()


# ---------------------------------------------------------------------------
# CompositeCallback
# ---------------------------------------------------------------------------

class CompositeCallback:
    """
    Bundles multiple callbacks into one object for cleaner ``ASDTuner`` init.

    Parameters
    ----------
    callbacks : list of callables
    """

    def __init__(self, callbacks: List[Callable]) -> None:
        self._callbacks = callbacks

    def __call__(self, study: optuna.Study, trial: FrozenTrial) -> None:
        for cb in self._callbacks:
            try:
                cb(study, trial)
            except Exception as e:
                logger.warning(
                    "Callback %s raised %s: %s",
                    type(cb).__name__, type(e).__name__, e,
                )
