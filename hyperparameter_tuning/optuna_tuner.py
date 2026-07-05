"""
ASDTuner — Optuna Bayesian hyperparameter optimiser for the ASD framework.

Wraps an Optuna study and injects sampled hyperparameters into a user-supplied
``train_fn``.  Key features:

  - Single-objective (maximize val_auc) and multi-objective (maximize AUC,
    minimize Brier score) via NSGA-II
  - Median and Hyperband pruning for early elimination of poor trials
  - SQLite persistence for resumable studies across runs
  - Parallel execution via Optuna's n_jobs
  - Per-fold intermediate reporting for fold-level pruning

``train_fn`` contract
---------------------
::

    def train_fn(config, variant_name: str) -> List[Dict[str, float]]:
        # config        — modified config for this trial
        # variant_name  — "trial_NNNN" for logging
        # Returns K per-fold metric dicts

This is identical to the AblationRunner.train_fn contract so the same
training function works for both modules.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler
from optuna.pruners import MedianPruner, HyperbandPruner, NopPruner
from optuna.trial import TrialState

from hyperparameter_tuning.search_spaces import suggest_params

logger = logging.getLogger(__name__)

# Suppress Optuna's own verbose INFO by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# TrialRecord
# ---------------------------------------------------------------------------

class TrialRecord:
    """
    Lightweight record of one completed, pruned, or failed trial.

    Stored in ``ASDTuner.records`` independently of Optuna's database so that
    the full fold-level metrics and dot-notation overrides are accessible even
    when using an in-memory (non-persistent) study.
    """

    def __init__(
        self,
        trial_number:    int,
        params:          Dict[str, Any],
        overrides:       Dict[str, Any],
        mean_metrics:    Dict[str, float],
        fold_metrics:    List[Dict[str, float]],
        elapsed_seconds: float,
        state:           str = "complete",
    ) -> None:
        self.trial_number    = trial_number
        self.params          = params
        self.overrides       = overrides
        self.mean_metrics    = mean_metrics
        self.fold_metrics    = fold_metrics
        self.elapsed_seconds = elapsed_seconds
        self.state           = state  # "complete" | "pruned" | "failed"

    def get_metric(self, key: str) -> float:
        return self.mean_metrics.get(key, float("nan"))

    def to_dict(self) -> dict:
        return {
            "trial_number":    self.trial_number,
            "params":          self.params,
            "overrides":       self.overrides,
            "mean_metrics":    self.mean_metrics,
            "fold_metrics":    self.fold_metrics,
            "elapsed_seconds": self.elapsed_seconds,
            "state":           self.state,
        }

    def __repr__(self) -> str:
        return (
            f"TrialRecord(trial={self.trial_number}, state={self.state}, "
            f"metrics={self.mean_metrics})"
        )


# ---------------------------------------------------------------------------
# ASDTuner
# ---------------------------------------------------------------------------

class ASDTuner:
    """
    Bayesian hyperparameter optimiser for the ASD detection framework.

    Parameters
    ----------
    train_fn : callable
        ``(config, variant_name) -> List[Dict[str, float]]``.
        Returns K per-fold metric dicts; one dict per fold.
    base_config : any
        Starting config (dict or dataclass); deep-copied before each trial.
    search_space : str
        One of the search space names registered in ``search_spaces.py``.
    direction : str or list of str
        ``"maximize"`` or ``"minimize"``.  Pass a list for multi-objective.
    objectives : list of str, optional
        Metric keys corresponding to each direction.
        Single-objective default: ``["val_auc"]``.
    n_trials : int
        Maximum number of trials to run.
    timeout_seconds : float, optional
        Wall-clock time budget (in addition to n_trials).
    pruner : str
        ``"median"`` | ``"hyperband"`` | ``"none"``
    sampler : str
        ``"tpe"`` | ``"cmaes"`` | ``"nsgaii"``
    study_name : str
    storage_path : path-like, optional
        SQLite file path.  ``None`` → in-memory (no persistence).
    config_modifier : callable, optional
        Defaults to ``default_config_modifier`` from the ablation module.
    seed : int, optional
    callbacks : list of callable, optional
        Optuna callbacks ``(study, trial) -> None``.
    """

    def __init__(
        self,
        train_fn:         Callable,
        base_config,
        search_space:     str                               = "quick",
        direction:        Union[str, List[str]]             = "maximize",
        objectives:       Optional[List[str]]               = None,
        n_trials:         int                               = 50,
        timeout_seconds:  Optional[float]                   = None,
        pruner:           str                               = "median",
        sampler:          str                               = "tpe",
        study_name:       str                               = "asd_tuning",
        storage_path:     Optional[Union[str, Path]]        = None,
        config_modifier:  Optional[Callable]                = None,
        seed:             Optional[int]                     = 42,
        callbacks:        Optional[List[Callable]]          = None,
    ) -> None:
        self.train_fn        = train_fn
        self.base_config     = base_config
        self.search_space    = search_space
        self.n_trials        = n_trials
        self.timeout_seconds = timeout_seconds
        self.study_name      = study_name
        self.seed            = seed
        self.callbacks       = callbacks or []
        self._records: List[TrialRecord] = []

        if config_modifier is None:
            from ablation.ablation_runner import default_config_modifier
            config_modifier = default_config_modifier
        self.config_modifier = config_modifier

        # Single-objective vs multi-objective
        if isinstance(direction, str):
            self._directions = [direction]
            self._objectives = objectives or ["val_auc"]
        else:
            self._directions = list(direction)
            self._objectives = objectives if objectives else ["val_auc"] * len(direction)

        if len(self._directions) != len(self._objectives):
            raise ValueError(
                f"len(direction)={len(self._directions)} must equal "
                f"len(objectives)={len(self._objectives)}"
            )

        self._is_multi     = len(self._directions) > 1
        self._storage      = self._build_storage(storage_path)
        self._sampler      = self._build_sampler(sampler)
        self._pruner       = self._build_pruner(pruner)
        self._study: Optional[optuna.Study] = None

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_storage(path) -> Optional[str]:
        if path is None:
            return None
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{p.as_posix()}"

    def _build_sampler(self, name: str):
        kw = {"seed": self.seed} if self.seed is not None else {}
        if name == "tpe":
            return TPESampler(**kw, multivariate=True)
        elif name == "cmaes":
            if self._is_multi:
                logger.warning("CMA-ES does not support multi-objective; falling back to TPE")
                return TPESampler(**kw, multivariate=True)
            return CmaEsSampler(**kw)
        elif name == "nsgaii":
            seed_val = kw.get("seed", None)
            return NSGAIISampler(seed=seed_val)
        else:
            raise ValueError(f"Unknown sampler '{name}'; choose from tpe, cmaes, nsgaii")

    @staticmethod
    def _build_pruner(name: str):
        if name == "median":
            return MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        elif name == "hyperband":
            return HyperbandPruner(min_resource=1, max_resource=10, reduction_factor=3)
        elif name == "none":
            return NopPruner()
        else:
            raise ValueError(f"Unknown pruner '{name}'; choose from median, hyperband, none")

    # ------------------------------------------------------------------
    # Objective closure
    # ------------------------------------------------------------------

    def _make_objective(self) -> Callable:
        """Build the Optuna objective function as a closure over self."""

        def objective(trial: optuna.Trial) -> Union[float, Tuple[float, ...]]:
            # Sample hyperparameters and convert to dot-notation overrides
            overrides = suggest_params(trial, self.search_space)
            variant_name = f"trial_{trial.number:04d}"

            config = self.config_modifier(self.base_config, overrides)

            t0 = time.perf_counter()
            try:
                fold_metrics: List[Dict[str, float]] = self.train_fn(config, variant_name)
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                logger.warning("Trial %d raised %s: %s", trial.number, type(e).__name__, e)
                self._records.append(TrialRecord(
                    trial_number    = trial.number,
                    params          = dict(trial.params),
                    overrides       = overrides,
                    mean_metrics    = {},
                    fold_metrics    = [],
                    elapsed_seconds = time.perf_counter() - t0,
                    state           = "failed",
                ))
                raise optuna.exceptions.TrialPruned() from e

            # Aggregate per-fold metrics
            all_keys = sorted(set().union(*[f.keys() for f in fold_metrics]))
            mean_m: Dict[str, float] = {}
            for k in all_keys:
                vals = [f[k] for f in fold_metrics if k in f]
                if vals:
                    mean_m[k] = float(np.mean(vals))

            # Report each fold as an intermediate step for pruning.
            # trial.report / should_prune are only supported for single-objective.
            primary_key = self._objectives[0]
            for fold_idx, fm in enumerate(fold_metrics):
                if not self._is_multi and primary_key in fm:
                    trial.report(float(fm[primary_key]), step=fold_idx)
                    if trial.should_prune():
                        self._records.append(TrialRecord(
                            trial_number    = trial.number,
                            params          = dict(trial.params),
                            overrides       = overrides,
                            mean_metrics    = mean_m,
                            fold_metrics    = fold_metrics[: fold_idx + 1],
                            elapsed_seconds = time.perf_counter() - t0,
                            state           = "pruned",
                        ))
                        raise optuna.exceptions.TrialPruned()

            self._records.append(TrialRecord(
                trial_number    = trial.number,
                params          = dict(trial.params),
                overrides       = overrides,
                mean_metrics    = mean_m,
                fold_metrics    = fold_metrics,
                elapsed_seconds = time.perf_counter() - t0,
                state           = "complete",
            ))

            obj_vals = [mean_m.get(obj, float("nan")) for obj in self._objectives]
            return tuple(obj_vals) if self._is_multi else obj_vals[0]

        return objective

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_study(self) -> optuna.Study:
        """Create or reload a persistent Optuna study."""
        common = dict(
            study_name     = self.study_name,
            storage        = self._storage,
            sampler        = self._sampler,
            load_if_exists = True,
        )
        if self._is_multi:
            study = optuna.create_study(directions=self._directions, **common)
        else:
            study = optuna.create_study(
                direction = self._directions[0],
                pruner    = self._pruner,
                **common,
            )
        self._study = study
        return study

    def optimize(self) -> optuna.Study:
        """
        Run the optimisation loop and return the completed study.

        Results are accessible via ``self.records``, ``self.best_trial``,
        ``self.best_params``, and ``self.pareto_front``.
        """
        if self._study is None:
            self.create_study()

        # n_trials is a TOTAL budget, not "additional" trials.
        # Subtract already-completed trials so reloading a study doesn't overrun.
        already_done = sum(
            1 for t in self._study.trials
            if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
        )
        remaining = max(0, self.n_trials - already_done)
        logger.info(
            "HPO: target=%d total trials, %d already done, running %d more",
            self.n_trials, already_done, remaining,
        )
        if remaining == 0:
            return self._study

        self._study.optimize(
            self._make_objective(),
            n_trials          = remaining,
            timeout           = self.timeout_seconds,
            callbacks         = self.callbacks or None,
            show_progress_bar = False,
            gc_after_trial    = True,
        )

        n_complete = sum(1 for t in self._study.trials if t.state == TrialState.COMPLETE)
        n_pruned   = sum(1 for t in self._study.trials if t.state == TrialState.PRUNED)
        logger.info(
            "Study '%s' done: %d complete, %d pruned",
            self.study_name, n_complete, n_pruned,
        )
        return self._study

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    @property
    def records(self) -> List[TrialRecord]:
        """All TrialRecord objects produced in this tuning session."""
        return list(self._records)

    @property
    def study(self) -> Optional[optuna.Study]:
        return self._study

    @property
    def best_trial(self) -> Optional[optuna.trial.FrozenTrial]:
        """Best trial (single-objective only; None for multi-objective)."""
        if self._study is None or self._is_multi:
            return None
        try:
            return self._study.best_trial
        except ValueError:
            return None

    @property
    def best_params(self) -> Dict[str, Any]:
        """
        Best hyperparameters as dot-notation overrides (single-objective).

        For multi-objective studies use ``pareto_front`` instead.
        """
        bt = self.best_trial
        if bt is None:
            return {}
        rec = next((r for r in self._records if r.trial_number == bt.number), None)
        return rec.overrides if rec is not None else {}

    @property
    def best_value(self) -> float:
        """Best objective value (single-objective; nan otherwise)."""
        bt = self.best_trial
        return bt.value if bt is not None and bt.value is not None else float("nan")

    @property
    def pareto_front(self) -> List[optuna.trial.FrozenTrial]:
        """Pareto-optimal trials (multi-objective only)."""
        if self._study is None or not self._is_multi:
            return []
        return self._study.best_trials

    def get_n_best(self, n: int = 5, metric: str = "val_auc") -> List[TrialRecord]:
        """Return top-n complete trial records sorted by metric (descending)."""
        complete = [r for r in self._records if r.state == "complete"]
        return sorted(complete, key=lambda r: r.get_metric(metric), reverse=True)[:n]

    def summary_dict(self) -> dict:
        """Compact summary dict for logging or reporting."""
        n_complete = sum(1 for r in self._records if r.state == "complete")
        n_pruned   = sum(1 for r in self._records if r.state == "pruned")
        n_failed   = sum(1 for r in self._records if r.state == "failed")
        best = self.get_n_best(1, self._objectives[0])
        return {
            "study_name":  self.study_name,
            "n_complete":  n_complete,
            "n_pruned":    n_pruned,
            "n_failed":    n_failed,
            "n_total":     len(self._records),
            "best_metric": {
                obj: best[0].get_metric(obj) if best else float("nan")
                for obj in self._objectives
            },
            "best_params": self.best_params,
        }
