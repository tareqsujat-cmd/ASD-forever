"""
AblationRunner — executes an AblationStudy by training and evaluating all
variant configurations.

The runner is fully decoupled from the model training code: all training
logic is injected as a callable ``train_fn``.  This makes the runner easily
testable with a mock and usable with any training framework.

``train_fn`` contract
---------------------
::

    def train_fn(config, variant_name: str) -> List[Dict[str, float]]:
        # config  : the (potentially modified) config object for this variant
        # Returns : list of K per-fold metric dicts, e.g.
        #           [{"val_auc": 0.85, "val_acc": 0.78}, ...]

``config_modifier`` contract
-----------------------------
::

    def config_modifier(base_config, overrides: dict):
        # base_config : original config object (deep-copied before calling)
        # overrides   : flat {dot.key: value} dict from AblationDimension
        # Returns     : modified config (may be a new object or the same)

A default modifier is provided that works for both plain dicts and objects
with attribute access.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from ablation.ablation_config import AblationStudy
from ablation.ablation_results import AblationResults

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default config modifier
# ---------------------------------------------------------------------------

def default_config_modifier(base_config, overrides: dict):
    """
    Apply flat dot-notation overrides to a config object or dict.

    Supports both attribute-access objects (dataclasses, SimpleNamespace) and
    plain nested dicts.  Creates a deep copy before modification.
    """
    cfg = copy.deepcopy(base_config)
    for dotkey, value in overrides.items():
        parts = dotkey.split(".")
        obj = cfg
        for part in parts[:-1]:
            if isinstance(obj, dict):
                obj = obj.setdefault(part, {})
            else:
                obj = getattr(obj, part)
        final = parts[-1]
        if isinstance(obj, dict):
            obj[final] = value
        else:
            setattr(obj, final, value)
    return cfg


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------

class AblationRunner:
    """
    Runs all variants in an ``AblationStudy``.

    Parameters
    ----------
    train_fn : callable
        ``(config, variant_name) -> List[Dict[str, float]]``
        Must return K per-fold metric dicts for the provided config variant.
    save_dir : str or Path
        Directory for incremental result checkpointing.
    config_modifier : callable, optional
        ``(base_config, overrides) -> new_config``
        Defaults to ``default_config_modifier``.
    verbose : bool
        Log progress per variant.
    """

    def __init__(
        self,
        train_fn: Callable,
        save_dir: Union[str, Path] = "ablation_results",
        config_modifier: Optional[Callable] = None,
        verbose: bool = True,
    ) -> None:
        self.train_fn = train_fn
        self.save_dir = Path(save_dir)
        self.config_modifier = config_modifier or default_config_modifier
        self.verbose = verbose

    # ------------------------------------------------------------------

    def run_study(
        self,
        study: AblationStudy,
        resume: bool = True,
    ) -> AblationResults:
        """
        Execute all variants in ``study``.

        Parameters
        ----------
        study : AblationStudy
        resume : bool
            If True, load existing results from ``save_dir/results.json`` and
            skip variants already present.  Enables recovery from crashes.

        Returns
        -------
        AblationResults — complete results for all variants
        """
        self.save_dir.mkdir(parents=True, exist_ok=True)
        results_path = self.save_dir / f"{study.name}_results.json"

        # Load existing results if resuming
        results = AblationResults(study_name=study.name)
        if resume and results_path.exists():
            try:
                results = AblationResults.load_json(results_path)
                logger.info(
                    "Resuming study '%s': %d/%d variants already done",
                    study.name, len(results), study.num_variants()
                )
            except Exception as e:
                logger.warning("Could not load checkpoint: %s — starting fresh", e)

        variants = study.generate_variants()
        n_total = len(variants)

        for idx, (variant_name, overrides) in enumerate(variants, 1):
            if resume and variant_name in results:
                if self.verbose:
                    logger.info("[%d/%d] Skipping '%s' (already done)",
                                idx, n_total, variant_name)
                continue

            if self.verbose:
                logger.info("[%d/%d] Running variant '%s'", idx, n_total, variant_name)

            try:
                config = self.config_modifier(study.base_config, overrides)
                t0 = time.perf_counter()
                fold_metrics = self.train_fn(config, variant_name)
                elapsed = time.perf_counter() - t0
            except Exception as e:
                logger.error(
                    "Variant '%s' failed with %s: %s",
                    variant_name, type(e).__name__, e
                )
                # Store failure so it doesn't block resume
                results.add(variant_name, [{"error": 1.0}], elapsed_seconds=0.0)
                results.save_json(results_path)
                continue

            results.add(variant_name, fold_metrics, elapsed_seconds=elapsed)

            # Save incrementally after every variant
            results.save_json(results_path)

            if self.verbose:
                r = results.get(variant_name)
                metrics_str = "  ".join(
                    f"{k}={v:.4f}" for k, v in r.mean_metrics.items()
                    if not k.startswith("error")
                )
                logger.info("  → %s", metrics_str)

        logger.info(
            "Study '%s' complete: %d/%d variants succeeded",
            study.name,
            sum(1 for r in results.all_results()
                if "error" not in r.mean_metrics),
            n_total,
        )
        return results

    def run_single(
        self,
        base_config,
        variant_name: str,
        overrides: dict,
    ) -> AblationResults:
        """
        Run a single variant outside of a study (useful for debugging).

        Returns
        -------
        AblationResults containing only this variant.
        """
        results = AblationResults(study_name=variant_name)
        config = self.config_modifier(base_config, overrides)
        t0 = time.perf_counter()
        fold_metrics = self.train_fn(config, variant_name)
        elapsed = time.perf_counter() - t0
        results.add(variant_name, fold_metrics, elapsed_seconds=elapsed)
        return results
