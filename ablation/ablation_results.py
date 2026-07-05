"""
Ablation result containers with JSON persistence.

``AblationResults`` stores per-variant, per-fold metrics and supports
incremental saving so that long studies can be resumed after interruption.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VariantResult
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VariantResult:
    """All measurements for one ablation variant."""

    name: str
    fold_metrics: List[Dict[str, float]]  # one dict per fold
    mean_metrics: Dict[str, float]        # mean across folds
    std_metrics:  Dict[str, float]        # std across folds
    elapsed_seconds: float = 0.0
    timestamp: str = ""

    @classmethod
    def from_fold_metrics(
        cls,
        name: str,
        fold_metrics: List[Dict[str, float]],
        elapsed_seconds: float = 0.0,
    ) -> "VariantResult":
        """Compute summary statistics from raw per-fold metrics."""
        # Collect all metric keys present in at least one fold
        all_keys = sorted(set().union(*[f.keys() for f in fold_metrics]))
        mean_m: Dict[str, float] = {}
        std_m:  Dict[str, float] = {}
        for k in all_keys:
            vals = [f[k] for f in fold_metrics if k in f]
            if vals:
                mean_m[k] = float(np.mean(vals))
                std_m[k]  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        return cls(
            name=name,
            fold_metrics=fold_metrics,
            mean_metrics=mean_m,
            std_metrics=std_m,
            elapsed_seconds=elapsed_seconds,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_metric(self, key: str) -> float:
        """Return mean value for a metric key (nan if missing)."""
        return self.mean_metrics.get(key, float("nan"))

    def get_fold_values(self, key: str) -> List[float]:
        """Return per-fold values for a metric key."""
        return [f[key] for f in self.fold_metrics if key in f]

    def summary_str(self, key: str) -> str:
        """Format as 'mean ± std' for a metric."""
        m = self.mean_metrics.get(key, float("nan"))
        s = self.std_metrics.get(key, float("nan"))
        return f"{m:.4f} ± {s:.4f}"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# AblationResults
# ---------------------------------------------------------------------------

class AblationResults:
    """
    Container for all variant results in an ablation study.

    Supports incremental saving so that interrupted studies can be resumed
    without re-running completed variants.

    Parameters
    ----------
    study_name : str
        Propagated from the ``AblationStudy`` for file naming.
    """

    def __init__(self, study_name: str = "ablation") -> None:
        self.study_name = study_name
        self._variants: Dict[str, VariantResult] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        fold_metrics: List[Dict[str, float]],
        elapsed_seconds: float = 0.0,
    ) -> VariantResult:
        """Add results for one variant (overwrites if already present)."""
        result = VariantResult.from_fold_metrics(name, fold_metrics, elapsed_seconds)
        self._variants[name] = result
        logger.info(
            "Added variant '%s': %s",
            name,
            {k: f"{v:.4f}" for k, v in result.mean_metrics.items()},
        )
        return result

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._variants

    def __len__(self) -> int:
        return len(self._variants)

    def get(self, name: str) -> Optional[VariantResult]:
        return self._variants.get(name)

    def all_variants(self) -> List[str]:
        return list(self._variants.keys())

    def all_results(self) -> List[VariantResult]:
        return list(self._variants.values())

    def metric_matrix(self, key: str) -> Dict[str, float]:
        """Return {variant_name: mean_value} for a given metric."""
        return {
            name: r.get_metric(key)
            for name, r in self._variants.items()
        }

    def available_metrics(self) -> List[str]:
        """Union of all metric keys seen across all variants."""
        keys = set()
        for r in self._variants.values():
            keys.update(r.mean_metrics.keys())
        return sorted(keys)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_json(self, path: str | Path) -> None:
        """Serialize all results to a JSON file (pretty-printed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "study_name": self.study_name,
            "n_variants": len(self._variants),
            "variants": {name: r.to_dict() for name, r in self._variants.items()},
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved %d variant results to %s", len(self._variants), path)

    @classmethod
    def load_json(cls, path: str | Path) -> "AblationResults":
        """Load results from a previously saved JSON file."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        results = cls(study_name=payload.get("study_name", "ablation"))
        for name, d in payload.get("variants", {}).items():
            vr = VariantResult(
                name=d["name"],
                fold_metrics=d["fold_metrics"],
                mean_metrics=d["mean_metrics"],
                std_metrics=d["std_metrics"],
                elapsed_seconds=d.get("elapsed_seconds", 0.0),
                timestamp=d.get("timestamp", ""),
            )
            results._variants[name] = vr
        logger.info("Loaded %d variant results from %s", len(results), path)
        return results

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AblationResults(study='{self.study_name}', "
            f"n_variants={len(self._variants)})"
        )
