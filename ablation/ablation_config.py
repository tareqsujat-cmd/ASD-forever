"""
Ablation study configuration: dimensions, variants, and study generation.

An ``AblationStudy`` is composed of one or more ``AblationDimension`` objects.
Each dimension names one aspect of the model that varies (e.g. fusion strategy,
MRI backbone), and lists all values that dimension can take together with a
designated default (baseline) value.

Two generation modes are supported:

  ``"ofat"`` (One-Factor-At-a-Time, default)
      One baseline run (all dimensions at their defaults) plus, for each
      dimension, one run per non-default value.  Total = 1 + Σ(|dim| - 1).
      This is the standard ablation protocol for MICCAI / IEEE papers.

  ``"factorial"``
      All |dim_1| × |dim_2| × … combinations.  Exponential; only practical
      for ≤ 3 small dimensions.

  ``"custom"``
      User-supplied list of (name, overrides) tuples; generation is a no-op.

Override format
---------------
Each variant override is a flat dict of dot-notation keys → values, e.g.::

    {"fusion.architecture": "gated", "training.epochs": 50}

The runner's ``config_modifier`` is responsible for applying these overrides
to the actual config object.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# AblationDimension
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AblationDimension:
    """
    One axis of variation in an ablation study.

    Parameters
    ----------
    name : str
        Short identifier, e.g. ``"fusion"``.  Used as a prefix in variant names.
    variants : dict[str, dict]
        Maps variant label → flat override dict.
        Example::

            {
                "cross_attention": {"fusion.architecture": "cross_attention"},
                "gated":           {"fusion.architecture": "gated"},
                "late":            {"fusion.architecture": "late"},
            }
    default : str
        Key in ``variants`` that is the baseline value.
    description : str
        Human-readable description for table captions.
    """

    name: str
    variants: Dict[str, Dict[str, Any]]
    default: str
    description: str = ""

    def __post_init__(self):
        if self.default not in self.variants:
            raise ValueError(
                f"AblationDimension '{self.name}': default '{self.default}' "
                f"not in variants {list(self.variants.keys())}"
            )

    @property
    def non_default_variants(self) -> Dict[str, Dict[str, Any]]:
        return {k: v for k, v in self.variants.items() if k != self.default}

    @property
    def default_overrides(self) -> Dict[str, Any]:
        return self.variants[self.default]


# ---------------------------------------------------------------------------
# AblationStudy
# ---------------------------------------------------------------------------

class AblationStudy:
    """
    An ablation study: a named collection of dimensions and a generation mode.

    Parameters
    ----------
    name : str
        Study identifier (used in filenames and table captions).
    base_config : any
        The starting config object (dict, dataclass, …).  Copied and then
        modified by the runner's ``config_modifier``.
    dimensions : list of AblationDimension
    mode : ``"ofat"`` | ``"factorial"`` | ``"custom"``
    custom_variants : list of (name, overrides_dict), required when mode="custom"
    """

    def __init__(
        self,
        name: str,
        base_config,
        dimensions: List[AblationDimension],
        mode: str = "ofat",
        custom_variants: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    ) -> None:
        if mode not in ("ofat", "factorial", "custom"):
            raise ValueError(f"mode must be 'ofat', 'factorial', or 'custom'; got '{mode}'")
        if mode == "custom" and not custom_variants:
            raise ValueError("mode='custom' requires custom_variants")

        self.name = name
        self.base_config = base_config
        self.dimensions = dimensions
        self.mode = mode
        self._custom_variants = custom_variants or []

    # ------------------------------------------------------------------

    def generate_variants(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Return all (variant_name, overrides_dict) pairs for this study.

        Variant names follow the convention::

            "baseline"                     — all defaults (OFAT only)
            "fusion=gated"                 — single dimension varied
            "fusion=gated__backbone=swin"  — multiple dimensions varied (factorial)
        """
        if self.mode == "custom":
            return list(self._custom_variants)
        if self.mode == "ofat":
            return self._generate_ofat()
        return self._generate_factorial()

    def _generate_ofat(self) -> List[Tuple[str, Dict[str, Any]]]:
        variants: List[Tuple[str, Dict[str, Any]]] = []

        # Baseline: merge all default overrides
        baseline_overrides: Dict[str, Any] = {}
        for dim in self.dimensions:
            baseline_overrides.update(dim.default_overrides)
        variants.append(("baseline", baseline_overrides))

        # One-factor-at-a-time
        for dim in self.dimensions:
            for val_name, val_overrides in dim.non_default_variants.items():
                # Start from all defaults, then override this dimension
                combined = copy.copy(baseline_overrides)
                combined.update(val_overrides)
                variant_name = f"{dim.name}={val_name}"
                variants.append((variant_name, combined))

        return variants

    def _generate_factorial(self) -> List[Tuple[str, Dict[str, Any]]]:
        variants: List[Tuple[str, Dict[str, Any]]] = []

        dim_items = [
            [(label, overrides) for label, overrides in dim.variants.items()]
            for dim in self.dimensions
        ]

        for combo in itertools.product(*dim_items):
            name_parts = []
            merged: Dict[str, Any] = {}
            for (dim, (label, overrides)) in zip(self.dimensions, combo):
                name_parts.append(f"{dim.name}={label}")
                merged.update(overrides)
            variant_name = "__".join(name_parts)
            variants.append((variant_name, merged))

        return variants

    def num_variants(self) -> int:
        return len(self.generate_variants())

    def __repr__(self) -> str:
        return (
            f"AblationStudy(name='{self.name}', mode='{self.mode}', "
            f"dimensions={[d.name for d in self.dimensions]}, "
            f"n_variants={self.num_variants()})"
        )
