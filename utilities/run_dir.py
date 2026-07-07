"""
Per-run output directory management for reproducible experiments.

Every training run gets its own auto-incrementing ``results/run_N/`` directory,
so runs never overwrite each other.  Each run directory captures everything
needed to reproduce and audit the run::

    results/run_1/
      run_manifest.json      # seed, device, git commit, args, versions, timing
      config_snapshot.yaml   # the exact resolved config used for this run
      run.log                # full log of the run
      <stage subdirs written by the pipeline: training/, evaluation/, ...>

The public entry points are :func:`create_run_dir` (allocate the directory),
:func:`attach_file_logger` (tee logs into it), :func:`snapshot_config`
(freeze the config), and :func:`write_manifest` / :func:`update_manifest`
(record reproducibility metadata).
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

RUN_PREFIX = "run_"
MANIFEST_NAME = "run_manifest.json"
CONFIG_SNAPSHOT_NAME = "config_snapshot.yaml"
LOG_NAME = "run.log"


# ---------------------------------------------------------------------------
# Run directory allocation
# ---------------------------------------------------------------------------

def _next_run_index(results_root: Path) -> int:
    """Return the next free integer N for ``run_N`` under ``results_root``."""
    existing = []
    for p in results_root.glob(f"{RUN_PREFIX}*"):
        if p.is_dir():
            suffix = p.name[len(RUN_PREFIX):]
            if suffix.isdigit():
                existing.append(int(suffix))
    return (max(existing) + 1) if existing else 1


def create_run_dir(results_root: str | Path, run_name: Optional[str] = None) -> Path:
    """
    Create and return a fresh run directory under ``results_root``.

    Parameters
    ----------
    results_root : path
        The results root (e.g. ``results/``).  Created if missing.
    run_name : str, optional
        Explicit run directory name.  If given and it already exists, a numeric
        suffix is appended to avoid clobbering.  If omitted, an auto-incrementing
        ``run_N`` name is used.

    Returns
    -------
    Path to the newly created run directory (guaranteed not to pre-exist).
    """
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    if run_name:
        run_dir = results_root / run_name
        if run_dir.exists():
            # Never overwrite an existing run — disambiguate with a suffix.
            i = 2
            while (results_root / f"{run_name}_{i}").exists():
                i += 1
            run_dir = results_root / f"{run_name}_{i}"
    else:
        idx = _next_run_index(results_root)
        run_dir = results_root / f"{RUN_PREFIX}{idx}"

    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info("Run directory: %s", run_dir.resolve())
    return run_dir


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def attach_file_logger(run_dir: str | Path, level: int = logging.INFO) -> logging.Handler:
    """
    Attach a file handler to the root logger so the full run log is written to
    ``run_dir/run.log`` (in addition to the console).  Returns the handler so
    callers can detach it when the run ends.
    """
    log_path = Path(run_dir) / LOG_NAME
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    return handler


def detach_file_logger(handler: Optional[logging.Handler]) -> None:
    """Detach and close a handler previously returned by attach_file_logger."""
    if handler is None:
        return
    logging.getLogger().removeHandler(handler)
    try:
        handler.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------

def snapshot_config(cfg: Any, run_dir: str | Path) -> Optional[Path]:
    """
    Write the fully resolved config to ``run_dir/config_snapshot.yaml``.

    Accepts a dataclass config (from configs.config_schema) or a plain dict.
    Returns the written path, or None if the config could not be serialised.
    """
    import dataclasses

    if dataclasses.is_dataclass(cfg):
        cfg_dict = dataclasses.asdict(cfg)
    elif isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        logger.warning("Cannot snapshot config of type %s", type(cfg).__name__)
        return None

    path = Path(run_dir) / CONFIG_SNAPSHOT_NAME
    try:
        import yaml
        path.write_text(
            yaml.safe_dump(cfg_dict, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
    except Exception:
        # Fall back to JSON if PyYAML is unavailable or a value is not YAML-safe.
        path = Path(run_dir) / "config_snapshot.json"
        path.write_text(json.dumps(cfg_dict, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Reproducibility manifest
# ---------------------------------------------------------------------------

def _git_info() -> Dict[str, Any]:
    def _run(cmd):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": (bool(status) if status is not None else None),
    }


def write_manifest(
    run_dir: str | Path,
    *,
    seed: int,
    device: Any,
    args: Any,
    mode: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write ``run_dir/run_manifest.json`` capturing everything needed to
    reproduce this run: seed, device, git commit/dirty state, CLI args,
    library versions, and platform.
    """
    import numpy
    import torch

    manifest: Dict[str, Any] = {
        "run_dir": str(Path(run_dir).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "seed": seed,
        "device": str(device),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {
            "torch": torch.__version__,
            "numpy": numpy.__version__,
        },
        "args": vars(args) if hasattr(args, "__dict__") else dict(args),
    }
    manifest.update(_git_info())
    if extra:
        manifest.update(extra)

    path = Path(run_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def update_manifest(run_dir: str | Path, extra: Dict[str, Any]) -> None:
    """Merge ``extra`` into an existing manifest (e.g. final metrics, timing)."""
    path = Path(run_dir) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    manifest.update(extra)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
