"""
Structured logging setup for the ASD framework.

Provides a single get_logger() factory that:
  - Routes DEBUG+ to console (colorized via rich if available)
  - Routes INFO+  to a rotating file in results/logs/
  - Attaches MLflow/WandB step logging when trackers are active

Usage
-----
    from utilities.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training epoch 1/100")
    logger.metric("val_auc", 0.834, step=1)   # custom level
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Custom METRIC level (between INFO=20 and WARNING=30)
# ---------------------------------------------------------------------------
METRIC_LEVEL = 25
logging.addLevelName(METRIC_LEVEL, "METRIC")


def _metric(self, message, *args, **kwargs):
    if self.isEnabledFor(METRIC_LEVEL):
        self._log(METRIC_LEVEL, message, args, **kwargs)


logging.Logger.metric = _metric  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Rich handler (optional, graceful fallback)
# ---------------------------------------------------------------------------
def _make_console_handler(level: int) -> logging.Handler:
    try:
        from rich.logging import RichHandler
        handler = RichHandler(
            level=level,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
    return handler


def _make_file_handler(log_dir: Path, filename: str, level: int) -> logging.Handler:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    return handler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_initialized_root = False


def setup_root_logger(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    filename: str = "asd_framework.log",
) -> None:
    """
    Initialize the root logger once at application startup.

    Call this in your main script before importing any modules that log.

    Parameters
    ----------
    level : str
        Minimum log level string: DEBUG | INFO | METRIC | WARNING | ERROR.
    log_dir : str, optional
        Directory for the rotating log file.  If None, file logging is skipped.
    filename : str
        Log file name.
    """
    global _initialized_root
    if _initialized_root:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Root captures everything; handlers filter

    # Console: show INFO+ by default (DEBUG in dev mode)
    console_level = logging.DEBUG if os.environ.get("ASD_DEBUG") else numeric_level
    root.addHandler(_make_console_handler(console_level))

    # File: always INFO+
    if log_dir:
        root.addHandler(_make_file_handler(Path(log_dir), filename, logging.INFO))

    # Silence noisy third-party loggers
    for noisy in ["matplotlib", "PIL", "numexpr", "h5py", "urllib3", "fsspec"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized_root = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Parameters
    ----------
    name : str
        Logger name, typically __name__.

    Returns
    -------
    logging.Logger
        Named logger with the custom .metric() method available.
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Experiment step logger — wraps MLflow / WandB log calls
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Thin wrapper that simultaneously logs metrics to:
      - Python logger (for file/console output)
      - MLflow (if active run exists)
      - Weights & Biases (if run is initialized)

    Parameters
    ----------
    name : str
        Module name for the Python logger.
    """

    def __init__(self, name: str) -> None:
        self._log = get_logger(name)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log a dict of scalar metrics to all active trackers."""
        # Python log
        parts = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in metrics.items())
        self._log.metric(f"step={step} | {parts}")  # type: ignore[attr-defined]

        # MLflow
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

        # WandB
        try:
            import wandb
            if wandb.run is not None:
                log_dict = dict(metrics)
                if step is not None:
                    log_dict["_step"] = step
                wandb.log(log_dict)
        except Exception:
            pass

    def log_params(self, params: dict) -> None:
        """Log hyperparameters to all active trackers."""
        self._log.info(f"Params: {params}")

        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_params(params)
        except Exception:
            pass

        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(params)
        except Exception:
            pass

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log a file artifact (figures, configs, checkpoints)."""
        self._log.info(f"Artifact: {path}")
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_artifact(path, artifact_path)
        except Exception:
            pass

        try:
            import wandb
            if wandb.run is not None:
                wandb.save(path)
        except Exception:
            pass
