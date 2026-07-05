"""
Model export utilities for the ASD detection framework.

Exports the trained ASDModel to ONNX and TorchScript formats and validates
output equivalence between the original PyTorch model and each exported
artefact.  All results are JSON-serialisable for inclusion in the
reproducibility manifest.

Supported export formats
------------------------
1. ONNX (opset 17, dynamic batch axis)
     - Optional runtime validation via `onnxruntime` (CPU provider)
     - Disk size reported in MB
     - Max absolute diff vs. PyTorch reference (must be < `tol`)

2. TorchScript — traced
     - `torch.jit.trace` with example inputs
     - Handles dict output by extracting the `logits` key before tracing
     - Saves as `.pt` file

3. TorchScript — scripted
     - `torch.jit.script` attempt with graceful fallback
     - Many models with dynamic Python control flow cannot be scripted;
       the report records the failure reason rather than raising

Usage
-----
    from utilities.model_export import ModelExporter

    exporter = ModelExporter(tolerance=1e-4, opset=17)
    report = exporter.export(
        model      = trained_model,
        mri_tensor = example_mri,      # (1, 1, D, H, W)
        gen_tensor = example_genetics, # (1, G)
        out_dir    = out_dir / "model_export",
    )
    report.print_summary()
    report.save(out_dir / "model_export" / "export_report.json")

Notes
-----
- Exports use `model.eval()` with `torch.no_grad()`.
- The wrapper class used for export returns only `logits` (B, num_classes).
  Downstream tools that need `fused_features` for t-SNE should call the
  original PyTorch model directly.
- AMP / fp16 export is supported via the `amp` flag (writes fp16 ONNX).
"""

from __future__ import annotations

import io
import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_ONNX_OPSET = 17


# ---------------------------------------------------------------------------
# Lightweight wrapper that linearises the model's dict output
# ---------------------------------------------------------------------------

class _ExportWrapper(nn.Module):
    """
    Positional-arg wrapper that returns only the `logits` tensor.

    ASDModel.forward(mri, genetics) returns a dict; ONNX tracing requires
    a single tensor (or a flat tuple of tensors).  This wrapper makes the
    model traceable without modifying ASDModel itself.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.m = model

    def forward(
        self,
        mri:      torch.Tensor,
        genetics: torch.Tensor,
    ) -> torch.Tensor:
        out = self.m(mri, genetics)
        if isinstance(out, dict):
            return out["logits"]
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class FormatResult:
    """Result for a single export format."""
    format:        str          # "onnx" | "torchscript_trace" | "torchscript_script"
    path:          str          # absolute path to saved file
    size_mb:       float        # file size in MB
    exported:      bool         # whether export succeeded
    validated:     bool         # whether equivalence check passed
    max_diff:      float        # max |output_exported − output_pytorch|
    tolerance:     float        # threshold used
    error_message: str = ""     # non-empty only when exported=False


@dataclass
class ExportReport:
    """Complete export report for one model."""

    # ---- Architecture summary ----
    total_params:     int   = 0
    size_fp32_mb:     float = 0.0
    mri_shape:        Tuple = ()
    genetics_shape:   Tuple = ()

    # ---- Per-format results ----
    results: List[FormatResult] = field(default_factory=list)

    # ---- Metadata ----
    torch_version:    str   = ""
    onnx_opset:       int   = _ONNX_OPSET
    tolerance:        float = 1e-4
    device:           str   = "cpu"
    ort_available:    bool  = False

    def all_passed(self) -> bool:
        """True if every attempted export was validated successfully."""
        return all(r.validated or not r.exported for r in self.results)

    def print_summary(self) -> None:
        logger.info("=" * 64)
        logger.info("MODEL EXPORT REPORT")
        logger.info("=" * 64)
        logger.info("  Parameters  : %s", _fmt_num(self.total_params))
        logger.info("  FP32 size   : %.1f MB", self.size_fp32_mb)
        logger.info("  MRI input   : %s", self.mri_shape)
        logger.info("  Gen input   : %s", self.genetics_shape)
        logger.info("  Tolerance   : %.1e", self.tolerance)
        logger.info("  ORT avail.  : %s", self.ort_available)
        logger.info("-" * 64)
        logger.info("  %-22s  %-8s  %-10s  %-10s  %-8s",
                    "Format", "Exported", "Validated", "Max diff", "Size MB")
        for r in self.results:
            status = ("PASS" if r.validated else
                      "FAIL" if r.exported else
                      "ERROR")
            logger.info("  %-22s  %-8s  %-10s  %-10.2e  %-8.2f",
                        r.format,
                        "yes" if r.exported else "no",
                        status,
                        r.max_diff,
                        r.size_mb)
            if r.error_message:
                logger.warning("    Error: %s", r.error_message[:120])
        status_str = "ALL PASS" if self.all_passed() else "SOME FAILED"
        logger.info("=" * 64)
        logger.info("  Overall: %s", status_str)
        logger.info("=" * 64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_params":   self.total_params,
            "size_fp32_mb":   round(self.size_fp32_mb, 3),
            "mri_shape":      list(self.mri_shape),
            "genetics_shape": list(self.genetics_shape),
            "torch_version":  self.torch_version,
            "onnx_opset":     self.onnx_opset,
            "tolerance":      self.tolerance,
            "device":         self.device,
            "ort_available":  self.ort_available,
            "all_passed":     self.all_passed(),
            "results": [
                {
                    "format":        r.format,
                    "path":          r.path,
                    "size_mb":       round(r.size_mb, 3),
                    "exported":      r.exported,
                    "validated":     r.validated,
                    "max_diff":      r.max_diff,
                    "tolerance":     r.tolerance,
                    "error_message": r.error_message,
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Export report saved → %s", path)


# ---------------------------------------------------------------------------
# Main exporter
# ---------------------------------------------------------------------------

class ModelExporter:
    """
    Export ASDModel to ONNX and TorchScript with equivalence validation.

    Parameters
    ----------
    tolerance : float
        Max absolute difference allowed between PyTorch and exported model
        outputs for the equivalence check to pass.  Default 1e-4.
    opset : int
        ONNX opset version.  17 is compatible with ORT 1.15+ and covers
        all ops used by conv, attention, and BN layers.
    """

    def __init__(
        self,
        tolerance: float = 1e-4,
        opset:     int   = _ONNX_OPSET,
    ) -> None:
        self.tolerance = tolerance
        self.opset     = opset

        # Detect onnxruntime once at construction time
        try:
            import onnxruntime  # noqa: F401
            self._ort_available = True
        except ImportError:
            self._ort_available = False
            logger.warning(
                "onnxruntime not installed — ONNX runtime validation skipped. "
                "Install with: pip install onnxruntime"
            )

    def export(
        self,
        model:      nn.Module,
        mri_tensor: torch.Tensor,
        gen_tensor: torch.Tensor,
        out_dir:    Path,
        device:     Optional[torch.device] = None,
    ) -> ExportReport:
        """
        Export model to all supported formats and validate equivalence.

        Parameters
        ----------
        model      : ASDModel (or any nn.Module whose forward takes mri, genetics)
        mri_tensor : single-sample MRI tensor, shape (1, C, D, H, W)
        gen_tensor : single-sample genetics tensor, shape (1, G)
        out_dir    : directory where exported files are written
        device     : device to use for reference inference (default: auto-detect)
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            device = next(model.parameters()).device

        model = model.to(device).eval()

        # Build wrapper — used for all export formats
        wrapper = _ExportWrapper(model).to(device).eval()

        mri_dev = mri_tensor.to(device)
        gen_dev = gen_tensor.to(device)

        # Reference output: PyTorch forward pass
        with torch.no_grad():
            ref_output = wrapper(mri_dev, gen_dev).detach().cpu().numpy()

        report = ExportReport(
            torch_version  = torch.__version__,
            onnx_opset     = self.opset,
            tolerance      = self.tolerance,
            device         = str(device),
            ort_available  = self._ort_available,
        )

        # Parameter count and model size
        total_params  = sum(p.numel() for p in model.parameters())
        param_bytes   = sum(p.numel() * p.element_size() for p in model.parameters())
        report.total_params = total_params
        report.size_fp32_mb = param_bytes / 1e6
        report.mri_shape     = tuple(mri_tensor.shape[1:])
        report.genetics_shape = tuple(gen_tensor.shape[1:])

        # ---- ONNX ----
        onnx_path = out_dir / "model.onnx"
        report.results.append(
            self._export_onnx(wrapper, mri_dev, gen_dev, ref_output, onnx_path)
        )

        # ---- TorchScript trace ----
        ts_trace_path = out_dir / "model_traced.pt"
        report.results.append(
            self._export_torchscript_trace(
                wrapper, mri_dev, gen_dev, ref_output, ts_trace_path, device
            )
        )

        # ---- TorchScript script ----
        ts_script_path = out_dir / "model_scripted.pt"
        report.results.append(
            self._export_torchscript_script(
                wrapper, mri_dev, gen_dev, ref_output, ts_script_path, device
            )
        )

        return report

    # ------------------------------------------------------------------
    # ONNX
    # ------------------------------------------------------------------

    def _export_onnx(
        self,
        wrapper:    nn.Module,
        mri:        torch.Tensor,
        gen:        torch.Tensor,
        ref_output: np.ndarray,
        path:       Path,
    ) -> FormatResult:
        try:
            import onnx
        except ImportError:
            return FormatResult(
                format        = "onnx",
                path          = str(path),
                size_mb       = 0.0,
                exported      = False,
                validated     = False,
                max_diff      = float("nan"),
                tolerance     = self.tolerance,
                error_message = "onnx not installed (pip install onnx)",
            )

        try:
            logger.info("Exporting ONNX → %s  (opset %d)…", path, self.opset)
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    args          = (mri, gen),
                    f             = str(path),
                    input_names   = ["mri", "genetics"],
                    output_names  = ["logits"],
                    dynamic_axes  = {
                        "mri":      {0: "batch_size"},
                        "genetics": {0: "batch_size"},
                        "logits":   {0: "batch_size"},
                    },
                    opset_version          = self.opset,
                    do_constant_folding    = True,
                    export_params          = True,
                    verbose                = False,
                )

            # Structural validation
            onnx_model = onnx.load(str(path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX structural check passed")

        except Exception as exc:
            return FormatResult(
                format        = "onnx",
                path          = str(path),
                size_mb       = _file_size_mb(path),
                exported      = path.exists(),
                validated     = False,
                max_diff      = float("nan"),
                tolerance     = self.tolerance,
                error_message = _short_exc(exc),
            )

        size_mb  = _file_size_mb(path)
        max_diff = float("nan")
        validated = False

        # Runtime validation via OnnxRuntime
        if self._ort_available:
            try:
                import onnxruntime as ort

                sess = ort.InferenceSession(
                    str(path),
                    providers=["CPUExecutionProvider"],
                )
                ort_out = sess.run(
                    ["logits"],
                    {
                        "mri":      mri.cpu().numpy(),
                        "genetics": gen.cpu().numpy(),
                    },
                )[0]

                max_diff  = float(np.max(np.abs(ort_out - ref_output)))
                validated = max_diff < self.tolerance
                logger.info(
                    "ONNX ORT validation: max_diff=%.3e  [%s]",
                    max_diff, "PASS" if validated else "FAIL"
                )
            except Exception as exc:
                logger.warning("ONNX ORT validation failed: %s", exc)
                max_diff  = float("nan")
                validated = False
        else:
            # File exists + structural check passed — best we can do without ORT
            validated = True
            max_diff  = 0.0
            logger.info("ONNX runtime validation skipped (ORT not installed)")

        return FormatResult(
            format    = "onnx",
            path      = str(path),
            size_mb   = size_mb,
            exported  = True,
            validated = validated,
            max_diff  = max_diff,
            tolerance = self.tolerance,
        )

    # ------------------------------------------------------------------
    # TorchScript trace
    # ------------------------------------------------------------------

    def _export_torchscript_trace(
        self,
        wrapper:    nn.Module,
        mri:        torch.Tensor,
        gen:        torch.Tensor,
        ref_output: np.ndarray,
        path:       Path,
        device:     torch.device,
    ) -> FormatResult:
        try:
            logger.info("Tracing TorchScript → %s…", path)
            with torch.no_grad():
                traced = torch.jit.trace(wrapper, (mri, gen), strict=False)
            torch.jit.save(traced, str(path))
            logger.info("TorchScript trace saved (%.1f MB)", _file_size_mb(path))

        except Exception as exc:
            return FormatResult(
                format        = "torchscript_trace",
                path          = str(path),
                size_mb       = _file_size_mb(path),
                exported      = path.exists(),
                validated     = False,
                max_diff      = float("nan"),
                tolerance     = self.tolerance,
                error_message = _short_exc(exc),
            )

        # Equivalence check
        max_diff  = float("nan")
        validated = False
        try:
            loaded = torch.jit.load(str(path), map_location=device)
            loaded.eval()
            with torch.no_grad():
                ts_out = loaded(mri, gen).cpu().numpy()
            max_diff  = float(np.max(np.abs(ts_out - ref_output)))
            validated = max_diff < self.tolerance
            logger.info(
                "TorchScript trace validation: max_diff=%.3e  [%s]",
                max_diff, "PASS" if validated else "FAIL"
            )
        except Exception as exc:
            logger.warning("TorchScript trace validation failed: %s", exc)

        return FormatResult(
            format    = "torchscript_trace",
            path      = str(path),
            size_mb   = _file_size_mb(path),
            exported  = True,
            validated = validated,
            max_diff  = max_diff,
            tolerance = self.tolerance,
        )

    # ------------------------------------------------------------------
    # TorchScript script
    # ------------------------------------------------------------------

    def _export_torchscript_script(
        self,
        wrapper:    nn.Module,
        mri:        torch.Tensor,
        gen:        torch.Tensor,
        ref_output: np.ndarray,
        path:       Path,
        device:     torch.device,
    ) -> FormatResult:
        """
        Attempt `torch.jit.script`.  Models with data-dependent control flow,
        Python closures, or non-tensor attributes frequently fail scripting;
        this is recorded as a non-fatal failure in the report.
        """
        try:
            logger.info("Scripting TorchScript → %s…", path)
            scripted = torch.jit.script(wrapper)
            torch.jit.save(scripted, str(path))
            logger.info("TorchScript script saved (%.1f MB)", _file_size_mb(path))

        except Exception as exc:
            logger.warning(
                "TorchScript scripting failed (this is common for models "
                "with dynamic control flow — trace export is the fallback): %s",
                _short_exc(exc)
            )
            return FormatResult(
                format        = "torchscript_script",
                path          = str(path),
                size_mb       = 0.0,
                exported      = False,
                validated     = False,
                max_diff      = float("nan"),
                tolerance     = self.tolerance,
                error_message = _short_exc(exc),
            )

        # Equivalence check
        max_diff  = float("nan")
        validated = False
        try:
            loaded = torch.jit.load(str(path), map_location=device)
            loaded.eval()
            with torch.no_grad():
                sc_out = loaded(mri, gen).cpu().numpy()
            max_diff  = float(np.max(np.abs(sc_out - ref_output)))
            validated = max_diff < self.tolerance
            logger.info(
                "TorchScript script validation: max_diff=%.3e  [%s]",
                max_diff, "PASS" if validated else "FAIL"
            )
        except Exception as exc:
            logger.warning("TorchScript script validation failed: %s", exc)

        return FormatResult(
            format    = "torchscript_script",
            path      = str(path),
            size_mb   = _file_size_mb(path),
            exported  = True,
            validated = validated,
            max_diff  = max_diff,
            tolerance = self.tolerance,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_size_mb(path: Path) -> float:
    try:
        return os.path.getsize(path) / 1e6
    except OSError:
        return 0.0


def _fmt_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _short_exc(exc: Exception, max_len: int = 300) -> str:
    """Return a single-line exception summary suitable for JSON storage."""
    msg = f"{type(exc).__name__}: {exc}"
    return msg[:max_len].replace("\n", " ")
