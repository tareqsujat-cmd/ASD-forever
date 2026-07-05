"""
Computational profiler for the ASD detection framework.

Measures every dimension of model performance required by an IEEE paper's
"Computational Complexity" section:

  - FLOPs and MACs (via `thop`, with parameter-based fallback)
  - Parameter count and model size in FP32 / FP16 / INT8 (bytes)
  - Inference latency: mean ± std, P50, P95, P99 (wall-clock and GPU-only)
  - Throughput: samples/second at multiple batch sizes
  - Peak GPU memory per batch size (torch.cuda.max_memory_allocated)
  - Per-layer timing: top-k bottleneck layers (torch.profiler)

All outputs are JSON-serialisable and logged at INFO level.

Usage
-----
    from utilities.profiler import ComputationalProfiler

    profiler = ComputationalProfiler(device=device)
    result = profiler.profile(
        model          = model,
        example_inputs = {"image": mri_tensor, "genetics": gen_tensor},
        batch_sizes    = [1, 4, 8],
        n_warmup       = 10,
        n_trials       = 100,
        profile_layers = True,
    )
    result.print_summary()
    result.save(out_dir / "computational_profile.json")

Notes
-----
- `thop` (pip install thop) is used for FLOPs/MACs when available.
  If not installed, a parameter-based estimate is computed and flagged.
- `torch.profiler` layer-level timing adds ~2× overhead; it is run
  separately from the latency benchmark to avoid interfering with numbers.
- All timings use CUDA events when on GPU (device-synchronised, not wall-clock).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Latency distribution at a specific batch size."""
    batch_size:   int
    n_trials:     int
    mean_ms:      float
    std_ms:       float
    median_ms:    float
    p95_ms:       float
    p99_ms:       float
    min_ms:       float
    max_ms:       float
    throughput_samples_per_sec: float
    peak_gpu_memory_mb:         float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class ProfilingReport:
    """Complete computational profile of the ASD model."""

    # ---- Architecture ----
    total_params:     int   = 0
    trainable_params: int   = 0
    frozen_params:    int   = 0

    # ---- Model size ----
    size_fp32_mb: float = 0.0
    size_fp16_mb: float = 0.0
    size_int8_mb: float = 0.0   # estimated (parameter bytes only, not activations)

    # ---- FLOPs / MACs ----
    flops:        float = 0.0   # floating point operations (forward pass)
    macs:         float = 0.0   # multiply-accumulate operations
    gflops:       float = 0.0
    gmacs:        float = 0.0
    flops_source: str   = "unknown"   # "thop" | "estimated"

    # ---- Input shapes ----
    mri_shape:     Tuple = ()
    genetics_shape: Tuple = ()

    # ---- Latency per batch size ----
    latency: Dict[int, LatencyStats] = field(default_factory=dict)

    # ---- Per-layer profiling ----
    top_layers_by_time: List[Dict[str, Any]] = field(default_factory=list)

    # ---- Device info ----
    device:      str = "cpu"
    gpu_name:    str = ""
    cuda_version: str = ""
    torch_version: str = ""

    def print_summary(self) -> None:
        logger.info("=" * 64)
        logger.info("COMPUTATIONAL PROFILE")
        logger.info("=" * 64)
        logger.info("  Device       : %s  (%s)", self.device, self.gpu_name or "N/A")
        logger.info("  Parameters   : %s total  (%s trainable  %s frozen)",
                    _fmt_num(self.total_params),
                    _fmt_num(self.trainable_params),
                    _fmt_num(self.frozen_params))
        logger.info("  Model size   : FP32=%.1f MB  FP16=%.1f MB  INT8≈%.1f MB",
                    self.size_fp32_mb, self.size_fp16_mb, self.size_int8_mb)
        logger.info("  FLOPs        : %.3f GFLOPs  [%s]", self.gflops, self.flops_source)
        logger.info("  MACs         : %.3f GMACs   [%s]", self.gmacs, self.flops_source)
        logger.info("-" * 64)
        logger.info("  Latency & Throughput:")
        logger.info("  %-6s  %-10s  %-10s  %-10s  %-14s  %-12s",
                    "BS", "mean(ms)", "P95(ms)", "P99(ms)",
                    "throughput(s/s)", "peak_mem(MB)")
        for bs, ls in sorted(self.latency.items()):
            logger.info("  %-6d  %-10.2f  %-10.2f  %-10.2f  %-14.1f  %-12.1f",
                        bs, ls.mean_ms, ls.p95_ms, ls.p99_ms,
                        ls.throughput_samples_per_sec, ls.peak_gpu_memory_mb)
        if self.top_layers_by_time:
            logger.info("-" * 64)
            logger.info("  Top-10 layers by CPU time:")
            for i, layer in enumerate(self.top_layers_by_time[:10], 1):
                logger.info("  %2d. %-40s  %.2f ms", i,
                            layer.get("name", "?")[:40],
                            layer.get("cpu_time_ms", 0.0))
        logger.info("=" * 64)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total_params":       self.total_params,
            "trainable_params":   self.trainable_params,
            "frozen_params":      self.frozen_params,
            "size_fp32_mb":       round(self.size_fp32_mb, 3),
            "size_fp16_mb":       round(self.size_fp16_mb, 3),
            "size_int8_mb":       round(self.size_int8_mb, 3),
            "flops":              self.flops,
            "macs":               self.macs,
            "gflops":             round(self.gflops, 6),
            "gmacs":              round(self.gmacs, 6),
            "flops_source":       self.flops_source,
            "mri_shape":          list(self.mri_shape),
            "genetics_shape":     list(self.genetics_shape),
            "device":             self.device,
            "gpu_name":           self.gpu_name,
            "cuda_version":       self.cuda_version,
            "torch_version":      self.torch_version,
            "latency": {
                str(bs): ls.to_dict() for bs, ls in self.latency.items()
            },
            "top_layers_by_time": self.top_layers_by_time,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Computational profile saved → %s", path)

    def to_latex_table(self) -> str:
        """
        LaTeX table for the paper's computational complexity section.
        Lists latency and throughput at each batch size.
        """
        lines = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \caption{Inference Latency and Throughput}",
            r"  \label{tab:compute}",
            r"  \begin{tabular}{rrrrr}",
            r"    \toprule",
            r"    BS & Latency (ms) & P95 (ms) & Throughput (s/s) & Peak Mem (MB) \\",
            r"    \midrule",
        ]
        for bs, ls in sorted(self.latency.items()):
            lines.append(
                f"    {bs} & {ls.mean_ms:.1f}$\\pm${ls.std_ms:.1f} & "
                f"{ls.p95_ms:.1f} & {ls.throughput_samples_per_sec:.0f} & "
                f"{ls.peak_gpu_memory_mb:.0f} \\\\"
            )
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

class ComputationalProfiler:
    """
    Profile a PyTorch model for FLOPs, latency, throughput, and memory.

    Parameters
    ----------
    device : torch.device
        Device to run profiling on.
    amp : bool
        Whether to measure under AMP (float16 forward pass).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        amp:    bool = False,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.amp = amp and (self.device.type == "cuda")

    def profile(
        self,
        model:          nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        batch_sizes:    List[int]  = None,
        n_warmup:       int        = 10,
        n_trials:       int        = 100,
        profile_layers: bool       = True,
        out_dir:        Optional[Path] = None,
    ) -> ProfilingReport:
        """
        Run the full profiling suite.

        Parameters
        ----------
        model          : the ASDModel (or any nn.Module)
        example_inputs : dict with "image" (1, C, D, H, W) and "genetics" (1, G)
                         tensors at batch_size=1; other batch sizes are constructed
                         by repeating these along dim 0.
        batch_sizes    : list of batch sizes to benchmark; default [1, 4, 8]
        n_warmup       : warm-up iterations (not timed) to heat up GPU cache
        n_trials       : timed iterations per batch size
        profile_layers : whether to run torch.profiler per-layer analysis
        out_dir        : if given, save JSON report to out_dir/computational_profile.json
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]

        model = model.to(self.device).eval()
        report = ProfilingReport(device=str(self.device))

        # ---- Device info ----
        self._fill_device_info(report)

        # ---- Architecture: parameter count + model size ----
        self._measure_parameters(model, report)

        # ---- Input shapes ----
        img = example_inputs.get("image",    example_inputs.get("mri"))
        gen = example_inputs.get("genetics", example_inputs.get("gen"))
        if img is not None:
            report.mri_shape      = tuple(img.shape[1:])  # drop batch dim
        if gen is not None:
            report.genetics_shape = tuple(gen.shape[1:])

        # ---- FLOPs / MACs ----
        self._measure_flops(model, example_inputs, report)

        # ---- Latency + throughput + memory ----
        for bs in batch_sizes:
            logger.info("Benchmarking batch_size=%d (%d warm-up + %d trials)…",
                        bs, n_warmup, n_trials)
            stats = self._benchmark_latency(
                model, example_inputs, bs, n_warmup, n_trials
            )
            report.latency[bs] = stats
            logger.info("  bs=%d  latency=%.2f±%.2fms  "
                        "throughput=%.0f s/s  peak_mem=%.0fMB",
                        bs, stats.mean_ms, stats.std_ms,
                        stats.throughput_samples_per_sec,
                        stats.peak_gpu_memory_mb)

        # ---- Per-layer profiling ----
        if profile_layers:
            self._profile_layers(model, example_inputs, report)

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            report.save(out_dir / "computational_profile.json")

        return report

    # ------------------------------------------------------------------
    # Device info
    # ------------------------------------------------------------------

    def _fill_device_info(self, report: ProfilingReport) -> None:
        report.torch_version = torch.__version__
        if self.device.type == "cuda" and torch.cuda.is_available():
            report.gpu_name    = torch.cuda.get_device_name(0)
            report.cuda_version = torch.version.cuda or ""
        else:
            report.gpu_name    = ""
            report.cuda_version = ""

    # ------------------------------------------------------------------
    # Parameter count and model size
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_parameters(model: nn.Module, report: ProfilingReport) -> None:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen    = total - trainable

        report.total_params     = total
        report.trainable_params = trainable
        report.frozen_params    = frozen

        # Sizes: each parameter is stored as the dtype it's declared as
        # We compute assuming FP32 weights (standard for inference)
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        report.size_fp32_mb = param_bytes / 1e6
        report.size_fp16_mb = param_bytes / 2e6   # FP16 = half the bytes
        report.size_int8_mb = param_bytes / 4e6   # INT8 = quarter the bytes

        logger.info("Parameters: %s total  (%s trainable, %s frozen)  "
                    "FP32 size: %.1f MB",
                    _fmt_num(total), _fmt_num(trainable), _fmt_num(frozen),
                    report.size_fp32_mb)

    # ------------------------------------------------------------------
    # FLOPs / MACs
    # ------------------------------------------------------------------

    def _measure_flops(
        self,
        model:          nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        report:         ProfilingReport,
    ) -> None:
        """
        Measure FLOPs and MACs using `thop` (primary) or a parameter-based
        estimate (fallback when thop is not installed).

        thop counts ops for Conv, Linear, BN, ReLU, etc.  The dual-stream
        model (MRI + genetics) is profiled by tracing the combined forward.
        """
        img = self._to_device(example_inputs.get("image",
                              example_inputs.get("mri")))
        gen = self._to_device(example_inputs.get("genetics",
                              example_inputs.get("gen")))

        # ---- Attempt thop ----
        try:
            from thop import profile as thop_profile, clever_format

            # Wrap model so thop receives positional args
            class _Wrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, image, genetics):
                    return self.m(image, genetics)

            wrapper = _Wrapper(model)
            with torch.no_grad():
                macs, params = thop_profile(
                    wrapper, inputs=(img, gen), verbose=False
                )
            flops = 2 * macs   # 1 MAC = 2 FLOPs (mul + add)

            report.macs        = float(macs)
            report.flops       = float(flops)
            report.gmacs       = float(macs)  / 1e9
            report.gflops      = float(flops) / 1e9
            report.flops_source = "thop"

            macs_str, params_str = clever_format([macs, params], "%.3f")
            logger.info("FLOPs (thop): %.3f GFLOPs  MACs: %.3f GMACs  "
                        "Params: %s",
                        report.gflops, report.gmacs, macs_str)
            return

        except ImportError:
            logger.warning(
                "thop not installed (pip install thop). "
                "Falling back to parameter-based FLOPs estimate."
            )
        except Exception as exc:
            logger.warning("thop profiling failed: %s — using estimate.", exc)

        # ---- Fallback: fvcore ----
        try:
            from fvcore.nn import FlopCountAnalysis

            class _FvcWrapper(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, image, genetics):
                    return self.m(image, genetics)

            with torch.no_grad():
                fca = FlopCountAnalysis(_FvcWrapper(model), (img, gen))
                fca.unsupported_ops_warnings(False)
                fca.uncalled_modules_warnings(False)
                flops = float(fca.total())

            report.flops       = flops
            report.macs        = flops / 2.0
            report.gflops      = flops / 1e9
            report.gmacs       = flops / 2e9
            report.flops_source = "fvcore"
            logger.info("FLOPs (fvcore): %.3f GFLOPs", report.gflops)
            return

        except ImportError:
            pass
        except Exception as exc:
            logger.warning("fvcore profiling failed: %s — using estimate.", exc)

        # ---- Fallback: parameter-based heuristic ----
        # For a typical CNN+Transformer, ~2 FLOPs per parameter per token.
        # This is a rough estimate — the actual number depends on input resolution.
        n_params = report.total_params
        # Estimate: 2 × params × typical_spatial_tokens (rough for 16^3 MRI)
        estimated_macs = n_params * 2.0
        report.macs        = estimated_macs
        report.flops       = estimated_macs * 2
        report.gmacs       = estimated_macs / 1e9
        report.gflops      = estimated_macs * 2 / 1e9
        report.flops_source = "estimated (install thop for accurate counts)"
        logger.warning(
            "FLOPs estimated from parameter count: %.3f GFLOPs  "
            "(INACCURATE — install thop or fvcore for exact values)",
            report.gflops,
        )

    # ------------------------------------------------------------------
    # Latency benchmark
    # ------------------------------------------------------------------

    def _benchmark_latency(
        self,
        model:          nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        batch_size:     int,
        n_warmup:       int,
        n_trials:       int,
    ) -> LatencyStats:
        """
        Benchmark inference latency using CUDA events (GPU) or time.perf_counter (CPU).

        CUDA events are device-synchronised — they measure only GPU compute time,
        not Python/CPU overhead.  This gives the most reproducible measurements.
        """
        img = self._batch(example_inputs.get("image", example_inputs.get("mri")),
                          batch_size)
        gen = self._batch(example_inputs.get("genetics", example_inputs.get("gen")),
                          batch_size)

        use_cuda = (self.device.type == "cuda")

        if use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        # Warm-up
        with torch.no_grad():
            ctx = torch.amp.autocast("cuda") if self.amp else _null_ctx()
            with ctx:
                for _ in range(n_warmup):
                    _ = model(img, gen)
                if use_cuda:
                    torch.cuda.synchronize()

        # Timed trials
        times_ms: List[float] = []

        if use_cuda:
            # CUDA event timing: excludes Python overhead, measures GPU time only
            for _ in range(n_trials):
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.no_grad():
                    ctx = torch.amp.autocast("cuda") if self.amp else _null_ctx()
                    with ctx:
                        _ = model(img, gen)
                end.record()
                torch.cuda.synchronize()
                times_ms.append(start.elapsed_time(end))
        else:
            for _ in range(n_trials):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(img, gen)
                times_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(times_ms, dtype=float)
        mean_ms   = float(np.mean(arr))
        std_ms    = float(np.std(arr, ddof=1))
        median_ms = float(np.median(arr))
        p95_ms    = float(np.percentile(arr, 95))
        p99_ms    = float(np.percentile(arr, 99))

        # Throughput in samples/second
        throughput = batch_size / (mean_ms / 1000.0)

        # Peak GPU memory (in MB) since start of session
        peak_mb = 0.0
        if use_cuda:
            peak_mb = torch.cuda.max_memory_allocated(self.device) / 1e6

        return LatencyStats(
            batch_size   = batch_size,
            n_trials     = n_trials,
            mean_ms      = mean_ms,
            std_ms       = std_ms,
            median_ms    = median_ms,
            p95_ms       = p95_ms,
            p99_ms       = p99_ms,
            min_ms       = float(arr.min()),
            max_ms       = float(arr.max()),
            throughput_samples_per_sec = throughput,
            peak_gpu_memory_mb         = peak_mb,
        )

    # ------------------------------------------------------------------
    # Per-layer profiling
    # ------------------------------------------------------------------

    def _profile_layers(
        self,
        model:          nn.Module,
        example_inputs: Dict[str, torch.Tensor],
        report:         ProfilingReport,
        n_steps:        int = 3,
        top_k:          int = 20,
    ) -> None:
        """
        Use torch.profiler to identify the top-k time-consuming layers.

        Runs n_steps forward passes under the profiler, then extracts
        per-operator CPU and CUDA times averaged across steps.
        """
        try:
            from torch.profiler import (
                profile as torch_profile,
                ProfilerActivity,
                record_function,
            )

            img = self._to_device(example_inputs.get("image",
                                  example_inputs.get("mri")))
            gen = self._to_device(example_inputs.get("genetics",
                                  example_inputs.get("gen")))

            activities = [ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)

            with torch_profile(
                activities       = activities,
                record_shapes    = False,
                with_flops       = True,
                with_stack       = False,
                profile_memory   = True,
            ) as prof:
                for _ in range(n_steps):
                    with record_function("forward"):
                        with torch.no_grad():
                            _ = model(img, gen)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()

            # Extract top-k ops by CPU time
            key_averages = prof.key_averages()
            sorted_ops = sorted(
                key_averages,
                key=lambda e: e.cpu_time_total,
                reverse=True,
            )[:top_k]

            report.top_layers_by_time = [
                {
                    "name":           e.key,
                    "cpu_time_ms":    round(
                        getattr(e, "cpu_time_total", 0) / (1000.0 * n_steps), 4
                    ),
                    "cuda_time_ms":   round(
                        getattr(e, "cuda_time_total", 0) / (1000.0 * n_steps), 4
                    ) if self.device.type == "cuda" else 0.0,
                    "n_calls":        getattr(e, "count", n_steps) // n_steps,
                    "cpu_memory_mb":  round(
                        getattr(e, "cpu_memory_usage", 0) / 1e6, 3
                    ),
                    "cuda_memory_mb": round(
                        getattr(e, "cuda_memory_usage", 0) / 1e6, 3
                    ) if self.device.type == "cuda" else 0.0,
                    "flops":          getattr(e, "flops", 0) // n_steps,
                }
                for e in sorted_ops
            ]
            logger.info("Per-layer profiling complete (%d ops captured)", len(sorted_ops))

        except Exception as exc:
            logger.warning("Per-layer profiling failed: %s", exc)
            report.top_layers_by_time = []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _to_device(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.to(self.device)

    def _batch(
        self, tensor: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        """Replicate a single-sample tensor along dim 0 to produce a batch."""
        if tensor is None:
            return None
        t = tensor.to(self.device)
        if t.shape[0] == 1:
            t = t.expand(batch_size, *t.shape[1:]).contiguous()
        elif t.shape[0] != batch_size:
            # Tile as many times as needed, then slice
            reps = (batch_size + t.shape[0] - 1) // t.shape[0]
            t = t.repeat(reps, *([1] * (t.ndim - 1)))[:batch_size].contiguous()
        return t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_num(n: int) -> str:
    """Format large integers with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class _null_ctx:
    """No-op context manager used when AMP is disabled."""
    def __enter__(self): return self
    def __exit__(self, *_): pass
