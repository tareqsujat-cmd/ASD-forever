"""
Robustness evaluation for the ASD detection framework.

Tests model resilience across three perturbation families required by
IEEE medical AI reviewers:

1. Noise injection (test-time)
   - Gaussian additive noise on MRI at σ ∈ {0.05, 0.15, 0.30} × signal_std
   - Rician noise (the correct model for MRI magnitude images) at two levels
   - Gaussian additive noise on genetics features at two levels
   - Salt-and-pepper masking on genetics (simulates missing SNP calls)

2. Missing-modality evaluation
   - MRI absent   → zero imputation / mean imputation / Gaussian noise fill
   - Genetics absent → zero imputation / mean imputation

3. Distribution-shift tests
   - MRI contrast shift (scanner gain: ×0.5, ×1.5)
   - MRI sinusoidal bias field (low-frequency spatial drift, amplitude ±0.2)
   - Site holdout: evaluate on subjects from a single held-out site
     (skipped when fewer than 2 sites have ≥ `min_site_n` subjects each)

All perturbations are applied at inference time only — no retraining.

Usage
-----
    from evaluation.robustness import RobustnessEvaluator

    evaluator = RobustnessEvaluator(threshold=0.5, batch_size=8)
    report = evaluator.evaluate(
        model      = model,
        mri        = mri_tensor,    # (N, 1, D, H, W)
        genetics   = gen_tensor,    # (N, G)
        y_true     = labels,        # (N,) int
        site_ids   = sites,         # (N,) str
        device     = device,
        out_dir    = out_dir / "robustness",
    )
    report.print_summary()
    report.save(out_dir / "robustness" / "robustness_report.json")

Notes
-----
- The model is called as `model(mri, genetics)` — positional tensors matching
  ASDModel.forward(mri, genetics).
- All metric computations use evaluation.metrics.compute_all_metrics for
  consistency with the main evaluation pipeline.
- Noise levels are expressed as fractions of the signal's population standard
  deviation (measured over the clean test set), not absolute values.
"""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perturbation base class and concrete implementations
# ---------------------------------------------------------------------------

class Perturbation(ABC):
    """Abstract base: perturbs (mri, genetics) tensor pair."""

    name: str = "base"

    @abstractmethod
    def apply(
        self,
        mri:      torch.Tensor,   # (N, 1, D, H, W)
        genetics: torch.Tensor,   # (N, G)
        mri_std:  float,          # population std of clean MRI (for relative noise)
        gen_std:  float,          # population std of clean genetics
        rng:      np.random.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class _GaussianMRI(Perturbation):
    """Additive Gaussian noise on MRI at sigma = level × mri_std."""

    def __init__(self, level: float) -> None:
        self.level = level
        self.name  = f"mri_gaussian_σ={level:.2f}×std"

    def apply(self, mri, genetics, mri_std, gen_std, rng):
        sigma = self.level * mri_std
        noise = torch.from_numpy(
            rng.normal(0, sigma, mri.shape).astype(np.float32)
        ).to(mri.device)
        return mri + noise, genetics


class _RicianMRI(Perturbation):
    """
    Rician noise — the correct noise model for MRI magnitude images.

    For a clean signal X, Rician noise gives:
        Y = sqrt((X + n1)^2 + n2^2),  n1, n2 ~ N(0, sigma^2)
    At high SNR this approximates Gaussian; at low SNR it introduces
    a systematic positive bias (the Rician floor).
    """

    def __init__(self, level: float) -> None:
        self.level = level
        self.name  = f"mri_rician_σ={level:.2f}×std"

    def apply(self, mri, genetics, mri_std, gen_std, rng):
        sigma = self.level * mri_std
        n1    = torch.from_numpy(
            rng.normal(0, sigma, mri.shape).astype(np.float32)
        ).to(mri.device)
        n2    = torch.from_numpy(
            rng.normal(0, sigma, mri.shape).astype(np.float32)
        ).to(mri.device)
        noisy = torch.sqrt((mri + n1) ** 2 + n2 ** 2)
        return noisy, genetics


class _GaussianGenetics(Perturbation):
    """Additive Gaussian noise on genetics features."""

    def __init__(self, level: float) -> None:
        self.level = level
        self.name  = f"gen_gaussian_σ={level:.2f}×std"

    def apply(self, mri, genetics, mri_std, gen_std, rng):
        sigma = self.level * gen_std
        noise = torch.from_numpy(
            rng.normal(0, sigma, genetics.shape).astype(np.float32)
        ).to(genetics.device)
        return mri, genetics + noise


class _SaltPepperGenetics(Perturbation):
    """
    Randomly zero-masks a fraction of genetics features per sample.

    Simulates missing SNP genotype calls (a common real-world scenario in
    GWAS data where quality filters blank out individual loci).
    """

    def __init__(self, mask_rate: float) -> None:
        self.mask_rate = mask_rate
        self.name      = f"gen_snp_mask_{int(mask_rate * 100)}pct"

    def apply(self, mri, genetics, mri_std, gen_std, rng):
        mask = torch.from_numpy(
            (rng.random(genetics.shape) > self.mask_rate).astype(np.float32)
        ).to(genetics.device)
        return mri, genetics * mask


class _MissingMRI(Perturbation):
    """Replace entire MRI with zeros or mean-imputation."""

    def __init__(self, mode: str = "zeros") -> None:
        assert mode in ("zeros", "mean", "gaussian")
        self.mode = mode
        self.name = f"missing_mri_{mode}"

    def apply(self, mri, genetics, mri_std, gen_std, rng):
        if self.mode == "zeros":
            return torch.zeros_like(mri), genetics
        if self.mode == "mean":
            # Impute with the per-channel population mean (= 0 after z-scoring)
            mean_val = float(mri.mean())
            return torch.full_like(mri, mean_val), genetics
        # Gaussian: replace with N(0, mri_std²) — pure noise, no signal
        noise = torch.from_numpy(
            rng.normal(0, mri_std, mri.shape).astype(np.float32)
        ).to(mri.device)
        return noise, genetics


class _MissingGenetics(Perturbation):
    """Replace entire genetics array with zeros or mean-imputation."""

    def __init__(self, mode: str = "zeros") -> None:
        assert mode in ("zeros", "mean")
        self.mode = mode
        self.name = f"missing_gen_{mode}"

    def apply(self, mri, genetics, mri_std, gen_std, rng):
        if self.mode == "zeros":
            return mri, torch.zeros_like(genetics)
        mean_val = float(genetics.mean())
        return mri, torch.full_like(genetics, mean_val)


class _ContrastShift(Perturbation):
    """
    Multiply MRI intensities by a constant gain factor.

    Simulates inter-scanner contrast variation (e.g., different receiver
    coil gains or flip angle calibration between sites).
    """

    def __init__(self, gain: float) -> None:
        self.gain = gain
        self.name = f"mri_contrast_gain×{gain:.1f}"

    def apply(self, mri, genetics, mri_std, gen_std, rng):
        return mri * self.gain, genetics


class _BiasField(Perturbation):
    """
    Multiplicative sinusoidal bias field along the first spatial dimension.

    Models low-frequency B1-field inhomogeneity — the dominant artefact
    in 3T structural MRI that causes intensity gradients across the volume.
    The field amplitude is ±amplitude (e.g., ±0.20 = ±20% intensity shift).
    """

    def __init__(self, amplitude: float = 0.20) -> None:
        self.amplitude = amplitude
        self.name      = f"mri_bias_field_amp={amplitude:.2f}"

    def apply(self, mri, genetics, mri_std, gen_std, rng):
        D = mri.shape[2]   # depth dimension
        # One full cycle of a sine wave across depth
        z = torch.linspace(0, 2 * math.pi, D, device=mri.device)
        field = 1.0 + self.amplitude * torch.sin(z)   # shape (D,)
        # Broadcast to (N, 1, D, H, W)
        field = field.view(1, 1, D, 1, 1)
        return mri * field, genetics


# ---------------------------------------------------------------------------
# Standard condition set
# ---------------------------------------------------------------------------

def _standard_perturbations() -> List[Perturbation]:
    return [
        # Gaussian MRI noise
        _GaussianMRI(0.05),
        _GaussianMRI(0.15),
        _GaussianMRI(0.30),
        # Rician MRI noise
        _RicianMRI(0.10),
        _RicianMRI(0.30),
        # Genetics noise
        _GaussianGenetics(0.10),
        _GaussianGenetics(0.30),
        # SNP masking
        _SaltPepperGenetics(0.10),
        _SaltPepperGenetics(0.30),
        # Missing modality
        _MissingMRI("zeros"),
        _MissingMRI("mean"),
        _MissingGenetics("zeros"),
        _MissingGenetics("mean"),
        # Distribution shift
        _ContrastShift(0.50),
        _ContrastShift(1.50),
        _BiasField(0.20),
    ]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    """Metrics for one perturbation condition."""
    condition:    str
    n_subjects:   int
    metrics:      Dict[str, float]
    delta_auc:    float = 0.0    # vs. baseline (negative = degraded)
    delta_f1:     float = 0.0
    delta_sens:   float = 0.0
    delta_spec:   float = 0.0
    is_baseline:  bool  = False

    def to_dict(self) -> dict:
        return {
            "condition":   self.condition,
            "n_subjects":  self.n_subjects,
            "metrics":     {k: round(v, 6) for k, v in self.metrics.items()},
            "delta_auc":   round(self.delta_auc, 6),
            "delta_f1":    round(self.delta_f1, 6),
            "delta_sens":  round(self.delta_sens, 6),
            "delta_spec":  round(self.delta_spec, 6),
            "is_baseline": self.is_baseline,
        }


@dataclass
class RobustnessReport:
    """Aggregated robustness evaluation results."""

    baseline:    Optional[ConditionResult] = None
    conditions:  List[ConditionResult]    = field(default_factory=list)
    threshold:   float                    = 0.5
    n_subjects:  int                      = 0
    mri_std:     float                    = 0.0
    gen_std:     float                    = 0.0

    def print_summary(self) -> None:
        logger.info("=" * 72)
        logger.info("ROBUSTNESS EVALUATION REPORT")
        logger.info("=" * 72)
        logger.info("  N=%d  threshold=%.2f  mri_std=%.4f  gen_std=%.4f",
                    self.n_subjects, self.threshold, self.mri_std, self.gen_std)
        if self.baseline:
            m = self.baseline.metrics
            logger.info("  Baseline: AUC=%.4f  Sens=%.4f  Spec=%.4f  F1=%.4f  MCC=%.4f",
                        m.get("auc", 0), m.get("sensitivity", 0),
                        m.get("specificity", 0), m.get("f1", 0),
                        m.get("mcc", 0))
        logger.info("-" * 72)
        logger.info("  %-40s  %7s  %7s  %7s  %7s",
                    "Condition", "ΔAUC", "ΔF1", "ΔSens", "ΔSpec")
        for r in self.conditions:
            flag = " ⚠" if abs(r.delta_auc) > 0.05 else ""
            logger.info("  %-40s  %+7.4f  %+7.4f  %+7.4f  %+7.4f%s",
                        r.condition[:40], r.delta_auc,
                        r.delta_f1, r.delta_sens, r.delta_spec, flag)
        logger.info("=" * 72)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "threshold":  self.threshold,
            "n_subjects": self.n_subjects,
            "mri_std":    round(self.mri_std, 6),
            "gen_std":    round(self.gen_std, 6),
            "baseline":   self.baseline.to_dict() if self.baseline else None,
            "conditions": [c.to_dict() for c in self.conditions],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Robustness report saved → %s", path)

    def to_latex_table(self) -> str:
        """
        IEEE-format LaTeX table of AUC degradation under each condition.
        """
        lines = [
            r"\begin{table}[h]",
            r"  \centering",
            r"  \caption{Robustness evaluation under test-time perturbations}",
            r"  \label{tab:robustness}",
            r"  \begin{tabular}{lrrrr}",
            r"    \toprule",
            r"    Condition & AUC & $\Delta$AUC & $\Delta$F1 & $\Delta$Sens \\",
            r"    \midrule",
        ]
        # Baseline row
        if self.baseline:
            m = self.baseline.metrics
            lines.append(
                f"    Baseline (clean) & "
                f"{m.get('auc', 0):.4f} & -- & -- & -- \\\\"
            )
        # Condition rows grouped by family
        families: Dict[str, List[ConditionResult]] = {}
        for c in self.conditions:
            fam = c.condition.split("_")[0]
            families.setdefault(fam, []).append(c)

        for fam, rows in families.items():
            lines.append(r"    \midrule")
            for r in rows:
                auc = r.metrics.get("auc", float("nan"))
                flag = r" $\dagger$" if abs(r.delta_auc) > 0.05 else ""
                lines.append(
                    f"    {r.condition[:48]} & "
                    f"{auc:.4f} & {r.delta_auc:+.4f} & "
                    f"{r.delta_f1:+.4f} & {r.delta_sens:+.4f}"
                    f"{flag} \\\\"
                )
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \begin{tablenotes}{\small",
            r"    $\dagger$ Clinically significant degradation ($|\Delta$AUC$| > 0.05$).",
            r"  }\end{tablenotes}",
            r"\end{table}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_inference(
    model:    nn.Module,
    mri:      torch.Tensor,     # (N, 1, D, H, W) already on device
    genetics: torch.Tensor,     # (N, G) already on device
    device:   torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Batch inference: returns P(ASD) for all N subjects.

    The model's dict output is expected to contain a "logits" key with shape
    (B, num_classes).  Softmax is applied to get class probabilities.
    """
    model.eval()
    n = mri.shape[0]
    probs_list: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        m_b  = mri[start:end].to(device)
        g_b  = genetics[start:end].to(device)

        out = model(m_b, g_b)

        if isinstance(out, dict):
            logits = out["logits"]
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        if logits.shape[-1] == 1:
            # Binary with single logit — sigmoid
            p_asd = torch.sigmoid(logits).squeeze(-1)
        else:
            # Multi-class — softmax, take class-1 column
            p_asd = torch.softmax(logits, dim=-1)[:, 1]

        probs_list.append(p_asd.cpu().numpy())

    return np.concatenate(probs_list, axis=0)


def _safe_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Compute key robustness metrics safely, returning NaN on failure.
    """
    from evaluation.metrics import compute_all_metrics
    try:
        m = compute_all_metrics(y_true, y_prob, threshold=threshold)
        return {
            "auc":         float(m.get("auc", float("nan"))),
            "sensitivity": float(m.get("sensitivity", float("nan"))),
            "specificity": float(m.get("specificity", float("nan"))),
            "f1":          float(m.get("f1",          float("nan"))),
            "mcc":         float(m.get("mcc",         float("nan"))),
            "ppv":         float(m.get("ppv",         float("nan"))),
            "npv":         float(m.get("npv",         float("nan"))),
        }
    except Exception as exc:
        logger.warning("Metric computation failed: %s", exc)
        nan = float("nan")
        return {"auc": nan, "sensitivity": nan, "specificity": nan,
                "f1": nan, "mcc": nan, "ppv": nan, "npv": nan}


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class RobustnessEvaluator:
    """
    Evaluate model robustness under test-time perturbations.

    Parameters
    ----------
    threshold  : decision threshold for binary predictions.
    batch_size : mini-batch size during inference (keep small on limited GPU).
    seed       : random seed for noise generation (reproducibility).
    min_site_n : minimum subjects per site to include in site-holdout tests.
    """

    def __init__(
        self,
        threshold:   float = 0.5,
        batch_size:  int   = 8,
        seed:        int   = 42,
        min_site_n:  int   = 5,
    ) -> None:
        self.threshold  = threshold
        self.batch_size = batch_size
        self.rng        = np.random.default_rng(seed)
        self.min_site_n = min_site_n

    def evaluate(
        self,
        model:        nn.Module,
        mri:          torch.Tensor,   # (N, 1, D, H, W) — CPU or GPU
        genetics:     torch.Tensor,   # (N, G)
        y_true:       np.ndarray,     # (N,) int
        site_ids:     Optional[np.ndarray] = None,  # (N,) str/int
        device:       Optional[torch.device] = None,
        out_dir:      Optional[Path]   = None,
        extra_perturbations: Optional[List[Perturbation]] = None,
    ) -> RobustnessReport:
        """
        Run the full robustness evaluation suite.

        Parameters
        ----------
        model        : trained ASDModel (eval mode set automatically)
        mri          : MRI tensor, (N, 1, D, H, W)
        genetics     : genetics tensor, (N, G)
        y_true       : ground-truth binary labels
        site_ids     : acquisition site identifiers (for site-holdout test)
        device       : inference device (defaults to model's device)
        out_dir      : if given, saves the JSON report and LaTeX table there
        extra_perturbations : user-defined additional perturbations to append

        Returns
        -------
        RobustnessReport
        """
        if device is None:
            device = next(model.parameters()).device

        model = model.to(device).eval()
        mri_dev = mri.to(device)
        gen_dev = genetics.to(device)

        y_true = np.asarray(y_true, dtype=int)
        n      = len(y_true)

        # Population statistics of clean test set (for relative noise levels)
        mri_std = float(mri_dev.std().item()) or 1.0
        gen_std = float(gen_dev.std().item()) or 1.0

        report = RobustnessReport(
            threshold  = self.threshold,
            n_subjects = n,
            mri_std    = mri_std,
            gen_std    = gen_std,
        )

        # ---- Baseline ----
        logger.info("Robustness — baseline inference (%d subjects)…", n)
        baseline_prob = _run_inference(
            model, mri_dev, gen_dev, device, self.batch_size
        )
        baseline_metrics = _safe_metrics(y_true, baseline_prob, self.threshold)
        report.baseline = ConditionResult(
            condition   = "baseline",
            n_subjects  = n,
            metrics     = baseline_metrics,
            is_baseline = True,
        )
        logger.info(
            "  Baseline: AUC=%.4f  F1=%.4f  Sens=%.4f  Spec=%.4f",
            baseline_metrics.get("auc", 0),
            baseline_metrics.get("f1", 0),
            baseline_metrics.get("sensitivity", 0),
            baseline_metrics.get("specificity", 0),
        )

        # ---- Perturbation conditions ----
        perturbations = _standard_perturbations()
        if extra_perturbations:
            perturbations.extend(extra_perturbations)

        for pert in perturbations:
            logger.info("  Testing: %s", pert.name)
            try:
                mri_p, gen_p = pert.apply(
                    mri_dev.clone(), gen_dev.clone(),
                    mri_std, gen_std, self.rng,
                )
                p_prob = _run_inference(
                    model, mri_p, gen_p, device, self.batch_size
                )
                metrics = _safe_metrics(y_true, p_prob, self.threshold)
                cond = _make_condition(
                    pert.name, n, metrics, baseline_metrics
                )
                report.conditions.append(cond)
            except Exception as exc:
                logger.warning("    Skipped (%s): %s", pert.name, exc)

        # ---- Site-holdout tests ----
        if site_ids is not None:
            site_conditions = self._site_holdout(
                model, mri_dev, gen_dev, y_true, site_ids,
                device, baseline_metrics,
            )
            report.conditions.extend(site_conditions)

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            report.save(out_dir / "robustness_report.json")
            latex_path = out_dir / "robustness_table.tex"
            latex_path.write_text(report.to_latex_table(), encoding="utf-8")
            logger.info("LaTeX robustness table → %s", latex_path)

        return report

    def _site_holdout(
        self,
        model:    nn.Module,
        mri:      torch.Tensor,
        genetics: torch.Tensor,
        y_true:   np.ndarray,
        site_ids: np.ndarray,
        device:   torch.device,
        baseline: Dict[str, float],
    ) -> List[ConditionResult]:
        """
        Leave-one-site-out evaluation: for each qualifying site, measure
        performance on subjects from that site only.

        A site qualifies if it has at least `min_site_n` subjects AND
        contains at least one sample from each class (both ASD and TC).
        Sites that fail this check are skipped with a warning.
        """
        site_ids = np.asarray(site_ids, dtype=str)
        sites    = np.unique(site_ids)
        results: List[ConditionResult] = []

        qualifying = []
        for site in sites:
            mask  = site_ids == site
            n_s   = int(mask.sum())
            n_pos = int(y_true[mask].sum())
            n_neg = n_s - n_pos
            if n_s >= self.min_site_n and n_pos >= 1 and n_neg >= 1:
                qualifying.append(site)

        if len(qualifying) < 2:
            logger.info(
                "Site holdout: only %d qualifying site(s) — skipped "
                "(need ≥2 with n≥%d and both classes)",
                len(qualifying), self.min_site_n,
            )
            return results

        logger.info(
            "Site holdout: evaluating %d sites individually", len(qualifying)
        )
        for site in qualifying:
            mask = site_ids == site
            idx  = np.where(mask)[0]

            mri_s = mri[idx]
            gen_s = genetics[idx]
            yt_s  = y_true[idx]

            try:
                p_prob = _run_inference(
                    model, mri_s, gen_s, device, self.batch_size
                )
                metrics = _safe_metrics(yt_s, p_prob, self.threshold)
                cond = _make_condition(
                    f"site_holdout_{site}", int(mask.sum()),
                    metrics, baseline,
                )
                results.append(cond)
                logger.info(
                    "    Site %s (n=%d): AUC=%.4f  ΔAUC=%+.4f",
                    site, mask.sum(), metrics.get("auc", 0), cond.delta_auc,
                )
            except Exception as exc:
                logger.warning("    Site %s skipped: %s", site, exc)

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_condition(
    name:     str,
    n:        int,
    metrics:  Dict[str, float],
    baseline: Dict[str, float],
) -> ConditionResult:
    return ConditionResult(
        condition  = name,
        n_subjects = n,
        metrics    = metrics,
        delta_auc  = metrics.get("auc", 0.0)         - baseline.get("auc", 0.0),
        delta_f1   = metrics.get("f1", 0.0)          - baseline.get("f1", 0.0),
        delta_sens = metrics.get("sensitivity", 0.0) - baseline.get("sensitivity", 0.0),
        delta_spec = metrics.get("specificity", 0.0) - baseline.get("specificity", 0.0),
    )
