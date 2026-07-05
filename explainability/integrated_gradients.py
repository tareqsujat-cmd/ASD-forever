"""
Integrated Gradients (IG) for both MRI and genetics inputs.

IG satisfies two key axioms:
  Sensitivity: if input and baseline differ in feature i and give different
    predictions, then IG_i ≠ 0.
  Implementation Invariance: attributions are identical for any two
    functionally equivalent networks.

The completeness axiom provides a built-in quality check:
  Σ_i IG_i(x) = F(x) - F(x')
up to the Riemann approximation error (typically < 1% with n_steps ≥ 50).

Smooth Integrated Gradients (SmoothIG) reduces noise by averaging IG over
inputs perturbed with small Gaussian noise.

Reference
---------
Sundararajan M et al. (2017). Axiomatic Attribution for Deep Networks.
  ICML 2017. arXiv:1703.01365

Smilkov D et al. (2017). SmoothGrad: removing noise by adding noise.
  arXiv:1706.03825
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Core Riemann sum
# ---------------------------------------------------------------------------

def _riemann_sum(
    x: torch.Tensor,
    baseline: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    n_steps: int,
) -> torch.Tensor:
    """
    Approximate IG by left-endpoint Riemann sum over the straight-line path
    from baseline to input.

    Returns (x - baseline) * (1/n_steps) * Σ_{k=1}^{n_steps} ∂F/∂x (interp_k)
    """
    grads_acc = torch.zeros_like(x, dtype=torch.float32)
    for k in range(1, n_steps + 1):
        alpha = k / n_steps
        interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        with torch.enable_grad():
            score = score_fn(interp)
        (grad,) = torch.autograd.grad(
            score.sum(), interp, create_graph=False, retain_graph=False
        )
        grads_acc = grads_acc + grad.detach()
    return (x - baseline).detach().float() * grads_acc / n_steps


def _convergence_delta(
    x: torch.Tensor,
    baseline: torch.Tensor,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    attributions: torch.Tensor,
) -> float:
    """
    Completeness check: |Σ_i IG_i - (F(x)-F(x'))| / (|F(x)-F(x')| + ε).

    A delta < 0.05 (5%) indicates good Riemann approximation.
    """
    with torch.no_grad():
        f_x = score_fn(x.detach()).float()
        f_b = score_fn(baseline.detach()).float()
    diff = (f_x - f_b).detach()  # (B,)
    ig_sum = attributions.view(x.shape[0], -1).sum(dim=1)  # (B,)
    delta = (ig_sum - diff).abs() / (diff.abs() + 1e-8)
    return float(delta.mean())


# ---------------------------------------------------------------------------
# IntegratedGradients
# ---------------------------------------------------------------------------

class IntegratedGradients:
    """
    Integrated Gradients for ASDModel (or any model with a dict output
    containing ``"logits"``).

    Parameters
    ----------
    model : nn.Module
    device : torch.device, optional
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or _get_model_device(model)

    # ------------------------------------------------------------------
    # MRI attribution
    # ------------------------------------------------------------------

    def attribute_mri(
        self,
        mri: torch.Tensor,
        genetics: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        target: int = 1,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50,
    ) -> Tuple[torch.Tensor, float]:
        """
        IG attribution w.r.t. the MRI input.

        Parameters
        ----------
        mri       : (B, C, D, H, W)
        genetics  : (B, n_genes) held fixed during integration
        target    : class index to explain (1 = ASD)
        baseline  : same shape as ``mri``; defaults to zeros
        n_steps   : Riemann steps (≥50 recommended)

        Returns
        -------
        attributions : (B, C, D, H, W) — IG scores per voxel channel
        delta : float — convergence error (< 0.05 is good)
        """
        mri = mri.to(self.device).float()
        if genetics is not None:
            genetics = genetics.to(self.device).detach()
        if adj is not None:
            adj = adj.to(self.device).detach()
        if baseline is None:
            baseline = torch.zeros_like(mri)
        else:
            baseline = baseline.to(self.device).float()

        training = self.model.training
        self.model.eval()

        def _score(mri_interp: torch.Tensor) -> torch.Tensor:
            if genetics is not None:
                out = self.model(mri_interp, genetics, adj=adj) \
                      if adj is not None else self.model(mri_interp, genetics)
            elif hasattr(self.model, "forward_mri_only"):
                out = self.model.forward_mri_only(mri_interp)
            else:
                out = self.model(mri_interp)
            logits = out["logits"] if isinstance(out, dict) else out
            return logits[:, target]

        attrs = _riemann_sum(mri, baseline, _score, n_steps)
        delta = _convergence_delta(mri, baseline, _score, attrs)

        self.model.train(training)
        return attrs.detach(), delta

    # ------------------------------------------------------------------
    # Genetics attribution
    # ------------------------------------------------------------------

    def attribute_genetics(
        self,
        genetics: torch.Tensor,
        mri: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        target: int = 1,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50,
    ) -> Tuple[torch.Tensor, float]:
        """
        IG attribution w.r.t. the genetics input.

        Parameters
        ----------
        genetics  : (B, n_genes)
        mri       : (B, C, D, H, W) held fixed during integration
        target    : class index to explain (1 = ASD)
        baseline  : same shape as ``genetics``; defaults to zeros
        n_steps   : Riemann steps

        Returns
        -------
        attributions : (B, n_genes) — IG scores per gene
        delta : float — convergence error
        """
        genetics = genetics.to(self.device).float()
        if mri is not None:
            mri = mri.to(self.device).detach()
        if adj is not None:
            adj = adj.to(self.device).detach()
        if baseline is None:
            baseline = torch.zeros_like(genetics)
        else:
            baseline = baseline.to(self.device).float()

        training = self.model.training
        self.model.eval()

        def _score(gen_interp: torch.Tensor) -> torch.Tensor:
            if mri is not None:
                out = self.model(mri, gen_interp, adj=adj) \
                      if adj is not None else self.model(mri, gen_interp)
            elif hasattr(self.model, "forward_gen_only"):
                out = self.model.forward_gen_only(gen_interp)
            else:
                out = self.model(gen_interp)
            logits = out["logits"] if isinstance(out, dict) else out
            return logits[:, target]

        attrs = _riemann_sum(genetics, baseline, _score, n_steps)
        delta = _convergence_delta(genetics, baseline, _score, attrs)

        self.model.train(training)
        return attrs.detach(), delta

    # ------------------------------------------------------------------
    # SmoothGrad (noise-averaged gradient)
    # ------------------------------------------------------------------

    def smooth_grad_mri(
        self,
        mri: torch.Tensor,
        genetics: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        target: int = 1,
        n_samples: int = 50,
        noise_std: float = 0.15,
    ) -> torch.Tensor:
        """
        SmoothGrad for MRI: average of gradients over noise-perturbed inputs.

        SmoothGrad_i(x) = E_{ε~N(0,σ²)}[∂F/∂x_i (x + ε)]

        Returns
        -------
        (B, C, D, H, W) averaged gradient tensor
        """
        mri = mri.to(self.device).float()
        if genetics is not None:
            genetics = genetics.to(self.device).detach()
        if adj is not None:
            adj = adj.to(self.device).detach()

        training = self.model.training
        self.model.eval()

        def _score(mri_in: torch.Tensor) -> torch.Tensor:
            if genetics is not None:
                out = self.model(mri_in, genetics, adj=adj) \
                      if adj is not None else self.model(mri_in, genetics)
            elif hasattr(self.model, "forward_mri_only"):
                out = self.model.forward_mri_only(mri_in)
            else:
                out = self.model(mri_in)
            logits = out["logits"] if isinstance(out, dict) else out
            return logits[:, target]

        grads_list = []
        sigma = noise_std * (mri.max() - mri.min()).item()
        for _ in range(n_samples):
            x_noisy = (mri + sigma * torch.randn_like(mri)).detach().requires_grad_(True)
            with torch.enable_grad():
                score = _score(x_noisy)
            (grad,) = torch.autograd.grad(
                score.sum(), x_noisy, create_graph=False, retain_graph=False
            )
            grads_list.append(grad.detach())

        self.model.train(training)
        return torch.stack(grads_list, dim=0).mean(dim=0)
