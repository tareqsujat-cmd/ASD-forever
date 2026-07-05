"""
GradCAM and GradCAM++ for 3D MRI volumes.

Both methods produce a spatial saliency map (same spatial dimensions as the
target convolutional layer's feature maps) showing which voxel regions most
influenced the model's prediction for a given class.

GradCAM weights each feature map channel by the global-average-pooled gradient
of the class score — fast but may miss localized discriminative regions.

GradCAM++ uses per-pixel alpha weights that account for second- and third-order
gradients, giving sharper localization when multiple regions are discriminative.

References
----------
Selvaraju RR et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization. ICCV 2017. arXiv:1610.02391

Chattopadhay A et al. (2018). Grad-CAM++: Generalized Gradient-Based Visual
  Explanations for Deep Convolutional Networks. WACV 2018. arXiv:1710.11063
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook context manager
# ---------------------------------------------------------------------------

class _CAMHooks:
    """Register forward and backward hooks on a single layer for one forward pass."""

    def __init__(self, layer: nn.Module) -> None:
        self._layer = layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_handle = None
        self._bwd_handle = None

    def __enter__(self) -> "_CAMHooks":
        self._fwd_handle = self._layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = self._layer.register_full_backward_hook(self._bwd_hook)
        return self

    def __exit__(self, *_) -> None:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()

    def _fwd_hook(self, module, inp, output) -> None:
        self.activations = output.detach()

    def _bwd_hook(self, module, grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_last_conv3d(model: nn.Module) -> Optional[nn.Module]:
    """Return the last ``nn.Conv3d`` in the model (depth-first pre-order scan)."""
    last_conv: Optional[nn.Module] = None
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            last_conv = m
    return last_conv


def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalization so each map lies in [0, 1]."""
    B = cam.shape[0]
    flat = cam.view(B, -1)
    lo = flat.min(dim=1).values.view(B, *([1] * (cam.ndim - 1)))
    hi = flat.max(dim=1).values.view(B, *([1] * (cam.ndim - 1)))
    return (cam - lo) / (hi - lo + 1e-8)


def _upsample_cam(
    cam: torch.Tensor,
    size: Tuple[int, int, int],
) -> torch.Tensor:
    """Trilinear upsample (B, d, h, w) → (B, D', H', W')."""
    return F.interpolate(
        cam.unsqueeze(1).float(),
        size=size,
        mode="trilinear",
        align_corners=False,
    ).squeeze(1)


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# GradCAM3D
# ---------------------------------------------------------------------------

class GradCAM3D:
    """
    GradCAM for 3D convolutional networks (MRI volumes).

    Usage
    -----
    ::

        cam_computer = GradCAM3D(model, target_layer=model.encoder.layer4)
        cam = cam_computer.compute({"mri": mri_tensor}, target_class=1,
                                   interpolate_to=(91, 109, 91))

    Parameters
    ----------
    model : nn.Module
        Must output a dict with key ``"logits"`` or a plain ``(B, C)`` tensor.
    target_layer : nn.Module, optional
        Conv layer at which CAM is computed.  Auto-detected (last Conv3d) if
        not given.
    device : torch.device, optional
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or _get_model_device(model)
        if target_layer is None:
            target_layer = find_last_conv3d(model)
        if target_layer is None:
            raise ValueError(
                "No nn.Conv3d found in model. Pass target_layer explicitly."
            )
        self.target_layer = target_layer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class: int = 1,
        interpolate_to: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        """
        Compute the GradCAM saliency map.

        Parameters
        ----------
        inputs : dict
            Must contain ``"mri"`` key with tensor ``(B, C, D, H, W)``.
            May also contain ``"genetics"`` ``(B, n_genes)`` and ``"adj"``.
        target_class : int
            Class index for which gradients are computed (1 = ASD).
        interpolate_to : (D', H', W'), optional
            Trilinear upsample to this spatial size (e.g. input MRI shape)
            so the map can be overlaid on the raw volume.

        Returns
        -------
        cam : (B, d, h, w) or (B, D', H', W') float tensor in ``[0, 1]``.
        """
        training = self.model.training
        self.model.eval()

        with _CAMHooks(self.target_layer) as hooks:
            out = self._forward(self._to_device(inputs))
            logits = out["logits"] if isinstance(out, dict) else out
            score = logits[:, target_class].sum()
            self.model.zero_grad()
            score.backward()

        self.model.train(training)

        A = hooks.activations   # (B, C, d, h, w)
        g = hooks.gradients     # (B, C, d, h, w)

        # alpha_k = global-average-pool of gradients
        weights = g.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        cam = F.relu((weights * A).sum(dim=1))          # (B, d, h, w)
        cam = _normalize_cam(cam)

        if interpolate_to is not None:
            cam = _upsample_cam(cam, interpolate_to)

        return cam.detach()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, inputs: Dict) -> Dict:
        mri = inputs["mri"]
        gen = inputs.get("genetics")
        adj = inputs.get("adj")
        if gen is not None:
            out = self.model(mri, gen, adj=adj) if adj is not None \
                  else self.model(mri, gen)
        elif hasattr(self.model, "forward_mri_only"):
            out = self.model.forward_mri_only(mri)
        else:
            out = self.model(mri)
        return out if isinstance(out, dict) else {"logits": out}

    def _to_device(self, inputs: Dict) -> Dict:
        return {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }


# ---------------------------------------------------------------------------
# GradCAM++3D
# ---------------------------------------------------------------------------

class GradCAMPlusPlus3D(GradCAM3D):
    """
    GradCAM++ for 3D CNNs.

    Replaces the global-average-pooled gradient weights with per-pixel alpha
    weights that account for second- and third-order derivatives of the class
    score with respect to the feature map activations.

    The per-pixel alpha for channel k at location (d, h, w):

        alpha_k_dhw = g² / ( 2g² + A_total * g³ + ε )

    Channel weight:

        w_k = Σ_{d,h,w} alpha_k_dhw * ReLU( g_k_dhw )
    """

    def compute(
        self,
        inputs: Dict[str, torch.Tensor],
        target_class: int = 1,
        interpolate_to: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        training = self.model.training
        self.model.eval()

        with _CAMHooks(self.target_layer) as hooks:
            out = self._forward(self._to_device(inputs))
            logits = out["logits"] if isinstance(out, dict) else out
            score = logits[:, target_class].sum()
            self.model.zero_grad()
            score.backward()

        self.model.train(training)

        A = hooks.activations   # (B, C, d, h, w)
        g = hooks.gradients     # (B, C, d, h, w)

        g2 = g ** 2
        g3 = g ** 3
        A_sum = A.sum(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        denom = 2.0 * g2 + A_sum * g3 + 1e-8
        alpha = g2 / denom                          # (B, C, d, h, w)

        weights = (alpha * F.relu(g)).sum(
            dim=(2, 3, 4), keepdim=True
        )  # (B, C, 1, 1, 1)

        cam = F.relu((weights * A).sum(dim=1))  # (B, d, h, w)
        cam = _normalize_cam(cam)

        if interpolate_to is not None:
            cam = _upsample_cam(cam, interpolate_to)

        return cam.detach()
