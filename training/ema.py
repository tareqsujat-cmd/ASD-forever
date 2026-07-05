"""
Exponential Moving Average (EMA) of model weights.

EMA maintains a shadow copy of model parameters updated as:
    shadow_t = decay * shadow_{t-1}  +  (1 - decay) * param_t

At inference time, the EMA model is used instead of the live model.
EMA smooths out parameter updates across many optimizer steps, acting
as an implicit ensemble of the model across its recent training history.

Benefits for ASD detection on small datasets
---------------------------------------------
- Reduces variance: EMA weights generalise better than last-step weights
- Published effect: 0.5–2 AUC points improvement on small medical datasets
- Standard practice in medical imaging AI (e.g., MeanTeacher, used by ABIDE papers)

Reference
---------
Tarvainen A, Valpola H. (2017). Mean teachers are better role models:
  weight-averaged consistency targets improve semi-supervised learning.
  NeurIPS 2017.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelEMA:
    """
    EMA of model parameters and buffers.

    Parameters
    ----------
    model : nn.Module
        The live training model.  EMA is initialised from its weights.
    decay : float
        EMA decay factor.  0.999 = very slow update (stable).
        0.99 = faster update (follows training more closely).
    device : torch.device, optional
        Device for the shadow model.  Defaults to model's device.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
    ) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model)

        if device is not None:
            self.shadow = self.shadow.to(device)

        # EMA shadow model is never trained directly
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

        self._update_count = 0
        logger.info("ModelEMA: decay=%.4f", decay)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update EMA shadow weights from the live model.
        Call after every optimizer.step().
        """
        self._update_count += 1

        # Bias correction for early training: prevents EMA from being
        # dominated by the initial zero-like state.
        # effective_decay = min(decay, (1 + t) / (10 + t))  — warmup EMA
        decay = min(self.decay, (1 + self._update_count) / (10 + self._update_count))

        for shadow_p, live_p in zip(
            self.shadow.parameters(), model.parameters()
        ):
            shadow_p.data.mul_(decay).add_(live_p.data, alpha=1.0 - decay)

        # Also update buffers (running mean/var in BatchNorm)
        for shadow_b, live_b in zip(
            self.shadow.buffers(), model.buffers()
        ):
            if shadow_b.dtype.is_floating_point:
                shadow_b.data.mul_(decay).add_(live_b.data, alpha=1.0 - decay)
            else:
                shadow_b.data.copy_(live_b.data)

    def state_dict(self) -> dict:
        return {
            "shadow": self.shadow.state_dict(),
            "decay": self.decay,
            "update_count": self._update_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state["shadow"])
        self.decay = state["decay"]
        self._update_count = state["update_count"]

    def __call__(self, *args, **kwargs):
        """Forward pass through the shadow model."""
        return self.shadow(*args, **kwargs)
