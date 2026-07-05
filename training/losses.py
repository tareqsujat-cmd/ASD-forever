"""
Loss functions for ASD classification.

Three options are provided for flexibility across ablation studies:

FocalLoss
---------
Focal Loss (Lin et al., 2017) down-weights easy examples and focuses training
on hard, misclassified examples.  For ASD detection:
  - Many TC subjects are "easy" (clear structural normalcy)
  - A subset of ASD subjects are hard (mild/atypical presentation)
  - Focal loss allocates more gradient to the hard ASD cases

  FL(p_t) = -α_t * (1 − p_t)^γ * log(p_t)
  α=0.25, γ=2.0 (default from the original paper)

BalancedCrossEntropyLoss
------------------------
Standard cross-entropy with inverse-frequency class weights computed
from the training set.  Simpler than focal loss, useful as a baseline.

LabelSmoothingCrossEntropyLoss
-------------------------------
Cross-entropy with label smoothing (Szegedy et al., 2016).
Prevents the model from becoming overconfident, reducing overfitting.
Particularly important for small datasets like ABIDE (~500 subjects/fold).

References
----------
Lin TY et al. (2017). Focal Loss for Dense Object Detection. ICCV.
Szegedy C et al. (2016). Rethinking the inception architecture. CVPR.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Multiclass focal loss.

    Parameters
    ----------
    alpha : float
        Weighting factor for the positive (ASD) class.
        Set < 0.5 to down-weight the majority class.
    gamma : float
        Focusing parameter.  0 → standard CE.  2 → standard focal.
    label_smoothing : float
        Optional label smoothing applied before focal weighting.
    reduction : str
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C)  — raw unnormalised scores
        targets : (B,)    — integer class indices

        Returns
        -------
        scalar loss
        """
        num_classes = logits.shape[-1]

        # Standard per-sample cross-entropy (no reduction)
        ce_loss = F.cross_entropy(
            logits, targets,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )  # (B,)

        # p_t: predicted probability of the true class
        p_t = torch.exp(-ce_loss)

        # Focal weighting: (1 − p_t)^γ — down-weights easy examples
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting: class-specific factor
        # alpha for positive class (1=ASD), 1-alpha for negative (0=TC)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(ce_loss, self.alpha),
                              torch.full_like(ce_loss, 1.0 - self.alpha))

        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # "none"


class BalancedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with inverse-frequency class weights.

    Call update_weights(labels) at the start of each fold to compute
    weights from the training set label distribution.
    """

    def __init__(
        self,
        num_classes: int = 2,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.register_buffer(
            "class_weights", torch.ones(num_classes)
        )

    def update_weights(self, labels: torch.Tensor) -> None:
        """Compute inverse-frequency weights from training labels."""
        counts = torch.bincount(labels.long(), minlength=self.num_classes).float()
        weights = counts.sum() / (self.num_classes * counts.clamp(min=1))
        self.class_weights = weights.to(self.class_weights.device)
        logger.info("BalancedCE class weights: %s", weights.tolist())

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


def build_criterion(cfg) -> nn.Module:
    """
    Build loss function from configuration.

    Supports:
      cfg.training.loss.type = "focal" | "cross_entropy" | "balanced"
    """
    loss_cfg = cfg.training.loss

    if loss_cfg.type == "focal":
        criterion = FocalLoss(
            alpha=loss_cfg.focal_alpha,
            gamma=loss_cfg.focal_gamma,
            label_smoothing=loss_cfg.label_smoothing,
        )
        logger.info(
            "Criterion: FocalLoss(α=%.2f, γ=%.1f, smoothing=%.2f)",
            loss_cfg.focal_alpha, loss_cfg.focal_gamma, loss_cfg.label_smoothing,
        )

    elif loss_cfg.type == "balanced":
        criterion = BalancedCrossEntropyLoss(
            label_smoothing=loss_cfg.label_smoothing,
        )
        logger.info(
            "Criterion: BalancedCrossEntropyLoss(smoothing=%.2f)",
            loss_cfg.label_smoothing,
        )

    elif loss_cfg.type == "cross_entropy":
        criterion = nn.CrossEntropyLoss(
            label_smoothing=loss_cfg.label_smoothing,
        )
        logger.info(
            "Criterion: CrossEntropyLoss(smoothing=%.2f)",
            loss_cfg.label_smoothing,
        )

    else:
        raise ValueError(f"Unknown loss type: '{loss_cfg.type}'")

    return criterion
