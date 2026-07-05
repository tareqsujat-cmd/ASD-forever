"""
Genetics feature importance methods.

Three complementary methods are implemented:

1. Gradient × Input (GradInput)
   Saliency = |∂F/∂x_i * x_i|.
   Fast, works for any differentiable architecture.

2. TabNet sparse attention masks
   TabNet naturally produces feature selection masks at each step.
   ``get_feature_importances()`` returns the mean mask across steps,
   giving a normalized per-feature importance vector.

3. Transformer attention from CLS token
   After rollout, the CLS-token row gives a gene-level importance vector
   that captures which genes the model attends to across all layers.

``GeneticsFeatureImportance.aggregate()`` returns a consensus importance
vector by taking the mean of all available methods after normalizing each
to sum to 1.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _normalize_to_sum1(x: torch.Tensor) -> torch.Tensor:
    """Normalize a non-negative vector so that its entries sum to 1."""
    total = x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return x / total


# ---------------------------------------------------------------------------
# GeneticsFeatureImportance
# ---------------------------------------------------------------------------

class GeneticsFeatureImportance:
    """
    Compute per-gene feature importance from a genetics encoder.

    Parameters
    ----------
    model : nn.Module
        Full ASDModel, a GeneticsClassifier, or any model with a forward
        that takes genetics (and optionally MRI) and returns a dict with
        ``"logits"`` or a plain ``(B, C)`` tensor.
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
    # Method 1: Gradient × Input
    # ------------------------------------------------------------------

    def gradient_times_input(
        self,
        genetics: torch.Tensor,
        mri: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        target: int = 1,
    ) -> torch.Tensor:
        """
        Saliency map via element-wise product of gradient and input value.

        GradInput_i(x) = |∂F(x)/∂x_i * x_i|

        This is equivalent to the first-order term of a Taylor expansion around
        zero and gives a quick approximation of feature influence.

        Returns
        -------
        saliency : (B, n_genes) non-negative float tensor
        """
        training = self.model.training
        self.model.eval()

        genetics = genetics.to(self.device).float()
        gen_req = genetics.detach().requires_grad_(True)

        if mri is not None:
            mri = mri.to(self.device).detach()
        if adj is not None:
            adj = adj.to(self.device).detach()

        with torch.enable_grad():
            out = self._forward_genetics(gen_req, mri, adj)
            logits = out["logits"] if isinstance(out, dict) else out
            score = logits[:, target].sum()
            (grad,) = torch.autograd.grad(
                score, gen_req, create_graph=False, retain_graph=False
            )

        self.model.train(training)
        return (grad.detach() * genetics.detach()).abs()  # (B, n_genes)

    # ------------------------------------------------------------------
    # Method 2: TabNet sparse attention masks
    # ------------------------------------------------------------------

    def tabnet_importance(
        self,
        genetics: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Feature importance from TabNet's sparse attention masks.

        Returns the mean attention mask across all sequential attention steps,
        averaged over the batch.  Returns None if the model does not use TabNet.

        Returns
        -------
        importance : (n_genes,) non-negative float tensor, or None
        """
        backbone = self._get_tabnet_backbone()
        if backbone is None:
            return None

        genetics = genetics.to(self.device).float()
        self.model.eval()
        with torch.no_grad():
            _ = backbone(genetics)
            imp = backbone.get_feature_importances()
        return imp.detach()  # (n_genes,)

    # ------------------------------------------------------------------
    # Method 3: Transformer attention-based importance
    # ------------------------------------------------------------------

    def attention_importance(
        self,
        genetics: torch.Tensor,
        discard_ratio: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """
        Per-gene importance from CLS-token attention rollout.

        Only applicable when the genetics encoder is a transformer that
        exposes ``get_attention_weights()``.

        Returns
        -------
        importance : (B, n_genes) float tensor in [0, 1] per sample, or None
        """
        from explainability.attention_viz import attention_rollout, rollout_cls_to_tokens

        backbone = self._get_transformer_backbone()
        if backbone is None or not hasattr(backbone, "get_attention_weights"):
            return None

        genetics = genetics.to(self.device).float()
        backbone.eval()
        with torch.no_grad():
            attn_list = backbone.get_attention_weights(genetics)

        rollout = attention_rollout(attn_list, discard_ratio=discard_ratio)
        return rollout_cls_to_tokens(rollout).detach()  # (B, n_genes)

    # ------------------------------------------------------------------
    # Ensemble aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        genetics: torch.Tensor,
        mri: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        target: int = 1,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Compute all available importance methods and return them in a dict.

        Also returns a ``"consensus"`` key: element-wise mean of all
        non-None methods after normalizing each to sum-to-1, averaged over
        the batch.

        Parameters
        ----------
        methods : list of str, optional
            Subset of ``["gradient_times_input", "tabnet", "attention"]``.
            Defaults to all three.

        Returns
        -------
        dict:
            "gradient_times_input" : (B, n_genes) or None
            "tabnet"               : (n_genes,) or None
            "attention"            : (B, n_genes) or None
            "consensus"            : (n_genes,) normalized or None
        """
        if methods is None:
            methods = ["gradient_times_input", "tabnet", "attention"]

        result: Dict[str, Optional[torch.Tensor]] = {}

        if "gradient_times_input" in methods:
            result["gradient_times_input"] = self.gradient_times_input(
                genetics, mri=mri, adj=adj, target=target
            )

        if "tabnet" in methods:
            result["tabnet"] = self.tabnet_importance(genetics)

        if "attention" in methods:
            result["attention"] = self.attention_importance(genetics)

        # Consensus: average available methods (normalized, batch-averaged)
        available = []
        for name in ["gradient_times_input", "tabnet", "attention"]:
            v = result.get(name)
            if v is None:
                continue
            if v.dim() == 2:
                v = v.mean(dim=0)   # batch mean → (n_genes,)
            available.append(_normalize_to_sum1(v))

        result["consensus"] = (
            torch.stack(available, dim=0).mean(dim=0)
            if available else None
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_genetics(
        self,
        genetics: torch.Tensor,
        mri: Optional[torch.Tensor],
        adj: Optional[torch.Tensor],
    ):
        if mri is not None:
            return self.model(mri, genetics, adj=adj) \
                   if adj is not None else self.model(mri, genetics)
        if hasattr(self.model, "forward_gen_only"):
            return self.model.forward_gen_only(genetics)
        return self.model(genetics)

    def _get_tabnet_backbone(self) -> Optional[nn.Module]:
        """Navigate model hierarchy to locate a TabNetEncoder."""
        # ASDModel → gen_encoder (GeneticsEncoder) → backbone (TabNetEncoder)
        for attr in ("gen_encoder", "gen_branch"):
            enc = getattr(self.model, attr, None)
            if enc is None:
                continue
            backbone = getattr(enc, "backbone", enc)
            if hasattr(backbone, "get_feature_importances"):
                return backbone
        return None

    def _get_transformer_backbone(self) -> Optional[nn.Module]:
        """Navigate model hierarchy to locate a GeneTransformerEncoder."""
        for attr in ("gen_encoder", "gen_branch"):
            enc = getattr(self.model, attr, None)
            if enc is None:
                continue
            backbone = getattr(enc, "backbone", enc)
            if hasattr(backbone, "get_attention_weights"):
                return backbone
        return None
