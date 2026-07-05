"""
ExplainabilityEngine — unified API for all explanation methods.

The engine automatically detects the model architecture and dispatches to
the appropriate explanation methods:

  MRI saliency    : GradCAM, GradCAM++, Integrated Gradients, SmoothGrad
  Genetics        : Gradient × Input, TabNet masks, Attention rollout
  Fusion          : Cross-modal attention weights

Unavailable methods (e.g. attention for a CNN-only model) return None
gracefully rather than raising errors.

Usage
-----
::

    engine = ExplainabilityEngine(model, device=device)

    # Full explanation
    result = engine.explain(mri, genetics)

    # Selective
    result = engine.explain_mri(mri, genetics, methods=["gradcam"])
    result = engine.explain_genetics(genetics, mri, methods=["gradient_times_input"])
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class ExplainabilityEngine:
    """
    Unified explanation API for ASDModel (or any compatible model).

    Parameters
    ----------
    model : nn.Module
        The trained model to explain.  Expected to output a dict with
        ``"logits"`` and optionally ``"mri_features"`` / ``"gen_features"``.
        Also expected to have ``forward_mri_only`` and ``forward_gen_only``
        methods (used for attribution without the other modality).
    device : torch.device, optional
    gradcam_target_layer : nn.Module, optional
        Override the auto-detected Conv3d target layer for GradCAM / GradCAM++.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        gradcam_target_layer: Optional[nn.Module] = None,
    ) -> None:
        self.model = model
        self.device = device or _get_model_device(model)
        self._gradcam_target_layer = gradcam_target_layer

        # Lazily initialized sub-explainers
        self._gradcam: Optional[object] = None
        self._gradcam_pp: Optional[object] = None
        self._ig: Optional[object] = None
        self._feat_imp: Optional[object] = None
        self._attn_ext: Optional[object] = None

    # ------------------------------------------------------------------
    # MRI explanations
    # ------------------------------------------------------------------

    def explain_mri(
        self,
        mri: torch.Tensor,
        genetics: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        target_class: int = 1,
        methods: Optional[List[str]] = None,
        interpolate_to_input: bool = True,
        n_steps: int = 50,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Compute MRI saliency maps.

        Parameters
        ----------
        mri           : (B, C, D, H, W)
        genetics      : (B, n_genes) — if provided, used as fixed context
        target_class  : class index (1 = ASD)
        methods       : list of ``["gradcam", "gradcam_pp", "integrated_gradients",
                        "smooth_grad"]``; defaults to first three
        interpolate_to_input : bool
            Upsample GradCAM / GradCAM++ maps to the input MRI spatial size.
        n_steps       : Riemann steps for IG (default 50)

        Returns
        -------
        dict with one key per requested method → (B, D, H, W) or None
        """
        if methods is None:
            methods = ["gradcam", "gradcam_pp", "integrated_gradients"]

        inputs = {"mri": mri}
        if genetics is not None:
            inputs["genetics"] = genetics
        if adj is not None:
            inputs["adj"] = adj

        interp_size = tuple(mri.shape[2:]) if interpolate_to_input else None
        result: Dict[str, Optional[torch.Tensor]] = {}

        if "gradcam" in methods:
            try:
                gc = self._get_gradcam()
                result["gradcam"] = gc.compute(
                    inputs, target_class=target_class, interpolate_to=interp_size
                )
            except Exception as e:
                logger.warning("GradCAM failed: %s", e)
                result["gradcam"] = None

        if "gradcam_pp" in methods:
            try:
                gc_pp = self._get_gradcam_pp()
                result["gradcam_pp"] = gc_pp.compute(
                    inputs, target_class=target_class, interpolate_to=interp_size
                )
            except Exception as e:
                logger.warning("GradCAM++ failed: %s", e)
                result["gradcam_pp"] = None

        if "integrated_gradients" in methods:
            try:
                ig = self._get_ig()
                attrs, delta = ig.attribute_mri(
                    mri, genetics=genetics, adj=adj,
                    target=target_class, n_steps=n_steps,
                )
                result["integrated_gradients"] = attrs
                result["ig_convergence_delta"] = delta
            except Exception as e:
                logger.warning("IG (MRI) failed: %s", e)
                result["integrated_gradients"] = None
                result["ig_convergence_delta"] = None

        if "smooth_grad" in methods:
            try:
                ig = self._get_ig()
                result["smooth_grad"] = ig.smooth_grad_mri(
                    mri, genetics=genetics, adj=adj, target=target_class
                )
            except Exception as e:
                logger.warning("SmoothGrad failed: %s", e)
                result["smooth_grad"] = None

        return result

    # ------------------------------------------------------------------
    # Genetics explanations
    # ------------------------------------------------------------------

    def explain_genetics(
        self,
        genetics: torch.Tensor,
        mri: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        target_class: int = 1,
        methods: Optional[List[str]] = None,
        n_steps: int = 50,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Compute genetics feature importance scores.

        Parameters
        ----------
        genetics      : (B, n_genes)
        mri           : (B, C, D, H, W) optional fixed context
        methods       : list of ``["gradient_times_input", "integrated_gradients",
                        "tabnet", "attention"]``; defaults to all four
        n_steps       : IG Riemann steps

        Returns
        -------
        dict with per-method tensors:
            "gradient_times_input" : (B, n_genes) or None
            "integrated_gradients" : (B, n_genes) or None
            "ig_convergence_delta" : float or None
            "tabnet"               : (n_genes,) or None
            "attention"            : (B, n_genes) or None
            "consensus"            : (n_genes,) or None
        """
        if methods is None:
            methods = [
                "gradient_times_input", "integrated_gradients", "tabnet", "attention"
            ]

        result: Dict[str, Optional[torch.Tensor]] = {}

        feat_methods = [m for m in methods if m in (
            "gradient_times_input", "tabnet", "attention"
        )]
        if feat_methods:
            try:
                fi = self._get_feat_imp()
                fi_result = fi.aggregate(
                    genetics, mri=mri, adj=adj, target=target_class,
                    methods=feat_methods,
                )
                result.update(fi_result)
            except Exception as e:
                logger.warning("FeatureImportance failed: %s", e)

        if "integrated_gradients" in methods:
            try:
                ig = self._get_ig()
                attrs, delta = ig.attribute_genetics(
                    genetics, mri=mri, adj=adj,
                    target=target_class, n_steps=n_steps,
                )
                result["integrated_gradients"] = attrs
                result["ig_convergence_delta"] = delta
            except Exception as e:
                logger.warning("IG (genetics) failed: %s", e)
                result["integrated_gradients"] = None
                result["ig_convergence_delta"] = None

        return result

    # ------------------------------------------------------------------
    # Fusion explanations
    # ------------------------------------------------------------------

    def explain_fusion(
        self,
        mri: torch.Tensor,
        genetics: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[object]]:
        """
        Extract attention weights from the fusion and genetics transformer.

        Returns
        -------
        dict:
            "genetics_attention"   : list of (B, n_heads, N, N) per layer
            "genetics_rollout"     : (B, N+1, N+1) attention rollout
            "gene_importance_attn" : (B, n_genes) per-gene from CLS rollout
            "fusion_attention"     : {"mri_to_gen": ..., "gen_to_mri": ...} or None
        """
        result: Dict[str, Optional[object]] = {}
        ae = self._get_attn_ext()

        # Genetics transformer attention
        result["genetics_attention"] = ae.get_genetics_attention(genetics)
        result["genetics_rollout"] = ae.get_genetics_rollout(genetics)
        result["gene_importance_attn"] = ae.get_gene_importance_from_attention(genetics)

        # Fusion cross-attention — needs encoded features
        mri_feat, gen_feat = self._encode_features(mri, genetics, adj)
        if mri_feat is not None and gen_feat is not None:
            result["fusion_attention"] = ae.get_fusion_attention(mri_feat, gen_feat)
        else:
            result["fusion_attention"] = None

        return result

    # ------------------------------------------------------------------
    # Unified explain
    # ------------------------------------------------------------------

    def explain(
        self,
        mri: torch.Tensor,
        genetics: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
        target_class: int = 1,
        mri_methods: Optional[List[str]] = None,
        gen_methods: Optional[List[str]] = None,
        include_fusion: bool = True,
    ) -> Dict[str, Dict]:
        """
        Full multi-modal explanation.

        Returns
        -------
        dict with keys ``"mri"``, ``"genetics"``, and optionally ``"fusion"``.
        """
        result = {}
        result["mri"] = self.explain_mri(
            mri, genetics=genetics, adj=adj,
            target_class=target_class, methods=mri_methods,
        )
        result["genetics"] = self.explain_genetics(
            genetics, mri=mri, adj=adj,
            target_class=target_class, methods=gen_methods,
        )
        if include_fusion:
            result["fusion"] = self.explain_fusion(mri, genetics, adj=adj)
        return result

    # ------------------------------------------------------------------
    # Lazy initializers
    # ------------------------------------------------------------------

    def _get_gradcam(self):
        if self._gradcam is None:
            from explainability.gradcam import GradCAM3D
            self._gradcam = GradCAM3D(
                self.model,
                target_layer=self._gradcam_target_layer,
                device=self.device,
            )
        return self._gradcam

    def _get_gradcam_pp(self):
        if self._gradcam_pp is None:
            from explainability.gradcam import GradCAMPlusPlus3D
            self._gradcam_pp = GradCAMPlusPlus3D(
                self.model,
                target_layer=self._gradcam_target_layer,
                device=self.device,
            )
        return self._gradcam_pp

    def _get_ig(self):
        if self._ig is None:
            from explainability.integrated_gradients import IntegratedGradients
            self._ig = IntegratedGradients(self.model, device=self.device)
        return self._ig

    def _get_feat_imp(self):
        if self._feat_imp is None:
            from explainability.feature_importance import GeneticsFeatureImportance
            self._feat_imp = GeneticsFeatureImportance(self.model, device=self.device)
        return self._feat_imp

    def _get_attn_ext(self):
        if self._attn_ext is None:
            from explainability.attention_viz import AttentionExtractor
            gen_backbone = self._find_gen_backbone()
            fusion_mod   = getattr(self.model, "fusion", None)
            mri_enc      = getattr(self.model, "mri_encoder", None)
            gen_enc      = getattr(self.model, "gen_encoder", None)
            self._attn_ext = AttentionExtractor(
                genetics_encoder=gen_backbone,
                fusion_module=fusion_mod,
                mri_encoder=mri_enc,
                gen_encoder=gen_enc,
                device=self.device,
            )
        return self._attn_ext

    # ------------------------------------------------------------------
    # Structural helpers
    # ------------------------------------------------------------------

    def _find_gen_backbone(self) -> Optional[nn.Module]:
        for attr in ("gen_encoder", "gen_branch"):
            enc = getattr(self.model, attr, None)
            if enc is None:
                continue
            backbone = getattr(enc, "backbone", enc)
            if hasattr(backbone, "get_attention_weights") or \
               hasattr(backbone, "get_feature_importances"):
                return backbone
            return enc
        return None

    def _encode_features(
        self,
        mri: torch.Tensor,
        genetics: torch.Tensor,
        adj: Optional[torch.Tensor],
    ):
        """
        Run forward pass and extract intermediate encoded features.
        Returns (mri_features, gen_features) or (None, None) on failure.
        """
        mri = mri.to(self.device)
        genetics = genetics.to(self.device)
        if adj is not None:
            adj = adj.to(self.device)

        with torch.no_grad():
            try:
                out = self.model(mri, genetics, adj=adj) \
                      if adj is not None else self.model(mri, genetics)
                mri_feat = out.get("mri_features") if isinstance(out, dict) else None
                gen_feat = out.get("gen_features") if isinstance(out, dict) else None
                return mri_feat, gen_feat
            except Exception as e:
                logger.debug("Feature encoding failed: %s", e)
                return None, None
