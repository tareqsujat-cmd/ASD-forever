"""Explainability suite tests -- CPU-only, no real MRI data needed."""
import sys
sys.path.insert(0, r'e:\ASD_forever')
import torch
import torch.nn as nn
import torch.nn.functional as F

PASS = 0
FAIL = 0

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS: {name}")
        PASS += 1
    else:
        print(f"  FAIL: {name} -- {msg}")
        FAIL += 1


# =========================================================================
# Toy models used throughout the tests
# =========================================================================

class _ToyConvModel(nn.Module):
    """Tiny 3D CNN: shape (B,1,D,H,W) -> dict{logits: (B,2)}."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv3d(4, 8, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool3d(1)
        self.fc    = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return {"logits": self.fc(self.pool(x).flatten(1))}

    def forward_mri_only(self, x):
        return self.forward(x)


class _ToyGeneticsModel(nn.Module):
    """Tiny MLP: shape (B,n_genes) -> dict{logits: (B,2)}."""
    def __init__(self, n_genes=20):
        super().__init__()
        self.fc1 = nn.Linear(n_genes, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        return {"logits": self.fc2(F.relu(self.fc1(x)))}

    def forward_gen_only(self, x):
        return self.forward(x)


class _ToyASDModel(nn.Module):
    """Combined model: (mri, genetics) -> dict{logits, mri_features, gen_features}."""
    def __init__(self, n_genes=20):
        super().__init__()
        self.mri_branch = _ToyConvModel()
        self.gen_branch = _ToyGeneticsModel(n_genes)
        self.fuse = nn.Linear(4, 2)

    def forward(self, mri, genetics, adj=None):
        m = self.mri_branch(mri)["logits"]
        g = self.gen_branch(genetics)["logits"]
        return {
            "logits":       self.fuse(torch.cat([m, g], dim=-1)),
            "mri_features": m,
            "gen_features": g,
        }

    def forward_mri_only(self, mri):
        return self.mri_branch(mri)

    def forward_gen_only(self, genetics):
        return self.gen_branch(genetics)


# =========================================================================
# Synthetic inputs
# =========================================================================
torch.manual_seed(0)
B, C, D, H, W = 2, 1, 8, 8, 8
N_GENES = 20

mri      = torch.randn(B, C, D, H, W)
genetics = torch.randn(B, N_GENES)

conv_model = _ToyConvModel()
gen_model  = _ToyGeneticsModel(N_GENES)
asd_model  = _ToyASDModel(N_GENES)

# =========================================================================
print("=== GradCAM3D ===")
from explainability.gradcam import GradCAM3D, GradCAMPlusPlus3D, find_last_conv3d

gc = GradCAM3D(conv_model)

# Target layer auto-detected
check("GradCAM target layer is Conv3d", isinstance(gc.target_layer, nn.Conv3d))

cam = gc.compute({"mri": mri}, target_class=1)
check("GradCAM output shape",        tuple(cam.shape) == (B, D, H, W),
      str(cam.shape))
check("GradCAM non-negative",        cam.min() >= -1e-6,
      f"min={cam.min():.4f}")
check("GradCAM max <= 1",            cam.max() <= 1.0 + 1e-4,
      f"max={cam.max():.4f}")
check("GradCAM finite",              cam.isfinite().all().item())

# Interpolation
cam_up = gc.compute({"mri": mri}, target_class=1, interpolate_to=(16, 16, 16))
check("GradCAM interpolate shape",   tuple(cam_up.shape) == (B, 16, 16, 16),
      str(cam_up.shape))

# Model stays in correct train state
conv_model.train()
_ = gc.compute({"mri": mri})
check("GradCAM restores train mode", conv_model.training)

# find_last_conv3d
last_conv = find_last_conv3d(conv_model)
check("find_last_conv3d returns Conv3d", isinstance(last_conv, nn.Conv3d))

# =========================================================================
print("\n=== GradCAMPlusPlus3D ===")
gc_pp = GradCAMPlusPlus3D(conv_model)

cam_pp = gc_pp.compute({"mri": mri}, target_class=1)
check("GradCAM++ shape",             tuple(cam_pp.shape) == (B, D, H, W))
check("GradCAM++ non-negative",      cam_pp.min() >= -1e-6)
check("GradCAM++ finite",            cam_pp.isfinite().all().item())

# GradCAM and GradCAM++ should differ (they use different weighting)
diff = (cam - cam_pp).abs().mean().item()
check("GradCAM != GradCAM++",        diff > 0, f"diff={diff:.6f}")

# Works with combined model (mri + genetics)
gc_asd = GradCAM3D(asd_model)
cam_asd = gc_asd.compute({"mri": mri, "genetics": genetics}, target_class=1)
check("GradCAM ASDModel shape",      tuple(cam_asd.shape) == (B, D, H, W))

# =========================================================================
print("\n=== IntegratedGradients (MRI) ===")
from explainability.integrated_gradients import IntegratedGradients

ig = IntegratedGradients(conv_model)
attrs, delta = ig.attribute_mri(mri, n_steps=20)
check("IG MRI attr shape",           tuple(attrs.shape) == (B, C, D, H, W),
      str(attrs.shape))
check("IG MRI attr finite",          attrs.isfinite().all().item())
check("IG MRI convergence delta",    delta < 0.15,
      f"delta={delta:.4f}")

# Baseline other than zeros
baseline = torch.ones_like(mri) * 0.5
attrs_b, delta_b = ig.attribute_mri(mri, baseline=baseline, n_steps=20)
check("IG MRI custom baseline shape", tuple(attrs_b.shape) == (B, C, D, H, W))
check("IG MRI custom baseline finite", attrs_b.isfinite().all().item())

# With genetics context (ASDModel)
ig_asd = IntegratedGradients(asd_model)
attrs_ctx, delta_ctx = ig_asd.attribute_mri(mri, genetics=genetics, n_steps=20)
check("IG MRI+gen context shape",    tuple(attrs_ctx.shape) == (B, C, D, H, W))
check("IG MRI+gen convergence",      delta_ctx < 0.15,
      f"delta={delta_ctx:.4f}")

# =========================================================================
print("\n=== IntegratedGradients (Genetics) ===")
ig_gen = IntegratedGradients(gen_model)
attrs_gen, delta_gen = ig_gen.attribute_genetics(genetics, n_steps=20)
check("IG genetics attr shape",      tuple(attrs_gen.shape) == (B, N_GENES),
      str(attrs_gen.shape))
check("IG genetics attr finite",     attrs_gen.isfinite().all().item())
check("IG genetics convergence",     delta_gen < 0.15,
      f"delta={delta_gen:.4f}")

# With MRI context
attrs_gctx, _ = ig_asd.attribute_genetics(genetics, mri=mri, n_steps=20)
check("IG gen+MRI context shape",    tuple(attrs_gctx.shape) == (B, N_GENES))

# SmoothGrad
sg = ig.smooth_grad_mri(mri, n_samples=10, noise_std=0.1)
check("SmoothGrad shape",            tuple(sg.shape) == (B, C, D, H, W))
check("SmoothGrad finite",           sg.isfinite().all().item())

# =========================================================================
print("\n=== attention_rollout ===")
from explainability.attention_viz import attention_rollout, rollout_cls_to_tokens, AttentionExtractor

N_TOK = 5
attn1 = torch.softmax(torch.randn(B, 4, N_TOK, N_TOK), dim=-1)
attn2 = torch.softmax(torch.randn(B, 4, N_TOK, N_TOK), dim=-1)

rollout = attention_rollout([attn1, attn2])
check("Rollout shape",               tuple(rollout.shape) == (B, N_TOK, N_TOK),
      str(rollout.shape))
check("Rollout rows sum to 1",       (rollout.sum(dim=-1) - 1).abs().max() < 1e-4,
      f"max row sum error={(rollout.sum(dim=-1) - 1).abs().max():.6f}")
check("Rollout non-negative",        rollout.min() >= 0.0,
      f"min={rollout.min():.6f}")

cls_attn = rollout_cls_to_tokens(rollout)
check("CLS attn shape",              tuple(cls_attn.shape) == (B, N_TOK - 1),
      str(cls_attn.shape))
check("CLS attn sums to 1",          (cls_attn.sum(dim=-1) - 1).abs().max() < 1e-4)

# Single-layer rollout
rollout1 = attention_rollout([attn1])
check("Rollout 1-layer shape",       tuple(rollout1.shape) == (B, N_TOK, N_TOK))

# Discard ratio
rollout_d = attention_rollout([attn1, attn2], discard_ratio=0.5)
check("Rollout discard shape",       tuple(rollout_d.shape) == (B, N_TOK, N_TOK))

# =========================================================================
print("\n=== AttentionExtractor (non-transformer model) ===")
ae_none = AttentionExtractor()
check("AE no encoder returns None",  ae_none.get_genetics_attention(genetics) is None)
check("AE rollout returns None",     ae_none.get_genetics_rollout(genetics) is None)
check("AE gene imp returns None",    ae_none.get_gene_importance_from_attention(genetics) is None)
check("AE fusion returns None",      ae_none.get_fusion_attention(
                                         torch.randn(B, 8), torch.randn(B, 8)) is None)
check("AE head imp returns None",    ae_none.get_head_importance(genetics) is None)

# AttentionExtractor with a mock encoder that provides attention weights
class _MockTransformerEncoder(nn.Module):
    def __init__(self, n_genes=20, n_heads=4, n_layers=2):
        super().__init__()
        self.n_genes = n_genes
        self.n_heads = n_heads
        self.n_layers = n_layers

    def get_attention_weights(self, x):
        B = x.shape[0]
        N = self.n_genes + 1  # +1 for CLS
        # Return random softmax attention weights (simulating transformer output)
        return [
            torch.softmax(torch.randn(B, self.n_heads, N, N), dim=-1)
            for _ in range(self.n_layers)
        ]

mock_enc = _MockTransformerEncoder(N_GENES, n_heads=4, n_layers=2)
ae = AttentionExtractor(genetics_encoder=mock_enc)

attn_list = ae.get_genetics_attention(genetics)
check("AE transformer returns list",  isinstance(attn_list, list))
check("AE transformer list length",   len(attn_list) == 2,
      str(len(attn_list)))
check("AE attn shape",                tuple(attn_list[0].shape) == (B, 4, N_GENES+1, N_GENES+1),
      str(attn_list[0].shape))

gene_imp = ae.get_gene_importance_from_attention(genetics)
check("AE gene importance shape",     tuple(gene_imp.shape) == (B, N_GENES),
      str(gene_imp.shape))
check("AE gene importance sums to 1", (gene_imp.sum(dim=-1) - 1).abs().max() < 1e-4)

head_imp = ae.get_head_importance(genetics)
check("AE head importance shape",     tuple(head_imp.shape) == (4,),
      str(head_imp.shape))

# =========================================================================
print("\n=== GeneticsFeatureImportance ===")
from explainability.feature_importance import GeneticsFeatureImportance

# With plain genetics model
gfi = GeneticsFeatureImportance(gen_model)
gti = gfi.gradient_times_input(genetics, target=1)
check("GFI grad*input shape",        tuple(gti.shape) == (B, N_GENES),
      str(gti.shape))
check("GFI grad*input non-negative", gti.min() >= 0.0,
      f"min={gti.min():.6f}")
check("GFI grad*input finite",       gti.isfinite().all().item())

# TabNet backbone not present -> None
tabnet_imp = gfi.tabnet_importance(genetics)
check("GFI tabnet=None (no TabNet)", tabnet_imp is None)

# Attention backbone not present -> None
attn_imp = gfi.attention_importance(genetics)
check("GFI attn=None (no transformer)", attn_imp is None)

# Aggregate
agg = gfi.aggregate(genetics, target=1, methods=["gradient_times_input"])
check("GFI aggregate has gti",       "gradient_times_input" in agg)
check("GFI consensus not None",      agg["consensus"] is not None)
check("GFI consensus shape",         tuple(agg["consensus"].shape) == (N_GENES,),
      str(agg["consensus"].shape if agg["consensus"] is not None else None))
check("GFI consensus sums ~1",       abs(agg["consensus"].sum().item() - 1.0) < 1e-3,
      f"sum={agg['consensus'].sum().item():.4f}")

# With ASDModel
gfi_asd = GeneticsFeatureImportance(asd_model)
gti_asd = gfi_asd.gradient_times_input(genetics, mri=mri)
check("GFI ASDModel grad*input shape", tuple(gti_asd.shape) == (B, N_GENES))

# =========================================================================
print("\n=== ExplainabilityEngine ===")
from explainability.explainability_engine import ExplainabilityEngine

engine = ExplainabilityEngine(asd_model)

# --- MRI explanations ---
mri_result = engine.explain_mri(
    mri, genetics=genetics,
    methods=["gradcam", "gradcam_pp"],
    interpolate_to_input=False,
)
check("Engine MRI has gradcam",      "gradcam" in mri_result)
check("Engine MRI gradcam not None", mri_result["gradcam"] is not None)
check("Engine MRI gradcam shape",    tuple(mri_result["gradcam"].shape) == (B, D, H, W),
      str(mri_result.get("gradcam", {}) if not isinstance(mri_result.get("gradcam"), torch.Tensor) else mri_result["gradcam"].shape))
check("Engine MRI gradcam_pp shape", tuple(mri_result["gradcam_pp"].shape) == (B, D, H, W))

# IG through engine
mri_ig = engine.explain_mri(
    mri, genetics=genetics,
    methods=["integrated_gradients"],
    n_steps=10,
)
check("Engine MRI IG not None",      mri_ig.get("integrated_gradients") is not None)
check("Engine MRI IG shape",         tuple(mri_ig["integrated_gradients"].shape) == (B, C, D, H, W),
      str(mri_ig.get("integrated_gradients", None)))
check("Engine MRI IG delta key",     "ig_convergence_delta" in mri_ig)

# --- Genetics explanations ---
gen_result = engine.explain_genetics(
    genetics, mri=mri,
    methods=["gradient_times_input"],
)
check("Engine gen gti not None",     gen_result.get("gradient_times_input") is not None)
check("Engine gen gti shape",        tuple(gen_result["gradient_times_input"].shape) == (B, N_GENES))

gen_ig = engine.explain_genetics(
    genetics, mri=mri,
    methods=["integrated_gradients"],
    n_steps=10,
)
check("Engine gen IG not None",      gen_ig.get("integrated_gradients") is not None)
check("Engine gen IG shape",         tuple(gen_ig["integrated_gradients"].shape) == (B, N_GENES))

# --- Fusion explanations ---
fusion_result = engine.explain_fusion(mri, genetics)
check("Engine fusion has keys",      all(k in fusion_result for k in
      ["genetics_attention", "genetics_rollout", "gene_importance_attn", "fusion_attention"]))

# --- Full explain ---
full_result = engine.explain(
    mri, genetics,
    mri_methods=["gradcam"],
    gen_methods=["gradient_times_input"],
)
check("Engine explain has mri",      "mri" in full_result)
check("Engine explain has genetics", "genetics" in full_result)
check("Engine explain has fusion",   "fusion" in full_result)

# Model training state preserved
asd_model.train()
_ = engine.explain_mri(mri, genetics, methods=["gradcam"])
check("Engine preserves train mode", asd_model.training)

# =========================================================================
print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL EXPLAINABILITY TESTS PASSED")
else:
    sys.exit(1)
