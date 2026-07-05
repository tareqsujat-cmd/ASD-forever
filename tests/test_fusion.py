"""Fusion module tests — CPU only."""
import sys
sys.path.insert(0, r'e:\ASD_forever')
import torch
import torch.nn as nn

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

B = 4
MRI_DIM = 128
GEN_DIM = 64
FUSION_DIM = 128
NUM_CLASSES = 2
N_HEADS = 4

mri = torch.randn(B, MRI_DIM)
gen = torch.randn(B, GEN_DIM)

# -------------------------------------------------------------------------
print("=== CrossAttentionFusion ===")
from models.fusion.cross_attention import CrossAttentionFusion

ca = CrossAttentionFusion(
    mri_dim=MRI_DIM, gen_dim=GEN_DIM, fusion_dim=FUSION_DIM,
    n_heads=N_HEADS, n_layers=2, n_tokens=4, ffn_dim=256,
    num_classes=NUM_CLASSES, dropout=0.1,
)
ca.eval()
with torch.no_grad():
    out = ca(mri, gen)
check("CA logits shape",   out["logits"].shape == (B, NUM_CLASSES), str(out["logits"].shape))
check("CA fused shape",    out["fused_features"].shape == (B, FUSION_DIM), str(out["fused_features"].shape))
check("CA logits finite",  out["logits"].isfinite().all().item())
check("CA fused finite",   out["fused_features"].isfinite().all().item())

# Attention weights extraction
attn = ca.get_attention_weights(mri, gen)
check("CA attn keys present", "mri_to_gen" in attn and "gen_to_mri" in attn)
check("CA attn n_layers",  len(attn["mri_to_gen"]) == 2)
mri_w = attn["mri_to_gen"][0]
check("CA attn weight shape", mri_w.shape[0] == B, str(mri_w.shape))
# shape: (B, n_heads, n_tokens, n_tokens) but n_tokens in Q and KV could differ
check("CA attn weight 4D",    mri_w.dim() == 4, str(mri_w.shape))

# Gradient flow
ca.train()
out = ca(mri, gen)
out["logits"].sum().backward()
check("CA gradients flow",
      any(p.grad is not None for p in ca.parameters() if p.requires_grad))

# n_tokens=1 edge case (degenerate but valid)
ca1 = CrossAttentionFusion(MRI_DIM, GEN_DIM, FUSION_DIM, N_HEADS, 1, 1, 128, 2, 0.1)
with torch.no_grad():
    out1 = ca1(mri, gen)
check("CA n_tokens=1 valid", out1["logits"].shape == (B, 2))

# -------------------------------------------------------------------------
print("\n=== IntermediateFusion ===")
from models.fusion.gated_fusion import IntermediateFusion

inter = IntermediateFusion(MRI_DIM, GEN_DIM, FUSION_DIM, NUM_CLASSES, 0.1)
inter.eval()
with torch.no_grad():
    out = inter(mri, gen)
check("Inter logits shape",  out["logits"].shape == (B, NUM_CLASSES))
check("Inter fused shape",   out["fused_features"].shape == (B, FUSION_DIM))
check("Inter logits finite", out["logits"].isfinite().all().item())

# -------------------------------------------------------------------------
print("\n=== GatedFusion (sigmoid) ===")
from models.fusion.gated_fusion import GatedFusion

gate_sig = GatedFusion(MRI_DIM, GEN_DIM, FUSION_DIM, NUM_CLASSES, 0.1, "sigmoid")
gate_sig.eval()
with torch.no_grad():
    out = gate_sig(mri, gen)
check("Gate[sig] logits shape",   out["logits"].shape == (B, NUM_CLASSES))
check("Gate[sig] gate_weights",   out["gate_weights"].shape == (B, 2))
check("Gate[sig] gates in [0,1]", (out["gate_weights"] >= 0).all() and
                                   (out["gate_weights"] <= 1).all())

# -------------------------------------------------------------------------
print("\n=== GatedFusion (softmax) ===")
gate_soft = GatedFusion(MRI_DIM, GEN_DIM, FUSION_DIM, NUM_CLASSES, 0.1, "softmax")
gate_soft.eval()
with torch.no_grad():
    out = gate_soft(mri, gen)
check("Gate[soft] gates sum=1",
      torch.allclose(out["gate_weights"].sum(-1), torch.ones(B), atol=1e-5))

# -------------------------------------------------------------------------
print("\n=== LateFusion ===")
from models.fusion.gated_fusion import LateFusion

late = LateFusion(MRI_DIM, GEN_DIM, FUSION_DIM, NUM_CLASSES, 0.1)
late.eval()
with torch.no_grad():
    out = late(mri, gen)
check("Late logits shape",    out["logits"].shape == (B, NUM_CLASSES))
check("Late ensemble_weights sum=1",
      torch.allclose(out["ensemble_weights"].sum(), torch.tensor(1.0), atol=1e-5))
check("Late mri_logits present", "mri_logits" in out)
check("Late gen_logits present", "gen_logits" in out)

# -------------------------------------------------------------------------
print("\n=== DynamicFusion ===")
from models.fusion.gated_fusion import DynamicFusion

dyn = DynamicFusion(MRI_DIM, GEN_DIM, FUSION_DIM, NUM_CLASSES, 0.1)
dyn.eval()
with torch.no_grad():
    out = dyn(mri, gen)
check("Dyn logits shape",    out["logits"].shape == (B, NUM_CLASSES))
check("Dyn fused shape",     out["fused_features"].shape == (B, FUSION_DIM))
check("Dyn weights sum=1",
      torch.allclose(out["modality_weights"].sum(-1), torch.ones(B), atol=1e-5))

# -------------------------------------------------------------------------
print("\n=== MultiModalFusion wrapper ===")
from models.fusion.fusion_module import MultiModalFusion

fusion_wrapper = MultiModalFusion(
    backend=CrossAttentionFusion(MRI_DIM, GEN_DIM, FUSION_DIM, N_HEADS, 1, 4, 256, 2, 0.1),
    fusion_dim=FUSION_DIM, num_classes=NUM_CLASSES, method="cross_attention",
)
fusion_wrapper.eval()
with torch.no_grad():
    out = fusion_wrapper(mri, gen)
check("Wrapper logits shape",  out["logits"].shape == (B, NUM_CLASSES))
check("Wrapper fused shape",   out["fused_features"].shape == (B, FUSION_DIM))

# Attention weights via wrapper
attn = fusion_wrapper.get_attention_weights(mri, gen)
check("Wrapper attn not None", attn is not None)

# Non-attention backend returns None
late_wrapper = MultiModalFusion(
    backend=LateFusion(MRI_DIM, GEN_DIM, FUSION_DIM, 2, 0.1),
    fusion_dim=4, num_classes=2, method="late",
)
check("Late wrapper attn=None", late_wrapper.get_attention_weights(mri, gen) is None)

# -------------------------------------------------------------------------
print("\n=== End-to-end gradient flow (all fusion types) ===")
for name, module in [
    ("CrossAttention", CrossAttentionFusion(MRI_DIM, GEN_DIM, FUSION_DIM, N_HEADS, 1, 4, 256, 2, 0.1)),
    ("Intermediate",   IntermediateFusion(MRI_DIM, GEN_DIM, FUSION_DIM, 2, 0.1)),
    ("Gated",          GatedFusion(MRI_DIM, GEN_DIM, FUSION_DIM, 2, 0.1, "sigmoid")),
    ("Late",           LateFusion(MRI_DIM, GEN_DIM, FUSION_DIM, 2, 0.1)),
    ("Dynamic",        DynamicFusion(MRI_DIM, GEN_DIM, FUSION_DIM, 2, 0.1)),
]:
    module.train()
    out = module(mri.detach().requires_grad_(True), gen.detach().requires_grad_(True))
    out["logits"].sum().backward()
    has_grad = any(p.grad is not None for p in module.parameters() if p.requires_grad)
    check(f"{name} backward pass", has_grad)

# -------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL FUSION TESTS PASSED")
else:
    sys.exit(1)
