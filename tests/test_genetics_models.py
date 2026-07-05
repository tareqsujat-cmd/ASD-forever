"""Genetics model tests — CPU only, small synthetic gene sets."""
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

B = 4           # batch size
N_GENES = 128   # genes after feature selection
FEAT_DIM = 64   # output feature dim (small for speed)

x = torch.randn(B, N_GENES)

# -------------------------------------------------------------------------
print("=== GeneTransformerEncoder ===")
from models.genetics.transformer_encoder import GeneTransformerEncoder

trans = GeneTransformerEncoder(
    n_genes=N_GENES, d_model=64, n_heads=4, n_layers=2,
    dim_feedforward=128, feature_dim=FEAT_DIM, dropout=0.1,
)
trans.eval()
with torch.no_grad():
    out = trans(x)
check("Transformer output shape", out.shape == (B, FEAT_DIM), str(out.shape))
check("Transformer output finite", out.isfinite().all().item())

# Attention weights
attn_maps = trans.get_attention_weights(x)
check("Attention maps returned", len(attn_maps) == 2)
check("Attention maps shape dim", attn_maps[0].dim() == 4,
      str(attn_maps[0].shape))  # (B, n_heads, N+1, N+1)
check("Attention maps non-None", all(m is not None for m in attn_maps))

# Gradient flow
trans.train()
out = trans(x)
loss = out.sum()
loss.backward()
grad_ok = all(
    p.grad is not None for p in trans.parameters() if p.requires_grad
)
check("Transformer gradients flow", grad_ok)

# -------------------------------------------------------------------------
print("\n=== TabNetEncoder ===")
from models.genetics.tabnet import TabNetEncoder

tab = TabNetEncoder(
    n_genes=N_GENES, n_d=32, n_a=32, n_steps=3, gamma=1.5,
    feature_dim=FEAT_DIM, dropout=0.0,
)
tab.eval()
with torch.no_grad():
    feats, sparse_loss = tab(x)
check("TabNet features shape", feats.shape == (B, FEAT_DIM), str(feats.shape))
check("TabNet sparsity_loss finite", sparse_loss.isfinite().item())
check("TabNet sparsity_loss >= 0", sparse_loss.item() >= 0)

importances = tab.get_feature_importances()
check("TabNet importances shape", importances.shape == (B, N_GENES), str(importances.shape))
check("TabNet importances in [0,1]",
      (importances >= 0).all() and (importances <= 1.0001).all())
check("TabNet importances sum <= 1 (sparse)",
      (importances.sum(dim=-1) <= 1.0 + 1e-4).all().item())

# Gradient through features
tab.train()
feats, sp = tab(x)
(feats.sum() + sp).backward()
check("TabNet gradients flow",
      any(p.grad is not None for p in tab.parameters() if p.requires_grad))

# -------------------------------------------------------------------------
print("\n=== Sparsemax ===")
from models.genetics.tabnet import _sparsemax
z = torch.randn(4, 128)
sm = _sparsemax(z, dim=-1)
check("Sparsemax non-negative", (sm >= 0).all().item())
check("Sparsemax sums to 1", torch.allclose(sm.sum(-1), torch.ones(4), atol=1e-5))
check("Sparsemax sparse (some zeros)", (sm == 0).any().item())

# -------------------------------------------------------------------------
print("\n=== GATLayer ===")
from models.genetics.gnn_encoder import GATLayer
N = 32
gat = GATLayer(in_features=16, out_features=8, n_heads=2, concat=True)
adj = (torch.rand(N, N) > 0.5).float()
adj.fill_diagonal_(1.0)  # self-loops
h = torch.randn(B, N, 16)
out_gat = gat(h, adj)
check("GATLayer concat output shape", out_gat.shape == (B, N, 16), str(out_gat.shape))

gat_mean = GATLayer(in_features=16, out_features=8, n_heads=2, concat=False)
out_mean = gat_mean(h, adj)
check("GATLayer mean output shape", out_mean.shape == (B, N, 8), str(out_mean.shape))

# Unbatched input
h_single = torch.randn(N, 16)
out_single = gat(h_single, adj)
check("GATLayer unbatched shape", out_single.shape == (N, 16), str(out_single.shape))

# -------------------------------------------------------------------------
print("\n=== GNNEncoder ===")
from models.genetics.gnn_encoder import GNNEncoder
adj_full = (torch.rand(N_GENES, N_GENES) > 0.7).float()
adj_full.fill_diagonal_(1.0)

gnn = GNNEncoder(
    n_genes=N_GENES, adj=adj_full,
    emb_dim=16, gat_hidden=16, gat_out=16, n_heads=2,
    feature_dim=FEAT_DIM, dropout=0.1,
)
gnn.eval()
with torch.no_grad():
    gnn_feats, gnn_aux = gnn(x)
check("GNN features shape", gnn_feats.shape == (B, FEAT_DIM), str(gnn_feats.shape))
check("GNN aux_loss zero", gnn_aux.item() == 0.0)
check("GNN features finite", gnn_feats.isfinite().all().item())

# GNN gradient
gnn.train()
feats, _ = gnn(x)
feats.sum().backward()
check("GNN gradients flow",
      any(p.grad is not None for p in gnn.parameters() if p.requires_grad))

# -------------------------------------------------------------------------
print("\n=== GeneticsEncoder wrapper ===")
from models.genetics.genetics_encoder import GeneticsEncoder, GeneticsClassifier

# Transformer backend
enc_trans = GeneticsEncoder(
    backbone=GeneTransformerEncoder(N_GENES, 64, 4, 2, 128, FEAT_DIM, 0.1),
    feature_dim=FEAT_DIM, dropout=0.1,
)
enc_trans.eval()
with torch.no_grad():
    enc_out = enc_trans(x)
check("GeneticsEncoder[Transformer] shape", enc_out.shape == (B, FEAT_DIM))
check("GeneticsEncoder aux_loss attr exists", hasattr(enc_trans, 'last_aux_loss'))

# TabNet backend
enc_tab = GeneticsEncoder(
    backbone=TabNetEncoder(N_GENES, 32, 32, 3, 1.5, 2, 1e-15, 8, FEAT_DIM, 0.0),
    feature_dim=FEAT_DIM, dropout=0.1,
)
enc_tab.eval()
with torch.no_grad():
    enc_out_tab = enc_tab(x)
check("GeneticsEncoder[TabNet] shape", enc_out_tab.shape == (B, FEAT_DIM))
check("GeneticsEncoder[TabNet] aux_loss >= 0",
      enc_tab.last_aux_loss.item() >= 0)

# GNN backend
enc_gnn = GeneticsEncoder(
    backbone=GNNEncoder(N_GENES, adj_full, 16, 16, 16, 2, FEAT_DIM, 0.1),
    feature_dim=FEAT_DIM, dropout=0.1,
)
enc_gnn.eval()
with torch.no_grad():
    enc_out_gnn = enc_gnn(x)
check("GeneticsEncoder[GNN] shape", enc_out_gnn.shape == (B, FEAT_DIM))

# GeneticsClassifier
clf = GeneticsClassifier(enc_trans, num_classes=2, dropout=0.1)
clf.eval()
with torch.no_grad():
    result = clf(x)
check("GeneticsClassifier logits shape", result["logits"].shape == (B, 2))
check("GeneticsClassifier features shape", result["features"].shape == (B, FEAT_DIM))

# -------------------------------------------------------------------------
print("\n=== get_feature_maps explainability hook ===")
enc_trans.train()  # enable dropout
attn = enc_trans.get_feature_maps(x)
check("Transformer get_feature_maps is list", isinstance(attn, list))
check("Transformer get_feature_maps n_layers", len(attn) == 2)

enc_tab.train()
enc_tab(x)  # populate mask
imp = enc_tab.backbone.get_feature_importances()
check("TabNet get_feature_importances shape",
      imp is not None and imp.shape == (B, N_GENES))

# -------------------------------------------------------------------------
print("\n=== MLP ablation encoder ===")
from models.genetics.genetics_encoder import _MLPEncoder
mlp = _MLPEncoder(N_GENES, [128, 64], FEAT_DIM, dropout=0.1)
mlp.eval()
with torch.no_grad():
    mlp_out = mlp(x)
check("MLP output shape", mlp_out.shape == (B, FEAT_DIM), str(mlp_out.shape))

# -------------------------------------------------------------------------
print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL GENETICS MODEL TESTS PASSED")
else:
    sys.exit(1)
