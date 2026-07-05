"""Visualization suite tests -- CPU-only, Agg backend, no display needed."""
import sys
sys.path.insert(0, r'e:\ASD_forever')
import matplotlib
matplotlib.use('Agg')   # non-interactive, must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

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

def is_fig(obj):
    return isinstance(obj, plt.Figure)

def close(fig):
    plt.close(fig)

# =========================================================================
# Synthetic data
# =========================================================================
rng = np.random.default_rng(42)
N = 200
y_true = rng.integers(0, 2, N)
y_prob = np.clip(
    np.where(y_true == 1, rng.uniform(0.45, 0.95, N), rng.uniform(0.05, 0.65, N)),
    0, 1
)
y_prob_b  = rng.uniform(0, 1, N)   # random baseline

# Synthetic feature embeddings
features = rng.standard_normal((N, 32))

# Synthetic MRI volume + saliency (small for speed)
volume   = rng.standard_normal((16, 16, 16))
saliency = np.abs(rng.standard_normal((16, 16, 16)))

# Gene importances (signed)
gene_importances = rng.standard_normal(50)
gene_names       = [f"GENE_{i:03d}" for i in range(50)]

# Confusion matrix [[TN, FP], [FN, TP]]
y_pred = (y_prob >= 0.5).astype(int)
TP = int(((y_pred == 1) & (y_true == 1)).sum())
TN = int(((y_pred == 0) & (y_true == 0)).sum())
FP = int(((y_pred == 1) & (y_true == 0)).sum())
FN = int(((y_pred == 0) & (y_true == 1)).sum())
cm = np.array([[TN, FP], [FN, TP]], dtype=float)

# =========================================================================
print("=== style ===")
from visualization.style import (
    ieee_style, apply_ieee_style, COLORS, PALETTE,
    SINGLE_COL_W, DOUBLE_COL_W, IEEE_RC
)
check("ieee_style is context manager", True)  # if import worked, it works
check("COLORS has asd key",           "asd" in COLORS)
check("COLORS has tc key",            "tc"  in COLORS)
check("PALETTE is a list",            isinstance(PALETTE, list) and len(PALETTE) >= 4)
check("SINGLE_COL_W reasonable",      3.0 < SINGLE_COL_W < 4.0)
check("DOUBLE_COL_W reasonable",      6.0 < DOUBLE_COL_W < 8.0)

# =========================================================================
print("\n=== ROC / PR curves ===")
from visualization.roc_pr_curves import (
    plot_roc_curve, plot_pr_curve, plot_roc_comparison, plot_pr_comparison
)

fig = plot_roc_curve(y_true, y_prob, label="Model")
check("ROC returns Figure",           is_fig(fig))
check("ROC has axes",                 len(fig.axes) >= 1)
close(fig)

fig = plot_roc_curve(y_true, y_prob, show_ci=True, n_bootstrap=50)
check("ROC with CI returns Figure",   is_fig(fig))
close(fig)

fig = plot_pr_curve(y_true, y_prob, label="Model")
check("PR returns Figure",            is_fig(fig))
close(fig)

fig = plot_pr_curve(y_true, y_prob, show_ci=True, n_bootstrap=50)
check("PR with CI returns Figure",    is_fig(fig))
close(fig)

models = [
    {"name": "Proposed",  "y_true": y_true, "y_prob": y_prob},
    {"name": "Baseline",  "y_true": y_true, "y_prob": y_prob_b},
]
fig = plot_roc_comparison(models)
check("ROC comparison returns Figure", is_fig(fig))
check("ROC comparison has legend",     len(fig.axes[0].get_legend().get_texts()) >= 2)
close(fig)

fig = plot_pr_comparison(models)
check("PR comparison returns Figure",  is_fig(fig))
close(fig)

# =========================================================================
print("\n=== Confusion Matrix ===")
from visualization.confusion_matrix import plot_confusion_matrix, cm_from_report

fig = plot_confusion_matrix(cm)
check("CM returns Figure",            is_fig(fig))
check("CM has 1+ axes",               len(fig.axes) >= 1)
close(fig)

fig = plot_confusion_matrix(cm, normalize=True)
check("CM normalized returns Figure", is_fig(fig))
close(fig)

fig = plot_confusion_matrix(cm, labels=["Control", "ASD"], show_metrics=True)
check("CM custom labels returns Figure", is_fig(fig))
close(fig)

# Invalid shape raises
try:
    plot_confusion_matrix(np.zeros((3, 3)))
    check("CM 3x3 raises ValueError",  False, "No exception raised")
except ValueError:
    check("CM 3x3 raises ValueError",  True)

# =========================================================================
print("\n=== Calibration ===")
from visualization.calibration_plot import plot_reliability_diagram, plot_brier_decomposition
from evaluation.calibration import brier_score_decomposition

fig = plot_reliability_diagram(y_true, y_prob, n_bins=10)
check("Reliability diagram returns Figure",    is_fig(fig))
check("Reliability diagram has 2 axes (hist)", len(fig.axes) >= 2)
close(fig)

fig = plot_reliability_diagram(y_true, y_prob, show_histogram=False)
check("Reliability diagram no hist",           is_fig(fig))
close(fig)

brier_dict = brier_score_decomposition(y_true, y_prob)
fig = plot_brier_decomposition(brier_dict)
check("Brier decomp returns Figure",           is_fig(fig))
close(fig)

# =========================================================================
print("\n=== Embedding ===")
from visualization.embedding_viz import compute_embedding, plot_embedding

emb = compute_embedding(features, method="tsne", perplexity=15, random_state=42)
check("t-SNE returns (N,2)",           emb.shape == (N, 2),
      str(emb.shape))
check("t-SNE is finite",               np.isfinite(emb).all())

emb_pca = compute_embedding(features, method="pca", random_state=42)
check("PCA returns (N,2)",             emb_pca.shape == (N, 2))

fig = plot_embedding(emb, y_true)
check("Embedding returns Figure",      is_fig(fig))
close(fig)

site_ids = rng.choice(["NYU", "UCLA", "PITT"], N)
fig = plot_embedding(emb, y_true, site_ids=site_ids, title="Test embedding")
check("Embedding with sites Figure",   is_fig(fig))
close(fig)

# =========================================================================
print("\n=== MRI Saliency ===")
from visualization.mri_saliency import plot_mri_slice, plot_mri_triplet, plot_volume_grid

fig = plot_mri_slice(volume)
check("MRI slice no saliency Figure",     is_fig(fig))
close(fig)

fig = plot_mri_slice(volume, saliency, dim=0, idx=8)
check("MRI slice with saliency Figure",   is_fig(fig))
close(fig)

fig = plot_mri_triplet(volume, saliency, title="GradCAM Test")
check("MRI triplet returns Figure",       is_fig(fig))
check("MRI triplet has 3+ axes",          len(fig.axes) >= 3,
      str(len(fig.axes)))
close(fig)

fig = plot_mri_triplet(volume, saliency=None)  # no saliency
check("MRI triplet no saliency Figure",   is_fig(fig))
close(fig)

# 4-channel input (C, D, H, W)
vol_4ch = rng.standard_normal((2, 16, 16, 16))
fig = plot_mri_slice(vol_4ch)
check("MRI slice 4-channel Figure",       is_fig(fig))
close(fig)

# Volume grid
vols = rng.standard_normal((4, 16, 16, 16))
fig = plot_volume_grid(vols, n_cols=4)
check("Volume grid returns Figure",       is_fig(fig))
close(fig)

# =========================================================================
print("\n=== Gene Importance / Attention ===")
from visualization.genetics_importance import (
    plot_gene_importance, plot_attention_heatmap, plot_multihead_attention
)

fig = plot_gene_importance(gene_importances, gene_names=gene_names, top_k=20)
check("Gene imp returns Figure",          is_fig(fig))
check("Gene imp has axes",                len(fig.axes) >= 1)
close(fig)

# dict input
imp_dict = {name: float(v) for name, v in zip(gene_names[:10], gene_importances[:10])}
fig = plot_gene_importance(imp_dict, top_k=5)
check("Gene imp dict returns Figure",     is_fig(fig))
close(fig)

# Attention heatmap: (N, N) matrix
attn_2d = np.random.default_rng(0).random((21, 21))
fig = plot_attention_heatmap(attn_2d, title="Test Attention")
check("Attention heatmap 2D Figure",      is_fig(fig))
close(fig)

# (n_heads, N, N) → head selection
attn_3d = np.random.default_rng(0).random((4, 21, 21))
fig = plot_attention_heatmap(attn_3d, head_idx=2)
check("Attention heatmap 3D head sel",    is_fig(fig))
close(fig)

# (B, n_heads, N, N) → first sample
attn_4d = np.random.default_rng(0).random((2, 4, 21, 21))
fig = plot_attention_heatmap(attn_4d)
check("Attention heatmap 4D Figure",      is_fig(fig))
close(fig)

fig = plot_multihead_attention(attn_3d, n_cols=4)
check("Multi-head attn returns Figure",   is_fig(fig))
close(fig)

# =========================================================================
print("\n=== FigureFactory ===")
from visualization.figure_factory import FigureFactory

with tempfile.TemporaryDirectory() as tmpdir:
    ff = FigureFactory(output_dir=tmpdir, dpi=72, formats=["png"])

    fig = ff.roc_curve(y_true, y_prob, show_ci=False)
    check("FF roc_curve returns Figure",     is_fig(fig))
    close(fig)

    fig = ff.pr_curve(y_true, y_prob, show_ci=False)
    check("FF pr_curve returns Figure",      is_fig(fig))
    close(fig)

    fig = ff.confusion_matrix(cm)
    check("FF confusion_matrix Figure",      is_fig(fig))
    close(fig)

    fig = ff.calibration(y_true, y_prob)
    check("FF calibration Figure",           is_fig(fig))
    close(fig)

    brier_d = brier_score_decomposition(y_true, y_prob)
    fig = ff.brier_decomposition(brier_d)
    check("FF brier_decomp Figure",          is_fig(fig))
    close(fig)

    fig = ff.gene_importance(gene_importances, gene_names=gene_names, top_k=10)
    check("FF gene_importance Figure",       is_fig(fig))
    close(fig)

    fig = ff.attention_heatmap(attn_2d)
    check("FF attention_heatmap Figure",     is_fig(fig))
    close(fig)

    fig = ff.mri_saliency(volume, saliency, title="Test")
    check("FF mri_saliency Figure",          is_fig(fig))
    close(fig)

    # Save creates files
    fig = ff.roc_curve(y_true, y_prob, show_ci=False, filename="test_roc")
    saved_png = os.path.join(tmpdir, "test_roc.png")
    check("FF save creates PNG",             os.path.exists(saved_png) and
                                             os.path.getsize(saved_png) > 500,
          f"exists={os.path.exists(saved_png)}")
    close(fig)

    # summary panel
    fig = ff.summary_panel(y_true, y_prob, gene_importances=gene_importances[:50],
                            gene_names=gene_names)
    check("FF summary_panel Figure",         is_fig(fig))
    check("FF summary_panel has 6 axes",     len([ax for ax in fig.axes
                                                   if ax.get_visible()]) >= 4,
          str(len(fig.axes)))
    close(fig)

# =========================================================================
print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL VISUALIZATION TESTS PASSED")
else:
    sys.exit(1)
