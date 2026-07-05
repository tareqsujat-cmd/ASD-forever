"""Tests for Module 12: Paper Figure Generation."""
import sys
sys.path.insert(0, r'e:\ASD_forever')
import matplotlib
matplotlib.use('Agg')
import os, tempfile, math
import numpy as np
import matplotlib.pyplot as plt

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
# Synthetic test data
# =========================================================================
RNG = np.random.default_rng(42)
N   = 120

y_true = np.array([0]*60 + [1]*60)
y_prob = np.clip(
    y_true * 0.5 + RNG.normal(0, 0.2, N) + 0.25,
    0.01, 0.99,
)

# MRI volume + saliency (32x32x32)
mri_vol  = RNG.standard_normal((32, 32, 32)).astype(np.float32)
saliency = np.abs(RNG.standard_normal((32, 32, 32))).astype(np.float32)
saliency /= saliency.max()

# Genetics
n_genes       = 50
gene_names    = [f"SNP_{i:04d}" for i in range(n_genes)]
gene_imp      = RNG.standard_normal(n_genes).astype(np.float32)

# Features for embedding
features = RNG.standard_normal((N, 32)).astype(np.float32)
labels   = y_true.copy()
site_ids = (RNG.integers(0, 5, N)).astype(int)

# Site / class counts
site_counts  = {f"Site_{i}": int(RNG.integers(15, 60)) for i in range(8)}
class_counts = {"ASD": 60, "TC": 60}

# CV results
cv_results = []
for fold in range(5):
    for model in ["Proposed", "MRI-only", "Genetics-only"]:
        base = {"Proposed": 0.82, "MRI-only": 0.74, "Genetics-only": 0.70}[model]
        cv_results.append({
            "model":   model,
            "fold":    fold,
            "val_auc": float(np.clip(base + RNG.normal(0, 0.02), 0.5, 1.0)),
            "val_acc": float(np.clip(base - 0.06 + RNG.normal(0, 0.02), 0.5, 1.0)),
            "val_f1":  float(np.clip(base - 0.03 + RNG.normal(0, 0.02), 0.5, 1.0)),
        })

# Models list for ROC/PR
models_list = [
    {"name": "Proposed", "y_true": y_true,
     "y_prob": y_prob, "color": "#e41a1c"},
    {"name": "MRI-only",
     "y_true": y_true,
     "y_prob": np.clip(y_true * 0.4 + RNG.normal(0, 0.25, N) + 0.3, 0.01, 0.99),
     "color": "#377eb8"},
]

# Brier decomposition (values must satisfy cal ≥ 0, res ≥ 0, unc ≥ 0)
brier_dict = {
    "brier_score": 0.12,
    "calibration": 0.02,
    "resolution":  0.15,
    "uncertainty": 0.25,
}

# Ablation results
from ablation.ablation_results import AblationResults
ablation_res = AblationResults("test_ablation")
for name, auc_base in [("baseline", 0.80), ("fusion=gated", 0.75),
                        ("backbone=swin3d", 0.84), ("fusion=late", 0.70)]:
    auc_vals = list(np.clip(auc_base + RNG.normal(0, 0.01, 5), 0.5, 1.0))
    ablation_res.add(name, [{"val_auc": float(a), "val_acc": float(a-0.05)}
                              for a in auc_vals])

# Minimal HPO tuner (already optimized)
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
from hyperparameter_tuning.optuna_tuner import ASDTuner

def _tiny_train(cfg, name):
    return [{"val_auc": float(np.clip(0.80 + RNG.normal(0, 0.03), 0.5, 1.0))}]

tuner = ASDTuner(
    train_fn=_tiny_train, base_config={},
    search_space="quick", n_trials=8, pruner="none", seed=1,
)
tuner.optimize()

# =========================================================================
print("=== FigureSpec ===")
from paper.figure_specs import FigureSpec, FIGURE_SPECS, get_spec

check("FIGURE_SPECS has 9 entries", len(FIGURE_SPECS) == 9, str(len(FIGURE_SPECS)))
check("get_spec by filename", get_spec("fig01_architecture").label == "fig:architecture")
check("get_spec by label",    get_spec("fig:roc_pr").filename == "fig03_roc_pr")

spec_roc = get_spec("fig03_roc_pr")
check("Double-col width correct",
      abs(spec_roc.width_inches - 7.166) < 0.01,
      str(spec_roc.width_inches))
spec_single = get_spec("fig04_ablation")
check("Single-col width correct",
      abs(spec_single.width_inches - 3.487) < 0.01,
      str(spec_single.width_inches))
check("panel_labels populated",
      len(get_spec("fig02_dataset_statistics").panel_labels) == 2)

latex_env = spec_roc.latex_figure_env()
check("latex_figure_env has figure*",   "figure*" in latex_env)
check("latex_figure_env has caption",   "caption" in latex_env)
check("latex_figure_env has label",     "fig:roc_pr" in latex_env)
check("latex_figure_env has textwidth", r"\textwidth" in latex_env)

try:
    get_spec("nonexistent_key")
    check("get_spec bad key raises KeyError", False, "no exception")
except KeyError:
    check("get_spec bad key raises KeyError", True)

# =========================================================================
print("\n=== PaperFigureGenerator construction ===")
from paper.paper_figures import PaperFigureGenerator

with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    check("Generator output_dir set",   str(gen.output_dir) == tmpdir)
    check("Generator formats set",      gen.formats == ["png"])
    check("Generator dpi set",          gen.dpi == 72)
    check("Manifest empty at start",    len(gen._manifest) == 0)

# =========================================================================
print("\n=== fig_architecture_overview ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    fig = gen.fig_architecture_overview()
    check("Returns Figure",        isinstance(fig, plt.Figure))
    check("PNG file created",      os.path.exists(
          os.path.join(tmpdir, "fig01_architecture.png")))
    check("Manifest has 1 entry",  len(gen._manifest) == 1)
    plt.close(fig)

# =========================================================================
print("\n=== fig_dataset_statistics ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    fig = gen.fig_dataset_statistics(site_counts, class_counts)
    check("Returns Figure",   isinstance(fig, plt.Figure))
    check("PNG file created", os.path.exists(
          os.path.join(tmpdir, "fig02_dataset_statistics.png")))
    check("Has 2 axes",       len(fig.axes) == 2, str(len(fig.axes)))
    plt.close(fig)

# =========================================================================
print("\n=== fig_roc_pr ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    fig = gen.fig_roc_pr(models_list)
    check("Returns Figure",    isinstance(fig, plt.Figure))
    check("PNG file created",  os.path.exists(
          os.path.join(tmpdir, "fig03_roc_pr.png")))
    check("Has 2 axes",        len(fig.axes) >= 2, str(len(fig.axes)))
    plt.close(fig)

# =========================================================================
print("\n=== fig_ablation ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    fig = gen.fig_ablation(ablation_res)
    check("Returns Figure",   isinstance(fig, plt.Figure))
    check("PNG file created", os.path.exists(
          os.path.join(tmpdir, "fig04_ablation.png")))
    plt.close(fig)

# =========================================================================
print("\n=== fig_explainability ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    fig = gen.fig_explainability(mri_vol, saliency, gene_imp, gene_names, top_k=15)
    check("Returns Figure",   isinstance(fig, plt.Figure))
    check("PNG file created", os.path.exists(
          os.path.join(tmpdir, "fig05_explainability.png")))
    plt.close(fig)

# =========================================================================
print("\n=== fig_cv_summary ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    fig = gen.fig_cv_summary(cv_results, metrics=["val_auc", "val_acc"])
    check("Returns Figure",   isinstance(fig, plt.Figure))
    check("PNG file created", os.path.exists(
          os.path.join(tmpdir, "fig06_cv_summary.png")))
    check("Has 2 metric axes", sum(1 for ax in fig.axes
                                    if ax.get_title()) >= 2)
    plt.close(fig)

# =========================================================================
print("\n=== fig_calibration ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    # With Brier decomposition
    fig = gen.fig_calibration(y_true, y_prob, brier_dict=brier_dict)
    check("Returns Figure (with brier)",  isinstance(fig, plt.Figure))
    check("PNG file created",             os.path.exists(
          os.path.join(tmpdir, "fig07_calibration.png")))
    plt.close(fig)

with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    # Without Brier decomposition
    fig = gen.fig_calibration(y_true, y_prob, brier_dict=None)
    check("Returns Figure (no brier)",    isinstance(fig, plt.Figure))
    plt.close(fig)

# =========================================================================
print("\n=== fig_embedding ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    fig = gen.fig_embedding(features, labels, site_ids=site_ids, method="pca")
    check("Returns Figure",   isinstance(fig, plt.Figure))
    check("PNG file created", os.path.exists(
          os.path.join(tmpdir, "fig08_embedding.png")))
    plt.close(fig)

# =========================================================================
print("\n=== fig_hpo_summary ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    fig = gen.fig_hpo_summary(tuner)
    check("Returns Figure",   isinstance(fig, plt.Figure))
    check("PNG file created", os.path.exists(
          os.path.join(tmpdir, "figS1_hpo.png")))
    check("Has 2 axes",       len(fig.axes) >= 2, str(len(fig.axes)))
    plt.close(fig)

# =========================================================================
print("\n=== generate_all ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png"], dpi=72)
    bundle = {
        "site_counts":     site_counts,
        "class_counts":    class_counts,
        "models":          models_list,
        "ablation_results":ablation_res,
        "cv_results":      cv_results,
        "mri_volume":      mri_vol,
        "saliency":        saliency,
        "gene_importances":gene_imp,
        "gene_names":      gene_names,
        "features":        features,
        "labels":          labels,
        "site_ids":        site_ids,
        "y_true":          y_true,
        "y_prob":          y_prob,
        "brier_dict":      brier_dict,
        "tuner":           tuner,
    }
    results = gen.generate_all(bundle)
    plt.close("all")
    check("generate_all returns dict",      isinstance(results, dict))
    check("All 9 figures produced",         len(results) == 9,
          f"got {len(results)}: {list(results.keys())}")
    check("Each entry has paths list",      all(isinstance(v, list)
                                                  for v in results.values()))

    # Verify files exist
    expected_stems = [
        "fig01_architecture", "fig02_dataset_statistics",
        "fig03_roc_pr", "fig04_ablation", "fig05_explainability",
        "fig06_cv_summary", "fig07_calibration", "fig08_embedding",
        "figS1_hpo",
    ]
    for stem in expected_stems:
        path = os.path.join(tmpdir, f"{stem}.png")
        check(f"File exists: {stem}", os.path.exists(path))

# =========================================================================
print("\n=== manifest ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["png", "pdf"], dpi=72)
    gen.fig_architecture_overview()
    gen.fig_dataset_statistics(site_counts, class_counts)
    plt.close("all")

    manifest_str = gen.latex_manifest()
    check("Manifest non-empty",              len(manifest_str) > 100)
    check("Manifest has figure env",         "begin{figure" in manifest_str)
    check("Manifest has architecture label", "fig:architecture" in manifest_str)

    manifest_list = gen.manifest_dict()
    check("manifest_dict returns list",      isinstance(manifest_list, list))
    check("manifest_dict has 2 entries",     len(manifest_list) == 2,
          str(len(manifest_list)))
    check("manifest_dict has label key",     "label" in manifest_list[0])
    check("manifest_dict has paths key",     "paths" in manifest_list[0])

# =========================================================================
print("\n=== _add_panel_labels ===")
fig_test, axes_test = plt.subplots(1, 3)
PaperFigureGenerator._add_panel_labels(
    list(axes_test.flat),
    [(0, "(a)"), (1, "(b)"), (2, "(c)")],
)
# Check that text annotations are present
texts = [t for ax in axes_test for t in ax.texts]
check("panel labels added",    len(texts) == 3, str(len(texts)))
check("first label is (a)",    any("(a)" in t.get_text() for t in texts))
plt.close(fig_test)

# =========================================================================
print("\n=== PDF format (vector output) ===")
with tempfile.TemporaryDirectory() as tmpdir:
    gen = PaperFigureGenerator(output_dir=tmpdir, formats=["pdf", "png"], dpi=150)
    fig = gen.fig_architecture_overview()
    check("PDF file created",  os.path.exists(
          os.path.join(tmpdir, "fig01_architecture.pdf")))
    check("PNG file also created", os.path.exists(
          os.path.join(tmpdir, "fig01_architecture.png")))
    check("Manifest paths has 2 entries", len(gen._manifest[0]["paths"]) == 2)
    plt.close(fig)

# =========================================================================
print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL PAPER FIGURE TESTS PASSED")
else:
    sys.exit(1)
