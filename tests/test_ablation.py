"""Ablation study tests -- no actual training, all mock."""
import sys
sys.path.insert(0, r'e:\ASD_forever')
import matplotlib
matplotlib.use('Agg')
import json, os, tempfile
import numpy as np

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

def approx(a, b, tol=1e-4):
    return abs(a - b) < tol

# =========================================================================
# Mock training function
# =========================================================================
# Deterministic: name hash → stable 5-fold AUC values so Wilcoxon tests
# are predictable.

PRESET = {
    "baseline":       [0.80, 0.81, 0.79, 0.82, 0.80],
    "fusion=gated":   [0.75, 0.74, 0.76, 0.75, 0.74],
    "fusion=late":    [0.70, 0.71, 0.69, 0.72, 0.70],
    "backbone=swin3d":[0.83, 0.84, 0.82, 0.85, 0.83],
}

def _mock_train_fn(cfg, name):
    """Returns 5-fold metrics. Uses preset values if available, else random."""
    aucs = PRESET.get(name)
    if aucs is None:
        seed = int.from_bytes(name.encode()[:8].ljust(8, b'\x00'), "big") % (2**32)
        rng  = np.random.default_rng(seed)
        aucs = list(0.75 + 0.1 * rng.standard_normal(5))
    return [{"val_auc": float(a), "val_acc": float(a - 0.05)} for a in aucs]

# =========================================================================
print("=== AblationDimension ===")
from ablation.ablation_config import AblationDimension, AblationStudy

dim_fusion = AblationDimension(
    name="fusion",
    variants={
        "cross_attention": {"fusion.architecture": "cross_attention"},
        "gated":           {"fusion.architecture": "gated"},
        "late":            {"fusion.architecture": "late"},
    },
    default="cross_attention",
    description="Fusion strategy",
)
check("AblationDimension name",              dim_fusion.name == "fusion")
check("AblationDimension default valid",     dim_fusion.default in dim_fusion.variants)
check("AblationDimension non_default count", len(dim_fusion.non_default_variants) == 2)
check("AblationDimension default_overrides", "fusion.architecture" in dim_fusion.default_overrides)

dim_backbone = AblationDimension(
    name="backbone",
    variants={
        "resnet10": {"model.mri.architecture": "resnet10"},
        "swin3d":   {"model.mri.architecture": "swin3d"},
    },
    default="resnet10",
)
check("AblationDimension 2nd dim ok",        dim_backbone.default == "resnet10")

# Invalid default raises
try:
    AblationDimension(name="x", variants={"a": {}}, default="b")
    check("Bad default raises ValueError",   False, "No exception")
except ValueError:
    check("Bad default raises ValueError",   True)

# =========================================================================
print("\n=== AblationStudy OFAT ===")
base_cfg = {"fusion": {"architecture": "cross_attention"},
            "model":  {"mri": {"architecture": "resnet10"}}}

study_ofat = AblationStudy(
    name="test_ofat",
    base_config=base_cfg,
    dimensions=[dim_fusion, dim_backbone],
    mode="ofat",
)
variants = study_ofat.generate_variants()
# OFAT: 1 baseline + 2 (fusion non-defaults) + 1 (backbone non-default) = 4
check("OFAT variant count",               len(variants) == 4,
      f"got {len(variants)}")
names = [n for n, _ in variants]
check("OFAT has baseline",                "baseline" in names)
check("OFAT has fusion=gated",            "fusion=gated" in names)
check("OFAT has fusion=late",             "fusion=late"  in names)
check("OFAT has backbone=swin3d",         "backbone=swin3d" in names)

# Overrides for baseline include both defaults
_, bl_overrides = next(v for v in variants if v[0] == "baseline")
check("Baseline has fusion override",     "fusion.architecture" in bl_overrides)
check("Baseline has backbone override",   "model.mri.architecture" in bl_overrides)

# Overrides for fusion=gated only changes fusion
_, gated_ovr = next(v for v in variants if v[0] == "fusion=gated")
check("gated override value correct",     gated_ovr["fusion.architecture"] == "gated")
check("gated keeps backbone default",     gated_ovr["model.mri.architecture"] == "resnet10")

# =========================================================================
print("\n=== AblationStudy Factorial ===")
study_fact = AblationStudy(
    name="test_fact",
    base_config=base_cfg,
    dimensions=[dim_fusion, dim_backbone],
    mode="factorial",
)
fact_variants = study_fact.generate_variants()
# 3 fusion × 2 backbone = 6
check("Factorial variant count",          len(fact_variants) == 6,
      f"got {len(fact_variants)}")
fact_names = [n for n, _ in fact_variants]
check("Factorial has cross+resnet",       "fusion=cross_attention__backbone=resnet10" in fact_names)
check("Factorial has gated+swin",         "fusion=gated__backbone=swin3d" in fact_names)

# =========================================================================
print("\n=== AblationStudy Custom ===")
custom = [("my_variant_a", {"x": 1}), ("my_variant_b", {"x": 2})]
study_custom = AblationStudy(
    name="custom", base_config={}, dimensions=[],
    mode="custom", custom_variants=custom,
)
check("Custom variants",                  study_custom.generate_variants() == custom)

# =========================================================================
print("\n=== AblationResults ===")
from ablation.ablation_results import AblationResults, VariantResult

results = AblationResults(study_name="test")
check("Empty results len",                len(results) == 0)

fold_m = [{"val_auc": 0.80, "val_acc": 0.75},
          {"val_auc": 0.82, "val_acc": 0.77},
          {"val_auc": 0.79, "val_acc": 0.74}]
r = results.add("baseline", fold_m)
check("Add returns VariantResult",        isinstance(r, VariantResult))
check("Results len after add",            len(results) == 1)
check("in operator",                      "baseline" in results)
check("not in operator",                  "nonexistent" not in results)

r2 = results.get("baseline")
check("Get returns VariantResult",        r2 is not None)
check("Mean AUC correct",                 approx(r2.get_metric("val_auc"),
                                                  np.mean([0.80,0.82,0.79])))
check("Std AUC correct",                  r2.std_metrics.get("val_auc", None) is not None)
check("summary_str format",               "±" in r2.summary_str("val_auc"),
      repr(r2.summary_str("val_auc")))
check("fold_values correct",              len(r2.get_fold_values("val_auc")) == 3)
check("missing metric is nan",            np.isnan(r2.get_metric("nonexistent")))

# Save and load
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "results.json")
    results.save_json(path)
    check("JSON file exists",             os.path.exists(path))
    check("JSON file non-empty",          os.path.getsize(path) > 50)

    loaded = AblationResults.load_json(path)
    check("Loaded results len",           len(loaded) == 1)
    lr = loaded.get("baseline")
    check("Loaded mean AUC",              lr is not None and
                                          approx(lr.get_metric("val_auc"),
                                                  np.mean([0.80,0.82,0.79])))
    check("Loaded fold metrics count",    len(lr.fold_metrics) == 3)

# available_metrics
results.add("v2", [{"val_auc": 0.85, "val_brier": 0.12}])
avail = results.available_metrics()
check("available_metrics includes val_auc",   "val_auc" in avail)
check("available_metrics includes val_brier", "val_brier" in avail)

# =========================================================================
print("\n=== default_config_modifier ===")
from ablation.ablation_runner import default_config_modifier

base = {"fusion": {"architecture": "cross_attention"}, "training": {"lr": 1e-3}}
modified = default_config_modifier(base, {"fusion.architecture": "gated", "training.lr": 5e-4})
check("modifier sets nested dict key",    modified["fusion"]["architecture"] == "gated")
check("modifier sets another nested key", approx(modified["training"]["lr"], 5e-4))
check("modifier does not mutate base",    base["fusion"]["architecture"] == "cross_attention")

# Object with attributes
class _Cfg:
    def __init__(self): self.lr = 1e-3; self.epochs = 10
cfg_obj = _Cfg()
modified_obj = default_config_modifier(cfg_obj, {"lr": 5e-4})
check("modifier sets attribute",          approx(modified_obj.lr, 5e-4))
check("modifier does not mutate orig",    approx(cfg_obj.lr, 1e-3))

# =========================================================================
print("\n=== AblationRunner ===")
from ablation.ablation_runner import AblationRunner

with tempfile.TemporaryDirectory() as tmpdir:
    runner = AblationRunner(
        train_fn=_mock_train_fn,
        save_dir=tmpdir,
        verbose=False,
    )
    study = AblationStudy(
        name="runner_test",
        base_config={"fusion": {"architecture": "cross_attention"},
                     "model":  {"mri": {"architecture": "resnet10"}}},
        dimensions=[dim_fusion, dim_backbone],
        mode="ofat",
    )
    results_run = runner.run_study(study, resume=False)
    check("Runner produces AblationResults",  isinstance(results_run, AblationResults))
    check("Runner ran all 4 variants",        len(results_run) == 4,
          str(len(results_run)))
    check("Results saved to disk",            os.path.exists(
          os.path.join(tmpdir, "runner_test_results.json")))

    # Resume: re-run with resume=True — should skip all
    import copy
    results_resume = runner.run_study(study, resume=True)
    check("Resume has same variant count",    len(results_resume) == 4)

    # Check baseline metric
    bl = results_resume.get("baseline")
    check("Baseline val_auc finite",          np.isfinite(bl.get_metric("val_auc")))

    # run_single
    single = runner.run_single(
        base_config=base_cfg,
        variant_name="fusion=gated",
        overrides={"fusion.architecture": "gated"},
    )
    check("run_single returns results",       len(single) == 1)
    check("run_single variant in results",    "fusion=gated" in single)

# Failure handling: train_fn raises → variant stored with error flag
def _failing_train_fn(cfg, name):
    if name == "fusion=gated":
        raise RuntimeError("Deliberate failure")
    return _mock_train_fn(cfg, name)

with tempfile.TemporaryDirectory() as tmpdir:
    runner_f = AblationRunner(train_fn=_failing_train_fn, save_dir=tmpdir, verbose=False)
    res_f = runner_f.run_study(
        AblationStudy("fail_test", base_cfg, [dim_fusion], "ofat"),
        resume=False,
    )
    check("Failed variant is still stored",   "fusion=gated" in res_f)
    check("Other variants ran ok",            "baseline" in res_f and
                                               "fusion=late" in res_f)

# =========================================================================
print("\n=== AblationAnalyzer ===")
from ablation.ablation_analyzer import AblationAnalyzer

# Build a result set with known values
an_results = AblationResults("analyzer_test")
for name, aucs in PRESET.items():
    an_results.add(name, [{"val_auc": a, "val_acc": a-0.05} for a in aucs])

analyzer = AblationAnalyzer(an_results, baseline_name="baseline")

# --- Ranking ---
ranked = analyzer.rank_variants("val_auc")
check("Rank returns list",                isinstance(ranked, list))
check("Rank has correct length",          len(ranked) == len(PRESET),
      str(len(ranked)))
# backbone=swin3d (mean 0.834) > baseline (0.804) > fusion=gated (0.748) > fusion=late (0.704)
check("Best variant is swin3d",           ranked[0][0] == "backbone=swin3d",
      ranked[0][0])
check("Worst variant is fusion=late",     ranked[-1][0] == "fusion=late",
      ranked[-1][0])

best_name, best_val = analyzer.best_variant("val_auc")
check("best_variant returns correct name", best_name == "backbone=swin3d",
      best_name)
check("best_variant returns float",        isinstance(best_val, float))

# --- Significance tests ---
cmp = analyzer.compare_to_baseline("val_auc")
check("compare_to_baseline returns dict",  isinstance(cmp, dict))
check("compare excludes baseline itself",  "baseline" not in cmp)
check("fusion=gated in comparisons",       "fusion=gated" in cmp)
check("fusion=gated has p_value",          "p_value" in cmp.get("fusion=gated", {}))
check("fusion=gated direction worse",      cmp["fusion=gated"].get("direction") == "worse",
      str(cmp["fusion=gated"]))
check("swin3d direction better",           cmp.get("backbone=swin3d", {}).get("direction") == "better",
      str(cmp.get("backbone=swin3d")))

# Significance matrix
sig_mat = analyzer.significance_matrix("val_auc")
check("Sig matrix is dict of dicts",       isinstance(sig_mat, dict) and
                                            all(isinstance(v, dict) for v in sig_mat.values()))
check("Sig matrix diagonal p=1",           approx(
      sig_mat.get("baseline", {}).get("baseline", {}).get("p_value", 0), 1.0))

# --- summary dict ---
summ = analyzer.summary(metrics=["val_auc"])
check("Summary has all variants",          set(summ.keys()) == set(PRESET.keys()))
check("Summary has rank",                  "rank" in summ.get("baseline", {}))
check("Summary rank is integer",           isinstance(summ["baseline"]["rank"], int))

# --- Markdown table ---
md = analyzer.markdown_table(metrics=["val_auc", "val_acc"])
check("Markdown table is non-empty",       len(md) > 50)
check("Markdown table has header row",     "AUC" in md or "val_auc" in md)
check("Markdown table has baseline row",   "baseline" in md)
check("Markdown table has pipe chars",     "|" in md)

# --- LaTeX table ---
latex = analyzer.latex_table(
    metrics=["val_auc", "val_acc"],
    caption="Test ablation.",
    label="tab:test",
)
check("LaTeX table non-empty",             len(latex) > 100)
check("LaTeX has tabular",                 "tabular" in latex)
check("LaTeX has toprule",                 "toprule" in latex)
check("LaTeX has textbf (best bolded)",    "textbf" in latex)
check("LaTeX has caption",                 "Test ablation." in latex)

# --- Plot ---
import matplotlib.pyplot as plt
fig = analyzer.plot_comparison("val_auc")
check("Plot returns Figure",               isinstance(fig, plt.Figure))
plt.close(fig)

# =========================================================================
print("\n=== Study factory functions ===")
from ablation.study_factory import (
    build_fusion_ablation, build_backbone_ablation,
    build_genetics_ablation, build_modality_ablation,
    build_full_ablation, build_fusion_backbone_factorial,
)

base_cfg_factory = {
    "fusion": {"architecture": "cross_attention"},
    "model":  {"mri": {"architecture": "resnet10"},
               "genetics": {"architecture": "transformer"},
               "modality": "multimodal"},
    "training": {"loss": "focal", "use_ema": True},
}

fs = build_fusion_ablation(base_cfg_factory)
check("Fusion study name",                 fs.name == "fusion_ablation")
check("Fusion OFAT variants = 5",         fs.num_variants() == 5,
      str(fs.num_variants()))  # 1 baseline + 4 non-defaults

bs = build_backbone_ablation(base_cfg_factory)
check("Backbone study variants = 5",      bs.num_variants() == 5)

gs = build_genetics_ablation(base_cfg_factory)
check("Genetics study variants = 4",      gs.num_variants() == 4)

ms = build_modality_ablation(base_cfg_factory)
check("Modality study variants = 3",      ms.num_variants() == 3)

full = build_full_ablation(base_cfg_factory)
# 1 baseline + 4 fusion + 4 backbone + 3 genetics + 2 modality + 2 loss + 1 ema = 17
check("Full ablation variant count = 17", full.num_variants() == 17,
      str(full.num_variants()))

fact = build_fusion_backbone_factorial(base_cfg_factory)
# 5 fusion × 5 backbone = 25
check("Factorial 5x5 = 25 variants",      fact.num_variants() == 25,
      str(fact.num_variants()))

# =========================================================================
print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL ABLATION TESTS PASSED")
else:
    sys.exit(1)
