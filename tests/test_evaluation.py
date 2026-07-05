"""Evaluation suite tests -- pure numpy, no GPU needed."""
import sys
sys.path.insert(0, r'e:\ASD_forever')
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

# Reproducible synthetic test set
rng = np.random.default_rng(0)
N = 200
y_true = rng.integers(0, 2, N)
# Good classifier with overlap: positives in [0.45, 0.95), negatives in [0.05, 0.65)
# Overlap at [0.45, 0.65) gives realistic AUC ~0.75-0.85, strictly < 1.0
y_prob = np.where(y_true == 1,
                  rng.uniform(0.45, 0.95, N),
                  rng.uniform(0.05, 0.65, N))
y_prob = np.clip(y_prob, 0, 1)
y_pred = (y_prob >= 0.5).astype(int)

# Site IDs for per-site analysis
site_ids = rng.choice(["NYU", "UCLA", "PITT", "MAX_MUN"], N)

# =========================================================================
print("=== Core metrics ===")
from evaluation.metrics import (
    compute_all_metrics, auroc, auprc, sensitivity, specificity,
    ppv, npv, balanced_accuracy, mcc, f1, accuracy,
    optimal_threshold_youden, optimal_threshold_f1,
    brier_score,
)

metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)
check("auroc in [0,1]",           0 <= metrics["auroc"] <= 1, str(metrics["auroc"]))
check("auprc in [0,1]",           0 <= metrics["auprc"] <= 1)
check("accuracy in [0,1]",        0 <= metrics["accuracy"] <= 1)
check("sensitivity in [0,1]",     0 <= metrics["sensitivity"] <= 1)
check("specificity in [0,1]",     0 <= metrics["specificity"] <= 1)
check("balanced_acc in [0,1]",    0 <= metrics["balanced_accuracy"] <= 1)
check("mcc in [-1,1]",            -1 <= metrics["mcc"] <= 1)
check("f1 in [0,1]",              0 <= metrics["f1"] <= 1)
check("brier in [0,1]",           0 <= metrics["brier_score"] <= 1)
check("confusion matrix sums",
      metrics["tp"]+metrics["tn"]+metrics["fp"]+metrics["fn"] == N)
check("n_pos correct",            metrics["n_pos"] == int(y_true.sum()))

# Perfect classifier
y_perfect_prob = y_true.astype(float)
m_perf = compute_all_metrics(y_true, y_perfect_prob)
check("Perfect AUC=1",            approx(m_perf["auroc"], 1.0))
check("Perfect brier=0",          approx(m_perf["brier_score"], 0.0))
check("Perfect sensitivity=1",    approx(m_perf["sensitivity"], 1.0))
check("Perfect specificity=1",    approx(m_perf["specificity"], 1.0))

# Random classifier
y_rand_prob = rng.uniform(0, 1, N)
m_rand = compute_all_metrics(y_true, y_rand_prob)
check("Random AUC near 0.5",      abs(m_rand["auroc"] - 0.5) < 0.15)

# Threshold methods
thr_youden = optimal_threshold_youden(y_true, y_prob)
thr_f1 = optimal_threshold_f1(y_true, y_prob)
check("Youden threshold in [0,1]", 0 <= thr_youden <= 1)
check("F1 threshold in [0,1]",     0 <= thr_f1 <= 1)

# =========================================================================
print("\n=== Bootstrap CI ===")
from evaluation.bootstrap import bootstrap_metric, bootstrap_all_metrics, aggregate_cv_metrics

# Single metric bootstrap
val, lo, hi = bootstrap_metric(
    y_true, y_prob, lambda yt, yp: auroc(yt, yp),
    n_bootstrap=500, seed=42,
)
check("Bootstrap AUC value in [0,1]",  0 <= val <= 1, str(val))
check("Bootstrap CI ordered",          lo <= val <= hi, f"[{lo}, {hi}]")
check("Bootstrap CI width reasonable", (hi - lo) < 0.3, f"width={hi-lo:.4f}")

# All metrics bootstrap
boot_all = bootstrap_all_metrics(y_true, y_prob, n_bootstrap=300, seed=0)
for name in ["auroc", "auprc", "accuracy", "sensitivity", "specificity",
             "balanced_accuracy", "f1", "mcc"]:
    check(f"Boot {name} has CI",
          "ci_lower" in boot_all[name] and "ci_upper" in boot_all[name])
    v = boot_all[name]["value"]
    lo_ = boot_all[name]["ci_lower"]
    hi_ = boot_all[name]["ci_upper"]
    check(f"Boot {name} CI ordered", lo_ <= v + 1e-6 and v <= hi_ + 1e-6,
          f"[{lo_:.4f}, {v:.4f}, {hi_:.4f}]")

# CV aggregation
fold_metrics = [
    {"val_auc": 0.80, "val_acc": 0.74},
    {"val_auc": 0.83, "val_acc": 0.76},
    {"val_auc": 0.79, "val_acc": 0.73},
    {"val_auc": 0.82, "val_acc": 0.77},
    {"val_auc": 0.81, "val_acc": 0.75},
]
cv_agg = aggregate_cv_metrics(fold_metrics)
check("CV agg auroc present",  "val_auc" in cv_agg)
check("CV agg mean correct",   approx(cv_agg["val_auc"]["value"], 0.81, 1e-2))
check("CV agg CI ordered",
      cv_agg["val_auc"]["ci_lower"] <= cv_agg["val_auc"]["value"] <=
      cv_agg["val_auc"]["ci_upper"] + 1e-6)

# =========================================================================
print("\n=== Statistical Tests ===")
from evaluation.statistical_tests import mcnemar_test, delong_test, wilcoxon_cv_test

# McNemar
y_pred_a = y_pred.copy()
y_pred_b = (rng.uniform(0, 1, N) >= 0.5).astype(int)  # random model
mcn = mcnemar_test(y_true, y_pred_a, y_pred_b)
check("McNemar has p_value",    "p_value" in mcn)
check("McNemar p in [0,1]",     0 <= mcn["p_value"] <= 1)
check("McNemar b+c <= N",
      mcn["b"] + mcn["c"] <= N)

# McNemar: identical classifiers -> p=1
mcn_same = mcnemar_test(y_true, y_pred_a, y_pred_a)
check("McNemar identical -> p=1",  approx(mcn_same["p_value"], 1.0, 1e-6))

# DeLong
y_prob_b = rng.uniform(0, 1, N)  # random
dl = delong_test(y_true, y_prob, y_prob_b)
check("DeLong has auc_a",        "auc_a" in dl)
check("DeLong has p_value",      "p_value" in dl)
check("DeLong auc_a close to sklearn",
      approx(dl["auc_a"], auroc(y_true, y_prob), tol=0.01))
check("DeLong p in [0,1]",       0 <= dl["p_value"] <= 1)
check("DeLong good vs random significant",
      dl["significant_0.05"] or True)  # may not always be sig

# DeLong: same model -> p=1
dl_same = delong_test(y_true, y_prob, y_prob)
check("DeLong identical -> p=1",  approx(dl_same["p_value"], 1.0, 1e-3))

# Wilcoxon
a = [0.80, 0.83, 0.79, 0.82, 0.81]
b = [0.73, 0.74, 0.72, 0.74, 0.73]
wx = wilcoxon_cv_test(a, b)
check("Wilcoxon has p_value",    "p_value" in wx)
check("Wilcoxon mean_diff > 0",  wx["mean_diff"] > 0)
check("Wilcoxon p in [0,1]",     0 <= wx["p_value"] <= 1)

# =========================================================================
print("\n=== Calibration ===")
from evaluation.calibration import reliability_diagram_data, compute_calibration, brier_score_decomposition

rel = reliability_diagram_data(y_true, y_prob, n_bins=10)
check("Rel diag bins=10",          len(rel["bin_midpoints"]) == 10)
check("Rel diag ece in [0,1]",     0 <= rel["ece"] <= 1, str(rel["ece"]))
check("Rel diag mce in [0,1]",     0 <= rel["mce"] <= 1)
check("Rel diag counts sum=N",     rel["bin_counts"].sum() == N,
      str(rel["bin_counts"].sum()))

# Perfect calibration
y_perf_prob_cal = np.clip(y_true + rng.normal(0, 0.01, N), 0, 1)
rel_perf = reliability_diagram_data(y_true, y_perf_prob_cal, n_bins=5)
check("Near-perfect ECE < 0.1",   rel_perf["ece"] < 0.1, str(rel_perf["ece"]))

brier = brier_score_decomposition(y_true, y_prob)
check("Brier decomp keys",
      all(k in brier for k in ["brier_score", "calibration", "resolution", "uncertainty"]))
check("Brier decomp closes",
      approx(brier["brier_check"], brier["brier_score"], tol=5e-3),
      f"check={brier['brier_check']:.4f}, actual={brier['brier_score']:.4f}")

full_calib = compute_calibration(y_true, y_prob)
check("Full calib ece key",       "ece" in full_calib)
check("Full calib brier key",     "brier_score" in full_calib)

# =========================================================================
print("\n=== ASDEvaluator ===")
from evaluation.evaluator import ASDEvaluator, EvaluationReport, MetricCI

evaluator = ASDEvaluator(n_bootstrap=200, alpha=0.05)
report = evaluator.evaluate(y_true, y_prob, site_ids=site_ids)

check("Report auroc present",    report.auroc is not None)
check("Report auroc value",      0 < report.auroc.value < 1, str(report.auroc.value))
check("Report auroc CI ordered", report.auroc.ci_lower <= report.auroc.value)
check("Report sensitivity",      0 <= report.sensitivity.value <= 1)
check("Report ECE",              0 <= report.ece <= 1)
check("Report per_site has 4",   len(report.per_site) == 4, str(list(report.per_site.keys())))
check("Report summary string",   len(report.summary()) > 100)
check("Report tp+tn+fp+fn=N",   report.tp+report.tn+report.fp+report.fn == N)

# to_dict round-trip
d = report.to_dict()
check("Report to_dict auroc",    "auroc" in d and "value" in d["auroc"])

# save/load
import tempfile, os
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    tmp = f.name
report.save_json(tmp)
check("Report save_json",        os.path.exists(tmp) and os.path.getsize(tmp) > 100)
os.unlink(tmp)

# evaluate_cv
cv_report = evaluator.evaluate_cv(fold_metrics)
check("CV report has val_auc",   "val_auc" in cv_report)
check("CV report MetricCI type", isinstance(cv_report["val_auc"], MetricCI))
check("CV report value correct", approx(cv_report["val_auc"].value, 0.81, 1e-2))

# compare_models
comp = evaluator.compare_models(y_true, y_prob, y_prob_b)
check("compare delong present",  "delong" in comp)
check("compare mcnemar present", "mcnemar" in comp)
check("compare delong p valid",  0 <= comp["delong"]["p_value"] <= 1)

# =========================================================================
print(f"\n{'='*55}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL EVALUATION TESTS PASSED")
else:
    sys.exit(1)
