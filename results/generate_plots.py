"""Generate performance metric plots from the evaluation report."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# ── paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent
EVAL_DIR    = RESULTS_DIR / "evaluation"
PLOT_DIR    = RESULTS_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

with open(EVAL_DIR / "evaluation_report.json") as f:
    report = json.load(f)

y_true = np.array(report["y_true"])
y_prob = np.array(report["y_prob"])

# ── style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
AMBER  = "#D97706"
GREY   = "#6B7280"
LIGHT  = "#F3F4F6"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.color":       "#E5E7EB",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ROC CURVE
# ═══════════════════════════════════════════════════════════════════════════════
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc     = report["auroc"]["value"]
ci_lo       = report["auroc"]["ci_lower"]
ci_hi       = report["auroc"]["ci_upper"]

# operating point
thresh      = report["threshold"]
diffs       = np.abs(np.array([0.0] + list(fpr)) - np.array([0.0] + list(fpr)))
sens_op     = report["sensitivity"]["value"]
spec_op     = report["specificity"]["value"]
fpr_op      = 1 - spec_op

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"AUROC = {roc_auc:.4f}  [{ci_lo:.3f}–{ci_hi:.3f}]")
ax.fill_between(fpr, tpr * 0.94, tpr * 1.0, alpha=0.10, color=BLUE)
ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Chance")
ax.scatter([fpr_op], [sens_op], s=90, color=RED, zorder=5,
           label=f"Operating point (t={thresh:.3f})")
ax.annotate(f"  Sens={sens_op:.2f}\n  Spec={spec_op:.2f}",
            xy=(fpr_op, sens_op), fontsize=9, color=RED)
ax.set_xlabel("False Positive Rate (1 – Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curve — ABIDE I ASD Detection", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(PLOT_DIR / "01_roc_curve.png", dpi=150)
plt.close()
print("Saved 01_roc_curve.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PRECISION-RECALL CURVE
# ═══════════════════════════════════════════════════════════════════════════════
prec, rec, _ = precision_recall_curve(y_true, y_prob)
auprc        = report["auprc"]["value"]
baseline_pr  = y_true.mean()

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color=GREEN, lw=2, label=f"AUPRC = {auprc:.4f}  [{report['auprc']['ci_lower']:.3f}–{report['auprc']['ci_upper']:.3f}]")
ax.axhline(baseline_pr, linestyle="--", color=GREY, lw=1, label=f"Baseline (prevalence={baseline_pr:.2f})")
ppv_op = report["ppv"]["value"]
npv_op = report["npv"]["value"]
ax.scatter([sens_op], [ppv_op], s=90, color=RED, zorder=5,
           label=f"Operating point  PPV={ppv_op:.2f}")
ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision (PPV)")
ax.set_title("Precision–Recall Curve", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(PLOT_DIR / "02_pr_curve.png", dpi=150)
plt.close()
print("Saved 02_pr_curve.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SCORE DISTRIBUTION BY CLASS
# ═══════════════════════════════════════════════════════════════════════════════
asd_scores = y_prob[y_true == 1]
tc_scores  = y_prob[y_true == 0]

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 1, 30)
ax.hist(tc_scores,  bins=bins, alpha=0.55, color=BLUE,  label="TC (n=570)",  density=True)
ax.hist(asd_scores, bins=bins, alpha=0.55, color=RED,   label="ASD (n=530)", density=True)
ax.axvline(thresh, color="black", lw=1.5, linestyle="--", label=f"Decision threshold ({thresh:.3f})")
ax.set_xlabel("Predicted ASD probability")
ax.set_ylabel("Density")
ax.set_title("Score Distribution by Class", fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "03_score_distribution.png", dpi=150)
plt.close()
print("Saved 03_score_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CALIBRATION CURVE
# ═══════════════════════════════════════════════════════════════════════════════
n_bins   = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_mids  = []
frac_pos  = []
avg_pred  = []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = (y_prob >= lo) & (y_prob < hi)
    if mask.sum() > 0:
        bin_mids.append((lo + hi) / 2)
        frac_pos.append(y_true[mask].mean())
        avg_pred.append(y_prob[mask].mean())

ece = report["ece"]
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Perfect calibration")
ax.plot(avg_pred, frac_pos, "o-", color=AMBER, lw=2, ms=7, label=f"Model  (ECE={ece:.4f})")
ax.fill_between(avg_pred, frac_pos, avg_pred,
                alpha=0.12, color=AMBER, label="Calibration gap")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve", fontweight="bold")
ax.legend()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(PLOT_DIR / "04_calibration.png", dpi=150)
plt.close()
print("Saved 04_calibration.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
tp = report["tp"]; tn = report["tn"]
fp = report["fp"]; fn = report["fn"]
cm = np.array([[tn, fp], [fn, tp]])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.04)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.6 else "black",
                fontsize=13, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred TC", "Pred ASD"]); ax.set_yticklabels(["True TC", "True ASD"])
ax.set_title("Confusion Matrix", fontweight="bold")
fig.tight_layout()
fig.savefig(PLOT_DIR / "05_confusion_matrix.png", dpi=150)
plt.close()
print("Saved 05_confusion_matrix.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CLASSIFICATION METRICS BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
metric_keys = ["auroc", "auprc", "accuracy", "balanced_accuracy",
               "sensitivity", "specificity", "ppv", "npv", "f1", "mcc"]
labels_nice = ["AUROC", "AUPRC", "Accuracy", "Bal. Acc.",
               "Sensitivity", "Specificity", "PPV", "NPV", "F1", "MCC"]
vals  = [report[k]["value"] for k in metric_keys]
lo_ci = [report[k]["value"] - report[k]["ci_lower"] for k in metric_keys]
hi_ci = [report[k]["ci_upper"] - report[k]["value"] for k in metric_keys]

colors = [BLUE if v >= 0.7 else GREEN if v >= 0.65 else AMBER if v >= 0.6 else RED
          for v in vals]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(labels_nice))
bars = ax.bar(x, vals, color=colors, width=0.6, alpha=0.85, zorder=3)
ax.errorbar(x, vals, yerr=[lo_ci, hi_ci], fmt="none",
            ecolor="black", elinewidth=1.2, capsize=4, zorder=4)
ax.axhline(0.5, color=GREY, lw=1, linestyle="--", alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels_nice, rotation=30, ha="right")
ax.set_ylabel("Value"); ax.set_ylim(0, 1.05)
ax.set_title("Classification Metrics  (95% Bootstrap CI)", fontweight="bold")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
            f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
legend_patches = [
    mpatches.Patch(color=BLUE,  label="≥ 0.70"),
    mpatches.Patch(color=GREEN, label="≥ 0.65"),
    mpatches.Patch(color=AMBER, label="≥ 0.60"),
    mpatches.Patch(color=RED,   label="< 0.60"),
]
ax.legend(handles=legend_patches, title="Value range", loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_DIR / "06_metrics_bar.png", dpi=150)
plt.close()
print("Saved 06_metrics_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PER-SITE AUROC BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
site_data = report["per_site"]
sites = sorted(site_data.keys(), key=lambda s: int(s))
site_aucs = [site_data[s]["auroc"] for s in sites]
site_ns   = [site_data[s]["n"] for s in sites]
site_labels = [f"Site {s}\n(n={n})" for s, n in zip(sites, site_ns)]

site_colors = [BLUE if v >= 0.7 else GREEN if v >= 0.65 else AMBER if v >= 0.6
               else "#F97316" if v >= 0.5 else RED for v in site_aucs]

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(sites))
bars = ax.bar(x, site_aucs, color=site_colors, width=0.7, alpha=0.85, zorder=3)
ax.axhline(0.5, color=GREY, lw=1, linestyle="--", alpha=0.6, label="Chance")
ax.axhline(report["auroc"]["value"], color=BLUE, lw=1.5,
           linestyle="-.", alpha=0.8, label=f"Overall AUROC ({report['auroc']['value']:.3f})")
ax.set_xticks(x); ax.set_xticklabels(site_labels, fontsize=8)
ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.05)
ax.set_title("Per-Site AUROC  (20 ABIDE Acquisition Sites)", fontweight="bold")
ax.legend(fontsize=9)
for bar, v in zip(bars, site_aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015,
            f"{v:.2f}", ha="center", fontsize=7.5, rotation=0)
fig.tight_layout()
fig.savefig(PLOT_DIR / "07_per_site_auc.png", dpi=150)
plt.close()
print("Saved 07_per_site_auc.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PER-FOLD CHECKPOINT AUC  (surviving checkpoints from filenames)
# ═══════════════════════════════════════════════════════════════════════════════
import re

CKPT_DIR = RESULTS_DIR / "training" / "checkpoints"
fold_ckpts = {}
for pt in sorted(CKPT_DIR.glob("**/*.pt")):
    m = re.match(r"fold(\d)_epoch(\d+)_val_auc([\d.]+)\.pt", pt.name)
    if m:
        fold, epoch, auc_val = int(m[1]), int(m[2]), float(m[3])
        fold_ckpts.setdefault(fold, []).append((epoch, auc_val))

fold_colors = [BLUE, RED, GREEN, AMBER, "#8B5CF6"]
fig, ax = plt.subplots(figsize=(8, 5))

for fold in sorted(fold_ckpts):
    pts = sorted(fold_ckpts[fold])
    epochs = [p[0] for p in pts]
    aucs   = [p[1] for p in pts]
    ax.scatter(epochs, aucs, s=60, color=fold_colors[fold], zorder=4)
    ax.plot(epochs, aucs, "--", lw=1, color=fold_colors[fold],
            alpha=0.5, label=f"Fold {fold}  (peak={max(aucs):.4f})")

ax.axhline(0.5, color=GREY, lw=1, linestyle=":", alpha=0.7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation AUROC")
ax.set_title("Surviving Checkpoint AUCs per Fold\n"
             "(sparse: checkpoint bug kept worst + last few epochs)", fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0, 105); ax.set_ylim(0.45, 0.80)
fig.tight_layout()
fig.savefig(PLOT_DIR / "08_fold_checkpoints.png", dpi=150)
plt.close()
print("Saved 08_fold_checkpoints.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SENSITIVITY / SPECIFICITY TRADE-OFF ACROSS THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════
thresholds = np.linspace(0.01, 0.99, 200)
sens_arr = []
spec_arr = []
for t in thresholds:
    pred = (y_prob >= t).astype(int)
    tp_  = ((pred == 1) & (y_true == 1)).sum()
    tn_  = ((pred == 0) & (y_true == 0)).sum()
    fp_  = ((pred == 1) & (y_true == 0)).sum()
    fn_  = ((pred == 0) & (y_true == 1)).sum()
    sens_arr.append(tp_ / (tp_ + fn_ + 1e-9))
    spec_arr.append(tn_ / (tn_ + fp_ + 1e-9))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(thresholds, sens_arr, color=RED,   lw=2, label="Sensitivity")
ax.plot(thresholds, spec_arr, color=BLUE,  lw=2, label="Specificity")
ax.axvline(thresh, color="black", lw=1.5, linestyle="--",
           label=f"Optimal threshold ({thresh:.3f})")
ax.set_xlabel("Classification threshold")
ax.set_ylabel("Value")
ax.set_title("Sensitivity / Specificity vs. Threshold", fontweight="bold")
ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(PLOT_DIR / "09_sens_spec_threshold.png", dpi=150)
plt.close()
print("Saved 09_sens_spec_threshold.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SUMMARY SCORECARD
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")

scorecard = [
    ("AUROC",           f"{report['auroc']['value']:.4f}",
     f"[{report['auroc']['ci_lower']:.3f} – {report['auroc']['ci_upper']:.3f}]"),
    ("AUPRC",           f"{report['auprc']['value']:.4f}",
     f"[{report['auprc']['ci_lower']:.3f} – {report['auprc']['ci_upper']:.3f}]"),
    ("Accuracy",        f"{report['accuracy']['value']:.4f}",
     f"[{report['accuracy']['ci_lower']:.3f} – {report['accuracy']['ci_upper']:.3f}]"),
    ("Balanced Acc.",   f"{report['balanced_accuracy']['value']:.4f}",
     f"[{report['balanced_accuracy']['ci_lower']:.3f} – {report['balanced_accuracy']['ci_upper']:.3f}]"),
    ("Sensitivity",     f"{report['sensitivity']['value']:.4f}",
     f"[{report['sensitivity']['ci_lower']:.3f} – {report['sensitivity']['ci_upper']:.3f}]"),
    ("Specificity",     f"{report['specificity']['value']:.4f}",
     f"[{report['specificity']['ci_lower']:.3f} – {report['specificity']['ci_upper']:.3f}]"),
    ("PPV",             f"{report['ppv']['value']:.4f}",
     f"[{report['ppv']['ci_lower']:.3f} – {report['ppv']['ci_upper']:.3f}]"),
    ("NPV",             f"{report['npv']['value']:.4f}",
     f"[{report['npv']['ci_lower']:.3f} – {report['npv']['ci_upper']:.3f}]"),
    ("F1",              f"{report['f1']['value']:.4f}",
     f"[{report['f1']['ci_lower']:.3f} – {report['f1']['ci_upper']:.3f}]"),
    ("MCC",             f"{report['mcc']['value']:.4f}",
     f"[{report['mcc']['ci_lower']:.3f} – {report['mcc']['ci_upper']:.3f}]"),
    ("ECE",             f"{report['ece']:.4f}", ""),
    ("Brier Score",     f"{report['brier_score']['value']:.4f}",
     f"[{report['brier_score']['ci_lower']:.3f} – {report['brier_score']['ci_upper']:.3f}]"),
    ("TP / TN / FP / FN",
     f"{report['tp']} / {report['tn']} / {report['fp']} / {report['fn']}", ""),
    ("Threshold (Youden)", f"{report['threshold']:.4f}", ""),
]

ax.set_title("Evaluation Scorecard — ABIDE I  (n=1,100, 5-fold CV)",
             fontweight="bold", fontsize=13, pad=16)

col_x = [0.03, 0.42, 0.72]
row_h = 0.065
y0    = 0.91

ax.text(col_x[0], y0, "Metric",   fontweight="bold", fontsize=10, transform=ax.transAxes)
ax.text(col_x[1], y0, "Value",    fontweight="bold", fontsize=10, transform=ax.transAxes)
ax.text(col_x[2], y0, "95% CI",   fontweight="bold", fontsize=10, transform=ax.transAxes)
ax.plot([0, 1], [y0 - 0.01, y0 - 0.01], color="#D1D5DB", lw=0.8, transform=ax.transAxes)

for i, (metric, val, ci) in enumerate(scorecard):
    y = y0 - (i + 1) * row_h
    bg = LIGHT if i % 2 == 0 else "white"
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, y - 0.015), 1, row_h - 0.005,
        boxstyle="square,pad=0", facecolor=bg, edgecolor="none",
        transform=ax.transAxes))
    ax.text(col_x[0], y, metric, fontsize=9, transform=ax.transAxes)
    v_color = BLUE if float(val.split()[0]) >= 0.70 else GREEN if float(val.split()[0]) >= 0.65 \
        else AMBER if float(val.split()[0]) >= 0.60 else "black" \
        if not val[0].isdigit() else RED
    try:
        fv = float(val.split()[0])
        v_color = BLUE if fv >= 0.70 else GREEN if fv >= 0.65 else AMBER if fv >= 0.60 else RED
    except Exception:
        v_color = "black"
    ax.text(col_x[1], y, val, fontsize=9, fontweight="bold",
            color=v_color, transform=ax.transAxes)
    ax.text(col_x[2], y, ci,  fontsize=8.5, color=GREY, transform=ax.transAxes)

fig.tight_layout()
fig.savefig(PLOT_DIR / "10_scorecard.png", dpi=150)
plt.close()
print("Saved 10_scorecard.png")


print(f"\nAll plots saved to: {PLOT_DIR}")
