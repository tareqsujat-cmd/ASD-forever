"""
Conditions for trustworthy use of the ensemble: site base rate, operating point, decision value, strata.

    a  site-level calibration offset against the site's autistic proportion (leave-one-site-out)
    b  how wrong the assumed site proportion may be before the correction stops helping
    c  realised sensitivity and specificity of thresholds fixed on the training partition only
    d  decision curve (net benefit) of the recalibrated and the native probabilities
    e  AUROC by stratum (age, head motion, site size)
    f  calibration-in-the-large by stratum

Every number is read from results/<study>/review/trust_checks.json (exp_trust_checks.py).

    .venv/bin/python viz_trust.py [--study results/latest_study] [--out DIR] [--no-copy]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz_summary import ASD, TD, ACC, GREY, style, panel

ROOT = Path(__file__).resolve().parent
JOURNAL = ROOT / "ASD-journal-I" / "figures"
MINUS = "−"


def build(study: Path, out: Path, copy: bool):
    d = json.loads((study / "review" / "trust_checks.json").read_text())
    A = d["A_new_site_calibration"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.9), layout="constrained")

    # ---- a: offset against the site's autistic proportion
    ax = axes[0, 0]
    ps = sorted(A["per_site"], key=lambda r: r["autistic_proportion"])
    x = np.array([r["autistic_proportion"] for r in ps])
    y1 = np.array([r["citl_training_only"] for r in ps])
    y2 = np.array([r["citl_base_rate_shift"] for r in ps])
    ax.axhline(0, color=GREY, ls=":", lw=.9)
    for xi, a_, b_ in zip(x, y1, y2):
        ax.plot([xi, xi], [a_, b_], color="#BBBBBB", lw=1.0, zorder=1)
    ax.scatter(x, y1, s=34, color=ACC, marker="s", zorder=3, label="training-only recalibration")
    ax.scatter(x, y2, s=34, facecolor="white", edgecolor="#333333", lw=1.1, zorder=3,
               label="after shift to the site's proportion")
    for r in ps:
        if r["site"] in ("KKI", "USM", "UM_2", "PITT"):
            dx, ha = (0.006, "left") if r["site"] != "USM" else (-0.006, "right")
            ax.text(r["autistic_proportion"] + dx, r["citl_training_only"], r["site"], fontsize=9, ha=ha, va="center")
    ax.axvline(A["per_site"][0]["training_proportion"], color=GREY, ls="--", lw=.8)
    ax.text(A["per_site"][0]["training_proportion"] - .011, -0.97, "training\nproportion",
            fontsize=8.5, color="#555555", va="center", ha="right")
    ax.set_xlabel("autistic proportion of the withheld site")
    ax.set_ylabel("calibration-in-the-large (logit)")
    ax.text(.03, .97, f"Spearman $\\rho$ = {A['spearman_site_proportion_vs_offset']:.2f}\n10 sites with $\\geq$ 30 participants",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    panel(ax, "a", "Site offsets follow the site proportion")

    # ---- b: tolerance of the correction to an error in the assumed proportion
    ax = axes[0, 1]
    t = A["tolerance"]
    e = np.array([r["error"] for r in t])
    m = np.array([r["mean_abs_citl"] for r in t])
    none = A["global_recalibration (current)"]["mean_abs_citl"]
    ax.axhline(none, color=ASD, ls="--", lw=1.2, label="no site information")
    ax.plot(e, m, "o-", color=TD, lw=1.6, ms=5, label="shifted to the assumed proportion")
    for r in t:
        side = 0 if r["error"] == 0 else (1 if r["error"] > 0 else -1)
        ax.text(r["error"] + side * .013, r["mean_abs_citl"] + (-.06 if side == 0 else .0),
                f"{int(r['n_sites_ci_excludes_zero'])}", ha=("center" if side == 0 else ("left" if side > 0 else "right")),
                va="center", fontsize=8.5, color="#333333",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.85), zorder=5)
    ax.axvspan(-0.05, 0.05, color=ACC, alpha=.14, lw=0)
    ax.set_xlim(-0.235, 0.235)
    ax.set_xlabel("error in the assumed autistic proportion")
    ax.set_ylabel("mean absolute site offset (logit)")
    ax.set_ylim(0, max(m) * 1.18)
    ax.text(.02, .97, "numbers: sites with an interval excluding zero", transform=ax.transAxes, va="top", fontsize=8.5, color="#555555")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    panel(ax, "b", "Correction needs the proportion to within 0.05")

    # ---- c: realised operating points
    ax = axes[0, 2]
    rows = []
    for proto, pn in (("pooled", "pooled"), ("loso", "site withheld")):
        B = d[f"B_operating_points_{proto}"]
        for rule, tgt, other in (("specificity >= 0.80", "specificity", "sensitivity"),
                                 ("sensitivity >= 0.80", "sensitivity", "specificity")):
            rows.append((pn, tgt, B[rule]))
    yy = np.arange(len(rows))[::-1].astype(float)
    ax.axvline(0.80, color=GREY, ls="--", lw=.9)
    for y_, (pn, tgt, v) in zip(yy, rows):
        other = "sensitivity" if tgt == "specificity" else "specificity"
        ax.plot(v[tgt + "_ci"], [y_, y_], color=ASD, lw=1.6)
        ax.plot(v[tgt], y_, "o", color=ASD, ms=7, zorder=3)
        ax.plot(v[other + "_ci"], [y_ - .22, y_ - .22], color=TD, lw=1.2, alpha=.8)
        ax.plot(v[other], y_ - .22, "o", mfc="white", mec=TD, ms=6, zorder=3)
        ax.text(v[tgt + "_ci"][1] + .012, y_, f"{v[tgt]:.2f}", fontsize=8.5, ha="left", va="center")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{pn}\n{tgt} target" for pn, tgt, _ in rows], fontsize=8.8)
    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(-0.6, 4.1)
    ax.set_xlabel("realised on held-out participants")
    ax.plot([], [], "o", color=ASD, label="targeted metric")
    ax.plot([], [], "o", mfc="white", mec=TD, label="the other metric")
    ax.plot([], [], "--", color=GREY, label="target 0.80")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.95, fontsize=8.3, loc="upper center",
              ncol=3, handletextpad=.3, columnspacing=1.0)
    panel(ax, "c", "Training-set thresholds hold")

    # ---- d: decision curve
    ax = axes[1, 0]
    dc = d["B_operating_points_pooled"]["decision_curve"]
    th = np.array([r["threshold"] for r in dc])
    rec, nat, tall = (np.array([r[k] for r in dc]) for k in ("recalibrated", "native", "treat_all"))
    lo = np.array([r["recalibrated_ci"][0] for r in dc])
    hi = np.array([r["recalibrated_ci"][1] for r in dc])
    ax.fill_between(th, lo, hi, color=ACC, alpha=.18, lw=0)
    ax.plot(th, rec, "-", color=ACC, lw=1.8, label="recalibrated")
    ax.plot(th, nat, "--", color=TD, lw=1.3, label="native")
    ax.plot(th, tall, "-", color=GREY, lw=1.2, label="treat everyone")
    ax.axhline(0, color="#222222", lw=.9, ls=":", label="treat no one")
    ax.set_ylim(-0.12, 0.40)
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.text(.98, .97, "research cohort: 46% autistic", transform=ax.transAxes, ha="right", va="top", fontsize=8.8, color="#555555")
    ax.legend(frameon=False, fontsize=8.8, loc="lower right", ncol=2)
    panel(ax, "d", "Net benefit above treating everyone")

    # ---- e, f: strata
    names = ["age < 12", "age 12-17", "age >= 18", "motion: lowest third", "motion: middle third",
             "motion: highest third", "site size < 40", "site size 40-80", "site size >= 80"]
    nice = ["age < 12", "age 12–17", "age ≥ 18", "motion, lowest third", "motion, middle third",
            "motion, highest third", "site < 40", "site 40–79", "site ≥ 80"]
    ypos = np.array([0, 1, 2, 3.6, 4.6, 5.6, 7.2, 8.2, 9.2])[::-1]
    for ax, key, title, letter in ((axes[1, 1], "auroc", "AUROC by stratum", "e"),
                                    (axes[1, 2], "citl_recalibrated", "Calibration by stratum", "f")):
        for proto, col, off, lab in (("pooled", TD, .13, "pooled ten-fold"), ("loso", ASD, -.13, "site withheld")):
            S = d[f"C_stratified_{proto}"]
            for yv, nm in zip(ypos, names):
                v = S[nm]
                if key == "auroc":
                    est, ci = v["auroc"], v["auroc_ci"]
                else:
                    est, ci = v["citl_recalibrated"], v["citl_recalibrated_ci"]
                ax.plot(ci, [yv + off] * 2, color=col, lw=1.3, alpha=.85)
                ax.plot(est, yv + off, "o", color=col, ms=4.8, zorder=3)
        ax.set_yticks(ypos)
        ax.set_yticklabels(nice if letter == "e" else [""] * len(nice), fontsize=9)
        if letter == "e":
            ax.axvline(0.5, color=GREY, ls=":", lw=.8)
            ax.set_xlim(0.40, 0.95)
            ax.set_xlabel("AUROC (95% site-clustered interval)")
            ax.plot([], [], "o", color=TD, label="pooled ten-fold")
            ax.plot([], [], "o", color=ASD, label="site withheld")
            ax.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(0.0, 0.5))
        else:
            ax.axvline(0, color=GREY, ls=":", lw=.9)
            ax.set_xlabel("calibration-in-the-large after recalibration (logit)")
        panel(ax, letter, title)
    fig.align_ylabels()
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_trust.{ext}", dpi=300)
    plt.close(fig)
    if copy:
        JOURNAL.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            shutil.copy2(out / f"fig_trust.{ext}", JOURNAL / f"fig_trust.{ext}")
    return out / "fig_trust.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=str(ROOT / "results" / "latest_study"))
    ap.add_argument("--out", default=str(ROOT / "results" / "latest_study" / "figures"))
    ap.add_argument("--no-copy", action="store_true")
    a = ap.parse_args()
    style()
    print(build(Path(a.study).resolve(), Path(a.out), not a.no_copy))


if __name__ == "__main__":
    main()
