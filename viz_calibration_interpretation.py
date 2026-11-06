"""Two Results figures for the calibration and interpretation analyses, in the house style of
viz_summary.py (same palette, Times New Roman, panel letters).

    fig_calibration     a, b  reliability of the native and recalibrated probabilities
                        c     calibration-in-the-large with site-clustered intervals
                        d     calibration-in-the-large at each site withheld
    fig_interpretation  a     contribution of each system pair to the group separation
                        b     held-out AUROC of each system pair against the permutation maximum
                        c     reproducibility across training partitions against the null
                        d     held-out AUROC after deleting / keeping the top-ranked pairs

Every number is read from results/latest_study/review/ at render time.

    .venv/bin/python viz_calibration_interpretation.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from viz_summary import ASD, TD, ACC, GREY, style, panel

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "results" / "latest_study" / "review"
SYSTEMS = ["visual", "sensorimotor", "dorsal attention", "salience / ventral attention",
           "limbic", "frontoparietal control", "default mode", "subcortical / cerebellar"]
SHORT = ["VIS", "SMN", "DAN", "SAL", "LIM", "FPN", "DMN", "SUB"]
ABBR = dict(zip(SYSTEMS, SHORT))
WARM = "#C77A1E"


def read(name):
    p = REVIEW / name
    return pd.read_csv(p) if p.exists() else None


def pair_label(block):
    a, b = block.split(" | ")
    return f"{ABBR[a]}–{ABBR[b]}"


# =========================================================================================
def fig_calibration(out: Path):
    rel, met = read("calibration_reliability.csv"), read("calibration_nested.csv")
    site, het = read("calibration_per_site.csv"), read("calibration_site_heterogeneity.csv")
    if rel is None or met is None:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.9), layout="constrained")
    styles = {"native": (TD, "o", "native"), "intercept only": (ACC, "s", "intercept recalibration"),
              "logistic": (WARM, "^", "logistic recalibration")}

    for ax, proto, letter, title in ((axes[0, 0], "pooled", "a", "Reliability, pooled ten-fold"),
                                     (axes[0, 1], "loso", "b", "Reliability, site withheld")):
        for v, (col, mk, nm) in styles.items():
            d = rel[(rel.protocol == proto) & (rel.variant == v) & (rel.scheme == "equal width")]
            if not len(d):
                continue
            ax.errorbar(d.mean_predicted, d.observed,
                        yerr=[d.observed - d.obs_lo, d.obs_hi - d.observed], fmt="none",
                        capsize=1.5, color=col, alpha=.55, elinewidth=.7)
            ax.plot(d.mean_predicted, d.observed, "-", color=col, lw=1.0, alpha=.9)
            ax.scatter(d.mean_predicted, d.observed, s=np.clip(d.n / 2.2, 6, 90), marker=mk,
                       color=col, alpha=.95, zorder=3, label=nm, lw=0)
        ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted probability (ten equal-width bins)")
        ax.text(.98, .03, "marker area $\\propto$ bin count\nbars: 95% Wilson interval", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color="#555555")
        ax.set_ylabel("observed proportion autistic")
        panel(ax, letter, title)
    for a_ in (axes[0, 0], axes[0, 1]):                     # the same key on both reliability panels
        a_.legend(frameon=False, loc="upper left", fontsize=9.5, markerscale=.8)

    # c: calibration-in-the-large and slope
    ax = axes[1, 0]
    rows, labels = [], []
    for proto, pn in (("pooled", "pooled"), ("loso", "site withheld")):
        for v, (col, mk, nm) in list(styles.items()) + [("unpenalised intercept",
                                                        (GREY, "D", "unpenalised intercept"))]:
            r = met[(met.protocol == proto) & (met.variant == v)
                    & (met.metric == "calibration-in-the-large")]
            if len(r):
                rows.append((pn, nm, col, mk, r.iloc[0]))
    ypos, heads = [], []
    y = 0
    for k, (pn, nm, col, mk, r) in enumerate(rows):
        if k and rows[k - 1][0] != pn:
            y += 0.6
        ax.hlines(y, r.lo, r.hi, color=col, lw=1.4)
        ax.plot(r.estimate, y, mk, color=col, ms=6, zorder=3)
        ypos.append((y, f"{nm}" + (" (diagnostic)" if nm == "unpenalised intercept" else "")))
        if nm == "native":
            heads.append((y, pn))
        y += 1
    ax.axvline(0, color=GREY, ls=":", lw=.9)
    ax.set_yticks([p for p, _ in ypos]); ax.set_yticklabels([t for _, t in ypos], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("calibration-in-the-large (logit; negative = over-prediction)\n"
                  "bars: 95% site-clustered bootstrap interval; diamond: diagnostic refit, not a correction",
                  fontsize=9.5)
    for yh, pn in heads:                                    # block headings in the free space on the right
        ax.text(.99, yh, {"pooled": "pooled ten-fold", "site withheld": "site withheld"}[pn],
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10,
                fontweight="bold", color="#222222")
    panel(ax, "c", "Average predicted risk against observed rate")

    # d: per-site calibration-in-the-large, site withheld
    ax = axes[1, 1]
    if site is not None:
        d = site[(site.protocol == "loso") & (site.n >= 30)].dropna(subset=["citl"])
        nat = d[d.variant == "native"].set_index("site")
        rec = d[d.variant == "intercept only"].set_index("site")
        order = nat.sort_values("citl").index
        yy = np.arange(len(order))
        ax.hlines(yy, nat.loc[order].citl_lo, nat.loc[order].citl_hi, color=TD, lw=1, alpha=.6)
        ax.plot(nat.loc[order].citl, yy, "o", color=TD, ms=4.5, label="native")
        ax.plot(rec.loc[order].citl, yy, "s", color=ACC, ms=4, label="intercept recalibration")
        ax.axvline(0, color=GREY, ls=":", lw=.9)
        ax.set_yticks(yy); ax.set_yticklabels(order, fontsize=8.5)
        ax.set_xlabel("calibration-in-the-large at the withheld site (logit)\n"
                      "bars: 95% Wald interval of the native estimate", fontsize=9.5)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        if het is not None:
            h = het[(het.protocol == "loso") & (het.variant == "intercept only")]
            if len(h):
                h = h.iloc[0]
                ax.text(.02, .50,
                        f"after recalibration:\n$I^2$ = {100 * h.I2:.0f}%, $\\tau$ = {h.tau:.2f}",
                        transform=ax.transAxes, ha="left", va="center", fontsize=9)
    panel(ax, "d", "Every site with at least 30 participants")
    return save(fig, out, "fig_calibration")


# =========================================================================================
def fig_interpretation(out: Path):
    blocks, stab = read("interpretation_blocks.csv"), read("interpretation_stability.csv")
    snull, bnull = read("interpretation_stability_null.csv"), read("interpretation_block_null.csv")
    dele = read("interpretation_deletion.csv")
    if blocks is None:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4), layout="constrained")
    B = blocks[(blocks.protocol == "pooled") & (blocks.view == "mean of views")].copy()
    B = B[B.block != "unassigned"]

    # a: system-pair matrix of contribution differences
    ax = axes[0, 0]
    M = np.full((8, 8), np.nan)
    star = np.full((8, 8), "", dtype=object)
    fw = None
    if bnull is not None:
        fw = bnull[bnull.view == "mean of views"].set_index("block").p_familywise
    for _, r in B.iterrows():
        a, b = r.block.split(" | ")
        i, j = SYSTEMS.index(a), SYSTEMS.index(b)
        M[max(i, j), min(i, j)] = r.diff_per_connection * 1000
        if fw is not None and fw.get(r.block, 1) < .05:
            star[max(i, j), min(i, j)] = "*"
    lim = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(8):
        for j in range(8):
            if star[i, j]:
                ax.text(j, i, star[i, j], ha="center", va="center", fontsize=14, color="k")
    ax.set_xticks(range(8)); ax.set_xticklabels(SHORT, fontsize=9, rotation=45)
    ax.set_yticks(range(8)); ax.set_yticklabels(SHORT, fontsize=9)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=.8)
    cb.set_label("contribution per connection, autistic $-$ non-autistic ($\\times10^{-3}$ logit)", fontsize=9)
    panel(ax, "a", "Where the score separates the groups")

    # b: held-out AUROC of each pair
    ax = axes[0, 1]
    D = B.sort_values("auroc").reset_index(drop=True)
    yy = np.arange(len(D))
    sig = np.zeros(len(D), bool)
    if fw is not None:
        sig = np.array([fw.get(b, 1) < .05 for b in D.block])
    ax.hlines(yy, D.auroc_lo, D.auroc_hi, color=GREY, lw=.9)
    ax.scatter(D.auroc[~sig], yy[~sig], s=22, color="white", edgecolor=TD, zorder=3, lw=1.1)
    ax.scatter(D.auroc[sig], yy[sig], s=26, color=TD, zorder=3)
    ax.axvline(.5, color=GREY, ls=":", lw=.8)
    ax.set_yticks(yy); ax.set_yticklabels([pair_label(b) for b in D.block], fontsize=6.5)
    ax.set_xlabel("held-out AUROC of the pair's contribution")
    ax.legend([Line2D([], [], marker="o", ls="", color=TD),
               Line2D([], [], marker="o", ls="", mfc="white", mec=TD)],
              ["familywise $p<0.05$", "not distinguished"], frameon=False, fontsize=8.5,
              loc="lower right")
    panel(ax, "b", "Each system pair on its own")

    # c: replication across independent halves of the cohort, against the permutation reference
    ax = axes[1, 0]
    sh = read("interpretation_splithalf.csv")
    if sh is not None:
        levels = ["connection", "region", "system pair"]
        views = ["correlation", "tangent (static)", "dynamic: mean"]
        cols = [TD, ASD, ACC]
        S = sh[sh.quantity == "struct"]
        w = .26
        for k, (vn, col) in enumerate(zip(views, cols)):
            vals = [S[(S.view == vn) & (S.level == lv)].observed_rho.iloc[0] for lv in levels]
            ax.bar(np.arange(3) + (k - 1) * w, vals, w, color=col, alpha=.88,
                   label={"correlation": "correlation", "tangent (static)": "static tangent",
                          "dynamic: mean": "windowed tangent"}[vn])
            hi = [S[(S.view == vn) & (S.level == lv)].null_p95.iloc[0] for lv in levels]
            for x, h in zip(np.arange(3) + (k - 1) * w, hi):
                ax.plot([x - w / 2, x + w / 2], [h, h], color="k", lw=1.4)
        ax.set_xticks(range(3)); ax.set_xticklabels(["connections", "regions", "system pairs"])
        ax.set_ylabel("Spearman $\\rho$ between independent halves")
        ax.axhline(0, color=GREY, lw=.6)
        ax.set_ylim(0, 1.0)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left",
                  title="black ticks: permutation 95th percentile", title_fontsize=8)
    panel(ax, "c", "Replication across independent halves")

    # d: deletion and keep-only curves
    ax = axes[1, 1]
    if dele is not None:
        d = dele[(dele.protocol == "pooled") & (dele.view == "mean of views")]
        for mode, col, ls, nm in (("delete top", ASD, "-", "delete top-ranked"),
                                  ("keep only top", TD, "-", "keep only top-ranked")):
            r = d[d["mode"] == mode].sort_values("n_pairs")
            ax.plot(r.n_pairs, r.auroc, ls, color=col, lw=1.6, marker="o", ms=3.5, label=nm)
            ax.fill_between(r.n_pairs, r.random_lo, r.random_hi, color=col, alpha=.15, lw=0)
            ax.plot(r.n_pairs, r.random_mean, "--", color=col, lw=.9)
        ax.set_xlabel("number of system pairs deleted or kept (of 36)")
        ax.set_ylabel("held-out AUROC")
        ax.legend(frameon=False, fontsize=8.5, loc="lower center",
                  title="dashed and shaded: random pairs", title_fontsize=8)
    panel(ax, "d", "Do the ranked pairs matter to the trained model?")
    return save(fig, out, "fig_interpretation")


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    return out / f"{name}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "latest_study" / "figures"))
    a = ap.parse_args()
    style()
    out = Path(a.out)
    for f in (fig_calibration, fig_interpretation):
        p = f(out)
        print(f"{f.__name__}: {p if p else 'skipped (inputs missing)'}")


if __name__ == "__main__":
    main()
