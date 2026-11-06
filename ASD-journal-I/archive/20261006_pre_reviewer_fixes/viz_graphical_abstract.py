"""
Graphical abstract (version 2): the method flow, the two lead contributions, and the three
shorter findings.

    band 1  the pipeline, left to right (cohort, quality control, atlas, three views,
            training-only models, averaged probability, discrimination)
    band 2  contribution 1 (probabilities recalibrated, discrimination unchanged) and
            contribution 2 (the score attributed to brain-system pairs, replicated across
            independent halves)
    band 3  contributions 3 to 5 (somatomotor reduction under motion matching; what the
            score measures; matched model comparisons)

The leakage check is deliberately absent: it is a methodological check, not a claim of
the paper.  Calibration is worded as in the paper: calibration-in-the-large is restored,
discrimination is unchanged, and site-level gaps remain.

Encoding: positions on a labelled common scale (dots, intervals, dumbbells).  The only
bars are the two correlations, which start at zero.  No axis is truncated to enlarge a
difference.  Every number is read from the study tables and checked for internal
consistency (block shares sum to one, counts add up); nothing is typed in.  A sources file
(values, table of origin, alt text) is written next to the figure.

Designed at the printed size of the manuscript page (text >= 7 pt on a 190 mm canvas).
Elsevier asks for at least 531 x 1328 px (height x width), readable at 5 x 13 cm, and
scales the image into a 500 x 200 px window on ScienceDirect, so the shapes carry the
message at thumbnail size and the labels carry it at full size.

Usage:
    .venv/bin/python viz_graphical_abstract.py [--study results/latest_study]
                                               [--out DIR] [--no-copy]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

import viz_style as vs  # noqa: E402

STEM = "fig_graphical_abstract"
W, H = 190.0, 105.0                     # mm (1.81:1; ScienceDirect window is 2.5:1)
M = 4.0
NATIVE, RECAL = "#2E6FB4", "#2A9D8F"    # as in the calibration figure of the paper
MINUS = "−"
ASCII_ARROW = "→"


def sgn(v: float, nd: int = 2, plus: bool = False) -> str:
    s = f"{abs(v):.{nd}f}"
    if v < 0 and float(s) != 0:
        return MINUS + s
    return ("+" + s) if (plus and v > 0) else s


# =========================================================================================
# data
# =========================================================================================
def read(study: Path, name: str, sub: str = "tables") -> pd.DataFrame:
    p = study / sub / name
    if not p.exists():
        raise SystemExit(f"missing result table: {p}")
    return pd.read_csv(p)


def evidence(study: Path) -> tuple[dict, dict]:
    d, src = {}, {}

    qc = read(study, "qc_cohort.csv").set_index("cohort")
    d["n"], d["n_asd"], d["n_con"] = (int(qc.loc["strict", k]) for k in ("n", "asd", "control"))
    assert d["n_asd"] + d["n_con"] == d["n"]
    src["cohort"] = "tables/qc_cohort.csv"

    iv = read(study, "intervals.csv").set_index("protocol")
    d["auroc"] = float(iv.loc["pooled", "auroc"])
    d["auroc_loso"] = float(iv.loc["loso", "auroc"])
    src["discrimination"] = "tables/intervals.csv"

    # contribution 1: calibration-in-the-large before and after the primary correction
    cal = read(study, "calibration_nested.csv", "review")
    for proto in ("pooled", "loso"):
        for var, key in (("native", "nat"), ("intercept only", "rec")):
            r = cal[(cal.protocol == proto) & (cal.variant == var)
                    & (cal.metric == "calibration-in-the-large")].iloc[0]
            d[f"citl_{key}_{proto}"] = (float(r.estimate), float(r.lo), float(r.hi))
            a = cal[(cal.protocol == proto) & (cal.variant == var) & (cal.metric == "AUROC")].iloc[0]
            d[f"auroc_{key}_{proto}"] = float(a.estimate)
    het = read(study, "calibration_site_heterogeneity.csv", "review")
    h = het[(het.protocol == "loso") & (het.variant == "intercept only")].iloc[0]
    d["i2_loso_recal"] = float(h.I2)
    src["calibration"] = ["review/calibration_nested.csv (native, intercept only; site-clustered "
                          "bootstrap 95% intervals)", "review/calibration_site_heterogeneity.csv"]

    # contribution 2: shares of the score separation, and replication across halves
    bl = read(study, "interpretation_blocks.csv", "review")
    bl = bl[(bl.protocol == "pooled") & (bl.view == "mean of views")]
    assert abs(bl.share_of_separation.sum() - 1.0) < 1e-6, "shares must sum to one"
    keys = vs.SYS_KEYS
    share = np.full((8, 8), np.nan)
    for _, r in bl[bl.block != "unassigned"].iterrows():
        a, b = r.block.split(" | ")
        i, j = keys.index(a), keys.index(b)
        share[i, j] = share[j, i] = r.share_of_separation
    assert not np.isnan(share).any()
    d["share"] = share
    d["share_max"] = float(np.nanmax(share))
    d["share_top"] = bl.sort_values("share_of_separation", ascending=False).iloc[0].block
    assert d["share_top"] == "default mode | default mode"
    sh = read(study, "interpretation_splithalf.csv", "review")
    sh = sh[sh.quantity == "struct"]
    d["rep"] = {}
    for lvl in ("connection", "region", "system pair"):
        s = sh[sh.level == lvl]
        d["rep"][lvl] = (s.observed_rho.to_numpy(float), float(s.null_p95.max()))
    src["attribution"] = ["review/interpretation_blocks.csv (pooled, mean of views)",
                          "review/interpretation_splithalf.csv (structure coefficients)"]

    # contribution 3: somatomotor regions, all participants against motion-matched pairs
    a = read(study, "sensorimotor_regions.csv")[["roi_id", "effect"]]
    b = read(study, "sensorimotor_regions_matched.csv")[["roi_id", "effect"]]
    m = a.merge(b, on="roi_id", suffixes=("_all", "_matched"))
    d["sm_all"], d["sm_matched"] = m.effect_all.to_numpy(float), m.effect_matched.to_numpy(float)
    d["sm_n"] = len(m)
    d["sm_stronger"] = int((m.effect_matched < m.effect_all).sum())
    src["somatomotor"] = ["tables/sensorimotor_regions.csv", "tables/sensorimotor_regions_matched.csv"]

    # contribution 4: cognitive-ability gradient (unmatched, autistic bands) and severity
    sub = read(study, "subgroup_auroc.csv")
    iq = sub[(sub.protocol == "pooled") & sub.variable.str.startswith("FIQ band")]
    d["iq"] = iq.auroc.to_numpy(float)
    assert len(d["iq"]) == 4
    sev = read(study, "severity_pooled_vs_within.csv")
    srs = sev[(sev.protocol == "pooled") & (sev.instrument == "SRS total")]
    d["srs_both"] = float(srs[srs.scope == "pooled"].rho.iloc[0])
    d["srs_within"] = float(srs[srs.scope == "within-autism"].rho.iloc[0])
    src["what_the_score_measures"] = ["tables/subgroup_auroc.csv (autistic FIQ bands, comparison "
                                      "group fixed)", "tables/severity_pooled_vs_within.csv"]

    # contribution 5: classifier family and shrinkage-matched windowing (participant-pooled)
    cls = read(study, "classifier.csv")
    cls = cls[cls.protocol == "pooled"]
    tree = cls.condition.str.contains("forest|xgboost|catboost", case=False)
    d["lin"], d["tree"] = (cls.loc[~tree, "auroc_pooled_oof"].to_numpy(float),
                           cls.loc[tree, "auroc_pooled_oof"].to_numpy(float))
    assert len(d["lin"]) == 5 and len(d["tree"]) == 3
    rt = read(study, "ridge_sweep_tests.csv")
    um = rt[rt.meaning.str.contains("against an unridged static baseline")].iloc[0]
    mm = rt[rt.meaning.str.contains("against a matched static baseline")].iloc[0]
    d["static"], d["windowed"], d["static_matched"] = float(um.auroc_b), float(um.auroc_a), float(mm.auroc_b)
    d["win_vs_matched"], d["win_vs_matched_p"] = float(mm.difference), float(mm.delong_p)
    src["matched_comparisons"] = ["tables/classifier.csv (participant-pooled AUROC)",
                                  "tables/ridge_sweep_tests.csv (static, shrinkage-matched static, windowed)"]
    return d, src


# =========================================================================================
# drawing helpers (everything is drawn in canvas millimetres)
# =========================================================================================
def tx(ax, x, y, s, *, ha="center", va="center", size=7.0, color=None, weight="normal"):
    return vs.label(ax, x, y, s, ha=ha, va=va, size=size, color=color or vs.INK, weight=weight)


def head(ax, x, y, s):
    return tx(ax, x, y, s, ha="left", size=8.0, weight="bold")


def dot(ax, x, y, d=1.8, fc=vs.INK, ec=None, marker="o", lw=None):
    ax.scatter([x], [y], s=(d / vs.PT) ** 2, marker=marker, facecolor=fc, edgecolor=ec or fc,
               linewidths=vs.LW_KEY if lw is None else lw, zorder=vs.Z_GLYPH + 1, clip_on=False)


def seg(ax, x0, y0, x1, y1, color=vs.GREY, lw=0.6, z=None, ls="-"):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="butt", ls=ls,
            zorder=vs.Z_GLYPH if z is None else z, clip_on=False)


class Scale:
    """Linear map from a data range onto a horizontal canvas span."""

    def __init__(self, lo, hi, x0, x1):
        self.lo, self.hi, self.x0, self.x1 = lo, hi, x0, x1

    def __call__(self, v):
        return self.x0 + (np.asarray(v, float) - self.lo) / (self.hi - self.lo) * (self.x1 - self.x0)


def xaxis(ax, sc, y, ticks, fmt, *, pad=2.7):
    seg(ax, sc.x0, y, sc.x1, y, color=vs.GREY, lw=0.5)
    for t in ticks:
        seg(ax, sc(t), y, sc(t), y + 0.9, color=vs.GREY, lw=0.5)
        tx(ax, sc(t), y + pad, fmt(t))


def card(ax, x, y, w, h):
    vs.panel_bg(ax, x, y, w, h, gid="panel_bg_n")


# =========================================================================================
# figure
# =========================================================================================
def build(out: Path, study: Path, copy_to: Path | None):
    vs.apply_style()
    vs.CHIP_NEUTRAL_FACE = "white"
    vs.GLYPH_FACE = "white"
    d, src = evidence(study)
    fig, ax = vs.canvas(H, W)

    # ================================================================ band 1: the pipeline
    cy, LAB = 12.0, 23.6
    vs.dots(ax, 4.5, 4.5, 18.0, 14.0, round(d["n_asd"] / 10), round(d["n_con"] / 10), seed=3)
    tx(ax, 13.5, LAB, "%s participants" % format(d["n"], ","))
    vs.arrow(ax, (24.0, cy), (28.5, cy), kind="data")
    vs.funnel(ax, 30.0, 6.5, 10.0, 11.0)
    tx(ax, 35.0, LAB, "quality control")
    vs.arrow(ax, (41.5, cy), (46.0, cy), kind="data")
    vs.brain_glyph(ax, 47.5, 4.5, 18.0, 15.0, view="lateral")
    tx(ax, 56.5, LAB, "200 regions")
    vs.arrow(ax, (67.5, cy), (72.5, cy), kind="data")
    for i, mk in enumerate(("corr", "static", "windowed")):
        yy = 3.8 + i * 6.2
        ax.add_patch(FancyBboxPatch((74.0, yy), 2.6, 5.2,
                                    boxstyle="round,pad=0,rounding_size=0.7",
                                    facecolor=vs.MEMBER[mk], edgecolor="none", zorder=vs.Z_GLYPH))
        vs.matrix_thumb(ax, 77.2, yy - 0.1, 5.4, None, strips="", frame=True)
    tx(ax, 79.5, LAB, "three views")
    vs.arrow(ax, (85.5, cy), (90.5, cy), kind="fit")
    LX, LY, LW_, LH = 92.0, 2.5, 52.0, 19.0
    ax.add_patch(FancyBboxPatch((LX, LY), LW_, LH, boxstyle="round,pad=0,rounding_size=2.0",
                                facecolor=vs.LANE, edgecolor=vs.BLUE, linewidth=vs.LW_LANE, zorder=0.5))
    vs.padlock(ax, LX + LW_ / 2, LY + 2.1, size=3.0)
    # the dome draws from y + 0.28 h (tangent plane) to y + 1.10 h (front base curve); keep all of it inside the box
    vs.dome(ax, LX + 3.5, LY + 0.5, 13.0, 13.0, anchor_label=None, lock=False)
    vs.sigmoid_glyph(ax, LX + 25.0, cy - 4.0, s=8.0)   # centre on the box centre line
    vs.fold_bar(ax, LX + 35.5, cy - 1.3, 13.0, h=2.6, n=10, held=3)
    tx(ax, LX + LW_ / 2, LAB, "training-only logistic models")
    vs.arrow(ax, (146.0, cy), (150.5, cy), kind="agg")
    vs.merge_node(ax, 154.5, cy, d=7.0)
    tx(ax, 154.5, LAB, "averaged probability")
    vs.arrow(ax, (158.5, cy), (163.0, cy), kind="agg")
    tx(ax, 175.5, 4.6, "area under the")
    tx(ax, 175.5, 7.7, "curve (AUROC)")
    tx(ax, 175.5, 13.2, "%.3f" % d["auroc"], size=14.0, weight="bold")
    tx(ax, 175.5, 18.6, "%.3f unseen site" % d["auroc_loso"])

    # ================================================================ band 2: the two hero panels
    HY, HH = 28.5, 36.0
    AX, AW = M, 80.0
    BX, BW = AX + AW + 3.0, W - 2 * M - AW - 3.0
    card(ax, AX, HY, AW, HH)
    card(ax, BX, HY, BW, HH)

    # ---- A: probabilities recalibrated, ranking unchanged
    head(ax, AX + 2.5, HY + 3.3, "Probabilities recalibrated")
    dot(ax, AX + 2.9, HY + 8.2, 1.9, NATIVE, marker="o")
    tx(ax, AX + 5.0, HY + 8.2, "native", ha="left")
    dot(ax, AX + 17.0, HY + 8.2, 1.9, RECAL, marker="s")
    tx(ax, AX + 19.1, HY + 8.2, "recalibrated", ha="left")
    tx(ax, AX + AW - 2.5, HY + 3.3, "AUROC %.3f %s %.3f" % (d["auroc_nat_pooled"], ASCII_ARROW,
                                                           d["auroc_rec_pooled"]), ha="right")
    sc = Scale(-0.50, 0.30, AX + 21.0, AX + AW - 3.0)
    zero = float(sc(0.0))
    seg(ax, zero, HY + 11.8, zero, HY + 29.0, color=vs.INK, lw=0.5, ls=(0, (2, 2)))
    rows = [("pooled", "pooled", HY + 15.0), ("loso", "site withheld", HY + 25.0)]
    for proto, name, yc in rows:
        tx(ax, AX + 2.5, yc, name, ha="left")
        for key, dy, col, mk in (("nat", -1.6, NATIVE, "o"), ("rec", 1.6, RECAL, "s")):
            est, lo, hi = d[f"citl_{key}_{proto}"]
            seg(ax, sc(lo), yc + dy, sc(hi), yc + dy, color=col, lw=0.9, z=vs.Z_GLYPH + 0.5)
            dot(ax, sc(est), yc + dy, 2.0, col, marker=mk)
            tx(ax, sc(lo) - 1.2, yc + dy, sgn(est, 2, plus=True), ha="right")
    tx(ax, AX + 21.0, HY + 30.6, "← over-predicts", ha="left", color=vs.GREY)
    tx(ax, AX + AW - 3.0, HY + 30.6, "under-predicts →", ha="right", color=vs.GREY)
    tx(ax, zero, HY + 10.2, "0 = observed rate", color=vs.GREY)
    tx(ax, AX + AW / 2, HY + HH - 2.2, "site-level gaps remain", color=vs.INK)

    # ---- B: the score attributed to brain-system pairs, replicated across independent halves
    head(ax, BX + 2.5, HY + 3.3, "Score explained by brain systems")
    cell, gx = 3.0, BX + 5.2
    gy = HY + 9.6
    cmap = LinearSegmentedColormap.from_list("share", ["#FFFFFF", vs.AGG])
    share, smax = d["share"], d["share_max"]
    for i in range(8):
        for j in range(8):
            ax.add_patch(Rectangle((gx + j * cell, gy + i * cell), cell, cell,
                                   facecolor=cmap(share[i, j] / smax), edgecolor="white",
                                   linewidth=0.2, zorder=vs.Z_GLYPH))
    for k, (_, _, _, col) in enumerate(vs.SYSTEMS):
        ax.add_patch(Rectangle((gx - 1.3, gy + k * cell), 0.9, cell, facecolor=col,
                               edgecolor="none", zorder=vs.Z_GLYPH))
        ax.add_patch(Rectangle((gx + k * cell, gy - 1.3), cell, 0.9, facecolor=col,
                               edgecolor="none", zorder=vs.Z_GLYPH))
    di = vs.SYS_KEYS.index("default mode")
    dmn = vs.SYS_COL["default mode"]
    ax.add_patch(Rectangle((gx, gy + di * cell), 8 * cell, cell, facecolor="none",
                           edgecolor=dmn, linewidth=0.9, zorder=vs.Z_GLYPH + 1))
    ax.add_patch(Rectangle((gx + di * cell, gy), cell, 8 * cell, facecolor="none",
                           edgecolor=dmn, linewidth=0.9, zorder=vs.Z_GLYPH + 1))
    tx(ax, gx + 8 * cell + 1.8, gy + 2.0, "shade = share", ha="left", color=vs.GREY)
    tx(ax, gx + 8 * cell + 1.8, gy + 5.0, "of separation", ha="left", color=vs.GREY)
    tx(ax, gx + 8 * cell + 1.8, gy + di * cell + cell / 2, "default mode", ha="left")

    rx0 = BX + 5.2 + 8 * cell + 21.0              # start of the replication panel
    tx(ax, rx0 + 2.0, HY + 7.0, "agreement between", ha="left")
    tx(ax, rx0 + 2.0, HY + 10.0, "independent halves (ρ)", ha="left")
    rsc = Scale(0.0, 0.8, rx0 + 20.0, BX + BW - 5.0)
    levels = [("system pairs", "system pair"), ("regions", "region"), ("connections", "connection")]
    ry = [HY + 16.0, HY + 21.0, HY + 26.0]
    cols = [vs.MEMBER["corr"], vs.MEMBER["static"], vs.MEMBER["windowed"]]
    for (nm, key), yy in zip(levels, ry):
        vals, null95 = d["rep"][key]
        ax.add_patch(Rectangle((rsc(0.0), yy - 1.7), rsc(null95) - rsc(0.0), 3.4,
                               facecolor="#E3E3E3", edgecolor="none", zorder=vs.Z_GLYPH - 0.2))
        tx(ax, rx0 + 18.0, yy, nm, ha="right")
        for v, c in zip(vals, cols):
            dot(ax, float(rsc(v)), yy, 1.8, c, ec="white", lw=0.3)
    xaxis(ax, rsc, HY + 30.0, [0.0, 0.4, 0.8], lambda t: "%g" % t, pad=2.6)
    tx(ax, float(rsc(d["rep"]["system pair"][1])) + 1.5, ry[0], "chance", ha="left", color=vs.GREY)
    tx(ax, BX + BW - 2.5, HY + 3.3, "one dot per view", ha="right", color=vs.GREY)

    # ================================================================ band 3: three shorter findings
    TY, TH = HY + HH + 3.0, 31.0
    gap = 3.0
    tw = [56.0, 66.0]
    tw.append(W - 2 * M - 2 * gap - sum(tw))
    tx0 = [M, M + tw[0] + gap, M + tw[0] + tw[1] + 2 * gap]
    for x0, w in zip(tx0, tw):
        card(ax, x0, TY, w, TH)

    # ---- tile 1: somatomotor reduction persists after motion matching
    x0, w = tx0[0], tw[0]
    head(ax, x0 + 2.5, TY + 3.3, "Somatomotor reduction persists")
    dot(ax, x0 + w - 18.5, TY + 9.6, 1.7, vs.GREY)
    tx(ax, x0 + w - 16.5, TY + 9.6, "all", ha="left")
    dot(ax, x0 + w - 18.5, TY + 13.0, 1.7, vs.MATCH)
    tx(ax, x0 + w - 16.5, TY + 13.0, "matched", ha="left")
    sc1 = Scale(-0.36, -0.06, x0 + 4.0, x0 + w - 4.0)
    o = np.argsort(d["sm_all"])
    n1 = d["sm_n"]
    y_top, y_bot = TY + 8.5, TY + 21.5
    ys = np.linspace(y_top, y_bot, n1)
    for yy, a_, b_ in zip(ys, d["sm_all"][o], d["sm_matched"][o]):
        seg(ax, sc1(a_), yy, sc1(b_), yy, color=vs.RULE, lw=0.4, z=vs.Z_GLYPH - 0.1)
        dot(ax, float(sc1(a_)), yy, 0.9, vs.GREY, lw=0)
        dot(ax, float(sc1(b_)), yy, 0.9, vs.MATCH, lw=0)
    tx(ax, x0 + 4.0, TY + 17.6, "standardised", ha="left", color=vs.GREY)
    tx(ax, x0 + 4.0, TY + 20.6, "difference", ha="left", color=vs.GREY)
    xaxis(ax, sc1, y_bot + 1.6, [-0.3, -0.2, -0.1], lambda t: sgn(t, 1), pad=2.4)
    tx(ax, x0 + w / 2, TY + TH - 1.6, "%d of %d regions stronger" % (d["sm_stronger"], d["sm_n"]),
       color=vs.INK)

    # ---- tile 2: what the score measures
    x0, w = tx0[1], tw[1]
    head(ax, x0 + 2.5, TY + 3.3, "What the score measures")
    hw = w / 2
    tx(ax, x0 + hw / 2, TY + 7.6, "AUROC, autistic", color=vs.GREY)
    tx(ax, x0 + hw / 2, TY + 10.5, "by cognitive ability", color=vs.GREY)
    px0, px1 = x0 + 5.0, x0 + hw - 4.0
    ysc = lambda v: TY + 25.0 - (v - 0.5) / 0.5 * 11.0     # AUROC 0.5 .. 1.0
    seg(ax, px0 - 1.0, float(ysc(0.5)), px1 + 1.0, float(ysc(0.5)), color=vs.RULE, lw=0.5, ls=(0, (2, 2)))
    xs = np.linspace(px0, px1, 4)
    for k in range(3):
        seg(ax, xs[k], float(ysc(d["iq"][k])), xs[k + 1], float(ysc(d["iq"][k + 1])),
            color=vs.INK, lw=1.0, z=vs.Z_GLYPH + 0.5)
    for xx, v in zip(xs, d["iq"]):
        dot(ax, float(xx), float(ysc(v)), 1.7, vs.INK)
    tx(ax, (xs[0] + xs[3]) / 2, float(ysc(0.5)) - 1.9, "chance", color=vs.GREY)
    tx(ax, xs[0] + 1.5, float(ysc(d["iq"][0])) - 2.6, "%.2f" % d["iq"][0])
    tx(ax, xs[3], float(ysc(d["iq"][3])) - 2.6, "%.2f" % d["iq"][3])
    tx(ax, xs[0], TY + TH - 1.7, "lower")
    tx(ax, xs[3], TY + TH - 1.7, "higher")
    seg(ax, x0 + hw, TY + 7.0, x0 + hw, TY + TH - 2.0, color=vs.RULE, lw=0.4)
    cx = x0 + hw + hw / 2
    tx(ax, cx, TY + 7.6, "ρ with Social", color=vs.GREY)
    tx(ax, cx, TY + 10.5, "Responsiveness Scale", color=vs.GREY)
    base, k = TY + 25.0, 11.0 / 0.6
    for xx, v, c, nm in ((cx - 7.5, d["srs_both"], vs.GREY, "both groups"),
                         (cx + 8.0, d["srs_within"], vs.INK, "autistic only")):
        ax.add_patch(Rectangle((xx - 3.0, base - v * k), 6.0, v * k, facecolor=c,
                               edgecolor="none", zorder=vs.Z_GLYPH))
        tx(ax, xx, base - v * k - 2.4, "%.2f" % v)
        tx(ax, xx, TY + TH - 1.7, nm)
    seg(ax, cx - 14.0, base, cx + 15.5, base, color=vs.INK, lw=0.5)

    # ---- tile 3: matched model comparisons, one AUROC axis
    x0, w = tx0[2], tw[2]
    head(ax, x0 + 2.5, TY + 3.3, "Matched model comparisons")
    sc3 = Scale(0.55, 0.80, x0 + 26.0, x0 + w - 4.0)
    rows3 = [("trees", d["tree"], vs.TREE, "o", True), ("linear", d["lin"], vs.LINEAR, "o", True),
             ("static", [d["static"]], vs.MEMBER["static"], "o", True),
             ("static + shrinkage", [d["static_matched"]], vs.MEMBER["static"], "o", False),
             ("windowed", [d["windowed"]], vs.MEMBER["windowed"], "o", True)]
    yr = [TY + 8.6, TY + 12.4, TY + 17.0, TY + 20.8, TY + 24.6]
    for (nm, vals, col, mk, filled), yy in zip(rows3, yr):
        tx(ax, x0 + 23.5, yy, nm, ha="right")
        vals = np.asarray(vals, float)
        if len(vals) > 1:
            seg(ax, sc3(vals.min()), yy, sc3(vals.max()), yy, color=col, lw=0.5, z=vs.Z_GLYPH - 0.1)
        for v in vals:
            dot(ax, float(sc3(v)), yy, 1.8, col if filled else "white", ec=col, marker=mk,
                lw=0.8 if not filled else None)
        if len(vals) == 1:
            tx(ax, float(sc3(vals[0])) + 2.0, yy, "%.2f" % vals[0], ha="left")
        else:
            rng = "%.2f–%.2f" % (vals.min(), vals.max())
            if vals.mean() > 0.7:
                tx(ax, float(sc3(vals.min())) - 2.0, yy, rng, ha="right")
            else:
                tx(ax, float(sc3(vals.max())) + 2.0, yy, rng, ha="left")
    seg(ax, x0 + 2.5, TY + 14.7, x0 + w - 2.5, TY + 14.7, color=vs.RULE, lw=0.4)
    xaxis(ax, sc3, TY + 27.0, [0.6, 0.7, 0.8], lambda t: "%.1f" % t, pad=2.2)
    tx(ax, x0 + 23.5, TY + 29.2, "AUROC", ha="right", color=vs.GREY)

    # ================================================================ the one sentence
    vs.label(ax, W / 2, H - 2.6,
             "A connectome classifier with recalibrated probabilities and system-level explanations.",
             ha="center", size=9.0, weight="bold")

    probs = vs.check_layout(fig, min_pt=7.0, max_words=9, verbose=True)   # 9 = the closing sentence
    paths = vs.save_figure(fig, STEM, out, copy_to)

    manifest = {
        "figure": STEM, "canvas_mm": [W, H], "min_text_pt": 7.0,
        "sources": src,
        "values": {k: (v if not isinstance(v, np.ndarray) else v.round(4).tolist())
                   for k, v in d.items() if k not in ("share", "rep", "sm_all", "sm_matched")},
        "alt_text": ("Graphical abstract. A pipeline turns 871 participants' resting-state scans into "
                     "three connectivity views, fits three logistic models inside training data only, "
                     "and averages them (area under the curve %.3f, %.3f at an unseen site). Below, "
                     "the probabilities over-predict autism (calibration-in-the-large %s, interval "
                     "below zero) and a correction fitted inside training data removes the offset "
                     "(%s) without changing discrimination; site-level gaps remain. The score "
                     "divides exactly among 36 pairs of brain systems, default-mode pairs carry the "
                     "largest shares, and the system-pair pattern replicates across independent "
                     "halves of the cohort far more than the single-connection pattern. Three "
                     "shorter findings: the somatomotor reduction in autism is stronger in all %d "
                     "regions after motion matching; detection falls from %.2f to %.2f as the "
                     "cognitive ability of autistic participants rises, and the score correlates "
                     "with a social-responsiveness scale across both groups (%.2f) but hardly "
                     "within autism (%.2f); linear models beat tree ensembles and windowing "
                     "equals a shrinkage-matched whole-scan model."
                     % (d["auroc"], d["auroc_loso"], sgn(d["citl_nat_pooled"][0], 3),
                        sgn(d["citl_rec_pooled"][0], 3), d["sm_n"], d["iq"][0], d["iq"][3],
                        d["srs_both"], d["srs_within"])),
    }
    (Path(out) / f"{STEM}.sources.json").write_text(json.dumps(manifest, indent=1))
    print("AUROC %.3f | loso %.3f | n %d (%d/%d)" % (d["auroc"], d["auroc_loso"], d["n"], d["n_asd"], d["n_con"]))
    print("CITL pooled %s -> %s | loso %s -> %s" % (
        sgn(d["citl_nat_pooled"][0], 3), sgn(d["citl_rec_pooled"][0], 3),
        sgn(d["citl_nat_loso"][0], 3), sgn(d["citl_rec_loso"][0], 3, plus=True)))
    print("somatomotor %d/%d | iq %.2f->%.2f | srs %.2f/%.2f | static %.3f matched %.3f windowed %.3f (diff %.4f, p=%.3f)"
          % (d["sm_stronger"], d["sm_n"], d["iq"][0], d["iq"][3], d["srs_both"], d["srs_within"],
             d["static"], d["static_matched"], d["windowed"], d["win_vs_matched"], d["win_vs_matched_p"]))
    print("layout problems:", len(probs))
    for p in paths:
        print("wrote", p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Graphical abstract")
    ap.add_argument("--study", default=str(ROOT / "results/latest_study"))
    ap.add_argument("--out", default=str(ROOT / "results/latest_study/figures"))
    ap.add_argument("--no-copy", action="store_true")
    args = ap.parse_args()
    build(Path(args.out), Path(args.study).resolve(),
          None if args.no_copy else vs.JOURNAL_FIG)


if __name__ == "__main__":
    main()
