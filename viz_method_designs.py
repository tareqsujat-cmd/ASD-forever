"""
Methods figure 3: evaluation and analysis designs (fig_designs).

Five schematic panels that follow ASD-journal-I/sections/method.tex, drawn with the shared
glyph system in viz_style.py (redesign brief, section 2 "Figure 3: designs"):

  (a) nested cross-validation and fold-level fitting (sec:m-4-1 to m-4-3, m-2-3, m-2-4,
      m-3): grey label-free pre-split zone, split line, blue training lane with steps
      1-5, vermilion-outlined held-out lane;
  (b) feature-selection scope (sec:m-6-7): paired connection-level strips (reference vs
      global selection) and the region-level label scope with the top-k ticks;
  (c) motion-matched group contrast (sec:m-7-1 to m-7-4, m-8-1, m-8-3);
  (d) cognitive-subgroup designs on one shared IQ axis (sec:m-6-9);
  (e) classifier family (sec:m-6-3).

No results are shown.  Real data enter as inputs only: the strict-cohort class counts
and site sizes (abide_processed/mri/metadata.csv; dot cloud and leave-one-site-out fold
widths) and the CC200 atlas with its system table (atlas/, brain glyph and system strips).
Participant markers: filled = autistic, open = non-autistic.  In (a) one marker stands
for about ten participants; markers in (c)-(e) and the IQ values in (d) are schematic.

The page is drawn at its printed size (165 mm wide) on one mm canvas (origin top-left).

Usage:
    .venv/bin/python viz_method_designs.py [--out results/latest_study/figures] [--no-copy]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

import viz_style as vs

ROOT = Path(__file__).resolve().parent
STEM = "fig_designs"
H_MM = 178.0                                  # printed height (limit 195 mm)
XC = vs.GRID.x(6)                             # left edge of panel (c): 84 mm
DASHED = (0, (2.2, 1.6))                      # dashed outline of the inner-CV inset


# ----------------------------------------------------------------------------- data
def cohort_counts() -> tuple[int, int]:
    """Autistic / non-autistic counts of the strict cohort (metadata.csv, label 1/0)."""
    with open(vs.META, newline="") as f:
        lab = [int(r["label"]) for r in csv.DictReader(f)]
    n_aut, n_non = sum(v == 1 for v in lab), sum(v == 0 for v in lab)
    assert (n_aut, n_non) == (403, 468), (n_aut, n_non)
    return n_aut, n_non


def greedy_iq_match(aut, non, caliper=5):
    """Sec:m-6-9: autistic IQ ascending, closest free partner, |diff| <= caliper, greedy
    without replacement (one site).  Returns {autistic IQ: partner IQ}."""
    free, pairs = list(non), {}
    for a in sorted(aut):
        cand = sorted((abs(a - n), n) for n in free)
        assert len(cand) < 2 or cand[0][0] != cand[1][0], "tie in the schematic example"
        if cand and cand[0][0] <= caliper:
            pairs[a] = cand[0][1]
            free.remove(cand[0][1])
    return pairs


# schematic IQ values shared by the three designs in (d) (one site)
IQ_AUT = [78, 89, 96, 103, 109, 116, 124, 134]
IQ_NON = [86, 91, 99, 102, 107, 112, 119, 127, 138]


# ----------------------------------------------------------------------------- glyphs
def zone(ax, x, y, w, h, gid="presplit_zone"):
    """Grey pre-split zone (label-free, before the outer split)."""
    with vs.group(ax, gid):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={vs.RADIUS}",
                                    facecolor="white", edgecolor=vs.RULE,
                                    linewidth=vs.LW_KEY, zorder=vs.Z_LANE))
    return vs.Box(x, y, w, h)


def split_line(ax, x, y0, y1, text=None, gid="split_line"):
    """The outer split: a solid ink rule, optionally labelled above."""
    with vs.group(ax, gid):
        ax.add_line(Line2D([x, x], [y0, y1], color=vs.INK, lw=vs.LW_ARROW,
                           zorder=vs.Z_ARROW, solid_capstyle="butt"))
        if text:
            vs.label(ax, x, y0 - 1.6, text, ha="center", va="center")
    return vs.Box(x, y0, 0, y1 - y0)


def _bg(ax, x, y, s):
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle=f"round,pad=0,rounding_size={vs.RADIUS}",
                                facecolor="white", edgecolor="none", zorder=vs.Z_GLYPH))


def mini_anchor(ax, x, y, w=8.0, h=6.0, gid="anchor_mini"):
    """Small SPD dome with the tangent plane and the blue anchor (grammar of step 1)."""
    with vs.group(ax, gid):
        cx, yb = x + w / 2, y + h
        a, b = 0.42 * w, 0.62 * h
        t = np.linspace(np.pi, 0, 50)
        ax.add_patch(Polygon(np.c_[cx + a * np.cos(t), yb - b * np.sin(t)], closed=True,
                             facecolor=vs.TINT, edgecolor=vs.GREY, linewidth=vs.LW_KEY,
                             zorder=vs.Z_GLYPH))
        ya = yb - b
        pw, pd, sk = 0.95 * w, 0.26 * h, 0.08 * w
        plane = [[cx - pw / 2 + sk, ya - pd / 2], [cx + pw / 2 + sk, ya - pd / 2],
                 [cx + pw / 2 - sk, ya + pd / 2], [cx - pw / 2 - sk, ya + pd / 2]]
        ax.add_patch(Polygon(plane, closed=True, facecolor=vs.LANE, edgecolor=vs.BLUE,
                             linewidth=vs.LW_KEY, alpha=0.92, zorder=vs.Z_GLYPH + 0.2))
        ax.add_patch(Circle((cx, ya), 0.65, facecolor=vs.BLUE, edgecolor="white",
                            linewidth=0.3, zorder=vs.Z_GLYPH + 0.5))
    return vs.Box(x, y, w, h)


def held_cell(ax, x, y, w, h, gid="heldout_cell"):
    """Small held-out fold: white, vermilion outline, three participant markers."""
    with vs.group(ax, gid):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={vs.RADIUS}",
                                    facecolor="white", edgecolor=vs.VERM,
                                    linewidth=vs.LW_LANE, zorder=vs.Z_LANE))
        for k, f in enumerate((True, False, True)):
            vs._dot(ax, x + w / 2 + (k - 1) * 2.0, y + h / 2, f, 1.3)
    return vs.Box(x, y, w, h)


def inner_folds(ax, x, y, w, h, n=3, held=1, gid="inner_cv_folds"):
    """Inner stratified CV inside the training partition (as in Figure 1): training cells
    in lane tint, the inner validation cell in solid blue (vermilion stays reserved for
    the outer held-out fold and leakage)."""
    with vs.group(ax, gid):
        cw = w / n
        for i in range(n):
            ax.add_patch(Rectangle((x + i * cw, y), cw, h,
                                   facecolor=vs.BLUE if i == held else vs.LANE,
                                   edgecolor="none", zorder=vs.Z_GLYPH))
        for i in range(1, n):
            ax.add_line(Line2D([x + i * cw] * 2, [y, y + h], color="white",
                               lw=vs.LW_MIN * 1.6, zorder=vs.Z_GLYPH + 0.1))
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=vs.BLUE,
                               linewidth=vs.LW_KEY, zorder=vs.Z_GLYPH + 0.2))
    return vs.Box(x, y, w, h)


def rank_glyph(ax, x, y, w=8.0, h=6.0, n=6, gid="rank"):
    """Ranked parcels: descending bars."""
    with vs.group(ax, gid):
        bw = w / n
        for k in range(n):
            hk = h * (1 - 0.15 * k)
            ax.add_patch(Rectangle((x + k * bw + 0.15 * bw, y + h - hk), 0.7 * bw, hk,
                                   facecolor=vs.GREY, edgecolor="none", zorder=vs.Z_GLYPH))
    return vs.Box(x, y, w, h)


def triangle_glyph(ax, x, y, size, gid="system_pairs"):
    """8 x 8 system grid with the 36 within- and between-system pairs (upper triangle,
    diagonal included) and system colour strips."""
    with vs.group(ax, gid):
        c = size / 8
        for r in range(8):
            for k in range(8):
                if k >= r:
                    ax.add_patch(Rectangle((x + k * c, y + r * c), c, c, facecolor="#D9D9D9",
                                           edgecolor="white", linewidth=vs.LW_MIN,
                                           zorder=vs.Z_GLYPH))
        vs.network_strip(ax, x - 1.2, y, size, "v", None, 0.8, gid="tri_strip_left")
        vs.network_strip(ax, x, y - 1.2, size, "h", None, 0.8, gid="tri_strip_top")
    return vs.Box(x, y, size, size)


def _points(ax, x, y, s, pts, cls):
    for (u, v), f in zip(pts, cls):
        vs._dot(ax, x + u * s, y + v * s, f, 0.95)


def ridge_glyph(ax, x, y, s=6.0, gid="ridge"):
    """Ridge classifier: least-squares line fitted to the class labels (rows 1 and 0)."""
    with vs.group(ax, gid):
        _bg(ax, x, y, s)
        _points(ax, x, y, s, [(.62, .26), (.84, .26), (.16, .76), (.38, .76)],
                [True, True, False, False])
        ax.add_line(Line2D([x + .10 * s, x + .90 * s], [y + .64 * s, y + .38 * s], color=vs.INK,
                           lw=vs.LW_ARROW, zorder=vs.Z_GLYPH + 0.6))
    return vs.Box(x, y, s, s)


def svc_glyph(ax, x, y, s=6.0, gid="linear_svc"):
    """Linear SVC: separating line with its two margins."""
    with vs.group(ax, gid):
        _bg(ax, x, y, s)
        _points(ax, x, y, s, [(.16, .60), (.30, .82), (.12, .82), (.62, .20), (.84, .34),
                              (.84, .14)], [True, True, True, False, False, False])
        for (u0, v0, u1, v1), col, ls in (((.10, .10, .90, .90), vs.INK, "-"),
                                          ((.22, .08, .92, .78), vs.GREY, (0, (1.5, 1.2))),
                                          ((.08, .22, .78, .92), vs.GREY, (0, (1.5, 1.2)))):
            ax.add_line(Line2D([x + u0 * s, x + u1 * s], [y + v0 * s, y + v1 * s], color=col,
                               lw=vs.LW_KEY if col == vs.GREY else vs.LW_ARROW, ls=ls,
                               zorder=vs.Z_GLYPH + 0.6))
    return vs.Box(x, y, s, s)


def lda_glyph(ax, x, y, s=6.0, gid="pca_lda"):
    """PCA + shrinkage LDA: two principal axes and a linear boundary."""
    with vs.group(ax, gid):
        _bg(ax, x, y, s)
        ax.add_line(Line2D([x + .12 * s, x + .88 * s], [y + .80 * s, y + .24 * s],
                           color=vs.GREY, lw=vs.LW_KEY, zorder=vs.Z_GLYPH + 0.3))  # PC1
        _points(ax, x, y, s, [(.22, .62), (.34, .76), (.16, .80), (.62, .30), (.82, .40),
                              (.78, .16)], [True, True, True, False, False, False])
        ax.add_line(Line2D([x + .28 * s, x + .72 * s], [y + .22 * s, y + .82 * s], color=vs.INK,
                           lw=vs.LW_ARROW, zorder=vs.Z_GLYPH + 0.6))          # boundary
    return vs.Box(x, y, s, s)


def spectral_glyph(ax, x, y, s=6.0, gid="lowrank_spectral"):
    """Low-rank spectral: components ordered by |eigenvalue|, the leading ones kept."""
    with vs.group(ax, gid):
        _bg(ax, x, y, s)
        n = 6
        bw = .7 * s / n
        for k in range(n):
            hk = .66 * s * (0.82 ** (1.6 * k))
            ax.add_patch(Rectangle((x + .15 * s + k * bw + .1 * bw, y + .84 * s - hk), .8 * bw, hk,
                                   facecolor=vs.INK if k < 2 else vs.RULE, edgecolor="none",
                                   zorder=vs.Z_GLYPH + 0.3))
    return vs.Box(x, y, s, s)


def _tree(ax, x, y, w, h):
    """Mini decision tree (root at top), top-left (x, y), w x h mm."""
    lv = [[(x + w / 2, y)], [(x + w / 4, y + h / 2), (x + 3 * w / 4, y + h / 2)],
          [(x + w / 8, y + h), (x + 3 * w / 8, y + h), (x + 5 * w / 8, y + h),
           (x + 7 * w / 8, y + h)]]
    for k in range(2):
        for i, p in enumerate(lv[k]):
            for q in lv[k + 1][2 * i:2 * i + 2]:
                ax.add_line(Line2D([p[0], q[0]], [p[1], q[1]], color=vs.GREY, lw=vs.LW_KEY,
                                   zorder=vs.Z_GLYPH + 0.3))
    for row in lv:
        for p in row:
            ax.add_patch(Circle(p, 0.28, facecolor=vs.INK, edgecolor="none",
                                zorder=vs.Z_GLYPH + 0.5))


def forest_glyph(ax, x, y, s=6.0, gid="random_forest"):
    """Random forest: independent trees side by side."""
    with vs.group(ax, gid):
        _bg(ax, x, y, s)
        for k in range(3):
            _tree(ax, x + .08 * s + k * .29 * s, y + .30 * s, .26 * s, .42 * s)
    return vs.Box(x, y, s, s)


def boost_glyph(ax, x, y, s=6.0, gid="boosting"):
    """Gradient boosting: trees fitted in sequence (tree, plus, tree)."""
    with vs.group(ax, gid):
        _bg(ax, x, y, s)
        _tree(ax, x + .10 * s, y + .30 * s, .30 * s, .42 * s)
        _tree(ax, x + .60 * s, y + .30 * s, .30 * s, .42 * s)
        c = (x + .5 * s, y + .51 * s)
        for d in ((.07 * s, 0), (0, .07 * s)):
            ax.add_line(Line2D([c[0] - d[0], c[0] + d[0]], [c[1] - d[1], c[1] + d[1]],
                               color=vs.INK, lw=vs.LW_ARROW, zorder=vs.Z_GLYPH + 0.5))
    return vs.Box(x, y, s, s)


def tick_axis(ax, x0, x1, y, ticks, labels, gid="k_axis"):
    """Horizontal axis with categorical ticks and labels below."""
    with vs.group(ax, gid):
        ax.add_line(Line2D([x0, x1], [y, y], color=vs.GREY, lw=vs.LW_KEY, zorder=vs.Z_GLYPH))
        for t, s in zip(ticks, labels):
            ax.add_line(Line2D([t, t], [y, y + 0.9], color=vs.GREY, lw=vs.LW_KEY,
                               zorder=vs.Z_GLYPH))
            vs.label(ax, t, y + 1.4, s, ha="center", va="top")
    return vs.Box(x0, y, x1 - x0, 4.0)


# ----------------------------------------------------------------------------- panel (a)
def participant_key(ax, xr, y):
    """Marker key, right-aligned at xr: filled = autistic, open = non-autistic."""
    with vs.group(ax, "participant_key"):
        w2 = vs.text_size_mm(ax, "non-autistic")[0]
        w1 = vs.text_size_mm(ax, "autistic")[0]
        x2 = xr - w2
        vs.label(ax, x2, y, "non-autistic")
        vs._dot(ax, x2 - 1.6, y, False, 1.5)
        x1 = x2 - 1.6 - 4.0 - w1
        vs.label(ax, x1, y, "autistic")
        vs._dot(ax, x1 - 1.6, y, True, 1.5)


def step_head(ax, x, y, n, word):
    """Circled step number with its one-word name (keyed to the caption)."""
    vs.step_badge(ax, x + 1.5, y, n, gid=f"step_{n}")
    vs.label(ax, x + 3.8, y, word, gid=f"step_{n}_name")


def panel_a(ax):
    n_aut, n_non = cohort_counts()
    vs.panel_letter(ax, 1.2, 0.8, "a")
    participant_key(ax, vs.PAGE_W - 1.5, 1.5)

    # grey pre-split zone: cohort (1 marker ~ 10 participants) and cached covariances
    z = zone(ax, 1.2, 4, 33.8, 37)
    vs.dots(ax, 2.5, 6.3, 30, 12.5, round(n_aut / 10), round(n_non / 10), seed=3,
            gid="cohort_dots")
    vs.chip(ax, z.cx, 21.4, "871", kind="outline", ha="center", gid="tag_871")
    vs.arrow(ax, (z.cx, 23.4), (z.cx, 27.0), gid="arrow_to_cov")
    dk = vs.deck(ax, 13.6, 27.6, 6.0, n=3, tag="×871", gid="cov_deck")
    vs.label(ax, 12.4, dk.meta["front"].cy, vs.SYM["Sigma_hat"], ha="right", gid="sigma_hat")
    vs.chip(ax, z.cx, 38.7, "once · cached", kind="outline", ha="center", gid="tag_cached")

    # outer split and the two outer schemes (real site sizes for leave-one-site-out)
    split_line(ax, 38, 3.6, 41.4, text="outer split")
    sites, sizes = vs.site_sizes()
    # the fold pair is centred between the two lanes so that the fork reads as one split
    vs.label(ax, 41, 14.2, "pooled 10-fold", gid="lab_10fold")
    vs.fold_bar(ax, 41, 16.0, 26, h=2.6, n=10, held=3, gid="fold_bar_10")
    vs.label(ax, 41, 22.6, "leave-one-site-out", gid="lab_loso")
    vs.fold_bar(ax, 41, 24.4, 26, h=2.6, sizes=sizes, held=sites.index("NYU"),
                gid="fold_bar_loso")
    yf = 21.5                                            # fork height
    with vs.group(ax, "fold_bracket"):
        ax.add_line(Line2D([67.3, 68.4, 68.4, 67.3], [17.3, 17.3, 25.7, 25.7], color=vs.GREY,
                           lw=vs.LW_KEY, zorder=vs.Z_ARROW))
    vs.arrow(ax, (68.4, yf), (70.2, yf), (70.2, 36.0), (95.6, 36.0),
             gid="arrow_to_heldout")
    vs.arrow(ax, (70.2, yf), (70.2, 14.5), (74.6, 14.5), gid="arrow_to_train")

    # blue training lane, steps 1-5 (per ensemble member); right edge kept 0.2 mm inside
    # the page so that the 1 pt border is not clipped
    lane_w = vs.PAGE_W - 73 - 0.2
    vs.lane(ax, 73, 4, lane_w, 23, kind="train", text="outer training", gid="lane_train")
    with vs.group(ax, "member_tabs"):
        for k, m in enumerate(("corr", "static", "windowed")):
            vs.member_tab(ax, 158.4 + k * 1.8, 5.0, m, name=False, gid=f"tab_{m}")
        vs.label(ax, 157.4, 6.6, "per member", ha="right")
    yb, yc = 10.0, 18.5                                  # badge row, icon centre
    step_head(ax, 75.5, yb, 1, "anchor")
    vs.dome(ax, 75.5, 12.0, 16, 12.5, anchor_label=vs.SYM["Sigma_T"], lock=True,
            gid="step1_anchor")
    step_head(ax, 96.0, yb, 2, "project")
    vs.matrix_thumb(ax, 96.0, yc - 3.0, 6.0, strips="", gid="step2_matrix")
    vs.vector_bar(ax, 103.8, yc - 4.0, 8.0, gid="step2_vector")
    step_head(ax, 111.5, yb, 3, "scale")
    sc = vs.scaler_glyph(ax, 111.5, yc - 3.25, 6.5, gid="step3_scaler")
    step_head(ax, 123.0, yb, 4, "inner CV")
    # italic C (as on the arrow) + plain-text grid; two runs, because a mixed string is
    # parsed as mathtext and the Unicode ∈ and superscript minus then have no glyph
    cgrid = r"$\in\{10^{-3}\ldots1\}$"                   # mathtext: Times/STIX glyphs only
    wc, wg = vs.text_size_mm(ax, "$C$")[0], vs.text_size_mm(ax, cgrid)[0]
    ctot = wc + 0.7 + wg
    iw = max(18.0, ctot + 2.6)
    with vs.group(ax, "step4_inner_cv"):
        ax.add_patch(FancyBboxPatch((123.0, 13.0), iw, 11.0,
                                    boxstyle=f"round,pad=0,rounding_size={vs.RADIUS}",
                                    facecolor="white", edgecolor=vs.GREY, linewidth=vs.LW_KEY,
                                    linestyle=DASHED, zorder=vs.Z_GLYPH))
        inner_folds(ax, 123.0 + iw / 2 - 6, 15.2, 12, 2.2, gid="fold_bar_inner")
        xc0 = 123.0 + iw / 2 - ctot / 2
        vs.label(ax, xc0, 21.7, "$C$", va="baseline", gid="lab_C_grid_symbol")
        vs.label(ax, xc0 + wc + 0.7, 21.7, cgrid, va="baseline", gid="lab_C_grid")
    x5 = 147.0
    vs.arrow(ax, (123.0 + iw + 0.5, yc), (x5 - 0.6, yc), gid="arrow_C")
    vs.label(ax, (123.0 + iw + x5) / 2, yc - 0.9, "$C$", ha="center", va="bottom", gid="lab_C")
    step_head(ax, x5, yb, 5, "refit")
    sg = vs.sigmoid_glyph(ax, x5, yc - 3.25, 6.5, gid="step5_sigmoid")
    vs.arrow(ax, (92.2, yc), (95.4, yc), gid="arrow_12")
    vs.arrow(ax, (106.0, yc), (110.9, yc), gid="arrow_23")
    vs.arrow(ax, (118.6, yc), (122.4, yc), gid="arrow_34")

    # vermilion-outlined held-out lane: training fits applied, member p, ensemble p-bar
    vs.lane(ax, 73, 30, lane_w, 11, kind="heldout", text="held-out", gid="lane_heldout")
    yh = 36.0
    vs.matrix_thumb(ax, 96.25, yh - 2.75, 5.5, strips="", seed=4, gid="heldout_matrix")
    vs.vector_bar(ax, 103.8, yh - 3.7, 7.4, seed=5, gid="heldout_vector")
    vs.scaler_glyph(ax, 111.5, yh - 3.25, 6.5, gid="heldout_scaler")
    vs.sigmoid_glyph(ax, x5, yh - 3.25, 6.5, gid="heldout_sigmoid")
    vs.arrow(ax, (106.0, yh), (110.9, yh), gid="arrow_h23")
    vs.arrow(ax, (118.6, yh), (x5 - 0.6, yh), gid="arrow_h35")
    vs.arrow(ax, (87.0, 26.3), (97.2, 32.4), kind="fit", rad=0.15, gid="fit_anchor")
    vs.arrow(ax, (sc.cx, sc.b + 0.6), (sc.cx, yh - 3.9), kind="fit", gid="fit_scaler")
    vs.arrow(ax, (sg.cx, sg.b + 0.6), (sg.cx, yh - 3.9), kind="fit", gid="fit_classifier")
    xm = 161.6
    for m, dy in (("corr", -2.6), ("static", 0.0), ("windowed", 2.6)):
        vs.strand(ax, (x5 + 6.9, yh + dy), (x5 + 9.4, yh + dy), (xm - 2.3, yh + dy * 0.15),
                  member=m, gid=f"strand_{m}")
    vs.merge_node(ax, xm, yh, d=4.4, gid="ensemble_pbar")


# ----------------------------------------------------------------------------- panel (b)
def panel_b(ax):
    y0 = 47.0
    vs.panel_letter(ax, 1.2, y0 - 0.2, "b")
    vs.label(ax, 5.0, y0 + 1.5, "19,900 connections", gid="lab_connections")
    sx = 32.0
    rows = (("reference", 52.8, 54.6), ("global selection", 68.2, 70.0))
    split_line(ax, sx, 53.9, 81.0, gid="split_b")
    for name, ly, top in rows:
        h = 10.0
        cy = top + h / 2 + 0.6
        glob = name.startswith("global")
        tag = "glob" if glob else "ref"
        vs.label(ax, 0, ly, name, gid=f"lab_{tag}")
        zone(ax, 0, top, 29, h, gid=f"zone_{tag}")
        vs.deck(ax, 2.5, cy - 3.3, 5.2, n=3, tag=None, seed=7, gid=f"deck_{tag}")
        vs.lane(ax, 35, top, 30, h, kind="train", gid=f"lane_{tag}")
        hc = held_cell(ax, 69, top + 1.5, 9, h - 3.0, gid=f"held_{tag}")
        vs.scaler_glyph(ax, 49.6, cy - 2.6, 5.2, gid=f"scaler_{tag}")
        vs.sigmoid_glyph(ax, 58.2, cy - 2.6, 5.2, gid=f"sigmoid_{tag}")
        vs.arrow(ax, (65.4, cy), (68.4, cy), kind="fit", gid=f"fit_{tag}")
        if not glob:
            mini_anchor(ax, 39.2, cy - 3.0, 8.0, 6.0, gid="anchor_ref")
            vs.arrow(ax, (29.6, cy), (38.6, cy), gid="arrow_ref")
        else:
            mini_anchor(ax, 11.3, cy - 3.0, 8.0, 6.0, gid="anchor_global")
            fn = vs.funnel(ax, 21.8, cy - 2.9, 5.6, 5.8, gid="select_2000")
            vs.label(ax, fn.cx, ly, "2,000", ha="center", gid="lab_2000")
            vs.arrow(ax, (29.6, cy), (49.0, cy), gid="arrow_glob")
            yl = top + h + 3.2
            vs.arrow(ax, (hc.cx, hc.b + 0.6), (hc.cx, yl), (fn.cx, yl), (fn.cx, fn.b + 0.5),
                     kind="leak", gid="leak_labels")
            vs.label(ax, (hc.cx + fn.cx) / 2, yl + 0.8, "labels", ha="center", va="top",
                     gid="lab_labels")

    # region level: nuisance model once, parcels ranked with one of two label scopes
    yr = 90.4
    vs.label(ax, 0, yr, "196 regions", gid="lab_regions")
    x = vs.text_size_mm(ax, "196 regions")[0] + 2.0
    c1 = vs.chip(ax, x, yr, "863", kind="outline", gid="tag_863_b")
    vs.chip(ax, c1.r + 1.5, yr, "$r$ = 0.30", kind="outline", gid="tag_r030")
    top, h = 93.4, 10.0
    cy = top + h / 2
    zone(ax, 0, top, 29, h, gid="zone_region")
    vs.chip(ax, 14.5, cy, "nuisance model", kind="outline", ha="center", gid="tag_nuisance")
    split_line(ax, sx, top - 0.6, top + h + 0.6, gid="split_region")
    rk = rank_glyph(ax, 36.0, cy - 3.0, 8.0, 6.0, gid="rank_parcels")
    vs.arrow(ax, (29.6, cy), (35.4, cy), gid="arrow_rank")
    vs.arrow(ax, (rk.r + 0.6, cy), (48.6, cy), gid="arrow_topk")
    ticks = np.linspace(51.0, 76.0, 6)
    tick_axis(ax, 49.6, 77.4, cy + 0.6, ticks, ["10", "20", "30", "50", "100", "all"],
              gid="k_ticks")
    vs.label(ax, 63.5, cy - 2.0, "top $k$ regions", ha="center", gid="lab_topk")
    ys = top + h + 3.0
    t1 = vs.chip(ax, 34.0, ys, "training-fold", kind="train", gid="scope_train")
    vs.chip(ax, t1.r + 1.0, ys, "all labels", kind="heldout", gid="scope_all")
    with vs.group(ax, "scope_link"):
        ax.add_line(Line2D([rk.cx, rk.cx], [rk.b + 0.4, ys - 1.8], color=vs.GREY,
                           lw=vs.LW_KEY, zorder=vs.Z_ARROW))


# ----------------------------------------------------------------------------- panel (c)
def panel_c(ax):
    x0, y0 = XC, 47.0
    vs.panel_letter(ax, x0 + 0.4, y0 - 0.2, "c")
    xl, xm, xr = 97.0, 124.5, 152.0                     # 0.01 mm | complete | 0.02 mm
    yt = 62.4                                            # tag row
    vs.dots(ax, xm - 8.0, 51.0, 16.0, 8.4, 11, 13, seed=5, gid="contrast_dots")
    vs.chip(ax, xm, yt, "863", kind="outline", ha="center", gid="tag_863")
    for xc_, tag, gid in ((xl, "256 pairs · 0.01 mm", "matched_001"),
                          (xr, "280 pairs · 0.02 mm", "matched_002")):
        with vs.group(ax, gid):
            for k in range(3):
                vs.dot_pair(ax, xc_ - 1.7, 52.4 + k * 3.0, gap=3.4, color=vs.MATCH)
        vs.chip(ax, xc_, yt, tag, kind="outline", ha="center", gid=f"tag_{gid}")
    for xa, xb in ((xm - 8.7, xl + 3.6), (xm + 8.7, xr - 3.6)):
        vs.arrow(ax, (xa, 55.4), (xb, 55.4), kind="match", gid="arrow_match")
        vs.label(ax, (xa + xb) / 2, 53.5, "FD match", ha="center", gid="lab_fd_match")

    # one nuisance model per sample, then the standardised difference d_e
    band = zone(ax, x0 + 2.0, 68.0, vs.PAGE_W - x0 - 2.0, 12.6, gid="nuisance_band")
    for xc_ in (xl, xm, xr):
        vs.arrow(ax, (xc_, yt + 2.2), (xc_, band.t - 0.5), gid="arrow_to_nuisance")
    names = ["age", "sex", "FD", "site"]
    ws = [vs.text_size_mm(ax, s)[0] + 1.8 for s in names]
    wl = vs.text_size_mm(ax, "adjust")[0]
    tot = wl + 1.6 + sum(ws) + 1.2 * (len(ws) - 1)
    cx = xm - tot / 2
    vs.label(ax, cx, 71.4, "adjust", gid="lab_adjust")
    cx += wl + 1.6
    for s, w in zip(names, ws):
        vs.chip(ax, cx, 71.4, s, kind="outline", gid=f"cov_{s}")
        cx += w + 1.2
    eq = (r"$d_e = (\bar{R}_{e,\mathrm{autistic}} - \bar{R}_{e,\mathrm{comparison}})"
          r"\,/\,\mathrm{SD}(R_e)$")
    vs.label(ax, xm, 76.6, eq, ha="center", gid="eq_de")

    # outputs: 36 system pairs (complete + 0.01 mm), SMN parcels (complete + 0.02 mm)
    tri = triangle_glyph(ax, 104.0, 87.0, 11.0, gid="system_pairs")
    br = vs.brain_glyph(ax, 131.5, 86.0, 18.0, 12.5, view="lateral", highlight="sensorimotor",
                        gid="brain_smn")
    yb = band.b + 0.6
    for (xa, xb, yb2) in ((xl, tri.l - 1.0, tri.t - 1.8), (xm, tri.r + 0.4, tri.t - 1.8),
                          (xm, br.l + 1.0, br.t + 0.2), (xr, br.r - 1.0, br.t + 0.2)):
        vs.arrow(ax, (xa, yb), (xb, yb2), kind="agg", gid="arrow_to_output")
    vs.label(ax, tri.cx - 0.6, 100.4, "36 pairs", ha="center", gid="lab_36pairs")
    vs.label(ax, br.cx, 100.4, "SMN", ha="center", gid="lab_smn")
    c1w = vs.text_size_mm(ax, "5,000 perms within site")[0] + 1.8
    c2w = vs.text_size_mm(ax, "BH-FDR")[0] + 1.8
    xs = xm - (c1w + 1.5 + c2w) / 2
    vs.chip(ax, xs, 104.6, "5,000 perms within site", kind="outline", gid="tag_perms")
    vs.chip(ax, xs + c1w + 1.5, 104.6, "BH-FDR", kind="outline", gid="tag_bhfdr")


# ----------------------------------------------------------------------------- panel (d)
def panel_d(ax):
    y0 = 112.0
    vs.panel_letter(ax, 1.2, y0 + 0.4, "d")
    px0, px1, lo, hi = 36.0, 114.0, 70.0, 150.0
    X = lambda v: px0 + (v - lo) / (hi - lo) * (px1 - px0)
    pool = sorted(IQ_AUT + IQ_NON)
    n = len(pool)
    cuts = [(pool[int(n * q) - 1] + pool[int(n * q)]) / 2 for q in (0.25, 0.5, 0.75)]
    pairs = greedy_iq_match(IQ_AUT, IQ_NON)
    rows = (("i", "site-centred quartiles", 114.0), ("ii", "fixed comparison", 122.8),
            ("iii", "within-site matching", 131.6))
    for num, name, top in rows:
        ya, yn, yl = top + 4.2, top + 6.8, top + 1.4
        with vs.group(ax, f"design_{num}"):
            vs.label(ax, 0, top + 5.5, num)
            vs.label(ax, 4.5, top + 5.5, name)
            if num == "i":
                edges = [lo] + cuts + [hi]
                for c in cuts:
                    ax.add_line(Line2D([X(c)] * 2, [top + 2.8, top + 7.9], color=vs.GREY,
                                       lw=vs.LW_KEY, ls=(0, (2, 1.5)), zorder=vs.Z_GLYPH))
                for k in range(4):
                    vs.label(ax, X((edges[k] + edges[k + 1]) / 2), yl, f"Q{k + 1}",
                             ha="center")
            elif num == "ii":
                for c in (85, 100, 115):
                    ax.add_line(Line2D([X(c)] * 2, [top + 2.8, top + 5.4], color=vs.GREY,
                                       lw=vs.LW_KEY, zorder=vs.Z_GLYPH))
                for (a, b), s in zip(((lo, 85), (85, 100), (100, 115), (115, hi)),
                                     ("<85", "85–99", "100–114", "≥115")):
                    vs.label(ax, X((a + b) / 2), yl, s, ha="center")
                ax.add_patch(FancyBboxPatch((X(83), yn - 1.0), X(141) - X(83), 2.0,
                                            boxstyle="round,pad=0,rounding_size=1.0",
                                            facecolor="none", edgecolor=vs.GREY,
                                            linewidth=vs.LW_KEY, zorder=vs.Z_GLYPH))
                vs.chip(ax, X(141) + 1.2, yn, "468", kind="outline", gid="tag_468")
            else:
                for a, b in pairs.items():
                    ax.add_line(Line2D([X(a), X(b)], [ya, yn], color=vs.MATCH, lw=vs.LW_ARROW,
                                       zorder=vs.Z_GLYPH))
                ex = [a for a in IQ_AUT if a not in pairs]
                for a in ex:
                    ax.add_patch(Circle((X(a), ya), 0.75, facecolor=vs.RULE, edgecolor=vs.GREY,
                                        linewidth=vs.LW_KEY, zorder=vs.Z_GLYPH + 0.5))
                    vs.label(ax, X(a), yl, "excluded", ha="center")
                a0 = 109
                vs.label(ax, X((a0 + pairs[a0]) / 2), yl, "≤5 points", ha="center")
                vs.chip(ax, X(141) + 1.2, top + 5.5, "Q1–Q4", kind="outline", gid="tag_q1q4")
            for a in IQ_AUT:
                if num != "iii" or a in pairs:
                    vs._dot(ax, X(a), ya, True, 1.5)
            for b in IQ_NON:
                vs._dot(ax, X(b), yn, False, 1.5)

    # shared full-scale IQ axis
    ya = 141.4
    vs.arrow(ax, (px0 - 1.0, ya), (px1 + 1.5, ya), gid="iq_axis")
    with vs.group(ax, "iq_ticks"):
        for v in (85, 100, 115):
            ax.add_line(Line2D([X(v)] * 2, [ya, ya + 0.9], color=vs.GREY, lw=vs.LW_KEY,
                               zorder=vs.Z_GLYPH))
            vs.label(ax, X(v), ya + 1.3, str(v), ha="center", va="top")
        vs.label(ax, px1 + 1.5, ya + 1.3, "full-scale IQ", ha="right", va="top")

    # (iv) score vs IQ within each diagnostic group (schematic points, no trend)
    gx0, gx1, gy0, gy1 = 134.0, 160.0, 119.0, 136.0
    with vs.group(ax, "design_iv"):
        vs.label(ax, 128.0, 115.4, "iv")
        vs.label(ax, 132.5, 115.4, "score–IQ")
        aut = [(137.5, 129.0), (141.0, 124.0), (146.0, 131.5), (150.0, 122.0), (155.0, 128.0),
               (157.5, 124.5), (143.5, 133.5)]
        non = [(139.0, 122.5), (144.0, 127.5), (148.5, 125.5), (152.0, 132.5), (153.5, 120.8),
               (158.0, 132.5), (136.5, 133.5)]
        for pts, f in ((aut, True), (non, False)):
            for u, v in pts:
                vs._dot(ax, u, v, f, 1.3)
        vs.label(ax, gx1, gy1 + 1.2, "IQ", ha="right", va="top")
        vs.label(ax, gx0 + 1.2, gy0 + 0.4, "score", va="center")
    vs.arrow(ax, (gx0, gy1), (gx1, gy1), gid="iv_xaxis")
    vs.arrow(ax, (gx0, gy1), (gx0, gy0 - 1.2), gid="iv_yaxis")
    vs.chip(ax, (gx0 + gx1) / 2, ya, "Spearman ρ", kind="outline", ha="center",
            gid="tag_spearman")


# ----------------------------------------------------------------------------- panel (e)
def panel_e(ax):
    y0 = 151.5
    vs.panel_letter(ax, 1.2, y0 + 0.1, "e")
    # identical inputs: static tangent features, scaling, outer folds
    vs.member_tab(ax, 4.0, 153.1, "static", gid="tab_static_e")
    vs.scaler_glyph(ax, 4.0, 158.7, 5.5, gid="scaler_e")
    vs.fold_bar(ax, 11.5, 160.25, 15.0, h=2.4, n=10, held=3, gid="fold_bar_e")
    ybus = 156.5
    lin = [(41.0, "$\\ell_2$-logistic", vs.sigmoid_glyph), (56.0, "ridge", ridge_glyph),
           (71.0, "linear SVC", svc_glyph), (86.0, "PCA +\nshrinkage LDA", lda_glyph),
           (101.0, "low-rank\nspectral", spectral_glyph)]
    tree = [(125.0, "random forest", forest_glyph), (140.0, "XGBoost", boost_glyph),
            (155.0, "CatBoost", boost_glyph)]
    with vs.group(ax, "input_bus"):
        # shared input in grey, then the bus takes the colour of the branch it feeds
        ax.add_line(Line2D([27.2, 31.5, 31.5], [161.45, 161.45, ybus], color=vs.GREY,
                           lw=vs.LW_ARROW, zorder=vs.Z_ARROW, solid_joinstyle="miter"))
        ax.add_line(Line2D([31.5, lin[-1][0]], [ybus, ybus], color=vs.LINEAR,
                           lw=vs.LW_ARROW, zorder=vs.Z_ARROW))
        ax.add_line(Line2D([lin[-1][0], tree[-1][0]], [ybus, ybus], color=vs.TREE,
                           lw=vs.LW_ARROW, zorder=vs.Z_ARROW))
    vs.label(ax, lin[0][0] - 3.0, 153.9, "linear", color=vs.LINEAR, gid="lab_linear")
    vs.label(ax, tree[0][0] - 3.0, 153.9, "tree ensembles", color=vs.TREE, gid="lab_trees")
    branch = {c: "linear" for c, _, _ in lin} | {c: "tree" for c, _, _ in tree}
    for cx, name, fn in lin + tree:
        gid = name.split("\n")[0].replace("$\\ell_2$", "l2").replace(" ", "_").strip("+_")
        if fn is vs.sigmoid_glyph:
            fn(ax, cx - 3.0, 158.7, 6.0, gid="clf_l2_logistic")
        else:
            fn(ax, cx - 3.0, 158.7, 6.0, gid=f"clf_{gid}")
        vs.arrow(ax, (cx, ybus), (cx, 158.4), kind=branch[cx], gid="bus_drop")
        vs.label(ax, cx, 165.9, name, ha="center", va="top", linespacing=1.1,
                 gid=f"name_{gid}")
    vs.chip(ax, lin[0][0], 170.4, "tuned", kind="train", ha="center", gid="tag_tuned")


# ----------------------------------------------------------------------------- build
def build(out: Path, copy_to: Path | None, stem: str = STEM):
    import matplotlib.pyplot as plt
    vs.apply_style()
    vs.CHIP_NEUTRAL_FACE = "white"      # grey chips and icon boxes would vanish on the
    vs.GLYPH_FACE = "white"             # light panel overlays
    fig, ax = vs.canvas(H_MM, bleed=vs.PANEL_BLEED)
    # light overlay blocks (1 mm white gaps): a full width; b and c share one band
    BG = {"a": (0.0, 0.0, vs.PAGE_W, 44.8),
          "b": (0.0, 45.8, XC - 1.8, 110.4 - 45.8),
          "c": (XC - 0.8, 45.8, vs.PAGE_W - (XC - 0.8), 110.4 - 45.8),
          "d": (0.0, 111.4, vs.PAGE_W, 149.6 - 111.4),
          "e": (0.0, 150.6, vs.PAGE_W, 172.4 - 150.6)}
    for k, (x, y, w, h) in BG.items():
        vs.panel_bg(ax, x, y, w, h, gid=f"panel_bg_{k}")
    for fn in (panel_a, panel_b, panel_c, panel_d, panel_e):
        fn(ax)
    # key of the functional line colours used in panels (c)-(e); the grey, blue and
    # vermilion arrows keep the meanings of the arrow key in Figure 1
    vs.arrow_key(ax, 4.0, 175.4, kinds=("match", "agg", "linear", "tree"), gid="line_key")
    probs = vs.check_layout(fig)
    paths = vs.save_figure(fig, stem, out_dir=out, copy_to=copy_to)
    plt.close(fig)
    return paths, probs


def main() -> None:
    ap = argparse.ArgumentParser(description="Methods figure 3: evaluation and analysis designs")
    ap.add_argument("--out", default=str(vs.FIG_OUT))
    ap.add_argument("--stem", default=STEM)
    ap.add_argument("--no-copy", action="store_true",
                    help="do not copy the exports to ASD-journal-I/figures")
    args = ap.parse_args()
    paths, _ = build(Path(args.out), None if args.no_copy else vs.JOURNAL_FIG, args.stem)
    for p in paths:
        print("wrote", p)
    print("fonts:", vs.pdf_fonts(paths[0]))


if __name__ == "__main__":
    main()
