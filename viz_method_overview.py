"""
Figure 1 (Methods): overview of the method, drawn with the shared viz_style glyphs.

Three horizontal bands, each read left to right:
  a  cohort strip - ABIDE-I dataset -> quality control (241 records excluded) -> strict
     cohort (871 participants; the dot cloud draws about one dot per ten participants,
     40 filled = autistic, 47 open = non-autistic) -> CC200 atlas coloured by system ->
     regional time series of one example participant with the K = 6 windows of L = 50
     volumes (Eq. 6) as translucent bands;
  b  the three connectivity representations of the same participant (real 200 x 200
     matrices, regions ordered by system) with member tabs and one shared colour key; a
     blue padlock follows the two tangent members, whose references are fitted on
     training participants only;
  c  the ensemble - one member strand per representation (19,900 features each) passes the
     scaler (Eq. 10) and the l2-logistic model (Eq. 11; C chosen by inner three-fold CV)
     inside the blue training lane, the strands merge at p-bar (Eq. 12) and p-bar >= 0.5
     gives the class; pooled ten-fold and leave-one-site-out fold bars (held-out fold in
     vermilion).  A one-line key of the three arrow meanings closes the figure.
The circled steps 1-5 are keyed to the caption.  No result (AUROC, p-value, effect size,
learned weight) is drawn.

The matrices are computed exactly as the Methods specify.  Covariances are Ledoit-Wolf
estimates (Eq. 1); window covariances receive additional isotropic shrinkage r = 0.15
(Eq. 5); windows follow Eq. 6.  The tangent references are affine-invariant Riemannian means
(Eq. 2; fixed-point iteration, at most ten updates, eigenvalues clamped at 1e-6) estimated
from an ILLUSTRATIVE training partition, namely the strict cohort minus a random 10 %
(numpy default_rng(42)); the dynamic reference is the Riemannian mean of each training
participant's arithmetic mean of the six window covariances.

Usage:
    python viz_method_overview.py [--out results/latest_study/figures] [--cache file.npz]
                                  [--no-copy]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

import viz_style as vs  # noqa: E402

TS_DIR = ROOT / "abide_raw/ABIDE_pcp/cpac/filt_noglobal"
META = ROOT / "abide_processed/mri/metadata.csv"
EXAMPLE = "NYU_0050952"
STEM = "fig_overview"
DEFAULT_CACHE = ROOT / "results/latest_study/cache/fig_overview_refs.npz"

K, L, R_WIN = 6, 50, 0.15          # windows, window length (volumes), window shrinkage (Eq. 5)
EIG_FLOOR = 1e-6                   # eigenvalue clamp
MAX_ITER = 10                      # Riemannian-mean updates

H_MM = 128.0                       # printed height (mm); width is the 165 mm text width


# ----------------------------------------------------------------------------------------
# computation (Methods, Sections 3.2.2-3.2.6)
# ----------------------------------------------------------------------------------------
def ts_path(site: str, sid: int) -> Path:
    hits = sorted(TS_DIR.glob(f"*_{int(sid):07d}_rois_cc200.1D"))
    if not hits:
        raise FileNotFoundError(f"no CC200 series for {site} {sid}")
    return hits[0]


def load_ts(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def lw_cov(x: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(x).covariance_                       # Eq. 1


def iso_shrink(c: np.ndarray, r: float) -> np.ndarray:
    p = c.shape[-1]
    return (1 - r) * c + r * np.trace(c) / p * np.eye(p)          # Eq. 5


def _spd_fun(c: np.ndarray, fun) -> np.ndarray:
    """f(C) through a symmetric eigendecomposition, eigenvalues clamped at 1e-6."""
    w, v = np.linalg.eigh(c)
    w = np.maximum(w, EIG_FLOOR)
    return (v * fun(w)[..., None, :]) @ np.swapaxes(v, -1, -2)


def _sym_exp(t: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(t)
    return (v * np.exp(w)[..., None, :]) @ np.swapaxes(v, -1, -2)


def riemann_mean(covs: np.ndarray, chunk: int = 128) -> np.ndarray:
    """Affine-invariant Riemannian mean (Eq. 2), fixed-point iteration, <= 10 updates."""
    g = covs.mean(axis=0)
    for _ in range(MAX_ITER):
        g_half = _spd_fun(g, np.sqrt)
        g_ihalf = _spd_fun(g, lambda w: 1.0 / np.sqrt(w))
        acc = np.zeros_like(g)
        for i in range(0, len(covs), chunk):
            acc += _spd_fun(g_ihalf @ covs[i:i + chunk] @ g_ihalf, np.log).sum(axis=0)
        t = acc / len(covs)
        g = g_half @ _sym_exp(t) @ g_half
        g = (g + g.T) / 2
        if np.linalg.norm(t) < 1e-8:
            break
    return g


def log_map(ref: np.ndarray, c: np.ndarray) -> np.ndarray:
    r_ihalf = _spd_fun(ref, lambda w: 1.0 / np.sqrt(w))
    return _spd_fun(r_ihalf @ c @ r_ihalf, np.log)               # Eq. 3


def window_starts(t: int) -> list[int]:
    delta = (t - L) / (K - 1)                                    # Eq. 6
    return [int(round(k * delta)) for k in range(K)]


def window_covs(x: np.ndarray) -> np.ndarray:
    return np.stack([iso_shrink(lw_cov(x[s:s + L]), R_WIN) for s in window_starts(len(x))])


def compute(cache: Path | None) -> dict:
    if cache is not None and cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    meta = pd.read_csv(META)
    n = len(meta)
    rng = np.random.default_rng(42)
    held = rng.choice(n, size=int(round(0.10 * n)), replace=False)
    train = np.setdiff1d(np.arange(n), held)

    static, dyn_mean = [], []
    for site, sid in zip(meta["site"], meta["subject_id"]):
        x = load_ts(ts_path(site, sid))
        static.append(lw_cov(x))
        dyn_mean.append(window_covs(x).mean(axis=0))
    static = np.stack(static)
    dyn_mean = np.stack(dyn_mean)

    ref_static = riemann_mean(static[train])
    ref_dyn = riemann_mean(dyn_mean[train])

    ts = load_ts(TS_DIR / f"{EXAMPLE}_rois_cc200.1D")
    c = lw_cov(ts)
    d = np.sqrt(np.diag(c))
    corr = c / np.outer(d, d)
    s_static = log_map(ref_static, c)
    wins = window_covs(ts)
    s_dyn = np.mean([log_map(ref_dyn, w) for w in wins], axis=0)   # Eq. 7 (elementwise)

    ex_id = int(EXAMPLE.split("_")[-1])
    ex_idx = int(np.flatnonzero(meta["subject_id"].to_numpy() == ex_id)[0])
    out = dict(ts=ts, corr=corr, s_static=s_static, s_dyn=s_dyn,
               starts=np.array(window_starts(len(ts))),
               ex_in_train=np.array(ex_idx in set(train.tolist())),
               n_train=np.array(len(train)))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out


# ----------------------------------------------------------------------------------------
# small local glyphs (built from viz_style tokens)
# ----------------------------------------------------------------------------------------
def _cmap():
    return matplotlib.colormaps[vs.MATRIX_CMAP].with_extremes(bad="white")


def vector_strip(ax, x, y, w, h, vec, vlim, gid):
    """Horizontal thumbnail of one real 19,900-feature vector (RdBu_r at +-vlim)."""
    with vs.group(ax, gid):
        ax.imshow(np.asarray(vec, float)[None, :], cmap=_cmap(), vmin=-vlim, vmax=vlim,
                  extent=(x, x + w, y + h, y), origin="upper", interpolation="nearest",
                  aspect="auto", zorder=vs.Z_GLYPH)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=vs.GREY,
                               linewidth=vs.LW_KEY, zorder=vs.Z_GLYPH + 0.2))
    return vs.Box(x, y, w, h)


def inner_folds(ax, x, y, w, h, n=3, held=1, gid="inner_cv_folds"):
    """Inner stratified CV inside the training partition: training cells in lane tint,
    the inner validation cell in solid blue (vermilion stays reserved for outer held-out)."""
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


def swatch(ax, x, y, face, edge="none", s=1.8):
    """Square legend swatch centred vertically on y (left edge x)."""
    ax.add_patch(Rectangle((x, y - s / 2), s, s, facecolor=face, edgecolor=edge,
                           linewidth=vs.LW_KEY, zorder=vs.Z_GLYPH))
    return vs.Box(x, y - s / 2, s, s)


def tw(ax, s):
    return vs.text_size_mm(ax, s)[0]


# ----------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------
def build(out: Path, cache: Path | None, copy_to: Path | None) -> None:
    data = compute(cache)
    vs.apply_style()
    vs.CHIP_NEUTRAL_FACE = "white"      # grey chips would vanish on the panel overlay
    fig, ax = vs.canvas(H_MM, bleed=vs.PANEL_BLEED)
    iu = np.triu_indices(200, 1)
    order, _ = vs.system_order()
    # light overlay blocks behind panels a, b and c (1 mm white gap between blocks)
    BG = {"a": (0.0, 32.8), "b": (33.8, 83.4), "c": (84.4, 120.8)}
    for k, (t, b) in BG.items():
        vs.panel_bg(ax, 0.0, t, vs.PAGE_W, b - t, gid=f"panel_bg_{k}")

    # ===================================== a: cohort strip ==============================
    vs.panel_letter(ax, 1.5, 1.2, "a")
    yc = 11.0                                           # icon centre line
    y_name, y_tag, y_leg = 21.2, 25.4, 29.8
    cyl = vs.cylinder(ax, 8.0, yc - 4.5, 8.0, 9.0, gid="a_dataset_abide")
    fun = vs.funnel(ax, 31.0, yc - 4.0, 8.0, 8.0, gid="a_quality_control")
    dts = vs.dots(ax, 51.0, yc - 5.5, 24.0, 11.0, 40, 47, seed=1, gid="a_strict_cohort")
    brn = vs.brain_glyph(ax, 86.0, yc - 6.5, 20.0, 13.0, view="lateral",
                         mark_excluded=False,        # all 200 regions enter prediction
                         gid="a_cc200_brain")
    tsb = vs.timeseries_glyph(ax, 116.0, yc - 6.5, 45.0, 13.0, data["ts"],
                              starts=[int(s) for s in data["starts"]], L=L,
                              bands="overlay", gid="a_example_timeseries")
    icons = [cyl, fun, dts, brn, tsb]
    for i0, i1 in zip(icons[:-1], icons[1:]):
        vs.arrow(ax, (i0.r + 1.2, yc), (i1.l - 1.2, yc), gid="a_flow")
    with vs.group(ax, "a_names"):
        for b, s in zip(icons, ["ABIDE-I", "quality control", "primary cohort",
                                "CC200 atlas", "regional time series"]):
            vs.label(ax, b.cx, y_name, s, ha="center")
    vs.chip(ax, cyl.cx, y_tag, "1,112 · 17 sites", ha="center", gid="a_tag_release")
    vs.chip(ax, fun.cx, y_tag, "−241", ha="center", gid="a_tag_excluded")
    vs.chip(ax, brn.cx, y_tag, "200 regions", ha="center", gid="a_tag_regions")
    vs.chip(ax, tsb.cx, y_tag, "K = 6, L = 50", ha="center", kind="outline",
            gid="a_tag_windows")
    # strict cohort: 871 tag, then 403 autistic (filled) and 468 non-autistic (open)
    d_mk, g1, g2 = 1.5, 0.7, 2.2
    w871 = tw(ax, "871") + 1.8
    w403, w468 = tw(ax, "403"), tw(ax, "468")
    total = w871 + g2 + w403 + g1 + d_mk + g2 + w468 + g1 + d_mk
    x = dts.cx - total / 2
    with vs.group(ax, "a_tag_cohort"):
        vs.chip(ax, x, y_tag, "871", gid="tag_871")
        x += w871 + g2
        vs.label(ax, x, y_tag, "403", va="center_baseline")
        x += w403 + g1
        vs.dot(ax, x + d_mk / 2, y_tag, filled=True, d=d_mk, gid="dot_autistic")
        x += d_mk + g2
        vs.label(ax, x, y_tag, "468", va="center_baseline")
        x += w468 + g1
        vs.dot(ax, x + d_mk / 2, y_tag, filled=False, d=d_mk, gid="dot_nonautistic")
    leg = vs.network_legend(ax, brn.l - 1.5, y_leg, gap=1.8, gid="a_system_legend")
    vs.step_badge(ax, fun.r + 0.8, fun.t - 0.4, 1, gid="step_1")
    vs.step_badge(ax, tsb.r + 1.7, tsb.t - 0.3, 2, gid="step_2")

    # ===================================== b: representations ===========================
    y_bus = 34.6
    vs.panel_letter(ax, 1.5, BG["b"][0] + 1.0, "b")
    y_head, y_mat, size = 39.4, 46.0, 32.0
    xs = [11.0, 62.0, 113.0]
    members = ["corr", "static", "windowed"]
    mats = {"corr": data["corr"], "static": data["s_static"], "windowed": data["s_dyn"]}
    mboxes = []
    for x0, mem in zip(xs, members):
        m = vs.matrix_thumb(ax, x0, y_mat, size, mats[mem], diag="blank",
                            gid=f"b_matrix_{mem}")
        mboxes.append(m)
        # tangent members depend on a reference fitted on training participants only:
        # a blue padlock (the training-lane symbol) follows their names
        lock = mem != "corr"
        lk, lg = 2.2, 0.8                                  # padlock size, gap (mm)
        wname = 1.5 + 1.0 + tw(ax, vs.MEMBER_NAME[mem]) + ((lg + lk) if lock else 0.0)
        tb = vs.member_tab(ax, m.cx - wname / 2, y_head, mem, gid=f"b_tab_{mem}")
        if lock:
            vs.padlock(ax, tb.r + lg, y_head + (3.2 - lk) / 2, lk,
                       gid=f"b_reference_lock_{mem}")
    m0 = mboxes[0]
    vs.network_labels(ax, vs.Box(m0.meta["outer"].l, m0.t, m0.r - m0.meta["outer"].l, m0.h),
                      gid="b_system_labels")
    key = vs.cmap_key(ax, 150.0, mboxes[2].b - 4.0, w=10.0, gid="b_shared_key")
    # bus: time series -> the three representations
    xr = 163.6
    vs.arrow(ax, (tsb.r + 0.6, yc), (xr, yc), (xr, y_bus), (mboxes[0].cx, y_bus),
             head=False, gid="bus_a_to_b")
    for m in mboxes:
        vs.arrow(ax, (m.cx, y_bus), (m.cx, y_head - 0.7), gid="to_representation")

    # ===================================== c: ensemble ==================================
    y_c = 85.0
    vs.panel_letter(ax, 1.5, BG["c"][0] + 1.0, "c")
    rows = {"corr": 96.0, "static": 99.0, "windowed": 102.0}
    sx0, sx1, sh = 8.0, 25.0, 1.5
    # bus: representations -> vectorised features
    y_bus2 = mboxes[0].b + 4.0
    for m in mboxes:
        vs.arrow(ax, (m.cx, m.b + 0.6), (m.cx, y_bus2), head=False, gid="bus_b_to_c")
    x_in = (sx0 + sx1) / 2
    vs.arrow(ax, (mboxes[2].cx, y_bus2), (x_in, y_bus2), (x_in, rows["corr"] - sh / 2 - 0.7),
             gid="bus_b_to_c_in")
    for mem in members:
        r = rows[mem]
        vec = mats[mem][np.ix_(order, order)][iu]
        vector_strip(ax, sx0, r - sh / 2, sx1 - sx0, sh, vec, mboxes[members.index(mem)]
                     .meta["vlim"], gid=f"c_features_{mem}")
        with vs.group(ax, f"c_tab_{mem}"):
            swatch(ax, sx0 - 2.4, r, vs.MEMBER[mem], s=1.5)
    vs.chip(ax, x_in, rows["windowed"] + 4.2, "19,900", ha="center", gid="c_tag_features")

    lanebox = vs.lane(ax, 30.0, 87.5, 66.0, 31.5, kind="train",
                      text="outer training partition", gid="c_training_lane")
    gy, gs = 94.0, 10.0
    sc = vs.scaler_glyph(ax, 40.0, gy, s=gs, gid="c_scaler")
    sg = vs.sigmoid_glyph(ax, 58.0, gy, s=gs, gid="c_l2_logistic")
    nx, ny, nd = 106.0, rows["static"], 5.5
    x_turn = 98.5
    for mem in members:
        r = rows[mem]
        vs.strand(ax, (sx1 + 0.4, r), (sc.l, r), member=mem, gid=f"strand_{mem}_1")
        vs.strand(ax, (sc.r, r), (sg.l, r), member=mem, gid=f"strand_{mem}_2")
        vs.strand(ax, (sg.r, r), (x_turn, r), (nx - nd / 2 - 0.5, ny), member=mem,
                  head=(mem == "static"), gid=f"strand_{mem}_3")
    vs.merge_node(ax, nx, ny, d=nd, gid="c_merge_pbar")
    # decision: p-bar >= 0.5 -> autistic (filled), < 0.5 -> non-autistic (open)
    xo = nx + nd / 2 + 5.2
    for dy, txt, filled in ((-3.0, "≥ 0.5", True), (3.0, "< 0.5", False)):
        vs.arrow(ax, (nx + nd / 2 + 0.5, ny + dy / 3.0), (xo - 0.7, ny + dy),
                 gid="c_threshold")
        vs.label(ax, xo, ny + dy, txt, va="center_baseline")
        vs.dot(ax, xo + tw(ax, txt) + 1.8, ny + dy, filled=filled, gid="c_class")
    # inner CV inset (C grid) under the l2-logistic glyph
    # plain-text tag (no element-of sign or superscript minus needed); grid in the text
    lab_in, lab_c = "inner 3-fold", "$C$: 0.001…1"
    w_in, w_c, w_bar = tw(ax, lab_in), tw(ax, lab_c), 11.0
    iw = 1.6 + w_in + 1.6 + w_bar + 1.6 + w_c + 1.6
    ih, iy = 5.0, 110.5
    ix = min(max(sg.cx - iw / 2, lanebox.l + 2.0), lanebox.r - 2.0 - iw)
    with vs.group(ax, "c_inner_cv_inset"):
        ax.add_patch(FancyBboxPatch((ix, iy), iw, ih,
                                    boxstyle=f"round,pad=0,rounding_size={vs.RADIUS}",
                                    facecolor="white", edgecolor=vs.GREY,
                                    linewidth=vs.LW_KEY, linestyle=(0, (2.5, 1.5)),
                                    zorder=vs.Z_GLYPH - 0.5))
        vs.label(ax, ix + 1.6, iy + ih / 2, lab_in, va="center_baseline")
        inner_folds(ax, ix + 1.6 + w_in + 1.6, iy + ih / 2 - 0.9, w_bar, 1.8)
        vs.label(ax, ix + iw - 1.6, iy + ih / 2, lab_c, ha="right", va="center_baseline")
    vs.arrow(ax, (sg.cx, iy - 0.5), (sg.cx, sg.b + 0.6), kind="fit", gid="c_C_selected")
    vs.step_badge(ax, sc.r + 0.4, sc.t - 0.4, 3, gid="step_3")
    vs.step_badge(ax, sg.r + 0.4, sg.t - 0.4, 4, gid="step_4")
    vs.step_badge(ax, nx + 3.3, ny - 4.1, 5, gid="step_5")         # clear of the node rim

    # outer validation: fold bars
    fx, fw = 130.0, 33.0
    sites, sizes = vs.site_sizes()
    with vs.group(ax, "c_outer_validation"):
        vs.label(ax, fx, 90.3, "pooled 10-fold")
        vs.fold_bar(ax, fx, 92.4, fw, h=2.6, n=10, held=3, gid="fold_10")
        vs.label(ax, fx, 98.7, "leave-one-site-out")
        vs.fold_bar(ax, fx, 100.8, fw, h=2.6, sizes=sizes, held=11, gid="fold_loso")
        yl = 108.0
        swatch(ax, fx, yl, vs.VERM)
        vs.label(ax, fx + 2.6, yl, "held-out fold", va="center_baseline")
        x2 = fx + 2.6 + tw(ax, "held-out fold") + 3.0
        swatch(ax, x2, yl, vs.LANE, edge=vs.BLUE)
        vs.label(ax, x2 + 2.6, yl, "training", va="center_baseline")

    assert leg.b < BG["a"][1] and y_bus > BG["b"][0] and y_c > BG["c"][0] - 1.0
    assert lanebox.b < BG["c"][1], "training lane must sit inside block c"
    # ===================================== keys =========================================
    y_key = 124.5
    ak = vs.arrow_key(ax, 8.0, y_key, kinds=("data", "fit"), gid="arrow_key")
    with vs.group(ax, "participant_key"):
        x = ak.r + 8.0
        vs.dot(ax, x + 0.75, y_key, filled=True)
        vs.label(ax, x + 2.4, y_key, "autistic", va="center_baseline")
        x += 2.4 + tw(ax, "autistic") + 4.0
        vs.dot(ax, x + 0.75, y_key, filled=False)
        vs.label(ax, x + 2.4, y_key, "non-autistic", va="center_baseline")

    # ------------------------------------------------------------------ QA and save
    probs = vs.check_layout(fig)
    paths = vs.save_figure(fig, STEM, out_dir=out, copy_to=copy_to)
    print("legend right edge (mm):", round(leg.r, 2), "| key right (mm):", round(key.r, 2))
    print("inset (mm):", round(ix, 2), round(ix + iw, 2), "| lane:", lanebox.l, lanebox.r)
    print("example participant in illustrative training partition:",
          bool(data["ex_in_train"]), "| training n =", int(data["n_train"]))
    print("window starts:", [int(s) for s in data["starts"]])
    print("colour ranges (corr, static, windowed):", [m.meta["vlim"] for m in mboxes])
    print("layout problems:", len(probs))
    for p in paths:
        print("wrote", p)


def main() -> None:
    ap = argparse.ArgumentParser(description="Methods overview figure (Figure 1)")
    ap.add_argument("--out", default=str(ROOT / "results/latest_study/figures"))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE),
                    help=".npz cache of the computed matrices (delete to recompute)")
    ap.add_argument("--no-copy", action="store_true",
                    help="do not copy the outputs to ASD-journal-I/figures")
    args = ap.parse_args()
    build(Path(args.out), Path(args.cache) if args.cache else None,
          None if args.no_copy else vs.JOURNAL_FIG)


if __name__ == "__main__":
    main()
