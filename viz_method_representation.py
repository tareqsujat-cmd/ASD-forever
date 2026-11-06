"""
Methods figure 2: connectivity representations (Section 3.2 of the manuscript).

Draws, for one example participant of the strict cohort (NYU_0050952, 176 volumes,
CC200 regional time series from the C-PAC ``filt_noglobal`` derivative), the chain of
real intermediate quantities that produces each tangent-space representation, in the
shared glyph grammar of ``viz_style.py`` (redesign brief, section 2, "Figure 2"):

(a) static tangent member -- X_i -> Ledoit-Wolf covariance (step 1, Eq. 1; additional
    isotropic shrinkage r of Eq. 5 as chips, r = 0 primary) -> whitening by the training
    reference and matrix logarithm on the SPD dome (step 3, Eq. 3) -> S_i -> vecu (step 4,
    Eq. 4) -> x_i (19,900).  The shrinkage-correlation member branches from the same
    Ledoit-Wolf estimate.  The blue training lane holds the deck of training covariances
    (784/871) -> affine-invariant Riemannian mean (step 2, Eq. 2) -> reference, applied to
    the dome by a dashed blue arrow.
(b) windowed tangent member -- six numbered window bands of L = 50 volumes placed by
    Eq. (6) (step 5) -> Ledoit-Wolf, r = 0.15, projection at the dynamic reference ->
    deck x6 of window tangent matrices -> mean m_i (Eq. 7).  The dynamic lane holds the
    per-participant arithmetic means of the six window covariances -> Riemannian mean ->
    dynamic reference.

The references are illustrative: they are estimated exactly as the Methods specify (the
affine-invariant Riemannian mean with at most ten fixed-point updates and eigenvalues
clamped at 1e-6) from an illustrative training partition, namely the strict cohort minus
a random 10 % (numpy default_rng(42)), 784 of 871.  The deck cards are the matrices of the
first three participants of that training partition other than the example.  No result
of any analysis is shown.

The reference computation takes about a minute and is cached (``--cache``); delete the
cache files to force a recompute.

Usage:
    python viz_method_representation.py [--out results/latest_study/figures] [--no-copy]
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import viz_style as vs

ROOT = Path("/Users/reshadsazid/ASD-forever")
TS_DIR = ROOT / "abide_raw/ABIDE_pcp/cpac/filt_noglobal"
META = ROOT / "abide_processed/mri/metadata.csv"
EXAMPLE = TS_DIR / "NYU_0050952_rois_cc200.1D"
EXAMPLE_ID = 50952
STEM = "fig_representation"

K, L = 6, 50              # windows per participant, window length (volumes)
R_WIN = 0.15              # additional isotropic shrinkage of window covariances
EIG_FLOOR = 1e-6          # eigenvalue clamp for log and inverse square root
MAX_ITER = 10             # fixed-point updates of the Riemannian mean
N_DECK = 3                # training participants drawn as deck cards

H_MM = 126.0              # printed height (mm); width is the 165 mm text width


# ----------------------------------------------------------------------------------
# numerics (mirrors gpu_tangent.py / dynamic_fc.py, in float64 numpy)
# ----------------------------------------------------------------------------------

def load_ts(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#", dtype=np.float64)


def ledoit_wolf(ts: np.ndarray) -> np.ndarray:
    from sklearn.covariance import LedoitWolf
    return LedoitWolf().fit(ts).covariance_


def iso_shrink(C: np.ndarray, r: float) -> np.ndarray:
    """Eq. (5): (1 - r) C + r tr(C)/P I."""
    P = C.shape[-1]
    mu = np.trace(C, axis1=-2, axis2=-1) / P
    return (1.0 - r) * C + r * mu[..., None, None] * np.eye(P)


def _eigfun(M: np.ndarray, f) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    w = f(np.maximum(w, EIG_FLOOR))
    return (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)


def _inv_sqrt(M: np.ndarray) -> np.ndarray:
    return _eigfun(M, lambda w: 1.0 / np.sqrt(w))


def riemannian_mean(covs: np.ndarray, max_iter: int = MAX_ITER, tol: float = 1e-6):
    """Eq. (2): affine-invariant Riemannian mean by fixed-point iteration."""
    g = covs.mean(0)
    for _ in range(max_iter):
        w, V = np.linalg.eigh(g)
        sw = np.sqrt(np.maximum(w, EIG_FLOOR))
        g_sqrt = (V * sw) @ V.T
        g_isqrt = (V * (1.0 / sw)) @ V.T
        logs = _eigfun(g_isqrt @ covs @ g_isqrt, np.log)
        lm = logs.mean(0)
        if np.linalg.norm(lm) < tol:
            break
        wl, Vl = np.linalg.eigh(lm)
        g = g_sqrt @ ((Vl * np.exp(wl)) @ Vl.T) @ g_sqrt
    return g


def window_starts(T: int):
    """Eq. (6): nominal stride and rounded window starts."""
    delta = (T - L) / (K - 1)
    return delta, [int(round(k * delta)) for k in range(K)]


def window_covs(ts: np.ndarray) -> np.ndarray:
    _, starts = window_starts(ts.shape[0])
    return np.stack([iso_shrink(ledoit_wolf(ts[s:s + L]), R_WIN) for s in starts])


def tangent_safe(covs: np.ndarray, ref: np.ndarray, tries: int = 4) -> np.ndarray:
    """Eq. (3) with the adaptive diagonal jitter used for window covariances."""
    P = covs.shape[-1]
    isq = _inv_sqrt(ref)
    scale = np.diagonal(covs, axis1=-2, axis2=-1).mean()
    for k in range(tries):
        try:
            jit = 0.0 if k == 0 else (10.0 ** (-6 + k)) * scale
            return _eigfun(isq @ (covs + jit * np.eye(P)) @ isq, np.log)
        except np.linalg.LinAlgError:
            if k == tries - 1:
                raise
    raise RuntimeError("unreachable")


def _partition():
    """Strict-cohort time-series paths, illustrative training indices, example index."""
    meta = pd.read_csv(META)
    paths = []
    for sid in meta.subject_id:
        g = glob.glob(str(TS_DIR / f"*_{int(sid):07d}_rois_cc200.1D"))
        assert len(g) == 1, sid
        paths.append(Path(g[0]))
    n = len(meta)
    held = np.random.default_rng(42).choice(n, size=int(round(0.1 * n)), replace=False)
    train = np.setdiff1d(np.arange(n), held)
    ex_idx = int(np.where(meta.subject_id.values == EXAMPLE_ID)[0][0])
    return paths, train, ex_idx, n


def compute(cache: Path) -> dict:
    if cache.exists():
        return dict(np.load(cache))
    paths, train, ex_idx, n = _partition()
    print(f"strict cohort {n}; illustrative training partition {len(train)}; "
          f"example in training partition: {ex_idx in set(train)}")

    static, dyn = [], []
    for j, i in enumerate(train):
        ts = load_ts(paths[i])
        static.append(ledoit_wolf(ts))
        dyn.append(window_covs(ts).mean(0))      # arithmetic mean of the six windows
        if (j + 1) % 100 == 0:
            print(f"  covariances {j + 1}/{len(train)}")
    static = np.stack(static)
    dyn = np.stack(dyn)
    print("  static reference ...")
    ref_s = riemannian_mean(static)
    del static
    print("  dynamic reference ...")
    ref_d = riemannian_mean(dyn)
    del dyn

    X = load_ts(EXAMPLE)
    T, P = X.shape
    cov = ledoit_wolf(X)
    isq = _inv_sqrt(ref_s)
    white = isq @ cov @ isq
    S = _eigfun(white, np.log)
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    delta, starts = window_starts(T)
    wt = tangent_safe(window_covs(X), ref_d)
    out = dict(X=X, cov=cov, ref_s=ref_s, white=white, S=S, corr=corr, ref_d=ref_d,
               win_tan=wt, m=wt.mean(0), starts=np.array(starts), delta=np.array(delta),
               n_train=np.array(len(train)), n_cohort=np.array(n),
               ex_in_train=np.array(ex_idx in set(train)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def compute_deck(cache: Path) -> dict:
    """Deck cards: whole-scan Ledoit-Wolf covariances and per-participant arithmetic
    means of the six window covariances for the first N_DECK training participants
    (illustrative partition order) other than the example."""
    if cache.exists():
        return dict(np.load(cache))
    paths, train, ex_idx, _ = _partition()
    pick = [int(i) for i in train if i != ex_idx][:N_DECK]
    ts = [load_ts(paths[i]) for i in pick]
    out = dict(deck_static=np.stack([ledoit_wolf(t) for t in ts]),
               deck_dyn=np.stack([window_covs(t).mean(0) for t in ts]),
               deck_idx=np.array(pick),
               deck_files=np.array([paths[i].name for i in pick]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


# ----------------------------------------------------------------------------------
# colour limits (symmetric, 99th percentile of |off-diagonal|, rounded up)
# ----------------------------------------------------------------------------------

def nice(v: float) -> float:
    mag = 10 ** np.floor(np.log10(v))
    for c in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if c * mag >= v * 0.999:
            return c * mag
    return 10 * mag


def offdiag_lim(M: np.ndarray, q=99) -> float:
    iu = np.triu_indices(M.shape[-1], 1)
    return float(nice(np.percentile(np.abs(M[..., iu[0], iu[1]]), q)))


# ----------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------

def anchor_matrix(ax, x, y, size, mat, vlim, gid):
    """Reference matrix inside a training lane: real matrix with a blue frame."""
    with vs.group(ax, gid):
        b = vs.matrix_thumb(ax, x, y, size, mat, vlim=vlim, gid=f"{gid}_matrix")
        ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor=vs.BLUE,
                               linewidth=vs.LW_ARROW, zorder=vs.Z_GLYPH + 0.5))
    return b


def training_lane(ax, x0, top, deck_mats, deck_vlim, deck_label, ref, ref_vlim, ref_label,
                  n_train, n_cohort, gid):
    """Blue lane: padlock + 'training partition', deck of training matrices tagged
    'n_train/n_cohort' -> Riemannian mean (step 2) -> reference.  Returns the lane and
    reference boxes; the lane is 60 mm wide and 22 mm tall."""
    W, H = 60.0, 22.0
    with vs.group(ax, gid):
        ln = vs.lane(ax, x0, top, W, H, kind="train", text="training partition",
                     gid=f"{gid}_lane")
        card = 9.0
        dk = vs.deck(ax, x0 + 3.0, top + 5.5, card, mats=list(deck_mats), tag="",
                     vlim=deck_vlim, offset=1.26, gid=f"{gid}_deck")
        fr = dk.meta["front"]
        y_lab = top + H - 2.4
        tag = vs.chip(ax, fr.cx, y_lab, f"{n_train}/{n_cohort}", kind="train",
                      ha="center", gid=f"{gid}_n")
        if deck_label:
            vs.label(ax, tag.r + 1.0, y_lab, deck_label, va="center_baseline")
        ref_box = anchor_matrix(ax, x0 + 42.0, top + 5.5, 12.0, ref, ref_vlim,
                                gid=f"{gid}_reference")
        y_ar = ref_box.cy
        a0, a1 = dk.r + 0.8, ref_box.l - 1.2
        vs.arrow(ax, (a0, y_ar), (a1, y_ar), kind="data", gid=f"{gid}_mean_arrow")
        xm = (a0 + a1) / 2
        vs.label(ax, xm, y_ar - 2.0, "Riemannian mean", ha="center", va="center")
        vs.step_badge(ax, xm, y_ar - 5.4, 2, gid=f"{gid}_step2")
        vs.label(ax, ref_box.cx, y_lab, ref_label, ha="center", va="center_baseline")
    return ln, ref_box


def build(d: dict, dk: dict, out: Path, copy_to):
    vs.apply_style()
    vs.CHIP_NEUTRAL_FACE = "white"      # grey chips would vanish on the panel overlay
    fig, ax = vs.canvas(H_MM, bleed=vs.PANEL_BLEED)
    iu = np.triu_indices(200, 1)
    X = d["X"]
    T = X.shape[0]
    ntr, nco = int(d["n_train"]), int(d["n_cohort"])

    lim = dict(cov=offdiag_lim(d["cov"]), ref_s=offdiag_lim(d["ref_s"]),
               S=offdiag_lim(d["S"]), corr=1.0, win=offdiag_lim(d["win_tan"]),
               m=offdiag_lim(d["m"]), ref_d=offdiag_lim(d["ref_d"]),
               deck_s=offdiag_lim(dk["deck_static"]), deck_d=offdiag_lim(dk["deck_dyn"]))

    # ============================== panel a ==============================
    vs.panel_letter(ax, 1.5, 1.2, "a")
    yc = 20.0                                              # chain centre line
    with vs.group(ax, "a_chain"):
        # X_i: five network-coloured regional time series
        ts = vs.timeseries_glyph(ax, 3.0, 11.0, 15.0, 18.0, X, bands=None, gid="a_X")
        vs.label(ax, ts.cx, 8.4, vs.SYM["X"], ha="center")
        vs.chip(ax, ts.cx, 32.3, f"{T} × 200", kind="outline", ha="center", gid="a_X_size")

        # step 1: Ledoit-Wolf -> Sigma_hat_i (hero matrix)
        cv = vs.matrix_thumb(ax, 35.0, 8.0, 24.0, d["cov"], vlim=lim["cov"], gid="a_Sigma")
        vs.label(ax, cv.cx, 4.6, vs.SYM["Sigma_hat"], ha="center")
        a0, a1 = ts.r + 1.5, cv.meta["outer"].l - 0.9
        vs.arrow(ax, (a0, yc), (a1, yc), gid="a_arrow_lw")
        xm = (a0 + a1) / 2
        vs.label(ax, xm, yc - 2.4, "Ledoit–Wolf", ha="center")
        vs.step_badge(ax, xm, yc - 6.6, 1, gid="a_step1")

        # dome: whitening by the training reference and matrix log (step 3)
        dm = vs.dome(ax, 94.0, 8.0, 28.0, 24.0, point_label=vs.SYM["Sigma_hat"],
                     proj_label=vs.SYM["S"], gid="a_dome")
        b0, b1 = cv.r + 0.8, 92.7
        vs.arrow(ax, (b0, yc), (b1, yc), gid="a_arrow_whiten")
        xm = (b0 + b1) / 2
        vs.label(ax, xm, yc - 3.0, vs.SYM["whiten"], ha="center")
        vs.step_badge(ax, xm, yc - 7.8, 3, gid="a_step3")
        # additional isotropic shrinkage r (Eq. 5): r = 0 primary | 0.15
        c1w = vs.text_size_mm(ax, r"$r$ = 0 (primary)")[0] + 1.2
        c2w = vs.text_size_mm(ax, "0.15")[0] + 1.2
        cx0 = xm - (c1w + 0.8 + c2w) / 2
        vs.chip(ax, cx0, yc + 3.4, r"$r$ = 0 (primary)", kind="neutral", pad=0.6,
                gid="a_r_primary")
        vs.chip(ax, cx0 + c1w + 0.8, yc + 3.4, "0.15", kind="outline", pad=0.6,
                gid="a_r_planned")

        # log -> S_i
        sm = vs.matrix_thumb(ax, 128.0, 11.0, 18.0, d["S"], vlim=lim["S"], gid="a_S")
        vs.label(ax, sm.cx, 8.2, vs.SYM["S"], ha="center")
        c0, c1 = 123.3, sm.meta["outer"].l - 0.6
        vs.arrow(ax, (c0, yc), (c1, yc), gid="a_arrow_log")
        vs.label(ax, (c0 + c1) / 2, yc - 2.4, "log", ha="center")

        # step 4: vecu -> x_i (19,900)
        xb = vs.vector_bar(ax, 153.2, 11.0, 18.0, vec=d["S"][iu], vlim=lim["S"], gid="a_x")
        e0, e1 = sm.r + 0.8, xb.l - 0.8
        vs.arrow(ax, (e0, yc), (e1, yc), gid="a_arrow_vecu")
        xm = (e0 + e1) / 2
        vs.label(ax, xm, yc - 2.4, "vecu", ha="center")
        vs.step_badge(ax, xm, yc - 6.6, 4, gid="a_step4")
        vs.label(ax, xb.cx, 8.4, vs.SYM["x"], ha="center")
        vs.label(ax, xb.cx, xb.b + 2.2, "19,900", ha="center")
        tw = 1.5 + 1.0 + vs.text_size_mm(ax, vs.MEMBER_NAME["static"])[0]
        vs.member_tab(ax, sm.cx - tw / 2, sm.b + 1.4, "static", gid="a_tab_static")

    # shrinkage-correlation branch from Sigma_hat_i
    with vs.group(ax, "a_correlation"):
        cr = vs.matrix_thumb(ax, 35.0, 44.5, 12.0, d["corr"], vlim=lim["corr"],
                             gid="a_corr")
        xa = 41.0
        vs.arrow(ax, (xa, cv.b + 0.9), (xa, cr.meta["outer"].t - 0.7), gid="a_arrow_corr")
        vs.label(ax, xa - 1.2, (cv.b + cr.meta["outer"].t) / 2,
                 r"$\hat{\Sigma}_{i,pq}/(\hat{\sigma}_p\hat{\sigma}_q)$", ha="right")
        cb = vs.vector_bar(ax, 53.0, cr.t, cr.h, vec=d["corr"][iu], vlim=lim["corr"],
                           gid="a_corr_vec")
        vs.arrow(ax, (cr.r + 0.8, cr.cy), (cb.l - 0.8, cr.cy), gid="a_arrow_corr_vec")
        vs.label(ax, cb.cx, cb.b + 2.2, "19,900", ha="center")
        tw = 1.5 + 1.0 + vs.text_size_mm(ax, vs.MEMBER_NAME["corr"])[0]
        vs.member_tab(ax, cr.meta["outer"].l - 1.2 - tw, cr.cy - 1.6, "corr",
                      gid="a_tab_corr")

    # training lane: training covariances -> Riemannian mean -> reference -> dome
    ln_a, ref_a = training_lane(ax, 60.0, 41.0, dk["deck_static"], lim["deck_s"],
                                "covariances", d["ref_s"], lim["ref_s"], vs.SYM["Sigma_T"],
                                ntr, nco, gid="a_training")
    assert abs(ref_a.cx - dm.cx) < 1e-6, "reference must sit under the dome"
    vs.arrow(ax, (ref_a.cx, ref_a.meta["outer"].t - 0.7), (dm.cx, dm.b + 3.2),
             kind="fit", gid="a_apply_reference")

    # ============================== panel b ==============================
    y0 = 69.0
    bg_a_bottom = ln_a.b + 1.8
    assert bg_a_bottom <= y0 - 2.2, f"block a ({bg_a_bottom:.1f} mm) runs into block b"
    vs.panel_bg(ax, 0.0, 0.0, vs.PAGE_W, bg_a_bottom, gid="panel_bg_a")
    vs.panel_letter(ax, 1.5, y0 - 0.2, "b")
    yb = 89.0                                              # chain centre line
    with vs.group(ax, "b_chain"):
        starts = [int(s) for s in d["starts"]]
        tsb = vs.timeseries_glyph(ax, 3.0, 75.0, 36.0, 42.0, X, starts=starts, L=L,
                                  bands="stair", numbered=True, bracket=True, gid="b_X")
        vs.label(ax, tsb.r, 76.2, vs.SYM["X"], ha="right")
        yk = tsb.b + 3.2
        s5 = vs.step_badge(ax, tsb.l + 1.5, yk, 5, gid="b_step5")
        k6 = vs.chip(ax, s5.r + 1.0, yk, r"$K$ = 6", kind="outline", gid="b_K")
        vs.chip(ax, k6.r + 1.0, yk, rf"$T_i$ = {T}", kind="outline", gid="b_T")

        # per-window steps: Ledoit-Wolf (1), r = 0.15, project (3).  seg = 7.6 mm pushes
        # "project" (and the lane anchored under it) clear of window band 5 in Times.
        x = tsb.r + 1.3
        seg = 7.6
        nodes = {}
        for key, s in (("lw", "Ledoit–Wolf"), ("r", r"$r$ = 0.15"), ("proj", "project")):
            vs.arrow(ax, (x, yb), (x + seg, yb), gid=f"b_arrow_{key}")
            x += seg + 0.8
            w = vs.text_size_mm(ax, s)[0]
            vs.label(ax, x, yb, s, ha="left", va="center")
            nodes[key] = (x + w / 2, w)
            x += w + 0.8
        vs.step_badge(ax, nodes["lw"][0], yb - 4.0, 1, gid="b_step1")
        vs.step_badge(ax, nodes["proj"][0], yb - 4.0, 3, gid="b_step3")

        # deck x6 of window tangent matrices (shared colour scale)
        dx = 102.0
        vs.arrow(ax, (x, yb), (dx - 0.8, yb), gid="b_arrow_deck")
        card = 10.0
        wd = vs.deck(ax, dx, yb + 1.5 - card / 2 - 5 * 1.4, card, mats=list(d["win_tan"]),
                     tag="×6", offset=1.4, vlim=lim["win"], gid="b_windows")

        # mean -> m_i (Eq. 7), aligned under S_i
        mm = vs.matrix_thumb(ax, 128.0, 80.0, 18.0, d["m"], vlim=lim["m"], gid="b_m")
        f0, f1 = wd.r + 0.9, mm.meta["outer"].l - 0.6
        vs.arrow(ax, (f0, yb), (f1, yb), gid="b_arrow_mean")
        vs.label(ax, (f0 + f1) / 2, yb - 2.4, "mean", ha="center")
        vs.label(ax, mm.cx, 77.2, vs.SYM["m"], ha="center")
        tw = 1.5 + 1.0 + vs.text_size_mm(ax, vs.MEMBER_NAME["windowed"])[0]
        vs.member_tab(ax, mm.cx - tw / 2, mm.b + 1.4, "windowed", gid="b_tab_windowed")

    # dynamic training lane: window means -> Riemannian mean -> dynamic reference
    x_proj = nodes["proj"][0]
    ln_b, ref_b = training_lane(ax, x_proj - 48.0, 101.0, dk["deck_dyn"], lim["deck_d"],
                                "window means", d["ref_d"], lim["ref_d"],
                                "dynamic reference", ntr, nco, gid="b_training")
    band_end = tsb.meta["win"][-1][1]                      # right end of window band 5
    assert ln_b.l >= band_end + vs.GUTTER - 1e-6, \
        f"lane (b) starts at {ln_b.l:.1f} mm, window band 5 ends at {band_end:.1f} mm"
    assert x < 101.2 - 4.0, "no room for the arrow into the window deck"
    vs.arrow(ax, (ref_b.cx, ref_b.meta["outer"].t - 0.7), (x_proj, yb + 1.9), kind="fit",
             gid="b_apply_reference")

    # ============================== shared keys ==============================
    with vs.group(ax, "keys"):
        kx = 124.5                                         # rows end about 161 mm
        vs.cmap_key(ax, kx, 107.0, w=12.0, gid="matrix_key")
        keys = vs.SYS_KEYS
        vs.network_legend(ax, kx, 115.3, keys=keys[:4], gid="network_legend_1")
        vs.network_legend(ax, kx, 119.8, keys=keys[4:], gid="network_legend_2")

    # block b spans from just above its letter to below its lane and the shared keys
    bg_b_top = y0 - 1.2
    bg_b_bottom = min(max(ln_b.b, 121.0) + 1.8, H_MM - 0.2)
    assert bg_b_top >= bg_a_bottom + 1.0
    vs.panel_bg(ax, 0.0, bg_b_top, vs.PAGE_W, bg_b_bottom - bg_b_top, gid="panel_bg_b")
    probs = vs.check_layout(fig)
    paths = vs.save_figure(fig, STEM, out_dir=out, copy_to=copy_to)
    return paths, lim, probs


def main() -> None:
    ap = argparse.ArgumentParser(description="Methods figure: connectivity representations")
    ap.add_argument("--out", default=str(ROOT / "results/latest_study/figures"))
    ap.add_argument("--cache", default=str(ROOT / "results/latest_study/cache/"
                                             "fig_representation_refs.npz"))
    ap.add_argument("--deck-cache", default=str(ROOT / "results/latest_study/cache/"
                                                  "fig_representation_deck.npz"))
    ap.add_argument("--no-copy", action="store_true",
                    help="do not copy the outputs to the manuscript figures folder")
    args = ap.parse_args()
    d = compute(Path(args.cache))
    dk = compute_deck(Path(args.deck_cache))
    paths, lim, _ = build(d, dk, Path(args.out), None if args.no_copy else vs.JOURNAL_FIG)
    print("colour limits (±):", {k: float(f"{v:.3g}") for k, v in lim.items()})
    print("deck cards:", list(dk["deck_files"]))
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
