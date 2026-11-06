"""
viz_style.py -- shared style system for the three Methods figures
(fig_overview, fig_representation, fig_designs).

Implements section 1 of the redesign brief: colour tokens, typography, line weights, a
millimetre canvas and reusable glyph functions.  Every glyph draws into ONE full-page
"canvas" axes whose data units are millimetres measured from the TOP-LEFT corner of the
page (x to the right, y downwards), so a layout written in mm is also the printed layout.
Each glyph call is wrapped in a named group, so the SVG export contains
``<g id="...">`` elements that can be selected and edited by hand.

Dependencies: matplotlib, numpy and nilearn (atlas loading), plus the standard library.

Typical use::

    import viz_style as vs
    vs.apply_style()
    fig, ax = vs.canvas(height_mm=130)
    vs.panel_letter(ax, 0, 0, "a")
    m = vs.matrix_thumb(ax, 10, 6, 33, mat=corr, diag="blank")
    vs.arrow(ax, m.right(), (60, m.cy))
    vs.check_layout(fig)
    vs.save_figure(fig, "fig_representation")

Typeface: Times New Roman throughout, to match the manuscript and the Results figures.
Plain-text symbols missing from Times New Roman (the element-of and proportional-to signs,
superscript minus) and the calligraphic training-set symbol are taken from STIXGeneral, a
Times-compatible face, by font fallback.  Mathtext is set in Times New Roman
(``mathtext.fontset = "custom"``) with sub- and superscripts shrunk to 6 pt; use
``SYM["T"]`` for the calligraphic symbol.  Use ``\\hat``, not
``\\widehat``: the wide hat is drawn from a 4.4 pt STIX size variant.
"""

from __future__ import annotations

import csv
import functools
import logging
import colorsys
import re
import shutil
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.text import Text

# =========================================================================================
# paths (real data used by the glyphs)
# =========================================================================================
ROOT = Path(__file__).resolve().parent
ATLAS = ROOT / "atlas/cc200_roi_atlas.nii.gz"
ROI_TABLE = ROOT / "atlas/roi_table.csv"
META = ROOT / "abide_processed/mri/metadata.csv"
TS_DIR = ROOT / "abide_raw/ABIDE_pcp/cpac/filt_noglobal"
EXAMPLE = "NYU_0050952"
EXAMPLE_TS = TS_DIR / f"{EXAMPLE}_rois_cc200.1D"
FIG_OUT = ROOT / "results/latest_study/figures"
JOURNAL_FIG = ROOT / "ASD-journal-I/figures"

# =========================================================================================
# tokens
# =========================================================================================
# page (Computers in Biology and Medicine, cas-sc: text width 16.5 cm, height 22.4 cm)
PAGE_W = 165.0                      # mm
PAGE_H_MAX = 224.0                  # mm
MM = 1 / 25.4                       # inch per mm
PT = 25.4 / 72                      # mm per point

# neutrals
INK = "#1A1A1A"                     # text, glyph strokes
GREY = "#6E6E6E"                    # secondary text, data-flow arrows
RULE = "#BFBFBF"                    # rules, keylines
TINT = "#F3F3F3"                    # neutral glyph fill
PANEL_BG = "#F4F6F9"                # light overlay behind each lettered panel block
CHIP_NEUTRAL_FACE = TINT            # set to "white" when panels carry PANEL_BG
GLYPH_FACE = TINT                   # scaler/sigmoid icon boxes; "white" on PANEL_BG
# one light tint per lettered block, kept clear of the semantic hues (training blue,
# vermilion accent, member colours) so a block tint never reads as a data role
PANEL_TINTS = {"a": "#F2F0FA", "b": "#FCF4E3", "c": "#EAF6EF", "d": "#FAEEF3",
               "e": "#EDF3F9"}
# functional line colours used in the designs figure
MATCH = "#7B3294"                   # participant matching (pair links, matching arrows)
AGG = "#1B7837"                     # aggregation into systems and regions
LINEAR = "#01665E"                  # linear-classifier branch
TREE = "#8C510A"                    # tree-ensemble branch
WHITE = "#FFFFFF"

# roles
BLUE = "#0072B2"                    # training role (fitted on training data)
LANE = "#E6F1F8"                    # training-lane tint (black text passes 4.5:1)
VERM = "#D55E00"                    # THE accent: held-out data and leakage only, never text

# ensemble members: 1.5 mm tabs and strand lines only, never fills
MEMBER = {"corr": "#E69F00", "static": "#009E73", "windowed": "#CC79A7"}
MEMBER_NAME = {"corr": "Correlation", "static": "Static tangent",
               "windowed": "Windowed tangent"}

# functional systems (Yeo-7 + subcortical/cerebellar), order and colours as in the current
# viz_method_* scripts; roi_table.csv key, abbreviation, display name, colour
SYSTEMS = [
    ("visual", "VIS", "visual", "#781286"),
    ("sensorimotor", "SMN", "somatomotor", "#4682B4"),
    ("dorsal attention", "DAN", "dorsal attention", "#00760E"),
    ("salience / ventral attention", "SAL", "salience / ventral attention", "#C43AFA"),
    ("limbic", "LIM", "limbic", "#C7B84A"),
    ("frontoparietal control", "FPN", "frontoparietal control", "#E69422"),
    ("default mode", "DMN", "default mode", "#CD3E4E"),
    ("subcortical / cerebellar", "SUB", "subcortical / cerebellar", "#8A8A8A"),
]
SYS_KEYS = [s[0] for s in SYSTEMS]
SYS_ABBR = {s[0]: s[1] for s in SYSTEMS}
SYS_NAME = {s[0]: s[2] for s in SYSTEMS}     # 'sensorimotor' displayed as 'somatomotor'
SYS_COL = {s[0]: s[3] for s in SYSTEMS}
NET_CMAP = ListedColormap([s[3] for s in SYSTEMS], name="yeo7_sub")
TS_SYSTEMS = ["visual", "sensorimotor", "dorsal attention", "default mode",
              "subcortical / cerebellar"]      # the five traces of the time-series glyph

MATRIX_CMAP = "RdBu_r"              # zero at white; symmetric ranges go in the caption

# typography (points at printed size) -- no other sizes
FS = 7.0                            # every label
FS_LETTER = 8.0                     # bold lowercase panel letters
FS_SUB = 6.0                        # mathtext sub/superscripts (floor)
_SHRINK = 0.8572                    # 7 pt x 0.8572 = 6.0 pt

# line weights (points)
LW_KEY = 0.5                        # keylines, frames, glyph strokes
LW_ARROW = 1.3                      # bold arrows (user-directed)
LW_LANE = 1.0                       # training / held-out lane border
LW_MIN = 0.25                       # nothing thinner

# geometry (mm)
RADIUS = 1.5                        # corner radius of the few remaining containers
HEAD_L, HEAD_W = 2.4, 1.9           # filled arrowheads, scaled to the bold shaft
GUTTER = 3.0                        # inside a panel
PANEL_GAP = 6.0                     # between panels
DASH = (0, (4.0, 2.5))              # dash pattern (scaled by line width)

# notation (Times New Roman mathtext; the calligraphic T comes from STIXGeneral)
SYM = {
    "T": "𝒯",
    "Sigma_T": r"$\bar{\Sigma}_{𝒯}$",
    "Sigma_hat": r"$\hat{\Sigma}_i$",
    "whiten": r"$\bar{\Sigma}_{𝒯}^{-1/2}\hat{\Sigma}_i\;\bar{\Sigma}_{𝒯}^{-1/2}$",
    "X": r"$X_i$", "S": r"$S_i$", "x": r"$x_i$", "m": r"$m_i$",
    "pbar": r"$\bar{p}$", "l2": r"$\ell_2$", "scaler": r"$\mu/\sigma$",
}

ARROW_KINDS = {
    "data": dict(color=GREY, ls="-"),          # solid grey: data flow
    "fit": dict(color=BLUE, ls=DASH),          # dashed blue: fitted on training, applied
    "leak": dict(color=VERM, ls=DASH),         # dashed vermilion: held-out labels used
    "match": dict(color=MATCH, ls="-"),        # purple: participant matching
    "agg": dict(color=AGG, ls="-"),            # green: aggregation into systems/regions
    "linear": dict(color=LINEAR, ls="-"),      # teal: linear-classifier branch
    "tree": dict(color=TREE, ls="-"),          # brown: tree-ensemble branch
}
ARROW_KEY_TEXT = {"data": "data flow", "fit": "fitted, then applied",
                  "leak": "held-out labels used", "match": "participant matching",
                  "agg": "aggregation", "linear": "linear branch",
                  "tree": "tree-ensemble branch"}

# z-order bands
Z_LANE, Z_GLYPH, Z_ARROW, Z_TEXT = 1, 3, 4, 6


# =========================================================================================
# style
# =========================================================================================
# Values of scientific-visualization/assets/publication.mplstyle, overridden where the
# brief differs (7 pt text, Times New Roman at the author's request, 600 dpi, bbox standard,
# no compositing).
_PUBLICATION = {
    "figure.dpi": 100, "figure.facecolor": "white", "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "font.size": FS, "font.family": ["Times New Roman", "STIXGeneral", "Times",
                                     "DejaVu Serif"],
    "axes.linewidth": LW_KEY, "axes.labelsize": FS, "axes.titlesize": FS,
    "axes.labelweight": "normal", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.axisbelow": True,
    "axes.grid": False, "axes.unicode_minus": True,
    "axes.prop_cycle": matplotlib.cycler(color=["#0072B2", "#D55E00", "#009E73",
                                                "#CC79A7", "#000000"]),
    "xtick.major.size": 2, "xtick.minor.size": 1.5, "xtick.major.width": LW_KEY,
    "xtick.minor.width": LW_MIN, "xtick.labelsize": FS, "xtick.direction": "out",
    "ytick.major.size": 2, "ytick.minor.size": 1.5, "ytick.major.width": LW_KEY,
    "ytick.minor.width": LW_MIN, "ytick.labelsize": FS, "ytick.direction": "out",
    "xtick.color": INK, "ytick.color": INK,
    "lines.linewidth": LW_ARROW, "lines.markersize": 3, "lines.markeredgewidth": LW_KEY,
    "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
    "patch.linewidth": LW_KEY,
    "legend.fontsize": FS, "legend.frameon": False,
    "text.color": INK,
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "mathtext.sf": "Times New Roman", "mathtext.tt": "Times New Roman",
    "mathtext.cal": "Times New Roman:italic", "mathtext.default": "it",
    "mathtext.fallback": "stix",
    "savefig.dpi": 600, "savefig.format": "pdf", "savefig.bbox": "standard",
    "savefig.pad_inches": 0.0, "savefig.transparent": False,
    "savefig.facecolor": "white", "savefig.edgecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "asd-methods-figures",
    "image.composite_image": False, "image.interpolation": "nearest",
    "image.cmap": MATRIX_CMAP, "image.aspect": "auto",
}


def apply_style() -> None:
    """Set rcParams for the Methods figures (publication.mplstyle + the brief)."""
    matplotlib.use("Agg")
    matplotlib.rcParams.update(_PUBLICATION)
    import matplotlib._mathtext as _mt          # sub/superscripts at 6 pt, not 4.9 pt
    _mt.SHRINK_FACTOR = _SHRINK
    warnings.filterwarnings("ignore", message=".*findfont.*")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# =========================================================================================
# canvas, grid, groups, anchors
# =========================================================================================
@dataclass
class Box:
    """Rectangle in canvas mm (y downwards) with anchor helpers; `meta` carries extras."""
    x: float
    y: float
    w: float
    h: float
    meta: dict = field(default_factory=dict)

    @property
    def l(self): return self.x
    @property
    def r(self): return self.x + self.w
    @property
    def t(self): return self.y
    @property
    def b(self): return self.y + self.h
    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2

    def left(self, dy=0.0, pad=0.8): return (self.l - pad, self.cy + dy)
    def right(self, dy=0.0, pad=0.8): return (self.r + pad, self.cy + dy)
    def top(self, dx=0.0, pad=0.8): return (self.cx + dx, self.t - pad)
    def bottom(self, dx=0.0, pad=0.8): return (self.cx + dx, self.b + pad)


PANEL_BLEED = 1.6                   # mm added around the page where an outlined panel block meets its edge


def canvas(height_mm: float, width_mm: float = PAGE_W, bleed: float = 0.0):
    """Figure at printed size with one full-page axes in mm (origin top-left, y down).

    bleed > 0 adds that margin on all four sides without moving any content: coordinates stay
    0..width and 0..height, and panel blocks that meet the page edge extend into the margin
    (see panel_bg), so their outline stroke clears labels that start at the old page edge."""
    fig = plt.figure(figsize=((width_mm + 2 * bleed) * MM, (height_mm + 2 * bleed) * MM),
                     facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-bleed, width_mm + bleed)
    ax.set_ylim(height_mm + bleed, -bleed)
    ax.set_autoscale_on(False)
    ax.axis("off")
    ax.set_gid("canvas")
    ax._vs_size = (width_mm, height_mm)
    ax._vs_bleed = bleed
    return fig, ax


def inset(ax, x, y, w, h, **kw):
    """Child axes at a canvas mm rectangle (for plots that need their own axes)."""
    W, H = ax._vs_size
    b = getattr(ax, "_vs_bleed", 0.0)
    Wf, Hf = W + 2 * b, H + 2 * b
    a = ax.figure.add_axes([(x + b) / Wf, (H - y - h + b) / Hf, w / Wf, h / Hf], **kw)
    return a


class Grid:
    """12-column grid across the text width: col width = (width - 11 * gutter) / 12."""

    def __init__(self, width=PAGE_W, ncol=12, gutter=GUTTER):
        self.width, self.ncol, self.gutter = width, ncol, gutter
        self.col = (width - (ncol - 1) * gutter) / ncol

    def x(self, c: int) -> float:
        return c * (self.col + self.gutter)

    def span(self, c0: int, n: int) -> tuple[float, float]:
        """(x, w) of n columns starting at column c0."""
        return self.x(c0), n * self.col + (n - 1) * self.gutter


GRID = Grid()


class _Group(Artist):
    """Draws its children inside one named SVG group ``<g id=gid>``."""

    def __init__(self, children, gid, zorder):
        super().__init__()
        self._kids = list(children)
        self.set_gid(gid)
        self.set_zorder(zorder)

    def get_children(self):
        return list(self._kids)

    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group("glyph", gid=self.get_gid())
        for a in sorted(self._kids, key=lambda a: a.get_zorder()):
            if a.get_visible():
                a.draw(renderer)
        renderer.close_group("glyph")
        self.stale = False


_GID_SEEN: dict[int, dict[str, int]] = {}


def _unique_gid(fig, gid: str) -> str:
    gid = re.sub(r"[^A-Za-z0-9_\-]+", "_", gid).strip("_") or "glyph"
    seen = _GID_SEEN.setdefault(id(fig), {})
    n = seen.get(gid, 0)
    seen[gid] = n + 1
    return gid if n == 0 else f"{gid}_{n + 1}"


@contextmanager
def group(ax, gid: str):
    """Collect every artist added to `ax` inside the block into one named SVG group."""
    before = {id(a) for a in ax._children}
    yield
    new = [a for a in ax._children if id(a) not in before]
    if not new:
        return
    ax._children = [a for a in ax._children if id(a) in before]
    z = min(a.get_zorder() for a in new)
    g = _Group(new, _unique_gid(ax.figure, gid), z)
    g.axes = ax
    g.set_figure(ax.figure)
    ax._children.append(g)
    g._remove_method = ax._children.remove


def _grouped(default_gid):
    """Decorator: run the glyph inside group(ax, gid or default)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(ax, *args, gid=None, **kw):
            with group(ax, gid or default_gid):
                return fn(ax, *args, **kw)
        return wrapper
    return deco


# =========================================================================================
# text
# =========================================================================================
def _renderer(fig):
    try:
        return fig.canvas.get_renderer()
    except AttributeError:                                     # pragma: no cover
        return fig._get_renderer()


def text_size_mm(ax, s: str, size: float = FS, weight: str = "normal") -> tuple[float, float]:
    """Rendered (width, height) of a text string in mm."""
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    bb = t.get_window_extent(_renderer(ax.figure))
    t.remove()
    dpi = ax.figure.dpi
    return bb.width / dpi * 25.4, bb.height / dpi * 25.4


def label(ax, x, y, s, *, ha="left", va="center", color=INK, weight="normal",
          size=FS, z=Z_TEXT, **kw):
    """7 pt label (<= 4 words).  Vermilion text is refused (contrast about 4:1)."""
    if str(color).upper() == VERM:
        raise ValueError("vermilion is never used for text")
    return ax.text(x, y, s, ha=ha, va=va, color=color, fontweight=weight, fontsize=size,
                   zorder=z, **kw)


def panel_outline(face: str, value: float = 0.66, boost: float = 2.6) -> tuple:
    """Darker, more saturated version of a panel tint, used as its outline stroke."""
    h, s, v = colorsys.rgb_to_hsv(*matplotlib.colors.to_rgb(face))
    return colorsys.hsv_to_rgb(h, min(0.60, s * boost + 0.10), min(v, value))


LW_PANEL = 0.8                      # outline of a panel block (pt)


def panel_bg(ax, x, y, w, h, gid=None):
    """Light rounded overlay marking one lettered panel block, with an outline in a darker
    shade of its tint; drawn under every other artist (including training lanes) so it groups
    a panel without competing with it."""
    key = (gid or "").rsplit("_", 1)[-1]          # "panel_bg_c" -> block c tint
    face = PANEL_TINTS.get(key, PANEL_BG)
    b = getattr(ax, "_vs_bleed", 0.0)
    if b > 0:                                     # blocks that meet the page edge extend into the margin
        W_, H_ = ax._vs_size
        inset_ = 0.5 * LW_PANEL * PT              # keep the stroke inside the canvas
        x0 = x - b + inset_ if x <= 1e-6 else x
        y0 = y - b + inset_ if y <= 1e-6 else y
        x1 = x + w + b - inset_ if x + w >= W_ - 1e-6 else x + w
        y1 = y + h + b - inset_ if y + h >= H_ - 1e-6 else y + h
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.0",
                       facecolor=face, edgecolor=panel_outline(face), linewidth=LW_PANEL,
                       zorder=0.1)
    ax.add_patch(p)
    p.set_gid(gid or "panel_bg")
    return Box(x, y, w, h)


def panel_letter(ax, x, y, letter: str):
    """Bold lowercase panel letter, 8 pt, anchored at its top-left corner."""
    t = ax.text(x, y, letter.lower(), ha="left", va="top", fontsize=FS_LETTER,
                fontweight="bold", color=INK, zorder=Z_TEXT)
    t.set_gid(f"panel_{letter.lower()}")
    return t


# =========================================================================================
# data helpers (standard library + numpy)
# =========================================================================================
@functools.lru_cache(maxsize=1)
def roi_table() -> dict:
    """CC200 table: system key, excluded flag and system fraction per roi_id (1-based)."""
    with open(ROI_TABLE, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["roi_id"]))
    return dict(roi_id=np.array([int(r["roi_id"]) for r in rows]),
                system=[r["system"] for r in rows],
                excluded=np.array([r["excluded"].strip().lower() == "true" for r in rows]),
                system_frac=np.array([float(r["system_frac"]) for r in rows]))


@functools.lru_cache(maxsize=1)
def system_order() -> tuple[np.ndarray, list[int]]:
    """Region order grouped by system (Yeo order, then roi index) and block sizes."""
    tab = roi_table()
    rank = {k: i for i, k in enumerate(SYS_KEYS)}
    order = sorted(range(len(tab["system"])), key=lambda i: (rank[tab["system"][i]], i))
    counts = [sum(1 for s in tab["system"] if s == k) for k in SYS_KEYS]
    return np.array(order), counts


def ts_rois(systems=TS_SYSTEMS) -> list[int]:
    """Representative region (0-based column) per system: highest system fraction,
    excluded regions skipped (as in viz_method_overview.py)."""
    tab = roi_table()
    out = []
    for key in systems:
        idx = [i for i, s in enumerate(tab["system"]) if s == key and not tab["excluded"][i]]
        out.append(max(idx, key=lambda i: (tab["system_frac"][i], -i)))
    return out


def site_sizes() -> tuple[list[str], np.ndarray]:
    """Strict-cohort participants per recorded site label (sorted by label)."""
    with open(META, newline="") as f:
        sites = [r["site"] for r in csv.DictReader(f)]
    names = sorted(set(sites))
    return names, np.array([sites.count(s) for s in names])


def load_example_ts() -> np.ndarray:
    """Regional time series of the example participant (176 x 200)."""
    return np.loadtxt(EXAMPLE_TS, comments="#")


def window_starts(T: int, K: int = 6, L: int = 50) -> list[int]:
    """Eq. (6): s_ik = round(k (T - L) / (K - 1))."""
    delta = (T - L) / (K - 1)
    return [int(round(k * delta)) for k in range(K)]


def sym_range(mat: np.ndarray, q: float = 98) -> float:
    """Symmetric colour limit: q-th percentile of |off-diagonal|, two significant digits."""
    m = np.asarray(mat, float)
    iu = np.triu_indices(m.shape[-1], 1)
    v = float(np.nanpercentile(np.abs(m[..., iu[0], iu[1]]), q))
    return float(f"{v:.2g}")


# =========================================================================================
# glyphs: matrices and networks
# =========================================================================================
def _matrix_cmap():
    return matplotlib.colormaps[MATRIX_CMAP].with_extremes(bad="white")


def _frame(ax, x, y, w, h, color=GREY, lw=LW_KEY, z=Z_GLYPH + 0.2):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw,
                           zorder=z, joinstyle="miter"))


@_grouped("network_strip")
def network_strip(ax, x, y, length, orient="v", counts=None, width=0.9):
    """Colour strip of the eight systems; segment lengths ∝ counts (default equal).
    (x, y) is the top-left corner; orient 'v' runs downwards, 'h' to the right."""
    counts = np.ones(len(SYSTEMS)) if counts is None else np.asarray(counts, float)
    edges = np.concatenate([[0], np.cumsum(counts)]) / counts.sum() * length
    for k, (_, _, _, col) in enumerate(SYSTEMS):
        a, b = edges[k], edges[k + 1]
        if orient == "v":
            r = Rectangle((x, y + a), width, b - a)
        else:
            r = Rectangle((x + a, y), b - a, width)
        r.set(facecolor=col, edgecolor="none", linewidth=0, zorder=Z_GLYPH)
        ax.add_patch(r)
    return Box(x, y, width, length) if orient == "v" else Box(x, y, length, width)


def _cartoon_matrix(n=8, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.uniform(-0.45, 0.45, (n, n))
    m = (m + m.T) / 2
    m[np.diag_indices(n)] = rng.uniform(0.55, 0.9, n)
    return m


@_grouped("matrix")
def matrix_thumb(ax, x, y, size, mat=None, *, vlim=None, diag="keep", reorder=True,
                 strips="lt", block_lines="auto", seed=0, n_blocks=8, frame=True):
    """Matrix thumbnail with its top-left corner at (x, y), `size` mm square.

    mat=None draws an 8 x 8 block cartoon (diagonal blocks positive).  A real 200 x 200
    matrix is reordered by system, drawn with RdBu_r at +-vlim (default: 98th percentile
    of |off-diagonal|, returned in meta['vlim']) and, if diag='blank', its diagonal is
    left white.  strips: 'l', 't', 'lt' or '' for the system colour strips."""
    cmap = _matrix_cmap()
    sw = max(0.6, min(1.0, 0.035 * size))              # strip width
    sg = max(0.25, 0.012 * size)                        # strip gap
    if mat is None:
        m = _cartoon_matrix(n_blocks, seed)
        vlim = 1.0 if vlim is None else vlim
        counts = None
    else:
        m = np.asarray(mat, float)
        order, counts = system_order()
        if reorder and m.shape == (200, 200):
            m = m[np.ix_(order, order)]
        if diag == "blank":
            m = m.copy()
            np.fill_diagonal(m, np.nan)
        vlim = sym_range(m) if vlim is None else vlim
    ax.imshow(m, cmap=cmap, vmin=-vlim, vmax=vlim, extent=(x, x + size, y + size, y),
              origin="upper", interpolation="nearest", aspect="auto", zorder=Z_GLYPH)
    n = m.shape[0]
    if mat is None:                                     # white hairlines between blocks
        for k in range(1, n):
            p = x + k * size / n
            q = y + k * size / n
            ax.add_line(Line2D([p, p], [y, y + size], color="white", lw=LW_MIN,
                               zorder=Z_GLYPH + 0.1))
            ax.add_line(Line2D([x, x + size], [q, q], color="white", lw=LW_MIN,
                               zorder=Z_GLYPH + 0.1))
    elif (block_lines == "auto" and size >= 14) or block_lines is True:
        b = np.cumsum(counts)[:-1] / sum(counts) * size
        for p in b:
            ax.add_line(Line2D([x + p, x + p], [y, y + size], color=INK, lw=LW_MIN,
                               alpha=0.55, zorder=Z_GLYPH + 0.1))
            ax.add_line(Line2D([x, x + size], [y + p, y + p], color=INK, lw=LW_MIN,
                               alpha=0.55, zorder=Z_GLYPH + 0.1))
    if frame:
        _frame(ax, x, y, size, size)
    if "l" in strips:
        network_strip(ax, x - sg - sw, y, size, "v", counts, sw, gid="strip_left")
    if "t" in strips:
        network_strip(ax, x, y - sg - sw, size, "h", counts, sw, gid="strip_top")
    ox = x - (sg + sw if "l" in strips else 0)
    oy = y - (sg + sw if "t" in strips else 0)
    return Box(x, y, size, size, meta=dict(vlim=vlim, outer=Box(ox, oy, x + size - ox,
                                                                 y + size - oy)))


@_grouped("network_labels")
def network_labels(ax, box: Box, side="left", counts=None, gap=1.6, min_sep=2.7):
    """System abbreviations beside a real matrix (drawn once per figure), leader lines
    to block centres when labels must be spread apart."""
    if counts is None:
        counts = system_order()[1]
    edges = np.concatenate([[0], np.cumsum(counts)]) / sum(counts)
    L = box.h if side == "left" else box.w
    centres = list((edges[:-1] + edges[1:]) / 2 * L)
    pos = list(centres)
    for _ in range(300):                                 # spread to >= min_sep
        moved = False
        for k in range(1, len(pos)):
            d = pos[k] - pos[k - 1]
            if d < min_sep - 1e-9:
                pos[k - 1] -= (min_sep - d) / 2
                pos[k] += (min_sep - d) / 2
                moved = True
        pos[0] = max(pos[0], 0.0)
        pos[-1] = min(pos[-1], L)
        if not moved:
            break
    for key, c, p in zip(SYS_KEYS, centres, pos):
        if side == "left":
            xt = box.l - gap
            label(ax, xt - 1.2, box.t + p, SYS_ABBR[key], ha="right")
            ax.add_line(Line2D([xt - 1.0, xt], [box.t + p, box.t + c], color=RULE,
                               lw=LW_MIN, zorder=Z_GLYPH))
        else:
            yt = box.t - gap
            label(ax, box.l + p, yt - 1.2, SYS_ABBR[key], ha="center", va="bottom")
            ax.add_line(Line2D([box.l + p, box.l + c], [yt - 1.0, yt], color=RULE,
                               lw=LW_MIN, zorder=Z_GLYPH))
    return box


@_grouped("network_legend")
def network_legend(ax, x, y, chip_mm=1.8, gap=2.2, keys=None):
    """One row of colour chips with black abbreviations (full names go in the caption)."""
    keys = SYS_KEYS if keys is None else keys
    cx = x
    for key in keys:
        ax.add_patch(Rectangle((cx, y - chip_mm / 2), chip_mm, chip_mm,
                               facecolor=SYS_COL[key], edgecolor="none", zorder=Z_GLYPH))
        label(ax, cx + chip_mm + 0.7, y, SYS_ABBR[key], va="center")
        tw, _ = text_size_mm(ax, SYS_ABBR[key])
        cx += chip_mm + 0.7 + tw + gap
    return Box(x, y - chip_mm / 2, cx - gap - x, chip_mm)


@_grouped("matrix_key")
def cmap_key(ax, x, y, w=10.0, h=1.2):
    """The single shared '− 0 +' key per figure; (x, y) is the top-left of the bar."""
    g = np.linspace(-1, 1, 101)[None, :]
    ax.imshow(g, cmap=_matrix_cmap(), vmin=-1, vmax=1, extent=(x, x + w, y + h, y),
              origin="upper", interpolation="bilinear", aspect="auto", zorder=Z_GLYPH)
    _frame(ax, x, y, w, h, lw=LW_MIN)
    for s, xx, ha in (("−", x, "left"), ("0", x + w / 2, "center"), ("+", x + w, "right")):
        label(ax, xx, y + h + 0.5, s, ha=ha, va="top")
    return Box(x, y, w, h + 3.0)


@_grouped("vector")
def vector_bar(ax, x, y, h, vec=None, w=1.5, vlim=None, seed=1):
    """Column glyph of a feature vector (real vector or a 24-cell cartoon)."""
    if vec is None:
        v = np.random.default_rng(seed).uniform(-1, 1, 24)
        vlim = 1.0
    else:
        v = np.asarray(vec, float).ravel()
        vlim = float(f"{np.percentile(np.abs(v), 98):.2g}") if vlim is None else vlim
    ax.imshow(v[:, None], cmap=_matrix_cmap(), vmin=-vlim, vmax=vlim,
              extent=(x, x + w, y + h, y), origin="upper", interpolation="nearest",
              aspect="auto", zorder=Z_GLYPH)
    _frame(ax, x, y, w, h)
    return Box(x, y, w, h, meta=dict(vlim=vlim))


@_grouped("deck")
def deck(ax, x, y, size, n=3, mats=None, tag="×K", offset=None, vlim=None, seed=0):
    """Stacked matrices ('×K' deck).  (x, y, size) is the whole footprint's top-left and
    the card size; back cards step up-right.  mats: list of real matrices (front = mats[0])
    or None for block cartoons.  meta['front'] is the front card's Box."""
    offset = size * 0.14 if offset is None else offset
    if mats is not None:
        n = len(mats)
        if vlim is None:
            vlim = sym_range(np.asarray(mats))
    for k in reversed(range(n)):                         # back to front
        cx, cy = x + k * offset, y + (n - 1 - k) * offset
        z0 = Z_GLYPH + (n - k) * 0.01
        ax.add_patch(Rectangle((cx - 0.25, cy - 0.25), size + 0.5, size + 0.5,
                               facecolor="white", edgecolor="none", zorder=z0))
        m = None if mats is None else mats[k]
        with group(ax, f"card_{k}"):
            box = matrix_thumb(ax, cx, cy, size, m, vlim=vlim, strips="",
                               block_lines=False, seed=seed + k)
        for a in ax._children[-1].get_children():
            a.set_zorder(z0 + 0.001 + 0.0001 * a.get_zorder())
        ax._children[-1].set_zorder(z0)
    front = Box(x, y + (n - 1) * offset, size, size)
    if tag:                                              # clear of the back cards
        label(ax, x + size + (n - 1) * offset + 0.8, front.b, tag, ha="left",
              va="bottom")
    return Box(x, y, size + (n - 1) * offset, size + (n - 1) * offset,
               meta=dict(front=front, vlim=vlim))


# =========================================================================================
# glyphs: manifold, folds, lanes
# =========================================================================================
@_grouped("padlock")
def padlock(ax, x, y, size=2.2, color=BLUE):
    """Padlock: (x, y) is the top-left of a size x size box."""
    bw, bh = 0.78 * size, 0.52 * size
    bx, by = x + (size - bw) / 2, y + size - bh
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle=f"round,pad=0,"
                                f"rounding_size={0.12 * size}", facecolor=color,
                                edgecolor="none", zorder=Z_GLYPH + 0.5))
    r = 0.25 * size
    t = np.linspace(0, np.pi, 40)
    sx = x + size / 2 + r * np.cos(t)
    sy = by - 0.02 * size - (size - bh - 0.1 * size) * np.sin(t) * 0.95 + 0.0
    sx = np.concatenate([[sx[0]], sx, [sx[-1]]])
    sy = np.concatenate([[by + 0.05], sy, [by + 0.05]])
    ax.add_line(Line2D(sx, sy, color=color, lw=0.9 * size / 2.2, zorder=Z_GLYPH + 0.4,
                       solid_capstyle="butt"))
    ax.add_patch(Circle((x + size / 2, by + bh * 0.45), 0.09 * size, facecolor="white",
                        edgecolor="none", zorder=Z_GLYPH + 0.6))
    return Box(x, y, size, size)


@_grouped("dome")
def dome(ax, x, y, w, h, anchor_label=SYM["Sigma_T"], point_label=None, proj_label=None,
         lock=True, point_t=0.72):
    """SPD manifold: dome with the tangent plane at the training anchor (blue dot and
    padlock), one participant point on the dome and its projection on the plane.
    Returns the footprint Box; meta: anchor, point, proj (xy in mm), plane (corners)."""
    cx = x + w / 2
    yb = y + h                          # base line
    a = 0.40 * w                        # dome half-width
    b = 0.60 * h                        # dome height
    be = 0.10 * h                       # base-ellipse depth
    ya = yb - b                         # apex
    # dome surface: top arc (left -> right), then the front half of the base ellipse
    t = np.linspace(np.pi, 0, 80)
    top = np.c_[cx + a * np.cos(t), yb - b * np.sin(t)]
    tf = np.linspace(0, np.pi, 60)
    front = np.c_[cx + a * np.cos(tf), yb + be * np.sin(tf)]
    ax.add_patch(Polygon(np.r_[top, front], closed=True, facecolor=TINT, edgecolor=GREY,
                         linewidth=LW_KEY, zorder=Z_GLYPH, joinstyle="round"))
    # two latitude lines for curvature
    for phi in (0.45, 0.85):
        yy = yb - b * np.sin(phi)
        aa = a * np.cos(phi)
        tl = np.linspace(0, np.pi, 50)
        ax.add_line(Line2D(cx + aa * np.cos(tl), yy + be * np.cos(phi) * np.sin(tl),
                           color=RULE, lw=LW_MIN, zorder=Z_GLYPH + 0.1))
    # tangent plane at the apex (perspective parallelogram)
    pw, pd, sk = 0.92 * w, 0.24 * h, 0.07 * w
    plane = np.array([[cx - pw / 2 + sk, ya - pd / 2], [cx + pw / 2 + sk, ya - pd / 2],
                      [cx + pw / 2 - sk, ya + pd / 2], [cx - pw / 2 - sk, ya + pd / 2]])
    ax.add_patch(Polygon(plane, closed=True, facecolor=LANE, edgecolor=BLUE,
                         linewidth=LW_KEY, alpha=0.92, zorder=Z_GLYPH + 0.2,
                         joinstyle="miter"))
    # participant point on the dome (angle from the right base), geodesic to the apex
    tp = point_t
    P = (cx + a * np.cos(tp), yb - b * np.sin(tp))
    tg = np.linspace(tp, np.pi / 2, 30)
    ax.add_line(Line2D(cx + a * np.cos(tg), yb - b * np.sin(tg), color=GREY, lw=LW_KEY,
                       ls=(0, (1.2, 1.2)), zorder=Z_GLYPH + 0.3))
    # its projection: same distance from the anchor along the plane (log map)
    arc = np.sum(np.hypot(np.diff(a * np.cos(tg)), np.diff(b * np.sin(tg))))
    Q = (min(cx + arc, cx + pw / 2 - sk - 1.2), ya)
    ax.add_patch(Circle(P, 0.55, facecolor=INK, edgecolor="white", linewidth=0.3,
                        zorder=Z_GLYPH + 0.6))
    ax.add_patch(Circle(Q, 0.55, facecolor=INK, edgecolor="white", linewidth=0.3,
                        zorder=Z_GLYPH + 0.6))
    _arrow_path(ax, [P, Q], kind="data", rad=-0.35, shrink=(0.8, 0.8))
    # anchor
    A = (cx, ya)
    ax.add_patch(Circle(A, 0.75, facecolor=BLUE, edgecolor="white", linewidth=0.4,
                        zorder=Z_GLYPH + 0.7))
    lab_y = ya - pd / 2 - 0.6
    if anchor_label:
        label(ax, cx - 0.4, lab_y, anchor_label, ha="right", va="bottom")
    if lock:
        padlock(ax, cx + 0.2, lab_y - 2.5, 2.0, gid="anchor_lock")
    if point_label:                       # right of the point, clear of the dome edge
        label(ax, P[0] + 1.4, P[1], point_label, ha="left", va="center")
    if proj_label:
        label(ax, Q[0], ya - pd / 2 - 0.6, proj_label, ha="left", va="bottom")
    return Box(x, y, w, h, meta=dict(anchor=A, point=P, proj=Q, plane=plane))


@_grouped("fold_bar")
def fold_bar(ax, x, y, w, h=2.2, n=10, held=3, sizes=None, train_face=LANE):
    """Outer folds as a bar: n equal cells, or cells with widths ∝ `sizes` (LOSO).
    Cell `held` is vermilion (held out); the others carry the training tint."""
    widths = np.ones(n) if sizes is None else np.asarray(sizes, float)
    n = len(widths)
    xs = x + np.concatenate([[0], np.cumsum(widths)]) / widths.sum() * w
    for i in range(n):
        ax.add_patch(Rectangle((xs[i], y), xs[i + 1] - xs[i], h,
                               facecolor=VERM if i == held else train_face,
                               edgecolor="none", zorder=Z_GLYPH))
    for p in xs[1:-1]:
        ax.add_line(Line2D([p, p], [y, y + h], color="white", lw=LW_MIN * 1.6,
                           zorder=Z_GLYPH + 0.1))
    _frame(ax, x, y, w, h, color=BLUE, lw=LW_KEY)
    return Box(x, y, w, h, meta=dict(edges=xs))


@_grouped("lane")
def lane(ax, x, y, w, h, kind="train", lock=True, text=None):
    """Tinted training lane (1.0 pt blue border, padlock) or vermilion-outlined held-out
    lane (kind='heldout').  Optional short `text` after the padlock."""
    if kind == "train":
        face, edge = LANE, BLUE
    elif kind == "heldout":
        face, edge = "white", VERM
        lock = False if lock is True else lock
    else:
        raise ValueError(kind)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={RADIUS}",
                                facecolor=face, edgecolor=edge, linewidth=LW_LANE,
                                zorder=Z_LANE))
    tx = x + 1.5
    if lock:
        padlock(ax, x + 1.3, y + 1.2, 2.2, gid="lane_lock")
        tx = x + 1.3 + 2.2 + 1.0
    if text:
        label(ax, tx, y + 1.2 + 1.1, text, va="center")
    return Box(x, y, w, h)


# =========================================================================================
# glyphs: participants, arrows, badges, chips
# =========================================================================================
@_grouped("dots")
def dots(ax, x, y, w, h, n_filled, n_open, seed=0, d=None):
    """Participant cloud in a w x h box: filled = autistic, open = non-autistic (ink).
    Hexagonal jittered grid; class positions shuffled with default_rng(seed)."""
    n = n_filled + n_open
    pitch = np.sqrt(w * h / (n * 0.866))                # hexagonal packing
    ncol = max(1, int(np.floor(w / pitch - 0.5)))
    nrow = int(np.ceil(n / ncol))
    px, py = w / (ncol + 0.5), h / nrow                 # stretch to fill the box
    pitch = min(px, py / 0.866)
    rng = np.random.default_rng(seed)
    last = n - (nrow - 1) * ncol                        # dots in the last row, centred
    pts = np.array([(x + (c + 0.5 + 0.5 * (r % 2)
                          + ((ncol - last) / 2 if r == nrow - 1 else 0)) * px,
                     y + (r + 0.5) * py)
                    for r in range(nrow) for c in range(ncol)][:n])
    pts += rng.uniform(-0.10, 0.10, pts.shape) * pitch
    cls = rng.permutation(np.r_[np.ones(n_filled, bool), np.zeros(n_open, bool)])
    d = 0.62 * pitch if d is None else d
    ms = (d / PT) ** 2                                   # scatter size in pt^2
    ax.scatter(pts[cls, 0], pts[cls, 1], s=ms, marker="o", facecolor=INK, edgecolor=INK,
               linewidths=LW_MIN, zorder=Z_GLYPH, clip_on=False)
    ax.scatter(pts[~cls, 0], pts[~cls, 1], s=ms, marker="o", facecolor="white",
               edgecolor=INK, linewidths=LW_KEY, zorder=Z_GLYPH, clip_on=False)
    return Box(x, y, w, h, meta=dict(pitch=pitch, n=n))


def _dot(ax, x, y, filled=True, d=1.5):
    ax.add_patch(Circle((x, y), d / 2, facecolor=INK if filled else "white", edgecolor=INK,
                        linewidth=LW_KEY, zorder=Z_GLYPH + 0.5))


@_grouped("dot")
def dot(ax, x, y, filled=True, d=1.5):
    """One participant marker centred at (x, y): filled = autistic, open = non-autistic."""
    _dot(ax, x, y, filled, d)
    return Box(x - d / 2, y - d / 2, d, d)


@_grouped("pair")
def dot_pair(ax, x, y, gap=3.0, d=1.5, color=GREY):
    """Matched pair glyph ●–○ centred on the left dot at (x, y); link in `color`."""
    ax.add_line(Line2D([x, x + gap], [y, y], color=color, lw=LW_ARROW, zorder=Z_GLYPH))
    _dot(ax, x, y, True, d)
    _dot(ax, x + gap, y, False, d)
    return Box(x - d / 2, y - d / 2, gap + d, d)


def _bezier(p0, c, p1, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.asarray(p0) + 2 * (1 - t) * t * np.asarray(c) + t ** 2 * \
        np.asarray(p1)


def _arrow_path(ax, pts, kind="data", rad=0.0, head=True, lw=LW_ARROW, color=None,
                shrink=(0.0, 0.0)):
    sty = ARROW_KINDS[kind]
    color = sty["color"] if color is None else color
    pts = [np.asarray(p, float) for p in pts]
    if len(pts) == 2 and rad:
        p0, p1 = pts
        mid = (p0 + p1) / 2
        d = p1 - p0
        c = mid + rad * np.array([-d[1], d[0]])
        path = _bezier(p0, c, p1, 60)
    else:
        path = np.array(pts)
    # shrink both ends along the path
    def _trim(path, s, from_start):
        if s <= 0:
            return path
        seg = np.hypot(*np.diff(path, axis=0).T)
        cum = np.cumsum(seg) if from_start else np.cumsum(seg[::-1])
        k = int(np.searchsorted(cum, s))
        if k >= len(seg):
            return path
        if from_start:
            f = (cum[k] - s) / seg[k]
            q = path[k + 1] - f * (path[k + 1] - path[k])
            return np.vstack([q, path[k + 1:]])
        segr = seg[::-1]
        f = (cum[k] - s) / segr[k]
        i = len(path) - 1 - k
        q = path[i - 1] + f * (path[i] - path[i - 1])
        return np.vstack([path[:i], q])
    path = _trim(path, shrink[0], True)
    path = _trim(path, shrink[1], False)
    tip = path[-1]
    if head:
        # direction from the point HEAD_L back along the path
        seg = np.hypot(*np.diff(path, axis=0).T)
        back = np.cumsum(seg[::-1])
        k = int(np.searchsorted(back, HEAD_L))
        k = min(k, len(seg) - 1)
        base_pt = path[len(path) - 2 - k]
        u = tip - base_pt
        u = u / (np.hypot(*u) + 1e-12)
        base = tip - HEAD_L * u
        nrm = np.array([-u[1], u[0]])
        ax.add_patch(Polygon([tip, base + nrm * HEAD_W / 2, base - nrm * HEAD_W / 2],
                             closed=True, facecolor=color, edgecolor="none",
                             zorder=Z_ARROW + 0.1))
        # shaft stops just inside the head base
        path = _trim(path, HEAD_L - 0.15, False)
    ax.add_line(Line2D(path[:, 0], path[:, 1], color=color, lw=lw, ls=sty["ls"],
                       zorder=Z_ARROW, solid_joinstyle="miter", dash_capstyle="butt",
                       solid_capstyle="butt"))
    return tip


@_grouped("arrow")
def arrow(ax, *pts, kind="data", rad=0.0, head=True, lw=LW_ARROW, shrink=(0.0, 0.0)):
    """Arrow through two or more (x, y) mm points (elbows allowed); rad bends a
    two-point arrow.  kind: 'data' solid grey, 'fit' dashed blue, 'leak' dashed
    vermilion.  Filled head 1.8 mm long."""
    if len(pts) == 1:
        pts = pts[0]
    tip = _arrow_path(ax, pts, kind, rad, head, lw, shrink=shrink)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return Box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), meta=dict(tip=tip))


@_grouped("strand")
def strand(ax, *pts, member="corr", lw=1.0, head=False):
    """Member-coloured strand line through the given points (optional head)."""
    if len(pts) == 1:
        pts = pts[0]
    col = MEMBER[member]
    pts = np.asarray(pts, float)
    if head:
        _arrow_path(ax, pts, "data", 0.0, True, lw, color=col)
    else:
        ax.add_line(Line2D(pts[:, 0], pts[:, 1], color=col, lw=lw, zorder=Z_ARROW,
                           solid_joinstyle="round", solid_capstyle="butt"))
    return Box(pts[:, 0].min(), pts[:, 1].min(), np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))


@_grouped("arrow_key")
def arrow_key(ax, x, y, seg=6.0, gap=4.0, kinds=("data", "fit", "leak")):
    """One-line key of the three arrow meanings; (x, y) is the left end, centre line."""
    cx = x
    for k in kinds:
        _arrow_path(ax, [(cx, y), (cx + seg, y)], k)
        label(ax, cx + seg + 1.0, y, ARROW_KEY_TEXT[k], va="center")
        tw, _ = text_size_mm(ax, ARROW_KEY_TEXT[k])
        cx += seg + 1.0 + tw + gap
    return Box(x, y - 1.5, cx - gap - x, 3.0)


@_grouped("step")
def step_badge(ax, x, y, n, d=3.0, color=INK):
    """Circled step number (1-5) centred at (x, y); keyed to the caption."""
    ax.add_patch(Circle((x, y), d / 2, facecolor=color, edgecolor="none",
                        zorder=Z_TEXT - 0.5))
    label(ax, x, y + 0.05, str(n), ha="center", va="center_baseline", color="white",
          weight="bold")
    return Box(x - d / 2, y - d / 2, d, d)


@_grouped("chip")
def chip(ax, x, y, text, kind="neutral", ha="left", pad=0.9, h=3.2):
    """Small rounded tag.  kind: 'neutral' (grey tint), 'train' (lane tint, blue edge),
    'heldout' (white, vermilion edge; text stays ink), 'outline' (white, grey edge).
    (x, y): left/centre/right end (per ha) at the vertical centre."""
    neutral_edge = "none" if CHIP_NEUTRAL_FACE == TINT else RULE
    face, edge = {"neutral": (CHIP_NEUTRAL_FACE, neutral_edge), "train": (LANE, BLUE),
                  "heldout": ("white", VERM), "outline": ("white", GREY)}[kind]
    tw, _ = text_size_mm(ax, text)
    w = tw + 2 * pad
    x0 = {"left": x, "center": x - w / 2, "right": x - w}[ha]
    ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={min(1.0, h / 2)}",
                                facecolor=face, edgecolor=edge, linewidth=LW_KEY,
                                zorder=Z_TEXT - 0.5))
    label(ax, x0 + w / 2, y, text, ha="center", va="center_baseline")
    return Box(x0, y - h / 2, w, h)


@_grouped("member_tab")
def member_tab(ax, x, y, member, name=True, h=3.2, w=1.5):
    """1.5 mm member-colour tab at top-left (x, y) plus the member's two-word name."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=MEMBER[member], edgecolor="none",
                           zorder=Z_GLYPH))
    tw = 0.0
    if name:
        s = MEMBER_NAME[member] if name is True else name
        label(ax, x + w + 1.0, y + h / 2, s, va="center_baseline")
        tw = text_size_mm(ax, s)[0] + 1.0
    return Box(x, y, w + tw, h)


# =========================================================================================
# glyphs: icons
# =========================================================================================
@_grouped("sigmoid")
def sigmoid_glyph(ax, x, y, s=6.0):
    """ℓ2-logistic icon: S-curve in a rounded tinted square (top-left x, y)."""
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle=f"round,pad=0,rounding_size={RADIUS}",
                                facecolor=GLYPH_FACE, edgecolor="none", zorder=Z_GLYPH))
    t = np.linspace(-6, 6, 60)
    ax.add_line(Line2D(x + 0.15 * s + 0.7 * s * (t + 6) / 12,
                       y + 0.82 * s - 0.64 * s / (1 + np.exp(-t)), color=INK, lw=LW_ARROW,
                       zorder=Z_GLYPH + 0.2))
    ax.add_line(Line2D([x + 0.15 * s, x + 0.85 * s], [y + s / 2] * 2, color=RULE,
                       lw=LW_MIN, zorder=Z_GLYPH + 0.1))
    return Box(x, y, s, s)


@_grouped("scaler")
def scaler_glyph(ax, x, y, s=6.0):
    """Feature-scaling icon: μ/σ in a rounded tinted square."""
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle=f"round,pad=0,rounding_size={RADIUS}",
                                facecolor=GLYPH_FACE, edgecolor="none", zorder=Z_GLYPH))
    label(ax, x + s / 2, y + s / 2, SYM["scaler"], ha="center", va="center")
    return Box(x, y, s, s)


@_grouped("cylinder")
def cylinder(ax, x, y, w=7.0, h=8.0):
    """Dataset icon (top-left x, y)."""
    e = 0.16 * h
    t = np.linspace(0, np.pi, 40)
    cx, a = x + w / 2, w / 2
    body = np.r_[[[x, y + e / 2]],
                 np.c_[cx + a * np.cos(np.pi - t), y + h - e / 2 + e / 2 * np.sin(t)],
                 [[x + w, y + e / 2]]]
    ax.add_patch(Polygon(body, closed=True, facecolor=TINT, edgecolor=GREY,
                         linewidth=LW_KEY, zorder=Z_GLYPH))
    tt = np.linspace(0, 2 * np.pi, 80)
    ax.add_patch(Polygon(np.c_[cx + a * np.cos(tt), y + e / 2 + e / 2 * np.sin(tt)],
                         closed=True, facecolor="white", edgecolor=GREY, linewidth=LW_KEY,
                         zorder=Z_GLYPH + 0.1))
    for f in (0.40, 0.68):
        ax.add_line(Line2D(cx + a * np.cos(np.pi - t),
                           y + e / 2 + f * (h - e) + e / 2 * np.sin(t), color=RULE,
                           lw=LW_MIN, zorder=Z_GLYPH + 0.1))
    return Box(x, y, w, h)


@_grouped("funnel")
def funnel(ax, x, y, w=7.0, h=7.0):
    """Quality-control funnel icon (top-left x, y)."""
    neck = 0.22 * w
    cx = x + w / 2
    pts = [(x, y), (x + w, y), (cx + neck / 2, y + 0.62 * h), (cx + neck / 2, y + h),
           (cx - neck / 2, y + h), (cx - neck / 2, y + 0.62 * h)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=TINT, edgecolor=GREY,
                         linewidth=LW_KEY, zorder=Z_GLYPH, joinstyle="round"))
    return Box(x, y, w, h)


@_grouped("merge")
def merge_node(ax, x, y, text=SYM["pbar"], d=5.0):
    """Merge node (probability average) centred at (x, y)."""
    ax.add_patch(Circle((x, y), d / 2, facecolor="white", edgecolor=INK,
                        linewidth=LW_ARROW, zorder=Z_GLYPH + 0.5))
    label(ax, x, y, text, ha="center", va="center")
    return Box(x - d / 2, y - d / 2, d, d)


# =========================================================================================
# glyphs: brain and time series (real data)
# =========================================================================================
def _blur(a: np.ndarray, r: int) -> np.ndarray:
    k = np.ones(2 * r + 1) / (2 * r + 1)
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, a)


def _fill_nearest(lbl: np.ndarray, iters: int = 6) -> np.ndarray:
    out = lbl.copy()
    for _ in range(iters):
        z = out == 0
        if not z.any():
            break
        for sh in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = np.roll(out, sh, axis=(0, 1))
            take = z & (nb > 0) & (out == 0)
            out[take] = nb[take]
    return out


@functools.lru_cache(maxsize=8)
def _brain_rgb(view: str, highlight: str | None, up: int = 4,
               mark_excluded: bool = True) -> np.ndarray:
    from nilearn import image                      # nilearn loads the CC200 atlas
    img = image.load_img(str(ATLAS))
    lab = np.asarray(img.dataobj).astype(int)       # (x, y, z); x = 93 - 3 i (mm)
    tab = roi_table()
    sys_of = np.zeros(lab.max() + 1, int)
    for rid, s, ex in zip(tab["roi_id"], tab["system"], tab["excluded"]):
        sys_of[rid] = -1 if (ex and mark_excluded) else SYS_KEYS.index(s) + 1
    nz = lab > 0
    if view == "lateral":          # left hemisphere seen from the left, anterior left
        n = lab.shape[0]
        first = np.where(nz.any(0), n - 1 - np.argmax(nz[::-1], axis=0), -1)   # (y, z)
        jj, kk = np.meshgrid(np.arange(lab.shape[1]), np.arange(lab.shape[2]),
                             indexing="ij")
        hit = np.where(first >= 0, lab[np.clip(first, 0, None), jj, kk], 0)
        depth = np.where(first >= 0, first, 0).astype(float)
        hit, depth = hit.T[::-1, ::-1], depth.T[::-1, ::-1]                    # rows z
    elif view == "dorsal":         # seen from above, anterior up, left hemisphere left
        n = lab.shape[2]
        first = np.where(nz.any(2), n - 1 - np.argmax(nz[:, :, ::-1], axis=2), -1)
        ii, jj = np.meshgrid(np.arange(lab.shape[0]), np.arange(lab.shape[1]),
                             indexing="ij")
        hit = np.where(first >= 0, lab[ii, jj, np.clip(first, 0, None)], 0)
        depth = np.where(first >= 0, first, 0).astype(float)
        hit, depth = hit.T[::-1, ::-1], depth.T[::-1, ::-1]                    # rows y
    else:
        raise ValueError(view)
    hit_u = np.kron(hit, np.ones((up, up), int))
    dep_u = np.kron(depth, np.ones((up, up)))
    mask = _blur((hit_u > 0).astype(float), up) > 0.5
    hit_u = _fill_nearest(hit_u, up + 2)
    hit_u[~mask] = 0
    d = dep_u[mask]
    dn = np.zeros_like(dep_u)
    dn[mask] = (d - d.min()) / (np.ptp(d) + 1e-9)
    shade = 0.72 + 0.28 * dn
    grey = np.array(to_rgb("#D4D4D4"))
    cols = np.vstack([[1, 1, 1], [to_rgb(s[3]) for s in SYSTEMS]])
    s_u = sys_of[hit_u]
    rgb = np.ones(hit_u.shape + (3,))
    for k in range(1, 9):
        on = s_u == k
        c = cols[k] if (highlight is None or SYS_KEYS[k - 1] == highlight) else grey
        rgb[on] = c
    rgb[s_u == -1] = grey                               # the 4 excluded regions
    rgb[mask] = 1 - (1 - rgb[mask]) * 1.0
    rgb[mask] *= shade[mask][:, None]
    # parcel boundaries (lightened) and silhouette (grey)
    bd = np.zeros(hit_u.shape, bool)
    bd[:, 1:] |= (hit_u[:, 1:] != hit_u[:, :-1]) & (hit_u[:, 1:] > 0) & (hit_u[:, :-1] > 0)
    bd[1:, :] |= (hit_u[1:, :] != hit_u[:-1, :]) & (hit_u[1:, :] > 0) & (hit_u[:-1, :] > 0)
    rgb[bd] = 0.5 * rgb[bd] + 0.5
    edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1)
                    & np.roll(mask, -1, 1))
    edge |= mask & ~(np.roll(mask, 2, 0) & np.roll(mask, -2, 0) & np.roll(mask, 2, 1)
                     & np.roll(mask, -2, 1))
    rgb[edge] = to_rgb(GREY)
    rgb[~mask] = 1.0
    rr, cc = np.where(mask)
    rgba = np.dstack([rgb, mask.astype(float)])
    return rgba[rr.min():rr.max() + 1, cc.min():cc.max() + 1]


@_grouped("brain")
def brain_glyph(ax, x, y, w, h=None, view="lateral", highlight=None, mark_excluded=True):
    """CC200 atlas (loaded with nilearn) projected to a lateral or dorsal view, parcels
    coloured by system (highlight: one system key in colour, others grey).  The picture
    is fitted into the w x h box (top-left x, y) keeping its aspect ratio.
    mark_excluded=True greys the four regions excluded from the group-difference
    analyses; False colours them by their system (all 200 regions enter prediction)."""
    pic = _brain_rgb(view, highlight, mark_excluded=mark_excluded)
    ar = pic.shape[1] / pic.shape[0]
    h = w / ar if h is None else h
    pw, ph = (w, w / ar) if w / ar <= h else (h * ar, h)
    px, py = x + (w - pw) / 2, y + (h - ph) / 2
    ax.imshow(pic, extent=(px, px + pw, py + ph, py), origin="upper",
              interpolation="nearest", aspect="auto", zorder=Z_GLYPH)
    return Box(px, py, pw, ph)


@_grouped("timeseries")
def timeseries_glyph(ax, x, y, w, h, ts, starts=None, L=50, bands="overlay",
                     numbered=False, bracket=False, systems=TS_SYSTEMS, band_h=None):
    """Five network-coloured regional time series (real, z-scored, no axes) with the
    window positions.  bands: 'overlay' (translucent bands over the traces), 'stair'
    (one translucent bar per window below the traces), or None.  numbered writes the
    window index k = 0..K-1 in the stair bars; bracket adds 'L = 50' over window 0.
    meta: x-positions of the windows ('win', list of (x0, x1)) and the trace box."""
    ts = np.asarray(ts, float)
    T = ts.shape[0]
    starts = window_starts(T, L=L) if starts is None else list(starts)
    K = len(starts)
    xv = lambda v: x + v / T * w                      # volume index -> mm
    top_pad = 3.4 if bracket else 0.0
    if bands == "stair":
        band_h = (2.5 if numbered else 0.9) if band_h is None else band_h
        stair_h = K * band_h + (K - 1) * 0.3
        tr_h = h - top_pad - stair_h - 1.0
    else:
        tr_h = h - top_pad
    ty = y + top_pad
    n = len(systems)
    pitch = tr_h / n
    for j, (key, col) in enumerate(zip(systems, ts_rois(systems))):
        v = ts[:, col]
        v = (v - v.mean()) / (v.std() + 1e-12)
        v = np.clip(v, -3, 3)
        yc = ty + (j + 0.5) * pitch
        ax.add_line(Line2D(xv(np.arange(T) + 0.5), yc - v / 3 * pitch * 0.62,
                           color=SYS_COL[key], lw=LW_KEY, zorder=Z_GLYPH + 0.2,
                           solid_joinstyle="round"))
    wins = [(xv(s), xv(s + L)) for s in starts]
    if bands == "overlay":
        for k, (a, b) in enumerate(wins):
            ax.add_patch(Rectangle((a, ty), b - a, tr_h, facecolor=GREY, alpha=0.06,
                                   edgecolor="none", zorder=Z_GLYPH))
            ax.add_line(Line2D([a, a], [ty, ty + tr_h], color=GREY, lw=LW_MIN,
                               alpha=0.8, zorder=Z_GLYPH + 0.1))
    elif bands == "stair":
        sy = ty + tr_h + 1.0
        for k, (a, b) in enumerate(wins):
            yy = sy + k * (band_h + 0.3)
            ax.add_patch(Rectangle((a, yy), b - a, band_h, facecolor=GREY, alpha=0.28,
                                   edgecolor="none", zorder=Z_GLYPH))
            if numbered:
                label(ax, a + 0.8, yy + band_h / 2, str(k), va="center_baseline")
    if bracket:
        a, b = wins[0]
        yb = y + top_pad - 0.6
        ax.add_line(Line2D([a, a, b, b], [yb + 0.7, yb, yb, yb + 0.7], color=GREY,
                           lw=LW_KEY, zorder=Z_GLYPH + 0.2))
        label(ax, (a + b) / 2, yb - 0.3, f"$L$ = {L}", ha="center", va="bottom")
    return Box(x, y, w, h, meta=dict(win=wins, trace=Box(x, ty, w, tr_h)))


# =========================================================================================
# QA and export
# =========================================================================================
def _words(s: str) -> int:
    """Words in a label: math counts as one, 'K = 6' as one, bare symbols as none."""
    s = re.sub(r"\$[^$]*\$", "X", s)
    s = re.sub(r"\s*=\s*", "=", s)
    return len([t for t in s.split() if re.search(r"\w", t)])


def check_layout(fig, min_pt: float = FS_SUB, max_words: int = 4, tol_pt: float = 0.3,
                 verbose: bool = True) -> list[str]:
    """Text QA on the drawn figure: size floor, <= max_words per label, text-text
    overlaps (> tol_pt in both directions), text leaving the page, vermilion text."""
    fig.canvas.draw()
    r = _renderer(fig)
    texts = [t for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and t.axes is not None]
    fb = fig.bbox
    k = 72.0 / fig.dpi                                    # px -> pt
    boxes = [t.get_window_extent(r) for t in texts]
    probs = []
    for t, bb in zip(texts, boxes):
        s = t.get_text()
        if t.get_fontsize() < min_pt - 1e-6:
            probs.append(f"size {t.get_fontsize():.2f} pt < {min_pt}: {s!r}")
        if _words(s) > max_words:
            probs.append(f"{_words(s)} words > {max_words}: {s!r}")
        if (bb.x0 < fb.x0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y0 < fb.y0 - 0.5
                or bb.y1 > fb.y1 + 0.5):
            probs.append(f"leaves the page: {s!r}")
        if matplotlib.colors.to_hex(t.get_color()).upper() == VERM:
            probs.append(f"vermilion text: {s!r}")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = boxes[i], boxes[j]
            ox = (min(a.x1, b.x1) - max(a.x0, b.x0)) * k
            oy = (min(a.y1, b.y1) - max(a.y0, b.y0)) * k
            if ox > tol_pt and oy > tol_pt:
                probs.append(f"overlap: {texts[i].get_text()!r} / {texts[j].get_text()!r}")
    if verbose:
        for p in probs:
            print("LAYOUT", p)
        print(f"layout check: {len(texts)} texts, {len(probs)} problem(s)")
    return probs


def save_figure(fig, stem: str, out_dir: Path | str | None = FIG_OUT,
                copy_to: Path | str | None = JOURNAL_FIG, dpi: int = 600,
                formats=("pdf", "svg", "png")) -> list[Path]:
    """Export <stem>.pdf/.svg/.png at printed size (bbox_inches=None, 600 dpi) to
    out_dir and copy them to copy_to (the manuscript figures folder) unless None."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"pdf": {"Creator": "viz_style.py (matplotlib)", "CreationDate": None},
            "svg": {"Creator": "viz_style.py (matplotlib)", "Date": None},
            "png": {"Software": "viz_style.py (matplotlib)"}}
    paths = []
    for ext in formats:
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches=None, facecolor="white",
                    metadata=meta.get(ext))
        paths.append(p)
    if copy_to is not None:
        copy_to = Path(copy_to)
        copy_to.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, copy_to / p.name)
    return paths


def pdf_fonts(path: Path | str) -> list[str]:
    """Embedded font names of a PDF (standard library; substitute for pdffonts)."""
    data = Path(path).read_bytes()
    names = set(re.findall(rb"/BaseFont\s*/(?:[A-Z]{6}\+)?([A-Za-z0-9\-_,]+)", data))
    return sorted(n.decode() for n in names)
