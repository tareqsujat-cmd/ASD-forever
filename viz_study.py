"""
Conference-style figures and a navigable report for the whole study.

Every comparison in the paper — representation, classifier, rank, atlas, fusion,
harmonisation, protocol, leakage audit, spectral structure and symptom association — is
rendered through the same small set of primitives, so that all panels share one palette,
one typography, one bar geometry and one way of marking the proposed method. Reading the
figures then requires learning the conventions once.

Conventions
-----------
* the proposed configuration is drawn in the accent colour, comparisons in muted blue,
  and a deliberately compromised condition (leakage audit) in red;
* every bar carries its value, and error bars show the standard deviation across folds;
* horizontal bars are used whenever labels are long, vertical bars when the axis is
  ordered (rank, coverage);
* chance level is marked wherever a score has one.

Usage:
    python viz_study.py --dir results/latest_study            # figures + report
    python viz_study.py --dir results/<run> --out figures     # figures elsewhere
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# shared appearance
# --------------------------------------------------------------------------

ACCENT = "#c0392b"      # the proposed configuration
BASE = "#4a7ab5"        # comparisons
MUTED = "#9aa5b1"       # context / weaker variants
WARN = "#b23b3b"        # compromised protocol
GOOD = "#1a8a3a"        # reference / honest protocol

# conditions drawn in the accent colour wherever they appear
HIGHLIGHT = {
    "tangent (static)", "logistic (dense linear)", "honest",
    "cc200+aal+ho", "fusion: mean", "pooled",
    "dynamic (fused): mean+dispersion+velocity",
    "low-rank spectral (r=20)", "no harmonisation",
}


def _style():
    from visualization.style import ieee_style
    return ieee_style


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _colors(labels, highlight=HIGHLIGHT, accent=ACCENT, base=BASE):
    return [accent if l in highlight else base for l in labels]


def _finish(ax, xlabel=None, ylabel=None, title=None, chance=None, vertical=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if chance is not None:
        (ax.axhline if vertical else ax.axvline)(
            chance, ls=":", lw=1.1, color="#777777", zorder=0)
    ax.grid(axis="y" if vertical else "x", lw=0.5, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9.5)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

def hbar_panel(ax, labels, values, errs=None, highlight=HIGHLIGHT, xlabel="AUROC",
               title=None, chance=0.5, sort=True, colors=None, annotate=True):
    """Horizontal comparison bars; used whenever condition names are long."""
    d = pd.DataFrame({"l": labels, "v": values,
                      "e": errs if errs is not None else np.zeros(len(labels))})
    d = d.dropna(subset=["v"])
    if sort:
        d = d.sort_values("v")
    y = np.arange(len(d))
    cols = colors or _colors(d["l"].tolist(), highlight)
    ax.barh(y, d["v"], xerr=d["e"], color=cols, edgecolor="black", linewidth=0.7,
            capsize=2.5, error_kw={"elinewidth": 0.9}, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["l"], fontsize=7.5)
    lo = min(chance if chance is not None else d["v"].min(), d["v"].min()) - 0.03
    hi = float((d["v"] + d["e"]).max())
    span = max(hi - lo, 1e-6)
    # labels are placed clear of the error bar, and the axis is widened to fit them
    if annotate:
        for yi, v, e in zip(y, d["v"], d["e"]):
            ax.text(v + e + 0.03 * span, yi, f"{v:.3f}", va="center",
                    fontsize=7, zorder=4)
        ax.set_xlim(max(0.0, lo), hi + 0.16 * span)
    else:
        ax.set_xlim(max(0.0, lo), hi + 0.04 * span)
    _finish(ax, xlabel=xlabel, title=title, chance=chance)
    return d


def curve_panel(ax, x, y, err=None, xlabel="", ylabel="", title=None, logx=False,
                color=ACCENT, marker="o", label=None, invert_x=False):
    """Ordered-axis comparison: rank sweeps, coverage curves, spectra."""
    ax.errorbar(x, y, yerr=err, fmt=f"-{marker}", lw=2.2, ms=4.5, capsize=2.5,
                color=color, label=label, zorder=3)
    if logx:
        ax.set_xscale("log")
    if invert_x:
        ax.invert_xaxis()
    _finish(ax, xlabel=xlabel, ylabel=ylabel, title=title, vertical=True)


# --------------------------------------------------------------------------
# figure builders
# --------------------------------------------------------------------------

TITLES = {
    "representation": "Connectivity representation",
    "classifier": "Classifier",
    "atlas": "Atlas combination",
    "fusion": "Fusion rule",
    "harmonisation": "Site harmonisation",
    "protocol": "Evaluation protocol",
}


def fig_main(d: Path, out: Path):
    """Headline figure: the comparisons a reader should see first."""
    plt = _mpl()
    ieee_style = _style()
    have = {k: (d / f"{k}.csv") for k in ("representation", "classifier")}
    rank_f, leak_f = d / "rank.csv", d / "leakage_audit.csv"
    panels = [k for k, f in have.items() if f.exists()]
    n = len(panels) + int(rank_f.exists()) + int(leak_f.exists())
    if n == 0:
        return
    with ieee_style():
        fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 3.6))
        axes = np.atleast_1d(axes)
        i = 0
        for k in panels:
            df = pd.read_csv(have[k])
            hbar_panel(axes[i], df["condition"], df["auroc"], df.get("auroc_std"),
                       title=TITLES[k])
            i += 1
        if rank_f.exists():
            df = pd.read_csv(rank_f)
            r = df["condition"].str.extract(r"rank=(\d+)").astype(float)[0]
            o = np.argsort(r.to_numpy())
            curve_panel(axes[i], r.to_numpy()[o], df["auroc"].to_numpy()[o],
                        df.get("auroc_std", pd.Series(np.zeros(len(df)))).to_numpy()[o],
                        xlabel="components retained $r$", ylabel="AUROC",
                        title="Compression", logx=True)
            i += 1
        if leak_f.exists():
            df = pd.read_csv(leak_f)
            cols = [GOOD if v == "honest" else WARN for v in df["variant"]]
            hbar_panel(axes[i], df["variant"], df["auroc"], df.get("auroc_std"),
                       title="Evaluation protocol", colors=cols)
            i += 1
        fig.tight_layout()
        _save(fig, out, "fig1_main_results")


def fig_ablations(d: Path, out: Path):
    """Every remaining ablation on one page, drawn identically."""
    plt = _mpl()
    ieee_style = _style()
    keys = [k for k in ("atlas", "fusion", "harmonisation", "protocol")
            if (d / f"{k}.csv").exists()]
    if not keys:
        return
    ncol = min(2, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    with ieee_style():
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.4 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, k in zip(axes, keys):
            df = pd.read_csv(d / f"{k}.csv")
            hbar_panel(ax, df["condition"], df["auroc"], df.get("auroc_std"),
                       title=TITLES.get(k, k))
        for ax in axes[len(keys):]:
            ax.axis("off")
        fig.tight_layout()
        _save(fig, out, "fig2_ablations")


def fig_representation_detail(d: Path, out: Path):
    """Dynamic-connectivity blocks: concatenation against late fusion."""
    f = d / "representation.csv"
    if not f.exists():
        return
    plt = _mpl(); ieee_style = _style()
    df = pd.read_csv(f)
    conc = df[~df["condition"].str.contains("fused", na=False)]
    fuse = df[df["condition"].str.contains("fused", na=False)]
    if fuse.empty:
        return
    with ieee_style():
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
        hbar_panel(axes[0], conc["condition"], conc["auroc"], conc.get("auroc_std"),
                   title="Blocks concatenated")
        hbar_panel(axes[1], fuse["condition"], fuse["auroc"], fuse.get("auroc_std"),
                   title="Blocks fused at the decision level")
        fig.tight_layout()
        _save(fig, out, "fig3_representation")


def _find(d: Path, name: str) -> Path | None:
    """Locate an artefact whether it sits with the tables or in the raw directory."""
    for cand in (d / name, d.parent / "raw" / name, d.parent / name,
                 d.parent / "ablations" / name):
        if cand.exists():
            return cand
    return None


def fig_spectrum(d: Path, out: Path):
    f = _find(d, "spectrum_analysis.json")
    if f is None:
        return
    plt = _mpl(); ieee_style = _style()
    s = json.loads(f.read_text())
    ar = s.get("auroc_by_rank", {})
    if not ar:
        return
    rs = sorted(int(k) for k in ar)
    with ieee_style():
        fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
        curve_panel(ax[0], rs, [ar[str(r)] for r in rs],
                    [s.get("auroc_std_by_rank", {}).get(str(r), 0) for r in rs],
                    xlabel="components retained", ylabel="AUROC",
                    title="Accuracy under compression", logx=True)
        bars = {"weights: top-10": s.get("weight_mass_top10"),
                "weights: top-50": s.get("weight_mass_top50"),
                "features: top-10": s.get("data_mass_top10"),
                "features: top-50": s.get("data_mass_top50")}
        bars = {k: v for k, v in bars.items() if v is not None}
        cols = [ACCENT if k.startswith("weights") else BASE for k in bars]
        hbar_panel(ax[1], list(bars), list(bars.values()), None,
                   xlabel="share of spectral mass", title="Where the spectrum sits",
                   chance=None, sort=False, colors=cols)
        fig.tight_layout()
        _save(fig, out, "fig4_spectrum")


def fig_symptoms(d: Path, out: Path):
    f = d / "symptom_axes.csv"
    if not f.exists():
        return
    plt = _mpl(); ieee_style = _style()
    df = pd.read_csv(f).sort_values("p").head(12)
    if df.empty:
        return
    labels = [f"{r.instrument} · comp {int(r.component)}" for r in df.itertuples()]
    cols = [ACCENT if q < 0.05 else MUTED for q in df["q_fdr"]]
    with ieee_style():
        fig, ax = plt.subplots(figsize=(6.4, 0.36 * len(df) + 1.5))
        y = np.arange(len(df))
        ax.barh(y, df["rho"], color=cols, edgecolor="black", linewidth=0.7, zorder=3)
        for yi, (rho, q) in enumerate(zip(df["rho"], df["q_fdr"])):
            ax.text(rho + (0.01 if rho >= 0 else -0.01), yi,
                    f"{rho:+.2f}" + ("*" if q < 0.05 else ""),
                    va="center", ha="left" if rho >= 0 else "right", fontsize=7)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
        ax.axvline(0, color="black", lw=0.9)
        _finish(ax, xlabel="partial Spearman $\\rho$ (within autism)",
                title="Component scores against symptom scores", chance=None)
        fig.tight_layout()
        _save(fig, out, "fig5_symptom_axes")


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    _mpl().close(fig)
    print(f"  {name}.png")


# --------------------------------------------------------------------------
# navigable report
# --------------------------------------------------------------------------

SECTION_ORDER = ["representation", "classifier", "rank", "atlas", "fusion",
                 "harmonisation", "protocol"]


def write_report(study: Path, d: Path, figdir: Path):
    lines = ["# Study results", "",
             "All numbers come from the CSV files in `tables/`; every figure is drawn "
             "from those same files and can be regenerated with `viz_study.py`.", ""]

    figs = sorted(figdir.glob("fig*.png"))
    if figs:
        lines += ["## Figures", ""]
        for f in figs:
            lines += [f"### {f.stem.replace('_', ' ')}", "",
                      f"![{f.stem}](figures/{f.name})", ""]

    lines += ["## Tables", ""]
    for key in SECTION_ORDER:
        f = d / f"{key}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).sort_values("auroc", ascending=False)
        lines += [f"### {TITLES.get(key, key)}", "",
                  "| condition | AUROC | accuracy |", "|---|---|---|"]
        for _, r in df.iterrows():
            sd = r.get("auroc_std", float("nan"))
            sd_s = f" ± {sd:.3f}" if pd.notna(sd) else ""
            lines.append(f"| {r['condition']} | {r['auroc']:.3f}{sd_s} | {r['accuracy']:.3f} |")
        lines.append("")

    f = d / "leakage_audit.csv"
    if f.exists():
        df = pd.read_csv(f).sort_values("auroc")
        lines += ["### Evaluation protocol", "",
                  "| variant | AUROC | change vs. in-fold protocol |", "|---|---|---|"]
        for _, r in df.iterrows():
            inf = r.get("auroc_inflation", float("nan"))
            s = "—" if r["variant"] == "honest" or pd.isna(inf) else f"{inf:+.3f}"
            lines.append(f"| {r['variant']} | {r['auroc']:.3f} | {s} |")
        lines.append("")

    f = d / "symptom_axes.csv"
    if f.exists():
        df = pd.read_csv(f).sort_values("p").head(10)
        lines += ["### Component scores against symptom scores", "",
                  "| instrument | component | n | rho | q (FDR) |", "|---|---|---|---|---|"]
        for _, r in df.iterrows():
            lines.append(f"| {r['instrument']} | {int(r['component'])} | {int(r['n'])} "
                         f"| {r['rho']:+.3f} | {r['q_fdr']:.3f} |")
        lines.append("")

    lines += ["## Directory", "", "```",
              "figures/   publication figures (png + pdf)",
              "tables/    one CSV per comparison, plus ablations_all.csv",
              "raw/       full JSON records including per-fold values",
              "model/     the fitted headline model and its own figures",
              "```", ""]
    (study / "RESULTS.md").write_text("\n".join(lines))
    print("report ->", study / "RESULTS.md")


def organise(study: Path) -> Path:
    """Move loose artefacts into tables/ , raw/ and figures/ ; return the tables dir."""
    src = study / "ablations"
    src = src if src.exists() else study
    tables, raw, figs = study / "tables", study / "raw", study / "figures"
    for p in (tables, raw, figs):
        p.mkdir(parents=True, exist_ok=True)
    for f in list(src.glob("*.csv")):
        if f.parent != tables:
            f.replace(tables / f.name)
    for f in list(src.glob("*.json")):
        if f.parent != raw:
            f.replace(raw / f.name)
    for f in list(src.rglob("*.png")) + list(src.rglob("*.pdf")):
        if f.parent != figs and "model" not in f.parts:
            f.replace(figs / f.name)
    old = study / "ablations"
    if old.exists() and not any(old.rglob("*")):
        import shutil
        shutil.rmtree(old, ignore_errors=True)
    return tables


def main() -> None:
    ap = argparse.ArgumentParser(description="Conference-style figures + report")
    ap.add_argument("--dir", required=True, help="study directory")
    ap.add_argument("--out", default=None, help="figure directory (default <dir>/figures)")
    ap.add_argument("--no_organise", action="store_true")
    args = ap.parse_args()

    study = Path(args.dir).resolve()
    tables = study / "tables" if args.no_organise else organise(study)
    if not tables.exists():
        tables = study / "ablations"
    figdir = Path(args.out) if args.out else study / "figures"

    print("figures:")
    for fn in (fig_main, fig_representation_detail, fig_ablations,
               fig_spectrum, fig_symptoms):
        try:
            fn(tables, figdir)
        except Exception as exc:
            print(f"  [skip] {fn.__name__}: {exc}")
    write_report(study, tables, figdir)


if __name__ == "__main__":
    main()
