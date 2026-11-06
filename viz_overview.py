"""
One figure showing every model run in the study.

Each experiment occupies a panel, drawn the same way, so the whole study can be taken in
at once: what was compared, how the conditions ordered, and how large the differences
were relative to one another. Panels share an axis range wherever the quantities are
comparable; the protocol audit is drawn on its own wider range because it extends far
above everything else, and that extension is the point of it.

Usage:
    python viz_overview.py --study results/latest_study
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ACCENT = "#c0392b"      # the configuration carried forward
BASE = "#4a7ab5"        # alternatives
MUTED = "#9aa5b1"       # context
WARN = "#b23b3b"        # protocol shortcuts
GOOD = "#1a8a3a"        # the honest protocol

# panel title, source file, and the condition to highlight
PANELS = [
    ("Connectivity representation", "representation.csv", "dynamic: mean"),
    ("Combining representations", "voting.csv", "ensemble (rank mean)"),
    ("Classifier", "classifier.csv", "logistic (dense linear)"),
    ("Components retained", "rank.csv", None),
    ("Atlas", "atlas.csv", "cc200"),
    ("Fusion rule", "fusion.csv", None),
    ("Site harmonisation", "harmonisation.csv", "no harmonisation"),
    ("Partitioning scheme", "protocol.csv", None),
    ("Sample definition", "qc_cohort.csv", None),
    ("Evaluation protocol", "leakage_audit.csv", "honest"),
]


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def panel(ax, df, title, highlight=None, label_col=None, leakage=False):
    label_col = label_col or ("condition" if "condition" in df.columns else
                              "variant" if "variant" in df.columns else "cohort")
    d = df.dropna(subset=["auroc"]).sort_values("auroc")
    labels = d[label_col].astype(str).tolist()
    vals = d["auroc"].to_numpy(float)
    errs = d["auroc_std"].to_numpy(float) if "auroc_std" in d else np.zeros(len(d))

    if leakage:
        cols = [GOOD if l == "honest" else WARN for l in labels]
    else:
        cols = [ACCENT if (highlight and l == highlight) else BASE for l in labels]

    y = np.arange(len(d))
    ax.barh(y, vals, xerr=errs, color=cols, edgecolor="black", linewidth=0.55,
            capsize=1.8, error_kw={"elinewidth": 0.7}, zorder=3, height=0.68)
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.006, yi, f"{v:.3f}", va="center", fontsize=6.2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([l if len(l) <= 30 else l[:29] + "…" for l in labels], fontsize=6.4)
    lo = 0.45 if leakage else 0.55
    hi = max(vals + errs) + (0.075 if leakage else 0.045)
    ax.set_xlim(lo, hi)
    ax.axvline(0.5, ls=":", lw=0.8, color="#888888", zorder=1)
    ax.set_title(title, fontsize=8.6, pad=4)
    ax.tick_params(axis="x", labelsize=6.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", lw=0.4, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def build(study: Path, out: Path, name="fig_overview"):
    plt = _mpl()
    from visualization.style import ieee_style

    T = study / "tables" if (study / "tables").is_dir() else study / "ablations"
    found = []
    for title, fname, hi in PANELS:
        f = T / fname
        if f.exists() and f.stat().st_size:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if "auroc" in df.columns and len(df):
                found.append((title, df, hi, fname == "leakage_audit.csv"))
    if not found:
        print("no result tables found")
        return

    ncol = 3
    nrow = int(np.ceil(len(found) / ncol))
    heights = [max(1.0, 0.24 * len(df) + 0.75) for _, df, _, _ in found]
    row_h = [max(heights[r * ncol:(r + 1) * ncol] or [1.4]) for r in range(nrow)]

    with ieee_style():
        fig = plt.figure(figsize=(4.9 * ncol, sum(row_h) + 0.9))
        gs = fig.add_gridspec(nrow, ncol, height_ratios=row_h,
                              hspace=0.55, wspace=0.42)
        for i, (title, df, hi, leak) in enumerate(found):
            ax = fig.add_subplot(gs[i // ncol, i % ncol])
            panel(ax, df, title, highlight=hi, leakage=leak)
        for j in range(len(found), nrow * ncol):
            fig.add_subplot(gs[j // ncol, j % ncol]).axis("off")

        fig.suptitle("Every comparison in the study — AUROC, mean and standard deviation "
                     "across folds", fontsize=11.5, fontweight="bold", y=0.997)
        fig.text(0.5, 0.004,
                 "Red marks the configuration carried forward. In the evaluation-protocol "
                 "panel, green is the protocol used throughout and red are variants in "
                 "which preparation steps were fitted before the data were split.",
                 ha="center", fontsize=7.4, color="#444444")
        out.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)
    print(f"wrote {out / f'{name}.png'}  ({len(found)} panels)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Overview of every model run")
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="fig_overview")
    args = ap.parse_args()
    study = Path(args.study).resolve()
    build(study, Path(args.out) if args.out else study / "figures", args.name)


if __name__ == "__main__":
    main()
