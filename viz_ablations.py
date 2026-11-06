"""
Figures for the ablation suite.

Reads the CSVs written by ``run_ablations.py`` and ``leakage_audit.py`` and produces one
scienceplots-styled figure per experiment, so that plots can be regenerated without
re-running any model.

Usage:
    python viz_ablations.py --dir results/latest_ablations/ablations
    python viz_ablations.py --dir results/<run>/ablations --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HIGHLIGHT = {"tangent (static)", "logistic (dense linear)", "honest",
             "dynamic: mean+dispersion+velocity", "low-rank spectral (r=20)"}


def _style():
    from visualization.style import ieee_style, COLORS
    return ieee_style, COLORS


def bar_figure(df: pd.DataFrame, title: str, out: Path, name: str,
               value="auroc", err="auroc_std", label_col="condition",
               highlight=HIGHLIGHT, sort=True, xlabel="AUROC"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ieee_style, COLORS = _style()

    d = df.dropna(subset=[value]).copy()
    if sort:
        d = d.sort_values(value)
    labels = d[label_col].astype(str).tolist()
    vals = d[value].to_numpy(float)
    errs = d[err].to_numpy(float) if err in d else np.zeros(len(d))
    colors = [COLORS.get("asd", "#c0392b") if l in highlight else "#5b8db8" for l in labels]

    with ieee_style():
        fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(labels) + 1.4))
        y = np.arange(len(labels))
        ax.barh(y, vals, xerr=errs, color=colors, edgecolor="black",
                linewidth=0.7, capsize=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=7.5)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        lo = max(0.45, float(np.nanmin(vals)) - 0.06)
        ax.set_xlim(lo, float(np.nanmax(vals)) + 0.06)
        ax.axvline(0.5, ls=":", lw=1.0, color="gray")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("figure ->", out / f"{name}.png")


def rank_figure(df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ieee_style, COLORS = _style()

    d = df.copy()
    d["rank"] = d["condition"].str.extract(r"rank=(\d+)").astype(float)
    d = d.dropna(subset=["rank"]).sort_values("rank")
    with ieee_style():
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.errorbar(d["rank"], d["auroc"], yerr=d.get("auroc_std"), fmt="-o",
                    color=COLORS.get("asd", "#c0392b"), lw=2.4, capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("Rank $r$ of the spectral classifier")
        ax.set_ylabel("AUROC (nested CV)")
        ax.set_title("Eigen-connectome rank vs. performance")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"rank_sweep.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("figure ->", out / "rank_sweep.png")


def leakage_figure(df: pd.DataFrame, out: Path):
    """Honest baseline vs each shortcut, annotated with the inflation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ieee_style, COLORS = _style()

    d = df.sort_values("auroc").copy()
    base = d.loc[d["variant"] == "honest", "auroc"]
    base = float(base.iloc[0]) if len(base) else float("nan")
    labels = d["variant"].tolist()
    vals = d["auroc"].to_numpy(float)
    colors = ["#1a8a3a" if l == "honest" else "#b23b3b" for l in labels]

    with ieee_style():
        fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(labels) + 1.4))
        y = np.arange(len(labels))
        ax.barh(y, vals, xerr=d.get("auroc_std"), color=colors, edgecolor="black",
                linewidth=0.7, capsize=3)
        for yi, v, lab in zip(y, vals, labels):
            txt = f"{v:.3f}"
            if lab != "honest" and np.isfinite(base):
                txt += f"  (+{v - base:.3f})"
            ax.text(v + 0.005, yi, txt, va="center", fontsize=7.5)
        if np.isfinite(base):
            ax.axvline(base, ls="--", lw=1.4, color="#1a8a3a",
                       label=f"leakage-free ({base:.3f})")
            ax.legend(loc="lower right", fontsize=7.5)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(max(0.45, float(np.nanmin(vals)) - 0.05), float(np.nanmax(vals)) + 0.10)
        ax.set_xlabel("AUROC (pooled 10-fold)")
        ax.set_title("Effect of common evaluation shortcuts")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"leakage_audit.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("figure ->", out / "leakage_audit.png")


TITLES = {
    "representation": "Connectivity representation",
    "classifier": "Classifier comparison (identical features)",
    "atlas": "Atlas and atlas combinations",
    "fusion": "Fusion rule",
    "harmonisation": "Site harmonisation",
    "protocol": "Evaluation protocol",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Figures from ablation CSVs")
    ap.add_argument("--dir", required=True, help="directory holding the ablation CSVs")
    ap.add_argument("--out", default=None, help="output directory (default: <dir>/figures)")
    args = ap.parse_args()

    d = Path(args.dir)
    out = Path(args.out) if args.out else d / "figures"
    out.mkdir(parents=True, exist_ok=True)

    for name, title in TITLES.items():
        f = d / f"{name}.csv"
        if f.exists():
            bar_figure(pd.read_csv(f), title, out, name)

    if (d / "rank.csv").exists():
        rank_figure(pd.read_csv(d / "rank.csv"), out)
        bar_figure(pd.read_csv(d / "rank.csv"), "Low-rank spectral classifier",
                   out, "rank_bars", sort=False)

    for cand in (d / "leakage_audit.csv", d.parent / "leakage_audit.csv"):
        if cand.exists():
            leakage_figure(pd.read_csv(cand), out)
            break

    if (d / "ablations_all.csv").exists():
        allf = pd.read_csv(d / "ablations_all.csv")
        print(f"\n{len(allf)} conditions across {allf['experiment'].nunique()} experiments")
        print(allf[["experiment", "condition", "auroc", "accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
