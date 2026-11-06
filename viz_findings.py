"""
Figures for the four findings that carry the paper.

Each panel is drawn from a table written by the experiment that produced it, so nothing
here recomputes a result and no number appears that is not in a stored file.

Usage:
    python viz_findings.py --study results/<study>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASD, TC, ACC, WARN = "#c0392b", "#4a7ab5", "#2f5d7c", "#a8551f"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=250, bbox_inches="tight", facecolor="white")
    print("  wrote", out / f"{name}.png")


# --------------------------------------------------------------------------- ridge
def fig_ridge(study: Path, out: Path, plt, style):
    f = study / "ablations" / "ridge_sweep.csv"
    if not f.exists():
        f = study / "tables" / "ridge_sweep.csv"
    if not f.exists():
        print("  [skip] ridge_sweep.csv not found"); return
    d = pd.read_csv(f)
    st = d[d.representation == "static"].sort_values("ridge")
    dy = d[d.representation == "windowed"]
    with style():
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        ax.errorbar(st["ridge"], st["auroc"], yerr=st["auroc_std"], marker="o", ms=6,
                    lw=2, color=ACC, capsize=3, label="whole-scan connectivity")
        if len(dy):
            r = float(dy["ridge"].iloc[0]); a = float(dy["auroc"].iloc[0])
            ax.axhline(a, color=ASD, ls="--", lw=1.8,
                       label=f"windowed average (ridge {r:.2f})")
            ax.axvline(r, color="#999999", ls=":", lw=1.2)
            ax.annotate("the windowed representation is\nregularised at this level",
                        xy=(r, a), xytext=(r + 0.045, a - 0.016), fontsize=8.5,
                        color="#444444",
                        arrowprops=dict(arrowstyle="->", color="#888888", lw=1))
        base = st[st.ridge == st.ridge.min()]
        if len(base):
            ax.axhline(float(base["auroc"].iloc[0]), color="#bbbbbb", lw=1,
                       label="whole-scan, unregularised (the usual baseline)")
        ax.set_xlabel("additional isotropic shrinkage applied to the covariance")
        ax.set_ylabel("AUROC")
        ax.set_title("The windowed advantage is a difference in regularisation,\n"
                     "not a difference in representation", fontsize=11.5)
        ax.legend(fontsize=8.5, loc="lower right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        save(fig, out, "fig_ridge_sweep"); plt.close(fig)


# ------------------------------------------------------------------------- scaling
def fig_scaling(study: Path, out: Path, plt, style):
    f = study / "tables" / "scaling_law.csv"
    if not f.exists():
        print("  [skip] scaling_law.csv not found"); return
    d = pd.read_csv(f)
    with style():
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
        dur = d[d.axis == "duration"].sort_values("value")
        if len(dur):
            axes[0].errorbar(dur["value"], dur["auroc"], yerr=dur["auroc_std"],
                             marker="o", ms=6, lw=2, color=ACC, capsize=3)
            axes[0].set_xlabel("volumes retained per participant")
            axes[0].set_ylabel("AUROC")
            axes[0].set_title(f"Minutes in the scanner\n(the same {int(dur['n'].iloc[0])} "
                              "people at every length)", fontsize=11)
        for ax_i, key, ttl in ((1, "subjects", "Number of participants"),):
            sub = d[d.axis == key].sort_values("value")
            if len(sub):
                axes[ax_i].errorbar(sub["value"], sub["auroc"], yerr=sub["auroc_std"],
                                    marker="s", ms=6, lw=2, color=ASD, capsize=3,
                                    label="all sites")
            ws = d[d.axis.str.startswith("subjects (")]
            if len(ws):
                ws = ws.sort_values("value")
                axes[ax_i].errorbar(ws["value"], ws["auroc"], yerr=ws["auroc_std"],
                                    marker="^", ms=5, lw=1.6, color="#7d5ba6",
                                    capsize=3, ls="--",
                                    label=ws["axis"].iloc[0].split("(")[1].rstrip(")"))
                axes[ax_i].legend(fontsize=8.5, loc="lower right")
            axes[ax_i].set_xlabel("participants")
            axes[ax_i].set_title(ttl + "\n(each at full recording length)", fontsize=11)
        lo = min(a.get_ylim()[0] for a in axes); hi = max(a.get_ylim()[1] for a in axes)
        for a in axes:
            a.set_ylim(lo, hi)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        fig.suptitle("Which axis still pays", fontsize=12.5, fontweight="bold", y=1.0)
        save(fig, out, "fig_scaling_law"); plt.close(fig)


# ------------------------------------------------------------------------ severity
def fig_severity(study: Path, out: Path, plt, style):
    f = study / "tables" / "severity_pooled_vs_within.csv"
    if not f.exists():
        print("  [skip] severity_pooled_vs_within.csv not found"); return
    d = pd.read_csv(f)
    d = d[d.protocol == d.protocol.iloc[0]]
    inst = list(dict.fromkeys(d["instrument"]))
    with style():
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        yy = np.arange(len(inst)); h = 0.36
        for off, scope, c, lab in ((+h/2, "pooled", ASD, "both groups pooled"),
                                   (-h/2, "within-autism", TC, "within the autism group")):
            sub = d[d.scope == scope].set_index("instrument").reindex(inst)
            ax.barh(yy + off, sub["rho"].fillna(0), height=h, color=c,
                    edgecolor="black", linewidth=0.6, label=lab)
            for i, (r, p) in enumerate(zip(sub["rho"], sub["p"])):
                if np.isfinite(r):
                    ax.text(r + (0.012 if r >= 0 else -0.012), yy[i] + off,
                            "*" if p < 0.05 else "", va="center",
                            ha="left" if r >= 0 else "right", fontsize=12, color="black")
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(yy); ax.set_yticklabels(inst, fontsize=9)
        ax.set_xlabel("rank correlation with the model's score")
        ax.set_title("A correlation with severity that only exists when controls are\n"
                     "included is a property of the groups, not of the model", fontsize=11.5)
        ax.legend(fontsize=8.5, loc="lower right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        save(fig, out, "fig_severity_pooled_vs_within"); plt.close(fig)


# ------------------------------------------------------------------------ conformal
def fig_conformal(study: Path, out: Path, plt, style):
    f = study / "raw" / "conformal_ensemble.json"
    if not f.exists():
        print("  [skip] conformal_ensemble.json not found"); return
    d_raw = json.loads(f.read_text())
    rows = []
    for mode, ev in (d_raw.items() if isinstance(d_raw, dict) else []):
        if not isinstance(ev, dict):
            continue
        h = ev.get("held_out") or {}
        if h.get("held_out_risk_mean") is None:
            continue
        rows.append({"tag": ev.get("tag", "unseen site"), "mode": mode,
                     "alpha": ev.get("alpha", 0.25),
                     "in_sample_bound": ev.get("risk_bound"),
                     "held_out_risk_mean": h["held_out_risk_mean"],
                     "held_out_risk_p90": h.get("held_out_risk_p90"),
                     "fraction_target_met": h.get("fraction_target_met")})
    if not rows:
        print("  [skip] no held-out conformal numbers in conformal_ensemble.json"); return
    d = pd.DataFrame(rows)
    with style():
        fig, ax = plt.subplots(figsize=(8.0, 4.6))
        lab = [f"{r['tag']}\n{r['mode']}" for _, r in d.iterrows()]
        x = np.arange(len(d)); w = 0.36
        ax.bar(x - w/2, d["in_sample_bound"], width=w, color="#c8c8c8",
               edgecolor="black", linewidth=0.6, label="bound fitted on the same data")
        ax.bar(x + w/2, d["held_out_risk_mean"], width=w, color=ACC,
               edgecolor="black", linewidth=0.6, label="risk on data held out from fitting")
        alpha = float(d["alpha"].iloc[0])
        ax.axhline(alpha, color=WARN, ls="--", lw=1.8, label=f"target ({alpha:g})")
        for i, r in enumerate(d.itertuples()):
            ax.text(i + w/2, r.held_out_risk_mean + 0.006,
                    f"{r.fraction_target_met:.0%}", ha="center", fontsize=8.5, color=ACC)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8.5)
        ax.set_ylabel("error among answered cases")
        ax.set_title("A bound fitted and reported on the same participants holds by\n"
                     "construction; the percentage is how often the target is really met",
                     fontsize=11.5)
        ax.legend(fontsize=8.5, loc="upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        save(fig, out, "fig_conformal_honest"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Figures for the four findings")
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    study = Path(args.study)
    out = Path(args.out) if args.out else study / "figures"
    plt = _plt()
    from visualization.style import ieee_style
    for fn in (fig_ridge, fig_scaling, fig_severity, fig_conformal):
        try:
            fn(study, out, plt, ieee_style)
        except Exception as exc:
            print(f"  [fail] {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
