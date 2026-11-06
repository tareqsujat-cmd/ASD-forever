"""
Per-site performance under leave-one-site-out, for the adopted ensemble.

Each bar is a site that contributed nothing to the model that was tested on it, so the
figure reports transfer to an unfamiliar acquisition rather than performance within the
cohort. Metrics are computed from the stored out-of-fold predictions of the condition
named by --condition, so the figure and the headline describe the same model.

Usage:
    python viz_per_site.py --study results/latest_study
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-site leave-one-site-out performance")
    ap.add_argument("--study", required=True)
    ap.add_argument("--condition", default="ensemble (probability mean) [loso]")
    ap.add_argument("--name", default="per_site_loso")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score, accuracy_score

    study = Path(args.study).resolve()
    raw = study / "raw" if (study / "raw").is_dir() else study / "ablations"

    rec = None
    for f in sorted(raw.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if isinstance(r, dict) and r.get("condition") == args.condition \
                    and "oof_site" in r:
                rec = r
                break
        if rec:
            break
    if rec is None:
        raise SystemExit(f"no stored predictions with sites for {args.condition!r}")

    y = np.asarray(rec["oof_true"])
    p = np.asarray(rec["oof_prob"])
    s = np.asarray(rec["oof_site"])

    rows = []
    for site in sorted(set(s)):
        m = s == site
        if len(set(y[m])) < 2:            # a site with one class has no defined AUROC
            continue
        rows.append({"site": site, "n": int(m.sum()),
                     "n_asd": int(y[m].sum()),
                     "auroc": float(roc_auc_score(y[m], p[m])),
                     "accuracy": float(accuracy_score(y[m], (p[m] >= 0.5).astype(int)))})
    df = pd.DataFrame(rows).sort_values("auroc").reset_index(drop=True)

    tdir = study / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tdir / f"{args.name}.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization.style import ieee_style

    with ieee_style():
        fig, ax = plt.subplots(figsize=(9.2, 4.4))
        # colour by how far the site sits from chance, so the weak sites read at a glance
        cols = ["#c0392b" if a < 0.65 else "#d08c1f" if a < 0.75 else "#4a7ab5"
                for a in df["auroc"]]
        ax.bar(range(len(df)), df["auroc"], color=cols, edgecolor="black", linewidth=0.6)
        ax.axhline(0.5, color="black", ls="--", lw=1.0, label="chance")
        pooled = float(roc_auc_score(y, p))
        ax.axhline(pooled, color="#1a8a3a", ls="-", lw=1.2,
                   label=f"all sites together ({pooled:.3f})")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([f"{r.site}\n(n={r.n})" for r in df.itertuples()],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.4, 1.0)
        ax.set_title("Each site tested by a model that never saw it", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fdir = study / "figures"; fdir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(fdir / f"{args.name}.{ext}", dpi=250, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)

    print(f"{len(df)} sites | worst {df.iloc[0]['site']} {df.iloc[0]['auroc']:.3f} | "
          f"best {df.iloc[-1]['site']} {df.iloc[-1]['auroc']:.3f}")
    print(f"below 0.65: {int((df['auroc'] < 0.65).sum())} | "
          f"below 0.60: {int((df['auroc'] < 0.60).sum())} | "
          f"at or below chance: {int((df['auroc'] <= 0.5).sum())}")
    print("->", tdir / f"{args.name}.csv")


if __name__ == "__main__":
    main()
