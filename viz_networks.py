"""
Where the difference sits, drawn for a reader who knows brains rather than matrices.

Three views of the same result. The first places every retained region on a transparent
brain, coloured by the system it belongs to, so the parcellation itself can be inspected.
The second gives the difference between autistic and non-autistic participants for each
pair of systems, which is the level at which this cohort can support a claim. The third
ranks those pairs, marking which survive correction for multiple comparisons.

Colours follow the convention of Yeo et al. (2011), so the systems are the ones a reader
of that literature already recognises.

Usage:
    python viz_networks.py --study results/<study>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# the canonical Yeo palette, plus a neutral for structures outside the cortical seven
YEO_COLORS = {
    "visual": "#781286",
    "sensorimotor": "#4682B4",
    "dorsal attention": "#00760E",
    "salience / ventral attention": "#C43AFA",
    "limbic": "#C7B84A",
    "frontoparietal control": "#E69422",
    "default mode": "#CD3E4E",
    "subcortical / cerebellar": "#8A8A8A",
}
SHORT = {
    "visual": "visual", "sensorimotor": "somatomotor", "dorsal attention": "dorsal attn",
    "salience / ventral attention": "salience", "limbic": "limbic",
    "frontoparietal control": "frontoparietal", "default mode": "default mode",
    "subcortical / cerebellar": "subcortical",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Figure of systems and their differences")
    ap.add_argument("--study", required=True)
    ap.add_argument("--roi_table", default="atlas/roi_table.csv")
    ap.add_argument("--name", default="fig_networks")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from nilearn import plotting
    from visualization.style import ieee_style

    study = Path(args.study)
    roi = pd.read_csv(args.roi_table)
    keep = roi[~roi["excluded"]].copy()
    con_f = study / "tables" / "network_contrast.csv"
    con = pd.read_csv(con_f) if con_f.exists() else None
    if con is None:
        print(f"  [note] {con_f} not present — drawing the anatomy panel only")

    order = [s for s in YEO_COLORS if (keep["system"] == s).any()]

    with ieee_style():
        fig = plt.figure(figsize=(14.5, 7.6))
        gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0], width_ratios=[1.15, 1.0],
                              hspace=0.42, wspace=0.34)

        # ---- (a) the regions, on a brain ----
        axA = fig.add_subplot(gs[0, :])
        coords = keep[["com_x", "com_y", "com_z"]].to_numpy()
        colors = [YEO_COLORS[s] for s in keep["system"]]
        sizes = 18 + 40 * (keep["n_voxels"] / keep["n_voxels"].max())
        # plot_connectome takes an explicit colour per node; plot_markers would map a
        # single value through a colormap and give every region the same colour
        plotting.plot_connectome(
            adjacency_matrix=np.zeros((len(keep), len(keep))), node_coords=coords,
            node_color=colors, node_size=sizes.to_numpy(), display_mode="lyrz",
            axes=axA, colorbar=False, annotate=False)
        axA.set_title(f"The {len(keep)} regions that carry usable signal, coloured by system",
                      fontsize=12, pad=6)
        axA.legend(handles=[Patch(facecolor=YEO_COLORS[s], edgecolor="none",
                                  label=f"{SHORT[s]} ({int((keep['system']==s).sum())})")
                            for s in order],
                   loc="lower center", ncol=4, fontsize=8.5, frameon=False,
                   bbox_to_anchor=(0.5, -0.30))

        # ---- (b) the system-by-system difference ----
        axB = fig.add_subplot(gs[1, 0])
        if con is not None:
            names = [s for s in YEO_COLORS if s in set(con.system_a) | set(con.system_b)]
            M = np.full((len(names), len(names)), np.nan)
            Q = np.ones_like(M)
            idx = {n: i for i, n in enumerate(names)}
            for _, r in con.iterrows():
                i, j = idx[r.system_a], idx[r.system_b]
                M[i, j] = M[j, i] = r.effect
                Q[i, j] = Q[j, i] = r.q_fdr
            lim = np.nanmax(np.abs(M)) or 1.0
            im = axB.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
            for i in range(len(names)):
                for j in range(len(names)):
                    if np.isfinite(M[i, j]) and Q[i, j] < 0.05:
                        axB.text(j, i, "*", ha="center", va="center", fontsize=15,
                                 color="black", fontweight="bold")
            axB.set_xticks(range(len(names)))
            axB.set_xticklabels([SHORT[n] for n in names], rotation=45, ha="right", fontsize=8)
            axB.set_yticks(range(len(names)))
            axB.set_yticklabels([SHORT[n] for n in names], fontsize=8)
            cb = fig.colorbar(im, ax=axB, fraction=0.040, pad=0.02)
            cb.set_label("autism − control", fontsize=8.5, labelpad=2)
            cb.ax.tick_params(labelsize=7.5)
            axB.set_title("Connectivity between systems, autism vs control\n"
                          "* marks pairs surviving correction", fontsize=11)
        else:
            axB.axis("off")
            axB.text(0.5, 0.5, "system contrast not yet computed", ha="center",
                     va="center", fontsize=10, color="#888888")

        # ---- (c) the ranking ----
        axC = fig.add_subplot(gs[1, 1])
        if con is not None:
            top = con.reindex(con.effect.abs().sort_values(ascending=False).index).head(10)
            top = top.iloc[::-1]
            lab = [f"{SHORT[a]} – {SHORT[b]}" for a, b in zip(top.system_a, top.system_b)]
            cols = ["#c0392b" if e > 0 else "#2f5d7c" for e in top.effect]
            axC.barh(range(len(top)), top.effect, color=cols, edgecolor="black", linewidth=0.6)
            for i, (e, q) in enumerate(zip(top.effect, top.q_fdr)):
                if q < 0.05:
                    axC.text(e + (0.004 if e > 0 else -0.004), i, "*", va="center",
                             ha="left" if e > 0 else "right", fontsize=13, fontweight="bold")
            axC.set_yticks(range(len(top)))
            axC.set_yticklabels(lab, fontsize=8.5)
            axC.axvline(0, color="black", lw=1)
            axC.set_xlabel("difference in connectivity (autism − control)", fontsize=9)
            axC.set_title("The ten largest system differences", fontsize=11)
            for sp in ("top", "right"):
                axC.spines[sp].set_visible(False)
        else:
            axC.axis("off")

        out = study / "figures"; out.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{args.name}.{ext}", dpi=250, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)
    print("wrote", out / f"{args.name}.png")


if __name__ == "__main__":
    main()
