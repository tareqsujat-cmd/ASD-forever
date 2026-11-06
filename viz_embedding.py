"""
The data seen at each stage of the pipeline, in three dimensions.

Every stage of the pipeline is a change of representation, and each can be viewed by
projecting it onto its leading principal components. Showing the same participants at
successive stages makes visible what the transformations do and, equally, what they do
not do: the group difference is small next to the variation that dominates these data,
and it does not lie along the directions of greatest variance.

Panels, in order:

  A  correlation connectivity, coloured by diagnosis
  B  the tangent embedding, coloured by diagnosis
  C  the same embedding coloured by acquisition site, which is what the leading
     directions actually track
  D  the out-of-fold decision score against the two leading directions, which is where
     the group difference does live
  E  the windowed trajectories of a few participants, each point one window of one scan
  F  variance explained, showing how little of it any single direction carries

Panel D uses out-of-fold scores, so no participant's own data contributed to their
position along that axis. The remaining panels are unsupervised and use every
participant, which is appropriate for describing the data but means they carry no claim
about performance.

Usage:
    python viz_embedding.py --out results/latest_study/figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")

import warnings
warnings.filterwarnings("ignore")
import numpy as np

ASD, TC = "#c0392b", "#4a7ab5"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def scatter3(ax, X, labels, colors, names, title, alpha=0.55, s=9):
    for lab, c, nm in zip(sorted(set(labels)), colors, names):
        m = np.asarray(labels) == lab
        ax.scatter(X[m, 0], X[m, 1], X[m, 2], s=s, c=c, alpha=alpha,
                   edgecolors="none", label=nm, depthshade=False)
    ax.set_xlabel("dim 1", fontsize=7, labelpad=-8)
    ax.set_ylabel("dim 2", fontsize=7, labelpad=-8)
    ax.set_zlabel("dim 3", fontsize=7, labelpad=-8)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_ticklabels([])
    ax.set_title(title, fontsize=9.5, pad=2)
    ax.grid(alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embedding spaces through the pipeline")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_dir", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--oof", default=None, help="ensemble out-of-fold predictions")
    ap.add_argument("--n_traj", type=int, default=6, help="participants drawn as trajectories")
    ap.add_argument("--out", default="results/latest_study/figures")
    ap.add_argument("--name", default="fig_embedding_flow")
    ap.add_argument("--cache", default=None,
                    help="npz of the computed embeddings (default: alongside --out)")
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild the embeddings even if the cache is present")
    args = ap.parse_args()

    # The embeddings take about an hour to rebuild, and the layout of the figure is
    # tuned separately from them, so they are cached and reused.
    cache = Path(args.cache) if args.cache else Path(args.out) / "embedding_cache.npz"
    if cache.exists() and not args.recompute:
        print(f"reusing {cache}", flush=True)
        z = np.load(cache, allow_pickle=True)
        draw(z, args)
        return

    from sklearn.decomposition import PCA
    from nilearn.connectome import ConnectivityMeasure
    from run_benchmark import load_dataset
    from gpu_tangent import GPUTangentFC
    from dynamic_fc import DynamicTangentFC

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_dir))
    print(f"{len(y)} participants", flush=True)

    print("correlation connectivity ...", flush=True)
    cm = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
    Xcorr = cm.fit_transform(ts)

    print("tangent embedding ...", flush=True)
    gt = GPUTangentFC(device="cpu", max_iter=10)
    covs = gt.compute_covariances(ts)
    gt.fit(covs)
    Xtan = gt.transform(covs)

    print("window trajectories ...", flush=True)
    dfc = DynamicTangentFC(window=50, n_windows=6)
    wcovs = dfc.compute_window_covariances(ts)
    dfc.fit(wcovs, list(range(len(y))))

    p_corr = PCA(n_components=3, random_state=0).fit(Xcorr)
    p_tan = PCA(n_components=8, random_state=0).fit(Xtan)
    Ecorr, Etan = p_corr.transform(Xcorr), p_tan.transform(Xtan)

    # out-of-fold decision score for the supervised view
    oof_path = Path(args.oof) if args.oof else None
    if oof_path is None:
        c = sorted(Path("results").glob("*_full_study/ensemble_oof.json"))
        oof_path = c[-1] if c else None
    score = None
    if oof_path and oof_path.exists():
        rec = [r for r in json.loads(oof_path.read_text()) if r["protocol"] == "pooled"]
        if rec and len(rec[0]["oof_prob"]) == len(y):
            score = np.asarray(rec[0]["oof_prob"])
    if score is None:
        print("  [note] no out-of-fold scores found; panel D omitted")

    # cross-validated AUROC per representation, for annotation
    auc = {}
    vt = sorted(Path("results").glob("*_full_study/tables/voting.csv"))
    if vt:
        import pandas as pd
        vd = pd.read_csv(vt[-1])
        for _, r in vd.iterrows():
            auc[str(r["condition"])] = float(r["auroc"])

    # how much of each leading direction is explained by site (one-way eta-squared)
    def eta_sq(v, groups):
        g = np.asarray(groups)
        gm = v.mean()
        between = sum((v[g == s].mean() - gm) ** 2 * (g == s).sum() for s in set(g))
        return float(between / ((v - gm) ** 2).sum())

    site_r2 = [eta_sq(Etan[:, k], sites) for k in range(3)]
    print(f"  site explains {site_r2[0]*100:.0f}% / {site_r2[1]*100:.0f}% / "
          f"{site_r2[2]*100:.0f}% of leading directions 1-3", flush=True)

    # within-scan movement relative to between-participant spread, in the same space
    import torch
    iu = np.triu_indices(wcovs[0].shape[-1], k=1)
    rng0 = np.random.RandomState(1)
    sub = rng0.choice(len(y), 40, replace=False)
    within = []
    for i in sub:
        tg = dfc._tangent_safe(torch.as_tensor(wcovs[i], dtype=torch.float64))
        P = p_tan.transform(tg[:, iu[0], iu[1]].cpu().numpy())[:, :3]
        within.append(np.linalg.norm(P - P.mean(0), axis=1).mean())
    ratio = float(np.mean(within) /
                  np.linalg.norm(Etan[:, :3] - Etan[:, :3].mean(0), axis=1).mean())
    print(f"  within-scan movement is {ratio:.2f}x the between-participant spread",
          flush=True)

    plt = _mpl()
    from visualization.style import ieee_style
    with ieee_style():
        fig = plt.figure(figsize=(15.5, 9.6))
        gs = fig.add_gridspec(2, 3, hspace=0.16, wspace=0.10)

        def note(ax, txt):
            ax.text2D(0.5, -0.02, txt, transform=ax.transAxes, ha="center",
                      fontsize=8, style="italic", color="#333333")

        ax = fig.add_subplot(gs[0, 0], projection="3d")
        scatter3(ax, Ecorr, y, [TC, ASD], ["control", "autism"],
                 "A  correlation connectivity")
        ax.legend(fontsize=7, loc="upper left", markerscale=1.6, framealpha=0.85)
        if "correlation" in auc:
            note(ax, f"cross-validated AUROC {auc['correlation']:.3f}")

        ax = fig.add_subplot(gs[0, 1], projection="3d")
        scatter3(ax, Etan, y, [TC, ASD], ["control", "autism"],
                 "B  tangent embedding")
        if "tangent (static)" in auc:
            note(ax, f"cross-validated AUROC {auc['tangent (static)']:.3f} — the "
                     f"embedding changes the geometry, not the visible overlap")

        ax = fig.add_subplot(gs[0, 2], projection="3d")
        top = [s for s, _ in sorted({s: (np.asarray(sites) == s).sum()
                                     for s in set(sites)}.items(),
                                    key=lambda kv: -kv[1])[:6]]
        pal = ["#c0392b", "#4a7ab5", "#1a8a3a", "#d08c1f", "#7d5ba6", "#3aa8a8"]
        for s_, c_ in zip(top, pal):
            m = np.asarray(sites) == s_
            ax.scatter(Etan[m, 0], Etan[m, 1], Etan[m, 2], s=9, c=c_, alpha=0.7,
                       edgecolors="none", label=s_, depthshade=False)
        other = ~np.isin(np.asarray(sites), top)
        ax.scatter(Etan[other, 0], Etan[other, 1], Etan[other, 2], s=6, c="#c8c8c8",
                   alpha=0.35, edgecolors="none", label="other", depthshade=False)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_ticklabels([])
        ax.set_title("C  the same embedding, coloured by site", fontsize=9.5, pad=2)
        ax.legend(fontsize=6, loc="upper left", markerscale=1.4, framealpha=0.85, ncol=2)
        note(ax, f"site explains {site_r2[0]*100:.0f}% of direction 1 — far more than "
                 f"diagnosis does")

        ax = fig.add_subplot(gs[1, 0], projection="3d")
        if score is not None:
            E = np.column_stack([Etan[:, 0], Etan[:, 1], score])
            scatter3(ax, E, y, [TC, ASD], ["control", "autism"],
                     "D  decision score against the leading directions")
            ax.set_zlabel("out-of-fold score", fontsize=7, labelpad=-8)
            ax.view_init(elev=6, azim=-72)      # near side-on: the score axis reads
            note(ax, f"AUROC {auc.get('ensemble (rank mean)', float('nan')):.3f} — the "
                     f"group difference lies along the score, not the leading directions")
        else:
            ax.axis("off")

        ax = fig.add_subplot(gs[1, 1], projection="3d")
        rng = np.random.RandomState(0)
        pick = np.concatenate([rng.choice(np.where(y == 1)[0], args.n_traj // 2, False),
                               rng.choice(np.where(y == 0)[0], args.n_traj // 2, False)])
        # the whole cohort, faintly, so the excursions have a scale to be read against
        ax.scatter(Etan[:, 0], Etan[:, 1], Etan[:, 2], s=4, c="#d5d5d5", alpha=0.30,
                   edgecolors="none", depthshade=False)
        for i in pick:
            # one tangent vector per window, exactly as DynamicTangentFC.transform does
            tang = dfc._tangent_safe(torch.as_tensor(wcovs[i], dtype=torch.float64))
            W = tang[:, iu[0], iu[1]].cpu().numpy()                  # (K, d)
            P = p_tan.transform(W)
            c = ASD if y[i] == 1 else TC
            ax.plot(P[:, 0], P[:, 1], P[:, 2], "-o", color=c, lw=1.3, ms=3.2, alpha=0.85)
            ax.scatter(*P[0, :3], s=26, c=c, marker="s", edgecolors="black", linewidths=0.5)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_ticklabels([])
        ax.set_title("E  within-scan trajectories (square = first window)", fontsize=9.5, pad=2)
        note(ax, f"each participant moves {ratio:.2f}x the spread between participants — "
                 f"the windows of one scan stay together")

        ax = fig.add_subplot(gs[1, 2])
        v = p_tan.explained_variance_ratio_
        ax.bar(np.arange(1, len(v) + 1), v * 100, color="#5b8db8", edgecolor="black",
               linewidth=0.6)
        ax.set_xlabel("direction"); ax.set_ylabel("variance explained (%)")
        ax.set_title(f"F  leading directions carry little of the variance\n"
                     f"(first eight together: {v.sum() * 100:.1f}%)", fontsize=9.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        fig.suptitle("The same participants at successive stages of the pipeline",
                     fontsize=12.5, fontweight="bold", y=0.965)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{args.name}.{ext}", dpi=250, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)
    print("wrote", out / f"{args.name}.png")


if __name__ == "__main__":
    main()
