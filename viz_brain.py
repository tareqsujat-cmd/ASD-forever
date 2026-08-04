"""
Brain figures from real ABIDE-I CC200 data (publication E5 / method overview):

  1. glass-brain **connectome** of the most discriminative FC connections,
  2. a **Yeo 7-network** block matrix of aggregated edge importance,
  3. an example subject FC matrix (what the model sees).

Discriminative edges come from cross-fold ℓ2-logistic coefficients on tangent/Pearson
FC (same signal the classifier uses).  Node coordinates are the CC200 ROI centroids
(Craddock 2012 atlas); each ROI is assigned to a Yeo-7 network by its centroid.

Usage:
    python viz_brain.py --top_k 60
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

N_ROIS = 200
YEO7 = {1: "Visual", 2: "Somatomotor", 3: "DorsAttn", 4: "SalVentAttn",
        5: "Limbic", 6: "Frontoparietal", 7: "Default"}


# --------------------------- atlas / coordinates ---------------------------

def cc200_atlas_img():
    """Locate the extracted Craddock atlas and return the ~200-region volume."""
    from nilearn import image
    base = Path.home() / "nilearn_data" / "craddock_2012"
    for name in ["tcorr05_2level_all.nii.gz", "scorr05_2level_all.nii.gz",
                 "tcorr05_mean_all.nii.gz", "scorr05_mean_all.nii.gz"]:
        f = base / name
        if not f.exists():
            continue
        img = image.load_img(str(f))
        if img.ndim == 4:
            data = img.get_fdata()
            counts = [len(np.unique(data[..., i])) - 1 for i in range(img.shape[3])]
            idx = int(np.argmin([abs(c - N_ROIS) for c in counts]))
            return image.index_img(img, idx), counts[idx]
        return img, len(np.unique(img.get_fdata())) - 1
    raise FileNotFoundError("Craddock atlas not found in ~/nilearn_data/craddock_2012")


def roi_coords(atlas_img):
    from nilearn.plotting import find_parcellation_cut_coords
    return np.asarray(find_parcellation_cut_coords(atlas_img))     # (n_rois, 3)


def assign_yeo(coords):
    """Assign each ROI centroid to a Yeo-7 network by sampling the Yeo atlas."""
    from nilearn import datasets, image
    import nibabel as nib
    yeo = datasets.fetch_atlas_yeo_2011(n_networks=7)
    net = image.load_img(yeo["maps"])
    if net.ndim == 4:
        net = image.index_img(net, 0)
    data = net.get_fdata(); aff = net.affine
    inv = np.linalg.inv(aff)
    labels = []
    for c in coords:
        v = inv @ np.array([c[0], c[1], c[2], 1.0])
        i, j, k = np.round(v[:3]).astype(int)
        if 0 <= i < data.shape[0] and 0 <= j < data.shape[1] and 0 <= k < data.shape[2]:
            labels.append(int(data[i, j, k]))
        else:
            labels.append(0)
    return np.array(labels)


# --------------------------- edge importance ---------------------------

def edge_importance(processed_dir: Path, n_folds=10, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    meta = pd.read_csv(processed_dir / "mri" / "metadata.csv")
    X, y = [], []
    for _, r in meta.iterrows():
        f = processed_dir / "mri" / f"{r['subject_id']}.npy"
        if f.exists():
            X.append(np.nan_to_num(np.load(f).astype(np.float64)))
            y.append(int(r["label"]))
    X, y = np.asarray(X), np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    coefs = []
    for tr, _ in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.01, solver="liblinear", max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        coefs.append(clf.coef_.ravel())
    mean_coef = np.mean(coefs, axis=0)
    iu = np.triu_indices(N_ROIS, k=1)
    return mean_coef, iu, X[0]


# --------------------------- figures ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Brain figures from real ABIDE-I CC200")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--top_k", type=int, default=60)
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--out", default="results/brain")
    args = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    atlas_img, n = cc200_atlas_img()
    print(f"atlas volume with {n} regions", flush=True)
    coords = roi_coords(atlas_img)
    print(f"got {len(coords)} ROI coordinates", flush=True)
    yeo = assign_yeo(coords)

    mean_coef, iu, fc0 = edge_importance(Path(args.processed_dir), args.n_folds)
    n_rois = len(coords)
    # build importance adjacency (align edge vector to n_rois; CC200 uses 200)
    W = np.zeros((n_rois, n_rois))
    if len(coords) == N_ROIS:
        W[iu] = mean_coef
        W = W + W.T
    # top-K edges by |coef|
    absv = np.abs(mean_coef)
    thr = np.sort(absv)[-args.top_k] if len(absv) >= args.top_k else 0
    Wt = np.where(np.abs(W) >= thr, W, 0.0)

    # (1) glass-brain connectome
    try:
        fig = plt.figure(figsize=(9, 3.2))
        disp = plotting.plot_connectome(
            Wt, coords, edge_threshold="0%", node_size=18,
            edge_cmap="coolwarm", colorbar=True,
            title=f"Top {args.top_k} discriminative FC connections (ABIDE-I, CC200)",
            display_mode="lyrz", figure=fig)
        for ext in ("png", "pdf"):
            disp.savefig(out / f"connectome.{ext}", dpi=300)
        disp.close()
        print("connectome ->", out / "connectome.png", flush=True)
    except Exception as e:
        print("connectome failed:", e)

    # (2) Yeo network block matrix of aggregated |importance|
    try:
        from visualization.style import ieee_style
        nets = sorted(set(int(v) for v in yeo if v > 0))
        M = np.zeros((len(nets), len(nets)))
        idx = {nn: i for i, nn in enumerate(nets)}
        for e, (a, b) in enumerate(zip(iu[0], iu[1])):
            na, nb = yeo[a], yeo[b]
            if na > 0 and nb > 0:
                M[idx[na], idx[nb]] += abs(mean_coef[e]); M[idx[nb], idx[na]] += abs(mean_coef[e])
        with ieee_style():
            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            im = ax.imshow(M, cmap="magma")
            names = [YEO7.get(nn, str(nn)) for nn in nets]
            ax.set_xticks(range(len(nets))); ax.set_yticks(range(len(nets)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(names, fontsize=8)
            ax.set_title("Discriminative connectivity by network (Yeo-7)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for ext in ("png", "pdf"):
                fig.savefig(out / f"network_matrix.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
        print("network matrix ->", out / "network_matrix.png", flush=True)
    except Exception as e:
        print("network matrix failed:", e)

    # (3) example FC matrix
    try:
        from visualization.style import ieee_style
        Fm = np.zeros((N_ROIS, N_ROIS)); Fm[iu] = fc0[:len(iu[0])]; Fm = Fm + Fm.T
        with ieee_style():
            fig, ax = plt.subplots(figsize=(4.0, 3.6))
            im = ax.imshow(Fm, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title("Example subject FC matrix (CC200)")
            ax.set_xlabel("ROI"); ax.set_ylabel("ROI")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for ext in ("png", "pdf"):
                fig.savefig(out / f"example_fc.{ext}", dpi=300, bbox_inches="tight")
            plt.close(fig)
        print("example FC ->", out / "example_fc.png", flush=True)
    except Exception as e:
        print("example FC failed:", e)


if __name__ == "__main__":
    main()
