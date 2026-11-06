"""
A region table a clinician can read: anatomy, network membership, and data quality.

Each of the 200 regions of the CC200 parcellation is given an anatomical name, the
functional system it belongs to, and a record of whether its signal is usable. The
anatomical names come from the table the Preprocessed Connectomes Project distributes with
the parcellation, so they are the labels the atlas authors assigned rather than labels
re-derived here. System membership is assigned by overlapping each region with the seven
cortical networks of Yeo et al. (2011).

The quality column exists because some parcels sit where the signal drops out. In those
regions the recorded time series is constant for some participants, which silently produces
undefined correlations; any region with a constant series in even one participant is marked
for exclusion.

Usage:
    python build_roi_table.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

YEO_NAMES = {
    1: "visual", 2: "sensorimotor", 3: "dorsal attention", 4: "salience / ventral attention",
    5: "limbic", 6: "frontoparietal control", 7: "default mode",
}


def top_label(cell: str):
    """Highest-overlap label and its fraction from PCP's bracketed encoding."""
    if not isinstance(cell, str):
        return None, np.nan
    hits = re.findall(r'"([^"]+)":\s*([0-9.]+)', cell)
    hits = [(n, float(f)) for n, f in hits if n != "None"]
    if not hits:
        return None, np.nan
    n, f = max(hits, key=lambda t: t[1])
    return n, f


def main() -> None:
    import nibabel as nib
    from nilearn import datasets, image

    atlas = nib.load("atlas/cc200_roi_atlas.nii.gz")
    A = atlas.get_fdata().astype(int)
    labels = pd.read_csv("atlas/CC200_ROI_labels.csv")
    labels.columns = [c.strip() for c in labels.columns]
    print(f"atlas {A.shape}, {len(labels)} labelled regions")

    # Yeo-7 resampled onto the parcellation grid; nearest-neighbour keeps labels intact
    yeo = datasets.fetch_atlas_yeo_2011()
    Y = image.resample_to_img(nib.load(yeo["maps"]), atlas,
                              interpolation="nearest", force_resample=True,
                              copy_header=True).get_fdata().astype(int)
    Y = Y.reshape(Y.shape[:3])
    print(f"Yeo-7 resampled to {Y.shape}")

    rows = []
    for _, r in labels.iterrows():
        rid = int(r["ROI number"])
        mask = A == rid
        n_vox = int(mask.sum())
        y_in = Y[mask]
        cortical = int((y_in > 0).sum())
        frac = {k: float((y_in == k).sum()) / n_vox if n_vox else 0.0 for k in YEO_NAMES}
        win = max(frac, key=frac.get) if cortical else None
        aal, aal_f = top_label(r.get("AAL"))
        ho, ho_f = top_label(r.get("Harvard-Oxford"))
        com = str(r.get("center of mass", "")).strip("() ").split(";")
        rows.append({
            "roi_id": rid, "n_voxels": n_vox,
            "com_x": float(com[0]) if len(com) == 3 else np.nan,
            "com_y": float(com[1]) if len(com) == 3 else np.nan,
            "com_z": float(com[2]) if len(com) == 3 else np.nan,
            "aal": aal, "aal_frac": aal_f, "harvard_oxford": ho, "ho_frac": ho_f,
            "cortical_frac": cortical / n_vox if n_vox else 0.0,
            "system": YEO_NAMES.get(win) if win and frac[win] > 0.2 else "subcortical / cerebellar",
            "system_frac": frac[win] if win else 0.0,
            **{f"yeo{k}": frac[k] for k in YEO_NAMES},
        })
    t = pd.DataFrame(rows)

    # which regions carry no signal in at least one participant
    ts_dir = Path("abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    files = sorted(ts_dir.glob("*_rois_cc200.1D"))
    print(f"scanning {len(files)} time series for constant regions ...", flush=True)
    dead = np.zeros(200, dtype=int)
    for i, f in enumerate(files):
        X = np.loadtxt(f, skiprows=1)
        dead += (X.std(axis=0) == 0).astype(int)
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    t["n_constant_subjects"] = dead[t["roi_id"].to_numpy() - 1]
    t["constant_rate"] = t["n_constant_subjects"] / max(len(files), 1)
    # A region is dropped when its signal is missing often enough that keeping it would
    # cost a meaningful number of participants.  Regions affected in only a handful of
    # scans are kept, and those individual participants are excluded instead: dropping a
    # region removes a column from everyone, dropping a participant removes one row.
    t["excluded"] = t["constant_rate"] > 0.01

    t["display_name"] = [
        f"{(r.aal or r.harvard_oxford or 'unlabelled')} [{r.roi_id}]" for r in t.itertuples()
    ]
    Path("atlas").mkdir(exist_ok=True)
    t.to_csv("atlas/roi_table.csv", index=False)

    print(f"\nregions dropped (signal missing in >1% of scans): {int(t.excluded.sum())}")
    print(f"regions retained: {int((~t.excluded).sum())}")
    keep_ids = t.loc[~t.excluded, "roi_id"].to_numpy() - 1
    bad_subj = 0
    for f in files:
        X = np.loadtxt(f, skiprows=1)
        if (X[:, keep_ids].std(axis=0) == 0).any():
            bad_subj += 1
    print(f"participants dropped (a retained region is flat for them): {bad_subj} "
          f"of {len(files)}\n")
    keep = t[~t.excluded]
    print(keep.groupby("system").size().sort_values(ascending=False).to_string())
    print("\n-> atlas/roi_table.csv")


if __name__ == "__main__":
    main()
