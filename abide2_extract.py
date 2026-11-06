"""
Streaming ROI extraction for ABIDE-II external validation.

ABIDE-II preprocessed BOLD is only available as ~1 GB denoised NIfTIs (fMRIPrep +
denoise), so the full set is terabyte-scale. This script therefore processes one
subject at a time — download BOLD -> extract ROI time series -> delete BOLD — so the
disk footprint never exceeds a single file.

Two important caveats, which must be reported with any external-validation result:
  (1) Pipeline mismatch. ABIDE-II here is fMRIPrep/denoise, whereas the model was
      trained on ABIDE-I C-PAC (filt_noglobal). This is a cross-pipeline test.
  (2) Atlas/space match. Region time series are extracted with the same Craddock-200
      volume used for ABIDE-I cc200 and the nilearn AAL atlas (116). Extraction is
      valid only if the BOLD is in MNI space; `--check_space` verifies this first.
HO (111 regions) is omitted unless an order-matched HO atlas is supplied, so external
validation runs on the cc200+aal subset.

Usage:
    python abide2_extract.py --check_space --labels abide2/labels.csv
    python abide2_extract.py --labels abide2/labels.csv --site ABIDEII-KKI_1 --limit 20 \
        --out_ts abide2/ts
"""

from __future__ import annotations

import argparse
import os
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

BUCKET = "https://fcp-indi.s3.amazonaws.com"
DENOISE = "data/Projects/ABIDE2/Outputs/denoise/denoise"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _list(prefix):
    url = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000"
    root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
    return [k.find(f"{NS}Key").text for k in root.iter(f"{NS}Contents")]


def find_bold_key(subject_id):
    """First resting denoised BOLD key for a subject, or None."""
    keys = _list(f"{DENOISE}/sub-{subject_id}/")
    bolds = sorted(k for k in keys if k.endswith("_bold_denoised.nii.gz") and "task-rest" in k)
    return bolds[0] if bolds else None


def _atlases():
    """cc200 (exact ABIDE Craddock-200 volume) + AAL (116)."""
    from viz_brain import cc200_atlas_img
    from nilearn import datasets
    cc200, _ = cc200_atlas_img()
    aal = datasets.fetch_atlas_aal()
    return {"cc200": cc200, "aal": aal.maps}


def _is_mni(affine, shape):
    # MNI-space functional grids: ~2–4 mm isotropic, sensible FOV.
    zooms = np.abs(np.diag(affine)[:3])
    return bool(np.all((zooms > 1.5) & (zooms < 5.0)) and all(20 < s < 120 for s in shape[:3]))


def extract_one(subject_id, atlases, out_ts: Path, tmp: Path):
    from nilearn.maskers import NiftiLabelsMasker
    import nibabel as nib
    key = find_bold_key(subject_id)
    if key is None:
        print(f"  [skip] {subject_id}: no rest BOLD"); return False
    dst = tmp / f"{subject_id}.nii.gz"
    urllib.request.urlretrieve(f"{BUCKET}/{key}", dst)          # ~1 GB
    try:
        img = nib.load(str(dst))
        if not _is_mni(img.affine, img.shape):
            print(f"  [skip] {subject_id}: BOLD not MNI-like (shape {img.shape})"); return False
        for name, atlas in atlases.items():
            masker = NiftiLabelsMasker(atlas, resampling_target="data", standardize=False)
            ts = masker.fit_transform(img)                     # (T, P)
            np.savetxt(out_ts / f"{subject_id}_rois_{name}.1D", ts)
        print(f"  [ok] {subject_id}: extracted {list(atlases)} "
              f"({img.shape[-1]} volumes)")
        return True
    finally:
        dst.unlink(missing_ok=True)                            # free the ~1 GB immediately


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream ABIDE-II BOLD -> ROI time series")
    ap.add_argument("--labels", default="abide2/labels.csv")
    ap.add_argument("--site", default=None, help="restrict to one site_id")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_ts", default="abide2/ts")
    ap.add_argument("--tmp", default="abide2/tmp")
    ap.add_argument("--check_space", action="store_true",
                    help="download ONE BOLD and report its affine/space, then exit")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.labels, dtype={"subject_id": str})
    if args.site:
        df = df[df["site"] == args.site]
    if args.limit:
        df = df.head(args.limit)
    out_ts = Path(args.out_ts); out_ts.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp); tmp.mkdir(parents=True, exist_ok=True)

    if args.check_space:
        import nibabel as nib
        sid = df["subject_id"].iloc[0]
        key = find_bold_key(sid)
        if not key:
            raise SystemExit(f"no BOLD for {sid}")
        print(f"downloading one BOLD for {sid} (~1 GB) to inspect space ...")
        dst = tmp / f"{sid}.nii.gz"
        urllib.request.urlretrieve(f"{BUCKET}/{key}", dst)
        img = nib.load(str(dst))
        print("shape:", img.shape, "| zooms:", np.round(img.header.get_zooms(), 2))
        print("MNI-like:", _is_mni(img.affine, img.shape))
        dst.unlink(missing_ok=True)
        return

    atlases = _atlases()
    ok = 0
    for sid in df["subject_id"]:
        try:
            ok += extract_one(sid, atlases, out_ts, tmp)
        except Exception as e:
            print(f"  [err] {sid}: {e}")
    print(f"\nextracted {ok}/{len(df)} subjects -> {out_ts}")


if __name__ == "__main__":
    main()
