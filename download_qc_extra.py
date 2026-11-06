"""
Fetch the ROI time series for participants in the wider quality-control cohort.

The strict cohort (871 participants) is already present.  Evaluating the model on the
1,035-participant definition additionally requires the participants who pass only the
third rater's functional assessment.  This script downloads exactly those files, for the
same pipeline, filtering strategy and atlases as the existing data, and skips anything
already on disk.

Files are small text matrices, so the transfer is modest.

Usage:
    python download_qc_extra.py --dry_run       # list what is missing
    python download_qc_extra.py                 # download it
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import urllib.request
from pathlib import Path

import pandas as pd

BASE = ("https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/"
        "{pipeline}/{strategy}/{deriv}/{file_id}_{deriv}.1D")


def missing_files(pheno_csv, ts_dir: Path, cohort: str, atlases, pipeline, strategy):
    from qc_cohort import cohort_ids
    ids = cohort_ids(pheno_csv, cohort)
    d = pd.read_csv(pheno_csv)
    d = d[d["FILE_ID"] != "no_filename"]
    d = d[d["SUB_ID"].astype(int).astype(str).isin(ids)]
    jobs = []
    for _, r in d.iterrows():
        for a in atlases:
            deriv = f"rois_{a}"
            dst = ts_dir / f"{r['FILE_ID']}_{deriv}.1D"
            if not dst.exists():
                jobs.append((BASE.format(pipeline=pipeline, strategy=strategy,
                                         deriv=deriv, file_id=r["FILE_ID"]), dst))
    return jobs


def fetch(job, timeout=120):
    url, dst = job
    try:
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return (dst.name, "empty")
        tmp.replace(dst)
        return (dst.name, "ok")
    except Exception as exc:
        Path(str(dst) + ".part").unlink(missing_ok=True)
        return (dst.name, f"{type(exc).__name__}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the wider QC cohort's ROI series")
    ap.add_argument("--pheno_csv",
                    default="./abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--ts_dir", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--cohort", default="func")
    ap.add_argument("--atlases", nargs="+", default=["cc200", "aal", "ho"])
    ap.add_argument("--pipeline", default="cpac")
    ap.add_argument("--strategy", default="filt_noglobal")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    ts_dir = Path(args.ts_dir); ts_dir.mkdir(parents=True, exist_ok=True)
    jobs = missing_files(args.pheno_csv, ts_dir, args.cohort, args.atlases,
                         args.pipeline, args.strategy)
    subjects = {j[1].name.split("_rois_")[0] for j in jobs}
    print(f"{len(jobs)} files missing across {len(subjects)} participants")
    if args.dry_run or not jobs:
        for _, dst in jobs[:6]:
            print("   ", dst.name)
        return

    ok = err = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (name, status) in enumerate(ex.map(fetch, jobs), 1):
            if status == "ok":
                ok += 1
            else:
                err += 1
                if err <= 10:
                    print(f"   [{status}] {name}")
            if i % 50 == 0:
                print(f"   {i}/{len(jobs)}  ok={ok} failed={err}", flush=True)
    print(f"\ndownloaded {ok}, failed {err}")

    from qc_cohort import cohort_ids, available
    have = available(ts_dir)
    for c in ("strict", "func"):
        ids = cohort_ids(args.pheno_csv, c)
        print(f"  cohort '{c}': {len(ids & have)} of {len(ids)} present")


if __name__ == "__main__":
    main()
