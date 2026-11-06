"""
Build a matched-encoding labels/phenotype table for ABIDE-II from the per-site
participants.tsv files on the fcp-indi S3 bucket.

Output columns match what external_validation.py expects, with encodings aligned to
the ABIDE-I processed data used to train the model:
    subject_id, label, site, age, sex, fiq, viq, piq, handedness
where label = 1 for ASD (dx_group==1), 0 for control (dx_group==2); sex is remapped
from the ABIDE convention (1=male, 2=female) to the ABIDE-I encoding (0=male, 1=female).

Usage:
    python abide2_labels.py --out abide2/labels.csv
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

BUCKET = "https://fcp-indi.s3.amazonaws.com"
PREFIX = "data/Projects/ABIDE2/RawData/"


def list_sites():
    import xml.etree.ElementTree as ET
    url = f"{BUCKET}/?list-type=2&prefix={PREFIX}&delimiter=/"
    xml = urllib.request.urlopen(url, timeout=60).read()
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    root = ET.fromstring(xml)
    sites = []
    for p in root.iter(f"{ns}CommonPrefixes"):
        pref = p.find(f"{ns}Prefix").text
        name = pref.rstrip("/").split("/")[-1]
        if name.startswith("ABIDEII-"):
            sites.append(name)
    return sites


def fetch_participants(site):
    url = f"{BUCKET}/{PREFIX}{site}/participants.tsv"
    try:
        raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
    except Exception as e:
        print(f"  [skip] {site}: {e}")
        return None
    df = pd.read_csv(io.StringIO(raw), sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]      # 'age_at_scan ' has a trailing space
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ABIDE-II labels/phenotype CSV")
    ap.add_argument("--out", default="abide2/labels.csv")
    args = ap.parse_args()

    sites = list_sites()
    print(f"{len(sites)} ABIDE-II sites")
    rows = []
    for s in sites:
        df = fetch_participants(s)
        if df is None:
            continue
        for _, r in df.iterrows():
            dx = pd.to_numeric(r.get("dx_group"), errors="coerce")
            if dx not in (1, 2):
                continue                                       # keep only ASD / control
            label = 1 if dx == 1 else 0
            sex_raw = pd.to_numeric(r.get("sex"), errors="coerce")
            sex = 0 if sex_raw == 1 else (1 if sex_raw == 2 else np.nan)   # 1=M->0, 2=F->1
            rows.append({
                "subject_id": str(int(r["participant_id"])),
                "label": label,
                "site": r.get("site_id", s),
                "age": pd.to_numeric(r.get("age_at_scan"), errors="coerce"),
                "sex": sex,
                "fiq": pd.to_numeric(r.get("fiq"), errors="coerce"),
                "viq": pd.to_numeric(r.get("viq"), errors="coerce"),
                "piq": pd.to_numeric(r.get("piq"), errors="coerce"),
                "handedness": pd.to_numeric(r.get("handedness_scores"), errors="coerce"),
            })
        print(f"  {s}: {sum(1 for x in rows if x['site']==df['site_id'].iloc[0] if 'site_id' in df) or ''}"
              f" (running total {len(rows)})")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {len(out)} participants -> {args.out}")
    print(f"  ASD={int((out.label==1).sum())}  control={int((out.label==0).sum())}  "
          f"sites={out.site.nunique()}")
    print(f"  age mean={out.age.mean():.1f} ({out.age.min():.0f}-{out.age.max():.0f}) | "
          f"male={int((out.sex==0).sum())} female={int((out.sex==1).sum())}")


if __name__ == "__main__":
    main()
