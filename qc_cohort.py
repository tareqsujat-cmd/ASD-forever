"""
Quality-control cohort definitions for ABIDE-I, and a comparison between them.

The ABIDE-I release provides quality ratings from three raters covering the anatomical
and the functional images. Two definitions of the analysed sample follow from them and
both appear in the literature:

    strict    every rating must pass -> 871 participants (403 ASD / 468 control)
    func      only the third rater's functional rating must pass -> 1,035 participants

The first is a subset of the second. Because the definition determines which published
results a number may be placed beside, the sample is selected explicitly from the
phenotypic table rather than inferred from whichever files happen to be on disk.

Running this module evaluates the same model on both cohorts, so that the effect of
admitting the lower-rated scans is measured rather than assumed.

Usage:
    python qc_cohort.py --summary                       # cohort sizes only
    python qc_cohort.py --compare --n_folds 10          # model on both cohorts
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

PHENO = "./abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"


def cohort_ids(pheno_csv: str | Path = PHENO, cohort: str = "strict") -> set[str]:
    """Subject identifiers belonging to the requested quality-control cohort."""
    d = pd.read_csv(pheno_csv)
    if cohort == "strict":
        m = ((d["qc_rater_1"] == "OK")
             & d["qc_anat_rater_2"].isin(["OK", "maybe"])
             & d["qc_func_rater_2"].isin(["OK", "maybe"])
             & (d["qc_anat_rater_3"] == "OK")
             & (d["qc_func_rater_3"] == "OK"))
    elif cohort == "func":
        m = d["qc_func_rater_3"] == "OK"
    elif cohort == "all":
        m = pd.Series(True, index=d.index)
    else:
        raise ValueError(f"unknown cohort {cohort!r}")
    return {str(int(s)) for s in d.loc[m, "SUB_ID"]}


def summarise(pheno_csv: str | Path = PHENO) -> pd.DataFrame:
    d = pd.read_csv(pheno_csv)
    rows = []
    for c in ("strict", "func", "all"):
        ids = cohort_ids(pheno_csv, c)
        sub = d[d["SUB_ID"].astype(int).astype(str).isin(ids)]
        rows.append({"cohort": c, "n": len(sub),
                     "asd": int((sub["DX_GROUP"] == 1).sum()),
                     "control": int((sub["DX_GROUP"] == 2).sum())})
    return pd.DataFrame(rows)


def available(ts_dir: Path, atlas: str = "rois_cc200") -> set[str]:
    """Subject identifiers for which a time series file is present on disk."""
    import re
    out = set()
    for f in Path(ts_dir).glob(f"*_{atlas}.1D"):
        m = re.search(r"_(\d{5,7})_" + atlas, f.name)
        if m:
            out.add(str(int(m.group(1))))
    return out


def load_from_phenotype(pheno_csv, ts_dir: Path, atlas: str = "rois_cc200"):
    """Load every participant that has a time-series file, using the phenotypic table.

    The processed metadata lists only the strict cohort, so reading the sample from it
    would silently cap the comparison at that cohort. Labels, sites and file names are
    therefore taken from the phenotypic table itself.
    """
    d = pd.read_csv(pheno_csv)
    d = d[d["FILE_ID"] != "no_filename"]
    ts, y, sites, sids = [], [], [], []
    for _, r in d.iterrows():
        f = Path(ts_dir) / f"{r['FILE_ID']}_{atlas}.1D"
        if not f.exists():
            continue
        arr = np.loadtxt(f)
        if arr.ndim != 2 or arr.shape[0] < 40:
            continue
        ts.append(arr)
        y.append(1 if int(r["DX_GROUP"]) == 1 else 0)
        sites.append(str(r["SITE_ID"]))
        sids.append(str(int(r["SUB_ID"])))
    return ts, np.array(y), np.array(sites), sids


def compare(args) -> None:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from run_benchmark import _outer_splits, _metrics
    from gpu_tangent import GPUTangentFC

    ts, y, sites, sids = load_from_phenotype(args.pheno_csv, Path(args.ts_dir))
    print(f"loaded {len(sids)} participants with time series "
          f"({int((y == 1).sum())} ASD / {int((y == 0).sum())} control)")
    gt = GPUTangentFC(device="cpu", max_iter=10)
    covs = gt.compute_covariances(ts)

    rows = []
    for cohort in args.cohorts:
        keep_ids = cohort_ids(args.pheno_csv, cohort)
        idx = np.array([i for i, s in enumerate(sids) if s in keep_ids])
        missing = len(keep_ids) - len(idx)
        print(f"\n=== cohort '{cohort}': {len(idx)} of {len(keep_ids)} defined "
              f"participants present on disk ({missing} not downloaded) ===", flush=True)
        if len(idx) < 50:
            print("  too few participants available, skipped")
            continue
        yc, sc_, cc = y[idx], np.asarray(sites)[idx], covs[idx]
        splits, _ = _outer_splits("pooled", yc, sc_, args.n_folds, args.seed)
        fm = []
        for tr, te in splits:
            gt.fit(cc[tr])
            Xtr, Xte = gt.transform(cc[tr]), gt.transform(cc[te])
            s = StandardScaler().fit(Xtr)
            Xtr, Xte = s.transform(Xtr), s.transform(Xte)
            gs = GridSearchCV(LogisticRegression(solver="liblinear", max_iter=2000),
                              {"C": [0.001, 0.01, 0.1, 1.0]}, scoring="roc_auc",
                              cv=3, n_jobs=-1).fit(Xtr, yc[tr])
            fm.append(_metrics(yc[te], gs.predict_proba(Xte)[:, 1]))
        keys = list(fm[0].keys())
        row = {"cohort": cohort, "n": int(len(idx)),
               "asd": int((yc == 1).sum()), "control": int((yc == 0).sum()),
               "not_downloaded": int(missing)}
        row.update({k: float(np.nanmean([f[k] for f in fm])) for k in keys})
        row.update({f"{k}_std": float(np.nanstd([f[k] for f in fm])) for k in ("auroc", "accuracy")})
        rows.append(row)
        print(f"  AUROC {row['auroc']:.3f} +/- {row['auroc_std']:.3f} | "
              f"accuracy {row['accuracy']:.3f} +/- {row['accuracy_std']:.3f}")

    if rows:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out / "qc_cohort.csv", index=False)
        (out / "qc_cohort.json").write_text(json.dumps(rows, indent=2))
        print("\n" + df[["cohort", "n", "auroc", "auroc_std", "accuracy"]].to_string(index=False))
        print("report ->", out / "qc_cohort.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="ABIDE-I quality-control cohorts")
    ap.add_argument("--pheno_csv", default=PHENO)
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_dir", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--cohorts", nargs="+", default=["strict", "func"])
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/ablations")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    if args.summary or not args.compare:
        print(summarise(args.pheno_csv).to_string(index=False))
        have = available(Path(args.ts_dir))
        for c in ("strict", "func"):
            ids = cohort_ids(args.pheno_csv, c)
            print(f"  cohort '{c}': {len(ids & have)} of {len(ids)} present on disk")
    if args.compare:
        compare(args)


if __name__ == "__main__":
    main()
