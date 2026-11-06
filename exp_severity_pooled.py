"""
Does the classifier score track how autistic someone is, or only whether they are?

Several studies report that a classifier's score correlates with symptom severity, and
read that as evidence the model measures a graded property.  Those correlations are
computed across both groups.  Controls are not scored on the autism instruments and sit
at or near the floor, so any classifier that separates the groups at all produces a
positive correlation with severity by arithmetic alone, whatever it has learned.

The test that distinguishes the two readings is to compute the same correlation twice:
once across both groups, reproducing the published value, and once within the autistic
group, where the floor effect is absent.

Usage:
    python exp_severity_pooled.py --study results/<study>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MISSING = -9999.0
INSTRUMENTS = {
    "ADOS total": "ADOS_TOTAL",
    "ADOS calibrated severity": "ADOS_GOTHAM_SEVERITY",
    "ADI-R social": "ADI_R_SOCIAL_TOTAL_A",
    "ADI-R communication": "ADI_R_VERBAL_TOTAL_BV",
    "ADI-R repetitive behaviour": "ADI_RRB_TOTAL_C",
    "SRS total": "SRS_RAW_TOTAL",
}


def clean(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce")
    return v.mask(v <= MISSING + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Severity correlation, pooled vs within-group")
    ap.add_argument("--study", required=True)
    ap.add_argument("--pheno", default="abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    args = ap.parse_args()

    from run_benchmark import load_dataset

    study = Path(args.study)
    _, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    sites = np.asarray(sites)
    order = [str(s).split("_")[-1].lstrip("0") for s in sids]

    oof_f = study / "ensemble_oof.json"
    if not oof_f.exists():
        raise SystemExit(f"no ensemble out-of-fold predictions at {oof_f}")
    entries = json.loads(oof_f.read_text())

    ph = pd.read_csv(args.pheno)
    ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    ph = ph.set_index("key")

    rows = []
    for e in entries:
        proto = e.get("protocol", "?")
        yy = np.asarray(e["oof_true"]); pp = np.asarray(e["oof_prob"])
        if len(yy) != len(order):
            print(f"  [skip] {proto}: {len(yy)} predictions vs {len(order)} participants")
            continue
        # the out-of-fold arrays are in fold-concatenation order, not cohort order;
        # rebuild the participant order the same way the folds were built
        from run_benchmark import _outer_splits
        splits, _ = _outer_splits("loso" if proto == "loso" else "pooled",
                                  y, sites, 10, 42)
        idx = np.concatenate([te for _, te in splits])
        if not np.array_equal(y[idx], yy):
            print(f"  [skip] {proto}: could not reconstruct participant order")
            continue
        keys = [order[i] for i in idx]
        sub = ph.reindex(keys)

        for name, col in INSTRUMENTS.items():
            if col not in sub.columns:
                continue
            v = clean(sub[col]).to_numpy()
            for scope in ("pooled", "within-autism"):
                m = np.isfinite(v)
                if scope == "within-autism":
                    m &= (yy == 1)
                if m.sum() < 25:
                    continue
                rho, pv = spearmanr(pp[m], v[m])
                rows.append({"protocol": proto, "instrument": name, "scope": scope,
                             "n": int(m.sum()), "rho": float(rho), "p": float(pv)})

    if not rows:
        raise SystemExit("no correlations computed")
    df = pd.DataFrame(rows)
    # the six instruments form one family within each combination of validation protocol
    # and scope, and are adjusted together within it
    from scipy.stats import false_discovery_control
    df["q_bh"] = np.nan
    for _, g in df.groupby(["protocol", "scope"]):
        df.loc[g.index, "q_bh"] = false_discovery_control(g["p"].to_numpy())
    t = study / "tables"; t.mkdir(parents=True, exist_ok=True)
    df.to_csv(t / "severity_pooled_vs_within.csv", index=False)

    print(f"\n{'instrument':30s}{'scope':16s}{'n':>6s}{'rho':>8s}{'p':>10s}")
    print("-" * 70)
    for _, r in df[df.protocol == df.protocol.iloc[0]].iterrows():
        print(f"{r['instrument']:30s}{r['scope']:16s}{r['n']:>6d}{r['rho']:>8.3f}{r['p']:>10.4f}")
    pool = df[df.scope == "pooled"]
    with_ = df[df.scope == "within-autism"]
    print(f"\npooled       : {(pool['p'] < 0.05).sum()}/{len(pool)} significant, "
          f"median rho {pool['rho'].median():+.3f}")
    print(f"within-autism: {(with_['p'] < 0.05).sum()}/{len(with_)} significant, "
          f"median rho {with_['rho'].median():+.3f}")
    print("\n->", t / "severity_pooled_vs_within.csv")


if __name__ == "__main__":
    main()
