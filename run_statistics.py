"""
Significance tests between the conditions that carry the study's claims.

Several of the differences reported are small — a hundredth or two of AUROC — and a
difference of that size on this sample may or may not be distinguishable from noise. The
comparisons are therefore tested rather than read off the means.

Because every condition stores its out-of-fold predictions, the tests operate on those
saved predictions and can span experiments that were run in separate processes. Each
pair is compared with DeLong's test, which accounts for the two areas being computed on
the same participants, and with McNemar's test on the paired decisions at the default
threshold. Only pairs that share the same participants in the same order are compared.

Usage:
    python run_statistics.py --study results/latest_study
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# (condition A, condition B, what the comparison establishes)
# The rank-averaged ensemble with demographics exists only in its withdrawn within-fold
# form, and the low-rank pair duplicated the rank=20 comparison, so neither is listed.
PAIRS = [
    ("ensemble + phenotype (probability mean)", "ensemble (probability mean)",
     "adding demographic variables to the ensemble"),
    ("ensemble (rank mean)", "phenotype only",
     "the ensemble against demographic variables alone"),
    ("ensemble (rank mean)", "dynamic: mean",
     "combining the representations against the best single representation"),
    ("ensemble (probability mean)", "dynamic: mean",
     "combining the representations against the best single representation"),
    ("dynamic: mean", "tangent (static)",
     "windowed averaging against the whole-scan estimate"),
    ("tangent (static)", "correlation",
     "tangent embedding against correlation"),
    ("logistic (dense linear)", "random forest", "linear model against random forest"),
    ("logistic (dense linear)", "xgboost", "linear model against gradient boosting"),
    ("logistic (dense linear)", "catboost", "linear model against gradient boosting"),
    ("rank=50", "logistic (dense linear)",
     "fifty components against the dense weight matrix"),
    ("rank=20", "logistic (dense linear)",
     "twenty components against the dense weight matrix"),
    ("no harmonisation", "in-fold ComBat", "leaving site effects against removing them"),
    ("cc200", "cc200+aal+ho", "one atlas against three"),
]


def collect(raw_dir: Path) -> dict:
    """Out-of-fold predictions keyed by ``(condition, n_participants)``.

    Keying by condition alone lets one file shadow another: the files are read in
    alphabetical order, so ``voting_pheno.json`` overwrote the identically named
    conditions from ``voting_func.json`` and every comparison on the wider cohort became
    silently unreachable.  Including the sample size separates the cohorts, and the
    duplicate-detection below keeps a byte-identical copy from being reported twice.
    """
    recs = {}
    for f in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if not (isinstance(r, dict) and "oof_prob" in r and "condition" in r):
                continue
            y = np.asarray(r["oof_true"]); p = np.asarray(r["oof_prob"])
            key = (r["condition"], len(y))
            if key in recs:                      # identical copy in another file
                continue
            recs[key] = (y, p, f.stem)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description="Significance tests between conditions")
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from evaluation.stat_compare import delong_roc_test, mcnemar_test
    from sklearn.metrics import roc_auc_score

    study = Path(args.study).resolve()
    raw = study / "raw" if (study / "raw").is_dir() else study / "ablations"
    out = Path(args.out) if args.out else (study / "tables" if (study / "tables").is_dir()
                                           else raw)
    recs = collect(raw)
    print(f"{len(recs)} conditions with stored predictions")

    cohorts = sorted({n for _, n in recs})
    print(f"cohorts present: {cohorts}")

    rows = []
    for n in cohorts:
      for a, b, meaning in PAIRS:
        if (a, n) not in recs or (b, n) not in recs:
            continue
        ya, pa, sa = recs[(a, n)]
        yb, pb, sb = recs[(b, n)]
        if len(ya) != len(yb) or not np.array_equal(ya, yb):
            print(f"  [skip] {a} vs {b} [n={n}]: different participants or ordering")
            continue
        try:
            auc_a, auc_b, p_del = delong_roc_test(ya, pa, pb)
        except Exception as exc:
            print(f"  [skip] {a} vs {b} [n={n}]: {exc}")
            continue
        # The rank-averaged score is a position among training scores, not a probability,
        # so it has no 0.5 decision threshold and McNemar's test does not apply to it.
        rank_score = "rank mean" in a or "rank mean" in b
        mc = {} if rank_score else mcnemar_test(ya, (pa >= 0.5).astype(int),
                                                (pb >= 0.5).astype(int))
        rows.append({
            "comparison": f"{a} vs {b} [n={n}]", "meaning": meaning,
            "n_participants": int(n),
            "auroc_a": float(auc_a), "auroc_b": float(auc_b),
            "difference": float(auc_a - auc_b),
            "delong_p": float(p_del), "mcnemar_p": float(mc.get("p_value", float("nan"))),
            "source_a": sa, "source_b": sb,
        })

    if not rows:
        print("no comparable pairs found")
        return
    df = pd.DataFrame(rows).sort_values("delong_p")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "statistics.csv", index=False)
    (raw / "statistics_detail.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'comparison':<56}{'ΔAUROC':>9}{'DeLong p':>11}{'McNemar p':>11}")
    for _, r in df.iterrows():
        print(f"{r['comparison'][:55]:<56}{r['difference']:>+9.4f}"
              f"{r['delong_p']:>11.4f}{r['mcnemar_p']:>11.4f}")
    sig = int((df["delong_p"] < 0.05).sum())
    print(f"\n{sig} of {len(df)} comparisons separated at p < 0.05 (DeLong, uncorrected)")
    print("report ->", out / "statistics.csv")


if __name__ == "__main__":
    main()
