"""
How much of the remaining error is attributable to the diagnostic label.

ABIDE records two independent diagnostic instruments for many autistic participants: the
ADI-R algorithm, scored from a caregiver interview, and the ADOS, scored from direct
observation.  Where the two disagree about whether a participant meets criteria, at least
one of them is wrong, so their disagreement rate is an estimate of the noise in the label
any classifier is trained against.  That rate bounds the accuracy any classifier can
reach, and it lets the reported accuracy be corrected for the noise it was measured under
(Menon et al., ICML 2015).

Two cautions are recorded with the result.  The instruments are administered to autistic
participants and essentially never to controls, so the control-side rate has to be
estimated separately and less directly.  And the SRS used for that purpose is
caregiver-reported and often collected during the same diagnostic workup, so it is not
independent of the diagnosis; the control-side rate it yields is a lower bound.

Usage:
    python exp_label_ceiling.py --study results/<study>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MISSING = -9999.0

# ADI-R algorithm cut-offs for autism, and the ADOS autism cut-off on the total.
ADI_CUTOFFS = {"ADI_R_SOCIAL_TOTAL_A": 10, "ADI_R_VERBAL_TOTAL_BV": 8, "ADI_RRB_TOTAL_C": 3}
ADOS_CUTOFF = 7


def clean(series: pd.Series) -> pd.Series:
    """Numeric values with the -9999 sentinel removed.

    The sentinel is smaller than every real score, so leaving it in place silently sorts
    those participants to the bottom of any stratification.
    """
    v = pd.to_numeric(series, errors="coerce")
    return v.mask(v <= MISSING + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Label-noise ceiling from two instruments")
    ap.add_argument("--study", required=True)
    ap.add_argument("--pheno", default="abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    args = ap.parse_args()

    from run_benchmark import load_dataset

    study = Path(args.study)
    _, y, _, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    keys = {str(s).split("_")[-1].lstrip("0") for s in sids}

    ph = pd.read_csv(args.pheno)
    ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    coh = ph[ph["key"].isin(keys)].copy()
    asd = coh[coh["DX_GROUP"] == 1]
    ctl = coh[coh["DX_GROUP"] == 2]
    print(f"cohort {len(coh)}  |  autism {len(asd)}  control {len(ctl)}")

    # ---- autism arm: do the two instruments agree? ----
    adi = pd.DataFrame({c: clean(asd[c]) for c in ADI_CUTOFFS})
    ados = clean(asd["ADOS_TOTAL"])
    have = adi.notna().all(axis=1) & ados.notna()
    n_dual = int(have.sum())
    print(f"\nautistic participants with BOTH a complete ADI-R algorithm and an ADOS total: "
          f"{n_dual}")

    adi_pos = np.ones(n_dual, dtype=bool)
    for c, cut in ADI_CUTOFFS.items():
        adi_pos &= (adi.loc[have, c].to_numpy() >= cut)
    ados_pos = ados[have].to_numpy() >= ADOS_CUTOFF

    both = int((adi_pos & ados_pos).sum())
    adi_only = int((adi_pos & ~ados_pos).sum())
    ados_only = int((~adi_pos & ados_pos).sum())
    neither = int((~adi_pos & ~ados_pos).sum())
    disagree = (adi_only + ados_only) / n_dual
    print(f"  both positive {both} ({both/n_dual:.1%}) | ADI-R only {adi_only} "
          f"({adi_only/n_dual:.1%}) | ADOS only {ados_only} ({ados_only/n_dual:.1%}) | "
          f"neither {neither} ({neither/n_dual:.1%})")
    print(f"  instrument disagreement: {disagree:.1%}")

    # ---- control arm, via the one instrument scored in both groups ----
    srs_c = clean(ctl["SRS_RAW_TOTAL"])
    n_srs = int(srs_c.notna().sum())
    rho0 = float((srs_c.dropna() >= 60).mean()) if n_srs else float("nan")
    print(f"\ncontrols with an SRS total: {n_srs} | at or above the clinical cut-off: "
          f"{int((srs_c.dropna() >= 60).sum())} ({rho0:.1%})")

    # ---- ceiling and noise-corrected performance ----
    rho1 = disagree / 2.0            # a disagreement implicates one of the two instruments
    pi1 = float((y == 1).mean()); pi0 = 1.0 - pi1
    ceiling = 1.0 - (pi0 * rho0 + pi1 * rho1)
    print(f"\nclass-conditional noise rates: autism {rho1:.4f} | control {rho0:.4f}")
    print(f"certified accuracy ceiling: {ceiling:.4f}")

    rows = []
    oof = study / "ensemble_oof.json"
    if oof.exists():
        from sklearn.metrics import roc_auc_score, accuracy_score
        for e in json.loads(oof.read_text()):
            yy = np.asarray(e["oof_true"]); pp = np.asarray(e["oof_prob"])
            auc = roc_auc_score(yy, pp); acc = accuracy_score(yy, (pp >= 0.5).astype(int))
            denom = 1.0 - rho0 - rho1
            corr_auc = (auc - 0.5) / denom + 0.5 if denom > 0 else float("nan")
            corr_acc = (acc - (rho0 + rho1) / 2.0) / denom if denom > 0 else float("nan")
            rows.append({"protocol": e.get("protocol", "?"), "n": int(len(yy)),
                         "auroc": float(auc), "auroc_noise_corrected": float(corr_auc),
                         "accuracy": float(acc), "accuracy_noise_corrected": float(corr_acc),
                         "ceiling_accuracy": float(ceiling)})
            print(f"  {rows[-1]['protocol']:7s} AUROC {auc:.4f} -> {corr_auc:.4f} | "
                  f"acc {acc:.4f} -> {corr_acc:.4f}  (ceiling {ceiling:.4f})")

    # ---- agreement beyond chance, and how far the ceiling moves with its two inputs ----
    # The two-rater model is not identified here (no participant is negative on both
    # instruments), so the ceiling is reported as a range over the two quantities it depends
    # on: the autism-side noise rate (none, half the disagreement, all of it) and the SRS
    # threshold that defines the control-side rate.  The threshold is applied to the raw
    # total, although 60 is conventionally a T-score cut-off, which is one reason to vary it.
    po = (both + neither) / n_dual
    pa, pb = (both + adi_only) / n_dual, (both + ados_only) / n_dual
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"\nCohen's kappa between the two instrument criteria: {kappa:+.3f}")
    sens = []
    for r1_lab, r1 in (("none", 0.0), ("half the disagreement", disagree / 2.0),
                       ("all of the disagreement", disagree)):
        for cut in (40, 50, 55, 60, 70):
            r0 = float((srs_c.dropna() >= cut).mean()) if n_srs else float("nan")
            sens.append({"rho_autism_rule": r1_lab, "rho_autism": float(r1),
                         "srs_raw_cutoff": cut, "rho_control": r0,
                         "ceiling_accuracy": float(1.0 - (pi0 * r0 + pi1 * r1))})
    sens_df = pd.DataFrame(sens)
    lo, hi = float(sens_df.ceiling_accuracy.min()), float(sens_df.ceiling_accuracy.max())
    print(f"ceiling across the sensitivity grid: {lo:.3f} to {hi:.3f}")

    summary = {
        "kappa": float(kappa), "ceiling_range_min": lo, "ceiling_range_max": hi,
        "n_dual_instrument": n_dual, "both_positive": both, "adi_only": adi_only,
        "ados_only": ados_only, "neither": neither,
        "instrument_disagreement": float(disagree),
        "rho_autism": float(rho1), "rho_control": float(rho0),
        "n_controls_with_srs": n_srs,
        "ceiling_accuracy": float(ceiling),
        "caveat": ("SRS is caregiver-reported and often collected during the diagnostic "
                   "workup, so it is not independent of diagnosis; the control-side rate "
                   "is a lower bound and the ceiling correspondingly an upper one."),
        "corrected": rows,
    }
    t = study / "tables"; t.mkdir(parents=True, exist_ok=True)
    (study / "raw").mkdir(parents=True, exist_ok=True)
    (study / "raw" / "label_ceiling.json").write_text(json.dumps(summary, indent=2))
    sens_df.to_csv(t / "label_ceiling_sensitivity.csv", index=False)
    pd.DataFrame([{k: v for k, v in summary.items() if not isinstance(v, (list, dict))}]
                 ).to_csv(t / "label_ceiling.csv", index=False)
    if rows:
        pd.DataFrame(rows).to_csv(t / "label_ceiling_corrected.csv", index=False)
    print("\n->", t / "label_ceiling.csv")


if __name__ == "__main__":
    main()
