"""
Abstention, calibration and clinical-utility analysis for the ensemble.

The headline configuration is the average of the three connectivity representations. The
abstention analysis was originally computed on a different configuration, so this module
recomputes it on the ensemble itself, letting the whole report describe one model from
the accuracy table through to the decision rule.

It extracts the ensemble's out-of-fold predictions from the voting experiment, writes
them in the layout the selective-prediction module expects, and then produces the
selective-prediction summary, the decision curve and the predicted-probability
distributions from those predictions.

Usage:
    python ensemble_report.py --study results/latest_study
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_MEMBER = "ensemble (probability mean)"
# The rank-mean variant ranks a participant against the others in their own held-out
# fold.  Under leave-one-site-out that fold is an entire site, so the value is a rank
# among patients at the same scanner: it cannot be computed for one incoming patient, and
# it is uniform by construction rather than calibrated.  The probability mean is within
# noise of it on ranking, is better calibrated, and is defined for a single participant.


def extract(voting_json: Path, member: str = DEFAULT_MEMBER):
    """Out-of-fold predictions for one ensemble variant, one entry per protocol."""
    records = json.loads(Path(voting_json).read_text())
    out = []
    for r in records:
        cond = r.get("condition", "")
        base = cond.split(" [")[0]
        if base != member:
            continue
        out.append({"protocol": r.get("protocol", "pooled"),
                    "oof_true": r["oof_true"], "oof_prob": r["oof_prob"],
                    "oof_site": r.get("oof_site", [])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Abstention analysis for the ensemble")
    ap.add_argument("--study", required=True)
    ap.add_argument("--member", default=DEFAULT_MEMBER)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--source", default="voting.json",
                    help="stored voting run to read, e.g. voting_func.json for the wider cohort")
    ap.add_argument("--tag", default="",
                    help="suffix for every output, so another cohort never overwrites these")
    args = ap.parse_args()

    study = Path(args.study).resolve()
    raw = study / "raw" if (study / "raw").is_dir() else study / "ablations"
    tables = study / "tables" if (study / "tables").is_dir() else raw
    figs = study / "figures"

    src = raw / args.source
    if not src.exists():
        raise SystemExit(f"{src} not found — run the voting experiment first")
    entries = extract(src, args.member)
    if not entries:
        raise SystemExit(f"no records for {args.member!r} in {src}")
    print(f"{args.member}: " + ", ".join(
        f"{e['protocol']} ({len(e['oof_true'])} participants)" for e in entries))

    oof_path = study / f"ensemble_oof{args.tag}.json"
    oof_path.write_text(json.dumps(entries, indent=2))

    # ---- selective prediction, calibration, risk-coverage ----
    import selective_prediction as SP
    out_dir = study / f"selective_ensemble{args.tag}"
    rows = []
    for e in entries:
        y = np.asarray(e["oof_true"]); p = np.asarray(e["oof_prob"])
        ece, xs, ys, ns = SP.expected_calibration_error(y, p)
        rc = SP.risk_coverage(y, p)
        at50 = next((r for r in rc if abs(r["coverage"] - 0.5) < 1e-9), {})
        from sklearn.metrics import roc_auc_score, accuracy_score
        rows.append({
            "tag": e["protocol"], "n": int(len(y)), "ece": float(ece),
            "auroc_full": float(roc_auc_score(y, p)),
            "acc_full": float(accuracy_score(y, (p >= 0.5).astype(int))),
            "auroc_at_50pct": float(at50.get("auroc", float("nan"))),
            "acc_at_50pct": float(at50.get("accuracy", float("nan"))),
            "risk_coverage": rc,
        })
        try:
            SP._figures(y, p, rc, xs, ys, ece, out_dir, e["protocol"])
        except Exception as exc:
            print(f"  [figure skipped] {e['protocol']}: {exc}")
    (out_dir / "selective_report.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "selective_report.json").write_text(json.dumps(rows, indent=2))

    for r in rows:
        print(f"  {r['tag']:6s} AUROC {r['auroc_full']:.3f} -> {r['auroc_at_50pct']:.3f} "
              f"@50% | accuracy {r['acc_full']:.3f} -> {r['acc_at_50pct']:.3f} | "
              f"ECE {r['ece']:.3f}")

    # ---- decision curve and probability distributions on the same predictions ----
    import viz_advanced as VA
    oof = {e["protocol"]: (e["oof_true"], e["oof_prob"]) for e in entries}
    VA.fig_decision_curve(oof, figs, name=f"decision_curve_ensemble{args.tag}")
    VA.fig_probability_distribution(oof, figs, name=f"probability_distribution_ensemble{args.tag}")

    # ---- site-conditional risk control on the unseen-site predictions ----
    import conformal_selective as CS
    loso = next((e for e in entries if e["protocol"] == "loso"), None)
    if loso and loso.get("oof_site"):
        y = np.asarray(loso["oof_true"]); p = np.asarray(loso["oof_prob"])
        s = np.asarray(loso["oof_site"])
        summary = {}
        for mode in ("marginal", "mondrian"):
            thr = CS.fit_thresholds(y, p, s, alpha=args.alpha, mode=mode)
            ev = CS.evaluate(y, p, s, thr)
            # The bound above is fitted and scored on the same participants, so it holds
            # by construction and says nothing about a future case.  The honest figure
            # splits the data: one half chooses the threshold, the other is scored with it.
            hon = CS.honest_evaluate(y, p, s, alpha=args.alpha, mode=mode)
            ev["held_out"] = hon
            summary[mode] = ev
            print(f"  conformal {mode:9s} coverage {ev['coverage']:.3f} | "
                  f"selective accuracy {ev['selective_accuracy']:.3f} | "
                  f"in-sample bound {ev['risk_bound']:.3f} (target {args.alpha})")
            if hon.get("n_repeats"):
                print(f"            {'':9s} held-out risk {hon['held_out_risk_mean']:.3f} "
                      f"(p90 {hon['held_out_risk_p90']:.3f}) | "
                      f"target met in {hon['fraction_target_met']:.0%} of splits | "
                      f"coverage {hon['held_out_coverage_mean']:.3f}")
        (tables.parent / "raw" / f"conformal_ensemble{args.tag}.json").write_text(
            json.dumps(summary, indent=2))

    print("report ->", out_dir / "selective_report.json")


if __name__ == "__main__":
    main()
