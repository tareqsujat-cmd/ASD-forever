"""
Checks added in response to the reviewer comments of 2026-10-06.  Every number that the
manuscript quotes from these checks is written to results/<study>/review/.

    1  ensemble_logit_approximation.csv   mean of the member logits against the probability-mean
                                          ensemble (what the score attribution sums, and what the
                                          adopted ensemble is)
    2  random_effects_weighting.json      unweighted, size-weighted and random-effects site means,
                                          and the coupling between a site's AUROC and its weight
    3  loso_accuracy_weighting.json       equal weight per site against equal weight per participant
    4  motion_somatomotor.json            head motion against the within-somatomotor connections
    5  cohort_exclusion_stats.json        participants of the wider cohort that the primary rule
                                          excludes: motion, diagnostic balance, rater notes
    6  windowing_matched_difference.json  site-clustered bootstrap interval for the AUROC difference
                                          between the windowed and the shrinkage-matched static model

Usage:
    .venv/bin/python review_checks.py [--study results/latest_study] [--boot 2000]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, rankdata, spearmanr

ROOT = Path(__file__).resolve().parent


def auroc(score, y):
    r = rankdata(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------ 1
def ensemble_vs_logit(study: Path):
    labels = np.load(study / "cache" / "nested" / "labels.npz")["y"].astype(int)
    rows = []
    for proto in ("pooled", "loso"):
        files = [f for f in sorted(glob.glob(str(study / "review" / "nested" / f"fold_{proto}_*.npz")))
                 if ".tmp" not in f]
        te, P = [], []
        for f in files:
            z = np.load(f)
            te.append(z["te_idx"])
            P.append(z["p_raw"])
        te, P = np.concatenate(te), np.concatenate(P, axis=1)
        y = labels[te]
        pm = P.mean(0)                                   # the adopted ensemble (probability mean)
        lm = np.log(P / (1 - P)).mean(0)                 # mean of the three member logits
        lpm = np.log(pm / (1 - pm))                      # logit of the probability-mean ensemble
        rows.append(dict(
            protocol=proto, n=len(y),
            auroc_probability_mean=auroc(pm, y), auroc_mean_logit=auroc(lm, y),
            spearman=float(spearmanr(pm, lm).statistic),
            separation_mean_logit=float(lm[y == 1].mean() - lm[y == 0].mean()),
            separation_logit_of_probability_mean=float(lpm[y == 1].mean() - lpm[y == 0].mean()),
            decisions_differing_at_0p5=float(((pm >= 0.5) != (lm >= 0)).mean())))
    pd.DataFrame(rows).to_csv(study / "review" / "ensemble_logit_approximation.csv", index=False)
    return rows


# ------------------------------------------------------------------------------ 2, 3
def site_summaries(study: Path):
    a = pd.read_csv(study / "tables" / "per_site_loso.csv")
    b = pd.read_csv(study / "tables" / "per_site_intervals.csv")
    m = a.merge(b[["site", "auroc_lo", "auroc_hi"]], on="site")
    n, au, acc = m.n.to_numpy(float), m.auroc.to_numpy(float), m.accuracy.to_numpy(float)
    se = ((m.auroc_hi - m.auroc_lo) / (2 * 1.96)).clip(lower=1e-3).to_numpy()
    w = 1 / se ** 2
    fe = float((w * au).sum() / w.sum())
    q = float((w * (au - fe) ** 2).sum())
    k = len(au)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    out = dict(
        n_sites=int(k),
        auroc_unweighted_site_mean=float(au.mean()),
        auroc_size_weighted_mean=float((au * n).sum() / n.sum()),
        auroc_random_effects_mean_wald_approximation=float((ws * au).sum() / ws.sum()),
        spearman_site_auroc_vs_standard_error=float(spearmanr(au, se).statistic),
        spearman_site_auroc_vs_weight=float(spearmanr(au, ws).statistic),
        note="the paper's DeLong-variance random-effects mean (0.803) is read from the stored interval table; "
             "the Wald approximation here reproduces it to about 0.001")
    (study / "review" / "random_effects_weighting.json").write_text(json.dumps(out, indent=1))
    acc_out = dict(accuracy_equal_weight_per_site=float(acc.mean()),
                   accuracy_equal_weight_per_participant=float((acc * n).sum() / n.sum()),
                   n_sites=int(k), n_participants=int(n.sum()))
    (study / "review" / "loso_accuracy_weighting.json").write_text(json.dumps(acc_out, indent=1))
    return out, acc_out


# ------------------------------------------------------------------------------ 4
def motion_somatomotor(study: Path):
    import exp_freedman_lane as fl                       # same inputs as the published group tests
    a = argparse.Namespace(processed_dir="./abide_processed",
                           ts_root="./abide_raw/ABIDE_pcp/cpac/filt_noglobal",
                           roi_table="atlas/roi_table.csv",
                           pheno="abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
                           match_motion=False, caliper=0.01)
    X, y, sites, C, roi, keep, system, iu = fl.build(a)
    n = len(y)
    fd, site_d = C[:, 3], C[:, 4:]
    sysk = np.array([system.loc[r + 1] for r in keep])
    smn = (sysk[iu[0]] == "sensorimotor") & (sysk[iu[1]] == "sensorimotor")
    S = np.column_stack([np.ones(n), site_d])

    def res(M):
        b, *_ = np.linalg.lstsq(S, M, rcond=None)
        return M - S @ b

    fdr, Xr = res(fd[:, None])[:, 0], res(X)
    R = np.apply_along_axis(rankdata, 0, Xr)
    R -= R.mean(0)
    rf = rankdata(fdr) - rankdata(fdr).mean()
    rho = (R * rf[:, None]).sum(0) / np.sqrt((R ** 2).sum(0) * (rf ** 2).sum())
    slope = (Xr * fdr[:, None]).sum(0) / (fdr ** 2).sum()
    dfd = float(fd[y == 1].mean() - fd[y == 0].mean())
    attr = slope * dfd                                    # shift expected in autism from motion alone
    obs = Xr[y == 1].mean(0) - Xr[y == 0].mean(0)         # site-adjusted raw group difference
    out = dict(n_participants=int(n), n_connections=int(X.shape[1]), n_somatomotor_connections=int(smn.sum()),
               fd_difference_autistic_minus_control_mm=dfd,
               spearman_fd_connection_all_mean=float(rho.mean()),
               spearman_fd_connection_all_fraction_positive=float((rho > 0).mean()),
               spearman_fd_connection_somatomotor_mean=float(rho[smn].mean()),
               spearman_fd_connection_somatomotor_fraction_positive=float((rho[smn] > 0).mean()),
               motion_attributable_shift_somatomotor=float(attr[smn].mean()),
               observed_group_difference_somatomotor=float(obs[smn].mean()),
               fraction_somatomotor_connections_with_positive_motion_shift=float((attr[smn] > 0).mean()),
               scale="Fisher-transformed correlation; site removed from both variables; no age or sex adjustment")
    (study / "review" / "motion_somatomotor.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------ 5
def cohort_exclusions(study: Path):
    import qc_cohort as q
    d = pd.read_csv(q.PHENO)
    strict, wide = set(q.cohort_ids(cohort="strict")), set(q.cohort_ids(cohort="func"))
    sid = d["SUB_ID"].astype(str)
    d["in_strict"], d["in_wide"] = sid.isin(strict), sid.isin(wide)
    fdc = "func_mean_fd"
    d[fdc] = pd.to_numeric(d[fdc], errors="coerce")
    d.loc[d[fdc] < -9000, fdc] = np.nan
    inc, exc = d[d.in_strict], d[d.in_wide & ~d.in_strict]
    a, c = int((exc.DX_GROUP == 1).sum()), int((exc.DX_GROUP == 2).sum())
    notes = [col for col in d.columns if "notes" in col]
    out = dict(n_primary=int(len(inc)), n_wider=int(d.in_wide.sum()), n_excluded_by_primary_rule=int(len(exc)),
               mean_fd_primary=float(inc[fdc].mean()), mean_fd_excluded=float(exc[fdc].mean()),
               median_fd_primary=float(inc[fdc].median()), median_fd_excluded=float(exc[fdc].median()),
               mannwhitney_p_fd=float(mannwhitneyu(inc[fdc].dropna(), exc[fdc].dropna()).pvalue),
               excluded_autistic=a, excluded_non_autistic=c, excluded_autistic_share=a / (a + c),
               primary_autistic_share=float((inc.DX_GROUP == 1).mean()),
               fisher_p_autistic_share=float(fisher_exact([[a, c], [int((inc.DX_GROUP == 1).sum()),
                                                                     int((inc.DX_GROUP == 2).sum())]])[1]),
               excluded_with_rater_note=float(exc[notes].notna().any(axis=1).mean()),
               primary_with_rater_note=float(inc[notes].notna().any(axis=1).mean()))
    (study / "review" / "cohort_exclusion_stats.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------ 6
def windowing_ci(study: Path, boot: int):
    recs = json.load(open(study / "ablations" / "raw" / "ridge_sweep.json"))
    by = {r["condition"]: r for r in recs}
    win = next(v for k, v in by.items() if k.startswith("dynamic: mean"))
    mat = by["static ridge=0.15"]
    y, pw, pm = (np.asarray(win["oof_true"]).astype(int), np.asarray(win["oof_prob"]),
                 np.asarray(mat["oof_prob"]))
    assert np.array_equal(y, np.asarray(mat["oof_true"]).astype(int))
    site = np.asarray(win["oof_site"])
    diff = auroc(pw, y) - auroc(pm, y)
    rng = np.random.default_rng(42)
    sites = np.unique(site)
    by_site = {s: np.flatnonzero(site == s) for s in sites}
    d = []
    for _ in range(boot):
        idx = np.concatenate([by_site[s] for s in rng.choice(sites, sites.size, replace=True)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        d.append(auroc(pw[idx], yy) - auroc(pm[idx], yy))
    lo, hi = np.percentile(d, [2.5, 97.5])
    out = dict(difference_windowed_minus_matched_static=float(diff), ci95_site_clustered_bootstrap=[float(lo), float(hi)],
               n_boot=len(d), seed=42)
    (study / "review" / "windowing_matched_difference.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=str(ROOT / "results" / "latest_study"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--skip-motion", action="store_true", help="skip the check that loads the time series")
    args = ap.parse_args()
    study = Path(args.study).resolve()
    for name, fn in (("ensemble_vs_logit", lambda: ensemble_vs_logit(study)),
                     ("site_summaries", lambda: site_summaries(study)),
                     ("cohort_exclusions", lambda: cohort_exclusions(study)),
                     ("windowing_ci", lambda: windowing_ci(study, args.boot))):
        print(name, json.dumps(fn(), default=float)[:600])
    if not args.skip_motion:
        print("motion_somatomotor", json.dumps(motion_somatomotor(study), default=float)[:700])


if __name__ == "__main__":
    main()
