"""Cohort description: participant characteristics, and the head-motion quantities that
the group-difference analyses depend on.

Writes three tables:
  participant_characteristics.csv  strict cohort (n = 871), autistic against non-autistic
  motion_connectivity.csv          association of mean framewise displacement with mean
                                   connection strength in the group-contrast subset
  motion_matching.csv              the two within-site matched samples, re-derived with
                                   the same greedy rule and calipers as the analyses

The matched samples are rebuilt here so that the pair counts and the before/after motion
tests are stored; the counts are asserted against the analyses' own outputs.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MISSING = -9999.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Cohort description tables")
    ap.add_argument("--study", required=True)
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--pheno", default="abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--roi_table", default="atlas/roi_table.csv")
    args = ap.parse_args()

    from run_benchmark import load_dataset
    study = Path(args.study); t = study / "tables"; t.mkdir(parents=True, exist_ok=True)

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    y = np.asarray(y); sites = np.asarray(sites)
    ph = pd.read_csv(args.pheno)
    ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    ph = ph.set_index("key")
    sub = ph.reindex([str(s).split("_")[-1].lstrip("0") for s in sids])

    def raw(c):          # sentinel-coded missing values become NaN; nothing is imputed
        v = pd.to_numeric(sub[c], errors="coerce")
        return v.mask(v <= MISSING + 1).to_numpy(float)

    # ---- participant characteristics, strict cohort ----
    rows = []
    for col, lab in (("AGE_AT_SCAN", "Age, years"), ("FIQ", "Full-scale IQ"),
                     ("VIQ", "Verbal IQ"), ("PIQ", "Performance IQ"),
                     ("func_mean_fd", "Mean framewise displacement, mm")):
        v = raw(col)
        a, b = v[(y == 1) & np.isfinite(v)], v[(y == 0) & np.isfinite(v)]
        d = 3 if col == "func_mean_fd" else 2          # FD at the precision of its median row
        rows.append({"variable": lab, "statistic": "mean (SD)",
                     "autistic": f"{a.mean():.{d}f} ({a.std(ddof=1):.{d}f})", "autistic_n": len(a),
                     "non_autistic": f"{b.mean():.{d}f} ({b.std(ddof=1):.{d}f})",
                     "non_autistic_n": len(b), "test": "Welch t",
                     "p": float(stats.ttest_ind(a, b, equal_var=False).pvalue)})
        if col == "func_mean_fd":
            rows.append({"variable": lab, "statistic": "median [IQR]",
                         "autistic": f"{np.median(a):.3f} [{np.percentile(a,25):.3f}, {np.percentile(a,75):.3f}]",
                         "autistic_n": len(a),
                         "non_autistic": f"{np.median(b):.3f} [{np.percentile(b,25):.3f}, {np.percentile(b,75):.3f}]",
                         "non_autistic_n": len(b), "test": "Mann-Whitney U",
                         "p": float(stats.mannwhitneyu(a, b).pvalue)})
    sex = raw("SEX")
    fa, fc = int(((y == 1) & (sex == 2)).sum()), int(((y == 0) & (sex == 2)).sum())
    na, nc = int((y == 1).sum()), int((y == 0).sum())
    rows.append({"variable": "Female", "statistic": "n (%)",
                 "autistic": f"{fa} ({fa/na*100:.1f})", "autistic_n": na,
                 "non_autistic": f"{fc} ({fc/nc*100:.1f})", "non_autistic_n": nc,
                 "test": "chi-squared (Yates)",
                 "p": float(stats.chi2_contingency([[fa, na - fa], [fc, nc - fc]])[1])})
    T = np.array([np.asarray(x).shape[0] for x in ts])
    rows.append({"variable": "Volumes per scan", "statistic": "median [range]",
                 "autistic": f"{np.median(T[y==1]):.0f} [{T[y==1].min()}, {T[y==1].max()}]",
                 "autistic_n": na,
                 "non_autistic": f"{np.median(T[y==0]):.0f} [{T[y==0].min()}, {T[y==0].max()}]",
                 "non_autistic_n": nc, "test": "Mann-Whitney U",
                 "p": float(stats.mannwhitneyu(T[y == 1], T[y == 0]).pvalue)})
    pc = pd.DataFrame(rows)
    pc.to_csv(t / "participant_characteristics.csv", index=False)
    print(pc.to_string(index=False))

    # ---- group-contrast subset: 196 retained regions, no constant signal ----
    roi = pd.read_csv(args.roi_table)
    keep = roi[~roi["excluded"]]["roi_id"].to_numpy() - 1
    ok = np.array([bool((np.asarray(x)[:, keep].std(0) > 0).all()) for x in ts])
    iu = np.triu_indices(len(keep), 1)
    strength = np.array([np.arctanh(np.clip(np.corrcoef(np.asarray(x)[:, keep].T),
                                            -0.999, 0.999))[iu].mean()
                         for x, o in zip(ts, ok) if o])
    fd_raw, st, yy = raw("func_mean_fd")[ok], sites[ok], y[ok]
    m = np.isfinite(fd_raw)
    r_all = stats.spearmanr(fd_raw[m], strength[m])
    D = pd.get_dummies(pd.Series(st[m]), drop_first=True).to_numpy(float)
    C = np.column_stack([np.ones(m.sum()), D])
    rf = fd_raw[m] - C @ np.linalg.lstsq(C, fd_raw[m], rcond=None)[0]
    rs = strength[m] - C @ np.linalg.lstsq(C, strength[m], rcond=None)[0]
    r_site = stats.spearmanr(rf, rs)
    mc = pd.DataFrame([
        {"scope": "pooled", "n": int(m.sum()), "n_connections": int(len(iu[0])),
         "spearman_rho": float(r_all.statistic), "p": float(r_all.pvalue)},
        {"scope": "within site (site removed from both)", "n": int(m.sum()),
         "n_connections": int(len(iu[0])),
         "spearman_rho": float(r_site.statistic), "p": float(r_site.pvalue)}])
    mc.to_csv(t / "motion_connectivity.csv", index=False)
    print("\n" + mc.to_string(index=False))

    # ---- the two matched samples, same rule as the analyses ----
    fd = pd.Series(fd_raw).fillna(np.nanmedian(fd_raw)).to_numpy()
    before = float(stats.mannwhitneyu(fd[yy == 1], fd[yy == 0]).pvalue)
    mrows = []
    for caliper, analysis, ref in ((0.01, "system-level", "network_contrast_motion_matched.json"),
                                   (0.02, "regional", "sensorimotor_regions_matched.json")):
        chosen = []
        for s_ in np.unique(st):
            at = np.where((st == s_) & (yy == 1))[0]
            ct = list(np.where((st == s_) & (yy == 0))[0])
            for i in sorted(at, key=lambda k: fd[k]):
                if not ct:
                    break
                j = min(ct, key=lambda k: abs(fd[k] - fd[i]))
                if abs(fd[j] - fd[i]) <= caliper:
                    chosen += [i, j]; ct.remove(j)
        sel = np.array(sorted(chosen))
        after = float(stats.mannwhitneyu(fd[sel][yy[sel] == 1], fd[sel][yy[sel] == 0]).pvalue)
        refp = study / "raw" / ref
        stored_n = json.loads(refp.read_text()).get("n_participants") if refp.exists() else None
        assert stored_n is None or stored_n == len(sel), (analysis, stored_n, len(sel))
        mrows.append({"analysis": analysis, "caliper_mm": caliper, "pairs": len(sel) // 2,
                      "participants": len(sel), "motion_p_before": before,
                      "motion_p_after": after, "matches_stored_analysis": stored_n == len(sel)})
    mm = pd.DataFrame(mrows)
    mm.to_csv(t / "motion_matching.csv", index=False)
    print("\n" + mm.to_string(index=False))


if __name__ == "__main__":
    main()
