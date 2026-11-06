"""Every result the paper reports, in one table.

Values are read from the stored tables rather than transcribed, so this file cannot drift from
the run that produced it.  Scope is deliberately restricted to results that are *reported as
findings* -- the main claims and the supplementary comparisons that support them.  Approaches
that were evaluated and not adopted (harmonisation, within-scan dynamics, multiple atlases, the
rank constraint, demographic variables, the duration/sample-size contrast, restriction to the
most discriminating regions) belong to the limitations section and are excluded here.

    python compile_results.py --study results/latest_study --out paper_figures/results_compiled.csv
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile reported results into one table")
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", default="paper_figures/results_compiled.csv")
    args = ap.parse_args()

    study = Path(args.study)
    T, R = study / "tables", study / "raw"
    rows = []

    def add(cat, section, finding, quantity, value, lo=None, hi=None, n=None,
            protocol=None, comparator=None, stat=None, p=None, source=None):
        rows.append({"category": cat, "section": section, "finding": finding,
                     "quantity": quantity, "value": value, "ci_low": lo, "ci_high": hi,
                     "n": n, "protocol": protocol, "comparator": comparator,
                     "statistic": stat, "p_value": p, "source": source})

    def csv(name):
        p = T / name
        return pd.read_csv(p) if p.exists() else None

    def js(name, base=R):
        p = base / name
        return json.loads(p.read_text()) if p.exists() else None

    # ---- 1. performance, with intervals -------------------------------------------------
    iv = csv("intervals.csv")
    if iv is not None:
        for _, r in iv.iterrows():
            proto = "pooled 10-fold" if r.protocol == "pooled" else "leave-one-site-out"
            for q in ("auroc", "accuracy", "sensitivity", "specificity"):
                add("main", "1. Performance and transfer",
                    "Primary ensemble (probability mean), strict cohort", q,
                    round(float(r[q]), 4), round(float(r[f"{q}_lo"]), 4),
                    round(float(r[f"{q}_hi"]), 4), int(r["n"]), proto,
                    stat="DeLong" if q == "auroc" else "Wilson",
                    source="tables/intervals.csv")

    # wider cohort: participant-pooled, from the stored out-of-fold predictions, so these
    # rows use the same estimand as the strict-cohort rows above
    vfj = js("voting_func.json")
    if vfj:
        from exp_intervals import delong_var, wilson, ci
        import numpy as np
        byc = {e["condition"]: e for e in vfj}
        for cond, proto in [("ensemble (probability mean)", "pooled 10-fold"),
                            ("ensemble (probability mean) [loso]", "leave-one-site-out")]:
            e = byc.get(cond)
            if not e:
                continue
            yv = np.asarray(e["oof_true"]); pv = np.asarray(e["oof_prob"], dtype=float)
            a_, v_ = delong_var(yv, pv); lo_, hi_ = ci(a_, v_)
            k = int(((pv >= 0.5).astype(int) == yv).sum()); wl, wh = wilson(k, len(yv))
            add("main", "1. Performance and transfer",
                "Primary ensemble, wider quality-control cohort (comparability only)", "auroc",
                round(a_, 4), round(lo_, 4), round(hi_, 4), n=len(yv), protocol=proto,
                stat="DeLong", source="raw/voting_func.json")
            add("main", "1. Performance and transfer",
                "Primary ensemble, wider quality-control cohort (comparability only)", "accuracy",
                round(k / len(yv), 4), round(wl, 4), round(wh, 4), n=len(yv), protocol=proto,
                stat="Wilson", source="raw/voting_func.json")

    ivj = js("intervals.json")
    if ivj and ivj.get("meta_analysis"):
        m = ivj["meta_analysis"]
        add("main", "1. Performance and transfer",
            "Per-site random-effects combination", "AUROC", round(m["random_effects_mean"], 4),
            n=871, protocol="leave-one-site-out", stat="DerSimonian-Laird",
            source="raw/intervals.json")
        add("main", "1. Performance and transfer",
            "Prediction interval for a site not yet seen", "AUROC", None,
            round(m["prediction_interval_lo"], 4), round(m["prediction_interval_hi"], 4),
            n=m["k_sites"], protocol="leave-one-site-out",
            stat=f"I2 = {m['I2']:.0f}%, Q = {m['Q']:.1f} on {m['df']} df",
            p=round(m["Q_p"], 4), source="raw/intervals.json")

    ps = csv("per_site_intervals.csv")
    if ps is not None and len(ps):
        add("main", "1. Performance and transfer", "Per-site range", "AUROC median",
            round(float(ps.auroc.median()), 4), round(float(ps.auroc.min()), 4),
            round(float(ps.auroc.max()), 4), int(ps.n.sum()),
            protocol="leave-one-site-out",
            comparator=f"lowest {ps.loc[ps.auroc.idxmin(),'site']}, "
                       f"highest {ps.loc[ps.auroc.idxmax(),'site']}",
            source="tables/per_site_intervals.csv")

    # ---- 2. calibration and the guarantee -----------------------------------------------
    sel = js("selective_report.json", base=study / "selective_ensemble")
    if sel:
        for r in sel:
            proto = "pooled 10-fold" if r["tag"] == "pooled" else "leave-one-site-out"
            add("main", "2. Calibration and guarantee", "Calibration of reported probabilities",
                "expected calibration error", round(r["ece"], 4), n=r.get("n"),
                protocol=proto, source="selective_ensemble/selective_report.json")

    cf = js("conformal_ensemble.json")
    if cf:
        for mode, label in (("marginal", "One threshold for the whole cohort"),
                            ("mondrian", "A threshold per site")):
            if mode not in cf:
                continue
            c = cf[mode]
            add("main", "2. Calibration and guarantee", label, "cases answered",
                round(c["coverage"], 4), n=c["n_total"], protocol="leave-one-site-out",
                comparator=f"target alpha = {c['alpha']}", source="raw/conformal_ensemble.json")
            h = c.get("held_out", {})
            if h:
                add("main", "2. Calibration and guarantee", label,
                    "splits meeting target (honest split evaluation)",
                    round(h["fraction_target_met"], 4), n=h.get("n_repeats"),
                    protocol="leave-one-site-out",
                    comparator="nominal 0.90", source="raw/conformal_ensemble.json")

    # ---- 3. severity ---------------------------------------------------------------------
    sev = csv("severity_pooled_vs_within.csv")
    if sev is not None:
        for _, r in sev.iterrows():
            proto = "pooled 10-fold" if r.protocol == "pooled" else "leave-one-site-out"
            add("main", "3. Severity is a pooling artefact",
                f"{r.instrument}, computed {r.scope}", "Spearman rho", round(float(r.rho), 4),
                n=int(r.n), protocol=proto, stat="Spearman", p=float(r.p),
                source="tables/severity_pooled_vs_within.csv")

    # ---- 4. shrinkage and the windowed comparison ---------------------------------------
    rt = csv("ridge_sweep_tests.csv")
    if rt is not None:
        for _, r in rt.iterrows():
            meaning = (r.meaning.replace("unridged", "unshrunk")
                                .replace("ridge alone", "shrinkage alone"))
            add("main", "4. Windowed advantage is shrinkage", meaning,
                "AUROC difference", round(float(r.difference), 4), n=871,
                protocol="pooled 10-fold", comparator=r.comparison.replace("ridge=", "r="),
                stat="DeLong",
                p=float(r.delong_p), source="tables/ridge_sweep_tests.csv")
    rn = csv("ridge_nested.csv")
    if rn is not None:
        for proto in ("pooled", "loso"):
            d = rn[rn.protocol == proto]
            if len(d) == 2:
                nest = d[d.condition == "nested"].iloc[0]
                base = d[d.condition == "baseline"].iloc[0]
                add("main", "4. Windowed advantage is shrinkage",
                    "Shrinkage selected by nested cross-validation", "AUROC difference",
                    round(float(nest.auroc - base.auroc), 4), n=int(nest.n),
                    protocol="pooled 10-fold" if proto == "pooled" else "leave-one-site-out",
                    comparator="no second shrinkage", stat="DeLong",
                    p=float(nest.delong_p_vs_baseline), source="tables/ridge_nested.csv")

    # ---- 5. leakage granularity ----------------------------------------------------------
    lk = csv("leakage_audit.csv")
    if lk is not None:
        L = lk.set_index("variant")
        if {"honest", "fs_outside"} <= set(L.index):
            add("main", "5. Leakage depends on granularity",
                "Selecting 2,000 connections of 19,900 outside the split", "AUROC",
                round(float(L.loc["fs_outside", "auroc"]), 4), n=871,
                protocol="pooled 10-fold",
                comparator=f"honest {L.loc['honest','auroc']:.4f}",
                stat=f"inflation +{L.loc['fs_outside','auroc'] - L.loc['honest','auroc']:.4f}",
                source="tables/leakage_audit.csv")
    rs = csv("region_selection_tests.csv")
    if rs is not None:
        for k in (10, 20, 30, 50, 100):
            a = rs[rs.condition == f"top {k} (chosen from everyone)"]
            b = rs[rs.condition == f"top {k} (chosen in-fold)"]
            if len(a) and len(b):
                add("main", "5. Leakage depends on granularity",
                    f"Selecting {k} regions of 196 outside the split", "AUROC difference",
                    round(float(a.iloc[0].auroc - b.iloc[0].auroc), 4), n=871,
                    protocol="pooled 10-fold", comparator="same selection made in-fold",
                    source="tables/region_selection_tests.csv")

    # ---- 6. group differences ------------------------------------------------------------
    for name, label in (("network_contrast.csv", "all participants"),
                        ("network_contrast_motion_matched.csv", "motion-matched pairs")):
        d = csv(name)
        if d is None:
            continue
        add("main", "6. Group differences between systems",
            f"System pairs differing after correction, {label}", "pairs significant",
            int((d.q_fdr < .05).sum()), n=None, comparator=f"of {len(d)} pairs",
            source=f"tables/{name}")
        sm = d[(d.system_a == "sensorimotor") & (d.system_b == "sensorimotor")]
        if len(sm):
            add("main", "6. Group differences between systems",
                f"Somatomotor within-system difference, {label}", "standardised difference",
                round(float(sm.iloc[0].effect), 4), stat=f"z = {sm.iloc[0].z:.2f}",
                p=float(sm.iloc[0].p), source=f"tables/{name}")

    for name, label in (("sensorimotor_regions.csv", "all participants"),
                        ("sensorimotor_regions_matched.csv", "motion-matched pairs")):
        d = csv(name)
        if d is None:
            continue
        add("supporting", "S6. Inside the somatomotor system",
            f"Somatomotor regions significant, {label}", "regions significant",
            int((d.q_fdr < .05).sum()), comparator=f"of {len(d)} regions",
            stat=f"mean effect {d.effect.mean():.4f}", source=f"tables/{name}")
    smj = js("sensorimotor_regions.json")
    if smj:
        add("supporting", "S6. Inside the somatomotor system",
            "Split-half reproducibility of the regional ordering", "Spearman rho",
            round(smj["split_half_rho"], 3), n=smj.get("n_participants"),
            comparator=f"random {smj.get('n_null_blocks')} equal-sized blocks, median "
                       f"{smj.get('null_block_rho_median', float('nan')):.3f}",
            stat=f"{smj.get('null_block_rho_pct_ge', float('nan'))*100:.0f}% of random blocks reach it",
            source="raw/sensorimotor_regions.json")

    # ---- 7. label ceiling ----------------------------------------------------------------
    lc = js("label_ceiling.json")
    if lc:
        add("main", "7. The label ceiling", "Disagreement between the two instruments",
            "proportion disagreeing", round(lc["instrument_disagreement"], 4),
            n=lc["n_dual_instrument"], source="raw/label_ceiling.json")
        add("main", "7. The label ceiling", "Accuracy attainable against these labels",
            "accuracy", None, round(lc.get("ceiling_range_min", float("nan")), 3),
            round(lc.get("ceiling_range_max", float("nan")), 3), lc["n_dual_instrument"],
            stat="reported as a range; two-rater model unidentified (Hui & Walter 1980)",
            source="raw/label_ceiling.json")

    # ---- supporting comparisons ----------------------------------------------------------
    st = csv("statistics.csv")
    if st is not None:
        for key, label in [("random forest", "Linear model against random forest"),
                           ("xgboost", "Linear model against XGBoost"),
                           ("catboost", "Linear model against CatBoost")]:
            d = st[st.comparison.str.contains(key, case=False, na=False)]
            if len(d):
                r = d.iloc[0]
                add("supporting", "S3. Classifier family", label, "AUROC difference",
                    round(float(r.difference), 4), n=int(r.n_participants),
                    protocol="pooled 10-fold", comparator=f"{r.auroc_a:.4f} vs {r.auroc_b:.4f}",
                    stat="DeLong", p=float(r.delong_p), source="tables/statistics.csv")

    sp = js("shrinkage_profile.json")
    if sp:
        add("supporting", "S2. Representation",
            "Shrinkage the Ledoit-Wolf rule selects, against 0.20-0.50 chosen by the "
            "classification objective", "median intensity", round(sp["median"], 4),
            round(sp["min"], 4), round(sp["max"], 4), sp["n"],
            stat=f"Spearman(volumes, intensity) = {sp['spearman_T_vs_shrinkage']:.2f}; "
                 f"{sp['n_T_below_P']} of {sp['n']} scans shorter than the number of regions",
            source="raw/shrinkage_profile.json")

    sg = csv("subgroup_auroc.csv")
    if sg is not None:
        for _, r in sg.iterrows():
            add("supporting", "Where the model works less well",
                f"{r.variable}, {r.group}",
                "Spearman rho" if str(r.variable).startswith("score vs") else "AUROC",
                round(float(r.auroc), 4),
                round(float(r.auroc_lo), 4), round(float(r.auroc_hi), 4), int(r.n),
                protocol="pooled 10-fold" if r.protocol == "pooled" else "leave-one-site-out",
                source="tables/subgroup_auroc.csv")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"{len(df)} rows -> {out}")
    print(df.groupby(["category", "section"]).size().to_string())


if __name__ == "__main__":
    main()
