"""
Feasibility analyses for the trustworthiness criteria that the study does not yet establish.
Everything is computed from the stored out-of-fold predictions (results/<study>/review/nested/),
so no model is refitted and nothing is fitted on held-out data.

    A  new-site calibration     leave-one-site-out: can a small amount of local information
                                (the site's autistic proportion, or a few labelled participants)
                                reduce the site-level calibration heterogeneity (I^2, tau)?
    B  operating point          thresholds fixed from the training partition only (specificity 0.80,
                                sensitivity 0.80, Youden), applied to the held-out fold; net benefit
    C  stratified validation    AUROC and calibration-in-the-large by age group, head-motion tercile
                                and site size, with site-clustered bootstrap intervals

Outputs: results/<study>/review/trust_checks.json and trust_checks_*.csv

Usage:
    .venv/bin/python exp_trust_checks.py [--study results/latest_study] [--boot 2000] [--reps 400]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
PHENO = ROOT / "abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"


# ------------------------------------------------------------------------------------ helpers
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def fit_delta(z, y, prior_var=None, iters=60):
    """Intercept delta of y ~ Bernoulli(expit(z + delta)); optional N(0, prior_var) prior on delta.
    Returns (delta, standard error). Calibration-in-the-large is this delta (negative = over-prediction)."""
    d = 0.0
    for _ in range(iters):
        p = expit(z + d)
        g, H = (y - p).sum(), (p * (1 - p)).sum()
        if prior_var:
            g -= d / prior_var
            H += 1 / prior_var
        step = g / max(H, 1e-9)
        d = float(np.clip(d + step, -6, 6))
        if abs(step) < 1e-9:
            break
    p = expit(z + d)
    info = (p * (1 - p)).sum() + (1 / prior_var if prior_var else 0.0)
    return d, 1 / np.sqrt(max(info, 1e-9))


def auroc(score, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def dl(est, se):
    """DerSimonian-Laird heterogeneity: I^2 and tau of the site estimates."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    w = 1 / se ** 2
    fe = (w * est).sum() / w.sum()
    q = (w * (est - fe) ** 2).sum()
    k = len(est)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    return max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0, float(np.sqrt(tau2))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(c - h), float(c + h))


# ------------------------------------------------------------------------------------ data
def load(study: Path):
    lab = np.load(study / "cache" / "nested" / "labels.npz", allow_pickle=True)
    y_all, site_all, sids = lab["y"].astype(int), np.asarray(lab["sites"]).astype(str), lab["sids"]
    out = {}
    for proto in ("pooled", "loso"):
        files = [f for f in sorted(glob.glob(str(study / "review" / "nested" / f"fold_{proto}_*.npz")))
                 if ".tmp" not in f]
        rows = []
        for k, f in enumerate(files):
            z = np.load(f, allow_pickle=True)
            tr, te = z["tr_idx"], z["te_idx"]
            zin = logit(z["inner_p"].mean(0))
            delta, _ = fit_delta(zin, y_all[tr])
            rows.append(dict(fold=k, tr=tr, te=te, pm=z["p_raw"].mean(0), pin=zin, yin=y_all[tr],
                             delta=delta, prev_tr=float(y_all[tr].mean())))
        out[proto] = rows
    ph = pd.read_csv(PHENO)
    ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    ph = ph.set_index("key")
    sub = ph.reindex([str(s).split("_")[-1].lstrip("0") for s in sids])
    num = lambda c: pd.to_numeric(sub[c], errors="coerce").mask(lambda v: v < -9000).to_numpy()
    pheno = dict(age=num("AGE_AT_SCAN"), fd=num("func_mean_fd"), sex=num("SEX"))
    return y_all, site_all, out, pheno


def flatten(rows):
    te = np.concatenate([r["te"] for r in rows])
    pm = np.concatenate([r["pm"] for r in rows])
    delta = np.concatenate([np.full(len(r["te"]), r["delta"]) for r in rows])
    return te, pm, delta


def boot_sites(site, reps, rng):
    sites = np.unique(site)
    by = {s: np.flatnonzero(site == s) for s in sites}
    for _ in range(reps):
        yield np.concatenate([by[s] for s in rng.choice(sites, sites.size, replace=True)])


# ------------------------------------------------------------------------------------ A
def new_site_calibration(y_all, site_all, rows, reps, rng):
    sites_n = pd.Series(site_all).value_counts()
    big = [s for s in sites_n.index if sites_n[s] >= 30]
    tau_prior = 0.37                                          # between-site SD found in the study
    per_site = {}
    for r in rows:
        s = site_all[r["te"][0]]
        if s in big:
            z = logit(r["pm"])
            per_site[s] = dict(z_nat=z, z_glob=z + r["delta"], y=y_all[r["te"]], prev_tr=r["prev_tr"],
                               prev_s=float(y_all[r["te"]].mean()), n=len(r["te"]))

    def hetero(key, adjust=None, idx_fn=None):
        est, se = [], []
        for s, d in per_site.items():
            z, y = d[key], d["y"]
            if idx_fn is not None:
                i = idx_fn(s)
                z, y = z[i], y[i]
            if adjust is not None:
                z = z + adjust(s, d, y)
            e, v = fit_delta(z, y)
            est.append(e)
            se.append(v)
        i2, tau = dl(est, se)
        return dict(I2=i2, tau=tau, mean_abs_citl=float(np.mean(np.abs(est))),
                    n_sites_ci_excludes_zero=int(sum(abs(e) > 1.96 * v for e, v in zip(est, se))),
                    citl=[float(e) for e in est])

    res = {"n_sites": len(big), "sites": big}
    res["native"] = hetero("z_nat")
    res["global_recalibration (current)"] = hetero("z_glob")
    # oracle: the site's own autistic proportion moves the intercept by the log odds ratio of the proportions
    orc = lambda s, d, y: logit(d["prev_s"]) - logit(d["prev_tr"])
    res["base-rate shift, true site proportion (oracle)"] = hetero("z_glob", orc)
    for k in (10, 15):
        for name, fn in (("base-rate shift from k labelled participants", "prior"),
                         ("local intercept from k labelled participants, partial pooling", "local")):
            agg = []
            for _ in range(reps):
                chosen = {s: rng.permutation(d["n"]) for s, d in per_site.items()}
                cal = {s: chosen[s][:k] for s in per_site}
                rem = {s: chosen[s][k:] for s in per_site}

                def adj(s, d, y_rem, fn=fn, cal=cal):
                    c = cal[s]
                    if fn == "prior":
                        m = 10.0                                 # pseudo-participants at the training proportion
                        pi = (d["y"][c].sum() + m * d["prev_tr"]) / (k + m)
                        return logit(pi) - logit(d["prev_tr"])
                    dl_, _ = fit_delta(d["z_glob"][c], d["y"][c], prior_var=tau_prior ** 2)
                    return dl_
                agg.append(hetero("z_glob", adj, idx_fn=lambda s, rem=rem: rem[s]))
            res[f"{name} (k={k})"] = dict(I2=float(np.mean([a["I2"] for a in agg])),
                                           tau=float(np.mean([a["tau"] for a in agg])),
                                           mean_abs_citl=float(np.mean([a["mean_abs_citl"] for a in agg])),
                                           n_sites_ci_excludes_zero=float(np.mean([a["n_sites_ci_excludes_zero"] for a in agg])))
        # the same remainder without any local information, for a like-for-like comparison
        agg = []
        for _ in range(reps):
            chosen = {s: rng.permutation(d["n"]) for s, d in per_site.items()}
            rem = {s: chosen[s][k:] for s in per_site}
            agg.append(hetero("z_glob", None, idx_fn=lambda s, rem=rem: rem[s]))
        res[f"global recalibration, same remainder (k={k})"] = dict(
            I2=float(np.mean([a["I2"] for a in agg])), tau=float(np.mean([a["tau"] for a in agg])),
            mean_abs_citl=float(np.mean([a["mean_abs_citl"] for a in agg])),
            n_sites_ci_excludes_zero=float(np.mean([a["n_sites_ci_excludes_zero"] for a in agg])))
    # tolerance: the assumed autistic proportion of the site is wrong by e
    tol = []
    for e in np.round(np.arange(-0.20, 0.2001, 0.05), 2):
        adj = lambda s, d, y, e=e: logit(np.clip(d["prev_s"] + e, 0.02, 0.98)) - logit(d["prev_tr"])
        h = hetero("z_glob", adj)
        tol.append(dict(error=float(e), I2=h["I2"], tau=h["tau"], mean_abs_citl=h["mean_abs_citl"],
                        n_sites_ci_excludes_zero=h["n_sites_ci_excludes_zero"]))
    res["tolerance"] = tol

    # label-free estimation of the site proportion by EM (Saerens et al.)
    def em_prior(p, pi_tr, iters=200):
        pi = pi_tr
        for _ in range(iters):
            a_, b_ = (pi / pi_tr) * p, ((1 - pi) / (1 - pi_tr)) * (1 - p)
            new = float((a_ / (a_ + b_)).mean())
            if abs(new - pi) < 1e-9:
                break
            pi = new
        return pi
    est = {s: em_prior(expit(d["z_glob"]), d["prev_tr"]) for s, d in per_site.items()}
    emadj = lambda s, d, y: logit(np.clip(est[s], 0.02, 0.98)) - logit(d["prev_tr"])
    h = hetero("z_glob", emadj)
    res["label-free EM estimate of the site proportion"] = dict(
        I2=h["I2"], tau=h["tau"], mean_abs_citl=h["mean_abs_citl"], n_sites_ci_excludes_zero=h["n_sites_ci_excludes_zero"],
        correlation_with_true_proportion=float(np.corrcoef([d["prev_s"] for d in per_site.values()],
                                                            [est[s] for s in per_site])[0, 1]))
    # per-site table for the figure
    orc_h = hetero("z_glob", orc)
    nat_h, glob_h = hetero("z_nat"), hetero("z_glob")
    res["per_site"] = [dict(site=s, n=int(per_site[s]["n"]), autistic_proportion=float(per_site[s]["prev_s"]),
                            training_proportion=float(per_site[s]["prev_tr"]), citl_native=nat_h["citl"][i],
                            citl_training_only=glob_h["citl"][i], citl_base_rate_shift=orc_h["citl"][i])
                       for i, s in enumerate(per_site)]
    res["site_prevalence"] = {s: round(d["prev_s"], 3) for s, d in per_site.items()}
    res["spearman_site_proportion_vs_offset"] = float(pd.Series([r["autistic_proportion"] for r in res["per_site"]])
                                                      .corr(pd.Series([r["citl_training_only"] for r in res["per_site"]]), method="spearman"))
    return res


# ------------------------------------------------------------------------------------ B
def pick_thresholds(pin, yin):
    """Thresholds from the training partition only (inner cross-fitted probabilities)."""
    grid = np.unique(np.round(pin, 4))
    sens = np.array([((pin >= t) & (yin == 1)).sum() / (yin == 1).sum() for t in grid])
    spec = np.array([((pin < t) & (yin == 0)).sum() / (yin == 0).sum() for t in grid])
    out = {}
    ok = np.flatnonzero(spec >= 0.80)
    out["specificity >= 0.80"] = grid[ok[0]] if len(ok) else grid[-1]       # lowest threshold that reaches it
    ok = np.flatnonzero(sens >= 0.80)
    out["sensitivity >= 0.80"] = grid[ok[-1]] if len(ok) else grid[0]       # highest threshold that reaches it
    out["Youden index"] = grid[np.argmax(sens + spec - 1)]
    return out


def operating_points(y_all, site_all, rows, reps, rng):
    res = {}
    te = np.concatenate([r["te"] for r in rows])
    y = y_all[te]
    site = site_all[te]
    z_rec = np.concatenate([logit(r["pm"]) + r["delta"] for r in rows])
    pr = expit(z_rec)
    for rule in ("specificity >= 0.80", "sensitivity >= 0.80", "Youden index", "fixed 0.5"):
        pred = np.concatenate([
            (pr[off:off + len(r["te"])] >= (0.5 if rule == "fixed 0.5" else
                                           pick_thresholds(expit(r["pin"] + r["delta"]), r["yin"])[rule])).astype(int)
            for r, off in zip(rows, np.cumsum([0] + [len(r["te"]) for r in rows[:-1]]))])
        tp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 0) & (y == 1)).sum()
        tn, fp = ((pred == 0) & (y == 0)).sum(), ((pred == 1) & (y == 0)).sum()
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        bs = []
        for idx in boot_sites(site, reps, rng):
            yy, pp = y[idx], pred[idx]
            if yy.min() == yy.max():
                continue
            bs.append((((pp == 1) & (yy == 1)).sum() / (yy == 1).sum(), ((pp == 0) & (yy == 0)).sum() / (yy == 0).sum()))
        bs = np.array(bs)
        res[rule] = dict(sensitivity=float(sens), sensitivity_ci=[float(v) for v in np.percentile(bs[:, 0], [2.5, 97.5])],
                         specificity=float(spec), specificity_ci=[float(v) for v in np.percentile(bs[:, 1], [2.5, 97.5])],
                         ppv=float(tp / max(tp + fp, 1)), npv=float(tn / max(tn + fn, 1)),
                         fraction_flagged=float(pred.mean()))
    # net benefit of the recalibrated and of the native probabilities
    pn = np.concatenate([r["pm"] for r in rows])
    prev = y.mean()
    nb = {}
    for pt in (0.30, 0.40, 0.50):
        w = pt / (1 - pt)
        f = lambda p, yy: (((p >= pt) & (yy == 1)).sum() - w * ((p >= pt) & (yy == 0)).sum()) / len(yy)
        allnb = prev - (1 - prev) * w
        bsd = []
        for idx in boot_sites(site, reps, rng):
            yy = y[idx]
            bsd.append((f(pr[idx], yy) - (yy.mean() - (1 - yy.mean()) * w),
                        f(pn[idx], yy) - (yy.mean() - (1 - yy.mean()) * w)))
        bsd = np.array(bsd)
        nb[f"{pt:.2f}"] = dict(model_recalibrated=float(f(pr, y)), model_native=float(f(pn, y)), treat_all=float(allnb),
                               gain_over_treat_all_recalibrated=float(f(pr, y) - allnb),
                               gain_ci=[float(v) for v in np.percentile(bsd[:, 0], [2.5, 97.5])],
                               gain_native=float(f(pn, y) - allnb),
                               gain_native_ci=[float(v) for v in np.percentile(bsd[:, 1], [2.5, 97.5])])
    res["net_benefit"] = nb
    for rule in ("specificity >= 0.80", "Youden index", "sensitivity >= 0.80", "fixed 0.5"):
        se_, sp_ = res[rule]["sensitivity"], res[rule]["specificity"]
        res[rule]["predictive_values"] = {
            f"{prev_:.2f}": dict(ppv=float(se_ * prev_ / (se_ * prev_ + (1 - sp_) * (1 - prev_))),
                                 npv=float(sp_ * (1 - prev_) / (sp_ * (1 - prev_) + (1 - se_) * prev_)))
            for prev_ in (0.03, 0.10, 0.25, 0.46)}
    curve = []
    for pt in np.round(np.arange(0.20, 0.7001, 0.05), 2):
        w = pt / (1 - pt)
        f = lambda p, yy: (((p >= pt) & (yy == 1)).sum() - w * ((p >= pt) & (yy == 0)).sum()) / len(yy)
        allnb = lambda yy: yy.mean() - (1 - yy.mean()) * w
        bs = np.array([(f(pr[i], y[i]), f(pn[i], y[i]), allnb(y[i])) for i in boot_sites(site, reps, rng)])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        curve.append(dict(threshold=float(pt), recalibrated=float(f(pr, y)), native=float(f(pn, y)),
                          treat_all=float(allnb(y)), recalibrated_ci=[float(lo[0]), float(hi[0])],
                          native_ci=[float(lo[1]), float(hi[1])], treat_all_ci=[float(lo[2]), float(hi[2])]))
    res["decision_curve"] = curve
    return res


# ------------------------------------------------------------------------------------ C
def stratified(y_all, site_all, rows, pheno, reps, rng):
    te, pm, delta = flatten(rows)
    y, site = y_all[te], site_all[te]
    z = logit(pm)
    zr = z + delta
    age, fd = pheno["age"][te], pheno["fd"][te]
    n_site = pd.Series(site_all).map(pd.Series(site_all).value_counts()).to_numpy()[te]
    strata = {}
    strata["age < 12"] = age < 12
    strata["age 12-17"] = (age >= 12) & (age < 18)
    strata["age >= 18"] = age >= 18
    q = np.nanpercentile(fd, [33.3, 66.7])
    strata["motion: lowest third"] = fd <= q[0]
    strata["motion: middle third"] = (fd > q[0]) & (fd <= q[1])
    strata["motion: highest third"] = fd > q[1]
    strata["site size < 40"] = n_site < 40
    strata["site size 40-80"] = (n_site >= 40) & (n_site < 80)
    strata["site size >= 80"] = n_site >= 80
    out = {}
    for name, m in strata.items():
        m = np.asarray(m, bool)
        a = auroc(pm[m], y[m])
        d_nat, se_nat = fit_delta(z[m], y[m])
        d_rec, se_rec = fit_delta(zr[m], y[m])
        bs = []
        for idx in boot_sites(site, reps, rng):
            mm = m[idx]
            if mm.sum() < 15:
                continue
            v = auroc(pm[idx][mm], y[idx][mm])
            if not np.isnan(v):
                bs.append(v)
        prev_tr = float(np.mean(np.concatenate([np.full(len(r["te"]), r["prev_tr"]) for r in rows])))
        d_shift, se_shift = fit_delta(zr[m] + logit(np.clip(y[m].mean(), 0.02, 0.98)) - logit(prev_tr), y[m])
        out[name] = dict(n=int(m.sum()), autistic_share=float(y[m].mean()), auroc=a,
                         citl_after_base_rate_shift=d_shift,
                         citl_after_base_rate_shift_ci=[d_shift - 1.96 * se_shift, d_shift + 1.96 * se_shift],
                         auroc_ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])] if len(bs) > 50 else None,
                         citl_native=d_nat, citl_native_ci=[d_nat - 1.96 * se_nat, d_nat + 1.96 * se_nat],
                         citl_recalibrated=d_rec, citl_recalibrated_ci=[d_rec - 1.96 * se_rec, d_rec + 1.96 * se_rec])
    # AUROC differences between the extreme strata (site-clustered bootstrap)
    diffs = {}
    for lo_name, hi_name in (("age < 12", "age >= 18"), ("age 12-17", "age >= 18"), ("age < 12", "age 12-17"),
                             ("motion: lowest third", "motion: highest third"),
                             ("site size < 40", "site size >= 80")):
        ma, mb = np.asarray(strata[lo_name], bool), np.asarray(strata[hi_name], bool)
        d_obs = auroc(pm[ma], y[ma]) - auroc(pm[mb], y[mb])
        bs = []
        for idx in boot_sites(site, reps, rng):
            a1, b1 = auroc(pm[idx][ma[idx]], y[idx][ma[idx]]), auroc(pm[idx][mb[idx]], y[idx][mb[idx]])
            if not (np.isnan(a1) or np.isnan(b1)):
                bs.append(a1 - b1)
        diffs[f"{lo_name} minus {hi_name}"] = dict(difference=float(d_obs),
                                                    ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])])
    out["_auroc_differences"] = diffs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=str(ROOT / "results" / "latest_study"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=400)
    a = ap.parse_args()
    study = Path(a.study).resolve()
    rng = np.random.default_rng(42)
    y_all, site_all, data, pheno = load(study)
    # the first check: the recalibration shift reproduces the stored parameters
    par = pd.read_csv(study / "review" / "calibration_recalibration_params.csv")
    for proto in ("pooled", "loso"):
        mine = np.array([r["delta"] for r in data[proto]])
        stored = par[par.protocol == proto].sort_values("fold").logistic_intercept.to_numpy()
        print(f"{proto}: max |recomputed shift - stored| = {np.abs(mine - stored).max():.2e}")
    out = {"A_new_site_calibration": new_site_calibration(y_all, site_all, data["loso"], a.reps, rng)}
    out["B_operating_points_pooled"] = operating_points(y_all, site_all, data["pooled"], a.boot, rng)
    out["B_operating_points_loso"] = operating_points(y_all, site_all, data["loso"], a.boot, rng)
    out["C_stratified_pooled"] = stratified(y_all, site_all, data["pooled"], pheno, a.boot, rng)
    out["C_stratified_loso"] = stratified(y_all, site_all, data["loso"], pheno, a.boot, rng)
    (study / "review" / "trust_checks.json").write_text(json.dumps(out, indent=1, default=float))
    rev = study / "review"
    pd.DataFrame(out["A_new_site_calibration"]["per_site"]).to_csv(rev / "trust_new_site_per_site.csv", index=False)
    pd.DataFrame(out["A_new_site_calibration"]["tolerance"]).to_csv(rev / "trust_base_rate_tolerance.csv", index=False)
    for proto in ("pooled", "loso"):
        pd.DataFrame(out[f"B_operating_points_{proto}"]["decision_curve"]).to_csv(rev / f"trust_decision_curve_{proto}.csv", index=False)
        rows_ = []
        for k, v in out[f"C_stratified_{proto}"].items():
            if k.startswith("_"):
                continue
            rows_.append(dict(stratum=k, n=v["n"], autistic_share=v["autistic_share"], auroc=v["auroc"],
                              auroc_lo=(v["auroc_ci"] or [np.nan, np.nan])[0], auroc_hi=(v["auroc_ci"] or [np.nan, np.nan])[1],
                              citl_native=v["citl_native"], citl_recalibrated=v["citl_recalibrated"],
                              citl_after_base_rate_shift=v["citl_after_base_rate_shift"]))
        pd.DataFrame(rows_).to_csv(rev / f"trust_strata_{proto}.csv", index=False)
    print("written", study / "review" / "trust_checks.json")


if __name__ == "__main__":
    main()
