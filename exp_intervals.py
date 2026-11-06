"""Confidence intervals for the headline results, and discrimination by subgroup.

Section 12 promises intervals; this computes them.  AUROC intervals use the DeLong variance,
accuracy and the operating-point rates use Wilson intervals, and per-site performance is
combined by a DerSimonian-Laird random-effects model so that a prediction interval for a new
site can be quoted -- the quantity a reader actually wants and that no comparison study reports.

The subgroup pass links stored out-of-fold predictions to phenotype.  Those predictions are
written in fold-concatenation order and carry no participant identifier, so the fold order is
replayed and the reconstruction is *verified* element-wise against both the stored labels and
the stored site codes before any phenotype is joined; nothing is reported unless that check
passes for every participant.  (New runs carry ``oof_sid`` and need none of this.)
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

MISSING = -9999.0


def delong_var(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """AUROC and its DeLong variance."""
    x, z = p[y == 1], p[y == 0]
    m, n = len(x), len(z)
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    # psi(a, b) = 1 if a > b, 0.5 if equal, 0 otherwise
    d = x[:, None] - z[None, :]
    psi = (d > 0).astype(float) + 0.5 * (d == 0)
    auc = psi.mean()
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    var = (v10.var(ddof=1) / m if m > 1 else 0.0) + (v01.var(ddof=1) / n if n > 1 else 0.0)
    return float(auc), float(var)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, ctr - half)), float(min(1.0, ctr + half))


def ci(auc: float, var: float, z: float = 1.96) -> tuple[float, float]:
    se = np.sqrt(var)
    return float(max(0.0, auc - z * se)), float(min(1.0, auc + z * se))


def main() -> None:
    ap = argparse.ArgumentParser(description="Intervals and subgroup discrimination")
    ap.add_argument("--study", required=True)
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--pheno", default="abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    args = ap.parse_args()

    from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
    from run_benchmark import load_dataset
    from scipy import stats

    study = Path(args.study)
    oof = {r["protocol"]: r for r in json.loads((study / "ensemble_oof.json").read_text())}

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    y = np.asarray(y); sites = np.asarray(sites); sids = np.asarray(sids)

    rows, per_site_rows = [], []
    order = {}
    for proto, splitter, groups in (
            ("pooled", StratifiedKFold(10, shuffle=True, random_state=42), None),
            ("loso", LeaveOneGroupOut(), sites)):
        if proto not in oof:
            continue
        r = oof[proto]
        yy = np.asarray(r["oof_true"]); pp = np.asarray(r["oof_prob"], dtype=float)
        ss = np.asarray(r["oof_site"])

        idx = []
        for _, te in (splitter.split(np.zeros(len(y)), y, groups) if groups is not None
                      else splitter.split(np.zeros(len(y)), y)):
            idx.extend(te.tolist())
        idx = np.array(idx)
        verified = bool((y[idx] == yy).all() and (sites[idx] == ss).all())
        print(f"{proto}: linkage verified against labels and sites -> {verified}")
        if not verified:
            raise SystemExit(f"{proto}: could not verify participant linkage; refusing to report")
        order[proto] = idx

        auc, var = delong_var(yy, pp)
        lo, hi = ci(auc, var)
        pred = (pp >= 0.5).astype(int)
        acc = float((pred == yy).mean())
        alo, ahi = wilson(int((pred == yy).sum()), len(yy))
        tp = int(((pred == 1) & (yy == 1)).sum()); npos = int((yy == 1).sum())
        tn = int(((pred == 0) & (yy == 0)).sum()); nneg = int((yy == 0).sum())
        slo, shi = wilson(tp, npos); plo, phi = wilson(tn, nneg)
        rows.append({"protocol": proto, "n": len(yy), "auroc": auc,
                     "auroc_lo": lo, "auroc_hi": hi, "accuracy": acc,
                     "accuracy_lo": alo, "accuracy_hi": ahi,
                     "sensitivity": tp / npos, "sensitivity_lo": slo, "sensitivity_hi": shi,
                     "specificity": tn / nneg, "specificity_lo": plo, "specificity_hi": phi})

        if proto == "loso":
            for st in np.unique(ss):
                m = ss == st
                if len(set(yy[m])) < 2:
                    continue
                a_, v_ = delong_var(yy[m], pp[m])
                l_, h_ = ci(a_, v_)
                per_site_rows.append({"site": st, "n": int(m.sum()), "auroc": a_,
                                      "var": v_, "auroc_lo": l_, "auroc_hi": h_})

    df = pd.DataFrame(rows)
    ps = pd.DataFrame(per_site_rows)

    # ---- random-effects combination across sites, and a prediction interval ----
    meta = {}
    if len(ps) > 2:
        yi, vi = ps["auroc"].to_numpy(), ps["var"].to_numpy()
        w = 1.0 / vi
        fixed = float((w * yi).sum() / w.sum())
        Q = float((w * (yi - fixed) ** 2).sum())
        k = len(yi)
        tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        ws = 1.0 / (vi + tau2)
        mu = float((ws * yi).sum() / ws.sum())
        se_mu = float(np.sqrt(1.0 / ws.sum()))
        tcrit = float(stats.t.ppf(0.975, max(k - 2, 1)))
        pi = (mu - tcrit * np.sqrt(tau2 + se_mu ** 2), mu + tcrit * np.sqrt(tau2 + se_mu ** 2))
        meta = {"k_sites": k, "random_effects_mean": mu, "se": se_mu,
                "Q": Q, "df": k - 1, "Q_p": float(1 - stats.chi2.cdf(Q, k - 1)),
                "I2": float(max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0),
                "tau": float(np.sqrt(tau2)),
                "prediction_interval_lo": float(pi[0]), "prediction_interval_hi": float(pi[1])}
        print(f"\nper-site random-effects mean {mu:.3f}, Q = {Q:.1f} on {k-1} df "
              f"(p = {meta['Q_p']:.3f}), I2 = {meta['I2']:.1f}%")
        print(f"95% prediction interval for a new site: "
              f"[{pi[0]:.3f}, {pi[1]:.3f}]")

    # ---- discrimination by full-scale IQ and by sex ----
    ph = pd.read_csv(args.pheno)
    ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    ph = ph.set_index("key")
    sub = ph.reindex([str(s).split("_")[-1].lstrip("0") for s in sids])

    def clean(col):
        v = pd.to_numeric(sub[col], errors="coerce")
        return v.mask(v <= MISSING + 1).to_numpy(dtype=float)   # sentinels -> NaN, never imputed

    fiq, sex = clean("FIQ"), clean("SEX")
    viq, piq = clean("VIQ"), clean("PIQ")
    age, fd = clean("AGE_AT_SCAN"), clean("func_mean_fd")
    sub_rows = []
    for proto, idx in order.items():
        r = oof[proto]
        yy = np.asarray(r["oof_true"]); pp = np.asarray(r["oof_prob"], dtype=float)
        f_ord, s_ord, site_ord = fiq[idx], sex[idx], sites[idx]

        # centre IQ within site: site means differ, so raw quartiles would partly index site
        ok = np.isfinite(f_ord)
        fc = np.full(len(f_ord), np.nan)
        for st in np.unique(site_ord):
            m = (site_ord == st) & ok
            if m.sum() > 1:
                fc[m] = f_ord[m] - np.nanmean(f_ord[m])
        good = np.isfinite(fc)
        q = pd.qcut(fc[good], 4, labels=False)
        for qi in range(4):
            sel = np.where(good)[0][q == qi]
            a_, v_ = delong_var(yy[sel], pp[sel])
            l_, h_ = ci(a_, v_)
            sub_rows.append({"protocol": proto, "variable": "FIQ (site-centred)",
                             "group": f"Q{qi+1}", "n": len(sel),
                             "median_FIQ": float(np.nanmedian(f_ord[sel])),
                             "auroc": a_, "auroc_lo": l_, "auroc_hi": h_})
        # first-to-fourth difference, and a trend test on the quartile index
        qq = np.full(len(yy), np.nan); qq[np.where(good)[0]] = q
        tr = stats.spearmanr(qq[np.isfinite(qq)],
                             (pp[np.isfinite(qq)] > 0.5).astype(int) == yy[np.isfinite(qq)])
        print(f"{proto}: IQ-quartile trend in correctness Spearman rho = {tr.statistic:+.3f} "
              f"(p = {tr.pvalue:.4f})")

        # The comparison group is held fixed at every control while only the autistic side
        # is stratified.  Every stratum is then scored against the same reference, so a
        # difference between strata cannot be produced by the strata differing in case mix
        # -- which is what makes this, rather than a quartile split of everyone, the
        # analysis that speaks to detectability.
        ctrl = np.where(yy == 0)[0]
        for lo, hi, lab in ((-np.inf, 85, "FIQ < 85"), (85, 100, "FIQ 85-99"),
                            (100, 115, "FIQ 100-114"), (115, np.inf, "FIQ >= 115")):
            aut = np.where((yy == 1) & np.isfinite(f_ord) & (f_ord >= lo) & (f_ord < hi))[0]
            if len(aut) < 20:
                continue
            take = np.concatenate([aut, ctrl])
            a_, v_ = delong_var(yy[take], pp[take])
            l_, h_ = ci(a_, v_)
            sub_rows.append({"protocol": proto,
                             "variable": "FIQ band, autistic side only (controls held fixed)",
                             "group": lab, "n": len(aut),
                             "median_FIQ": float(np.nanmedian(f_ord[aut])),
                             "auroc": a_, "auroc_lo": l_, "auroc_hi": h_})

        # within the autistic group there is no group separation to confound the question,
        # so the same gradient is tested directly against a control-group null
        def partial_rho(mask, x):
            m = mask & np.isfinite(x) & np.isfinite(age_o) & np.isfinite(fd_o)
            if m.sum() < 30:
                return None
            D = pd.get_dummies(pd.Series(site_ord[m]), drop_first=True).to_numpy(float)
            C = np.column_stack([np.ones(m.sum()), age_o[m], fd_o[m], D])
            rx = x[m] - C @ np.linalg.lstsq(C, x[m], rcond=None)[0]
            rp = pp[m] - C @ np.linalg.lstsq(C, pp[m], rcond=None)[0]
            r_ = stats.spearmanr(rx, rp)
            return int(m.sum()), float(r_.statistic), float(r_.pvalue)

        age_o, fd_o = age[idx], fd[idx]
        for lab, xv in (("FIQ", f_ord), ("VIQ", viq[idx]), ("PIQ", piq[idx])):
            for grp, gl in ((1, "autistic"), (0, "control")):
                out_ = partial_rho(yy == grp, xv)
                if out_:
                    n_, r_, p_ = out_
                    sub_rows.append({"protocol": proto,
                                     "variable": f"score vs {lab}, residualised on site/age/motion",
                                     "group": gl, "n": n_, "median_FIQ": float("nan"),
                                     "auroc": r_, "auroc_lo": float("nan"),
                                     "auroc_hi": float("nan"), "p": p_})
                    print(f"{proto}: score vs {lab} within {gl}s rho = {r_:+.3f} "
                          f"(p = {p_:.4f}, n = {n_})")

        # An IQ gradient across everyone is not by itself evidence about detectability: autistic
        # participants have lower FIQ, so low-IQ strata are also the strata in which the two
        # groups differ most on IQ.  Matching each autistic participant to a control at the
        # same site with near-identical FIQ removes that separation by design.
        chosen = []
        for st in np.unique(site_ord):
            at = [i for i in np.where((site_ord == st) & (yy == 1) & ok)[0]]
            ct = [i for i in np.where((site_ord == st) & (yy == 0) & ok)[0]]
            for i in sorted(at, key=lambda k: f_ord[k]):
                if not ct:
                    break
                j = min(ct, key=lambda k: abs(f_ord[k] - f_ord[i]))
                if abs(f_ord[j] - f_ord[i]) <= 5.0:
                    chosen += [i, j]; ct.remove(j)
        msel = np.array(sorted(chosen))
        if len(msel) >= 100:
            fm, ym, pm, sm_ = f_ord[msel], yy[msel], pp[msel], site_ord[msel]
            after = stats.mannwhitneyu(fm[ym == 1], fm[ym == 0]).pvalue
            fcm = np.full(len(fm), np.nan)
            for st in np.unique(sm_):
                m = sm_ == st
                if m.sum() > 1:
                    fcm[m] = fm[m] - np.nanmean(fm[m])
            g2 = np.isfinite(fcm)
            q2 = pd.qcut(fcm[g2], 4, labels=False)
            for qi in range(4):
                s2 = np.where(g2)[0][q2 == qi]
                a_, v_ = delong_var(ym[s2], pm[s2])
                l_, h_ = ci(a_, v_)
                sub_rows.append({"protocol": proto,
                                 "variable": "FIQ (site-centred, IQ-matched pairs)",
                                 "group": f"Q{qi+1}", "n": len(s2),
                                 "median_FIQ": float(np.nanmedian(fm[s2])),
                                 "auroc": a_, "auroc_lo": l_, "auroc_hi": h_})
            print(f"{proto}: {len(msel)//2} FIQ-matched pairs, group IQ difference "
                  f"p = {after:.3f} after matching")

        for label, val in (("female", 2.0), ("male", 1.0)):
            sel = np.where(s_ord == val)[0]
            if len(sel) < 20 or len(set(yy[sel])) < 2:
                continue
            a_, v_ = delong_var(yy[sel], pp[sel])
            l_, h_ = ci(a_, v_)
            sub_rows.append({"protocol": proto, "variable": "sex", "group": label,
                             "n": len(sel), "median_FIQ": float("nan"),
                             "auroc": a_, "auroc_lo": l_, "auroc_hi": h_})

    sd = pd.DataFrame(sub_rows)
    t = study / "tables"; t.mkdir(parents=True, exist_ok=True)
    df.to_csv(t / "intervals.csv", index=False)
    ps.drop(columns=["var"]).to_csv(t / "per_site_intervals.csv", index=False)
    sd.to_csv(t / "subgroup_auroc.csv", index=False)
    (study / "raw").mkdir(parents=True, exist_ok=True)
    (study / "raw" / "intervals.json").write_text(json.dumps(
        {"headline": rows, "meta_analysis": meta,
         "linkage_verified": True}, indent=2))

    print("\n" + "=" * 74)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n" + sd.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {t/'intervals.csv'}, {t/'per_site_intervals.csv'}, {t/'subgroup_auroc.csv'}")


if __name__ == "__main__":
    main()
