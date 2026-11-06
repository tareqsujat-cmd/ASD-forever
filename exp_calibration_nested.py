"""
Calibration of the primary ensemble with recalibration fitted strictly inside each outer
training partition.

Input is the set of outer-fold refits written by exp_nested_outer.py.  For every outer fold
the ensemble probability of each outer-training participant was obtained from a model that
had not seen that participant (inner cross-fitting), and a recalibration map is fitted on
those predictions and the training labels only.  The map is then applied to the held-out
fold, so no held-out label reaches it.

Three maps are compared with the native probabilities.

  intercept only   logit p' = logit p + a            (slope fixed at one)
  logistic         logit p' = a + b logit p          (Platt scaling)
  isotonic         p' = monotone step function of p  (clipped to [0.005, 0.995])

The intercept-only map is designated the primary recalibration before any recalibrated
value was computed.  The reason is that the native probabilities already showed a slope
interval that included one, so the only defect to correct was the average offset; the
logistic and isotonic maps are sensitivity analyses.  A fifth row is a diagnostic rather
than a recalibration: the same ensemble refitted with an unpenalised intercept, which tests
whether the native offset comes from the solver.

Every measure is reported with a site-clustered bootstrap interval, and differences from the
native probabilities are computed on the same resamples so that they are paired.

Usage:
    .venv/bin/python exp_calibration_nested.py [--reps 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
STUDY = ROOT / "results" / "latest_study"
NESTED = STUDY / "review" / "nested"
OUT = STUDY / "review"
SEED = 42
EPS = 1e-6
ISO_CLIP = (0.005, 0.995)
VARIANTS = ["native", "intercept only", "logistic", "isotonic", "unpenalised intercept"]
ROLE = {"native": "reference", "intercept only": "primary recalibration",
        "logistic": "sensitivity", "isotonic": "sensitivity",
        "unpenalised intercept": "diagnostic (not a recalibration)"}
METRICS = ["AUROC", "accuracy", "sensitivity", "specificity", "Brier", "Brier skill score",
           "log loss", "calibration-in-the-large", "calibration slope",
           "ECE (5 bins)", "ECE (10 bins)", "ECE (20 bins)"]


# =========================================================================================
# small numerical helpers (Newton steps, so a bootstrap of thousands of fits stays fast)
# =========================================================================================
def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def citl_offset(z, y, iters=50):
    """Intercept of y ~ 1 with z as a fixed offset (calibration-in-the-large)."""
    a = 0.0
    for _ in range(iters):
        q = expit(z + a)
        h = np.sum(q * (1 - q))
        if h < 1e-12:
            break
        step = np.sum(y - q) / h
        a += step
        if abs(step) < 1e-10:
            break
    return a


def logistic_ab(z, y, iters=50):
    """Intercept and slope of y ~ a + b z by Newton's method."""
    a, b = 0.0, 1.0
    for _ in range(iters):
        q = expit(a + b * z)
        w = q * (1 - q)
        g = np.array([np.sum(y - q), np.sum((y - q) * z)])
        H = np.array([[w.sum(), (w * z).sum()], [(w * z).sum(), (w * z * z).sum()]])
        try:
            step = np.linalg.solve(H + 1e-10 * np.eye(2), g)
        except np.linalg.LinAlgError:
            break
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-9:
            break
    return a, b


def ece(y, p, bins):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return tot


def metrics(y, p):
    z = logit(p)
    pi = y.mean()
    brier = float(np.mean((p - y) ** 2))
    pc = np.clip(p, EPS, 1 - EPS)
    pred = p >= 0.5
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "accuracy": float((pred == y).mean()),
        "sensitivity": float(pred[y == 1].mean()),
        "specificity": float((~pred[y == 0]).mean()),
        "Brier": brier,
        "Brier skill score": 1.0 - brier / (pi * (1 - pi)),
        "log loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "calibration-in-the-large": float(citl_offset(z, y)),
        "calibration slope": float(logistic_ab(z, y)[1]),
        "ECE (5 bins)": ece(y, p, 5),
        "ECE (10 bins)": ece(y, p, 10),
        "ECE (20 bins)": ece(y, p, 20),
    }


# =========================================================================================
# assembling the held-out predictions and the within-training recalibration maps
# =========================================================================================
def load_folds(protocol: str):
    files = [f for f in sorted(NESTED.glob(f"fold_{protocol}_*.npz")) if ".tmp" not in f.name]
    if not files:
        raise SystemExit(f"no outer-fold files for {protocol} in {NESTED}")
    return [np.load(f) for f in files]


def stored_reference(protocol: str):
    d = json.loads((STUDY / "raw" / "voting.json").read_text())
    for r in d:
        if r["condition"].startswith("ensemble (probability mean)") and r["protocol"] == protocol:
            return (np.asarray(r["oof_true"], int), np.asarray(r["oof_prob"], float),
                    np.asarray(r["oof_site"]))
    raise SystemExit("stored ensemble predictions not found")


def assemble(protocol: str):
    folds = load_folds(protocol)
    lab = np.load(STUDY / "cache" / "nested" / "labels.npz")
    y_all, sites_all = lab["y"].astype(int), lab["sites"]

    order = np.concatenate([f["te_idx"] for f in folds])
    y, site = y_all[order], sites_all[order]
    native = np.concatenate([f["p_raw"].mean(0) for f in folds])
    unpen = np.concatenate([f["p_unpen"].mean(0) for f in folds])

    # the refit must reproduce the stored predictions, in the stored order
    ys, ps, ss = stored_reference(protocol)
    n = len(y)                                        # a partial run is compared on its prefix
    check = dict(labels_match=bool((ys[:n] == y).all()), sites_match=bool((ss[:n] == site).all()),
                 max_abs_diff_native=float(np.abs(ps[:n] - native).max()),
                 n_heldout=int(n), n_stored=int(len(ys)))
    if not (check["labels_match"] and check["sites_match"]):
        raise SystemExit(f"{protocol}: stored order does not match the refit order: {check}")

    rec = {k: [] for k in ("intercept only", "logistic", "isotonic")}
    params = []
    for f in folds:
        tr, te = f["tr_idx"], f["te_idx"]
        p_in = f["inner_p"].mean(0)                      # ensemble, cross-fitted inside training
        if np.isnan(p_in).any():
            raise SystemExit("inner predictions are missing; run without --outer-only")
        y_in, z_in, p_out = y_all[tr], logit(p_in), f["p_raw"].mean(0)
        z_out = logit(p_out)
        a0 = citl_offset(z_in, y_in)
        a1, b1 = logistic_ab(z_in, y_in)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=ISO_CLIP[0], y_max=ISO_CLIP[1])
        iso.fit(p_in, y_in)
        rec["intercept only"].append(expit(z_out + a0))
        rec["logistic"].append(expit(a1 + b1 * z_out))
        rec["isotonic"].append(iso.predict(p_out))
        params.append(dict(protocol=protocol, fold=len(params), n_train=len(tr), n_test=len(te),
                           train_prevalence=float(y_in.mean()),
                           inner_mean_prediction=float(p_in.mean()),
                           inner_citl=float(a0), logistic_intercept=float(a1),
                           logistic_slope=float(b1)))
    probs = {"native": native, "unpenalised intercept": unpen}
    probs.update({k: np.concatenate(v) for k, v in rec.items()})
    return y, site, {k: probs[k] for k in VARIANTS}, pd.DataFrame(params), check


# =========================================================================================
# bootstrap
# =========================================================================================
def cluster_bootstrap(y, site, probs, reps, rng):
    sites = np.unique(site)
    by_site = {s: np.flatnonzero(site == s) for s in sites}
    out = np.full((reps, len(VARIANTS), len(METRICS)), np.nan)
    for r in range(reps):
        idx = np.concatenate([by_site[s] for s in rng.choice(sites, sites.size, replace=True)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for v, name in enumerate(VARIANTS):
            m = metrics(yy, probs[name][idx])
            out[r, v] = [m[k] for k in METRICS]
    return out


def summarise(protocol, y, probs, boot):
    rows, paired = [], []
    for v, name in enumerate(VARIANTS):
        point = metrics(y, probs[name])
        for k, metric in enumerate(METRICS):
            lo, hi = np.nanpercentile(boot[:, v, k], [2.5, 97.5])
            rows.append(dict(protocol=protocol, variant=name, role=ROLE[name], metric=metric,
                             estimate=point[metric], lo=lo, hi=hi,
                             n_boot=int(np.isfinite(boot[:, v, k]).sum())))
            if name == "native":
                continue
            d = boot[:, v, k] - boot[:, 0, k]
            dl, dh = np.nanpercentile(d, [2.5, 97.5])
            base = metrics(y, probs["native"])[metric]
            paired.append(dict(protocol=protocol, variant=name, metric=metric,
                               native=base, variant_estimate=point[metric],
                               difference=point[metric] - base, lo=dl, hi=dh))
    return rows, paired


# =========================================================================================
# reliability bins and per-site calibration
# =========================================================================================
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def reliability(protocol, y, probs):
    rows = []
    for name in VARIANTS:
        p = probs[name]
        for scheme in ("equal width", "equal mass"):
            if scheme == "equal width":
                idx = np.clip(np.digitize(p, np.linspace(0, 1, 11)[1:-1]), 0, 9)
            else:
                edges = np.quantile(p, np.linspace(0, 1, 11)[1:-1])
                idx = np.digitize(p, edges)
            for b in range(10):
                m = idx == b
                if not m.any():
                    continue
                lo, hi = wilson(int(y[m].sum()), int(m.sum()))
                rows.append(dict(protocol=protocol, variant=name, scheme=scheme, bin=b,
                                 n=int(m.sum()), mean_predicted=float(p[m].mean()),
                                 observed=float(y[m].mean()), obs_lo=lo, obs_hi=hi))
    return rows


def per_site(protocol, y, site, probs):
    rows, het = [], []
    for name in ("native", "intercept only", "logistic"):
        z_all = logit(probs[name])
        est, var = [], []
        for s in np.unique(site):
            m = site == s
            n, k = int(m.sum()), int(y[m].sum())
            row = dict(protocol=protocol, variant=name, site=s, n=n, n_autistic=k,
                       prevalence=k / n, mean_predicted=float(probs[name][m].mean()))
            if 0 < k < n:
                a = citl_offset(z_all[m], y[m])
                q = expit(z_all[m] + a)
                se = 1 / np.sqrt(np.sum(q * (1 - q)))
                row.update(citl=a, citl_se=se, citl_lo=a - 1.96 * se, citl_hi=a + 1.96 * se)
                if n >= 30:
                    est.append(a)
                    var.append(se ** 2)
            rows.append(row)
        est, var = np.asarray(est), np.asarray(var)
        w = 1 / var
        mu = np.sum(w * est) / w.sum()
        Q = float(np.sum(w * (est - mu) ** 2))
        dfree = len(est) - 1
        tau2 = max(0.0, (Q - dfree) / (w.sum() - np.sum(w ** 2) / w.sum()))
        het.append(dict(protocol=protocol, variant=name, n_sites=len(est),
                        mean_site_citl=float(est.mean()), sd_site_citl=float(est.std(ddof=1)),
                        min_site_citl=float(est.min()), max_site_citl=float(est.max()),
                        Q=Q, df=dfree, I2=max(0.0, (Q - dfree) / Q) if Q > 0 else 0.0,
                        tau=float(np.sqrt(tau2)),
                        n_sites_ci_excludes_zero=int(sum(
                            abs(e) > 1.96 * np.sqrt(v) for e, v in zip(est, var)))))
    return rows, het


# =========================================================================================
def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--protocols", nargs="+", default=["pooled", "loso"])
    ap.add_argument("--out", default=str(OUT), help="directory for the result files")
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows, all_paired, rel, site_rows, het_rows, params, checks = [], [], [], [], [], [], {}
    for protocol in args.protocols:
        y, site, probs, par, check = assemble(protocol)
        checks[protocol] = check
        print(f"{protocol}: {len(y)} held-out predictions; native max|diff| vs stored = "
              f"{check['max_abs_diff_native']:.2e}", flush=True)
        boot = cluster_bootstrap(y, site, probs, args.reps, np.random.default_rng(SEED))
        rows, paired = summarise(protocol, y, probs, boot)
        all_rows += rows
        all_paired += paired
        rel += reliability(protocol, y, probs)
        sr, hr = per_site(protocol, y, site, probs)
        site_rows += sr
        het_rows += hr
        params.append(par)
        for v in VARIANTS:
            m = metrics(y, probs[v])
            print(f"  {v:22s} AUROC {m['AUROC']:.3f}  acc {m['accuracy']:.3f}  Brier {m['Brier']:.4f}  "
                  f"CITL {m['calibration-in-the-large']:+.3f}  slope {m['calibration slope']:.3f}  "
                  f"ECE10 {m['ECE (10 bins)']:.3f}", flush=True)

    pd.DataFrame(all_rows).to_csv(OUT / "calibration_nested.csv", index=False)
    pd.DataFrame(all_paired).to_csv(OUT / "calibration_nested_paired.csv", index=False)
    pd.DataFrame(rel).to_csv(OUT / "calibration_reliability.csv", index=False)
    pd.DataFrame(site_rows).to_csv(OUT / "calibration_per_site.csv", index=False)
    pd.DataFrame(het_rows).to_csv(OUT / "calibration_site_heterogeneity.csv", index=False)
    pd.concat(params).to_csv(OUT / "calibration_recalibration_params.csv", index=False)
    (OUT / "calibration_nested_summary.json").write_text(json.dumps(
        dict(reproduction_check=checks, reps=args.reps, seed=SEED, iso_clip=ISO_CLIP,
             inner_folds="5 (stratified under pooled, grouped by site under leave-one-site-out)"),
        indent=2))
    print("wrote calibration_nested*.csv, calibration_reliability.csv, calibration_per_site.csv, "
          "calibration_site_heterogeneity.csv, calibration_recalibration_params.csv")


if __name__ == "__main__":
    main()
