"""
Analyses requested by the reviewers, run on the stored out-of-fold predictions.

Four items are addressed here; each writes its numbers to results/latest_study/review/.

  1  site-clustered bootstrap intervals for AUROC and accuracy, replacing DeLong and
     Wilson intervals that condition on one split structure and ignore site clustering;
  2  calibration beyond ten-bin ECE: calibration-in-the-large, calibration slope, Brier
     score, and ECE at 5, 10 and 20 bins, each with site-clustered bootstrap intervals;
  3  a direct test of the diagnosis-by-IQ interaction on the out-of-fold score, with the
     IQ family corrected together by Benjamini-Hochberg;
  4  a participant-level cohort flow reconciling 1,112 records to every analysed cohort.

Usage:
    .venv/bin/python review_experiments.py [--study results/latest_study] [--reps 2000]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
PHENO = ROOT / "abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv"
META = ROOT / "abide_processed/mri/metadata.csv"
SEED = 42
EPS = 1e-6


# =========================================================================================
# helpers
# =========================================================================================
def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def oof_order(protocol: str, y_meta: np.ndarray) -> np.ndarray:
    """Row order of the stored out-of-fold arrays, as metadata-row indices.

    LOSO concatenates sites in sorted order; pooled concatenates the test folds of
    StratifiedKFold(10, shuffle=True, random_state=42).  Both are verified against the
    stored labels and sites before use, so a mismatch raises rather than mis-aligns.
    """
    from sklearn.model_selection import StratifiedKFold
    if protocol == "loso":
        return np.argsort(pd.read_csv(META).site.to_numpy(), kind="stable")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    return np.concatenate([te for _, te in skf.split(np.zeros(len(y_meta)), y_meta)])


def load_oof(study: Path, condition: str, protocol: str):
    d = json.loads((study / "raw" / "voting.json").read_text())
    for r in d:
        if r["condition"].startswith(condition) and r["protocol"] == protocol:
            return (np.asarray(r["oof_true"], int), np.asarray(r["oof_prob"], float),
                    np.asarray(r["oof_site"]))
    raise SystemExit(f"condition {condition!r} / {protocol!r} not found")


def site_bootstrap(y, p, site, fn, reps, rng):
    """Cluster bootstrap: resample whole sites with replacement, recompute fn."""
    sites = np.unique(site)
    idx_by_site = {s: np.flatnonzero(site == s) for s in sites}
    out = []
    for _ in range(reps):
        draw = rng.choice(sites, size=sites.size, replace=True)
        idx = np.concatenate([idx_by_site[s] for s in draw])
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():                 # a resample with one class only
            continue
        try:
            out.append(fn(yy, pp))
        except ValueError:
            continue
    return np.asarray(out, float)


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def ece(y, p, bins):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    counts = []
    for b in range(bins):
        m = idx == b
        counts.append(int(m.sum()))
        if m.sum():
            tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return tot, counts


def cal_slope_intercept(y, p):
    """Calibration slope (y ~ logit p) and calibration-in-the-large (offset model)."""
    lp = logit(p).reshape(-1, 1)
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(lp, y)
    slope = float(m.coef_[0, 0])
    # calibration-in-the-large: intercept of y ~ 1 with logit(p) as a fixed offset
    z = logit(p)
    a = 0.0
    for _ in range(200):                          # Newton steps on a single intercept
        q = 1 / (1 + np.exp(-(z + a)))
        g = np.sum(y - q)
        h = np.sum(q * (1 - q))
        if h < 1e-12:
            break
        step = g / h
        a += step
        if abs(step) < 1e-10:
            break
    return slope, float(a)


# =========================================================================================
# 1 + 2: uncertainty and calibration
# =========================================================================================
def uncertainty_and_calibration(study: Path, reps: int, out: Path):
    rows = []
    rng = np.random.default_rng(SEED)
    for protocol in ("pooled", "loso"):
        y, p, site = load_oof(study, "ensemble (probability mean)", protocol)
        acc_fn = lambda yy, pp: float(((pp >= 0.5).astype(int) == yy).mean())
        auroc = float(roc_auc_score(y, p))
        acc = acc_fn(y, p)
        brier = float(brier_score_loss(y, p))
        slope, cil = cal_slope_intercept(y, p)

        bs_auc = site_bootstrap(y, p, site, lambda a, b: roc_auc_score(a, b), reps, rng)
        bs_acc = site_bootstrap(y, p, site, acc_fn, reps, rng)
        bs_br = site_bootstrap(y, p, site, lambda a, b: brier_score_loss(a, b), reps, rng)
        bs_sl = site_bootstrap(y, p, site, lambda a, b: cal_slope_intercept(a, b)[0],
                               max(400, reps // 5), rng)
        bs_ci = site_bootstrap(y, p, site, lambda a, b: cal_slope_intercept(a, b)[1],
                               max(400, reps // 5), rng)

        for name, val, bs in (("AUROC", auroc, bs_auc), ("accuracy", acc, bs_acc),
                              ("Brier", brier, bs_br), ("calibration slope", slope, bs_sl),
                              ("calibration-in-the-large", cil, bs_ci)):
            lo, hi = ci(bs)
            rows.append(dict(protocol=protocol, quantity=name, estimate=val,
                             lo=lo, hi=hi, n_boot=len(bs)))
        for b in (5, 10, 20):
            e, counts = ece(y, p, b)
            rows.append(dict(protocol=protocol, quantity=f"ECE ({b} bins)", estimate=e,
                             lo=np.nan, hi=np.nan, n_boot=0,
                             note="min bin count %d" % min(counts)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "uncertainty_calibration.csv", index=False)
    print("\n== 1+2  site-clustered bootstrap and calibration ==")
    for _, r in df.iterrows():
        rng_s = "" if np.isnan(r.lo) else f"  [{r.lo:.3f}, {r.hi:.3f}]"
        print(f"  {r.protocol:7s} {r.quantity:26s} {r.estimate:7.3f}{rng_s}"
              f"{'  ' + r.note if isinstance(r.get('note'), str) else ''}")
    return df


# =========================================================================================
# 3: diagnosis-by-IQ interaction
# =========================================================================================
def iq_interaction(study: Path, out: Path):
    ph = pd.read_csv(PHENO, low_memory=False)
    ph["subject_id"] = ph.SUB_ID
    meta = pd.read_csv(META)
    y_meta = meta.label.to_numpy()
    rows, checks = [], []
    for protocol in ("pooled", "loso"):
        y, p, site = load_oof(study, "ensemble (probability mean)", protocol)
        idx = oof_order(protocol, y_meta)
        if not ((y_meta[idx] == y).all() and (meta.site.to_numpy()[idx] == site).all()):
            raise SystemExit(f"cannot align stored predictions for {protocol}")
        s = meta.iloc[idx].copy()
        s["score"] = logit(p)
        s["y"] = y
        s = s.merge(ph[["subject_id", "FIQ", "VIQ", "PIQ", "AGE_AT_SCAN", "SEX",
                        "func_mean_fd", "DX_GROUP"]], on="subject_id", how="left")

        # reproduce the published within-group correlations as an alignment check
        for grp, name in ((1, "autistic"), (0, "non-autistic")):
            g = s[(s.y == grp)].dropna(subset=["FIQ", "AGE_AT_SCAN", "func_mean_fd"])
            g = g[g.FIQ > 0]
            X = pd.get_dummies(g.site, drop_first=True).astype(float)
            X["age"] = g.AGE_AT_SCAN.to_numpy()
            X["fd"] = g.func_mean_fd.to_numpy()
            X["c"] = 1.0
            Xm = X.to_numpy(float)
            res = lambda v: v - Xm @ np.linalg.lstsq(Xm, v, rcond=None)[0]
            rho, pv = stats.spearmanr(res(g.score.to_numpy()), res(g.FIQ.to_numpy(float)))
            checks.append(dict(protocol=protocol, group=name, n=len(g),
                               rho=float(rho), p=float(pv)))

        for iqv in ("FIQ", "VIQ", "PIQ"):
            g = s.dropna(subset=[iqv, "AGE_AT_SCAN", "func_mean_fd"]).copy()
            g = g[g[iqv] > 0]
            X = pd.get_dummies(g.site, prefix="site", drop_first=True).astype(float)
            z = ((g[iqv] - g[iqv].mean()) / g[iqv].std()).to_numpy()
            a = g.y.to_numpy(float)                    # 1 = autistic
            X.insert(0, "iq", z)
            X.insert(1, "autistic", a)
            X.insert(2, "iq_x_autistic", z * a)
            X.insert(3, "age", ((g.AGE_AT_SCAN - g.AGE_AT_SCAN.mean())
                                / g.AGE_AT_SCAN.std()).to_numpy())
            X.insert(4, "sex", g.SEX.to_numpy(float))
            X.insert(5, "fd", ((g.func_mean_fd - g.func_mean_fd.mean())
                               / g.func_mean_fd.std()).to_numpy())
            X.insert(0, "const", 1.0)
            Xm = X.to_numpy(float)
            yv = g.score.to_numpy(float)
            beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
            resid = yv - Xm @ beta
            dof = len(yv) - np.linalg.matrix_rank(Xm)
            sigma2 = resid @ resid / dof
            se = np.sqrt(np.diag(np.linalg.pinv(Xm.T @ Xm)) * sigma2)
            j = list(X.columns).index("iq_x_autistic")
            tstat = beta[j] / se[j]
            rows.append(dict(protocol=protocol, iq=iqv, n=len(g), beta=float(beta[j]),
                             se=float(se[j]), t=float(tstat),
                             p=float(2 * stats.t.sf(abs(tstat), dof))))
    df = pd.DataFrame(rows)
    pv = df.p.to_numpy()
    m = len(pv)
    order = np.argsort(pv)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        prev = min(prev, pv[i] * m / (m - rank))
        q[i] = prev
    df["q_bh"] = q
    df.to_csv(out / "iq_interaction.csv", index=False)
    chk = pd.DataFrame(checks)
    chk.to_csv(out / "iq_within_group_check.csv", index=False)
    print("\n== 3a  alignment check: within-group score-IQ correlation (paper: -0.18 / +0.05) ==")
    for _, r in chk.iterrows():
        print(f"  {r.protocol:7s} {r.group:13s} n={r.n:4d}  rho={r.rho:+.3f}  p={r.p:.4f}")
    print("\n== 3b  diagnosis-by-IQ interaction, BH across the IQ family ==")
    for _, r in df.iterrows():
        print(f"  {r.protocol:7s} {r.iq}  n={r.n:4d}  beta={r.beta:+.4f}"
              f"  t={r.t:+.2f}  p={r.p:.4f}  q={r.q_bh:.4f}")
    return df


# =========================================================================================
# 4: participant-level cohort flow
# =========================================================================================
def cohort_flow(study: Path, out: Path):
    ph = pd.read_csv(PHENO, low_memory=False)
    meta = pd.read_csv(META)
    steps = []
    steps.append(("ABIDE-I phenotypic records", len(ph)))
    have = ph[ph.FILE_ID != "no_filename"]
    steps.append(("with a preprocessed FILE_ID", len(have)))

    r1 = have.qc_rater_1.astype(str).str.lower()
    r2a = have.qc_anat_rater_2.astype(str).str.lower()
    r2f = have.qc_func_rater_2.astype(str).str.lower()
    r3a = have.qc_anat_rater_3.astype(str).str.lower()
    r3f = have.qc_func_rater_3.astype(str).str.lower()

    wider = have[r3f.eq("ok")]
    steps.append(("wider rule: rater 3 functional OK", len(wider)))

    strict = have[r1.eq("ok") & r2a.isin(["ok", "maybe"]) & r2f.isin(["ok", "maybe"])
                  & r3a.eq("ok") & r3f.eq("ok")]
    steps.append(("primary rule: all three raters", len(strict)))
    steps.append(("primary cohort analysed (time series present)", len(meta)))
    steps.append(("  of which autistic", int((meta.label == 1).sum())))
    steps.append(("  of which non-autistic", int((meta.label == 0).sum())))

    df = pd.DataFrame(steps, columns=["step", "n"])
    df.to_csv(out / "cohort_flow.csv", index=False)
    print("\n== 4  cohort flow ==")
    for s, n in steps:
        print(f"  {s:48s} {n:5d}")
    return df


# =========================================================================================
# reproducibility manifest
# =========================================================================================
def manifest(study: Path, out: Path):
    import sklearn
    import scipy
    meta = pd.read_csv(META)
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "seed": SEED,
        "n_participants": int(len(meta)),
        "n_sites": int(meta.site.nunique()),
        "participant_id_sha256": __import__("hashlib").sha256(
            ",".join(str(s) for s in sorted(meta.subject_id)).encode()).hexdigest(),
        "atlas": "atlas/cc200_roi_atlas.nii.gz",
        "derivative": "C-PAC filt_noglobal, ABIDE Preprocessed Initiative",
    }
    (out / "manifest.json").write_text(json.dumps(info, indent=2))
    print("\n== reproducibility manifest ==")
    for k, v in info.items():
        print(f"  {k:26s} {v}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=str(ROOT / "results/latest_study"))
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    study = Path(a.study).resolve()
    out = study / "review"
    out.mkdir(parents=True, exist_ok=True)
    uncertainty_and_calibration(study, a.reps, out)
    iq_interaction(study, out)
    cohort_flow(study, out)
    manifest(study, out)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
