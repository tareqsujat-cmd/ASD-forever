"""
Do the classifier's eigen-connectome axes track autism symptom severity?

The rank-constrained classifier of `models/lowrank_spectral.py` describes a participant
by a small number of scores, z_ik = u_k^T S_i u_k, each measuring how strongly that
participant expresses one whole-brain component. This script asks whether those scores
carry graded clinical information within the autism group: whether a participant who
scores highly on a component also scores highly on a symptom instrument.

Three instruments are used, chosen because they separate the domains by which autism is
defined: the ADI-R social, verbal-communication and restricted/repetitive subscales.
Overall severity is additionally represented by the ADOS total and calibrated severity
scores and by the SRS total.

Design
------
Components are estimated from the whole sample using the *diagnostic* labels only. The
symptom scores never enter the fitting, so relating them to the component scores
afterwards is a separate question rather than a restatement of the fit. Associations are
computed **within the autism group**, since across groups any component that separates the
groups would correlate with a symptom score simply because controls are not scored.

Age, sex, full-scale IQ and acquisition site are partialled out of both variables before
the association is measured: instruments differ in administration across sites, and both
connectivity and symptom scores vary with age and ability. Spearman correlation is used on
the residuals, and p-values are corrected across all component-by-instrument tests by the
Benjamini-Hochberg procedure.

Component stability across cross-validation folds is reported alongside, so that an
association can be judged against how reproducible the component itself is.

Usage:
    python symptom_axes.py --rank 20 --out results/ablations
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from run_benchmark import load_dataset, _outer_splits
from gpu_tangent import GPUTangentFC
from models.lowrank_spectral import LowRankSpectralClassifier

MISSING = -9999.0
INSTRUMENTS = {
    "ADI_R_SOCIAL_TOTAL_A": "ADI-R social",
    "ADI_R_VERBAL_TOTAL_BV": "ADI-R communication",
    "ADI_RRB_TOTAL_C": "ADI-R repetitive behaviour",
    "ADOS_TOTAL": "ADOS total",
    "ADOS_GOTHAM_SEVERITY": "ADOS severity",
    "SRS_RAW_TOTAL": "SRS total",
}
COVARIATES = ["AGE_AT_SCAN", "SEX", "FIQ"]


def _clean(series: pd.Series) -> np.ndarray:
    # ``to_numpy`` may return a read-only view onto the frame, so copy before masking
    v = np.array(pd.to_numeric(series, errors="coerce"), dtype=float, copy=True)
    v[v <= MISSING + 1] = np.nan
    return v


def load_phenotype_table(pheno_csv: Path, sids) -> pd.DataFrame:
    df = pd.read_csv(pheno_csv)
    df["SUB_ID"] = df["SUB_ID"].astype(str)
    idx = pd.DataFrame({"SUB_ID": [str(s) for s in sids]})
    out = idx.merge(df, on="SUB_ID", how="left")
    return out


def residualise(v: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Remove the linear contribution of the covariates C from v (both may contain NaN)."""
    ok = np.isfinite(v) & np.all(np.isfinite(C), axis=1)
    r = np.full_like(v, np.nan, dtype=float)
    if ok.sum() < 10:
        return r
    A = np.column_stack([np.ones(ok.sum()), C[ok]])
    beta, *_ = np.linalg.lstsq(A, v[ok], rcond=None)
    r[ok] = v[ok] - A @ beta
    return r


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    ps = p[ok]
    order = np.argsort(ps)
    ranked = ps[order]
    m = len(ranked)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def component_stability(covs, y, splits, rank):
    """Absolute cosine similarity between fold-wise components and the full-sample ones."""
    from sklearn.preprocessing import StandardScaler
    gt = GPUTangentFC(device="cpu", max_iter=10)
    gt.fit(covs); X = gt.transform(covs)
    sc = StandardScaler().fit(X)
    full = LowRankSpectralClassifier(rank=rank).fit(sc.transform(X), y)
    sims = []
    for tr, _ in splits:
        gt.fit(covs[tr])
        Xtr = gt.transform(covs[tr])
        s2 = StandardScaler().fit(Xtr)
        m = LowRankSpectralClassifier(rank=rank).fit(s2.transform(Xtr), y[tr])
        M = np.abs(full.components_.T @ m.components_)      # (r, r) alignment
        sims.append(M.max(axis=1))                          # best match per component
    return full, np.mean(np.stack(sims), axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Eigen-connectome axes vs symptom severity")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_dir", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--pheno_csv",
                    default="./abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--rank", type=int, default=20)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/ablations")
    args = ap.parse_args()

    from scipy.stats import spearmanr
    from sklearn.preprocessing import StandardScaler

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_dir))
    gt = GPUTangentFC(device="cpu", max_iter=10)
    covs = gt.compute_covariances(ts)
    splits, _ = _outer_splits("pooled", y, sites, args.n_folds, args.seed)

    print(f"fitting rank-{args.rank} components and checking their stability ...", flush=True)
    model, stability = component_stability(covs, y, splits, args.rank)

    gt.fit(covs); X = StandardScaler().fit_transform(gt.transform(covs))
    Z = model.transform(X)                                   # (n, r) component scores

    pheno = load_phenotype_table(Path(args.pheno_csv), sids)
    cov_mat = np.column_stack([_clean(pheno[c]) for c in COVARIATES])
    site_codes = pd.get_dummies(pd.Series(sites)).to_numpy(float)[:, 1:]   # site as dummies
    C = np.column_stack([cov_mat, site_codes])

    asd = y == 1
    print(f"autism group: {asd.sum()} participants\n")

    rows = []
    for col, label in INSTRUMENTS.items():
        if col not in pheno.columns:
            continue
        s = _clean(pheno[col])
        s_res = residualise(np.where(asd, s, np.nan), C)
        n_avail = int(np.isfinite(s_res).sum())
        if n_avail < 30:
            print(f"  {label}: only {n_avail} scored participants, skipped")
            continue
        for k in range(Z.shape[1]):
            z_res = residualise(np.where(asd, Z[:, k], np.nan), C)
            ok = np.isfinite(z_res) & np.isfinite(s_res)
            if ok.sum() < 30:
                continue
            rho, p = spearmanr(z_res[ok], s_res[ok])
            rows.append({"instrument": label, "column": col, "component": k + 1,
                         "n": int(ok.sum()), "rho": float(rho), "p": float(p),
                         "stability": float(stability[k])})

    df = pd.DataFrame(rows)
    if df.empty:
        print("no association could be computed")
        return
    df["q_fdr"] = bh_fdr(df["p"].to_numpy())
    df = df.sort_values("p")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "symptom_axes.csv", index=False)
    (out / "symptom_axes.json").write_text(json.dumps({
        "rank": args.rank,
        "n_tests": int(len(df)),
        "n_significant_fdr05": int((df["q_fdr"] < 0.05).sum()),
        "component_stability": stability.tolist(),
        "top": df.head(15).to_dict(orient="records"),
    }, indent=2))

    print("=== strongest associations (within autism, covariates removed) ===")
    print(f"{'instrument':<28}{'comp':>5}{'n':>6}{'rho':>8}{'p':>10}{'q(FDR)':>9}{'stab':>7}")
    for _, r in df.head(12).iterrows():
        print(f"{r['instrument']:<28}{r['component']:>5.0f}{r['n']:>6.0f}"
              f"{r['rho']:>8.3f}{r['p']:>10.4f}{r['q_fdr']:>9.3f}{r['stability']:>7.2f}")
    n_sig = int((df["q_fdr"] < 0.05).sum())
    print(f"\n{n_sig} of {len(df)} tests significant after FDR correction")
    if n_sig == 0:
        print("No association survives correction: the components separate the groups "
              "without tracking graded severity in this sample.")
    print("report ->", out / "symptom_axes.csv")


if __name__ == "__main__":
    main()
