"""
Which brain systems differ in autism, at a resolution the sample can support.

Individual connections cannot be identified reliably from a cohort of this size: the fifty
strongest reproduce across random halves of the data at a correlation near zero. Grouping
the regions into the seven cortical systems of Yeo et al. (2011) reduces nineteen thousand
comparisons to a few dozen, each averaging over hundreds of connections, which is stable.

Every connection is first adjusted for acquisition site, head motion, age and sex, so a
system-level difference cannot be produced by autistic participants being scanned
elsewhere, moving more, or being younger. Significance is assessed by permuting the
diagnosis within each site, which preserves the site structure that the adjustment cannot
fully remove.

Usage:
    python exp_network_contrast.py --study results/<study>
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

MISSING = -9999.0


def main() -> None:
    ap = argparse.ArgumentParser(description="System-level connectivity contrast")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--roi_table", default="atlas/roi_table.csv")
    ap.add_argument("--pheno", default="abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv")
    ap.add_argument("--match_motion", action="store_true",
                    help="pair each autistic participant with a control at the same site and "
                         "nearly the same head motion, and analyse only those pairs")
    ap.add_argument("--caliper", type=float, default=0.01,
                    help="largest head-motion difference allowed within a matched pair")
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--study", required=True)
    args = ap.parse_args()

    from run_benchmark import load_dataset
    from scipy.stats import false_discovery_control

    roi = pd.read_csv(args.roi_table)
    keep = roi.loc[~roi["excluded"], "roi_id"].to_numpy() - 1        # 0-based columns
    system = roi.set_index("roi_id")["system"]
    print(f"{len(keep)} regions retained of {len(roi)}")

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    y = np.asarray(y); sites = np.asarray(sites)

    # participants whose retained regions are all usable
    ok = np.array([bool((np.asarray(t)[:, keep].std(axis=0) > 0).all()) for t in ts])
    print(f"{int(ok.sum())} of {len(y)} participants have complete signal in those regions")
    ts = [ts[i] for i in np.where(ok)[0]]
    y, sites, sids = y[ok], sites[ok], [sids[i] for i in np.where(ok)[0]]

    # connectivity, Fisher transformed, upper triangle only
    iu = np.triu_indices(len(keep), k=1)
    X = np.stack([np.arctanh(np.clip(np.corrcoef(np.asarray(t)[:, keep].T), -0.999, 0.999))[iu]
                  for t in ts])
    print(f"connectivity matrix {X.shape}")

    # nuisance variables
    ph = pd.read_csv(args.pheno)
    ph["key"] = ph["SUB_ID"].astype(str).str.lstrip("0")
    ph = ph.set_index("key")
    keys = [str(s).split("_")[-1].lstrip("0") for s in sids]
    sub = ph.reindex(keys)

    def clean(col):
        v = pd.to_numeric(sub[col], errors="coerce")
        return v.mask(v <= MISSING + 1).fillna(v[v > MISSING + 1].median()).to_numpy()

    age, sex, fd = clean("AGE_AT_SCAN"), clean("SEX"), clean("func_mean_fd")

    if args.match_motion:
        # Head motion is higher in autism and lowers measured connectivity, so a reduction
        # found after regressing motion out can still be a motion effect.  Pairing each
        # autistic participant with a control scanned at the same site and moving almost
        # exactly as much removes the difference by design rather than by adjustment.
        from scipy.stats import mannwhitneyu
        chosen = []
        for st in np.unique(sites):
            at = np.where((sites == st) & (y == 1))[0]
            ct = list(np.where((sites == st) & (y == 0))[0])
            for i in sorted(at, key=lambda k: fd[k]):
                if not ct:
                    break
                j = min(ct, key=lambda k: abs(fd[k] - fd[i]))
                if abs(fd[j] - fd[i]) <= args.caliper:
                    chosen += [i, j]; ct.remove(j)
        sel = np.array(sorted(chosen))
        if len(sel) < 100:
            raise SystemExit(f"only {len(sel)//2} pairs within caliper {args.caliper}")
        before = mannwhitneyu(fd[y == 1], fd[y == 0]).pvalue
        X, y, sites = X[sel], y[sel], sites[sel]
        age, sex, fd = age[sel], sex[sel], fd[sel]
        after = mannwhitneyu(fd[y == 1], fd[y == 0]).pvalue
        print(f"motion-matched: {len(sel)//2} pairs, {len(sel)} participants "
              f"({int(y.sum())} autistic)")
        print(f"  head-motion difference between groups: p={before:.2e} before matching, "
              f"p={after:.3f} after")
        print(f"  median motion now {np.median(fd[y==1]):.4f} vs {np.median(fd[y==0]):.4f}",
              flush=True)
    site_d = pd.get_dummies(pd.Series(sites), drop_first=True).to_numpy(dtype=float)
    C = np.column_stack([np.ones(len(y)), age, sex, fd, site_d])
    print(f"adjusting for age, sex, head motion and {site_d.shape[1]+1} sites")

    # residualise every connection on the nuisance variables
    beta, *_ = np.linalg.lstsq(C, X, rcond=None)
    R = X - C @ beta

    def block_means(labels):
        """Mean standardised group difference within each pair of systems."""
        a, b = R[labels == 1], R[labels == 0]
        sd = R.std(axis=0, ddof=1)
        d = (a.mean(axis=0) - b.mean(axis=0)) / np.where(sd > 0, sd, np.nan)
        sysname = system.reindex(roi.loc[~roi["excluded"], "roi_id"]).to_numpy()
        si, sj = sysname[iu[0]], sysname[iu[1]]
        out = {}
        for s1 in np.unique(sysname):
            for s2 in np.unique(sysname):
                if s1 > s2:
                    continue
                m = ((si == s1) & (sj == s2)) | ((si == s2) & (sj == s1))
                if m.sum() >= 20:
                    out[(s1, s2)] = (float(np.nanmean(d[m])), int(m.sum()))
        return out

    obs = block_means(y)
    print(f"{len(obs)} system pairs with at least 20 connections\n")

    # permute diagnosis within site, so the site structure is preserved
    rng = np.random.RandomState(args.seed)
    null = {k: [] for k in obs}
    for i in range(args.n_perm):
        perm = y.copy()
        for s in np.unique(sites):
            m = sites == s
            perm[m] = rng.permutation(perm[m])
        for k, (v, _) in block_means(perm).items():
            null[k].append(v)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{args.n_perm} permutations", flush=True)

    rows = []
    for k, (val, n_edges) in obs.items():
        nd = np.asarray(null[k])
        p = (1 + int((np.abs(nd) >= abs(val)).sum())) / (1 + len(nd))
        rows.append({"system_a": k[0], "system_b": k[1], "n_edges": n_edges,
                     "effect": val, "null_sd": float(nd.std()),
                     "z": float(val / nd.std()) if nd.std() > 0 else np.nan, "p": p})
    df = pd.DataFrame(rows)
    df["q_fdr"] = false_discovery_control(df["p"].to_numpy())
    df["direction"] = np.where(df["effect"] > 0, "higher in autism", "lower in autism")
    df = df.sort_values("p")

    study = Path(args.study)
    t = study / "tables"; t.mkdir(parents=True, exist_ok=True)
    suffix = "_motion_matched" if args.match_motion else ""
    df.to_csv(t / f"network_contrast{suffix}.csv", index=False)

    print(f"\n{'system pair':56s}{'edges':>7s}{'effect':>9s}{'z':>7s}{'q':>8s}")
    print("-" * 88)
    for _, r in df.head(14).iterrows():
        pair = f"{r.system_a} <-> {r.system_b}"
        print(f"{pair:56s}{int(r.n_edges):>7d}{r.effect:>+9.4f}{r.z:>7.2f}{r.q_fdr:>8.4f}")
    sig = df[df.q_fdr < 0.05]
    print(f"\n{len(sig)} of {len(df)} system pairs survive correction at q<0.05")
    (study / "raw").mkdir(parents=True, exist_ok=True)
    (study / "raw" / f"network_contrast{suffix}.json").write_text(json.dumps(
        {"n_participants": int(len(y)), "n_asd": int(y.sum()), "n_regions": len(keep),
         "n_permutations": args.n_perm, "n_significant": int(len(sig))}, indent=2))
    print("\n->", t / f"network_contrast{suffix}.csv")


if __name__ == "__main__":
    main()
