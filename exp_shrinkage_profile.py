"""How much shrinkage does Ledoit-Wolf actually choose, and is it enough?

Contribution 3 rests on a mechanism: the Ledoit-Wolf rule picks a shrinkage intensity that is
right for estimating the covariance but too small for the classification that follows.  That is
an assertion about two numbers -- what LW chooses, and what the classifier wants -- so this
script measures the first one instead of asserting it.

Bartz & Muller (NIPS 2014) give the reason to expect under-shrinkage: BOLD time series are
autocorrelated, so the effective number of independent samples is smaller than the number of
volumes, while the LW estimator's variance term does not compensate for that.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Measured Ledoit-Wolf shrinkage profile")
    ap.add_argument("--processed_dir", default="./abide_processed")
    ap.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    ap.add_argument("--roi_table", default="atlas/roi_table.csv")
    ap.add_argument("--study", required=True)
    args = ap.parse_args()

    from run_benchmark import load_dataset
    from sklearn.covariance import LedoitWolf
    from scipy.stats import spearmanr

    roi = pd.read_csv(args.roi_table)
    keep = roi[~roi["excluded"]]["roi_id"].to_numpy() - 1

    ts, y, sites, sids = load_dataset(Path(args.processed_dir), Path(args.ts_root))
    y = np.asarray(y); sites = np.asarray(sites)
    print(f"{len(ts)} participants, {len(keep)} regions", flush=True)

    rows = []
    for i, t in enumerate(ts):
        X = np.asarray(t)[:, keep]
        if X.std(axis=0).min() <= 0:
            continue
        lw = LedoitWolf(assume_centered=False).fit(X)
        rows.append({"sid": sids[i], "site": sites[i], "dx": int(y[i]),
                     "T": int(X.shape[0]), "P": int(X.shape[1]),
                     "lw_shrinkage": float(lw.shrinkage_)})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ts)}", flush=True)

    df = pd.DataFrame(rows)
    t = Path(args.study) / "tables"; t.mkdir(parents=True, exist_ok=True)
    df.to_csv(t / "shrinkage_profile.csv", index=False)

    rho = df["lw_shrinkage"]
    sp = spearmanr(df["T"], rho)
    site_med = df.groupby("site")["lw_shrinkage"].median().sort_values()

    print("\n" + "=" * 74)
    print("LEDOIT-WOLF SHRINKAGE ACTUALLY CHOSEN, PER PARTICIPANT")
    print("=" * 74)
    print(f"  median {rho.median():.4f}   IQR {rho.quantile(.25):.4f}-{rho.quantile(.75):.4f}"
          f"   range {rho.min():.4f}-{rho.max():.4f}")
    print(f"  participants with fewer volumes than regions: "
          f"{int((df['T'] < df['P']).sum())}/{len(df)}")
    print(f"\n  scan length vs shrinkage chosen: Spearman rho = {sp.statistic:+.3f} "
          f"(p = {sp.pvalue:.2e})")
    print("  (negative means shorter scans get more shrinkage, as they should)")
    print(f"\n  site medians span {site_med.min():.4f} ({site_med.index[0]}) to "
          f"{site_med.max():.4f} ({site_med.index[-1]}), a factor of "
          f"{site_med.max()/max(site_med.min(),1e-12):.1f}")
    print("\n  Compare against the extra shrinkage the classification objective selects in")
    print("  Eq. (1a): r = 0.20-0.50.  If LW were already choosing enough, that extra term")
    print("  would buy nothing.")

    (Path(args.study) / "raw").mkdir(parents=True, exist_ok=True)
    (Path(args.study) / "raw" / "shrinkage_profile.json").write_text(json.dumps(
        {"median": float(rho.median()), "q25": float(rho.quantile(.25)),
         "q75": float(rho.quantile(.75)), "min": float(rho.min()), "max": float(rho.max()),
         "spearman_T_vs_shrinkage": float(sp.statistic), "spearman_p": float(sp.pvalue),
         "n_T_below_P": int((df["T"] < df["P"]).sum()), "n": int(len(df)),
         "site_median_min": float(site_med.min()), "site_median_max": float(site_med.max()),
         "site_median_ratio": float(site_med.max() / max(site_med.min(), 1e-12))}, indent=2))
    print(f"\n-> {t / 'shrinkage_profile.csv'}")


if __name__ == "__main__":
    main()
