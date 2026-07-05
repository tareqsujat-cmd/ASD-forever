"""
Download ABIDE I preprocessed ROI timeseries via nilearn.

Fetches the CPAC pipeline, no global signal regression, bandpass filtered,
using the Harvard-Oxford atlas (111 ROIs) parcellation.

Run once:
    python data/download_abide.py --data_dir ./abide_raw

This downloads ~3 GB.  Expected time: 20–60 min depending on connection.
The output is a folder ./abide_raw/ABIDE_pcp/ with one .1D file per subject.
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _fetch_with_retry(
    data_dir: str,
    pipeline: str,
    atlas: str,
    max_retries: int = 20,
    base_wait: float = 10.0,
) -> object:
    """
    Call nilearn.fetch_abide_pcp with automatic retry on transient HTTP errors.

    The S3 bucket hosting ABIDE data returns 503 Service Unavailable under load.
    nilearn caches every successfully downloaded file, so re-running skips all
    already-fetched subjects and retries only the failed one.

    Back-off: 10s, 20s, 40s, … up to 5 min, then flat 5 min intervals.
    """
    import time
    from nilearn import datasets

    wait = base_wait
    for attempt in range(1, max_retries + 1):
        try:
            abide = datasets.fetch_abide_pcp(
                data_dir              = data_dir,
                pipeline              = pipeline,
                band_pass_filtering   = True,
                global_signal_regression = False,
                derivatives           = [atlas],
                verbose               = 1,
            )
            return abide
        except Exception as exc:
            is_last = (attempt == max_retries)
            if is_last:
                raise
            logger.warning(
                "Attempt %d/%d failed (%s). Retrying in %.0f s…",
                attempt, max_retries, exc, wait,
            )
            time.sleep(wait)
            wait = min(wait * 2, 300)   # cap at 5-minute intervals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="./abide_raw",
                   help="Root folder where ABIDE data will be saved")
    p.add_argument("--pipeline",    default="cpac",
                   choices=["cpac", "dparsf"],
                   help="Preprocessing pipeline (cpac recommended)")
    p.add_argument("--atlas",       default="rois_cc200",
                   choices=["rois_ho", "rois_cc200", "rois_aal"],
                   help="ROI parcellation atlas (cc200=200 ROIs recommended, ho=111 ROIs)")
    p.add_argument("--max_retries", type=int, default=20,
                   help="Max retry attempts on transient 503 errors")
    args = p.parse_args()

    logger.info("Fetching ABIDE I via nilearn (this may take 20–60 min)…")
    logger.info("  Pipeline    : %s", args.pipeline)
    logger.info("  Atlas       : %s", args.atlas)
    logger.info("  Output      : %s", args.data_dir)
    logger.info("  Max retries : %d", args.max_retries)

    abide = _fetch_with_retry(
        data_dir    = args.data_dir,
        pipeline    = args.pipeline,
        atlas       = args.atlas,
        max_retries = args.max_retries,
    )

    import pandas as pd
    pheno = abide.phenotypic
    if not isinstance(pheno, pd.DataFrame):
        pheno = pd.DataFrame(pheno)

    logger.info("Download complete.")
    logger.info("  Subjects  : %d", len(pheno))
    logger.info("  ASD       : %d", (pheno["DX_GROUP"] == 1).sum())
    logger.info("  TC        : %d", (pheno["DX_GROUP"] == 2).sum())
    logger.info("  Sites     : %s", pheno["SITE_ID"].nunique())
    logger.info("Phenotypic columns: %s", list(pheno.columns))


if __name__ == "__main__":
    main()
