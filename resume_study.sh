#!/usr/bin/env bash
# Resume the study after the completed stages, then finalise.
# The classifier experiment runs with single-threaded OpenMP, because it is the only
# stage that loads torch and the gradient-boosting libraries together.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
export KMP_DUPLICATE_LIB_OK=TRUE
FOLDS=${FOLDS:-10}

STUDY=${1:-$(ls -dt results/*_full_study 2>/dev/null | head -1)}
ABL="$STUDY/ablations"; mkdir -p "$ABL"
echo "=== resuming $STUDY (folds=$FOLDS) ==="
echo "already present: $(ls "$ABL"/*.csv 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"

collect () {                      # copy a stage's artefacts into the study directory
  local d; d=$(ls -dt results/*_"$1" 2>/dev/null | head -1)
  [ -n "$d" ] && cp -f "$d"/ablations/*.csv "$d"/ablations/*.json "$ABL"/ 2>/dev/null || true
  rm -rf "$d" 2>/dev/null || true
}

echo "--- classifier (single-threaded OpenMP) ---"
OMP_NUM_THREADS=1 $PY run_ablations.py --experiments classifier \
    --n_folds "$FOLDS" --out_dir results --run_name tmp_clf 2>&1 \
    | grep -avE "resource_tracker" || true
collect tmp_clf

echo "--- rank sweep ---"
OMP_NUM_THREADS=6 $PY run_ablations.py --experiments rank \
    --n_folds "$FOLDS" --out_dir results --run_name tmp_rank 2>&1 \
    | grep -avE "resource_tracker" || true
collect tmp_rank

echo "--- atlas / fusion / harmonisation / protocol / statistics ---"
OMP_NUM_THREADS=6 $PY run_ablations.py \
    --experiments atlas fusion harmonisation protocol statistics \
    --n_folds "$FOLDS" --out_dir results --run_name tmp_design 2>&1 \
    | grep -avE "resource_tracker" || true
collect tmp_design

echo "--- merging condition tables ---"
$PY - "$ABL" <<'PY'
import sys, glob, os
import pandas as pd
abl = sys.argv[1]
parts = []
for f in glob.glob(os.path.join(abl, "*.csv")):
    if os.path.basename(f) in ("ablations_all.csv", "leakage_audit.csv", "qc_cohort.csv"):
        continue
    try:
        head = pd.read_csv(f, nrows=0)
    except Exception:
        continue
    if "experiment" in head.columns:
        parts.append(pd.read_csv(f))
if parts:
    df = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["experiment", "condition"], keep="last")
    df.to_csv(os.path.join(abl, "ablations_all.csv"), index=False)
    print(f"merged {len(df)} conditions from {len(parts)} experiments")
PY

echo "--- finalising ---"
./finalize_study.sh "$STUDY"
