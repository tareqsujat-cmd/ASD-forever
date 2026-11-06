#!/usr/bin/env bash
# Complete a finished study directory: run the remaining analyses, build the
# conference-style figures, and organise everything into a navigable layout.
#
#   ./finalize_study.sh                      # newest *_full_study directory
#   ./finalize_study.sh results/<run>        # a specific one
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export KMP_DUPLICATE_LIB_OK=TRUE

STUDY=${1:-$(ls -dt results/*_full_study 2>/dev/null | head -1)}
[ -z "$STUDY" ] && { echo "no study directory found"; exit 1; }
ABL="$STUDY/ablations"; [ -d "$ABL" ] || ABL="$STUDY/tables"; mkdir -p "$ABL"
echo "=== finalising $STUDY ==="

echo "--- evaluation-protocol audit (re-run with corrected variant nesting) ---"
$PY leakage_audit.py --n_folds 10 --out "$ABL" 2>&1 | grep -avE "resource_tracker" || true

echo "--- spectral structure (supports the compression result) ---"
$PY spectrum_analysis.py --n_folds 5 --out "$ABL" 2>&1 | grep -avE "resource_tracker" || true

echo "--- component scores against symptom severity ---"
$PY symptom_axes.py --rank 20 --n_folds 5 --out "$ABL" 2>&1 | grep -avE "resource_tracker" || true

echo "--- quality-control cohort comparison (871 vs 1035) ---"
$PY qc_cohort.py --compare --n_folds 10 --out "$ABL" 2>&1 | grep -avE "resource_tracker" || true

echo "--- figures and report ---"
$PY viz_study.py --dir "$STUDY" 2>&1 | tail -20 || true

ln -sfn "$(basename "$STUDY")" results/latest_study
echo "=== done: $STUDY ==="
echo "open $STUDY/RESULTS.md"
find "$STUDY" -maxdepth 1 | sort
