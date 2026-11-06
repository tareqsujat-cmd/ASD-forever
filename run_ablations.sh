#!/usr/bin/env bash
# Full ablation suite for the ABIDE-I paper.
# Every experiment writes JSON + CSV so figures can be regenerated without re-running.
#
#   ./run_ablations.sh            # everything
#   ./run_ablations.sh novelties  # only the three contributions (N1-N3)
#   ./run_ablations.sh leakage    # only the leakage audit
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}
export KMP_DUPLICATE_LIB_OK=TRUE
FOLDS=${FOLDS:-10}
MODE=${1:-all}

echo "=== ablation suite (mode=$MODE, folds=$FOLDS) ==="

if [ "$MODE" = "leakage" ] || [ "$MODE" = "all" ]; then
  echo "--- Experiment 7: leakage audit ---"
  $PY leakage_audit.py --n_folds "$FOLDS" --out results/ablations
fi

if [ "$MODE" = "novelties" ] || [ "$MODE" = "all" ]; then
  echo "--- Experiments 1-3: representation / classifier / rank (N1, N2) ---"
  $PY run_ablations.py --experiments representation classifier rank \
      --n_folds "$FOLDS" --run_name ablations_novelty
fi

if [ "$MODE" = "all" ]; then
  echo "--- Experiments 4-6, 8-9: design, protocol, statistics (N3) ---"
  $PY run_ablations.py --experiments atlas fusion harmonisation protocol statistics \
      --n_folds "$FOLDS" --run_name ablations_design
fi

# figures for the most recent ablation run
LATEST=$(ls -dt results/*_ablations_* results/*_ablations 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  echo "--- figures ---"
  $PY viz_ablations.py --dir "$LATEST/ablations" || true
  echo "results: $LATEST/ablations"
fi
echo "=== done ==="
