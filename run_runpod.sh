#!/usr/bin/env bash
# =============================================================================
# ASD-forever — RunPod training + evaluation + figure generation (CUDA)
#
#   bash run_runpod.sh                 # full run, default hyperparameters
#   N_FOLDS=10 MAX_EPOCHS=100 bash run_runpod.sh   # override via env vars
#
# Runs the end-to-end pipeline on the real ABIDE-I FC data:
#   training (pooled stratified 10-fold) -> evaluation (bootstrap CI, per-site,
#   calibration) -> ablation -> HPO -> explainability -> paper figures -> report.
#
# Everything is written to a fresh, self-contained results/run_N/ directory
# (run_manifest.json records seed, device, git commit, versions).
# All figures are rendered with the scienceplots (science+ieee) style.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate

DEVICE="${DEVICE:-cuda}"
N_FOLDS="${N_FOLDS:-10}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
N_HPO_TRIALS="${N_HPO_TRIALS:-20}"
SEED="${SEED:-42}"

echo "==> Training + evaluation + figures  (device=${DEVICE}, folds=${N_FOLDS}, epochs=${MAX_EPOCHS})"
python run_experiment.py \
  --real_data \
  --mri_dir ./abide_processed/mri \
  --gen_dir ./abide_processed/gen \
  --device "${DEVICE}" \
  --n_folds "${N_FOLDS}" \
  --max_epochs "${MAX_EPOCHS}" \
  --n_hpo_trials "${N_HPO_TRIALS}" \
  --seed "${SEED}"

RUN_DIR="$(ls -dt results/run_* 2>/dev/null | head -1 || true)"
echo ""
echo "Done.  Results in: ${RUN_DIR:-results/}"
echo "  - metrics:      ${RUN_DIR}/evaluation/evaluation_report.json"
echo "  - figures:      ${RUN_DIR}/paper_figures/   (scienceplots-styled)"
echo "  - HTML report:  ${RUN_DIR}/experiment_report.html"
echo "  - manifest:     ${RUN_DIR}/run_manifest.json"
