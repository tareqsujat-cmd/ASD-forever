#!/usr/bin/env bash
# =============================================================================
# ASD-forever — one-command multi-atlas SSL SOTA training push (CUDA/RunPod).
#
#   bash run_sota.sh
#   ATLASES="cc200 aal ho" SSL_EPOCHS=100 EPOCHS=200 SEEDS=3 bash run_sota.sh
#
# Runs train_sota.py: per-fold tangent-FC (fit in-fold) -> SSL masked-connectome
# pretraining -> fine-tuned connectome transformer -> multi-atlas x seed ensemble,
# under pooled + LOSO nested CV.  Leakage-free.  Results -> results/run_N/sota/.
#
# Requires the extra atlas time series (run: ATLASES="cc200 aal ho" bash setup_runpod.sh).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .venv/bin/activate

ATLASES="${ATLASES:-cc200 aal ho}"
PROTOCOL="${PROTOCOL:-both}"
N_FOLDS="${N_FOLDS:-10}"
SSL_EPOCHS="${SSL_EPOCHS:-100}"
EPOCHS="${EPOCHS:-200}"
SEEDS="${SEEDS:-3}"
DEVICE="${DEVICE:-cuda}"
EXTRA=""
[ "${COMBAT:-0}" = "1" ] && EXTRA="--combat"

echo "==> Multi-atlas SSL ensemble (atlases='${ATLASES}', ssl=${SSL_EPOCHS}, ft=${EPOCHS}, seeds=${SEEDS})"
# shellcheck disable=SC2086
python train_sota.py \
  --atlases ${ATLASES} \
  --protocol "${PROTOCOL}" \
  --n_folds "${N_FOLDS}" \
  --ssl_epochs "${SSL_EPOCHS}" \
  --epochs "${EPOCHS}" \
  --seeds "${SEEDS}" \
  --device "${DEVICE}" \
  ${EXTRA}

RUN_DIR="$(ls -dt results/run_* results/sota* 2>/dev/null | head -1 || true)"
echo ""
echo "Done.  SOTA results in: ${RUN_DIR}/sota/sota_report.json"
