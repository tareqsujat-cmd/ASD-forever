#!/usr/bin/env bash
# =============================================================================
# ASD-forever — RunPod one-shot setup (CUDA GPU box, Linux)
#
#   bash setup_runpod.sh
#
# Does: create venv -> install CUDA PyTorch + deps -> get ABIDE-I data ->
#       preprocess into FC vectors + phenotypics -> sanity check.
#
# Idempotent / resumable.  If `abide_processed.zip` is present in the repo root
# it is unzipped and the ~30-min download+preprocess is skipped — upload that zip
# to RunPod to start training immediately.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

echo "==> [1/6] Python virtual environment"
if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null

echo "==> [2/6] PyTorch (CUDA build)"
CUDA_TAG="cu121"
if command -v nvidia-smi >/dev/null 2>&1; then
  CV=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  case "$CV" in
    12.6*|12.5*|12.4*) CUDA_TAG="cu124" ;;
    12.3*|12.2*|12.1*) CUDA_TAG="cu121" ;;
    11.8*)             CUDA_TAG="cu118" ;;
    *)                 CUDA_TAG="cu121" ;;
  esac
  echo "    Detected CUDA ${CV:-unknown} -> ${CUDA_TAG}"
else
  echo "    nvidia-smi not found; defaulting to ${CUDA_TAG}"
fi
python -c "import torch" 2>/dev/null && echo "    torch already installed" || \
  pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "==> [3/6] Python dependencies"
pip install -r requirements.txt

echo "==> [4/6] Verify GPU"
python - <<'PY'
import torch
print("    torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("    GPU:", torch.cuda.get_device_name(0))
else:
    print("    WARNING: CUDA not available — training will fall back to CPU.")
PY

echo "==> [5/6] Data (download + preprocess, or unzip prepared data)"
# ATLASES controls which ROI time series to fetch.  CC200 is required (FC pipeline
# + single-atlas benchmark); AAL + HO are needed for the multi-atlas SOTA run
# (train_sota.py).  Set ATLASES="cc200" to skip the extra atlases.
ATLASES="${ATLASES:-cc200 aal ho}"
declare -A ATLAS_SUFFIX=( [cc200]=rois_cc200 [aal]=rois_aal [ho]=rois_ho )

if [ -f "abide_processed/mri/metadata.csv" ]; then
  echo "    abide_processed/ already present."
elif [ -f "abide_processed.zip" ]; then
  echo "    Found abide_processed.zip — unzipping (skips CC200 download + preprocess)."
  unzip -q -o abide_processed.zip
else
  echo "    Downloading ABIDE-I CC200 ROI time series (~340 MB)…"
  python data/download_abide.py --data_dir ./abide_raw --pipeline cpac --atlas rois_cc200
  echo "    Preprocessing CC200 -> 19,900-d FC vectors + 6 phenotypics…"
  python data/preprocess_abide.py \
    --pheno_csv ./abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv \
    --ts_dir    ./abide_raw/ABIDE_pcp/cpac/filt_noglobal \
    --atlas rois_cc200 --out_dir ./abide_processed --n_jobs 4 --resume
fi

# Extra atlases for the multi-atlas SOTA run (raw time series only — no preprocess).
for a in $ATLASES; do
  suffix="${ATLAS_SUFFIX[$a]}"
  [ "$a" = "cc200" ] && continue
  if ls ./abide_raw/ABIDE_pcp/cpac/filt_noglobal/*_"$suffix".1D >/dev/null 2>&1; then
    echo "    atlas $a already downloaded."
  else
    echo "    Downloading atlas $a ($suffix)…"
    python data/download_abide.py --data_dir ./abide_raw --pipeline cpac --atlas "$suffix"
  fi
done

echo "==> [6/6] Sanity check"
python - <<'PY'
import glob, pandas as pd
m = pd.read_csv("abide_processed/mri/metadata.csv")
print(f"    Subjects: {len(m)} | ASD={int((m.label==1).sum())} TC={int((m.label==0).sum())} "
      f"| sites={m.site.nunique()} | FC files={len(glob.glob('abide_processed/mri/*.npy'))}")
PY

echo ""
echo "Setup complete.  Next:  bash run_runpod.sh"
