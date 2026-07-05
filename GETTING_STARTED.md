# Getting Started — Collaborator Guide

This document tells you exactly what the project does, what data it uses, and
how to reproduce results from scratch.

---

## Quick start (automated)

The fastest path — one command sets up the environment, downloads the data,
and runs a smoke test:

```bash
# 1. Clone the repo
git clone https://github.com/tareqsujat-cmd/ASD-forever.git
cd ASD-forever

# 2. Run the bootstrap script  (needs Python 3.10+ and a CUDA GPU)
python setup.py
```

`setup.py` will:
- Detect your CUDA version and install the matching PyTorch build
- Create a `.venv/` virtual environment and install all dependencies
- Download and preprocess ABIDE I (~340 MB download, ~80 MB stored)
- Run a smoke test on synthetic data to confirm everything works

After it finishes, activate the environment and run the full experiment:

```bash
# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

# Full experiment (~50-60 min on a GTX 1650)
python run_experiment.py \
    --real_data \
    --mri_dir ./abide_processed/mri \
    --gen_dir ./abide_processed/gen \
    --max_epochs 100 \
    --n_folds 5 \
    --skip_profile \
    --n_hpo_trials 10
```

Optional flags for `setup.py`:

| Flag | Use when |
|------|----------|
| `--skip-data` | You already have `abide_processed/` from a previous run |
| `--skip-smoke` | You want to skip the 2-min verification step |
| `--cpu-only` | No GPU available (training will be much slower) |
| `--no-venv` | You prefer to manage your own environment |

---

## What this project is

A deep-learning framework that classifies subjects as **ASD (Autism Spectrum
Disorder)** or **TC (Typical Control)** using two input streams:

| Stream | What it is | Dimensionality |
|--------|-----------|----------------|
| "MRI" branch | Functional connectivity (FC) vector from resting-state fMRI | 19,900 floats |
| "Genetics" branch | Phenotypic proxy features (age, sex, IQ scores) | 6 floats |

The two streams are fused via a **Cross-Modal Attention** layer before a binary
classifier head.

> **Important:** Despite the name "MRI encoder", the model does **not** see raw
> brain scans.  It sees a flattened 200×200 ROI correlation matrix — a number
> that measures how synchronized each pair of brain regions is during rest.
> This is explained in detail below.

---

## The ABIDE I dataset

ABIDE I (Autism Brain Imaging Data Exchange, release 2013) is a public
resting-state fMRI dataset assembled from 20 independent acquisition sites.

| Property | Value |
|----------|-------|
| Total subjects | 1,112 (after QC: **1,100** used here) |
| ASD | 530 |
| Typical Control | 570 |
| Acquisition sites | 20 |
| Preprocessing pipeline | CPAC |
| Brain atlas | CC200 (200 cortical ROIs) |
| Access | Free, no registration required |

### What "resting-state fMRI" means

Subjects lie still in an MRI scanner for ~6 minutes while BOLD (blood-oxygen-
level-dependent) signal is recorded.  The scanner captures a 4D volume (x, y,
z, time).  Each voxel's time series reflects neural activity indirectly.

### What "CC200 atlas" means

The brain volume is parcellated into 200 regions of interest (ROIs) using the
CC200 atlas.  Each ROI's time series is the average BOLD signal within that
region.  This reduces a huge 4D volume (~2 GB/subject) to a 200×T matrix.

### What this project extracts — the FC vector

From the 200-ROI time matrix, we compute a **functional connectivity matrix**:
the pairwise Pearson correlation between every ROI pair.  The full matrix is
200×200 and symmetric.  We keep only the upper triangle:

```
upper triangle of 200×200  =  (200 × 199) / 2  =  19,900 values
```

This 19,900-dimensional vector is the actual input to the model's "MRI" branch.
It encodes **how connected each brain region is to every other region**.

### How it is fed to the 3D ResNet

The MRI encoder backbone is a 3D ResNet designed for volumetric inputs shaped
`(B, 1, D, H, W)`.  To reuse it without modifications, the 19,900-element
vector is **zero-padded** to 21,952 = 28³ and **reshaped** to `(1, 28, 28, 28)`.
The ResNet's `AdaptiveAvgPool3d(1)` output layer is spatially agnostic, so it
works correctly despite the input not being a real volume.

This is an acknowledged workaround — a proper fix would be a 1D CNN or MLP
backbone — but it is functional and produces the reported results.

### Phenotypic / "Genetics" features (6 values per subject)

ABIDE does not include genetic sequencing data.  The 6-element "genetics" vector
is a phenotypic proxy built from ABIDE's metadata:

| Feature | Description |
|---------|-------------|
| Age | Subject age in years |
| Sex | 0=female, 1=male |
| FIQ | Full-scale IQ |
| VIQ | Verbal IQ |
| PIQ | Performance IQ |
| Site index | Acquisition site (0–19) |

---

## Environment setup

**Requirements:** Python 3.10+, pip, a CUDA-capable GPU (4 GB+ VRAM), Windows
or Linux.

```bash
# 1. Clone the repository
git clone https://github.com/tareqsujat-cmd/ASD-forever.git
cd ASD-forever

# 2a. GPU build — replace cu118 with your CUDA version (check: nvcc --version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2b. Install remaining dependencies
pip install -r requirements.txt
```

Verify CUDA is available:
```python
import torch
print(torch.cuda.is_available())   # should print True
print(torch.cuda.get_device_name(0))
```

---

## Smoke test (no data needed, ~2 min)

Before downloading any data, verify the pipeline runs end-to-end on synthetic
subjects:

```bash
python run_experiment.py
```

This generates 40 fake subjects in memory, runs all 8 pipeline stages, and
writes results to `results/smoke_test/`.  If this completes without errors the
environment is correctly set up.

---

## Download and preprocess ABIDE I

```bash
python prepare_abide.py --dataset abide1 --out_dir abide_processed
```

This script:
1. Downloads ABIDE I ROI time series from the Preprocessed Connectomes Project
   S3 bucket via `nilearn` (~340 MB download, discarded after processing)
2. Computes the 19,900-dimensional FC vector for each subject
3. Extracts the 6 phenotypic features
4. Writes everything to `abide_processed/`

**Output layout** (total on disk: ~80 MB):
```
abide_processed/
  mri/
    metadata.csv          — subject_id, label, site, split columns
    000050001.npy         — (19900,) float32 FC vector per subject
    000050002.npy
    ...
  gen/
    000050001.npy         — (6,) float32 phenotypic vector per subject
    ...
```

The `label` column in `metadata.csv` is `1` for ASD and `0` for TC.

---

## Full experiment run

```bash
python run_experiment.py \
    --real_data \
    --mri_dir ./abide_processed/mri \
    --gen_dir ./abide_processed/gen \
    --max_epochs 100 \
    --n_folds 5 \
    --skip_profile \
    --n_hpo_trials 10
```

| Flag | Meaning |
|------|---------|
| `--real_data` | Use the downloaded ABIDE I data instead of synthetic subjects |
| `--mri_dir` | Path to `abide_processed/mri/` |
| `--gen_dir` | Path to `abide_processed/gen/` |
| `--max_epochs 100` | Maximum training epochs per fold |
| `--n_folds 5` | 5-fold cross-validation (880 train / 220 val per fold) |
| `--skip_profile` | Skip the memory profiling stage |
| `--n_hpo_trials 10` | Run 10 Optuna hyperparameter search trials |

**Expected wall time:** ~50–60 minutes on an NVIDIA GTX 1650 (4 GB VRAM).
Early stopping with patience=15 usually triggers before epoch 100.

---

## Results

After the run, results are written to `results/`:

```
results/
  evaluation/
    evaluation_report.json    — all metrics with 95% bootstrap CIs
  hpo/
    hpo_best_trial.json       — best hyperparameter configuration found
  training/
    checkpoints/              — top-k model checkpoints per fold
  plots/                      — generated after running generate_plots.py
```

Generate the performance plots:
```bash
python results/generate_plots.py
```

Plots are saved to `results/plots/` (10 figures: ROC curve, PR curve,
calibration, confusion matrix, per-site AUC, fold checkpoints, etc.).

### Reported results (latest run)

| Metric | Value | 95% CI |
|--------|-------|--------|
| AUROC | 0.7110 | [0.685 – 0.744] |
| AUPRC | 0.6797 | — |
| Accuracy | 67.8% | — |
| Sensitivity (ASD recall) | 64.3% | — |
| Specificity (TC recall) | 71.1% | — |
| F1 | 0.658 | — |
| MCC | 0.355 | — |
| ECE (calibration) | 0.110 | — |

HPO best trial: AUROC=0.7187, `lr=7.3e-4`, `fusion=late`, `batch_size=32`.

---

## Model architecture summary

```
Input A: FC vector (19,900,) → zero-pad → reshape → (1, 28, 28, 28)
    └── 3D ResNet MRI encoder (47.2M params, random init) → 256-d embedding

Input B: phenotypic features (6,)
    └── GeneTransformerEncoder (3.26M params, 4 layers, 8 heads) → 256-d embedding

Fusion: CrossAttentionFusion (11M params, 4 tokens × 2 layers) → 512-d
    └── Linear classifier → ASD / TC logit

Total parameters: ~61.5M
Loss: FocalLoss (α=0.25, γ=2.0, label smoothing=0.10)
Optimizer: AdamW + cosine warmup scheduler
```

---

## Common issues

**`CUDA out of memory`** — Reduce `--batch_size` to 4 or 8.

**`nilearn download fails`** — The S3 bucket sometimes throttles.  Re-run
`prepare_abide.py`; it resumes from where it left off.

**`ModuleNotFoundError: optuna`** — Run `pip install optuna>=4.9.0`.

**Training is very slow on CPU** — The model requires a CUDA GPU.  CPU training
will work but takes 10× longer per epoch.

---

## Key files to understand

| File | What it does |
|------|-------------|
| [data/abide_dataset.py](data/abide_dataset.py) | Dataset class — explains the FC vector → 28³ reshape |
| [prepare_abide.py](prepare_abide.py) | Downloads and preprocesses ABIDE I/II |
| [run_experiment.py](run_experiment.py) | Orchestrates all 8 pipeline stages |
| [models/fusion/](models/fusion/) | CrossAttentionFusion and ablation variants |
| [training/checkpointing.py](training/checkpointing.py) | Checkpoint management |
| [results/generate_plots.py](results/generate_plots.py) | Generates performance figures from evaluation JSON |
| [configs/config.yaml](configs/config.yaml) | All hyperparameters in one place |
