# ASD Multimodal Detection Framework

> **Multimodal Autism Spectrum Disorder Detection via Functional Connectivity and Phenotypic Features**
>
> Targeting IEEE EMBC / ISBI / BIBM or *Transactions on Neural Systems and Rehabilitation Engineering*

---

## Quick Start — New Collaborators

### Before you begin — check these three things

**1. Python 3.10 or later**

```bash
python --version
```

Expected output:
```
Python 3.10.12        ← any 3.10.x or higher is fine
```

If you see `Python 3.8` or lower, download a newer version from python.org/downloads.

---

**2. Git**

```bash
git --version
```

Expected output:
```
git version 2.43.0    ← any recent version is fine
```

If not found, install from git-scm.com.

---

**3. NVIDIA GPU and drivers**

```bash
nvidia-smi
```

Expected output (values will match your GPU):
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 551.61       Driver Version: 551.61       CUDA Version: 12.4                |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA GeForce GTX 1650        Off |   00000000:01:00.0 Off |                  N/A |
+-----------------------------------------------------------------------------------------+
```

The important line is **CUDA Version** in the top-right corner — the setup script
reads this and automatically installs the matching PyTorch build.

If `nvidia-smi` is not found, install NVIDIA drivers from nvidia.com before continuing.

No GPU? You can still run with `python setup.py --cpu-only` but training will take ~8 hours instead of ~1 hour.

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/tareqsujat-cmd/ASD-forever.git
cd ASD-forever
```

Expected output:
```
Cloning into 'ASD-forever'...
remote: Enumerating objects: 185, done.
remote: Counting objects: 100% (185/185), done.
remote: Compressing objects: 100% (120/120), done.
Receiving objects: 100% (185/185), 2.35 MiB | 4.20 MiB/s, done.
Resolving deltas: 100% (55/55), done.
```

---

### Step 2 — Run the bootstrap script

```bash
python setup.py
```

This one command handles everything. It runs in 7 steps — here is exactly what you will see:

---

**Step 1/7 — Python version check**
```
======================================================================
  Step 1 / 7 — Checking Python version
======================================================================
  Python 3.10 detected
  OK
```

---

**Step 2/7 — CUDA detection**
```
======================================================================
  Step 2 / 7 — Detecting CUDA
======================================================================
  Detected CUDA 12.4
  Selected PyTorch wheel index: cu124
```

The script matches your CUDA version to the right PyTorch build automatically.
Supported: CUDA 11.8, 12.1, 12.4, 12.6. If your version is older than 11.8 it
will fall back to CPU-only PyTorch.

---

**Step 3/7 — Virtual environment**
```
======================================================================
  Step 3 / 7 — Virtual environment
======================================================================
  Creating .venv/ ...
  Done
  To activate later:  .venv\Scripts\activate        (Windows)
  To activate later:  source .venv/bin/activate     (Linux / Mac)
```

A `.venv/` folder will appear in the project directory. All packages are
installed inside it — nothing is installed system-wide.

---

**Step 4/7 — PyTorch installation**
```
======================================================================
  Step 4 / 7 — Installing PyTorch
======================================================================
  $ pip install torch>=2.1.0 torchvision>=0.16.0 --index-url https://...
  Collecting torch
  Downloading torch-2.3.0+cu124-cp310-cp310-win_amd64.whl (2.4 GB)
  ████████████████████ 100%  2.4 GB  4.5 MB/s
  ...
  PyTorch 2.3.0+cu124 | CUDA available: True
```

The download is ~2 GB — this is normal. The last line confirms CUDA is working.
If it says `CUDA available: False`, your NVIDIA drivers may need updating.

---

**Step 5/7 — Dependencies**
```
======================================================================
  Step 5 / 7 — Installing dependencies (requirements.txt)
======================================================================
  $ pip install -r _setup_requirements_tmp.txt
  Collecting numpy>=1.26.0
  ...
  Successfully installed einops-0.7.0 matplotlib-3.8.2 nilearn-0.10.3
    numpy-1.26.4 optuna-4.9.0 pandas-2.1.4 scikit-learn-1.4.0
    scipy-1.11.4 seaborn-0.13.2 tqdm-4.66.1 umap-learn-0.5.5 ...
```

---

**Step 6/7 — ABIDE I data download**
```
======================================================================
  Step 6 / 7 — Downloading and preprocessing ABIDE I
======================================================================
  Downloading ABIDE I ROI time series and computing FC vectors ...
  (This will download ~340 MB; processed output is ~80 MB)
  Expected time: 5-15 minutes depending on your internet speed

  Downloading subject 000050001 ...  [1/1100]
  Downloading subject 000050002 ...  [2/1100]
  ...
  Computing FC vectors ... 100%|████████████| 1100/1100
  Written: abide_processed/mri/  (1100 .npy files + metadata.csv)
  Written: abide_processed/gen/  (1100 .npy files)
```

After this step, an `abide_processed/` folder will appear in the project
directory containing ~80 MB of preprocessed data. The raw fMRI downloads are
discarded after processing to save disk space.

---

**Step 7/7 — Smoke test**
```
======================================================================
  Step 7 / 7 — Smoke test (synthetic data, ~2 min)
======================================================================
  Running pipeline on 40 synthetic subjects ...

  [Stage 1/8] Config + data loading ............. OK
  [Stage 2/8] MRI preprocessing ................. OK
  [Stage 3/8] Genetics preprocessing ............ OK
  [Stage 4/8] Training (2 epochs, 2 folds) ....... OK
  [Stage 5/8] Evaluation ........................ OK
  [Stage 6/8] Explainability .................... OK
  [Stage 7/8] HPO (1 trial) ..................... OK
  [Stage 8/8] Ablation .......................... OK

  Smoke test passed!
```

If any stage fails, the error message will tell you exactly what went wrong.
The most common cause is a missing dependency — re-run `pip install -r requirements.txt` inside the `.venv`.

---

**Setup complete — final message**
```
======================================================================
  Setup complete
======================================================================

  Your environment is ready.  To run the full experiment:

    .venv\Scripts\activate          (Windows)
    source .venv/bin/activate       (Linux / Mac)

    python run_experiment.py \
        --real_data \
        --mri_dir ./abide_processed/mri \
        --gen_dir ./abide_processed/gen \
        --max_epochs 100 \
        --n_folds 5 \
        --skip_profile \
        --n_hpo_trials 10
```

---

### Step 3 — Activate the environment

Every time you open a new terminal, activate the environment first:

```bash
# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate
```

Expected output (your prompt will change):
```
(asd-detection) C:\Users\you\ASD-forever>       (Windows)
(asd-detection) user@machine:~/ASD-forever$     (Linux / Mac)
```

The `(asd-detection)` prefix confirms the virtual environment is active.

---

### Step 4 — Run the full experiment

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

Expected training output (one block per fold, 5 folds total):

```
[HPO] Running 10 Optuna trials to find best hyperparameters ...
  Trial 1 | lr=3.2e-4 | fusion=cross_attention | AUROC=0.6821
  Trial 2 | lr=7.3e-4 | fusion=late            | AUROC=0.7187  ← best
  ...
  Best trial: AUROC=0.7187 | lr=7.3e-4 | batch_size=32

[Fold 1/5] Training ...
  Epoch  1/100 | loss=0.6923 | val_auc=0.5134 | lr=1.43e-05
  Epoch  2/100 | loss=0.6815 | val_auc=0.5602 | lr=2.86e-05
  Epoch  5/100 | loss=0.6541 | val_auc=0.6118 | lr=7.14e-05
  Epoch 10/100 | loss=0.6203 | val_auc=0.6634 | lr=9.87e-05
  Epoch 20/100 | loss=0.5891 | val_auc=0.6893 | lr=9.54e-05
  Epoch 35/100 | loss=0.5744 | val_auc=0.7033 | lr=8.21e-05  ← checkpoint saved
  Epoch 50/100 | loss=0.5699 | val_auc=0.6981 | lr=6.87e-05
  Early stopping at epoch 50 (no improvement for 15 epochs)
  Fold 1 best AUC: 0.7033

[Fold 2/5] Training ...
  ...
  Fold 2 best AUC: 0.6946

... (folds 3, 4, 5) ...

[Stage 4] Evaluation on held-out test set ...
  AUROC:       0.7110  [0.685 – 0.744]
  AUPRC:       0.6797
  Accuracy:    67.8%
  Sensitivity: 64.3%   (how often ASD subjects are correctly identified)
  Specificity: 71.1%   (how often typical controls are correctly identified)
  F1:          0.658
  MCC:         0.355

Results written to: results/evaluation/evaluation_report.json
```

Total expected time: **~50–60 minutes** on a GTX 1650 / RTX 3060.

---

### Step 5 — Generate performance plots

```bash
python results/generate_plots.py
```

Expected output:
```
Loading results/evaluation/evaluation_report.json ...
Saved results/plots/01_roc_curve.png
Saved results/plots/02_pr_curve.png
Saved results/plots/03_score_distribution.png
Saved results/plots/04_calibration.png
Saved results/plots/05_confusion_matrix.png
Saved results/plots/06_metrics_bar.png
Saved results/plots/07_per_site_auc.png
Saved results/plots/08_fold_checkpoints.png
Saved results/plots/09_sens_spec_threshold.png
Saved results/plots/10_scorecard.png
All 10 plots written to results/plots/
```

Open the `results/plots/` folder to view the figures. The most important ones:
- `01_roc_curve.png` — ROC curve with 95% confidence band (target: AUROC ≈ 0.71)
- `05_confusion_matrix.png` — TP/FP/TN/FN breakdown
- `07_per_site_auc.png` — performance across all 20 acquisition sites
- `10_scorecard.png` — single-page summary of all metrics

---

## What the model actually does

Despite the "MRI encoder" name, the model does **not** process raw brain scans.

**What it processes instead:**

1. A resting-state fMRI scan is recorded (~6 min, subject lies still in scanner)
2. The brain is divided into 200 regions using the CC200 atlas
3. Pearson correlations are computed between every pair of regions → 200×200 matrix
4. The upper triangle (19,900 values) is extracted as a "functional connectivity" (FC) vector
5. This vector is zero-padded to 28³ = 21,952 and reshaped to `(1, 28, 28, 28)` to fit the ResNet backbone
6. Alongside this, 6 phenotypic values (age, sex, FIQ, VIQ, PIQ, site) form the "genetics" input

The FC vector encodes *how synchronized each pair of brain regions is at rest* — a well-established biomarker for ASD.

See [GETTING_STARTED.md](GETTING_STARTED.md) for a full explanation of the dataset and model inputs.

---

## Architecture

```
Resting-state fMRI  →  CC200 atlas  →  200×200 FC matrix
                                              │
                              upper triangle  (19,900 floats)
                              zero-pad     →  (21,952 = 28³)
                              reshape      →  (1, 28, 28, 28)
                                              │
                                ┌─────────────▼──────────┐
                                │  MRI Encoder (3D ResNet)│
                                │  47.2M params           │
                                └─────────────┬──────────┘
                                              │ f_mri (d=256)

Phenotypic: age · sex · FIQ · VIQ · PIQ · site  (6 floats)
                                              │
                                ┌─────────────▼──────────┐
                                │  Genetics Encoder       │
                                │  Transformer · 3.26M p  │
                                │  4 layers · 8 heads     │
                                └─────────────┬──────────┘
                                              │ f_gen (d=256)

                                ┌─────────────▼──────────┐
                                │  CrossAttentionFusion   │
                                │  11M params             │
                                └─────────────┬──────────┘
                                              │
                                       ASD / TC logit

Total: ~61.5M parameters
Loss:  FocalLoss (α=0.25, γ=2.0, label smoothing=0.10)
```

---

## Results (latest run — ABIDE I, 5-fold CV)

| Metric | Value | 95% CI |
|--------|-------|--------|
| **AUROC** | **0.7110** | [0.685 – 0.744] |
| AUPRC | 0.6797 | — |
| Accuracy | 67.8% | — |
| Sensitivity (ASD recall) | 64.3% | — |
| Specificity (TC recall) | 71.1% | — |
| F1 | 0.658 | — |
| MCC | 0.355 | — |
| ECE (calibration error) | 0.110 | — |

Per-fold AUCs: 0.695 · 0.703 · 0.739 · 0.765 · 0.717

---

## Dataset — ABIDE I

| Property | Value |
|----------|-------|
| Subjects (after QC) | 1,100 — 530 ASD / 570 TC |
| Acquisition sites | 20 independent sites worldwide |
| Preprocessing pipeline | CPAC |
| Brain atlas | CC200 (200 cortical ROIs) |
| Processed data on disk | ~80 MB |
| Access | Free, no registration required |

---

## Repository Structure

```
ASD_forever/
├── setup.py                 ← START HERE — one-command bootstrap
├── GETTING_STARTED.md       ← full onboarding guide
│
├── run_experiment.py        End-to-end pipeline (all 8 stages)
├── prepare_abide.py         Download + preprocess ABIDE I
├── predict.py               Single-subject inference
│
├── configs/config.yaml      All hyperparameters
├── data/abide_dataset.py    Dataset — FC vector → 28³ reshape logic
├── models/fusion/           CrossAttention, Gated, Late, Dynamic variants
├── training/                FocalLoss · AdamW · EMA · K-fold CV
├── evaluation/              AUC/F1/MCC/Brier + bootstrap CI
├── results/generate_plots.py   Performance figures
│
├── requirements.txt
└── environment.yml
```

---

## Citation

*(To be updated after acceptance)*

## License

MIT — see [LICENSE](LICENSE) for details.
