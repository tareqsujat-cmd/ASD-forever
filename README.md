# ASD Multimodal Detection Framework

> **Multimodal Autism Spectrum Disorder Detection via Functional Connectivity and Phenotypic Features**
>
> Targeting IEEE EMBC / ISBI / BIBM or *Transactions on Neural Systems and Rehabilitation Engineering*

---

## Quick Start — New Collaborators

**Prerequisites:** Python 3.10+, a CUDA-capable GPU (4 GB+ VRAM), internet connection.

```bash
# 1. Clone the repository
git clone https://github.com/tareqsujat-cmd/ASD-forever.git
cd ASD-forever

# 2. Run the one-command bootstrap
python setup.py
```

`setup.py` automates everything:

| Step | What it does |
|------|-------------|
| 1 | Checks Python ≥ 3.10 |
| 2 | Detects your CUDA version via `nvidia-smi` |
| 3 | Creates a `.venv/` virtual environment |
| 4 | Installs PyTorch with the correct CUDA build (cu118 / cu121 / cu124 / cu126) |
| 5 | Installs all dependencies from `requirements.txt` |
| 6 | Downloads and preprocesses ABIDE I — ~340 MB download, ~80 MB stored on disk |
| 7 | Runs a 2-minute smoke test on synthetic data to confirm everything works |

When it finishes, activate the environment and run the full experiment:

```bash
# Windows
.venv\Scripts\activate

# Linux / Mac
source .venv/bin/activate

# Full training run  (~50–60 min on a GTX 1650 / RTX 3060)
python run_experiment.py \
    --real_data \
    --mri_dir ./abide_processed/mri \
    --gen_dir ./abide_processed/gen \
    --max_epochs 100 \
    --n_folds 5 \
    --skip_profile \
    --n_hpo_trials 10

# Generate performance plots after training
python results/generate_plots.py
```

**Optional flags for `setup.py`:**

```
--skip-data     abide_processed/ already exists — skip the download
--skip-smoke    Skip the 2-min smoke test
--cpu-only      No GPU available (training will be ~10x slower)
--no-venv       Install into your current Python instead of creating .venv/
```

> For a full explanation of the project, dataset, and what the model actually
> processes, read **[GETTING_STARTED.md](GETTING_STARTED.md)**.

---

## Overview

A complete, reproducible pipeline for binary ASD / TC classification that fuses
two input streams through a **Cross-Modal Attention** layer:

| Stream | Input | Dimension |
|--------|-------|-----------|
| "MRI" branch | Functional connectivity vector (CC200 atlas, ABIDE I) | 19,900 floats |
| "Genetics" branch | Phenotypic proxy features (age, sex, IQ scores) | 6 floats |

> **Note on the "MRI" branch:** The model does **not** process raw brain scans.
> It processes a *functional connectivity* (FC) vector — the upper triangle of
> the 200×200 ROI-to-ROI Pearson correlation matrix computed from resting-state
> fMRI.  This 19,900-element vector is zero-padded to 28³ and reshaped to
> `(1, 28, 28, 28)` to match the 3D ResNet backbone's expected input shape.
> See [GETTING_STARTED.md](GETTING_STARTED.md) for a full explanation.

---

## Architecture

```
Resting-state fMRI  →  CC200 atlas parcellation  →  200×200 FC matrix
                                                            │
                                        upper triangle  (19,900 floats)
                                        zero-pad  →  (21,952 = 28³)
                                        reshape   →  (1, 28, 28, 28)
                                                            │
                                              ┌─────────────▼──────────┐
                                              │  MRI Encoder (3D ResNet)│
                                              │  47.2M params           │
                                              └─────────────┬──────────┘
                                                            │ f_mri (d=256)

Phenotypic features: age, sex, FIQ, VIQ, PIQ, site  (6 floats)
                                                            │
                                              ┌─────────────▼──────────┐
                                              │  Genetics Encoder       │
                                              │  Transformer, 3.26M p   │
                                              │  4 layers · 8 heads     │
                                              └─────────────┬──────────┘
                                                            │ f_gen (d=256)

                                              ┌─────────────▼──────────┐
                                              │  CrossAttentionFusion   │  ← novel
                                              │  11M params             │
                                              │  4 tokens · 2 layers    │
                                              └─────────────┬──────────┘
                                                            │
                                                     ASD / TC logit

Total parameters: ~61.5M
Loss:             FocalLoss (α=0.25, γ=2.0, label smoothing=0.10)
Optimizer:        AdamW + cosine warmup (lr=1e-4, wd=1e-5)
Training:         5-fold CV · AMP · gradient accumulation · EMA
```

---

## Dataset — ABIDE I

| Property | Value |
|----------|-------|
| Subjects (after QC) | **1,100** — 530 ASD / 570 TC |
| Acquisition sites | 20 independent sites |
| Preprocessing pipeline | CPAC |
| Brain atlas | CC200 (200 cortical ROIs) |
| Processed data size | ~80 MB total |
| Access | Free, no registration required |

Data is downloaded automatically by `setup.py` / `prepare_abide.py` via
`nilearn`.  No manual registration or sign-up required.

---

## Results (latest run)

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

Per-fold best AUCs (5-fold CV): 0.695 · 0.703 · 0.739 · 0.765 · 0.717

HPO best: AUROC=0.7187, `lr=7.3e-4`, `fusion=late`, `batch_size=32`

---

## Repository Structure

```
ASD_forever/
├── setup.py                 ← START HERE — one-command bootstrap
├── GETTING_STARTED.md       ← full onboarding guide for new collaborators
│
├── run_experiment.py        End-to-end pipeline (all 8 stages)
├── prepare_abide.py         Download + preprocess ABIDE I or II
├── predict.py               Single-subject inference from checkpoint
│
├── configs/
│   ├── config.yaml          All hyperparameters (master config)
│   └── config_schema.py     Type-safe dataclass schema
│
├── data/
│   └── abide_dataset.py     Dataset class — FC vector → 28³ reshape logic
│
├── models/
│   ├── mri/                 ResNet3D encoder + SEBlock3D
│   ├── genetics/            GeneTransformerEncoder
│   └── fusion/              CrossAttention, Gated, Late, Dynamic variants
│
├── training/
│   │   FocalLoss · AdamW · cosine warmup · ModelEMA · EarlyStopping
│   └── ASDTrainer           AMP + gradient accumulation + K-fold CV
│
├── evaluation/              AUC/F1/MCC/Brier + bootstrap CI, calibration,
│                            per-site breakdown, DeLong / McNemar tests
│
├── hyperparameter_tuning/   Optuna 4.x — TPE/CMA-ES, fANOVA importance
│
├── ablation/                OFAT / factorial ablation runner
│
├── explainability/          GradCAM3D, IntegratedGradients, AttentionRollout
│
├── visualization/           IEEE-format figures
│
├── results/
│   ├── evaluation/          evaluation_report.json (metrics + CIs)
│   ├── hpo/                 hpo_best_trial.json
│   ├── training/checkpoints/ top-k model checkpoints per fold
│   └── plots/               generated by results/generate_plots.py
│
├── requirements.txt
└── environment.yml
```

---

## Reproducibility

- Master seed: `42` (set in `configs/config.yaml`)
- All hyperparameters in one place — no hard-coded values in source
- Optuna studies persist to SQLite; interrupted HPO runs resume automatically
- Checkpoint manager keeps top-k checkpoints per fold, evicts worst on save

---

## Citation

*(To be updated after acceptance)*

---

## License

MIT — see [LICENSE](LICENSE) for details.
