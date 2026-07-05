# ASD Multimodal Detection Framework

> **Multimodal Autism Spectrum Disorder Detection via Functional MRI and Phenotypic Genetics**
>
> Targeting IEEE EMBC / ISBI / BIBM or *Transactions on Neural Systems and Rehabilitation Engineering*

---

## Overview

A complete, reproducible, publication-ready framework for ASD detection that fuses resting-state fMRI features with phenotypic/genetic features through a **Cross-Modal Attention** mechanism.  All 12 modules are fully implemented, tested, and wired end-to-end.

**Key claim:** Cross-modal attention fusion between a 3D MRI encoder and a Transformer genetics encoder outperforms every single-modality baseline and simpler fusion strategies (concatenation, gating, late fusion) on ABIDE I, with generalisation confirmed on the held-out ABIDE II cohort.

---

## Architecture

```
Resting-state fMRI (96×96×96)
        │
  ┌─────▼──────────────────────────┐
  │  MRI Encoder                   │
  │  ResNet3D / DenseNet3D /        │
  │  SwinTransformer3D / ConvNeXt3D│
  │  + SEBlock3D attention          │
  └─────────────────┬──────────────┘
                    │ f_mri (d=256)
                    │
Phenotypic / Gene expression (256-D PCA)
        │
  ┌─────▼──────────────────────────┐
  │  Genetics Encoder              │
  │  TransformerEncoder / TabNet / │
  │  GNNEncoder                    │
  └─────────────────┬──────────────┘
                    │ f_gen (d=256)
                    │
  ┌─────────────────▼──────────────┐
  │  Multimodal Fusion             │
  │  CrossAttention  ←  (novel)    │
  │  (also: Gated, Intermediate,   │
  │   Late, Dynamic ablations)     │
  └─────────────────┬──────────────┘
                    │
              ASD / TC logit
```

---

## Repository Structure

```
ASD_forever/
├── configs/
│   ├── config.yaml              Master configuration (all hyperparameters)
│   └── config_schema.py         Type-safe Python dataclass schema
│
├── preprocessing/
│   ├── mri/                     NIfTI I/O, resampling, normalisation, QC,
│   │                            skull stripping, bias correction, registration
│   └── genetics/                Imputation, ComBat, feature selection,
│                                PCA/VAE, gene graph builder
│
├── models/
│   ├── mri/                     ResNet3D, DenseNet3D, SwinTransformer3D,
│   │                            ConvNeXt3D, MRIEncoder, SEBlock3D
│   ├── genetics/                GeneTransformerEncoder, TabNet, GNNEncoder,
│   │                            GeneticsEncoder
│   └── fusion/                  CrossAttention, Gated, Intermediate,
│                                Late, Dynamic + MultiModalFusion
│
├── training/
│   │   FocalLoss, AdamW param groups, cosine warmup,
│   │   ModelEMA, EarlyStopping, CheckpointManager,
│   └── ASDTrainer (AMP + grad accumulation + K-fold CV)
│
├── evaluation/                  Full metric suite (AUC, F1, MCC, Brier…),
│                                bootstrap CI, DeLong, McNemar, Wilcoxon,
│                                calibration, per-site breakdown
│
├── explainability/              GradCAM3D, GradCAM++3D, IntegratedGradients,
│                                SmoothGrad, AttentionRollout,
│                                GeneticsFeatureImportance, ExplainabilityEngine
│
├── ablation/                    AblationStudy (OFAT / factorial / custom),
│                                AblationRunner, AblationAnalyzer,
│                                Wilcoxon significance, LaTeX/markdown reports
│
├── hyperparameter_tuning/       Optuna 4.x — TPE / CMA-ES / NSGA-II,
│                                MedianPruner / HyperbandPruner,
│                                fANOVA importance, TuningAnalyzer
│
├── visualization/               IEEE-format figures: ROC/PR + CI bands,
│                                confusion matrix, calibration, t-SNE/UMAP,
│                                MRI saliency triplets, gene importance
│
├── paper/                       PaperFigureGenerator — all 9 camera-ready
│                                figures (PDF + PNG) + LaTeX manifest
│
├── tests/                       Pytest suite — 549 tests total, all passing
│
├── prepare_abide.py             Download + preprocess ABIDE I or II
├── run_experiment.py            End-to-end pipeline (all 8 stages)
├── predict.py                   Single-subject inference from checkpoint
├── requirements.txt
└── environment.yml
```

---

## Quick Start

### 1 — Environment

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate asd-detection

# Or pip
pip install -r requirements.txt

# GPU build (replace cu118 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2 — Smoke test (no data download required, ~2 min on CPU)

```bash
python run_experiment.py
```

This runs all 8 pipeline stages on 40 synthetic subjects and writes results to `results/smoke_test/`.

### 3 — Prepare ABIDE I (training corpus, ~8 GB download)

```bash
python prepare_abide.py \
    --dataset abide1 \
    --out_dir datasets/abide1
```

### 4 — Prepare ABIDE II (held-out external test set, ~9 GB download)

```bash
python prepare_abide.py \
    --dataset abide2 \
    --role held_out \
    --abide1_dir datasets/abide1 \
    --out_dir datasets/abide2
```

The `--abide1_dir` flag loads the ABIDE I fitted intensity normaliser and applies it to ABIDE II without re-fitting, preventing data leakage.

### 5 — Full experiment

```bash
python run_experiment.py \
    --real_data \
    --mri_dir datasets/abide1/mri \
    --gen_dir  datasets/abide1/genetics \
    --held_out_mri_dir datasets/abide2/mri \
    --held_out_gen_dir datasets/abide2/genetics \
    --n_folds 5 \
    --max_epochs 100 \
    --out_dir results/full_run
```

### 6 — Inference on a new subject

```bash
python predict.py \
    --checkpoint results/full_run/training/checkpoints/best_model.pt \
    --mri_nifti  path/to/subject.nii.gz \
    --genetics   path/to/subject_genetics.npy \
    --out_json   output/prediction.json
```

---

## Datasets

| Dataset | Modality | Subjects | Sites | Access |
|---|---|---|---|---|
| ABIDE I | Resting-state fMRI | ~1,112 | 17 | Public via nilearn |
| ABIDE II | Resting-state fMRI | ~1,114 | 27 | Public via nilearn |

Both datasets are downloaded automatically by `prepare_abide.py` using `nilearn.datasets.fetch_abide_pcp()` / `fetch_abide2()`.  No manual registration required.

> **Note on genetics:** ABIDE does not include SNP or RNA-seq data.  By default, `prepare_abide.py --genetics_mode phenotypic` builds a genetics proxy vector from ABIDE's phenotypic variables (age, sex, FIQ, VIQ, PIQ, ADOS, ADI-R) via PCA.  If you have a matched GEO expression matrix, pass `--genetics_mode geo --geo_csv path/to/expression.csv` to run the full ComBat → feature-selection → PCA pipeline.

---

## Modules

| # | Module | Tests | Description |
|---|---|---|---|
| 0 | Config + scaffold | — | YAML config, type-safe dataclass schema, seeding, hardware utils |
| 1 | MRI preprocessing | — | NIfTI I/O, resampling, pad/crop, intensity normalisation (site-aware), QC |
| 2 | Genetics preprocessing | — | Imputation, ComBat batch correction, feature selection, PCA/VAE |
| 3 | MRI feature extraction | — | ResNet3D, DenseNet3D, SwinTransformer3D, ConvNeXt3D, SEBlock3D |
| 4 | Genetics feature extraction | — | GeneTransformerEncoder, TabNet, GNNEncoder, GeneticsEncoder wrapper |
| 5 | Multimodal fusion | — | CrossAttention, Gated, Intermediate, Late, Dynamic |
| 6 | Training engine | — | FocalLoss, AdamW + cosine warmup, ModelEMA, AMP, K-fold CV |
| 7 | Evaluation suite | 78 | AUC/F1/MCC/Brier + bootstrap CI, DeLong, McNemar, Wilcoxon, calibration |
| 8 | Explainability | 70 | GradCAM3D, GradCAM++3D, IntegratedGradients, SmoothGrad, AttentionRollout |
| 9 | Visualisation | 53 | ROC/PR + CI, confusion matrix, calibration, t-SNE/UMAP, saliency, gene bars |
| 10 | Ablation study | 85 | OFAT/factorial/custom runner, Wilcoxon significance, LaTeX/markdown reports |
| 11 | Hyperparameter tuning | 85 | Optuna 4.x — TPE/CMA-ES/NSGA-II, fANOVA importance, TuningAnalyzer |
| 12 | Paper figures | 63 | 9 IEEE camera-ready figures (PDF + PNG), LaTeX manifest, `generate_all()` |

---

## Reproducibility

- Master seed: `42` (set in `configs/config.yaml`)
- Per-scope seeds derived via SHA-256 from master seed + scope string
- All hyperparameters live in `configs/config.yaml` — no hard-coded values in code
- Optuna studies persist to SQLite; interrupted HPO runs resume automatically
- Ablation runner checkpoints progress; interrupted runs resume from last completed variant

---

## Hyperparameter Tuning

```python
from hyperparameter_tuning import ASDTuner

tuner = ASDTuner(
    train_fn     = your_train_fn,
    base_config  = cfg,
    search_space = "full",      # full / optimizer / architecture / fusion / quick
    n_trials     = 100,
    sampler      = "tpe",       # tpe / cma / nsga2
    pruner       = "hyperband",
    storage_path = "results/hpo/study.db",  # SQLite resume
)
tuner.optimize()
print(tuner.best_params)
```

---

## Explainability

```python
from explainability import ExplainabilityEngine

engine = ExplainabilityEngine(model, cfg)
result = engine.explain(batch)

# result.mri_gradcam       — (D, H, W) saliency map
# result.mri_gradcam_pp    — GradCAM++ variant
# result.ig_attributions   — Integrated Gradients
# result.gene_importance   — per-gene importance scores
# result.attention_rollout — cross-attention weights
```

---

## Citation

*(To be updated after acceptance)*

---

## License

MIT — see [LICENSE](LICENSE) for details.
