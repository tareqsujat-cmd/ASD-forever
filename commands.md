# Commands — running the ABIDE-I experiments

All results land in an auto-incrementing **`results/run_N/`** directory (self-contained:
`run_manifest.json`, `config_snapshot.yaml`, `run.log`, and per-stage outputs). Figures use
the **scienceplots** (science + ieee) style. See [publication_plan.md](publication_plan.md)
for the full experiment matrix (E1–E8).

---

## A. RunPod / CUDA (recommended for training)

```bash
# 1. Setup: venv + CUDA PyTorch + deps + data + preprocessing.
#    If abide_processed.zip is uploaded to the repo root, it is unzipped
#    and the download+preprocess step is skipped.
bash setup_runpod.sh

# 2. Train + evaluate + figures (pooled stratified 10-fold, CUDA).
bash run_runpod.sh

# Override run size via env vars:
N_FOLDS=10 MAX_EPOCHS=100 N_HPO_TRIALS=20 bash run_runpod.sh
```

---

## B. Local (Mac / Apple MPS or CPU)

```bash
# One-command bootstrap (plain venv, no conda): env + deps + data + smoke test.
python3.12 setup.py            # add --skip-data if abide_processed/ exists

# Activate the environment.
source .venv/bin/activate

# Smoke test — synthetic data, no ABIDE needed (~2 min); verifies the pipeline.
python run_experiment.py --device mps        # or --device cpu
```

---

## C. Data preparation (manual, if not using the setup scripts)

```bash
# Download ABIDE-I CC200 ROI time series (~340 MB) into ./abide_raw
python data/download_abide.py --data_dir ./abide_raw --pipeline cpac --atlas rois_cc200

# Preprocess -> 19,900-d FC vectors + 6 phenotypics into ./abide_processed
python data/preprocess_abide.py \
    --pheno_csv ./abide_raw/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv \
    --ts_dir    ./abide_raw/ABIDE_pcp/cpac/filt_noglobal \
    --atlas rois_cc200 --out_dir ./abide_processed --n_jobs 4 --resume

# Verify processed data
python data/preprocess_abide.py --out_dir ./abide_processed --verify_only
```

---

## D. Full experiment run (real ABIDE-I)

```bash
python run_experiment.py \
    --real_data \
    --mri_dir ./abide_processed/mri \
    --gen_dir ./abide_processed/gen \
    --device cuda \            # cuda | mps | cpu | auto
    --n_folds 10 \            # pooled stratified 10-fold (headline protocol)
    --max_epochs 100 \
    --n_hpo_trials 20 \
    --seed 42
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--run_name NAME` | name the run dir (`results/NAME/`) instead of `run_N` |
| `--device` | `auto` (cuda→mps→cpu), or force `cuda`/`mps`/`cpu` |
| `--skip_profile` | skip FLOPs/latency profiling + ONNX export + robustness |
| `--seed` | global seed (full determinism) |

---

## E. Evaluation protocol variants (pooled vs LOSO)

The headline is **pooled** 10-fold. Switch to **leave-one-site-out** for the rigor number
via the config flag:

```bash
# Pooled (headline) — configs/config.yaml: cross_validation.group_by_site: false
python run_experiment.py --real_data --mri_dir ./abide_processed/mri \
    --gen_dir ./abide_processed/gen --device cuda --n_folds 10 --run_name pooled_10fold

# Leave-one-site-out (rigor) — set group_by_site: true in configs/config.yaml first,
# then:
python run_experiment.py --real_data --mri_dir ./abide_processed/mri \
    --gen_dir ./abide_processed/gen --device cuda --n_folds 10 --run_name loso
```

---

## F. Figures & report

Paper figures (scienceplots-styled) are generated automatically in the run
(`results/run_N/paper_figures/`) and an HTML report at
`results/run_N/experiment_report.html`. To regenerate the standalone metric plots from a
run's evaluation JSON:

```bash
python results/generate_plots.py     # reads results/evaluation/evaluation_report.json
```

---

## G. Tests

```bash
pytest                               # pytest-style suites (config, preprocessing)
python tests/test_fusion.py          # script-style module checks (run individually)
```

---

## H. Planned accuracy experiments (see publication_plan.md §4)

These are the next builds toward the 80–85% target (not yet wired into `run_experiment.py`):

- **E1.3** tangent-space FC + linear baseline (nested CV) — *reference number*
- **E2.x** connectome transformer + multi-atlas + SSL pretraining + ensemble
- **E3.x** component ablations · **E5.x** permutation / DeLong / bootstrap
- **E6.x** explainability (edge importance → Yeo networks → neurobiology)
- **E7.x** external validation on ABIDE-II

Commands for these will be added here as each experiment lands.
