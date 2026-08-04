# Publication Plan — A Reliable, Interpretable, Cross-Site ASD Connectome Biomarker (ABIDE-I)

**Working title:** *A reliable, interpretable, cross-site autism biomarker from resting-state
functional connectivity with calibrated selective prediction.*

**Target:** Q1 venue (Medical Image Analysis / NeuroImage / IEEE J-BHI / Human Brain Mapping).

---

## 0. Thesis & honest framing (read first)

On the **full** ABIDE-I cohort (~1000 subjects) with a **leakage-free, site-aware** protocol,
classification accuracy is capped around **~0.80 AUROC**, and — critically — **deep models do
not beat a well-tuned linear model** on functional connectivity (FC). This is not our opinion;
it is the replicated finding of independent rigorous benchmarks (Luo et al. 2025, arXiv
2501.17207: Logistic Regression ≈ MLP ≈ BrainNetTF ≈ 0.737, message-passing *hurts*;
PMC11912182: SVM ≈ GCN, no significant difference). We reproduced it: our multi-atlas
self-supervised connectome **transformer underperformed at ~0.62 AUROC**, while a plain
**tangent-FC linear model reaches ~0.77**.

**Therefore the contribution is NOT "highest accuracy"** (that road is leakage). It is a
**reliable, interpretable, deployable biomarker**, plus the honest methodological finding that
simple ≥ complex here. Novelty lives on the axes of **reliability (calibrated selective
prediction), cross-site generalization, and interpretability** — not the leaderboard.

| Reference point | Value | Note |
|---|---|---|
| Heinsfeld 2018 (baseline to beat) | 0.70 acc pooled, ~0.65 LOSO | full cohort, citable |
| Honest full-cohort ceiling (fair CV) | **~0.80 AUROC** | deep = linear (2501.17207) |
| Our linear baseline (real, leakage-free) | **0.756–0.77 AUROC** pooled | already at the ceiling |
| Our headline model target | **~0.79–0.81 AUROC** pooled | multi-atlas linear + phenotype |
| Reliability headline (novel) | **~0.90 AUROC at ~50% coverage** | selective prediction |

**Integrity rule:** every data-dependent transform (tangent reference, scaler, feature
selection, ComBat, calibration) is fit **inside the training fold only**. Site is **never** an
input feature (it leaks label prevalence). We report **pooled 10-fold AND leave-one-site-out**.

---

## 1. Contributions

1. **Calibrated selective prediction for ASD (novel core, Option A).** Report performance *at
   coverage*: with in-fold probability calibration and an abstention rule, the model achieves
   **high AUROC on the most-confident subjects** and declines to predict on the rest — turning a
   population-limited ~0.77 classifier into a clinically-usable, high-precision tool. Selective
   prediction + calibration is under-explored on ABIDE.
2. **Cross-site generalization treatment.** Systematic characterization of the **pooled↔LOSO
   gap** (which sites/subjects fail, and why), with honestly-reported harmonization/reliability
   remedies. The gap itself is a finding most "SOTA" papers hide.
3. **Stable, interpretable connectome biomarker.** Directly interpretable linear tangent-FC edge
   weights → cross-fold-**stable** discriminative connections → aggregated to brain networks →
   **validated against known ASD neurobiology** (default-mode, salience, cortico-cerebellar).
4. **Mechanistic "why simple wins" (Option C, support).** Show the ASD-discriminative signal
   lives in a **low-rank linear subspace** of tangent-FC (quantified via SVD/effective rank),
   explaining why nonlinear models add no value — a scientific *explanation*, not a number.
5. **A rigorous, reproducible, leakage-free benchmark** (pooled + LOSO, everything in-fold, fixed
   seeds, released code) + a validated **GPU-accelerated tangent-FC** implementation, on which a
   multi-atlas linear ensemble + phenotype late-fusion matches or beats deep models.

---

## 2. Data & preprocessing (DONE)

- ABIDE-I, C-PAC, `filt_noglobal`, atlases **CC200 (+ AAL, Harvard-Oxford)**, ~1030 subjects with
  usable data. FC = per-subject ROI time series → covariance.
- **Tangent-space FC** (Ledoit-Wolf covariance → Riemannian tangent at a **train-fold** reference),
  computed by `gpu_tangent.py` (validated numerically identical to nilearn; GPU-accelerated).
- 6 phenotypic features (age, sex, FIQ, VIQ, PIQ, handedness) — used via late-fusion, **not site**.
- Splits: stratified; connectivity/scaling/selection/ComBat/calibration all fit in-fold.

## 3. Model (DONE)

**Multi-atlas tangent-FC linear ensemble + phenotype late-fusion.** Per atlas: in-fold tangent-FC →
standardize → ℓ2-logistic with inner-CV `C`. Soft-vote across atlases (+ a phenotype-only logistic
member). Interpretable, fast, at the honest ceiling. `train_sota.py --model logreg --phenotype`.

*(The connectome transformer + SSL are retained only as the reported **negative-result** baseline
that motivates "simple ≥ complex"; they are not the proposed model.)*

## 4. Evaluation protocol

Nested, subject-independent, everything in-fold. **Headline: pooled stratified 10-fold. Rigor:
leave-one-site-out.** Metrics: AUROC (primary), accuracy, sensitivity, specificity, F1, MCC —
mean±std across folds + **bootstrap 95% CI**. Significance: **DeLong** (AUROC vs baselines),
**McNemar** (accuracy), **permutation test**, vs majority-class + permutation-null baselines.

---

## 5. Experiment matrix

| ID | Experiment | Status | Feeds |
|---|---|---|---|
| **E1** | Baselines: Pearson-SVM, **tangent-linear** (0.77), Heinsfeld-AE reimpl | ✅ tangent; ⧗ AE | Table 1 |
| **E2** | **Proposed:** multi-atlas tangent-linear + phenotype, pooled+LOSO | ✅ | Table 2 |
| **E3** | **Selective prediction:** calibration + risk–coverage curves + AUROC@coverage | ⧗ **build** | Fig, Table (novel) |
| **E4** | **Site generalization:** pooled↔LOSO gap, per-site table, harmonization remedies | ⧗ partial | Table, Fig |
| **E5** | **Interpretability biomarker:** stable edges → Yeo networks → neurobiology validation; connectome glass-brain | ✅ edges; ⧗ networks/brain | Figs |
| **E6** | **Low-rank analysis:** effective rank of discriminative subspace; acc-vs-rank | ⧗ **build** | Fig (why simple wins) |
| **E7** | **Stats:** DeLong / McNemar / permutation / bootstrap CIs | ✅ DeLong+McNemar | Table footnotes |
| **E8** | **Negative-result ablation:** transformer/GNN vs linear (deep ≤ linear) | ✅ (0.62 logged) | Table/Fig |
| **E9** | **External validation:** train ABIDE-I → test **ABIDE-II** | ⧗ (needs download) | Table |
| **E10** | **Leakage audit:** quantify what leaky feature-selection *would* have bought | ⧗ small | rigor Fig |

⧗ = to build. The three new modules: `selective_prediction.py` (E3), `lowrank_analysis.py` (E6),
biomarker/brain figure extension of `analyze_edges.py` (E5).

---

## 6. Codebase status

**Built & working:** `run_benchmark.py` (E1), `gpu_tangent.py` (validated), `train_sota.py`
(`--model logreg --phenotype` = the proposed model; `--model transformer` = negative-result),
`analyze_edges.py` (stable edges), `evaluation/stat_compare.py` (DeLong+McNemar, validated),
`viz_results.py` + `viz_compare.py` (Q1 scienceplots figures + SOTA comparison), per-run directory
+ checkpoint export.

**To build:** `selective_prediction.py`, `lowrank_analysis.py`, brain/network XAI figure,
Heinsfeld-AE baseline, ABIDE-II external eval.

---

## 7. Final run commands

```bash
# 1. Baseline table (E1): correlation vs tangent, pooled + LOSO
python run_benchmark.py --model logreg --metrics correlation tangent --protocol both --n_folds 10

# 2. THE MODEL (E2): multi-atlas tangent-linear + phenotype fusion, pooled + LOSO
python train_sota.py --model logreg --atlases cc200 aal ho --phenotype \
    --protocol both --n_folds 10 --run_name final
#    -> results/final/sota/{sota_report.json, checkpoints/, figures/}

# 3. Interpretability biomarker (E5)
python analyze_edges.py --n_folds 10

# 4. (after build) selective prediction (E3) + low-rank analysis (E6)
python selective_prediction.py --oof results/final/sota/sota_oof.json
python lowrank_analysis.py
```

---

## 8. Paper structure & figures

- **Fig 1** method overview on real ABIDE-I data (fMRI → 200 ROIs → FC → tangent → linear →
  prediction) · **Fig 2** ROC/PR pooled+LOSO vs baselines · **Fig 3** **risk–coverage /
  AUROC@coverage** (novel) · **Fig 4** per-site heatmap + pooled↔LOSO gap · **Fig 5** glass-brain
  connectome of stable discriminative edges + Yeo-network matrix · **Fig 6** effective-rank /
  acc-vs-rank (why simple wins) · **Fig 7** vs-SOTA comparison (honest, best legitimate methods) ·
  **Fig 8** external ABIDE-II calibration.
- **Tables:** baselines · main results (pooled+LOSO) · deep-vs-linear ablation · external.

## 9. Target venues
Medical Image Analysis · NeuroImage · IEEE J-BHI · Human Brain Mapping · NeuroImage: Clinical.

## 10. Key references
- Luo et al. 2025, *Rethinking Functional Brain Connectome Analysis* — arXiv 2501.17207 (deep = linear; message-passing hurts).
- Rigorous ABIDE classifier comparison — PMC11912182 (SVM ≈ GCN).
- Dadi et al. 2019, *NeuroImage* — tangent-space FC is the best parametrization.
- Abraham et al. 2016/2017 — reproducible tangent-FC pipeline, inter-site evaluation (~0.67).
- Heinsfeld et al. 2018 (PMID 29034163) — baseline 0.70.
- MADE-for-ASD 2024, arXiv 2407.07076 — multi-atlas + demographics ablation (levers that pay).
- Kim et al. 2021, *Sci. Rep.* s41598-021-87157-3 — feature-selection leakage inflation.
- Ferrari et al. 2023, PMC10676338 — ComBat must be fit train-only.
