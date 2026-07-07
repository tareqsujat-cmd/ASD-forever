# Publication Plan — ASD Classification on ABIDE-I (Q1 target)

**Goal:** A Q1-publishable paper that *honestly* beats the ABIDE-I state of the art on
resting-state fMRI functional connectivity (FC), pushing pooled accuracy toward
**80–85%**, with rigorous evaluation and clinically-grounded explainability.

**Status (living doc):** data ready (871 subjects, CC200 FC + 6 phenotypics), pipeline
hardened (per-run dirs, full reproducibility, MPS acceleration, leakage-controlled,
fabricated outputs removed). Protocol committed: **pooled nested 10-fold headline +
leave-one-site-out (LOSO) for rigor.**

---

## 0. The honest target (read this first)

| Number | Protocol | Verdict |
|---|---|---|
| **Baseline to beat** | Heinsfeld 2018 = **70%** (pooled 10-fold), ~65% LOSO | citable, full cohort |
| **Credible ceiling (full cohort)** | MADE-for-ASD 2024 = **75.2%** pooled; METAFormer = **83.7%** (882-subj subset, multi-atlas, SSL) | top edge of legitimate |
| **Our headline target** | **80–85% pooled nested 10-fold** | stretch, reachable with full stack |
| **Our rigor number** | **~68–72% LOSO** | report alongside; the gap is a contribution |

**Hard rule for 85% to be real, not retracted:** every data-dependent transform
(tangent reference mean, ComBat, scaler, any feature selection, PCA) is `fit` on the
**training fold only** and `transform`-applied to validation/test, *inside* the CV loop.
Feature-selection-outside-CV is exactly what earned a 98% ABIDE paper an Expression of
Concern (eClinicalMedicine 2026). We never do it.

**Why a number alone won't get accepted:** Q1 venues reject "we got high accuracy," and
high accuracy alone now reads as suspicious. Acceptance needs **four legs**:
1. **Novel method** (multi-view multi-atlas connectome transformer + SSL).
2. **Rigorous evaluation** (pooled + LOSO, nested, CIs, permutation, per-site).
3. **Interpretability tied to ASD neurobiology** (the credibility multiplier).
4. **Reproducibility** (clean per-run pipeline, fixed seeds, released code) — *done*.

---

## 1. Dataset & preprocessing (DONE — verify only)

- **Cohort:** ABIDE-I, CPAC pipeline, `filt_noglobal`, **CC200** atlas, **871 subjects**
  (403 ASD / 468 TC), 20 sites. Downloaded via `data/download_abide.py`.
- **Features per subject:** 19,900-d FC (upper-triangle Pearson→Fisher-z) in
  `abide_processed/mri/`, 6 phenotypics (age, sex, FIQ, VIQ, PIQ, handedness) in
  `abide_processed/gen/`. No NaNs; train-only `RobustScaler` on phenotypics.
- **Splits:** `metadata.csv` has stratified `train/val/test`; CV re-splits within the
  pooled cohort. Raw ROI time series retained in `abide_raw/` (needed for tangent-FC and
  multi-atlas).
- **TODO-data:** also download **AAL** and **Dosenbach-160** ROI series (multi-atlas);
  cache motion (mean FD) per subject from ABIDE phenotypics (ComBat covariate).

---

## 2. Method / architecture to build

The current model (3D-ResNet on a fake 28³ reshape of the FC vector) is a placeholder and
is replaced. Target architecture, built in ranked order of evidence:

| # | Component | Rationale | Expected Δ (pp) |
|---|---|---|---|
| 1 | **Tangent-space FC** (Ledoit-Wolf covariance → Riemannian tangent, per-fold reference) | Best FC parametrization (Dadi 2019) | +2–5 |
| 2 | **Connectome Transformer** (ROI-token self-attention; BrainNetTF/METAFormer lineage) | Purpose-built for FC; interpretable | core novelty |
| 3 | **Multi-atlas** (CC200 + AAL + Dosenbach-160), late-fused | Complementary parcellations | +2–4 |
| 4 | **Self-supervised pretraining** (masked-connectome reconstruction, unlabeled) | Biggest DL lever; leakage-safe | +2–4 |
| 5 | **Population-graph GNN** view (phenotypic-similarity edges; **never site as feature**) | Complementary view → ensemble | +1–3 |
| 6 | **Cross-modal fusion** connectome + phenotypics (reuse repo cross-attention) | Principled multimodal | +0–2 |
| 7 | **Ensemble** across atlases + seeds + views (soft-vote) | Most reliable route past 72% | +2–4 |
| + | **In-fold ComBat** harmonization (age/sex/motion covariates) | Removes site confound (helps DL) | +2–5 (DL) |

**One-line contribution:** *"A multi-view, multi-atlas connectome Transformer with
self-supervised masked-connectome pretraining and cross-attention phenotypic fusion,
evaluated under pooled and leave-one-site-out protocols, with connectome-level
explainability validated against ASD neurobiology."*

---

## 3. Evaluation protocol (locks legitimacy)

- **Headline:** pooled, subject-independent, **nested** stratified **10-fold** (inner loop
  tunes hyperparameters + any selection; outer loop evaluates). Repeat 10× with different
  seeds; report mean of means to kill fold-variance.
- **Rigor:** **LOSO** (20 site-folds) + grouped-nested CV.
- **Metrics:** AUROC (primary), accuracy, sensitivity, specificity, F1, MCC, balanced
  accuracy — each **mean ± std** across folds with **bootstrap 95% CI**.
- **Significance:** **permutation test** (shuffle labels, rerun the *entire* nested
  pipeline ≥1,000×) → p-value; **DeLong test** for AUROC vs each baseline; **McNemar** for
  accuracy vs baseline. Report the **majority-class** and **permutation-null** baselines.
- **Everything fit in-fold.** No transform sees validation/test data.

---

## 4. THE EXPERIMENT MATRIX (what to run for acceptance)

Grouped by purpose. Each row: run it, log to `results/run_N/`, feed the named table/figure.
"P" = pooled 10-fold, "L" = LOSO.

### E1 — Baselines (establish the floor; Table 1)
| ID | Experiment | Protocol | Feeds | Expected |
|---|---|---|---|---|
| E1.1 | Majority-class + permutation null | P,L | Table 1 | ~51% / chance |
| E1.2 | Linear SVM on **Pearson** FC | P,L | Table 1 | ~65–68% |
| E1.3 | ℓ2-Logistic / Linear SVM on **tangent** FC (in-fold ref) | P,L | Table 1, Fig ROC | ~70–73% |
| E1.4 | Random forest on FC | P | Table 1 | ~60–66% |
| E1.5 | **Heinsfeld AE + MLP reimplementation** (the paper we beat) | P,L | Table 1 | ~70% |
| E1.6 | Current repo model (3D-ResNet-on-28³ FC + cross-attn) | P | Table 1 (ablation ref) | ~68–72% |

### E2 — Proposed model & the climb (Table 2, main result)
| ID | Experiment | Protocol | Feeds | Expected |
|---|---|---|---|---|
| E2.1 | Connectome Transformer, single atlas (CC200), tangent FC | P,L | Table 2 | ~73–76% |
| E2.2 | + Self-supervised pretraining (masked connectome) | P,L | Table 2 | ~75–78% |
| E2.3 | + Multi-atlas (CC200+AAL+DOS160) late fusion | P,L | Table 2 | ~77–82% |
| E2.4 | + In-fold ComBat harmonization | P,L | Table 2 | +1–3 |
| E2.5 | + Cross-modal phenotypic fusion | P,L | Table 2 | +0–2 |
| E2.6 | + Population-GNN view, **soft-vote ensemble** (final model) | P,L | Table 2, Fig ROC | **80–85% (P)** |
| E2.7 | Final model, **10× repeated** 10-fold (variance) | P | Table 2 | mean ± std |

### E3 — Ablations (prove each component earns its place; Table 3)
Run the final model minus one component at a time (leave-one-out):
| ID | Ablation | Isolates |
|---|---|---|
| E3.1 | Pearson vs tangent FC | connectivity metric |
| E3.2 | single- vs multi-atlas | atlas fusion |
| E3.3 | no-SSL vs SSL pretraining | pretraining |
| E3.4 | no-ComBat vs in-fold ComBat | harmonization |
| E3.5 | FC-only vs +phenotypic fusion | multimodality |
| E3.6 | single model vs ensemble | ensembling |
| E3.7 | fusion strategy sweep (cross-attn / gated / late / concat) | fusion design |
| E3.8 | **leakage control**: ComBat/selection in-fold vs whole-dataset | *shows the inflation we avoid* |

> E3.8 is a deliberate, honest demonstration: run the pipeline the *wrong* (leaky) way to
> quantify how many points of accuracy leakage buys — a reviewer-proofing figure.

### E4 — Protocol / generalization (Table 4 + per-site)
| ID | Experiment | Feeds |
|---|---|---|
| E4.1 | Pooled vs LOSO gap for final model | Table 4, headline discussion |
| E4.2 | Per-site accuracy/AUROC table (LOSO) | Fig site-heatmap |
| E4.3 | Site-confound probe: predict *site* from features; predict dx from ComBat-residuals | rigor appendix |
| E4.4 | Motion confound: correlate mean-FD with prediction; control via covariate | rigor appendix |

### E5 — Statistical rigor (Table 1–2 footnotes)
| ID | Experiment |
|---|---|
| E5.1 | Bootstrap 95% CIs on all headline metrics |
| E5.2 | Permutation test (≥1000×) → p-value for final model |
| E5.3 | DeLong AUROC test: final vs each baseline |
| E5.4 | McNemar accuracy test: final vs Heinsfeld reimpl |

### E6 — Explainability (Section + Figs; leg 3)
| ID | Experiment | Feeds |
|---|---|---|
| E6.1 | Attention rollout over ROI tokens → salient ROIs | Fig brain-saliency |
| E6.2 | Integrated gradients on FC edges → top discriminative connections | Fig connectome-circos |
| E6.3 | Aggregate edges to Yeo-7/17 networks → network-level ASD signature | Fig network-matrix |
| E6.4 | Cross-fold **stability** of top-k edges (Jaccard/rank-corr) | Table stability |
| E6.5 | **Neurobiology validation**: compare top networks to ASD literature (DMN hypoconnectivity, salience, cortico-cerebellar) | Discussion |
| E6.6 | SHAP on phenotypic features (age/sex/IQ contribution) | Fig pheno-importance |

### E7 — External validation (strengthens generalization claim)
| ID | Experiment | Feeds |
|---|---|---|
| E7.1 | Train on full ABIDE-I → evaluate on **ABIDE-II** (true external set) | Table 5 |
| E7.2 | Calibration (reliability curve, ECE) on external set | Fig calibration |

### E8 — Clinical subgroup analyses (optional, boosts clinical relevance)
Stratify final-model performance by **age band, sex, IQ, symptom severity** (ADOS/ADI where
available — for *analysis only, never as input features*).

---

## 5. Phased execution & milestones

| Phase | Work (experiments) | Milestone | Target (P) |
|---|---|---|---|
| **P0** | Repo hardening + data | reproducible pipeline, honest baseline | done / ~70% |
| **P1** | E1.1–E1.6 baselines + nested-CV/LOSO harness + tangent-FC | Table 1 done | ~70–73% |
| **P2** | E2.1–E2.4 (transformer, SSL, multi-atlas, ComBat) | main model working | ~77–82% |
| **P3** | E2.5–E2.7, E3 ablations, E5 stats | Tables 2–3 done | **80–85%** |
| **P4** | E4 protocol, E6 XAI, E7 external, E8 subgroups | Tables 4–5, XAI figs | (paper-ready) |
| **P5** | Writing, figures, polishing, submission | manuscript | — |

---

## 6. Paper structure (figures & tables)

- **Fig 1** architecture · **Fig 2** dataset/pipeline · **Fig 3** ROC/PR (proposed vs
  baselines) · **Fig 4** ablation bars · **Fig 5** connectome-circos of top edges · **Fig 6**
  Yeo-network importance matrix · **Fig 7** per-site heatmap (LOSO) · **Fig 8** calibration
  (external) · **Fig S\*** attention maps, stability, HPO, decision curve.
- **Table 1** baselines · **Table 2** main results (pooled+LOSO) · **Table 3** ablations ·
  **Table 4** protocol/generalization · **Table 5** external ABIDE-II.
- **Sections:** Intro · Related work · Data · Method · Experiments · Results · Interpretability ·
  Discussion (incl. pooled↔LOSO gap, limitations, leakage-avoidance) · Conclusion.

---

## 7. Target Q1 venues

Medical Image Analysis · NeuroImage · IEEE Trans. Medical Imaging · IEEE J-BHI · Human Brain
Mapping · Neuroimage: Clinical (the Heinsfeld venue). Method+interpretability framing fits
MedIA/TMI; neuroscience framing fits NeuroImage/HBM.

---

## 8. Reviewer-proofing checklist (acceptance gate)

- [ ] Headline number is **pooled nested 10-fold**, all transforms in-fold, seeds fixed.
- [ ] **LOSO** reported; pooled↔LOSO gap discussed openly.
- [ ] **Permutation p-value + bootstrap CIs** on every headline metric.
- [ ] **DeLong/McNemar** vs baselines (incl. Heinsfeld reimpl).
- [ ] **Leakage-avoidance** explicitly described; E3.8 quantifies what leakage would have added.
- [ ] **No site ID / diagnostic (ADOS/ADI) features** used as model inputs.
- [ ] **Interpretability validated** against ASD neurobiology, with cross-fold stability.
- [ ] **External ABIDE-II** validation.
- [ ] **Code + configs + seeds released**; per-run manifests included.
- [ ] Any number >78% (P) audited for leakage before it goes in the paper.

---

## 9. Key references (from the research council)

- Heinsfeld 2018, *NeuroImage:Clinical* (PMID 29034163) — baseline 70%.
- Abraham 2017, *NeuroImage* (arXiv:1611.06066) — 67%, reproducible pipeline.
- Dadi 2019, *NeuroImage* (PMID 30836146) — tangent-space FC best.
- Parisot/Ktena 2018, *Med. Image Anal.* — population-GCN 70.4%.
- MADE-for-ASD 2024 (arXiv:2407.07076) — 75.2% multi-atlas ensemble.
- METAFormer 2023 (arXiv:2307.01759) — 83.7% multi-atlas + SSL transformer.
- Kim 2021, *Sci. Rep.* (s41598-021-87157-3) — feature-selection leakage inflation.
- Ferrari 2023, *Brain Informatics* (PMC10676338) — ComBat must be fit train-only.
- Traut 2022, *NeuroImage* — blinded challenge ceiling AUC ~0.80.
