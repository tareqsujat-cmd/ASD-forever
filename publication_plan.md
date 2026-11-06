# Publication Plan — Geometry-Aware, Rank-Structured and Risk-Controlled ASD Connectome Classification (ABIDE-I)

*Last updated: 2026-08-05*

---

## 0. Framing (read first)

The paper is a **methods paper on ABIDE-I**, not an accuracy-leaderboard entry. Three
methodological contributions are introduced, each ablated in isolation, and the whole is
evaluated under a protocol in which every learned step is fit inside the training fold.

Two facts set the target. First, rigorously evaluated work on ABIDE-I reports roughly
**0.79–0.81 AUROC / 76–80% accuracy** (CAS-GNN 79.25%, GNN-LSTM 80.4%); our current
multi-atlas baseline sits at **0.761 AUROC / 0.699 accuracy**, so there is genuine
headroom of about +0.03–0.05 AUROC. Second, the reported range in the wider literature
extends to 98%, which is not reproducible under leakage-free evaluation; quantifying that
gap is itself one of our experiments.

**Target:** superseded. Chasing accuracy on this benchmark is the framing this work argues
against: the achievable ceiling implied by disagreement between the two diagnostic
instruments is ~0.95, the observed figure is ~0.71, and correcting for label noise closes
only about two points of that gap. The claim is not a higher number but a correctly
regularised and honestly evaluated one. Current: AUROC 0.775 / accuracy 0.707 (871),
0.791 / 0.707 (1,032), on the averaged-probability ensemble.
No claim above that range will be made.

---

## 1. Methodological contributions

### N1 — Riemannian dynamic-FC trajectory features
Existing dynamic-FC pipelines slide a window, compute Pearson correlations and cluster
windows into discrete states, all in Euclidean space. We instead treat each participant
as a **trajectory on the SPD manifold** and summarise it with three geometry-native
statistics, each projected to the tangent space at a training-fold reference: the
**Fréchet mean** (where connectivity sits), the **dispersion** (how variable it is) and
the **geodesic velocity** (how fast it reconfigures). The mean block reproduces the
static representation, so dispersion and velocity isolate the dynamic contribution.
Motivated by reports of atypical network flexibility in autism.
*Implementation:* `dynamic_fc.py`. **Outcome: not supported.** Dispersion and velocity both
lowered AUROC (−0.026 and −0.027 alone, −0.048 together, worse again under decision-level
fusion). The surviving mean block is not separable from the static representation
(+0.010, DeLong p = 0.281) — and once the whole-scan baseline receives the same shrinkage
the windowed path applies to its short windows, the difference reverses to −0.0025
(p = 0.709). What looked like a dynamics effect was an asymmetry in regularisation.
Report as a negative result and as the control the comparison required.

### N2 — Low-rank spectral tangent classifier
A linear model on vectorised tangent features learns one weight per edge and discards
the fact that the features are the upper triangle of a symmetric matrix. Writing the
decision as `f(S) = <W, S>` and constraining `W = Σ λ_k u_k u_kᵀ` reduces the parameter
count from 19,900 to ≈rP, and each `u_k` becomes an interpretable whole-brain
**eigen-connectome**. **The premise did not survive measurement.** The weight matrix requires 131 components for
90% of its spectral mass, so it is not low-rank; the earlier participation-ratio figure
described the data, not the fitted weights.
*Implementation:* `models/lowrank_spectral.py` (two-stage spectral truncation + refit;
optional jointly optimised variant). *Verified:* at r=50 it **matches** the dense model
(0.733 vs 0.735 AUROC, 3-fold smoke test) with half the parameters. The claim is
parameter efficiency and interpretability at equal accuracy — **not** an accuracy gain.

### N3 — Site-conditional conformal risk control
The previous selective predictor thresholded `max(p, 1−p)` heuristically, with no
guarantee on the error rate among answered cases and no notion of site. We replace it
with **selective risk control**: a threshold is calibrated so that a Clopper–Pearson
upper bound on the error among answered cases is ≤ α in finite samples, either marginally
or **per site** (site as the Mondrian taxonomy). For a site with no labels, the marginal
threshold is inflated in proportion to that site's Riemannian distance from the training
reference — a label-free shift score.
*Implementation:* `conformal_selective.py`. *Verified on real OOF:* marginal coverage
0.652 at realised risk 0.225 with bound 0.249 ≤ α=0.25; Mondrian reaches 0.837 selective
accuracy; under LOSO the bound still holds (0.202). It correctly **abstains entirely on
sites too small to certify** (CMU n=11, CALTECH n=15).
*Positioning:* an adaptation of conformal risk control to cross-scanner neuroimaging, not
a new conformal theory — the 2026 literature on conformal-under-shift is active.

---

## 2. Experiment matrix

All experiments emit **JSON (full detail, per-fold) and CSV (flat, one row per
condition)** into `results/<timestamp>_ablations/ablations/`, so figures never require a
re-run.

| # | Experiment | Question | Runner |
|---|-----------|----------|--------|
| 1 | **Representation** | correlation → tangent → +dispersion → +velocity | `run_ablations.py --experiments representation` |
| 2 | **Classifier** | dense linear vs low-rank spectral vs ridge/SVM/LDA vs RF/XGB/CatBoost | `--experiments classifier` |
| 3 | **Rank sweep** | where does the eigen-connectome saturate? | `--experiments rank` |
| 4 | **Atlas** | each atlas, every pair, all three | `--experiments atlas` |
| 5 | **Fusion** | probability averaging vs stacking vs feature concatenation | `--experiments fusion` |
| 6 | **Harmonisation** | none vs in-fold ComBat | `--experiments harmonisation` |
| 7 | **Leakage audit** ⭐ | how much does each common shortcut inflate accuracy? | `leakage_audit.py` |
| 8 | **Protocol** | pooled vs LOSO, with conformal abstention | `--experiments protocol` |
| 9 | **Statistics** | DeLong / McNemar between headline conditions | `--experiments statistics` |

**Experiment 7 is a contribution in its own right.** Seven variants share one model and
one dataset and differ by a single shortcut each — scaler fit on all data, tangent
reference fit on all data, feature selection outside the folds, hyperparameters tuned on
the test fold, site as a predictor, and all of them combined. It explains the spread in
the published literature and reframes a modest honest number as the correct one.

---

## 3. Established results carried forward

| Finding | Value |
|---------|-------|
| Multi-atlas tangent-linear + phenotype | AUROC 0.761 pooled / 0.757 LOSO |
| Selective prediction (heuristic) | 0.833 AUROC, 0.805 accuracy at 50% coverage |
| Calibration | ECE 0.102 pooled / 0.092 LOSO |
| Linear vs trees | 0.755 vs 0.609–0.623 — decisive |
| Low-rank structure | AUROC saturates at rank ≈50; participation ratio 12.5 |
| Interpretability | default-mode-centred; SHAP agrees with coefficients at r=0.996 |
| ComBat harmonisation | **negative result** — worse AUROC and sensitivity, did not fix weak sites |
| Weak sites | MAX_MUN ≈0.51, OHSU ≈0.50 under LOSO — reported as a limitation |

---

## 4. Run order

```bash
# Experiment 7 — leakage audit (standalone, ~30 min)
python leakage_audit.py --n_folds 10 --out results/ablations

# Experiments 1-3 — the novelties (representation, classifier, rank)
python run_ablations.py --experiments representation classifier rank --n_folds 10

# Experiments 4-6, 8-9 — design justification and protocol
python run_ablations.py --experiments atlas fusion harmonisation protocol statistics --n_folds 10

# or everything at once
python run_ablations.py --all --n_folds 10
```

Figures are produced from the emitted CSVs; the model run itself is unchanged
(`train_sota.py --model logreg --atlases cc200 aal ho --phenotype --protocol both`).

---

## 5. Paper structure

1. **Introduction** — cross-site ASD classification; reliability over leaderboard.
2. **Methods** — `METHODOLOGY.md` (dataset, tangent embedding, harmonisation, fusion,
   classification, calibration, selective prediction, interpretation), extended with N1–N3.
3. **Experiments** — the nine above.
4. **Results** — `results.md`, regenerated from the ablation CSVs.
5. **Discussion** — why linear/low-rank suffices; what the leakage audit implies for the
   field; weak sites as an open problem.
6. **Limitations** — single dataset (ABIDE-II external validation not run: no
   pre-extracted ROI release exists); two sites fail; cross-pipeline transfer untested.

---

## 6. Target venues

| Venue | Fit | Note |
|-------|-----|------|
| **NeuroImage: Clinical** | strong | Rolling submission; published the Heinsfeld baseline |
| **Medical Image Analysis / IEEE TMI** | plausible with N1–N3 | Methods contributions lift it into range |
| **MICCAI** (main track) | plausible | ~30% accept; lead with the methods, not the accuracy |
| **MICCAI-MLCN workshop / MIDL / ISBI** | strong | Lower bar, respected |
| **CHIL / ML4H** | strong | Calibration and abstention are their currency |
| ICLR / NeurIPS / CHI | **no** | Insufficient ML novelty; no human-subjects study |

---

## 7. Honest positioning

Claim: a geometry-aware, rank-structured, risk-controlled pipeline for cross-site ASD
classification, evaluated without leakage, with an interpretable eigen-connectome basis
and finite-sample abstention guarantees.

Do **not** claim: state-of-the-art accuracy; that ComBat helped; that the low-rank model
improves accuracy (it matches at fewer parameters); that conformal prediction is novel in
general; or external validation on ABIDE-II.
