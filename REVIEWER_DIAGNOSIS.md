# Reviewer Diagnosis — what the paper is, what survives, what to cut

Working document. Source-of-truth for the edits to `METHODOLOGY.md` and `RESULTS_SUMMARY.md`
before the first draft. Written from a hostile-reviewer standpoint: every item here is
something a referee can check against our own stored tables.

---

## 0. The title

> **What Survives When Autism Classifiers Are Compared Fairly**

Plain words, names the subject, promises constructive content rather than an attack, and is
honest about both halves of the paper — the things that die and the things that live. It also
commits us publicly to the structure, which is useful discipline: if a fourth "novelty" creeps
back in during drafting, the title will not fit it.

Alternates, in order: *Three Effects in Autism Brain Imaging That Vanish Under Their Controls*;
*Why Brain Scans Predict an Autism Diagnosis but Not Its Severity* (this one is the title of a
different, narrower paper — hold it, see §7); *The Comparison Is Making the Finding*.

---

## 1. The one sentence the paper is defending

> Three effects reported in the resting-state fMRI autism literature — the dynamic-connectivity
> advantage, the correlation between classifier score and symptom severity, and the accuracy
> attributable to feature selection — are properties of how the comparison was constructed
> rather than of the brain; each disappears when run against its own control, and the residual
> gap between 0.71 accuracy and perfect accuracy is not explained by diagnostic label noise
> either.

The method that makes this a study rather than an opinion piece goes in the Introduction
explicitly: **in each case we first reproduce the published effect at its published magnitude on
our own data, then show it vanish under its control.** That sentence is the reason it is
publishable.

Two things that must *not* be folded into that sentence: the ensemble-scoring defect is a bug in
*our* pipeline, not a correction to the literature, and the abstention result is a limitation of
the data (twenty sites impose a resolution floor of 1/(K+1) ≈ 0.048). Both belong in a
self-audit section, not in the field-level claim.

---

## 2. Findings ranked by how well they survive attack

### Tier 1 — headline

**The severity pooling artefact. This is the best result in the paper and currently gets one
paragraph.**
SRS ρ = 0.455 pooled (n = 300) → ρ = 0.061 within-autism (p = 0.44, n = 163). The ADI-R
subscales, administered only to autistic participants, return identical values in both scopes —
a built-in positive control that *must* hold and does. It reproduces Nielsen (2013) and Plitt
(2015) at their published magnitude on their dataset and then dissolves them. It does not depend
on a small AUROC difference, a seed, motion, shrinkage, or sample definition.
*Required correction:* within-autism ADOS total ρ = 0.125, p = 0.036 must be reported as
**failing** BH inside its own six-instrument family (0.036 × 12/4 = 0.108) and as collapsing to
ρ = 0.065, p = 0.279 under LOSO scores in the same stored table. That is the one loose thread in
an otherwise airtight section.

**The two evaluation defects.**
(a) Transductive fold-internal ranking, worth 0.020–0.028 AUROC — and the sole basis of the
0.716 that would have placed this work above published results. (b) An abstention bound fitted
and evaluated on the same participants (0.2496 against α = 0.2500, held by construction).
(c) A threshold search that continued past its first failure; fixed to fixed-sequence, after
which the cohort-wide rule certifies nothing in sample.
*Generalise it — this is not a bug report.* Any score normalised within the test fold is
transductive; under LOSO the fold is an entire site; the quantity is undefined for one incoming
patient. Clarke et al. (2025) is transductive by their own admission on this dataset, which
gives an external instance and removes the "they only found their own bugs" reading. Zero of
~20 surveyed ABIDE papers report calibration.

**Windowed vs shrinkage-matched static — restated as *equivalence*, not reversal.**
The reviewer who recommended rejection nevertheless certified this proposition unpublished after
searching for it. From Δ = −0.0025, p = 0.709, SE ≈ 0.0067: **95% CI [−0.016, +0.011]**. The data
exclude a windowed advantage larger than ~0.011 AUROC. |z| = 0.37 is not a reversal, and the
current wording ("a property of the estimator, not evidence that connectivity changes during the
scan") is stronger than the data support.

### Tier 2 — support, not claims

**The shrinkage gain. Demote to an estimate with a CI. Not confirmatory, not in the abstract.**
Nested: +0.0142 [−0.0013, +0.0297], p = 0.0722 pooled; +0.0132 [−0.0025, +0.0289], p = 0.0983
LOSO. Fails our own pre-registered correction under any family size. The nested grid is censored
at 0.50 and selects over a plateau spanning 0.002 on inner folds whose AUROC SE is ~0.03 — that
selection is noise purchase (SE 0.0044 → 0.0079 while the point estimate barely moves).
*Decision:* stop selecting; fix r by a data-independent rule, report the fixed-r comparison as
the method and the nested result as sensitivity. **Never combine the pooled and LOSO p-values** —
Fisher 0.042, Stouffer 0.015, both invalid on the same 871 participants. Any combined p-value
anywhere is a reject on its own.

**Leakage granularity — and reverse the exclusion decision.**
Region-level selection in-fold vs all-data: 0.006, inconsistent sign. Edge-level selection
outside the split: 0.756 → 0.987. That is a dose–response of leakage against selection
granularity, and it is the only available explanation for the 0.82–0.99 band this literature
reports and for why our 0.707 sits below it.
*The ground moved since we decided to exclude it.* Kurkin & Hramov (eClinicalMedicine 2026)
have now published the qualitative claim — "10–30 percentage points" from generic methods
literature. We have the single-variable controlled measurement on ABIDE showing one shortcut
accounts for essentially all of it (fs_outside 0.987 vs all_shortcuts 0.988). Publish it as one
figure panel supporting the granularity finding, explicitly labelled as non-commensurable
quantities on a shared axis. Do **not** publish 0.987 as a standalone number; it will be
stripped of context and mis-cited.
*The "distributed, not focal" framing dies* — Chen et al. (2015) closes with "distributed,
rather than localized," on ABIDE, at top-100. Keep the numbers, drop the framing.

**The label-noise ceiling. Keep the conclusion, delete the number.**
κ on the 2×2 is **−0.078** — the instruments agree *less* than their marginals predict, which
destroys the "two independent raters" premise. The "neither positive" cell is zero by ABIDE's
inclusion rule, so the two-rater latent-class model is not identified (Hui & Walter 1980):
ρ₁ = 0 fits the table as well as ρ₁ = 0.075. `rho1 = disagree/2.0` is an uninstrumented halving
worth ±0.035. `SRS_RAW_TOTAL >= 60` applies a **T-score cutoff to a raw total** running 0–164,
pooled across two SRS versions with medians 43 and 68.
*Write:* "under the stated assumptions, accuracy attainable against these labels lies between
roughly 0.85 and 0.99, and the observed 0.71 falls below all of it." At the pessimistic end that
is still a 15-point gap. That sentence is unattackable; 0.9535 is not.
*Concede the clinical objection before a clinical reviewer leads with it:* ADI-R scores
caregiver-reported developmental history, ADOS scores directly observed current behaviour; their
disagreement is documented and structured, varying with age, verbal ability and IQ. The noise is
therefore instance-dependent, not class-conditional, which invalidates a Menon-style correction
as an exact statement. **Delete `label_ceiling_corrected`** rather than defending 0.778 → 0.808
— it is also computed on the withdrawn transductive score.

### Tier 3 — demote

**The network contrast → one short replication section at one-third current length.**
Not new. Ilioska et al., *Biological Psychiatry* 94(1):29–39 (2023), n = 1824, ages 5–58,
ABIDE-I + ABIDE-II + EU-AIMS LEAP, 390 regions — a strict superset of this cohort testing this
exact contrast, reporting sensory and attentional hypoconnectivity. Chen et al. (2015) has the
somatosensory/DMN/visual result on ABIDE at n = 252. **Neither is cited anywhere in the
repository.** A domain reviewer finds both in one search; the absence of the largest existing
analysis of exactly our contrast is grounds for rejection on its own.
*What we can honestly claim:* independent confirmation with site and motion confounds removed,
with a signed, FDR-corrected, within-site-permutation-tested contrast — where Chen's map is
unsigned random-forest importance and cannot state direction at all.
*Cut the globality framing.* The whole-brain mean over 19,110 edges is d ≈ −0.12, p ≈ 0.07;
"34 of 36 negative" is a sign test over massively non-independent blocks.
*Cut the limbic claims entirely.* Seven parcels; two assigned on 22–26% voxel overlap; one is an
insula parcel Yeo-7 calls salience and Harvard–Oxford calls frontal orbital cortex; one straddles
the midline; three siblings were excluded *for zero-variance time series* in the
orbitofrontal/anterior-temporal dropout zone. Limbic also drops out of the top tier in our own
motion-matched run.

**The 21 somatomotor regions → supplement, with the strong part promoted.**
"All 21 significant after FDR" against a null of zero is one finding restated 21 times: each
region's statistic is the mean of its 20 within-block edges, so the 21 statistics share every
edge pairwise. Rank 1 to rank 2 is 0.14 permutation SD; 16 of 21 rows sit at the 1/5001
permutation floor sharing q = 0.000262. The ordering is not resolvable.
*Replace the ranked table with* one pre-specified contrast (postcentral + paracentral +
precentral vs the rest of the system) and one post-hoc enrichment test (6 of top 8 postcentral
against 8 of 21, hypergeometric p = 0.011), both labelled exploratory. **Keep the specificity
nulls in the main text** — 2000 random 21-region blocks give P(≤ −0.2155) = 0, and the most
negative of all 196 compact 21-region neighbourhoods is −0.138. That is the strong part and it
is currently unwritten.

### Tier 4 — one "tested and dropped" table, one line each

Falsified scan-duration scaling law; site homogeneity; ComBat; multi-atlas fusion; dispersion
and velocity; demographic variables; low-rank spectral. Half a page total.

### Tier 5 — cut entirely as contributions

The ensemble (p = 0.0664 / 0.0554 — only the withdrawn rank-mean reaches 0.0288). The low-rank
classifier (premise falsified at 131 components; matches at p = 0.428). The abstention rule as a
deployable system. The clinical-utility claim. The IQ/sex subgroup audit — stays withdrawn,
replaced rather than reinstated.

---

## 3. Two results the panel did not have

**(a) The blocking motion check passed, and the finding got stronger.**
`exp_sensorimotor.py` now carries `--match_motion`, ported from the network-level matcher. On
280 within-site FD-caliper-matched pairs (560 participants) the group difference in head motion
is removed — p = 7.6 × 10⁻⁷ before matching, **p = 0.28 after** — and **21 of 21 regions remain
significant**, with the ordering preserved (rank correlation 0.887 against the unmatched result)
and the effects *larger*: mean −0.2155 → **−0.2489**. Top region Postcentral_R [28] goes
−0.269 → −0.312.

This mirrors the system level, where matching raised significant pairs from 10/36 to 26/36 and
somatomotor from −0.2155 to −0.2925. The effect is found *against* the direction of the
confound: in this no-GSR pipeline FD vs mean edge strength is ρ = +0.51, so motion *inflates*
connectivity while the group difference is a reduction. The honest statement is that residual
motion is not a sufficient explanation — not that it is excluded.

**(b) The fabricated 0.09 is gone, replaced by measured figures.**
The "whole-brain edge lists reproduce at about 0.09" claim was a hard-coded literal, computed
nowhere. Now measured in the same 200 splits, with nuisance coefficients **refit inside each
half** (fitting them once on everyone couples the halves and inflates agreement — this alone
moved 0.564 → 0.546):

| ranking | split-half ρ |
|---|---|
| the 21 regions | **0.55** |
| all 210 within-system connections | 0.38 |
| its top 50 connections | 0.34 |

The edge map is **not** less reproducible. Block-level reporting is justified on multiplicity —
21 comparisons rather than 19,110 — and only modestly on stability. And against a
cardinality-matched comparator, 100 random 21-region blocks scored identically give median
ρ = 0.35 (range 0.06–0.63), with 9% reaching this system's value: more stable than most equally
sized blocks, but not categorically so.

Also fixed: the two hemisphere contradictions (Paracentral_Lobule_L [111], Supp_Motor_Area_L
[123]) — parcels with centroids within 5 mm of the midline are now labelled M rather than forced
to a side.

---

## 4. What to omit — unsentimental list

Everything here invites an attack we cannot answer.

1. **"the improvement is separated from chance by the test"** (METHODOLOGY §2 bullet 1). Attaches
   a fixed-r p-value to the nested procedure; two stored tables contradict it. In a field with
   two 2025–26 ABIDE retractions this is the single most dangerous sentence in the manuscript.
   Delete before anything else.
2. **"raises the AUROC by more than any change of representation examined in this work."** Under
   the nested estimate the ensemble beats correlation by +0.028. False.
3. **"The combination beats the best single description by a margin the test separates from
   chance."** p = 0.0664 / 0.0554. The only version that separates is the score we retract —
   this is the *second* instance of the defect we disclose, and a reviewer who reads our
   disclosure will hunt for exactly it.
4. **"the harmonisation that was tested made accuracy worse."** AUROC p = 0.0208 (q ≈ 0.13);
   McNemar on paired decisions p = 0.083. §12 promises both because they can disagree; here they
   disagree and only one is shown. Rewrite as "no evidence of benefit."
5. **"Certified", and the number 0.9535.** Unidentified across ~[0.85, 0.99].
6. **Any combined p-value across pooled and LOSO.** Non-negotiable.
7. **The "accuracy, confident half" column** and the fig20 caption clause. A 50%-coverage number
   beside three 100%-coverage numbers, computed as a quantile *of the held-out set* —
   structurally identical to the transductive defect we correct four sections earlier. Highest-risk
   item in the results document.
8. **§10 scaling law, entire section.** Duration ρ = +0.93, subjects ρ = +0.89, no contrast; and
   the two axes are not on the same footing (duration fixed at n = 266, subjects running to 825).
   It is a live overclaim still asserting "the binding constraint is time in the scanner."
9. **§8: "so the model is reading brain organisation rather than an artefact."** No Haufe
   transform was applied; a weight map over correlated features licenses no anatomical reading.
10. **§12 opening: "...cannot be produced by autistic participants being scanned elsewhere,
    moving more, or being younger."** The permutation null certifies the p-value on residualised
    edges, not the residualisation itself.
11. ~~The ~0.09 comparator.~~ **Done** — see §3(b).
12. **The superseded three-atlas + phenotype table (0.761/0.699).** Redraw the figures it fed.
13. **The SHAP waterfall figure.** r = 0.996 is true by construction for a linear model. One
    sentence in Methods.
14. **The decision curve as drawn.** π = 0.4627 is the case-control sampling fraction. At
    π = 0.028 the model has negative net benefit everywhere, and the minimum predicted probability
    is 0.061 — it *cannot emit a probability low enough to rule anyone out* at population
    prevalence. Cut, or recompute at π = 0.03/0.10/0.20/0.30 with the case-control correction.
15. **Every sentence implying the model could direct specialist referral**
    (RESULTS_SUMMARY:87, :107, :436; METHODOLOGY:19). At 13.7% coverage, PPV 0.70 and NPV 0.93 at
    π = 0.30, the instrument avoids zero ADOS assessments. If we do not compute that table, a
    reviewer will.
16. **The early-identification motivation** (METHODOLOGY §1 para 2). Minimum age 6.47, exactly
    one participant under 7, median 14.66, and every label is a diagnosis obtained years before
    the scan. We cannot motivate with early screening and evaluate on adolescents and adults.
17. **`publication_plan.md` and `RESEARCH_COUNCIL_FINDINGS.md` as source or supplementary
    documents.** Neither ships in any form. The plan's headers still assert N1/N2 as
    contributions while its bodies retract them; the council findings still carry a withdrawn
    Cochran–Armitage z = −2.15 with no inline withdrawal stamp.
18. **The embedding-flow figure's four-paragraph reading, and the CC200-vs-AAL atlas reversal.**
    Attractive, off-thesis. Supplement.
19. **The nuclear-norm variant.** Never run, code deleted. It must not appear even as future
    work; a reviewer reads that as an unrun ablation.

---

## 5. Framing

### Accuracy not being state of the art

Put it in the abstract, second or third sentence, unadorned:

> "accuracy 0.707 (95% CI 0.679–0.734) on the matched sample definition, inside and not above the
> published range — an interval that contains Heinsfeld (0.700), ASD-DiagNet (0.703) and
> ASD-SAENet (0.708)."

A reviewer who discovers this in Table 6 rejects; a reviewer who sees us volunteer it reads the
rest generously. State that every accuracy comparison in that section is therefore uninformative
**in the table itself**, not only in prose.

Then supply the alternative currency in the same paragraph: a 95% prediction interval for a new
site of **[0.724, 0.882]** (random-effects mean 0.803, Q = 23.5 on 19 df, p = 0.218, I² = 19.0%,
τ = 0.033; caveat that LOSO per-site estimates share training data so I² is an underestimate).
No comparator paper offers one. Add Abraham et al. (2017)'s 66.8% inter-site on this exact
871/16-site cohort as the correct LOSO comparator — our LOSO accuracies beat it, and that
comparison is currently missing entirely.

### The p = 0.072 shrinkage result

Never write "significant." The permitted sentence, verbatim:

> "+0.014 (95% CI −0.001 to +0.030) pooled and +0.013 (−0.002 to +0.029) under leave-one-site-out;
> neither reaches conventional significance, and because both are computed on the same
> participants they are not independent replications."

The fixed-r comparison (+0.0127, p = 0.0036) stays under a different label: **"shrinkage held
fixed at the level the windowed pipeline applies to its own windows."** It is a single externally
pre-specified comparison — 0.15 is `dynamic_fc.py`'s hard-coded default, not the sweep optimum
(0.30–0.40) — so no selection adjustment is owed, and it is the control that makes the confound
argument airtight. It must never be described as the method's gain and must never appear in the
abstract. Say "matching the shrinkage removes the apparent windowing advantage," not "adding
shrinkage improves performance."

### The self-discovered defects

Nothing was previously published, so there is nothing to correct and no erratum obligation.
**Never write "we previously reported."** Write: *"Averaging member ranks within the held-out
fold — a fusion rule in common use — is transductive. We measure the inflation at 0.020–0.028
AUROC."* That converts an internal fix into a general result other people can use, and it is
true.

State once, plainly, that all three defects were found in our own pipeline during a
pre-submission audit — but only after items 3 and 7 above are fixed. Correcting one artefact that
put us above published results while leaving a structurally identical one in the comparison table
is worse than never disclosing.

*Before submitting, verify externally:* is the GitHub repository public with the old README and
result tables? Is any preprint, thesis chapter or conference abstract carrying 0.716 or the 0.833
selective figures? If yes, this **is** a correction and needs a version note — still fine,
provided it is labelled.

### The Clarke et al. collision

Paragraph one of the Introduction, not Related Work. Clarke et al., *GigaScience* 14:giaf091
(2025), PMID 40899917. Claim the four openings and concede the one loss in the same breath:

> "Clarke et al. (2025) applied conformal abstention to ABIDE with an ABIDE-II replication we
> cannot match. Their procedure is transductive by their own description, reports a single
> operating point at 2.4% coverage without a marginal guarantee, includes no calibration
> analysis, states that they were unable to account for site effects, and is restricted to male
> participants. We report split rather than transductive conformal prediction, a full
> risk–coverage curve, calibration, per-site risk control at 13.7% coverage meeting its target in
> 91% of held-out splits, and both sexes — and no external replication."

Do not claim a risk-control *contribution*. The honest framing: finite-sample selective risk
control is not achievable at usable coverage on ABIDE-I with twenty sites, and the resolution
limit 1/(K+1) ≈ 0.048 explains why. That is a legitimate negative result about the field.

Same treatment for BrainLinear (arXiv 2608.15266) — nearest methodological neighbour, Top-K
ROI-pair selection, which the granularity finding directly refutes. Name it.

---

## 6. METHODOLOGY.md — add and remove

**Remove**
- §1 para 2: the early-diagnosis motivation entirely (referral delay, Zwaigenbaum, "reliable
  diagnosis in the third year of life"). Replace with the decade-long 0.70–0.75 plateau across
  model classes and the question of what part of apparent progress is a property of how
  comparisons are built.
- §1 para 3: "and is feasible in young and clinical populations."
- §1 "Three questions remain open" — those three *are* the three failed contributions. Replace
  with the three dissolved effects plus the ceiling.
- §2 bullet 1: both clauses in omit-list items 1 and 2. §2 bullet 3: omit-list item 3.
- §2 bullet 6 ("a classifier readable as a few whole-brain patterns") — premise falsified by our
  own measurement.
- §12: the two unfulfilled promises — the alternative-QC-rule sensitivity check (no such table
  exists) and "positive and negative predictive value at stated prevalences" (nothing computes
  them). Fulfil or delete.

**Add**
- §2: replace entirely with the five contribution bullets in §5 above. The ensemble, equal
  windows, cross-site transfer and low-rank readability become apparatus in §§5–9; the abstention
  guarantee becomes a self-audit item.
- §2 bullet 5: "the harmonisation that was tested showed no evidence of benefit (Δ = −0.029
  AUROC, q > 0.05; McNemar p = 0.083), consistent with Serra et al. (2023)."
- §3: the full analysis flow in one place — 1,112 released → 871 strict / 1,032 matched → 863 and
  196 regions for the group contrast, counts and reasons at every node, as a STARD-style diagram.
  Age range 6.5–58, median 14.8, 727 M / 144 F, ASD mean FIQ 105.8, healthy controls only, no
  differential-diagnosis comparators. State that **n = 871 is primary and carries every claim**;
  n = 1,032 appears only as a pre-specified matched-sample comparison for comparability.
- §5: concede the estimator's provenance — Eq. (1a) is the standard shrinkage-to-scaled-identity
  estimator (scikit-learn `shrunk_covariance`, `pyriemann.estimation.Shrinkage`, Chen/Wiesel/
  Eldar/Hero 2010). We claim nothing about the estimator.
- §5: the mechanism with its citation — Bartz & Müller (NIPS 2014) showed that under
  autocorrelation the effective sample size falls while the Ledoit–Wolf estimator does not
  compensate, so LW systematically under-shrinks. Report the measured LW ρᵢ distribution
  (median ≈ 0.050, range 0.016–0.149, Spearman(T, ρᵢ) = −0.65) against the 0.20–0.50 the
  objective selects. This converts the grid search into a mechanism that cites its own scoop.
- §5: that the L2 penalty C is re-selected by inner CV at every shrinkage level
  (`run_ablations.py:110–114`), with the selected C at each r. Currently undocumented — one
  reviewer built a fatal objection on the assumption it was not done.
- §6: windows are equalised in **volumes** (50), not seconds, and ABIDE-I TR ranges ~1.5–3.0 s, so
  the ≥100 s criterion is violated at fast-TR sites and violated systematically by site. Also
  state that `L = min(window, T)` silently shortens the window for short scans.
- §12 QC: **correct** the >1% constant-signal rule — it is estimated by globbing the whole
  `filt_noglobal` directory, not the strict cohort. Within the strict cohort it drops **one**
  region, not four; thresholds 0.2/0.5/1/2% drop 11/3/1/0. Report the free defence: dropout is
  not group-differential (ASD 5/403 vs TD 10/468, Fisher p = 0.43).
- §12 Uncertainty: **deliver the promise** — no stored table currently contains a CI. DeLong
  intervals on both headline rows and every retained contrast, Wilson on accuracy, per-site
  intervals. Concede these are optimistic (they condition on the fitted model and treat dependent
  OOF predictions as independent; cite Bates/Hastie/Tibshirani). Add ≥10 seeds for the
  confirmatory comparisons and report between-seed spread. State that LOSO is seed-free and is
  therefore primary.
- §12 Group differences: Freedman–Lane (Winkler et al. 2014) — and state honestly what it does
  not fix: it corrects nuisance handling across permutations, not misspecification of a linear
  term in a single scalar FD. The direct answer to the motion mechanism is the matched subsample.
  Add `func_perc_fd`, `func_dvars` and `EYE_STATUS_AT_SCAN` to the covariate list, with
  median-imputation counts.
- §12 Confirmatory family: **replace it.** Two rejection-seeking hypotheses under Holm at FWER
  0.05 — (H1) the global network contrast as a single Westfall–Young max-|z| over the 36 pairs
  under within-site permutation, (H2) the per-site selective-risk rule against its target.
  Windowed-vs-static becomes an equivalence claim with its CI. Shrinkage magnitude and ensemble
  gain move out to estimates with CIs. Do **not** add "ensemble vs best member" as H3 — at
  p = 0.066 it dies under Holm and we would have volunteered our own headline model into a family
  that kills it.
- §12: the test count — ~624 computed conditions across 31 stored tables, roughly 250–400 formal
  tests. Print BH q beside every exploratory number and put "exploratory" in section headings.
- §12 Reproducibility: the software versions actually used (only torch and numpy are recorded), a
  lockfile or container, and a Zenodo-archived clean tagged commit. The reported run's manifest
  records `git_dirty: true` — re-run the confirmatory family from a clean commit. Release SUB_ID
  lists for every analysed set (871, 1032, 863, 196, 213, 137, 300/163); add `oof_sid` to
  `run_benchmark.py`. That gap is what produced the withdrawn IQ/sex audit.
- §13: the Yeo-7 assignment rule in full — nearest-neighbour resample of `fetch_atlas_yeo_2011`
  (note it defaults to `thickness='thick'`), winner-take-all above a 0.20 voxel-overlap floor,
  else "subcortical/cerebellar". State that 41 of 165 cortically-labelled parcels fall below 0.50
  overlap, minimum 0.214, and publish `system_frac` for every parcel. Add a ≥0.50 sensitivity
  analysis, split the 35-parcel subcortical/cerebellar catch-all into thalamus / striatum /
  hippocampus-amygdala / cerebellum, and name or exclude ROI 30 and ROI 130 (no AAL and no
  Harvard–Oxford label). State that region indices are keyed on the explicit ROI-number column,
  not row position, and add a column-identity assertion on the `.1D` header at
  `run_benchmark.py:111` (currently only a shape check).
- §13: an explicit statement that **no Haufe transform was applied**, so the group contrast is a
  univariate analysis of residualised edges and licenses no statement about what the classifier
  uses. Compress the SHAP paragraph to two sentences and a footnote. Change "sensorimotor" to
  **"somatomotor"** throughout, matching Yeo's own term.
- References, all load-bearing: Ilioska 2023; Chen 2015; Cerliani 2015; Nomi & Uddin 2015; Tyszka
  2014; Hull 2017; He/Byrge/Kennedy 2020; King 2019; Traut 2022; Power 2012; Satterthwaite 2012;
  Ciric 2017; Parkes 2018; Winkler 2014; Bartz & Müller 2014; Abbas 2023; Honnorat & Habes 2022;
  Honnorat 2019; Pervaiz 2020; Chen/Wiesel/Eldar/Hero 2010; Hindriks 2016; Serra 2023; Clarke
  2025; Kurkin & Hramov 2026; the Vidya et al. Expression of Concern; Gemein 2020; Kiessner 2024;
  Schulz/Bzdok/Haufe 2024; Nielsen 2013; Plitt 2015; Varoquaux 2018; Bouthillier 2021; Marek 2022;
  Risi 2006; Hui & Walter 1980; TRIPOD+AI (Collins 2024). Fix the mis-attribution of arXiv
  2501.17207 to "Luo et al." — it is Han et al. Resolve both `[VERIFY]` entries and the GitHub
  link.

---

## 7. RESULTS_SUMMARY.md — add and remove, section by section

- **§1 voting.** Delete the superseded three-atlas + phenotype table and redraw the figures it
  fed. Delete "The average is better than any of them alone" (p = 0.066). Add the adopted
  ensemble's sensitivity/specificity (pooled 0.7047/0.7094; LOSO 0.7419/0.6838) and likelihood
  ratios (LR+ 2.43, LR− 0.42 pooled) — these appear nowhere and are the only numbers a clinician
  reads. Reconcile the three headline AUROCs that circulate (0.7746 fold-mean, 0.7760 stored OOF,
  0.7811 all-196-regions) in one table stating cohort, folds, representation and r for each.
- **§2 statistics.** Print q alongside p on all rows. Dedupe `rank=20 vs logistic` and `low-rank
  spectral (r=20) vs logistic` — the same comparison twice, and a duplicated p-value inside a
  correction family is itself an error. ComBat verdict → "no evidence of benefit." Mark the
  rank-mean rows as using the withdrawn transductive score.
- **§3 protocol/selective.** Propagate the honest post-fix numbers, which currently appear in
  neither source document: cohort-wide rule certifies nothing in sample (bound 1.000); per-site
  rule meets target in 91% of splits against nominal 90% at 13.7% coverage; cohort-wide honest
  split evaluation 82%. Add the per-site coverage table **including the seventeen sites at
  coverage 0.000**, and state that 170 of 871 answered cases are 84% from NYU. State the
  local-calibration requirement and the minimum answerable n (8 at δ = 0.1) — and that the
  smallest sites abstain because their error rates are too high for any threshold to certify, not
  merely because they are small. Delete "the per-site rule is therefore the one the model should
  be deployed with."
- **§7 leakage/QC.** Remove the "Not for publication" banner; reinstate the leakage table as one
  supporting panel. Add the QC-only classifier for both cohorts and concede what it shows:
  motion/quality-only AUROC rises 0.571 → 0.597 strict-to-wider, a gain **larger** than the
  model's own 0.011–0.016, and the FD group effect size rises from d = 0.28 to d = 0.36.
- **§8 spectrum/symptom axes.** Delete "so the model is reading brain organisation rather than an
  artefact." Compress SHAP to one sentence; drop the waterfall. Move the 120-test symptom-axes
  null to supplement. Mark ADOS-total within-autism as failing BH and note the LOSO value
  (ρ = 0.065, p = 0.279). **Promote this section to §2 or §3** — it is the paper's strongest
  result.
- **§9 ridge sweep.** Relabel "ridge alone, on the static representation" as **"shrinkage held
  fixed at the level the windowed pipeline applies to its own windows."** Add the nested rows
  with p = 0.0722 / 0.0983 and their CIs (in neither source document today). Add the equivalence
  statement with CI [−0.016, +0.011]. Soften the "property of the estimator" sentence. State the
  grid actually used — the sweep reports 10 values to 0.80 while `exp_ridge_sweep.py`'s default is
  6 to 0.30 — and whether it was extended after seeing the curve.
- **§10 scaling law. Delete the entire section.** One row in the falsified table.
- **§11 label ceiling.** Delete the 0.9535 point estimate and the corrected-AUROC table. Replace
  with the range [~0.85, ~0.99] plus a sensitivity table over ρ₁ ∈ {0, 0.075, 0.15} and the SRS
  cut ∈ {40, 50, 55, 60, 70}. Rewrite the caveat: ρ₀ is *unidentified*, not bounded below. Add
  κ = −0.078 and Hui & Walter. Add that the estimand is P(assigned label ≠ true state) while
  DX_GROUP is a best-estimate clinical diagnosis informed by both instruments, so 15.0% bounds
  noise in an instrument-derived label. Reconcile the 3/137 vs 3/175 drift in one sentence.
- **§12 network contrast.** Promote the motion-matched analysis to primary or co-primary: 256
  matched pairs, n = 512, **26 of 36 pairs significant** versus 10, somatomotor −0.2925
  (z = −4.89) versus −0.2155. Explain the strengthening (site composition and n change; report
  both). Delete "cannot be produced by ... moving more." Delete the claim that motion lowers
  connectivity — in this no-GSR pipeline FD vs mean edge strength is ρ = +0.51, within-site
  partial r = +0.30, so motion *inflates* connectivity and the reduction is found against the
  direction of the confound. Delete the limbic claims and the globality framing. Add Ilioska and
  Chen in the first sentence; retitle as replication under confound control. Replace all
  "q = 0.000" with "p < 2 × 10⁻⁴".
- **§13 somatomotor regions.** *(Partly done — see §3 above.)* Still to do: replace the ranked
  table with the pre-specified contrast plus the enrichment test (p = 0.011, labelled post-hoc);
  add the specificity nulls (2000 random 21-region blocks, P = 0; best compact neighbourhood
  −0.138); add the z-coordinate gradient (ρ = 0.50) and own it with the counter-evidence
  (|effect| vs per-region FD susceptibility ρ = −0.55; no distance dependence across the 36
  pairs); report both AAL and Harvard–Oxford labels with overlap fractions for every named
  region; delete the sentence about sensory processing being part of the clinical definition.
  Move the section to supplement.
- **§14 region selection.** Keep. Rewrite the leakage-contrast sentence to cite the now-published
  leakage table. Give the interval on the 0.006 null rather than stating it as a positive result,
  and note that in-fold selection *beat* all-data selection at top-30 and top-50 — evidence the
  comparison is noise-dominated.
- **§15 embedding flow.** Cut to supplement.
- **§16 comparison.** Delete the "accuracy, confident half" column and the fig20 caption clause.
  Add Wilson CIs to every accuracy and state in the table that the comparison is uninformative.
  Redraw fig20 so it does not plot our bar beside theirs as commensurable. Add Abraham 2017's
  66.8% inter-site as the correct LOSO comparator. Add the per-site meta-analysis (Q = 23.5,
  p = 0.218, I² = 19.0%, prediction interval [0.724, 0.882]).
- **§17 status.** Replace with the compact falsified table.
- **Throughout.** Rewrite the plain-language "What this means" paragraphs into journal register.
  Fix **"mean FD 0.089 vs 0.068"** — those are medians; the means are 0.126 vs 0.094 (Welch
  p = 8.2 × 10⁻⁵, d = 0.27). Mislabelling a median as a mean in the direction that understates the
  confound, in this paper, is not survivable.

---

## 8. Work still required

**Blocking**
1. ~~Motion-match the region-level somatomotor analysis.~~ **Done — passed, see §3(a).**
2. **Rerun the network-contrast permutations storing the null draws, and run the
   deviation-from-global-mean test.** `exp_network_contrast.py` writes only summary counts; the
   5,000 draws are discarded and only per-pair `null_sd` survives. Needs a full re-run of the
   permutation pass, twice (matched and unmatched) — not a same-day fix. The conservative version
   already indicates somatomotor↔somatomotor deviates negatively (z ≈ −3.0) and
   somatomotor↔subcortical/cerebellar positively (z ≈ +3.4, q ≈ 0.022) — the latter is the
   direction Cerliani 2015 and Ilioska 2023 report, and it collapses to z = +0.35 under motion
   matching. Report both. **Until this runs we cannot name ten pairs.**
3. ~~Regenerate `label_ceiling_corrected.csv`~~ — dissolves into "delete the table" (Tier 2), but
   note it was computed on the withdrawn transductive score. Volunteer this if it comes up.

**High value, non-blocking**
4. The static-vs-dynamic literature survey — 12–15 published dFC-vs-static comparisons, what each
   does to the window covariance and the whole-scan baseline. Converts the confound from an
   anecdote about two of our own scripts into a claim about the field. Note Honnorat 2019 *do*
   shrink dFC windows — that evidences half our premise, and means we cannot say "every."
5. Quality-only classifier, both cohorts plus the low-motion stratum: FD-alone 0.597, cross-
   validated quality model 0.573; in the low-motion subset (FD < 0.15, perc_fd < 10, n ≈ 600) the
   connectivity model rises to ~0.788 while the FD-only baseline falls to ~0.565. If quality
   carried the model, both would fall together.
6. QC-FC and QC-FC distance dependence (Ciric/Parkes). Only the edge matrix and `func_mean_fd` are
   needed; centroids are in `atlas/roi_table.csv`. An afternoon.
7. The `filt_global` replication — a download, not a reprocessing. Reporting a direction-of-effect
   claim under exactly one denoising choice is the He/Byrge/Kennedy 2020 failure mode.
8. **Age stratification.** The claim that it would be underpowered is false and must not be
   written: n = 222/379/262, effect present in all three bands, strongest in adolescence
   (d = −0.42, p = 4 × 10⁻⁵), age × diagnosis interaction p = 0.33 — agreeing with Ilioska's null
   interaction at n = 1824. A null interaction is a result. Ten minutes.
9. Freedman–Lane. Ten lines. Run it because a methods reviewer will ask for it by name, but do not
   claim it addresses the residual-motion mechanism.
10. Sex-stratified AUROC and sensitivity on the probability-mean score. 144 females. Report it
    whatever it shows, with wide CIs — omitting it after withdrawing a subgroup analysis is itself
    a flag.
11. Decision curve at π = 0.03/0.10/0.20/0.30 with the case-control correction, plus the PPV/NPV
    table §12 promises. Then write the operational conclusion in our own voice: at π = 0.30 the
    instrument answers 14 of 100 referred children and avoids zero specialist assessments.
12. Housekeeping: rewrite or delete `publication_plan.md`; stamp RESEARCH_COUNCIL_FINDINGS §2 C4
    inline as WITHDRAWN and ensure neither document ships; add `oof_sid`; tag a clean release and
    archive with a DOI; complete the TRIPOD+AI checklist as supplementary — in 2026 that is worth
    more to acceptance odds than 0.01 of AUROC.

---

## 9. Venue

**Imaging Neuroscience.** Not IEEE JBHI — two of six reviewers returned outright rejections and
both rest on venue mismatch rather than on the work. JBHI's filter is a method that does something
better, and no restructuring makes 0.707 accuracy plus a set of dissolved effects read as a
biomedical-informatics method. The currencies JBHI would accept — a deployable system with a
guarantee, a clinical-utility demonstration, a benchmark resource — are precisely what our own
honesty disqualifies.

Imaging Neuroscience explicitly welcomes methodological critique, replication and negative
results; its readership is exactly the community that cares that a static-vs-dynamic comparison
varies two things at once and that a pooled correlation across groups is arithmetic; no page-charge
cliff; open access, which matters for a paper whose value is corrective. Non-SOTA accuracy is a
non-issue there.

Second: **GigaScience** — it published Clarke et al. 2025, our direct collision, and our stored
per-participant OOF predictions plus 31 ablation tables are its currency. Third:
**NeuroImage: Clinical** — it published Heinsfeld and Chen 2015, though it will want a
clinical-utility claim we cannot make. Fourth, only if those fail: **Computers in Biology and
Medicine**. **Not** Human Brain Mapping or NeuroImage — both would want the network and
somatomotor findings to *be* the paper, and those are the half that cannot survive Ilioska.

**The structural option.** Split it. The evaluation content — the shrinkage confound, the two
defects, the leakage-granularity result, the severity pooling artefact, the calibrated abstention
with honest split evaluation, the per-site prediction interval — is a complete, self-contained
paper today. The neuroscience content still needs items 2, 7 and 9 above plus the Ilioska
reconciliation. Carrying both in one manuscript is what forces the somatomotor ranking and the
ceiling point estimate to be stated more strongly than the evidence supports. **If the blocking
item slips, split rather than weaken.**
