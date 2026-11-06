# Results summary

Autism classification from resting-state functional connectivity on ABIDE-I (871 participants: 403 autism, 468 control, about twenty sites). Everything the model learns is fitted inside the training data only, and results are given for two splits: **pooled**, where participants are divided at random, and **leave-one-site-out**, where a whole site is held back so the model is tested on a scanner it has never seen.

---

## Summary of findings

**Two findings concern autism.**

1. **Detection is weaker in autistic participants with higher cognitive ability.** Holding the comparison group fixed at all 468 non-autistic participants and stratifying only the autistic side, discrimination falls monotonically from 0.873 below full-scale IQ 85 to 0.719 at IQ 115 and above, under both protocols. The same gradient holds within the autistic group, where nothing separates the groups to confound it (rho = -0.18 against full-scale IQ, p = 0.0004, pooled scores), and is absent in controls (+0.05, p = 0.32) — a null that makes the association a property of how autism presents in this measurement rather than a general effect of ability. A connectivity-based screening aid is therefore least effective in cognitively able autistic people. Section 1.2.

2. **A somatomotor difference that strengthens under motion control.** In 256 within-site pairs matched for head motion at the system level, the group motion difference is removed (p = 7.6 × 10⁻⁷ before matching, 0.59 after) and significant system pairs rise from 10 of 36 to 26 of 36; in 280 pairs matched with the wider regional caliper (p = 0.28 after matching), the somatomotor block moves from -0.215 to -0.249 and all 21 of its regions remain significant. The reduction runs against the direction of the confound: mean framewise displacement correlates positively with mean connection strength (Spearman rho = +0.51 across participants, +0.45 within site), so motion raises rather than lowers connectivity in this pipeline. The direction agrees with the somatomotor involvement reported by Chen et al. (2015) and the somatomotor reductions reported by Holiga et al. (2019) and Ilioska et al. (2023); here it is obtained in within-site pairs matched for head motion. Sections 6 and S6.

**Two concern what actually determines measured performance on this dataset.**

3. **Feature selection outside the training partition is worth 0.231 AUROC, and the inflation depends on feature granularity.** Selecting 2,000 of 19,900 connections before rather than inside cross-validation takes AUROC from 0.756 to 0.987 and accuracy from 0.692 to 0.948 on identical folds; the same selection over 10 to 100 of 196 regions changes AUROC by 0.006 on average in absolute value (0.001 at 10 regions, at most 0.012), with inconsistent sign. This is the largest effect in the study, so a pipeline that selects connections before the split can report accuracies well above those attainable without that step. Without any such step our accuracy is competitive with those figures — 0.706 (95% CI 0.678-0.733, participant-pooled, n = 1,032) against 0.700, 0.703 and 0.708 — so matching them does not require the shortcut. Sections 5 and 1.3.

4. **The linear classifiers perform alike, and fixed-configuration tree ensembles are markedly lower.** In participant-pooled out-of-fold AUROC, penalised logistic regression reaches 0.757 against random forest 0.611, XGBoost 0.617 and CatBoost 0.624 (DeLong p from 6.3 × 10⁻¹⁵ to 2.4 × 10⁻¹²); ridge (0.755) and linear SVM (0.753) are close to it, while shrinkage LDA (0.741) and the rank-20 spectral model (0.735) are lower. The tree ensembles used fixed configurations rather than nested tuning, so the gap describes these configurations rather than the best attainable member of each family. Sections S3 and S1.

Approaches evaluated and not adopted are collected under Limitations, and are reported as limitations on what these data can support rather than as findings.

---

![Pipeline overview](results/2026-08-24_23-58-52_full_study/figures/fig01_pipeline.png)

*Pipeline overview. The whole procedure, from the released scans to a decision.*

# Main findings

## 1. Classification performance and transfer to an unseen site

![Discrimination, per-site transfer, calibration and abstention](results/2026-08-24_23-58-52_full_study/figures/fig02_figS_performance.png)

*Discrimination, per-site transfer, calibration and abstention. (a) Discrimination with 95% intervals under both protocols. (b) Every site with the model trained without it, marker area proportional to n, with the random-effects mean and the prediction interval for a site not yet seen. (c) Predicted probability against observed frequency. (d) What the abstention rule answers, against the target it promises.*

| configuration | split | AUROC | sd | accuracy | adopted |
|---|---|---|---|---|---|
| ensemble (probability mean) | pooled | 0.775 | 0.040 | 0.707 | yes |
| dynamic: mean | pooled | 0.767 | 0.044 | 0.695 |  |
| ensemble (rank mean) | pooled | 0.756 | 0.038 | — |  |
| tangent (static) | pooled | 0.756 | 0.048 | 0.692 |  |
| correlation | pooled | 0.750 | 0.040 | 0.683 |  |
| ensemble (probability mean) | unseen site | 0.773 | 0.087 | 0.684 | yes |
| dynamic: mean | unseen site | 0.771 | 0.082 | 0.670 |  |
| tangent (static) | unseen site | 0.757 | 0.104 | 0.643 |  |
| ensemble (rank mean) | unseen site | 0.750 | 0.091 | — |  |
| correlation | unseen site | 0.743 | 0.083 | 0.655 |  |

The averaged probability is the configuration carried forward. The alternative, averaging each member's *rank*, places each held-out participant's score within the distribution of that member's training-set scores (Methods Section 3.4), so every participant is scored independently of the others. It is reported for comparison only; under this construction it scores below the best single description (Section S1). Because the rank score is not a probability, it has no 0.5 decision threshold, and no accuracy is reported for it.

**What this means.** Three different ways of describing the same scans were each given their own classifier and their answers averaged. Under both splits the averaged probability has the highest AUROC of the connectivity-only configurations in the table, but its gain over the best single description, the windowed mean, is small (+0.009 in participant-pooled AUROC) and is not distinguished from chance (DeLong p = 0.066; Section S1), so it is not presented as an established improvement. Performance on an unseen site is close to performance within the cohort, so the combination is not relying on knowing the scanner.

**On the demographic variants.** Age, sex and IQ alone reach an AUROC of about 0.62, so they do carry some signal, and adding them to the ensemble raises participant-pooled AUROC by about 0.004 (p = 0.041 unadjusted, q = 0.11 across the comparisons in Section S1). The gain was not adopted because it is small; the q-value is given for reference only (Methods Section 8.1). The connectivity features appear to capture the same information already.

For reference, the three-atlas model with demographic variables gave the following under both splits. It is not the primary ensemble, and the abstention results below are not computed on it. The ROC curve in Section S7 and the connectome, network-matrix and SHAP figures in Section S5 are drawn from this model:

| split | AUROC | accuracy | sensitivity | specificity |
|---|---|---|---|---|
| Pooled | 0.761 | 0.699 | 0.685 | 0.711 |
| Leave-one-site-out | 0.757 | 0.658 | 0.683 | 0.650 |

**What this means.** Performance barely changes when a whole site is held back, so the model is not relying on knowing the scanner. Sensitivity and specificity are close to each other, so it is not simply favouring one answer.

### 1.2 Intervals, per-site variation, and a prediction interval

Headline figures with intervals. Areas under the curve carry DeLong intervals; accuracy, sensitivity and specificity carry Wilson intervals. Both condition on the fitted model and treat dependent out-of-fold predictions as independent, so they are the narrowest defensible intervals rather than a full accounting of uncertainty.

| split | AUROC | accuracy | sensitivity | specificity |
|---|---|---|---|---|
| pooled | 0.776 (0.745-0.807) | 0.707 (0.676-0.736) | 0.705 (0.658-0.747) | 0.709 (0.667-0.749) |
| loso | 0.774 (0.744-0.805) | 0.711 (0.680-0.740) | 0.742 (0.697-0.782) | 0.684 (0.640-0.724) |

Combining the 20 per-site estimates under a random-effects model gives a mean of **0.803**, with between-site heterogeneity not distinguishable from chance (Q = 23.5 on 19 df, p = 0.22; I-squared = 19%). The 95% prediction interval for a site not yet seen is **0.725 to 0.882** — the quantity a reader deciding whether to use this instrument at their own centre actually needs, and one no comparison study reports. Because leave-one-site-out estimates share training data, the heterogeneity is if anything understated.

**Detection is weaker at higher cognitive ability.** Holding the comparison group fixed at every non-autistic participant and stratifying only the autistic side, each stratum is scored against the same reference, so differences between strata do not arise from changes in the comparison group. Discrimination declines monotonically under both protocols. The autistic strata still differ from the reference group in full-scale IQ; that separation is addressed by the IQ-matched analysis that follows.

| autistic participants | n | pooled | site withheld |
|---|---|---|---|
| FIQ < 85 | 40 | 0.873 | 0.871 |
| FIQ 85-99 | 97 | 0.815 | 0.816 |
| FIQ 100-114 | 126 | 0.781 | 0.788 |
| FIQ >= 115 | 113 | 0.719 | 0.709 |

**Stratifying both groups by IQ.** Dividing all participants into quartiles of full-scale IQ centred within site reproduces the gradient, from 0.833 in the lowest quartile to 0.709 in the highest (pooled). Because autistic participants have lower full-scale IQ, the lower quartiles are also those in which the groups differ most on IQ. Matching each autistic participant to a non-autistic participant at the same site whose full-scale IQ differs by at most five points (274 pairs pooled, 275 with the site withheld; Methods Section 6.9) removes that separation. Discrimination then remains highest in the lowest quartile (0.855 pooled, 0.848 site withheld) but no longer declines across the upper three (0.700–0.756 pooled; 0.735–0.756 site withheld). Part of the full-range gradient therefore reflects the group difference in IQ; the within-group association below involves no between-group comparison and is the direct evidence.

| site-centred FIQ quartile | n (all) | all, pooled | all, site withheld | n matched (pooled / site withheld) | matched, pooled | matched, site withheld |
|---|---|---|---|---|---|---|
| Q1 | 204 | 0.833 | 0.825 | 137 / 144 | 0.855 | 0.848 |
| Q2 | 203 | 0.794 | 0.793 | 137 / 132 | 0.700 | 0.735 |
| Q3 | 201 | 0.746 | 0.745 | 138 / 137 | 0.751 | 0.737 |
| Q4 | 203 | 0.709 | 0.705 | 136 / 137 | 0.756 | 0.756 |

The same gradient appears within the autistic group, where there is no group separation to confound it, and is absent in non-autistic participants. Scores are residualised on site, age and head motion before Spearman correlation; values are given for both validation protocols, as rho (p).

| ability measure | autistic, pooled | autistic, site withheld | non-autistic, pooled | non-autistic, site withheld |
|---|---|---|---|---|
| FIQ | -0.180 (0.0004), n = 376 | -0.179 (0.0005), n = 376 | +0.048 (0.3208), n = 435 | +0.022 (0.6466), n = 435 |
| VIQ | -0.182 (0.0007), n = 342 | -0.189 (0.0004), n = 342 | +0.039 (0.4547), n = 379 | +0.001 (0.9892), n = 379 |
| PIQ | -0.152 (0.0047), n = 344 | -0.147 (0.0062), n = 344 | +0.007 (0.8862), n = 391 | -0.005 (0.9271), n = 391 |

The control-group null is what makes this interpretable: the association is a property of how autism presents in the connectivity measurement, not a general relationship between ability and the score. Nielsen et al. (2013) reported that classifier scores covaried with verbal IQ on this dataset, so the association is anticipated; what is added here is the control-group comparison and the graded loss of detectability.

**Discrimination by sex shows no deficit.** Area under the curve is somewhat higher in female participants — female 0.827 (0.760-0.895, n = 144) under pooled; male 0.764 (0.730-0.798, n = 727) under pooled; female 0.806 (0.735-0.878, n = 144) under loso; male 0.767 (0.733-0.801, n = 727) under loso — but with 144 female participants the intervals overlap and no difference is claimed.

*Linkage note.* Stored out-of-fold predictions are written in fold order and carried no participant identifier, so the fold order was replayed and the reconstruction verified element-wise against both the stored labels and the stored site codes for all 871 participants before any phenotype was joined. Sentinel-coded missing values are excluded rather than imputed. Subsequent runs record identifiers directly.

### 1.3 Placement against published work on ABIDE-I

All entries use ABIDE-I resting-state functional connectivity. Sample size and testing procedure are listed because both change what the number means. Values for the other studies are as reported by their authors and were not recomputed here. The matched-sample rows use the same quality-control definition as those studies, which admits 1,035 participants; three of them have no usable time series in the release, so 1,032 are analysed.

**The accuracy comparison in this table is uninformative and is included only because its absence would be conspicuous.** None of the comparison studies reports a confidence interval, the intervals implied by these sample sizes are roughly +/- 0.03, and the three published values span 0.008. The last two columns record properties those studies do not report: how much accuracy survives on a scanner never seen in training, and whether the stated probabilities match observed frequencies. Those are the columns on which this work differs. AUROC values for this work are participant-pooled out-of-fold estimates; accuracy is the fold mean, which agrees with the pooled value to three decimals. For the matched sample, the participant-pooled accuracy is 0.706 (95% Wilson interval 0.678–0.733, n = 1,032).

| study | accuracy | AUROC | sample | procedure | AUROC on unseen site | calibration error |
|---|---|---|---|---|---|---|
| Heinsfeld et al. 2018 | 0.700 | not reported | 1,035 | 10-fold | not reported | not reported |
| Eslami et al. 2019 (ASD-DiagNet) | 0.703 | not reported | 1,035 | 10-fold | not reported | not reported |
| Almuqhim & Saeed 2021 (ASD-SAENet) | 0.708 | not reported | 1,035 | 10-fold | not reported | not reported |
| **This work — ensemble** | 0.707 | 0.776 | 871 | nested 10-fold | 0.774 (-0.002) | 0.046 |
| **This work — ensemble, matched sample** | 0.706 | 0.789 | 1,032 | nested 10-fold | 0.783 (-0.006) | 0.023 |
| This work — tangent model, matched sample | 0.696 | 0.766 | 1,032 | nested 10-fold | — | — |

**What this means.** Accuracy is in the same range as the established results on this dataset, reached under a stricter procedure in which nothing is fitted before the data are split. Higher figures reported elsewhere could not be checked against their sources and are therefore not listed.

**Where the difference actually lies.** The accuracy column compares models on all cases at a fixed cut-off, which is the least of what matters for a screening instrument. The two columns beside it are the substantive difference: performance is essentially unchanged on an unseen scanner, and the stated probabilities can be taken at face value. Abstention was also examined: a per-site rule held its error target on held-out half-splits but answered only 170 of 871 participants (19.5%), 143 of them at a single site, and the cohort-wide rule answered none, so abstention is reported as a measurement rather than as a feature of the model (Section 2).

**The main limitation.** Performance is uneven across sites: it ranges from 0.589 at SBL to 0.912 at UM_2. No site's point estimate falls to chance, but 2 of 20 sit below 0.65 (SBL, MAX_MUN). The 95% intervals of 5 sites (CALTECH, LEUVEN_1, MAX_MUN, OHSU, SBL) include 0.5. Whether that spread reflects real differences between scanners or the small size of most sites is not settled by the range alone: the median site contributes about thirty participants, so a wide interval is expected even if every site performed identically. It is reported as a range, not as evidence of site-dependent failure.

## 2. Calibration, and the cost of a guarantee

Computed on the ensemble, so this section describes the same model as the headline above.

The figures here are computed over all participants' held-out predictions pooled together, whereas Section 1 averages each fold's score. The two agree closely under the pooled split. They differ more under leave-one-site-out, where averaging folds gives every site equal weight regardless of size while pooling gives every participant equal weight; the accuracies quoted below are therefore slightly higher than the per-site averages in Section 1.

| split | accuracy, all cases | calibration error | accuracy on the more confident half* |
|---|---|---|---|
| Pooled | 0.707 | 0.046 | 0.805 |
| Leave-one-site-out | 0.711 | 0.040 | 0.819 |

\* This last column is reported for completeness and should not be read as an operating point. The half is defined by the median confidence *of the held-out set itself*, which is a transductive quantity: it cannot be computed for a single incoming participant; it corresponds to the exploratory risk–coverage analysis of Methods Section 5.2. The valid version of this analysis — a threshold fixed on calibration data and applied to participants who took no part in choosing it — is reported below and gives a substantially less favourable picture.

**What this means.** The calibration is sound: where the model reports seventy per cent it is correct about seventy per cent of the time, with an expected calibration error near 0.04 under both protocols. Whether that calibration can be converted into a guarantee on the cases the model chooses to answer is a separate question, addressed next.

![A guarantee, and what it is worth when checked honestly](results/2026-08-24_23-58-52_full_study/figures/fig07_figS_guarantee.png)

*A guarantee, and what it is worth when checked honestly. (a) The error bound each rule reports on the participants that calibrated it, against the error those rules actually incur on held-out cases. (b) What each rule costs to obtain that bound: the share of cases it answers, and the share of held-out splits in which it meets its target.*

**Is the guarantee real?** A threshold chosen so that a bound passes, then scored on the same participants, passes by construction and says nothing about a future case. Splitting the data — one half to choose the threshold, the other to be scored with it — is what tests it. Under that test the single cohort-wide threshold does not hold its target, whereas the per-site variant does but answers few cases: with the threshold set on the whole sample it answers 170 of 871 participants (19.5%), 143 of them at a single site, and its mean coverage across held-out half-splits is 13.7%. It is therefore reported as a measurement rather than as a deployable configuration.

## 3. The classifier score does not track symptom severity

![The four effects that change when their control is applied](results/2026-08-24_23-58-52_full_study/figures/fig15_figS_corrections.png)

*The four effects that change when their control is applied. (a) Severity correlation, pooled against within-group; rows marked with a dagger are instruments given only to autistic participants, for which the two scopes must coincide. (b) Whole-scan connectivity across shrinkage levels, with the windowed representation placed at the shrinkage it applies to its own windows. (c) Inflation from selecting outside the training split, against the number of candidates. (d) Observed accuracy against the interval attainable given the diagnostic labels.*

Several published studies report that a classifier's score correlates with symptom severity. Those correlations are computed across both groups. Non-autistic participants who complete these instruments score at the low end of the scale (in this cohort 19 have an ADOS total and 137 an SRS total; the ADI-R was administered only to autistic participants), so any model that separates the groups produces a pooled correlation largely by construction. Computing it both ways separates the two readings:

| instrument | n (pooled) | rho (pooled) | q (pooled) | n (within autism) | rho (within autism) | q (within autism) |
|---|---|---|---|---|---|---|
| ADI-R communication | 277 | -0.058 | 0.398 | 277 | -0.058 | 0.478 |
| ADI-R repetitive behaviour | 277 | 0.077 | 0.300 | 277 | 0.077 | 0.400 |
| ADI-R social | 276 | -0.051 | 0.398 | 276 | -0.051 | 0.478 |
| ADOS calibrated severity | 203 | 0.196 | 0.010 | 193 | 0.128 | 0.227 |
| ADOS total | 301 | 0.207 | 0.001 | 282 | 0.115 | 0.227 |
| SRS total | 300 | 0.438 | 0.000 | 163 | 0.050 | 0.522 |

Scores are out-of-fold ensemble probabilities from the pooled validation protocol; 'pooled' in the column headers refers to the correlation scope (both diagnostic groups). q is the Benjamini-Hochberg-adjusted p-value within each scope's six-instrument family.

**What this means.** Pooled across groups, 3 of 6 associations survive Benjamini-Hochberg correction within the six-instrument family. Within the autism group, where the floor effect is absent, 0 of 6 do. The largest pooled value reproduces the magnitude reported in the literature and then all but vanishes once controls are removed, which is the signature of a group-separation artefact rather than a graded measurement. The instruments administered only to autistic participants show the same correlation and the same uncorrected p-value in both scopes, as they must, because both scopes contain the same people; their q-values differ only because each scope is corrected as its own six-instrument family. This is a check that the comparison is doing what it claims. The model distinguishes autistic from non-autistic participants; it carries little information about how autistic someone is.

## 4. The windowed advantage is a difference in shrinkage

| condition | representation | extra shrinkage | AUROC | sd |
|---|---|---|---|---|
| static r=0.00 | static | 0.00 | 0.756 | 0.048 |
| static r=0.05 | static | 0.05 | 0.762 | 0.046 |
| static r=0.10 | static | 0.10 | 0.767 | 0.045 |
| static r=0.15 | static | 0.15 | 0.769 | 0.044 |
| static r=0.20 | static | 0.20 | 0.771 | 0.044 |
| static r=0.30 | static | 0.30 | 0.773 | 0.043 |
| static r=0.40 | static | 0.40 | 0.773 | 0.042 |
| static r=0.50 | static | 0.50 | 0.772 | 0.041 |
| static r=0.65 | static | 0.65 | 0.768 | 0.038 |
| static r=0.80 | static | 0.80 | 0.759 | 0.039 |
| dynamic: mean (r=0.15) | windowed | 0.15 | 0.767 | 0.044 |

| what is compared | difference in AUROC | p |
|---|---|---|
| shrinkage alone, on the static representation | 0.013 | 0.004 |
| the windowed representation against an unshrunk static baseline | 0.010 | 0.281 |
| the windowed representation against a matched static baseline | -0.002 | 0.709 |

**What this means.** A covariance estimated from a short window is far noisier than one estimated from a whole scan, so the windowed representation applies an extra shrinkage towards the identity that the whole-scan representation never receives. Every comparison between the two therefore changes the amount of regularisation at the same time as the representation. Applying the same shrinkage to the whole-scan estimate, and changing nothing else, recovers the difference that was attributed to windowing. The effect is a property of the estimator, not evidence that connectivity changes during the scan.

## 5. Feature-selection leakage depends on granularity

| how the pipeline was fitted | AUROC | sd |
|---|---|---|
| scaler_outside | 0.753 | 0.049 |
| tangent_outside | 0.756 | 0.048 |
| honest | 0.756 | 0.048 |
| site_feature | 0.756 | 0.049 |
| nonnested_tuning | 0.758 | 0.048 |
| fs_outside | 0.987 | 0.008 |
| all_shortcuts | 0.988 | 0.008 |

**What this means.** Fitting the tangent reference, or the tangent reference and the feature scaler, on all participants before splitting changed AUROC by at most 0.003 (0.756 and 0.753 against 0.756 for the reference protocol), and choosing the regularisation constant on the test fold (0.758) or adding acquisition site as a predictor (0.756) changed it by a similar amount. Selecting the 2,000 connections with an F-test on every participant's diagnosis before splitting raised AUROC from 0.756 to 0.987, and applying every shortcut together gave 0.988. Only the step that sees the labels inflates the score, so figures from pipelines that select connections before cross-validation are not comparable with estimates in which selection is nested.

| sample definition | n | AUROC | sd | accuracy |
|---|---|---|---|---|
| strict | 871 | 0.756 | 0.048 | 0.692 |
| func | 1032 | 0.766 | 0.020 | 0.696 |

**What this means.** Studies differ in how strictly they discard poor-quality scans, and that choice changes the score. Allowing the lower-rated scans back in raises performance slightly. Since the excluded scans are mostly those with head motion or failed processing, and are not evenly split between the two groups, part of that gain may reflect data quality rather than brain differences. Both definitions are reported so comparisons with other studies are like for like.

### 5.2 Region-level selection, in-fold against all-data

| regions kept | chosen from everyone | chosen inside each fold |
|---|---|---|
| 10 | 0.628 | 0.627 |
| 20 | 0.660 | 0.656 |
| 30 | 0.699 | 0.702 |
| 50 | 0.738 | 0.748 |
| 100 | 0.764 | 0.752 |

Using **all 196 regions** gives **0.781**, higher than any reduced set above.

**What this means.** Keeping only the most discriminating regions makes screening worse, and steadily so: the smallest set loses 0.154 of area under the curve against using every region. There is no subset that improves on the whole, which says the difference is spread across the cortex rather than concentrated where it is strongest. The somatomotor result of Section 6 identifies where the effect is largest, not where it lives, and a reduced set of regions is not a route to a better screening test.

**A second reading.** The regions were chosen twice: once from every participant, which uses the answer before measuring it, and once inside each training partition only. The two differ by 0.006 on average in absolute value (at most 0.012, at 100 regions), and the difference does not have a consistent sign, so selection at this granularity gives no consistent advantage. That contrasts sharply with the same procedure applied to individual connections, which inflates the score enormously (Section 5). The risk of a selection step depends on how fine-grained the choice is: ten of 196 regions is too coarse a decision to fit noise, whereas two thousand of 19,900 connections is not.

## 6. Group differences between functional systems

![Group differences, and what motion matching does to them](results/2026-08-24_23-58-52_full_study/figures/fig18_figS_brain.png)

*Group differences, and what motion matching does to them. (a, b) Standardised difference for every pair of systems, on all participants and on within-site pairs matched for head motion, on a shared colour scale; asterisks mark pairs surviving correction. (c) Each somatomotor region's effect on all participants plotted against its effect in motion-matched pairs; points below the diagonal are stronger once motion is matched. (d) Discrimination by full-scale IQ band among autistic participants (below 85, 85–99, 100–114, 115 and above), each against all 468 non-autistic participants, under both protocols.*

There are too many individual connections (19,110) to test one at a time with useful power, and their ordering was checked only inside the somatomotor system (Section S6), where the fifty strongest agree across random halves at a correlation of 0.34. Grouping the regions into the seven cortical systems of Yeo et al. plus a subcortical and cerebellar class reduces the comparisons to thirty-six system pairs, each averaging over 21 to 1,462 connections (median 441). Every connection is first adjusted for acquisition site, head motion, age and sex, and significance comes from permuting diagnosis within each site. The permutation certifies the p-value on residualised connections but not the residualisation itself. Within-site pairs matched for head motion (panels a–c, and Section S6) therefore address the possibility that a difference reflects autistic participants being scanned elsewhere or moving more; age is handled by adjustment alone.

| system | system | connections | difference | z | q |
|---|---|---|---|---|---|
| somatomotor | somatomotor | 210 | -0.215 | -4.769 | 0.002 |
| limbic | visual | 210 | -0.140 | -3.633 | 0.002 |
| limbic | salience / ventral attention | 147 | -0.133 | -3.247 | 0.013 |
| default mode | default mode | 903 | -0.127 | -3.777 | 0.002 |
| default mode | limbic | 301 | -0.125 | -3.677 | 0.004 |
| limbic | limbic | 21 | -0.118 | -3.052 | 0.014 |
| dorsal attention | somatomotor | 399 | -0.116 | -2.653 | 0.024 |
| dorsal attention | dorsal attention | 171 | -0.115 | -2.875 | 0.024 |
| limbic | somatomotor | 147 | -0.114 | -2.839 | 0.024 |
| frontoparietal control | limbic | 147 | -0.102 | -2.733 | 0.024 |
| somatomotor | visual | 630 | -0.101 | -2.317 | 0.058 |
| frontoparietal control | salience / ventral attention | 441 | -0.093 | -2.317 | 0.058 |

**What this means.** 10 of 36 system pairs differ after correction, and **every one of them is lower in autism** (34 of 36 pairs are negative overall). Nothing is significantly raised. The largest difference is within the somatomotor system, and the limbic system is the most widely involved, differing from most others. A reduction spread across many systems rather than concentrated in one is the pattern long proposed for autism, recovered here from the same data the classifier reads, with the acquisition differences that usually confound it removed.

## 7. What the diagnostic label can and cannot explain

*Reported for completeness; not claimed as a contribution.*

Of the autistic participants, 213 have both a complete ADI-R algorithm and an ADOS total. The two disagree about whether the participant meets criteria in **15.0%** of those cases (13 on the interview only, 19 on the observation only, 181 on both, and none on neither).

Because at least one instrument is wrong wherever they disagree, that rate estimates the noise in the label a classifier is trained against, and with it an upper bound on attainable accuracy. Under the stated assumptions the bound lies between approximately **0.85 and 0.99**. Correcting the observed figures for label noise at the central rates moves accuracy from 0.707 to 0.729 and AUROC from 0.776 to 0.806, while observed accuracy lies 0.14 below even the lower end of that range. The shortfall is therefore not explained by uncertainty in the diagnosis alone, although at the most pessimistic rates the correction is larger, so the size of the remaining gap depends on assumptions these data cannot settle.

A point estimate is not reported, for three reasons. The two instruments are not independent raters of a common quantity: one queries developmental history through a caregiver, the other observes current behaviour. Agreement beyond chance is in fact slightly negative (Cohen's kappa = -0.078), which is inconsistent with the two-rater model the correction assumes. And two instruments applied to a single population do not identify their error rates whatever the cell counts, because the 2 x 2 table supplies three independent proportions against five unknowns (Hui & Walter, 1980); here, in addition, no participant was negative on both instruments. The correlation between instrument errors is therefore not estimable from these data rather than merely imprecise.

*SRS is caregiver-reported and often collected during the diagnostic workup, so it is not independent of diagnosis; the control-side rate is a lower bound and the ceiling correspondingly an upper one.*


---

# Supplementary results

These sections record the comparisons behind the design choices made above. None carries a claim of its own; each is included so that a choice reported in the main results rests on evidence rather than on preference.

## S1. All model comparisons (unadjusted DeLong p-values)

p-values are unadjusted, as stated in Methods Section 8.1; Benjamini-Hochberg q-values across the comparisons in this table are given for reference.

| comparison | difference in AUROC | p (unadjusted) | BH q | p < 0.05 |
|---|---|---|---|---|
| logistic (dense linear) vs xgboost [n=871] | +0.1396 | 0.0000 | 0.0000 | yes |
| logistic (dense linear) vs random forest [n=871] | +0.1461 | 0.0000 | 0.0000 | yes |
| logistic (dense linear) vs catboost [n=871] | +0.1327 | 0.0000 | 0.0000 | yes |
| ensemble (rank mean) vs phenotype only [n=871] | +0.1336 | 0.0000 | 0.0000 | yes |
| no harmonisation vs in-fold ComBat [n=871] | +0.0290 | 0.0208 | 0.0708 | yes |
| ensemble + phenotype (probability mean) vs ensemble (probability mean) [n=871] | +0.0043 | 0.0408 | 0.1065 | yes |
| rank=20 vs logistic (dense linear) [n=871] | -0.0214 | 0.0439 | 0.1065 | yes |
| ensemble (probability mean) vs dynamic: mean [n=1032] | +0.0085 | 0.0554 | 0.1178 | no |
| ensemble (probability mean) vs dynamic: mean [n=871] | +0.0089 | 0.0664 | 0.1254 | no |
| dynamic: mean vs tangent (static) [n=1032] | +0.0145 | 0.0974 | 0.1656 | no |
| ensemble (rank mean) vs dynamic: mean [n=871] | -0.0131 | 0.1926 | 0.2977 | no |
| ensemble (rank mean) vs dynamic: mean [n=1032] | -0.0110 | 0.2152 | 0.3048 | no |
| dynamic: mean vs tangent (static) [n=871] | +0.0102 | 0.2811 | 0.3676 | no |
| rank=50 vs logistic (dense linear) [n=871] | -0.0049 | 0.4280 | 0.5197 | no |
| tangent (static) vs correlation [n=871] | +0.0085 | 0.5684 | 0.6442 | no |
| tangent (static) vs correlation [n=1032] | +0.0036 | 0.7915 | 0.8409 | no |
| cc200 vs cc200+aal+ho [n=871] | +0.0017 | 0.8443 | 0.8443 | no |

**What this means.** A small gap between two numbers may be nothing more than which participants happened to land in which fold, so each comparison is tested. The linear model is separated from the fixed-configuration tree ensembles by a wide margin. The gain from combining the three descriptions is not separated from chance: under the probability-averaging rule used throughout, p = 0.066 in the strict cohort and p = 0.055 in the wider cohort, and the rank-averaging rule, reported for comparison, scores below the best single description. Several smaller differences are *not* distinguishable from chance, including the gain from windowed averaging on its own and the gap between using one atlas and three — those should not be presented as improvements.

## S2. Connectivity representations compared

| description | AUROC | sd | accuracy |
|---|---|---|---|
| dynamic: mean | 0.767 | 0.044 | 0.695 |
| tangent (static) | 0.756 | 0.048 | 0.692 |
| correlation | 0.750 | 0.040 | 0.683 |
| dynamic: mean+dispersion | 0.741 | 0.042 | 0.669 |
| dynamic: mean+velocity | 0.740 | 0.033 | 0.661 |
| dynamic (fused): mean+dispersion | 0.734 | 0.039 | 0.661 |
| dynamic (fused): mean+velocity | 0.727 | 0.037 | 0.669 |
| dynamic: mean+dispersion+velocity | 0.719 | 0.041 | 0.648 |
| dynamic (fused): mean+dispersion+velocity | 0.697 | 0.043 | 0.646 |

**What this means.** The windowed mean reaches a slightly higher AUROC than the whole-scan estimate (0.767 against 0.756, fold means), but the difference is not distinguished from chance (p = 0.28; Section S1), and it disappears when the same extra shrinkage is applied to the whole-scan estimate (0.767 against 0.769, p = 0.71; Section 4). It therefore reflects the regularisation applied to the windows rather than a benefit of splitting the scan. Adding how much connectivity varies during the scan, or how fast it changes, gave lower AUROCs (0.697 to 0.741 across the six combined conditions, against 0.767 for the windowed mean alone); these differences were not formally tested and are reported as a negative finding.

![Ways of describing connectivity](results/2026-08-24_23-58-52_full_study/figures/fig08_representation.png)

*Ways of describing connectivity. *

## S3. Classifier family, and compressing the weight matrix

| classifier | AUROC (participant-pooled) | AUROC (fold mean) | sd (folds) | accuracy (fold mean) |
|---|---|---|---|---|
| logistic (dense linear) | 0.757 | 0.756 | 0.048 | 0.692 |
| ridge | 0.755 | 0.755 | 0.047 | 0.693 |
| linear SVM | 0.753 | 0.753 | 0.048 | 0.687 |
| shrinkage LDA (PCA-200) | 0.741 | 0.739 | 0.052 | 0.669 |
| low-rank spectral (rank 20) | 0.735 | 0.737 | 0.044 | 0.672 |
| catboost | 0.624 | 0.623 | 0.042 | 0.597 |
| xgboost | 0.617 | 0.616 | 0.050 | 0.584 |
| random forest | 0.611 | 0.609 | 0.049 | 0.577 |

Participant-pooled AUROC is computed over all 871 out-of-fold predictions together; it is the value quoted in the summary and tested with DeLong's test. Fold mean and sd summarise the ten outer folds.

**What this means.** The linear and spectral methods land close together, and the three tree ensembles land far below. Only the logistic model was tuned; the tree ensembles used fixed configurations, so the contrast describes these configurations rather than the best attainable member of each model family.

![Ranking with significance](results/2026-08-24_23-58-52_full_study/figures/fig09_critical_difference.png)

*Ranking with significance. Methods joined by a bar cannot be told apart. The linear group is separated from the fixed-configuration tree ensembles.*

| components kept | AUROC | sd |
|---|---|---|
| rank=100 | 0.752 | 0.049 |
| rank=50 | 0.751 | 0.043 |
| rank=20 | 0.737 | 0.044 |
| rank=30 | 0.733 | 0.043 |
| rank=10 | 0.718 | 0.051 |
| rank=5 | 0.716 | 0.064 |
| rank=2 | 0.712 | 0.069 |
| rank=1 | 0.648 | 0.081 |

**What this means.** The model can be rewritten using a small number of whole-brain patterns instead of twenty thousand separate weights. Keeping fifty patterns leaves the ranking of participants statistically indistinguishable from the full model, which makes it far easier to inspect. The individual decisions do shift, however, so the compression is not entirely free even where the summary score is unchanged. Keeping twenty is a genuine step down.

![How much can be compressed](results/2026-08-24_23-58-52_full_study/figures/fig10_spectrum.png)

*How much can be compressed. Accuracy against the number of patterns retained.*

## S4. Atlas, fusion and harmonisation

| atlas combination | AUROC | sd |
|---|---|---|
| cc200 | 0.756 | 0.048 |
| cc200+ho | 0.755 | 0.050 |
| cc200+aal+ho | 0.752 | 0.045 |
| cc200+aal | 0.749 | 0.037 |
| aal+ho | 0.728 | 0.048 |
| ho | 0.717 | 0.054 |
| aal | 0.692 | 0.035 |

**What this means.** Using three different maps of the brain instead of one did not help; a single map is as good. The three maps apparently capture much the same information, so averaging them adds little.

A published meta-analysis reports the opposite ordering — studies using the AAL atlas performing better than studies using CC200 — so the comparison was repeated under both connectivity descriptions, holding participants, preprocessing, classifier and folds fixed. This is a separate five-fold comparison with its own regularisation sweep, so its values are not directly comparable with the nested-CV table above; only the ordering within it should be read:

| parcellation | regions | features | AUROC (correlation) | AUROC (tangent) |
|---|---|---|---|---|
| CC200 | 200 | 19900 | 0.734 | 0.738 |
| Harvard-Oxford | 111 | 6105 | 0.726 | 0.710 |
| AAL90 (cerebrum only) | 90 | 4005 | not computed | 0.695 |
| AAL116 (full) | 116 | 6670 | 0.681 | 0.688 |
| AAL cerebellum only | 26 | 325 | not computed | 0.513 |

**What this means.** The ordering here is the reverse of the meta-analytic one, and it holds under both connectivity descriptions, so it is not an artefact of the tangent embedding. It is not a size effect either: Harvard-Oxford has fewer measurements than AAL yet performs better. Two observations bear on it. CC200, with the most regions, gives the highest AUROC under both descriptions, although AUROC does not rise monotonically with region count (Harvard-Oxford, with 111 regions, exceeds the 116-region AAL atlas). And the 26 cerebellar regions of the full AAL atlas are at chance in this cohort (tangent AUROC 0.513); removing them raises the tangent AUROC from 0.688 to 0.695. The meta-analytic result compares whole studies rather than holding conditions fixed, and that same analysis found smaller studies reporting higher performance, so atlas choice there may stand in for study size.

| site correction | AUROC | sd | accuracy |
|---|---|---|---|
| no harmonisation | 0.756 | 0.048 | 0.692 |
| in-fold ComBat | 0.726 | 0.048 | 0.665 |

**What this means.** Statistically removing scanner-related differences made results worse, and the drop is larger than chance. The correction appears to remove some of the signal along with the scanner effect, so it is not used.

![Remaining design comparisons](results/2026-08-24_23-58-52_full_study/figures/fig11_ablations.png)

*Remaining design comparisons. *

## S5. Where the weight map falls

![The connections that matter most](results/2026-08-24_23-58-52_full_study/figures/fig12_connectome.png)

*The connections that matter most. Drawn on a see-through brain, viewed from three sides.*

![Grouped into brain systems](results/2026-08-24_23-58-52_full_study/figures/fig13_network_matrix.png)

*Grouped into brain systems. The same information summarised over the seven major brain networks.*

**What this means.** The weights fall on connections within recognised functional systems rather than scattering at random, so the weight map has anatomical structure. No Haufe transform (Haufe et al., 2014) is applied, so this describes how the decision is formed and not which measurements carry the group difference. Which systems, and how strongly, is answered directly in Section 6 rather than read off this figure: the difference is largest within the somatomotor system, with the default-mode network also affected but less strongly. An earlier version of this document attributed the decision mainly to the default-mode network, on a superseded model and without the network-level test that now settles it.

![Why one person was classified this way](results/2026-08-24_23-58-52_full_study/figures/fig14_shap_subject_waterfalls.png)

*Why one person was classified this way. Three participants: two clear cases and one the model was unsure about.*

**What this means.** The weights say which connections matter on average; this says which mattered for one person, which is the form an explanation has to take if it is to accompany an individual result. For the uncertain participant the evidence points both ways and cancels out, which is exactly why the model hesitates.

For a linear model this attribution has a closed form — the weight multiplied by how far the participant departs from the average — so it re-expresses the fitted weights per participant rather than providing independent evidence about them. The agreement with the coefficients (0.996) is therefore expected by construction, and is reported as a check that the implementation is correct, not as corroboration of the pattern itself.

**Do these patterns track how severe the autism is?** Across 120 tests against standard symptom scales, none survived correction for multiple testing.

## S6. Inside the somatomotor system (exploratory)

Everything in this section is exploratory: the system was chosen because it carried the largest difference in Section 6, so the regional ordering inside it is a description of that result, not an independent test of it.

| region | hemisphere | difference | z | q |
|---|---|---|---|---|
| Postcentral_L [96] | L | -0.280 | -5.458 | 0.000 |
| Postcentral_L [154] | L | -0.273 | -5.413 | 0.000 |
| Postcentral_R [28] | R | -0.269 | -5.321 | 0.000 |
| Postcentral_L [8] | L | -0.261 | -5.109 | 0.000 |
| Postcentral_R [65] | R | -0.256 | -4.873 | 0.000 |
| Paracentral_Lobule_L [21] | L | -0.254 | -4.915 | 0.000 |
| Precentral_R [60] | R | -0.238 | -4.649 | 0.000 |
| Postcentral_L [88] | L | -0.236 | -4.678 | 0.000 |
| Rolandic_Oper_R [185] | R | -0.231 | -4.445 | 0.000 |
| Paracentral_Lobule_L [111] | M | -0.224 | -4.330 | 0.000 |

**Does the ordering reproduce?** Splitting the participants in half repeatedly, refitting the nuisance adjustment inside each half and comparing the two orderings gives a correlation of **0.55** across the twenty-one regions. Measured the same way in the same splits, the ordering of the 210 individual connections inside the system reproduces at 0.38 and its top fifty at 0.34. Region-level reporting is therefore preferred on multiplicity — twenty-one comparisons rather than 19,110 — and modestly on stability, not because connection-level effects are unreproducible.

That figure only means something against a comparator of the same size. Drawing 100 random blocks of twenty-one regions from the whole brain and scoring them identically gives a median of **0.35** (range 0.06–0.63); 9% of them reach this system's value. The ordering is more stable than most equally sized blocks, but not categorically so.

The regions at the top are the postcentral gyrus, the paracentral lobule and the precentral gyrus — primary somatosensory and motor cortex, on both sides and to a similar degree. The effect is spread across the system rather than sitting in one place: every one of the twenty-one regions differs, and the strongest accounts for only a small share of the total.

**The motion check.** These regions lie along the central sulcus, where head movement writes its largest artefact, and autistic participants move more — so the analysis was repeated on within-site pairs matched on mean framewise displacement. In 280 matched pairs (560 participants) the group difference in motion is removed (p = 7.6 × 10⁻⁷ before matching, p = 0.28 after), and **21 of 21 regions remain significant** with the ordering preserved (rank correlation 0.89 against the unmatched result). The effects are *larger* after matching (-0.215 → -0.249 on average), as they were at the system level. Residual motion is not a sufficient explanation for this difference; it is not thereby excluded as a contributor.

| region (motion-matched) | hemisphere | difference | z | q |
|---|---|---|---|---|
| Postcentral_R [28] | R | -0.312 | -4.896 | 0.000 |
| Postcentral_R [65] | R | -0.301 | -4.561 | 0.000 |
| Postcentral_L [96] | L | -0.296 | -4.586 | 0.000 |
| Postcentral_L [154] | L | -0.295 | -4.639 | 0.000 |
| Precentral_R [60] | R | -0.289 | -4.472 | 0.000 |

*Limitations of this section.* This is a univariate contrast of residualised connections, computed without reference to the classifier, and licenses no statement about what the classifier uses. Hemisphere is assigned from the parcel centroid while the anatomical name comes from the atlas label of greatest overlap; for parcels straddling the midline the two can disagree. Results are reported under a single denoising choice.

## S7. Individual panels

Single-panel versions of figures that appear merged in the main results, kept because a reader checking one number should not have to read a composite.

![ROC curve](results/2026-08-24_23-58-52_full_study/figures/fig03_roc_pooled.png)

*ROC curve. How well the model ranks autism cases above controls, at every possible cut-off. Further above the diagonal is better.*

![Predicted probabilities by group](results/2026-08-24_23-58-52_full_study/figures/fig04_probability_distribution_ensemble.png)

*Predicted probabilities by group. The two groups separate but overlap in the middle. Participants in that overlap are the ones the model is unsure about.*

![Are the stated probabilities honest?](results/2026-08-24_23-58-52_full_study/figures/fig05_calibration_pooled.png)

*Are the stated probabilities honest? Predicted probability against how often the model was actually right. Closer to the diagonal is better.*

![Accuracy when answering fewer cases](results/2026-08-24_23-58-52_full_study/figures/fig06_coverage_pooled.png)

*Accuracy when answering fewer cases. Moving left means answering only the more confident participants.*

![The same correlations computed two ways](results/2026-08-24_23-58-52_full_study/figures/fig16_fig_severity_pooled_vs_within.png)

*The same correlations computed two ways. Asterisks mark associations reaching p < 0.05.*

![Whole-scan connectivity across shrinkage levels](results/2026-08-24_23-58-52_full_study/figures/fig17_fig_ridge_sweep.png)

*Whole-scan connectivity across shrinkage levels. The dashed line is the windowed representation at its own shrinkage level.*

![The systems, and where they differ](results/2026-08-24_23-58-52_full_study/figures/fig19_fig_networks.png)

*The systems, and where they differ. Regions coloured by the system they belong to; the matrix gives the difference between autistic and non-autistic participants for each pair, with asterisks on the pairs that survive correction.*

![Performance site by site](results/2026-08-24_23-58-52_full_study/figures/fig21_per_site_loso.png)

*Performance site by site. Each bar is a site the model had never seen during training. Computed from the adopted ensemble's own stored predictions.*


---

## Limitations

**Scope of the cohort.** Participants range from 6.5 to 58 years with a median of 14.8 and only one below seven, so nothing here bears on early identification. The sample is 727 male to 144 female, the comparison group is typically developing rather than clinically referred, and every label is a diagnosis obtained before the scan; the results therefore describe discrimination within an aggregated research cohort, not performance in a differential-diagnostic setting.

**Denoising and interpretation.** All results are obtained under a single preprocessing strategy, without global signal regression, so the direction of any group difference is conditional on that choice. Group differences are univariate contrasts of residualised connections computed without reference to the classifier, and therefore support statements about how the groups differ but none about which measurements the classifier uses.

**Abstention has limited reach on these data.** The per-site rule holds its target but answers few cases — 170 of 871 (19.5%) with the threshold set on the whole sample, 143 of them at a single site, and 13.7% on average across held-out half-splits — and seventeen of twenty sites answer no one. This characterises the cohort rather than the method.

**Approaches evaluated and not adopted.** Site harmonisation showed no evidence of benefit; within-scan variability and rate of change lowered accuracy; three atlases performed no better than one; a rank constraint on the weight matrix matched the dense model without improving on it, and the premise motivating it does not hold; demographic variables added about 0.004 AUROC; restricting the model to the most discriminating regions lowered performance at every list length; the extracted components do not track symptom severity within the autism group; and neither the scan-duration nor the sample-size axis shows evidence of saturation, so the two cannot be separated on these data. Each is reported because a negative result on a public dataset bounds what the next study can expect.
