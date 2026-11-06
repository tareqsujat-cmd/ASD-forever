# Plan for the trustworthiness analyses (written 2026-10-06, before the site-disjoint run)

Purpose: the title and Conclusion speak of trustworthy classification, and the Discussion lists criteria that the study did not establish. This file fixes, before the last analysis is run,
what will be reported and what counts as a result. It is part of the analysis record and is not part of the manuscript.

## Status of the evidence
- Analyses A to C below were first run as feasibility checks on the stored out-of-fold predictions (`exp_trust_checks.py`) before this plan was written. Their results were seen. They are therefore
  reported as secondary or exploratory, not as confirmatory, with the evidence-status wording already used in the manuscript (added after the primary discrimination results were examined).
- Analysis D (site-disjoint reproducibility) had not been run when this file was written. Its success criterion below is pre-specified and its result is reported whichever way it falls.

## A. New-site calibration (leave-one-site-out, the 10 sites with at least 30 participants)
Quantities: site-level calibration-in-the-large (offset model, Wald interval), DerSimonian-Laird I^2 and tau, mean absolute offset, number of sites whose interval excludes zero.
Scenarios: native; training-only recalibration (current); base-rate shift using the site's autistic proportion (oracle, a diagnostic of the mechanism and not a deployable result);
the same shift when the assumed proportion is wrong by -0.20 to +0.20 (tolerance curve); base-rate shift or local intercept estimated from k = 10 or 15 labelled participants; label-free estimation by EM (reported as failed if unstable).
Claim allowed: heterogeneity is explained by the site's base rate, and calibration at a new site requires that base rate. Claim not allowed: a deployable label-free or few-label correction unless it reduces heterogeneity relative to the same remainder without it.

## B. Operating point
Rules fixed on the training partition only (inner cross-fitted probabilities, after training-only recalibration): specificity at least 0.80; sensitivity at least 0.80; Youden index; fixed 0.5 for reference.
Quantities: realised sensitivity and specificity on the held-out fold (pooled and leave-one-site-out) with site-clustered bootstrap intervals; prevalence-adjusted predictive values at 3, 10, 25 and 46 per cent;
net benefit against treating everyone, with the native and the recalibrated probabilities, at threshold probabilities 0.30, 0.40 and 0.50, and a decision curve from 0.20 to 0.70.
Claim allowed: a threshold fixed on training data reaches its target on held-out data and at unseen sites. Claim not allowed: screening utility.

## C. Stratified validation
Strata fixed in advance: age (below 12, 12 to 17, 18 and above), mean framewise displacement (thirds), site size (below 40, 40 to 79, 80 and above); sex and IQ bands are reported in the main text already.
Quantities: AUROC with a site-clustered bootstrap interval and calibration-in-the-large (native, recalibrated, and after a shift by the stratum's own autistic proportion).
Clinically referred comparison groups are not available in ABIDE-I and are stated as a limitation.

## D. Site-disjoint reproducibility of the explanations
Design: 40 random divisions of the 20 sites into two halves of 10 sites each (so no site is in both halves), with each half holding 42 to 58 per cent of the participants and autistic proportions within 0.08 of each other.
Each half is analysed on its own, from the tangent reference and the scaler to the classifier (penalty 0.001, as in the primary analysis). Two variants: (raw) the classifier and the structure coefficients as in the primary analysis;
(adjusted) site indicators, age, sex and mean framewise displacement are regressed out of every feature within the half before the classifier is fitted, so the structure coefficients describe connectivity net of these variables.
Null: 30 permutations of diagnosis within site. Agreement is the Spearman correlation between the two halves, at connection, region and system-pair level, for each of the three views.
Pre-specified success criterion: the system-pair structure-coefficient agreement exceeds the 95th percentile of the permutation null in at least 2 of the 3 views. It is evaluated separately for the two variants.
If it is not met, the paper states that system-level patterns do not reproduce across sites.

## Outcome of D (run completed 2026-10-06, 40 divisions, 30 permutations; results/latest_study/review/interpretation_splithalf_sitedisjoint.csv and sitedisjoint_summary.json)
The criterion was met for both variants (3 of 3 views each). System-pair structure-coefficient agreement, raw: 0.54 (correlation), 0.63 (static tangent), 0.57 (windowed) against permutation 95th percentiles of 0.09, 0.07, 0.10;
lowest single division 0.14. Adjusted for site, age, sex and mean framewise displacement: 0.60, 0.56, 0.53 against 0.06, 0.04, 0.04. Connection level, raw: 0.46, 0.15, 0.25. System-pair sign agreement, raw 63/75/67 per cent, adjusted 93/76/81 per cent, permutation 49 to 53 per cent.
Reported in Results (attribution subsection) and carried to the Discussion, Limitations and Conclusion. At the author's request the abstract and the Introduction contributions were later updated to carry this result (see EDIT_LOG.md); the highlights and the graphical abstract were not changed for it.
