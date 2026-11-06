# Edit log: calibration and interpretation contributions (2026-10-01)

Source of every new number: `results/latest_study/review/` (ladder run `ladder/RUN_REPORT.md`, 168 files hashed).
Backup of the files before editing: `archive/20261001_pre_calibration_edit/`.
New text is not wrapped in `\reviewhl`, because `soul`'s `\hl` breaks on the math in it; every change is listed here.
The title and `\shorttitle` were deliberately not changed (decision pending).

| file | change |
|---|---|
| main.tex | abstract rewritten (248 words; removes the stale reliability-oriented framing and the calibration tag); five highlights of at most 85 characters; keyword "Selective prediction" replaced by "Interpretability" |
| sections/introduction.tex | new paragraph on the two questions about the fitted model (after the problem statement); six to eight analyses; recalibration labelled primary descriptive, attribution and reproducibility secondary, both stated as added after the primary results; two new contributions (7 calibration, 8 attribution) |
| sections/related_work.tex | calibration and interpretation sentences with Van Calster 2019, Steyerberg 2010, Platt 1999, Zadrozny and Elkan 2002, Haufe 2014; closing statement no longer mentions low-rank patterns or selective prediction |
| sections/method.tex | overview wording (nested at the outer level), ten subsections, six to eight analyses; intercept-penalty sentence; "nested scope" paragraph now points to the fully nested sensitivity analysis; random-effects sentence no longer says "new site"; new subsubsections: fully nested penalty selection, primary descriptive assessment of calibration (retitled), primary descriptive recalibration within training partitions; new subsection: interpretation of the linear members (five subsubsections); SHAP paragraph removed (superseded; it had no Results); analysis status and multiplicity paragraphs extended; reproducibility paragraph extended |
| sections/result.tex | opening lists eight contributions; fully nested paragraph and cohort-flow clarification (two different sets of 1,035); Freedman-Lane effect-scale clarification; captions of Fig. 3a, 3b, 3c corrected; two new subsections (calibration and recalibration; attribution to system pairs) with Tables 12-14 and Figures 7-8; calibration paragraphs removed from the secondary results |
| sections/discussion.tex | eight findings; implications; two new interpretation paragraphs; stale inline tag replaced; secondary-results paragraph trimmed; new limitations paragraph; "requirements" paragraph rewritten (two items it listed as open were already closed); future-work sentence |
| sections/conclusion.tex | two sentences for the new findings |
| references.bib | seven entries appended (Haufe 2014, Van Calster 2019, Steyerberg 2010, Platt 1999, Zadrozny and Elkan 2002, LIBLINEAR, Westfall and Young 1993) |
| figures/ | new fig_calibration, fig_interpretation (png and pdf); fig_performance regenerated (descriptive interval wording, DeLong/Wilson label, colours consistent with the guarantee figure) |
| (tables) | the three new tables are written directly in sections/result.tex; `review_tex_tables.py` generates their LaTeX into results/latest_study/review/tables_tex/, outside the journal folder (the separate tables_generated/ folder was removed 2026-10-06) |

Build check (scratch copy with a `soul` stub and `unsrtnat`): 0 errors, 0 undefined references or citations, 59 pages (49 before),
the only overfull box is the pre-existing front-matter one (117 pt). Table audit: 192 of 192 cells of the existing tables match the stored outputs.

## Highlighting
All text added or changed in this edit is wrapped in `\addhl{...}` (light green; defined in main.tex next to `\reviewhl`, which stays yellow
for the earlier review edits). Captions of the new tables and figures, and the corrected captions of Fig. 3, are highlighted. Not highlighted,
because `soul` cannot hold them: headings, equations, table bodies, `\label`, `\item`, cross-references and citations (a reference inside a
highlighted sentence is a small gap in the colour). Verified with the real `soul` package (0 errors) and with the local stub. To remove the
marking, replace `\addhl{X}` by `X`.

## Figure files to upload to Overleaf (figures/)
New: `fig_calibration.png`, `fig_calibration.pdf`, `fig_interpretation.png`, `fig_interpretation.pdf`.
Replaced: `fig_performance.png`. All other figure files are unchanged.

# 2026-10-02 figure order and abbreviation pass

## Figure order (Methods diagrams)
First mentions of Figs 1-3 now follow the figure numbers (p5, p15, p20 in the scratch build); the premature Fig. 3 pointer at the end of the
study-design overview was replaced by a pointer to Section 3.4.1. The Fig. 4 caption no longer points to Fig. 7. Captions of Figs 1-3 define the
circled steps, the lock/arrow key, the system codes and the classifier abbreviations that their diagrams draw.

## Abbreviation rule
Each abbreviation is defined once, at its first use in the running text (full form followed by the acronym), and only the acronym is used afterwards.
The abstract, highlights and keywords stand alone (full forms, no acronyms; "fMRI" kept as a keyword). Section headings stay spelled out.
Definition sites: ASD, rs-fMRI, ABIDE, ABIDE-I, ADOS, ADI-R, IQ, FD, CV, LOSO, AUROC (Introduction); C-PAC, CC200, QC, the eight system codes
(VIS SMN DAN SAL LIM FPN DMN SUB), BH-FDR, SVC, LDA, PCA, SD, IQR, ECE, SRS, L-BFGS (Methods); SHAP, AI (Related Work); ABIDE-II (Discussion).
Removed: repeated definitions of ASD, rs-fMRI, ABIDE-I, AUROC, ECE, FD, LOSO, SD; the acronym FIQ (table rows now read "Full-scale IQ ...") and the acronym CV in captions is now the defined CV.
About 160 spelled-out occurrences became acronyms (leave-one-site-out 35, cross-validation 18, framewise displacement 16, standard deviation 7,
Benjamini-Hochberg 8, quality control 10, expected calibration error 6, plus single sites). All are wrapped in `\addhl{}` except in headings, table bodies and
text that was already highlighted.
Table 1 caption now defines IQR; the "Calibration error" column of the published-comparison table is headed ECE.
ADOS-2 in the Introduction is now ADOS (the later text uses ADOS); the author's blue note about ADOS-Generic is kept.
Abstract: 250 words (no slack). "ABIDE-I dataset" is spelled "first Autism Brain Imaging Data Exchange release"; "IQ" is spelled "intelligence quotient".

## Figure files to upload to Overleaf (figures/)
Replaced: `fig_groupdiff.png` (Fig. 5: system codes in the canonical order instead of "salience/VA", "subcort/cbl"), `fig_corrections.png`
(Fig. 6: "2,000 connections of 19,900"), `fig_interpretation.png` and `.pdf` (Fig. 8: SAL instead of VAN; legend "static tangent",
"windowed tangent" instead of "tangent", "dynamic mean"). Sources: `viz_summary.py`, `viz_calibration_interpretation.py`.
Also upload main.tex and sections/*.tex (all six).

# 2026-10-02 graphical abstract rebuilt (version 2)
Generator: `viz_graphical_abstract.py` (version 1 kept outside the repo). Canvas 190 x 105 mm (1.81:1), text >= 7 pt, 78 text objects, 0 layout problems.
Band 1: the pipeline (871 participants, quality control, 200 regions, three views, training-only logistic models, averaged probability, AUROC 0.776 and 0.774 at an unseen site).
Band 2: contribution 1 (calibration-in-the-large, native against intercept-recalibrated, pooled and site withheld, with site-clustered 95% intervals; AUROC 0.776 to 0.776; "site-level gaps remain") and contribution 2 (share of the score separation per system-pair, default mode outlined; replication across independent halves for system pairs, regions and connections against the permutation chance band).
Band 3: contributions 3 to 5 (somatomotor reduction in all 21 regions stronger after motion matching; AUROC by cognitive ability of the autistic participants and the correlation with the Social Responsiveness Scale in both groups against autism only; classifier family and static, shrinkage-matched static and windowed on one AUROC axis).
Removed: the leakage-audit tile (0.99 "not real"), the old tagline, the lavender/cream tile coding, the big AUROC as hero, and every truncated-baseline bar.
Encoding: dots, intervals and dumbbells on labelled axes; the only bars are the two correlations, which start at zero. Sources, values and alt text: `results/latest_study/figures/fig_graphical_abstract.sources.json`.
Checks: block shares sum to 1.000; counts add up (403 + 468 = 871); static, shrinkage-matched static and windowed AUROC 0.757, 0.770, 0.767 (difference -0.0025, p = 0.709) as in the paper; deuteranopia simulation reviewed; text contrast 4.7:1 or more on the panel tint.
Files to upload to Overleaf (figures/): `fig_graphical_abstract.png` and `.pdf` (and `.svg` if kept). For the separate Elsevier graphical-abstract upload: `results/latest_study/figures/fig_graphical_abstract.tiff` (600 dpi, LZW).
Fix (same day): the dome glyph in the "training-only logistic models" box overran the box's lower border (its base curve extends 0.10 of its height below the nominal box). It is now sized and centred so its whole extent lies inside the box; the figure and TIFF were regenerated.

# 2026-10-06 reviewer-comment fixes (main.tex, introduction.tex and related_work.tex NOT touched; backup in archive/20261006_pre_reviewer_fixes/)
New analysis script: `review_checks.py` (outputs in results/latest_study/review/): ensemble_logit_approximation.csv, random_effects_weighting.json,
loso_accuracy_weighting.json, motion_somatomotor.json, cohort_exclusion_stats.json, windowing_matched_difference.json, table_system_pairs_fl.tex.
Methods: Freedman--Lane is the principal group-difference test, within-site permutation of residualised connections the sensitivity analysis; regional analysis
labelled exploratory and descriptive; somatomotor motion check; approximation check of the three-member average against the probability-mean ensemble;
fold-mean versus participant-pooled weighting; random-effects weighting; site-stratified halves stated; wider-cohort exclusion comparison; windowing bootstrap interval;
"Classifier family" -> "Classifier configurations"; selective prediction, atlas combination and label-noise paragraphs moved to supplementary.tex.
Results: system-level results rewritten around Freedman--Lane (14 of 36, 27 of 36 matched; the 10 and 26 of the sensitivity analysis are subsets; four pairs at q=0.0499); new Table 8;
regional results descriptive; somatomotor-specific motion argument (all 210 connections positive, mean rho 0.18; shift +0.006 against observed -0.046); member scores exact and the view
average an approximation (rho 0.999, AUROC 0.7759 vs 0.7760, separation 1.07 vs 1.00 logit); within-cohort reproducibility; LOSO accuracy weighting sentence; random-effects weighting sentence;
wider-cohort statistics (164 excluded: mean FD 0.239 vs 0.109, 62.2% vs 46.3% autistic, 98.8% vs 36.2% rater notes); calibration mechanism softened; windowing "no benefit detected", interval -0.015 to +0.009;
section 4.10 condensed to one paragraph pointing to the supplement; captions of Figs 4, 5, 6 updated.
Discussion/Conclusion: matching wording; new subsection "Assessment against criteria for trustworthy classification" with Table 15; Conclusion states which criteria are met.
Figures: fig_groupdiff (Freedman--Lane stars, unmatched IQ panel), fig_performance (abstention panel removed), fig_corrections (label-ceiling panel removed, interval on panel b),
fig_graphical_abstract (png, pdf, svg; wording), new fig_label_ceiling (supplement only). Sources: viz_summary.py, viz_graphical_abstract.py.
Supplement: supplementary.tex (standalone; compile separately with the same references.bib).
Build (stand-in soul package): 0 errors, 0 undefined references, 53 pages (60 before); supplement 5 pages.
Not applied (main.tex and Introduction are protected): see the wording suggestions given in the chat of this date.

# 2026-10-06 (later) requested edits, figure fixes, panel outlines, graphical abstract, PDFs
Requested edits applied (explicit instruction): main.tex abstract (3 edits) and highlights 2, 4, 5; introduction.tex lines 36, 38, 42, 44. The abstract is now 252 words.
Bibliography: references.bib (updated by the author) checked against every \cite in main.tex, sections/*.tex and supplementary.tex: 56 entries, 0 missing keys.
Figures regenerated and installed: fig_guarantee (labels off the dashed lines, "nominal 90%" inside the axes), fig_performance (random-effects note moved to the empty upper left on a white box),
fig_calibration (same key on panels a and b, bold block headings in c, key and heterogeneity note moved clear of the data in d; axis labels state what the bars are).
Panel outlines: viz_style.panel_bg now draws an outline in a darker, more saturated shade of the panel tint (0.8 pt); viz_style.canvas(bleed=PANEL_BLEED) adds a 1.6 mm margin where a panel meets the page edge,
so labels that start at the old edge are not touched by the stroke. Regenerated: fig_designs, fig_representation, fig_overview (the figures are now 168.2 mm wide before scaling to the text width).
Graphical abstract rebuilt around the title and Introduction: pipeline ending in "Discrimination (AUROC 0.776; 0.774 unseen site)", row "Beyond discrimination: reliability of the fitted model" (panels 1 and 2),
row "what measured performance depends on" (panels 3 to 5), panels numbered as the five findings, chip "discrimination unchanged (AUROC 0.776)" replacing "0.776 -> 0.776", tagline "Beyond discrimination: toward trustworthy
connectome-based autism classification.", panel outlines.
PDFs: main.pdf (53 pages) and supplementary.pdf (5 pages) were rebuilt locally and placed in this folder (the previous main.pdf is archive/20261006_pre_reviewer_fixes/main.pdf.old).
They were built with a stand-in for the soul package (no highlight colours) and the unsrtnat bibliography style, so Overleaf output will differ in those two respects.
New analysis script (feasibility only, not yet in the paper): exp_trust_checks.py -> results/latest_study/review/trust_checks.json.

# 2026-10-06 (trust round) conditions for trustworthy use
Pre-registration: TRUSTWORTHINESS_PLAN.md (analyses A to D, written before the site-disjoint run; outcome appended). Scripts: exp_trust_checks.py (A to C -> review/trust_checks.json and CSVs), exp_interpretation_splithalf_sitedisjoint.py (D), viz_trust.py (fig_trust).
Results: A. site offsets follow the site's autistic proportion (Spearman 0.79, 10 sites); shifting to the true proportion removes the heterogeneity (I^2 63% to 0%) but only if the proportion is known to within about 0.05; it cannot be recovered from the predictions (EM correlates 0.22). B. thresholds fixed on the training partition reach their targets (specificity 0.82, sensitivity 0.80 pooled; 0.84 and 0.80 with a site withheld); net benefit over treating everyone 0.05, 0.11, 0.25 at 0.30, 0.40, 0.50. C. AUROC by age, motion, site size; offsets by stratum do NOT follow the stratum's autistic proportion (an earlier statement to the contrary was wrong and was corrected). D. site-disjoint halves: criterion met in both variants (system-pair 0.54/0.63/0.57 raw, 0.60/0.56/0.53 adjusted, permutation 95th percentile at most 0.10).
Methods: new subsection "Conditions for trustworthy use" (base-rate shift Equation eq:base-rate-shift, operating points, strata) and "Secondary reproducibility across sites". Results: subsection "Conditions for trustworthy use" with Figure fig:trust; site-disjoint sentences in the attribution subsection (replacing the statement that both halves share every site).
Discussion: Table 15 (tab:trustworthiness) and its subsection removed at the author's request (no verdict table); replaced by a short prose subsection "Conditions for trustworthy use"; limitation on halves updated. Conclusion: status sentence rewritten.
Introduction: contribution 1 ("traced to the autistic proportion of the site") and one sentence designating the base-rate, operating-point and cross-site analyses as secondary and the stratified validation as exploratory. Abstract and highlights: highlight 1 only at this stage (superseded by the entry below).
Highlights: new text of this round is wrapped in \addhl (green) in main.tex (highlight 1), introduction.tex, method.tex, result.tex (caption of fig:trust only for the figure), discussion.tex and conclusion.tex; baseline archive/20261006_pre_trust_additions/. Headings, equations and cross-references are left plain because soul cannot contain them.
Folder tables_generated/ removed: the three tables are in sections/result.tex; review_tex_tables.py now writes to results/latest_study/review/tables_tex/.
Build (stand-in soul that colours highlighted text dark green, unsrtnat): 0 errors, 0 undefined references, 55 pages; supplement 5 pages.

# 2026-10-06 (final) abstract, Introduction contributions and short title, at the author's request
main.tex: \shorttitle{Trustworthy Autism Classification from Functional Connectomes} added (the running header showed the template placeholder). Abstract rewritten to 246 words (limit set by the author: 240 to 250):
two additions ("residual between-site offsets follow the site's autistic proportion"; "(including halves that share no site)") and trims of redundant words only (no figure or claim removed): "brain", "also", "predictive", "within the same framework",
the two validation sentences merged into one with the AUROCs, "remains evident and" -> "persists and", "within the autistic group" -> "within autistic participants".
introduction.tex, contributions: (1) offsets "found to follow" the site's autistic proportion (Spearman 0.79, 10 sites), the shift to that proportion removes the heterogeneity (I^2 63% to 0%) only if the proportion is known to within about 0.05, and thresholds fixed on the training partition reach their targets at unseen sites (specificity 0.84, sensitivity 0.80 for targets of 0.80; 46% autistic cohort, not screening use);
(2) system-level agreement 0.60--0.70 between halves and 0.53--0.63 between halves that share no site; (3) 27 of 36 system pairs lower in the matched sample against 14 of 36 in the full cohort (worded without a motion effect, because the Results state that the matched sample also differs in size and composition).
Highlights (soul, green) were stripped and re-applied against archive/20261006_pre_trust_additions/ so that they mark this round's text, the abstract and Introduction included. The graphical abstract already carries the chip "site gaps follow base rate" and was not changed.
Build: 0 errors, 56 pages (one more than before because of the longer Introduction contributions).
