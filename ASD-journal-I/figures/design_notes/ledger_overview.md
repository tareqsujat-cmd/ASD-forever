# Element ledger: Figure 1 (fig_overview)

This ledger lists every string, number, symbol and equation reference in the previous figure (`ASD-journal-I/figures/fig_overview.png`; that figure and its script are backed up in `scratchpad/overview_backup/`). Each item has its new location:
- **in figure**: kept as a label, tag or glyph;
- **caption**: moved to `caption_overview.tex`;
- **Methods text**: already stated in `sections/method.tex`, with the sentence quoted.

The last column is a tick box for the pre-submission check.

## Panel (a): cohort and time series

| # | Old element | New location | Where exactly / quote | ✓ |
|---|---|---|---|---|
| 1 | "(a)" panel letter | in figure | bold lowercase "a", 8 pt | ☐ |
| 2 | "ABIDE-I release" | in figure + caption | label "ABIDE-I" under the dataset cylinder; "ABIDE-I release" in caption (a) | ☐ |
| 3 | "1,112 participants" | in figure + caption | tag "1,112 · 17 sites"; caption (a) "which contains 1,112 participants" | ☐ |
| 4 | "17 sites" | in figure + caption | same tag; caption (a) | ☐ |
| 5 | "20 site labels" | caption + glyph | caption (a) and (c); the leave-one-site-out bar has 20 cells | ☐ |
| 6 | "Quality control" | in figure | label "quality control" under the funnel glyph, step ① | ☐ |
| 7 | "every required rating / of three raters must pass" | caption + Methods text | caption ①; Methods: "This rule is applied so that the primary analyses include only scans that pass the quality criterion of every rater." | ☐ |
| 8 | "241 records excluded" | in figure + caption | tag "−241"; caption ① "241 records are excluded" | ☐ |
| 9 | "Strict cohort" | in figure | label "strict cohort" under the dot cloud | ☐ |
| 10 | "871 participants" | in figure | tag "871" | ☐ |
| 11 | "403 autistic" | in figure | "403 ●"; bottom key "● autistic" | ☐ |
| 12 | "468 non-autistic" | in figure | "468 ○"; bottom key "○ non-autistic" | ☐ |
| 13 | "Regional time series" | in figure | label "regional time series" | ☐ |
| 14 | "C-PAC filt_noglobal" | caption + Methods text | caption ②; Methods: "The imaging inputs are taken from the \texttt{filt\_noglobal} derivative of the Configurable Pipeline for the Analysis of Connectomes (C-PAC)." | ☐ |
| 15 | "0.01–0.10 Hz band-pass" | caption + Methods text | caption ②; Methods: "After nuisance regression, band-pass filtering from 0.01 to 0.10 Hz is applied." | ☐ |
| 16 | "CC200 atlas, 200 regions" | in figure | label "CC200 atlas" + tag "200 regions" | ☐ |
| 17 | Brain renders (sagittal and axial nilearn `plot_roi` slices) | in figure (changed) | one lateral surface projection of the same CC200 atlas (loaded with nilearn), all 200 parcels coloured by system (the four regions excluded only from the group-difference analyses, 3 LIM and 1 SUB, keep their system colour because all 200 enter prediction). The axial view is dropped; it added no new information | ☐ |
| 18 | "CC200 parcels by system" (legend heading) | in figure + caption | one row of 8 colour chips; caption (a) "Parcels and traces are coloured by system" | ☐ |
| 19 | "VIS" / "visual" | in figure + caption | chip "VIS" (legend and matrix rows); "visual (VIS)" in caption | ☐ |
| 20 | "SMN" / "somatomotor" | in figure + caption | chip "SMN"; "somatomotor (SMN)" in caption | ☐ |
| 21 | "DAN" / "dorsal attention" | in figure + caption | chip "DAN"; caption | ☐ |
| 22 | "SAL" / "salience/ventral attn." | in figure + caption | chip "SAL"; "salience/ventral attention (SAL)" in caption | ☐ |
| 23 | "LIM" / "limbic" | in figure + caption | chip "LIM"; caption | ☐ |
| 24 | "FPN" / "frontoparietal control" | in figure + caption | chip "FPN"; caption | ☐ |
| 25 | "DMN" / "default mode" | in figure + caption | chip "DMN"; caption | ☐ |
| 26 | "SUB" / "subcortical/cerebellar" | in figure + caption | chip "SUB"; caption | ☐ |
| 27 | Coloured trace labels VIS, SMN, DAN, DMN, SUB (coloured text) | in figure (changed) | the five traces keep their system colours; the coloured text is replaced by the black-text legend chips | ☐ |
| 28 | Five real time series of the example participant | in figure | time-series glyph with the same five representative regions (highest system fraction, excluded regions skipped) | ☐ |
| 29 | "K = 6" / "windows" | in figure + caption | tag "K = 6, L = 50"; six translucent bands on the traces; caption ② | ☐ |
| 30 | Window bars at the real starts 0, 25, 50, 76, 101, 126 (Eq. 6) | in figure + caption + Methods text | bands drawn at these real positions; caption ② "placed by Equation (6)"; Methods: "The window positions are given in Equation~\eqref{eq:6}, which spaces the windows evenly from the start to the end of each scan." | ☐ |
| 31 | Axis ticks "1", "50", "100", "176" | caption | the axis is removed; caption ② "the 176 volumes of one example participant" | ☐ |
| 32 | "volume (example participant, 176 volumes)" | caption | caption ② | ☐ |
| 33 | Grey arrows between the four boxes and to the time series | in figure | grey data-flow arrows between the five icons | ☐ |

## Panel (b): representations

| # | Old element | New location | Where exactly / quote | ✓ |
|---|---|---|---|---|
| 34 | "(b)" | in figure | bold "b" | ☐ |
| 35 | "Correlation" (header) | in figure | orange member tab + "Correlation" | ☐ |
| 36 | "Eq. (1)" (correlation) | caption | caption (b) "Shrinkage correlation is computed by Equation (1)" | ☐ |
| 37 | "Ledoit–Wolf covariance / converted to correlation" | caption + Methods text | caption (b) "shrinkage correlation"; Methods: "The covariance estimate is then converted to a correlation matrix by dividing each covariance by the product of the standard deviations of the two regions involved." | ☐ |
| 38 | "Static tangent" | in figure | green member tab + "Static tangent" | ☐ |
| 39 | "Eqs. (1)–(4)" | caption | caption (b) | ☐ |
| 40 | "whole-scan covariance mapped / at the training reference" | caption + Methods text | caption (b) "The tangent references are estimated from an illustrative training partition"; Methods: "Training and held-out covariances are then projected into the tangent space with the training reference." | ☐ |
| 41 | "Windowed tangent" | in figure | purple member tab + "Windowed tangent" | ☐ |
| 42 | "Eqs. (1)–(7), r = 0.15" | caption | caption (b) "by Equations (1)–(7) with r = 0.15" | ☐ |
| 43 | "K = 6 windows of L = 50 volumes" | in figure + caption | tag "K = 6, L = 50" in (a); caption ② | ☐ |
| 44 | "mean of projected windows" | caption + Methods text | caption (b), Equation (7) in the range (1)–(7); Methods: "First, the mean in Equation~\eqref{eq:7} averages the window vectors and represents the average location of the window trajectory." | ☐ |
| 45 | Three real 200 × 200 matrices, ordered by system, diagonal blank | in figure | same computed values (cache identical to the representation-figure cache to 1e-13) | ☐ |
| 46 | System colour strips on all three matrices | in figure | kept on all three (left and top) | ☐ |
| 47 | Row labels VIS…SUB (first matrix) | in figure | kept once, with leader lines | ☐ |
| 48 | Block lines between systems | in figure | kept | ☐ |
| 49 | Colourbar "−0.7 · 0 · 0.7", label "correlation" | in figure (key) + caption | one shared "− 0 +" key; "±0.7" in caption (b) | ☐ |
| 50 | Colourbar "−0.22 · 0 · 0.22", label "tangent coordinate" | caption | caption (b) "±0.22"; the axis name is covered by "static tangent" | ☐ |
| 51 | Colourbar "−0.096 · 0 · 0.096", label "mean tangent coordinate" | caption | caption (b) "±0.096" | ☐ |
| 52 | Bus arrows from the time series to the three headers | in figure | grey bus with three drop arrows | ☐ |
| 53 | Old caption "The tangent references are estimated from an illustrative training partition." | caption | caption (b), with "784 of the 871 participants" added from Figure 2 | ☐ |

## Panel (c): ensemble and validation

| # | Old element | New location | Where exactly / quote | ✓ |
|---|---|---|---|---|
| 54 | "(c)" | in figure | bold "c" | ☐ |
| 55 | "upper triangle vectorised: 19,900 features per representation" | in figure + caption | three real feature-vector strips + tag "19,900"; caption (c) "The upper triangle of each matrix is vectorised into 19,900 features." | ☐ |
| 56 | Lane chips "Correlation", "Static tangent", "Windowed tangent" | in figure (changed) | member-colour tabs + coloured strands; the names appear once, in (b) | ☐ |
| 57 | "Scaling" (×3) | in figure | μ/σ glyph, step ③ | ☐ |
| 58 | "Eq. (10)" (×3) | caption | caption ③ | ☐ |
| 59 | "ℓ2-logistic" (×3) | in figure + caption | sigmoid glyph, step ④; caption ④ "ℓ2-regularised logistic regression" | ☐ |
| 60 | "Eq. (11)" (×3) | caption | caption ④ | ☐ |
| 61 | Per-member (separate) scaler and classifier, implied by three rows | caption + Methods text | caption ③/④ "Each member is standardised with its own training statistics … fitted to each member"; Methods: "In addition, separate scalers are fitted for the correlation, static tangent, and dynamic tangent members, so that each member is standardised with its own training statistics." | ☐ |
| 62 | "Average / of the three / probabilities" | in figure + caption | merge node "p̄", step ⑤; caption ⑤ "averaged with equal weights" | ☐ |
| 63 | "Eq. (12)" | caption | caption ⑤ | ☐ |
| 64 | "C ∈ {0.001, 0.01, 0.1, 1.0} per representation," | in figure + caption | inset tag "C: 0.001…1" (Arial-only; Arial lacks ∈ and ⁻, which pulled in Arial Unicode MS); full grid "C ∈ {0.001, 0.01, 0.1, 1.0}", "inverse penalty" and the refit with the selected C in caption ④ | ☐ |
| 65 | "chosen by inner three-fold stratified CV" | in figure + caption | inset "inner 3-fold" + 3-cell bar + dashed blue "fitted, then applied" arrow to the sigmoid; "stratified" in caption ④ | ☐ |
| 66 | "Class label" / "threshold 0.5" | in figure + caption | fork "≥ 0.5 ●" / "< 0.5 ○"; caption ⑤ | ☐ |
| 67 | "Outer validation" | caption | caption (c) "outer training partition"; the fold bars are the outer validation schemes | ☐ |
| 68 | "Pooled ten-fold (stratified, seed 42)" | in figure + caption | label "pooled 10-fold" + 10-cell bar; "stratified, seed 42" in caption (c) | ☐ |
| 69 | "Leave-one-site-out (20 site labels)" | in figure + caption | label "leave-one-site-out" + 20-cell bar; "20 site labels" in caption (c) | ☐ |
| 70 | Fold-bar cells (10 equal; 20 equal in the old figure) | in figure (changed) + caption | 20 LOSO cells now have widths ∝ strict-cohort site size; caption (c) "Cell widths are proportional to site size" | ☐ |
| 71 | Swatch "held-out fold" | in figure | vermilion swatch "held-out fold" | ☐ |
| 72 | Swatch "training" | in figure | lane-tint swatch "training" | ☐ |
| 73 | "All learned quantities are fitted / within the outer training partition" | in figure + caption | blue lane + padlock + label "outer training partition"; caption (c) sentence | ☐ |
| 74 | Arrows between the 3 × 2 boxes and the average | in figure | strands through two glyphs into p̄ | ☐ |

## Panel (d): removed (approved by the author)

The whole panel is removed. The four analyses are drawn in Figure 3 (b–e). Each string is already stated in Methods, quoted below.

| # | Old element | New location | Quote from method.tex | ✓ |
|---|---|---|---|---|
| 75 | "(d)" and old caption "(d) The four analyses that correspond to the contributions are shown schematically, and no results are displayed." | Methods text + caption | "The four analyses that correspond to these contributions are shown schematically in Figure~\ref{fig:designs}b--e."; new caption ends "No results are shown." | ☐ |
| 76 | "Cognitive ability" | Methods text (+ Fig. 3d) | "In this exploratory analysis, discrimination is examined across levels of cognitive ability and across sex." | ☐ |
| 77 | IQ bands "< 85", "85–99", "100–114", "≥ 115" | Methods text | "The autistic participants, in turn, are divided into full-scale IQ bands below 85, 85--99, 100--114, and 115 or above." | ☐ |
| 78 | "non-autistic" box; "Autistic participants in four full-scale IQ bands, each compared with the same comparison group of all 468 non-autistic participants" | Methods text | "In the second analysis, all 468 non-autistic participants are held fixed as the comparison group." and "As a result, each autistic band is evaluated against the same non-autistic participants." | ☐ |
| 79 | "Motion-matched system contrast" | Methods text (+ Fig. 3c) | "The sensitivity of the group contrasts to head motion is assessed in motion-matched samples." | ☐ |
| 80 | ●–○ pair glyphs; "Autistic (●) and non-autistic (○) participants are paired within site on mean framewise displacement" | Methods text (+ Fig. 3c pair glyphs) | "For this purpose, autistic participants and non-autistic participants from the same site are matched greedily and without replacement by mean framewise displacement." | ☐ |
| 81 | "(caliper 0.01 mm)" | Methods text | "A maximum within-pair difference of 0.01 mm is used for the system-level analysis, whereas a maximum difference of 0.02 mm is used for the regional analysis." | ☐ |
| 82 | 8 × 8 system grid; "contrasts for 36 system pairs" | Methods text (+ Fig. 3c triangle glyph) | "As a result, the assignment produces eight system classes and 36 unordered within- and between-system pairs." | ☐ |
| 83 | "Feature-selection scope" | Methods text (+ Fig. 3b) | "In this primary descriptive comparison, the effect of the scope of data access on performance is examined." | ☐ |
| 84 | "training fold" / "all labels" bars, brackets, "test" (×2) | Methods text | "For this outcome-dependent ranking, either all diagnostic labels or the labels in each outer training fold are used." | ☐ |
| 85 | "Regions are ranked with the labels under the bracket: training fold only, or all labels before the split" | Methods text | "Next, parcels are ranked by the average absolute group difference over their incident connections, that is, the connections that involve each parcel." (+ quote in row 84) | ☐ |
| 86 | "(connections: all labels)" | Methods text | "In addition, the 2,000 strongest connections are selected with an F test using all labels." | ☐ |
| 87 | "Classifier family" | Methods text (+ Fig. 3e) | "The evaluated models range from linear classifiers to tree ensembles." | ☐ |
| 88 | "same features and outer folds" | Methods text | "To this end, the same static tangent features, feature scaling, and outer folds are used for all classifiers." | ☐ |
| 89 | "Linear (5)" | Methods text | "First, nested $\ell_2$-regularised logistic regression, the classifier of each primary member, is used." … "Fifth, a two-stage low-rank spectral classifier of rank 20 is used." | ☐ |
| 90 | "ℓ2-logistic (tuned)" | Methods text | "Hyperparameter tuning is applied to the logistic model only." | ☐ |
| 91 | "ridge, linear SVC" | Methods text | "Second, ridge classification is used with penalty strength 100." and "Third, linear support-vector classification is used with $C=0.01$; in this method, the linear boundary is placed to maximise the margin between the two classes." | ☐ |
| 92 | "PCA + shrinkage LDA" | Methods text | "Fourth, shrinkage linear discriminant analysis is used after 200-component principal component analysis (PCA)." | ☐ |
| 93 | "low-rank spectral" | Methods text | "Fifth, a two-stage low-rank spectral classifier of rank 20 is used." | ☐ |
| 94 | "Tree ensembles (3)" | Methods text | "The remaining three classifiers are tree ensembles, in which the outputs of many decision trees are combined." | ☐ |
| 95 | "random forest" | Methods text | "Sixth, a random forest with 300 trees of unrestricted depth is used." | ☐ |
| 96 | "XGBoost, CatBoost" | Methods text | "Seventh, XGBoost is used with 300 trees of maximum depth 4, learning rate 0.1, row subsampling 0.8, and column subsampling 0.3." and "Finally, CatBoost is used with 300 trees of depth 4, learning rate 0.1, and feature subsampling 0.3." | ☐ |
| 97 | "fixed configurations" | Methods text | "No other classifier is tuned; instead, each of the other classifiers is used with its fixed configuration" | ☐ |

**Action for the Figure 3 author:** add the classifier-family schematic (5 linear, 3 tree) as Figure 3e, as the brief requires. Method.tex already cites "Figure~\ref{fig:designs}e" for it.

## New elements added (not in the old figure)

| New element | Meaning | Explained in |
|---|---|---|
| Circled steps ①–⑤ | ① quality control, ② time series and windows, ③ scaling, ④ ℓ2-logistic with inner CV, ⑤ probability average and threshold | caption |
| Blue padlocks after "Static tangent" and "Windowed tangent" (b) | the tangent reference is a learned quantity fitted on training participants only (Methods: "For the primary ensemble, the tangent reference and scaler are fitted once on the complete outer training partition."); correlation has no fitted reference, so no padlock | caption (b) and last sentence of (c) |
| p̄ defined as p_i^ens | the merge-node symbol is tied to Equation (12) | caption ⑤ |
| Colour-key saturation rule | symmetric limits at the 98th percentile of absolute off-diagonal values; the feature strips share them | caption (b) |
| Dot cloud (40 filled, 47 open) | about one dot per ten participants | caption (a) |
| Inner 3-cell bar, inner validation cell in solid blue | inner stratified three-fold CV inside the training lane | caption ④ |
| Dashed blue arrow, inset → sigmoid | C is selected on training data, then applied to the refit | arrow key + caption ④ |
| Arrow key (data flow / fitted, then applied / held-out labels used) | shared grammar for Figures 1–3 | caption opening |
| Participant key "● autistic ○ non-autistic" | shape coding | in figure |
| Shared "− 0 +" key | one key for the three matrices and three vector strips | caption (b) ranges |
| Real 19,900-feature strips | vecu of each member matrix, same colour limits as its matrix | caption (c) |

No result (AUROC, accuracy, p-value, effect size, learned weight) appears in the old or the new figure.
