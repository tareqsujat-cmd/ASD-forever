# Element ledger: fig_designs (old -> new)

Old figure: `ASD-journal-I/figures/fig_designs.png` (backed up to
`scratchpad/designs_old/`). Strings were extracted from the old SVG text nodes and checked
against the old source (`designs_old/viz_method_designs.orig.py`).
New figure: `results/latest_study/figures/fig_designs.{pdf,svg,png}` (copied to
`ASD-journal-I/figures/`). New caption: `scratchpad/caption_designs.tex`.

Where it lives now:
- **fig** = in the figure (label, tag or glyph)
- **cap** = in the new caption
- **MT** = in method.tex; the sentence is quoted

## Panel (a): nested cross-validation and fold-level fitting
| # | Old element | Now | Where / quote |
|---|---|---|---|
| a1 | "(a)", title "Nested cross-validation and fold-level fitting" | fig + MT | letter **a**; MT sec:m-4-3: "The nested cross-validation and the order of the fold-level fitting steps are illustrated in Figure~\ref{fig:designs}a." |
| a2 | "Strict cohort" | fig + MT | 1:10 dot cloud in the grey pre-split zone; MT: "The participants who meet all three conditions form the strict cohort of 871 participants." |
| a3 | "871 participants" | fig | tag "871"; the cloud has 40 filled and 47 open markers (403/468 ÷ 10); cap: "Markers in (a) represent about ten participants" |
| a4 | "Covariance per participant" | fig | deck glyph Σ̂_i with tag "×871" |
| a5 | "Ledoit–Wolf, Eq. (1)" | cap | "Label-free Ledoit--Wolf covariances (Equation (1))" |
| a6 | "estimated once, before the outer split, and cached" | fig + cap + MT | chip "once · cached" left of the split line; cap: "are cached before the outer split"; MT: "Therefore, it is cached before fold-specific fitting without using information from other participants." |
| a7 | "Outer split" | fig | "outer split" label on the split line |
| a8 | "Pooled ten-fold" | fig | "pooled 10-fold" + 10-cell fold bar, held-out cell vermilion |
| a9 | "stratified, shuffled, seed 42" | MT | "the primary strict cohort is partitioned by stratified ten-fold cross-validation with shuffling and random seed 42." |
| a10 | "or" | fig | the two fold bars joined by one bracket into a single fork (alternatives) |
| a11 | "Leave-one-site-out" | fig | "leave-one-site-out" + fold bar with real site sizes, NYU held out |
| a12 | "20 recorded site labels, each held out once" | fig + MT | 20 cells in the bar; MT: "In this protocol, each of the 20 recorded site labels is held out once." |
| a13 | "width ∝ site size" | cap | "leave-one-site-out cell widths scale with site size" |
| a14 | "Outer training partition" | fig + cap | blue lane "outer training" with padlock; cap: "In each outer training partition" and "Per member" |
| a15 | "per member" | fig + cap | "per member" + three member-colour tabs; cap "Per member" |
| a16 | step 1 "Tangent reference fitted to the training covariances, Eq. (2) (tangent members)" | fig + cap | badge ① "anchor", dome with Σ̄_𝒯 and padlock; cap: "tangent members fit the reference (step 1, Equation (2))" (review fix: steps 1–2 are now tied to the tangent members: the correlation member has no reference and no projection, MT sec:m-2-2) |
| a17 | step 2 "Training and held-out covariances projected with it, Eqs. (3)–(4)" | fig + cap | badge ② "project" (matrix → vector) in both lanes, dashed blue arrow from the anchor to the held-out projection; cap: "tangent members … project covariances (step 2, Equations (3)–(4))" |
| a18 | step 3 "Feature scaling fitted, Eq. (10)" | fig + cap | badge ③ "scale", μ/σ glyph, dashed blue arrow to the held-out μ/σ; cap: "Per member, scaling is fitted (step 3, Equation (10))" |
| a19 | step 4 "C selected from {0.001, 0.01, 0.1, 1.0} by inner three-fold stratified cross-validation (highest AUROC)" | fig + cap + MT | badge ④ "inner CV", dashed inset with a 3-cell bar (inner validation cell solid blue, as in Figure 1: it lies inside the training partition, so it takes no vermilion) and "*C* ∈ {10⁻³…1}" (italic C, as on the arrow), arrow tagged *C*; cap: "the penalty $C$ is tuned by inner cross-validation (CV; step 4)"; MT: "To this end, $C$ is selected from $\{0.001,0.01,0.1,1.0\}$ by three-fold stratified cross-validation in the outer training partition." and "The value of $C$ with the highest AUROC in this inner cross-validation is then selected." |
| a20 | "inner" (label on the 3-cell strip) | fig | "inner CV" |
| a21 | step 5 "Classifier refitted on the complete outer training partition, Eq. (11)" | fig + cap | badge ⑤ "refit", sigmoid glyph; cap: "the classifier is refitted (step 5, Equation (11))" |
| a22 | step numerals 1–5 | fig | circled badges ①–⑤ |
| a23 | "Held-out partition" | fig | vermilion-outlined lane "held-out" |
| a24 | "training fits applied; probabilities predicted, Eq. (11)" | fig + MT | three dashed blue arrows into the held-out lane + held-out sigmoid; MT: "Next, probabilities are predicted for the held-out partition." (the Eq. (11) reference is carried by step 5 in the caption) |
| a25 | "Ensemble" / "member probabilities averaged, Eq. (12)" | fig + cap | three member strands merging at p̄; cap: "member probabilities are averaged (Equation (12))" |
| a26 | "held-out indices" (arrow over the top) | fig | fork arrow from the fold bars into the held-out lane (brief item 3) |

## Panel (b): feature-selection scope
| # | Old element | Now | Where / quote |
|---|---|---|---|
| b1 | "(b)", title "Feature-selection scope" | fig + MT | letter **b**; MT: "The reference and global-selection conditions and the region-level scope comparison are illustrated in Figure~\ref{fig:designs}b." |
| b2 | "Connection level: 19,900 static tangent features" | fig + MT | header "19,900 connections"; MT: "seven data-access conditions are evaluated with the 19,900 static tangent features from the primary strict cohort." |
| b3 | "Condition 1: reference" | fig + MT | strip label "reference"; MT: "The first condition serves as the reference condition" |
| b4 | "strict cohort (871); nothing fitted before the split" | fig + MT | empty grey zone left of the split (only the covariance deck); MT: "The same outer folds and primary strict cohort are retained in each condition" |
| b5 | "Condition 4: global selection" | fig + cap + MT | strip label "global selection"; cap "(b) Global selection fits the reference …"; MT: "In the fourth condition, a global-selection condition, the tangent reference is fitted globally and scaling is fitted within each fold." |
| b6 | "before the split: global tangent reference" | fig + cap | anchor glyph (no padlock) in the grey zone; cap: "Global selection fits the reference and selects … before the split" |
| b7 | "F test with all labels selects the 2,000 strongest connections" | fig + cap + MT | funnel tagged "2,000" + dashed vermilion "labels" arrow; cap: "selects 2,000 connections by an all-label F test before the split"; MT: "the 2,000 strongest connections are selected with an F test using all labels." |
| b8 | "(≈ 10%)" | MT | "This subset of approximately 10\% of the connections serves as the connection-level selection condition." |
| b9 | "Outer split (same folds)" | fig + MT | one split line shared by both strips; MT: "The same outer folds and primary strict cohort are retained in each condition" |
| b10 | "Training fold" (×2) | fig | blue lanes with padlocks |
| b11 | "tangent reference, scaler and classifier fitted" | fig | anchor, μ/σ, sigmoid in the reference lane |
| b12 | "scaler, classifier fitted; 2,000 selected connections" | fig | μ/σ, sigmoid in the global-selection lane (anchor slot empty); 2,000 on the funnel |
| b13 | "Held-out fold" (×2) | fig | vermilion-outlined held-out cells with participant markers |
| b14 | "held-out labels" (dashed arrow label) | fig + cap | "labels" on the dashed vermilion arrow; cap: "Arrows and system colours follow Figure 1" (dashed vermilion = held-out labels used, overview key) |
| b15 | "Region level: 863 participants, 196 regions" | fig | header "196 regions" + tag "863" |
| b16 | "r = 0.30 in Eq. (5)" | fig + cap | tag "r = 0.30"; cap: "at shrinkage $r$ (Equation (5))" |
| b17 | "nuisance model fitted once to all participants" | fig + MT | chip "nuisance model" in the grey pre-split zone; MT: "First, the nuisance model is fitted once to the complete analysis cohort." |
| b18 | "parcels ranked by mean absolute group difference over incident connections, using" | fig + MT | rank glyph (descending bars); MT: "Next, parcels are ranked by the average absolute group difference over their incident connections, that is, the connections that involve each parcel." |
| b19 | "training-fold labels" / "all labels" (dashed red box) | fig + cap | chips "training-fold" (blue) and "all labels" (vermilion outline); cap: "Regions are ranked with training-fold or all labels" |
| b20 | "→ top 10, 20, 30, 50 or 100 regions retained" | fig + MT | "top k regions" + ticks 10 20 30 50 100 "all"; MT: "Then, the top 10, 20, 30, 50, or 100 regions are retained in the models. Finally, a model containing all retained regions serves as the reference condition." |

## Panel (c): motion-matched group contrast
| # | Old element | Now | Where / quote |
|---|---|---|---|
| c1 | "(c)", title "Motion-matched group contrast" | fig + MT | letter **c**; MT: "The construction of the group contrasts and the motion-matched sensitivity analysis are illustrated in Figure~\ref{fig:designs}c." |
| c2 | "Group-contrast cohort", "863 participants" | fig + cap | centre cloud tagged "863"; cap: "the complete cohort" |
| c3 | "196 regions", "19,110 connections" | fig + MT | "196 regions" in (b); MT: "the group-contrast cohort contains 863 participants and 196 regions, corresponding to 19,110 unique connections." |
| c4 | "Pearson r → Fisher z" | MT | "the Pearson correlations between regions are Fisher transformed and then vectorised." |
| c5 | brain inset (196 retained parcels coloured by system, sagittal x = −42 / axial z = 18; 4 excluded uncoloured) | fig + cap | CC200 lateral brain glyph (nilearn-loaded atlas) with SMN in colour and the other parcels, including the 4 excluded, in grey; the full system colouring is in Figure 1; cap: "Arrows and system colours follow Figure 1" |
| c6 | legend VIS SMN DAN SAL LIM FPN DMN SUB | fig + cap | removed as a duplicate of Figure 1 (brief item 9); system colour strips remain on the 36-pair triangle; "SMN" label + cap "SMN, somatomotor"; the other abbreviations are defined with the Figure 1 legend and caption |
| c7 | "Within-site greedy matching on mean FD, without replacement" | fig + cap + MT | two "FD match" arrows; cap: "two within-site samples matched on framewise displacement (FD)"; MT: "autistic participants and non-autistic participants from the same site are matched greedily and without replacement by mean framewise displacement." |
| c8 | "Unmatched", "complete cohort (863)" | fig + cap | centre cloud "863"; cap "the complete cohort" |
| c9 | "Caliper 0.01 mm", "256 pairs (512)" | fig + MT | pair glyphs tagged "256 pairs · 0.01 mm"; MT: "the system-level caliper retains 256 pairs (512 participants), and the regional caliper retains 280 pairs (560 participants)." |
| c10 | "Caliper 0.02 mm", "280 pairs (560)" | fig + MT | pair glyphs tagged "280 pairs · 0.02 mm"; same MT sentence as c9 |
| c11 | "Nuisance adjustment in each sample" | fig + MT | grey band "adjust" fed by all three samples; MT: "Nuisance adjustment and site-stratified permutation are then repeated within each matched sample." |
| c12 | "per connection: age, sex, mean FD, one-hot site indicators" | fig + MT | chips "age" "sex" "FD" "site"; MT: "A separate linear nuisance model is then fitted for each connection, with age, sex, mean framewise displacement, and one-hot acquisition-site indicators as predictors." |
| c13 | d_e = (R̄_e,autistic − R̄_e,comparison)/SD(R_e) | fig | the equation, once, in the band |
| c14 | "Eq. (20)" | cap | "Group effects $d_e$ (Equation (20))" (review fix: the previous "Contrasts (Equations (20)–(21))" tied Eq. (21), the permutation p-value, to the contrast) |
| c15 | "System contrasts" | fig | 8×8 upper-triangle system glyph (36 cells) with system strips |
| c16 | "mean d_e in 36 within- and between-system pairs (≥ 20 connections)" | fig + MT | tag "36 pairs"; MT: "For each system pair containing at least 20 connections, the connection-level standardised effects are then averaged." |
| c17 | "unmatched, 0.01 mm" (system) | fig | arrows from the 863 and 0.01 mm columns into the triangle; MT: "A maximum within-pair difference of 0.01 mm is used for the system-level analysis" |
| c18 | "Regional analysis" | fig | brain glyph with SMN highlighted, label "SMN" |
| c19 | "mean d_e per somatomotor parcel over within-system connections" | MT | "For each parcel in that system, the effect is the mean standardised group difference across its within-system connections." and "the default target is the somatomotor system." |
| c20 | "unmatched, 0.02 mm" (regional) | fig + MT | arrows from the 863 and 0.02 mm columns into the brain; MT: "whereas a maximum difference of 0.02 mm is used for the regional analysis." |
| c21 | "Diagnosis permuted 5,000 times within site" | fig | chip "5,000 perms within site" |
| c22 | "on fixed nuisance residuals" | MT | "For this purpose, the nuisance model is fitted, and its residuals are computed, once before permutation." |
| c23 | "Eq. (21)" | cap | "are tested by within-site permutation (Equation (21))" |
| c24 | "Benjamini–Hochberg FDR" | fig + cap | chip "BH-FDR"; cap: "BH-FDR, Benjamini--Hochberg false-discovery rate" |
| c25 | "within the network and regional families" | MT | "the Benjamini--Hochberg false-discovery-rate procedure … is applied to the network, regional, symptom-association, and component--symptom families." |

## Panel (d): cognitive-subgroup designs
| # | Old element | Now | Where / quote |
|---|---|---|---|
| d1 | "(d)", title "Cognitive-subgroup designs" | fig + MT | letter **d**; MT: "The three cognitive-subgroup designs are illustrated in Figure~\ref{fig:designs}d." |
| d2 | legend "autistic" / "non-autistic" (orange / blue dots) | fig | key "● autistic ○ non-autistic" at the top right (shape coding, ink) |
| d3 | "(i) Site-centred IQ quartiles" | fig | row "i site-centred quartiles" |
| d4 | "full-scale IQ centred within site and divided into quartiles" | fig + MT | Q1–Q4 cuts on the shared axis; MT: "full-scale IQ is centred within site and divided into quartiles, that is, four groups of approximately equal size." |
| d5 | "AUROC within each quartile" | MT | "AUROC is then estimated within each quartile." |
| d6 | "Q1", "Q2", "Q3", "Q4" | fig | quartile labels |
| d7 | axis "full-scale IQ minus site mean" | fig + MT | shared axis "full-scale IQ" + row label "site-centred"; MT sentence of d4 |
| d8 | "(ii) Fixed comparison group" | fig | row "ii fixed comparison" |
| d9 | "autistic full-scale IQ bands, each compared with the same 468 non-autistic participants" | fig + cap | bands on the ● row, outlined ○ group; cap: "All autistic IQ bands share the 468 non-autistic participants (ii)" |
| d10 | "< 85", "85–99", "100–114", "≥ 115" | fig | "<85", "85–99", "100–114", "≥115" + axis ticks 85, 100, 115 |
| d11 | "all 468 non-autistic (fixed)" | fig + cap | chip "468" on the outlined ○ group; cap "the 468 non-autistic participants" |
| d12 | "(iii) Within-site IQ matching" | fig | row "iii within-site matching" |
| d13 | "ascending autistic IQ; closest same-site partner" | fig + MT | pair lines drawn by the greedy rule in code (`greedy_iq_match`); MT: "For this purpose, the autistic participants at each site are taken in ascending order of full-scale IQ. Each autistic participant is then matched to the non-autistic participant at the same site with the closest full-scale IQ." |
| d14 | "at most five points apart" | fig | label "≤5 points" |
| d15 | "greedy, without replacement" | MT | "The matching is greedy and without replacement; that is, each autistic participant is assigned the closest partner still available, and no partner is used twice." |
| d16 | "then site-centred quartiles" | fig + MT | chip "Q1–Q4" at the row end; MT: "Finally, AUROC is estimated within site-centred IQ quartiles of the matched sample." |
| d17 | "excluded" (hollow marker) | fig + MT | "excluded" over a grey ● (hollow now means non-autistic); MT: "Autistic participants without such a match are excluded." |
| d18 | axis "IQ" | fig | "full-scale IQ" (shared axis) |
| d19 | footer sentence: "Within each diagnostic group, the out-of-fold classifier score is related to full-scale, verbal and performance IQ by Spearman correlation after site, age and mean FD are regressed from both variables." | fig + cap + MT | (iv) glyph "score–IQ" with axes "score", "IQ" and chip "Spearman ρ"; cap: "score--IQ correlations are within-group (iv)"; MT: "First, the scores are related separately to full-scale, verbal, and performance IQ, within autistic participants and within non-autistic participants. Before the Spearman correlation is computed, site, age, and mean framewise displacement are regressed from both variables." |

## Independent-review changes (presentation only; nothing scientific added or removed)
| Change | Reason |
|---|---|
| (a) inner-CV validation cell drawn solid blue instead of vermilion | vermilion is reserved for the outer held-out fold and leakage; the inner validation cell lies inside the training partition (matches Figure 1) |
| (a) inset tag "C ∈ {10⁻³…1}" now uses italic *C* | same symbol as the arrow tag *C* and the caption |
| (a) fold-bar pair and fork moved down so that the fork sits between the two lanes | removes the empty block under the fold bars; the split reads as one fork into the training and held-out lanes |
| (a) lane right edge 0.2 mm inside the page | the 1 pt lane border was half-clipped at the 165 mm page edge |
| cap: "tangent members fit the reference … and project covariances" | the correlation member has neither a reference nor a projection (MT sec:m-2-2: "Shrinkage-correlation features are computed for each participant from a Ledoit--Wolf covariance estimate.") |
| cap: Eq. (20) now labels $d_e$ and Eq. (21) the within-site permutation | Eq. (21) is the permutation p-value, not a contrast |
| cap: "by an all-label F test", "at shrinkage $r$ (Equation (5))", "the penalty $C$" | the previous caption left the selection rule, $r$ and $C$ undefined |
| cap: "other markers, IQ values and matrices are schematic" | the (a)/(b) matrix and vector thumbnails are block cartoons, not participant data |
| cap: "PCA, principal component analysis" added to the abbreviations | "PCA" appears in (e) |

## Items in the old caption / old figure notes that are not drawn
| Old element | Now | Where / quote |
|---|---|---|
| other five data-access conditions (global reference, global scaling, held-out tuning, site features, combined) | MT | "In the second condition, a global-reference condition, …", "In the third condition, a global-scaling condition, …", "In the fifth condition, a held-out-tuning condition, …", "In the sixth condition, a site-feature condition, …", "Finally, the seventh condition is a combined condition." |
| median imputation before matching; Mann–Whitney U on FD | MT | "Before matching, missing nuisance values are imputed with medians from the complete pre-matching cohort." / "Finally, the group difference in mean framewise displacement is tested by the Mann--Whitney U test before and after matching." |

## New in-figure elements (none invented; source quoted)
| New element | Source in method.tex |
|---|---|
| (e) "Static tangent" tab + μ/σ + fold bar feeding all classifiers | "the same static tangent features, feature scaling, and outer folds are used for all classifiers." |
| (e) "linear": ℓ2-logistic, ridge, linear SVC, PCA + shrinkage LDA, low-rank spectral | sec:m-6-3, first to fifth classifiers ("nested $\ell_2$-regularised logistic regression", "ridge classification", "linear support-vector classification", "shrinkage linear discriminant analysis is used after 200-component principal component analysis (PCA)", "a two-stage low-rank spectral classifier of rank 20") |
| (e) "tree ensembles": random forest, XGBoost, CatBoost | "The remaining three classifiers are tree ensembles …"; hyperparameters (300 trees, depth, learning rate, subsampling; ridge penalty 100; SVC C = 0.01; rank 20) stay in MT |
| (e) "tuned" chip on ℓ2-logistic only | "Hyperparameter tuning is applied to the logistic model only." / "No other classifier is tuned" (cap: "only the ℓ2-logistic model is tuned") |
| (a) Σ̂_i, "×871", Σ̄_𝒯, p̄, member tabs | Eqs. (1), (2), (12); "each representation is fitted and scaled separately" |
| (b) "labels" dashed vermilion arrow | "In addition, the 2,000 strongest connections are selected with an F test using all labels." |
