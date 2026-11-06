# Element ledger: Figure 2 (`fig_representation`)

Every string, number, symbol and equation reference in the previous figure (`ASD-journal-I/figures/fig_representation.png`, the DejaVu version) is listed below with where it now lives:

- **in figure**: as a label, tag, chip or glyph in the new render.
- **caption**: in the new caption, `caption_representation.tex`.
- **Methods text**: already stated in `sections/method.tex` (quoted).

Circled steps (reviewer revision): ① Eq. (1) · ② Eq. (2) · ③ Eq. (3) · ④ Eq. (4) · ⑤ Eq. (6). Each badge now sits only where its equation acts. Eq. (5) (the r chips in (a) and the "r = 0.15" step in (b)) and Eq. (7) (the "mean" arrow in (b)) are cited by number in the caption at the sentence describing that element, without a circle. The earlier mapping (⑤ = Eq. 6 + Eq. 7) keyed ⑤ to the mean step, but the ⑤ badge sits only at the window bands.

Colour limits used in the new figure (symmetric, 99th percentile of |off-diagonal|, rounded up; identical to the old colourbars wherever the matrix is still drawn):

| Matrix | Limit |
|---|---|
| Σ̂_i | ±2,000 |
| training covariances (new deck) | ±2,000 |
| Σ̄_𝒯 | ±60 |
| S_i and x_i | ±0.25 |
| correlation | ±1 |
| window tangents | ±0.2 |
| m_i | ±0.12 |
| window means (new deck) | ±1,500 |
| dynamic reference (new) | ±120 |

## Panel (a)

| # | Old element | Now lives | Where exactly |
|---|---|---|---|
| 1 | "(a)" | in figure | panel letter **a** (8 pt bold, top-left) |
| 2 | "Time series" | caption | "the regional time series $X_i$" and "Five z-scored regional traces are drawn, one each from VIS, SMN, DAN, DMN and SUB" |
| 3 | "$X_i$" | in figure | label above the 5-trace glyph |
| 4 | "176 × 200" | in figure | outline chip under $X_i$ |
| 5 | Row labels VIS, SMN, DAN, SAL, LIM, FPN, DMN, SUB (drawn on both time-series heatmaps) | in figure + caption | Once, as the two-row network legend (colour chip + black abbreviation) at the bottom right. Colour strips stay on every matrix. Caption: "ordered by functional system, with the system colours and abbreviations of Figure 1". Full names: caption of Fig. 1, and Methods: "the assignment produces eight system classes" (sec:m-1-6). An inline alternative is given at the end of this file. |
| 6 | System brackets and leader lines beside the heatmaps | in figure | replaced by the colour strips (presentation only) |
| 7 | Time-series heatmap (176 × 200, RdBu) | in figure | replaced by the time-series glyph: five real traces of the same participant (one region per VIS, SMN, DAN, DMN, SUB), z-scored. Caption: "Five z-scored regional traces are drawn, one each from VIS, SMN, DAN, DMN and SUB". |
| 8 | Colourbar "−150 0 150" (time-series heatmap) | not carried over | Display scale of the heatmap that item 7 replaces; the traces are z-scored and carry no colour scale. It is not a scientific quantity. Flagged for the editor. |
| 9 | "Ledoit–Wolf" | in figure | arrow label $X_i$ → $\hat\Sigma_i$, with ① |
| 10 | "$\widehat{\Sigma}_i$" | in figure | label above the hero matrix, and the point label on the dome |
| 11 | "Eq. (1)" | in figure + caption | ① in figure; caption "① ... (Equation (1))" at the start of the step sentence |
| 12 | Covariance heatmap (real) | in figure | hero matrix, 24 mm, strips + system block lines |
| 13 | Colourbar "−2,000 0 2,000" | caption | "±2,000 for $\widehat{\Sigma}_i$ and the training covariances" |
| 14 | "Additional isotropic shrinkage" | caption | "The chips give the additional isotropic shrinkage $r$ of Equation (5)" |
| 15 | "Eq. (5)" | caption | same sentence ("The chips give the additional isotropic shrinkage $r$ of Equation (5) ..."); no circle |
| 16 | "primary member: $r$ = 0" | in figure + caption | chip "$r$ = 0 (primary)" on the arrow $\hat\Sigma_i$ → dome; caption "0 in the primary member" |
| 17 | "planned comparisons: $r$ = 0.15" | in figure + caption | chip "0.15" next to the primary chip; caption "0.15 in planned comparisons" |
| 18 | Open-circle junction and dashed connector from the shrinkage box | in figure | replaced by chips placed directly on the arrow (presentation) |
| 19 | "Whitened" | caption | "The covariance is whitened by this reference" |
| 20 | Formula $\overline{\Sigma}_{\mathcal T}^{-1/2}\widehat{\Sigma}_i\overline{\Sigma}_{\mathcal T}^{-1/2}$ | in figure | label on the arrow into the dome, with ③ |
| 21 | Whitened-covariance heatmap | in figure | replaced by the SPD dome: tangent plane at the blue anchor with a padlock, and the participant point $\hat\Sigma_i$ projected to $S_i$ (brief item 4) |
| 22 | Colourbar "−5 0 5" (whitened heatmap) | not carried over | Display scale of the heatmap that item 21 replaces; there is no whitened matrix left to scale. Flagged for the editor. |
| 23 | Filled junction dot where the reference joined the flow | in figure | dashed blue "fitted, then applied" arrow from the lane reference to the dome |
| 24 | "log" | in figure | arrow label dome → $S_i$ |
| 25 | "Tangent" | in figure + caption | member tab "Static tangent"; caption "mapped to $S_i$ by the matrix logarithm" |
| 26 | "$S_i$" | in figure | label above the matrix, and the projection label on the dome |
| 27 | "Eq. (3)" | in figure + caption | ③; caption "③ ... (Equation (3))" |
| 28 | Tangent heatmap $S_i$ (real) | in figure | 18 mm matrix with strips |
| 29 | Colourbar "−0.25 0 0.25" | caption | "±0.25 for $S_i$ and $x_i$" |
| 30 | "vecu" | in figure | arrow label $S_i$ → $x_i$, with ④ |
| 31 | "$x_i$" | in figure | label above the vector bar |
| 32 | "Eq. (4)" | in figure + caption | ④; caption "④ ... (Equation (4))" |
| 33 | "19,900" (under $x_i$) | in figure + caption | tag under the $x_i$ bar; caption "the 19,900 features $x_i$" |
| 34 | $x_i$ vector strip (real) | in figure | vector bar (real $S_i$ upper triangle, ±0.25) |
| 35 | "Shrinkage-correlation member:" | in figure + caption | orange member tab "Correlation"; caption "In the shrinkage-correlation branch, ..." |
| 36 | Formula $\widehat{\Sigma}_{i,pq}/(\widehat{\sigma}_p\widehat{\sigma}_q)$ | in figure + caption | label on the down-arrow $\hat\Sigma_i$ → correlation; caption "each covariance is divided by the product of the two regional standard deviations" |
| 37 | Correlation heatmap (real) | in figure | 12 mm matrix with strips |
| 38 | Colourbar "−1 0 1" | caption | "±1 for correlation" |
| 39 | "19,900" (correlation) | in figure | tag under the correlation vector bar. Methods: "the strict upper triangle of the correlation matrix ... is vectorised" (sec:m-2-2) |
| 40 | Correlation vector strip (real) | in figure | vector bar |
| 41 | "Training partition only" (bold) | in figure + caption | blue lane with padlock and label "training partition"; caption "Quantities estimated from the training partition only are shown in blue lanes" |
| 42 | Reference heatmap $\overline{\Sigma}_{\mathcal T}$ (real) | in figure | 12 mm matrix with blue frame at the lane's right end |
| 43 | Colourbar "−60 0 60" | caption | "±60 for $\overline{\Sigma}_{\mathcal{T}}$" |
| 44 | "Reference $\overline{\Sigma}_{\mathcal T}$" | in figure + caption | label $\bar\Sigma_{\mathcal T}$ under the lane matrix and at the dome anchor; caption "The reference $\overline{\Sigma}_{\mathcal{T}}$ ..." |
| 45 | "Eq. (2)" | in figure + caption | ② on the lane's Riemannian-mean arrow; caption "② ... (Equation (2))" |
| 46 | "refitted in each outer fold" | caption | "It is refitted in each outer fold ..." Methods: "The reference matrix is estimated separately for each outer fold" (sec:m-2-4) |
| 47 | "affine-invariant Riemannian mean" | in figure + caption | arrow label "Riemannian mean"; caption "affine-invariant Riemannian mean of the training covariances" |
| 48 | "of the training covariances" | in figure + caption | lane deck of three real training covariances, labelled "covariances"; caption |
| 49 | "(at most 10 fixed-point updates)" | caption | "with at most ten fixed-point updates". Methods: "at most ten updates are performed" (sec:m-2-4) |
| 50 | "applied unchanged to held-out participants" | in figure + caption | dashed blue arrow; caption "applied unchanged to held-out participants" |
| 51 | "Illustrative partition: 784 of 871 participants" | in figure + caption | chip "784/871" in both lanes; caption "an illustrative training partition of 784 of the 871 strict-cohort participants" |
| 52 | Grey arrows (data flow) | in figure | solid grey arrows (0.75 pt, 1.8 mm heads) |
| 53 | Grey dashed containers and grey fill | in figure | removed; the lanes are the only tinted containers (brief) |
| 54 | System block lines on the heatmaps | in figure | kept on matrices ≥ 14 mm ($\hat\Sigma_i$, $S_i$, $m_i$), dropped on small thumbnails |

## Panel (b)

| # | Old element | Now lives | Where exactly |
|---|---|---|---|
| 55 | "(b)" | in figure | panel letter **b** |
| 56 | "Time series $X_i$" | in figure | label $X_i$ at the top right of the time-series glyph |
| 57 | "$T_i$ = 176" | in figure + caption | outline chip "$T_i$ = 176"; caption "$\Delta_i = (176-50)/5$" |
| 58 | System row labels (repeat) | in figure | see item 5 (shown once) |
| 59 | Time-series heatmap (b) | in figure | same 5-trace glyph with window bands |
| 60 | "$s_{i0}$ = 0" | caption | "at volumes 0, 25, 50, 76, 101 and 126" |
| 61 | "$s_{i1}$ = 25" | caption | same |
| 62 | "$s_{i2}$ = 50" | caption | same |
| 63 | "$s_{i3}$ = 76" | caption | same |
| 64 | "$s_{i4}$ = 101" | caption | same |
| 65 | "$s_{i5}$ = 126" | caption | same |
| 66 | Six purple window bars | in figure | six grey bands numbered 0–5, drawn to scale under the traces (starts 0, 25, 50, 76, 101, 126; width 50) |
| 67 | Volume axis ticks "0 50 100 150 175" | in figure + caption | bands drawn to scale along the 176 volumes (the "$T_i$ = 176" chip gives the extent, last index 175); caption gives the starts in volumes |
| 68 | Axis title "volume" | caption | "at volumes 0, 25, ..." |
| 69 | "Eq. (6)" | in figure + caption | ⑤ next to the chips; caption "⑤ ... (Equation (6))" |
| 70 | "$K$ = 6" | in figure | chip "$K$ = 6"; also the six bands and the deck tag "×6" |
| 71 | "$L$ = 50" | in figure + caption | bracket "$L$ = 50" over window 0; caption "Six windows of $L = 50$ volumes" |
| 72 | "$\Delta_i$ = (176 − 50)/5 = 25.2" | caption | "$\Delta_i = (176-50)/5 = 25.2$" |
| 73 | "$s_{ik}$ = round($k\Delta_i$), $k$ = 0, …, 5" | caption + Methods | caption "$s_{ik} = \operatorname{round}(k\Delta_i)$"; $k$ = 0…5 are the band numbers in the figure. Methods Eq. (6): "$s_{ik} = \operatorname{round}(k\Delta_i), \quad k=0,\ldots,K-1$" |
| 74 | "Each window" (box title) | caption | "Each window covariance is estimated by ..." |
| 75 | "Ledoit–Wolf, Eq. (1)" | in figure + caption | inline step "Ledoit–Wolf" with ①; caption "(①)" |
| 76 | "additional shrinkage $r$ = 0.15, Eq. (5)" | in figure + caption | inline step "$r$ = 0.15" between ① and ③; caption "shrunk further with $r=0.15$", with Eq. (5) defined in the (a) sentence on the chips |
| 77 | "tangent projection at the dynamic reference, Eq. (3)" | in figure + caption | inline step "project" with ③ and a dashed blue arrow from "dynamic reference"; caption "projected at the dynamic reference (③)" |
| 78 | Small down-arrows inside the box | in figure | inline grey arrows between the three steps |
| 79 | Lilac "Each window" box fill | in figure | removed (presentation) |
| 80 | "Training partition only" (b) | in figure | second blue lane with padlock and "training partition" |
| 81 | "Dynamic reference" | in figure + caption | label "dynamic reference" under the lane matrix; caption |
| 82 | "Riemannian mean, Eq. (2)" | in figure + caption | arrow label "Riemannian mean" with ②; caption "Riemannian mean (②)" |
| 83 | "of each training participant's arithmetic mean of its six window covariances" | in figure + caption | lane deck of three real per-participant window means, labelled "window means"; caption "each training participant's arithmetic mean of its six window covariances" |
| 84 | "Window tangent matrices" | in figure + caption | staggered deck of the six real window tangent matrices tagged "×6"; caption "The six window tangent matrices" |
| 85 | "$k$ = 0" … "$k$ = 5" (thumbnail titles) | in figure | band numbers 0–5 on the time series (the deck front card is $k$ = 0) |
| 86 | Six window-tangent heatmaps (real) | in figure | the "×6" deck (real) |
| 87 | Colourbar "−0.2 0 0.2" | caption | "±0.2 for the window tangent matrices" |
| 88 | "Mean $m_i$" | in figure | arrow label "mean" and label "$m_i$" above the matrix |
| 89 | "Eq. (7)" | caption | "averaged elementwise into $m_i$ (Equation (7))" |
| 90 | $m_i$ heatmap (real) | in figure | 18 mm matrix aligned under $S_i$, with purple tab "Windowed tangent" |
| 91 | Colourbar "−0.12 0 0.12" | caption | "±0.12 for $m_i$" |

## New elements (not in the old figure; all real or specified by Methods)

| Element | Source |
|---|---|
| Member tabs "Correlation" (orange), "Static tangent" (green), "Windowed tangent" (purple) | brief item 11; Methods "Three connectivity representations are used in the primary ensemble" (sec:m-2-1) |
| Step badges ①–⑤ | brief item 5; defined in the caption |
| Shared "− 0 +" key | brief item 1; caption "One diverging scale with zero at white" |
| Network legend (8 chips) | brief item 6 |
| Dome labels $\hat\Sigma_i$, $S_i$, $\bar\Sigma_{\mathcal T}$ with padlock | brief item 4; Methods sec:m-2-4 |
| Training deck: Ledoit–Wolf covariances of the first three training participants (Pitt_0050003/4/5), ±2,000 | brief "stacked deck of training covariances"; caption "three of these participants are drawn as deck cards" |
| Dynamic deck: per-participant arithmetic means of the six window covariances of the same three participants, ±1,500 | brief item 10; Methods "the arithmetic mean of the six window covariances is computed for each training participant" (sec:m-2-6) |
| Dynamic reference matrix (real), ±120 | Methods "the Riemannian mean of these participant-level matrices is taken as the dynamic training reference" (sec:m-2-6) |
| Correlation 19,900 tag | Methods "the strict upper triangle of the correlation matrix ... is vectorised" (sec:m-2-2) |

## Reviewer revision (independent review, 2026-09-15)
- **Collision fixed.** Window band 5 (28.8–39.0 mm) ran under the left border of the panel (b) lane, which started at about 36 mm. The per-window chain arrows went from 5 to 6.5 mm, so "project" and the lane anchored beneath it moved right. The lane now starts about 42 mm, at least the 3 mm gutter from band 5, and an assert in the script enforces this.
- **Caption title corrected.** It read "tangent-space connectivity representations", but the figure also shows the correlation branch. It now reads "Construction of the connectivity representations ...".
- **Circled-step keying corrected.** See the header line above.
- **Circle markup made consistent with Fig. 1.** It is now `\textcircled{\scriptsize n}~` at the start of each step sentence, matching `caption_overview.tex`.
- **Window shrinkage wording fixed.** The caption said "Ledoit–Wolf shrinkage with r = 0.15", which merged the two steps. It now reads "Ledoit–Wolf shrinkage (①), shrunk further with r = 0.15", as Methods sec:m-2-6 does.
- **Trace systems named in the caption.** "one each from VIS, SMN, DAN, DMN and SUB" (item 7).
- **Correlation 19,900 in the caption.** "the upper triangle gives 19,900 features" (item 39).
- **"No results are shown." added.** This matches the Fig. 1 caption.
- **Item 5 verified.** `caption_overview.tex` carries the full system names ("visual (VIS), somatomotor (SMN), ... subcortical/cerebellar (SUB)"), so the cross-reference to Fig. 1 holds once the editor installs that caption. The inline alternative below is needed only if it is not installed.

## Inline alternative for item 5 (if Fig. 1's caption does not carry the full names)
"The systems are visual (VIS), somatomotor (SMN), dorsal attention (DAN), salience/ventral attention (SAL), limbic (LIM), frontoparietal control (FPN), default mode (DMN) and subcortical/cerebellar (SUB)." (24 words)

## Tick-off
- **All 91 old elements are accounted for:** 89 now live in the figure, the caption or Methods.
- **2 are not carried over:** the colourbar ranges ±150 and ±5, which belonged to heatmaps the brief replaces with glyphs. Neither is a scientific quantity. Both are flagged for the editor.
- **No result (AUROC, accuracy, p-value, effect size) appears** in the old or new figure.
