# Redesign brief: Methods figures 1–3 (overview, representation, designs)

## 0. What I checked

- **Figure size.** All three scripts draw at 6.5 in, i.e. 16.5 cm (the `cas-sc` text width), with DejaVu Sans at 7.5–8.5 pt.
- **Editable text is already on.** Every script sets `svg.fonttype="none"` and `pdf.fonttype=42`. The research note guessed the SVG text was outlined; it is not. The 0.8–1.6 MB SVG size comes from inline PNG heatmaps.
- **Nothing is rasterized on purpose.** No script calls `rasterized`.
- **Fonts.** Arial and Helvetica are both installed.
- **Content rule.** Every scientific element stays, but sentences leave the figure. Keep an **element ledger**: every string you remove is listed with where it now lives (caption or Methods) and ticked off before submission.

## 1. Global style system (applies to all three figures)

| Token | Spec |
|---|---|
| Canvas | 165 mm wide. Height: overview ≤ 200 mm, representation about 130 mm, designs ≤ 210 mm. `bbox_inches=None` so the width stays exact. |
| Neutrals | ink `#1A1A1A`; secondary text and arrows `#6E6E6E`; rules `#BFBFBF`; tint `#F3F3F3` |
| Training role | blue `#0072B2`; lane tint `#E6F1F8` (black text on it passes 4.5:1) |
| **One accent** | vermilion `#D55E00`, used **only** for held-out data and leakage paths. Never use it for text (contrast about 4:1); use it for chips, cells and strokes. |
| Member identity | correlation `#E69F00`, static tangent `#009E73`, windowed tangent `#CC79A7`. Use these only as 1.5 mm tabs or strand lines, never as fills. Keep them identical in all three figures. |
| Participants | shape coding, not colour: ● autistic, ○ non-autistic, in ink. Blue now means training. |
| Networks | the Yeo-7 + SUB colours, only on matrix edge strips and the brain glyph. Labels are black abbreviations next to colour chips; drop the coloured text. |
| Matrices | `RdBu_r` with zero at white. One shared "− 0 +" key per figure. The actual symmetric ranges go in the caption. |
| Type | Arial throughout, including mathtext (`mathtext.fontset="custom"`). Labels 7 pt; subscripts no smaller than 6 pt; panel letters bold lowercase **a**, **b** at 8 pt, top-left and flush with the grid. No other sizes. |
| Lines | keylines 0.5 pt; arrows 0.75 pt; the training-lane border 1.0 pt; nothing below 0.25 pt. |
| Corners | 1.5 mm radius on the few containers that remain; matrices and fold cells stay square. |
| Arrows | exactly three meanings, defined once in a one-line key in the overview: **solid grey** = data flow; **dashed blue** = a quantity fitted on training data, then applied; **dashed vermilion** = held-out labels used (leakage). Filled heads 1.8 mm long. |
| Icons | brain glyph (real nilearn CC200 render, 12 mm); matrix (one real 200×200 at 30–35 mm per figure, 8×8 block cartoons of 6–8 mm elsewhere); time-series glyph (5 network-coloured traces, no axes); vector bar (1.5 mm column); fold bar (10 cells, or 20 for LOSO with widths ∝ site size; held-out cell vermilion); stacked deck "×K"; SPD dome with tangent plane and the anchor Σ̄_T in blue with a padlock; sigmoid glyph = ℓ2-logistic; μ/σ glyph = scaler. |
| Text budget | ≤ 4 words per label and ≤ about 40 words per panel. Numbers appear as tags (871, −241, 19,900). No sentences. |
| Where removed text goes | Eq. numbers, seeds, C grids, rater rules and matching rules move into the **caption**, keyed to circled step numbers ①–⑤ in the figure. Long derivations go to Methods. |
| Whitespace | no panel boxes and no pastel fills. 3 mm gutters inside a panel, 6 mm between panels. Group by proximity and alignment on a 12-column grid. |

## 2. Per figure

### Figure 1: overview
**5-second message:** three connectome views of each brain, averaged by one ensemble that never sees held-out data.

**Layout:** three horizontal bands read left to right.
- **(a) Cohort strip.** ABIDE-I dataset cylinder "1,112 · 17 sites" → QC funnel "−241" → dot cloud "871 (403 ● / 468 ○)" → CC200 brain glyph → time-series glyph with 6 translucent window bands.
- **(b) Representations.** Three equal real matrices, each with its member tab.
- **(c) Ensemble.** Three coloured strands pass through scaler and sigmoid glyphs inside the blue training lane, merge at "p̄", then "p̄ ≥ 0.5". The fold bars sit at the right.

Before → after:
1. The four sentence boxes in (a) become the icon strip with number tags. The rater rule, C-PAC `filt_noglobal` and 0.01–0.10 Hz move to the caption.
2. The "CC200 parcels by system" table with full names becomes one row of 8 chips with abbreviations. Full names go to the caption.
3. The separate K = 6 window bar chart and volume axis become bands on the time-series glyph with a "K = 6, L = 50" tag.
4. Coloured network text (the purple VIS, and so on) becomes black text beside colour chips.
5. The lilac headers carrying "Eq. (1)–(7), r = 0.15" become a member tab plus a two-word name. The equations go to the caption.
6. Three colourbars (±0.7, ±0.22, ±0.096) become one shared key. Network strips stay on all three matrices; row labels appear once.
7. The 3×2 "Scaling Eq. (10) / ℓ2-logistic Eq. (11)" boxes become coloured strands through two glyphs.
8. The "Average of the three probabilities" box and the class-label box become a merge node "p̄" and the tag "≥ 0.5".
9. The C-grid sentence becomes a dashed zoom inset under the sigmoid: an inner 3-fold bar with the tag "C ∈ {10⁻³…1}".
10. The grey "Outer validation" box and its bold sentence become white-ground fold bars (10-fold and LOSO) plus the blue lane with a padlock.
11. **Remove panel (d).** It duplicates Figure 3. **Move "Classifier family" (5 linear, 3 tree) to Figure 3**, because it appears nowhere else.
12. The type changes from DejaVu 8.5/7.5 pt to Arial 8/7 pt.

### Figure 2: representation
**5-second message:** a training-only anchor flattens each covariance into a 19,900-dimensional tangent vector; the windowed member does the same for 6 windows and averages them.

**Layout:**
- **(a) Before-and-after chain:** X_i → Σ̂_i → dome glyph → S_i → x_i. The formula Σ̄_T^{-1/2} Σ̂_i Σ̄_T^{-1/2} and "log" label the arrows. The shrinkage-correlation member branches below Σ̂_i with its own tab.
- **Blue training lane** along the bottom: a stacked deck of training covariances tagged "784/871" → Riemannian mean → anchor, with a dashed blue arrow up to the dome.
- **(b)** has the same grammar with ×6 decks, aligned under (a).

Before → after:
1. Six colourbars (±150, ±2,000, ±1, ±60, ±5, ±0.25) and two more in (b) become one shared key. All ranges go in the caption.
2. The dashed "Additional isotropic shrinkage" callout becomes two chips on the arrow: "r = 0 (primary) | 0.15".
3. The grey "Training partition only" sentence box becomes the blue lane with a padlock. "At most 10 fixed-point updates" and "refitted each outer fold" go to the caption.
4. The whitened heatmap plus the separate log arrow become the dome/tangent-plane glyph. The formula stays on the arrow.
5. Every "Eq. (n)" becomes circled ①–⑤, keyed in the caption.
6. The network row labels repeated on each heatmap become labels once; the colour strips stay on all.
7. In (b), the s_ik bar chart and the Eq. (6)/Δ_i block become 6 numbered bands (0–5) on the time series with an "L = 50" bracket. The start values go to the caption.
8. The lilac "Each window" box with three steps becomes three arrow labels: "Ledoit–Wolf", "r = 0.15", "project".
9. The six k = 0…5 thumbnails become one staggered deck "×6" → m_i.
10. The second grey box in (b) becomes the same blue lane, where "mean of 6 window covs per participant" → Riemannian mean → dynamic anchor.
11. Member tabs: green on (a), purple on (b), matching Figure 1.

### Figure 3: designs
**5-second message:** every learned quantity sits inside the training lane, and each secondary analysis changes exactly one element.

**Layout:**
- **(a) Full width.** A grey pre-split zone (871 dots and Σ̂ "once · cached", which is label-free) → split line → blue training lane with icons ① anchor, ② project, ③ scaler, ④ inner-CV inset → C, ⑤ refit sigmoid. Beneath it, a vermilion-outlined held-out lane: project → p → ensemble.
- **(b) and (c)** side by side.
- **(d)** is three small multiples on a shared IQ axis.
- **(e)** is the classifier family.

Before → after:
1. The 20-plus pastel sentence boxes become white ground; only the training lane is tinted.
2. The numbered list of sentences in (a) becomes five icons; the sentences go to the caption.
3. The "held-out indices" arrow over the top becomes the held-out lane.
4. The fold bars are kept and enlarged. The "width ∝ site size" note goes to the caption.
5. The Condition 1/4 boxes in (b) become nnU-Net-style paired strips that are identical except for where the "select 2,000" icon sits relative to the split line. The leaky strip is marked only by a vermilion dashed "labels" arrow.
6. The region-level paragraph becomes the same paired grammar: a "training-fold | all labels" chip plus a k-tick bar at 10/20/30/50/100. "r = 0.30" and the nuisance fit go to tags or the caption.
7. The 8-box cascade in (c) becomes three pair glyphs (●–○) tagged "863", "256 pairs, 0.01 mm" and "280 pairs, 0.02 mm", then covariate chips "age · sex · FD · site", then the d_e equation once.
8. The system/regional boxes become two thumbnails: the 36-pair triangle glyph (reused from the old overview (d)) and a brain with SMN highlighted. The permutation sentence becomes "5,000 perms within site · BH-FDR".
9. The duplicated network legend is removed; the brain stays.
10. The captions in (d) become glyphs. Keep the (iii) matching glyph, which already works. The Spearman footer becomes a tiny (iv) score-vs-IQ glyph, with details in the caption.
11. Orange/blue participant dots become ●/○.

## 3. Tooling

- **Renderer: keep matplotlib + nilearn.** They are reproducible, draw from real data, and their text is already editable.
  - Add `viz_style.py` with the tokens above and the glyph functions (`matrix_thumb`, `deck`, `dome`, `fold_bar`, `lane`, `dots`, `arrow`). Import it from all three `viz_method_*.py` scripts.
  - Call `set_gid()` on each glyph group so it becomes a named `<g id>` in the SVG, which makes hand editing easier.
- **Export settings:**
  - rcParams: `font.family="Arial"`, `svg.fonttype="none"`, `pdf.fonttype=42`, `image.composite_image=False` (keeps each thumbnail a separate object).
  - Save with `savefig(dpi=600, bbox_inches=None)`.
  - For heatmaps and brains, use `imshow(interpolation="nearest")` or `rasterized=True`.
  - Deliver the PDF and SVG together.
- **Hand editing:** do the post-editing in Inkscape or Illustrator on the SVG. Treat the script as the source of truth, and record hand edits so they can be re-applied after a data change. Final submission is a vector PDF with fonts embedded (CBM accepts it).
- **Skills.** Copy files by hand; do not use the installer.
  1. **nature-figure, trimmed.** Leave out the OpenRouter script, its reference and `assets/figures4papers`. Set the backend to Python and override its export to `bbox_inches=None`. Use its per-figure plan, the alignment gate and `audit_pdf_text.py` (6 pt floor).
  2. **scientific-visualization.** Take `publication.mplstyle` and its palettes. Run `palette_audit.py` (grayscale and colour-blind checks) and `export_plan.py --publisher elsevier` with `.venv/bin/python`. Delete its "Citing" section.
  3. **academic-figures (optional).** Use it for 2–3 SVG layout wireframes before coding (Figs 1 and 3); relax its 18–22 mm gap rule.
  4. **Avoid** thesis-figure, canvas-design, figure-spec, the TikZ skills (`standalone.cls` is missing here) and every AI-image route.
- **QA before submission:**
  - `pdffonts` shows only embedded Arial.
  - 6 pt floor and element ledger complete.
  - Print at 16.5 cm and view from arm's length.
  - A colleague's "what do you see first?" test.
## Panel outlines (2026-10-06)
Every lettered panel block carries a 0.8 pt outline in a darker, more saturated shade of its own tint (viz_style.panel_outline). Blocks that meet the page edge extend into a 1.6 mm margin
(viz_style.canvas(bleed=PANEL_BLEED), panel_bg) so that labels and boxes drawn from the old page edge clear the stroke. Content coordinates are unchanged.
