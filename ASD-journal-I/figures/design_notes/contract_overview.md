# Figure contract: Figure 1 (fig_overview), Methods overview

## 1. Core conclusion
- **Results-level question:** How does one ABIDE-I scan become a single ensemble probability, and how is that probability validated without held-out information?
- **Figure-level claim (5-second message):** Each brain gets three connectome views, and one ensemble averages them without ever seeing held-out data.
- **What could overturn it:** a learned quantity drawn outside the training lane, members that appear to share a scaler or penalty, or a figure that implies results. No results are shown: no AUROC, accuracy, p-values or effect sizes.

## 2. Evidence chain (panel roles)
| Panel | Question | Evidence role | Unique inference gain |
|---|---|---|---|
| a | Which participants and signals enter? | Setup: cohort strip | Flow from 1,112 to 871 participants (the funnel removes 241); CC200 parcellation by system; one real example time series with the K = 6 windows |
| b | What are the three views of one scan? | Representative example: real data | Three real 200 x 200 matrices of the same participant (correlation, static tangent, windowed tangent), ordered by system, sharing one colour key |
| c | How are the views combined and validated? | Model and validation schematic | Member strands pass a scaler and an l2-logistic model inside the blue training lane, with inner CV selecting C; equal-weight merge to p-bar; threshold 0.5; pooled ten-fold and leave-one-site-out fold bars |

**Displaced material:**
- The old panel (d), the four analyses, is removed because Figure 3 (b-e) already draws it. The classifier-family schematic (5 linear, 3 tree) moves to Figure 3e.
- Equation numbers, the rater rule, C-PAC `filt_noglobal`, 0.01-0.10 Hz, seed 42, the C grid, the colour ranges and the full system names move to the caption, keyed to panel letters and the circled steps 1-5.

## 3. Archetype
Schematic-led composite with real-data thumbnails. It has three full-width horizontal bands that read left to right, and panel (b) holds the only real matrices, so it is the visual hero.

## 4. Backend
Python only: matplotlib, with nilearn to load the atlas. All drawing comes from the glyphs in `viz_style.py`. No R, no AI images, no external API.

## 5. Journal and export contract
- **Size:** 165 mm wide (the CBM `cas-sc` text width) by 128 mm tall, within the 185 mm cap. The height was set in the script before the first render. At that height the figure and caption fit on one 22.4 cm page.
- **Type:** Arial; labels 7 pt; bold lowercase panel letters 8 pt; mathtext sub- and superscripts 6 pt. The floor is 6 pt, stricter than the skill's 5 pt. Only Arial faces are embedded (ArialMT, Arial-ItalicMT, Arial-BoldMT): the inset tag avoids ∈ and ⁻, which Arial lacks.
- **Learned quantities:** scaler and classifier sit inside the blue lane; the tangent references (not drawn in this figure; see Figure 2) are marked by blue padlocks on the two tangent member tabs in (b).
- **Text budget:** at most 4 words per label, about 40 words per panel, no sentences.
- **Colour:**
  - Neutrals are ink and grey.
  - Blue (#0072B2) and its lane tint mean training.
  - Vermilion (#D55E00) marks the held-out fold only and is never used for text.
  - Member colours appear as tabs and strands only.
  - Yeo-7 + SUB colours appear only on matrix strips, the brain, the time-series traces and the legend chips.
  - Matrices use RdBu_r with zero at white and one shared key.
- **Lines:** keylines 0.5 pt, arrows 0.75 pt, lane border 1.0 pt, nothing below 0.25 pt.
- **Export:** `.pdf`, `.svg` and `.png` at 600 dpi with `bbox_inches=None`, `svg.fonttype='none'`, `pdf.fonttype=42` and `image.composite_image=False`. Each glyph is a named SVG `<g id>`. Files go to `results/latest_study/figures` and are copied to `ASD-journal-I/figures`.
- **Source data:**
  - Example participant: NYU_0050952, CC200, 176 volumes.
  - Strict-cohort metadata: 871 participants, 20 site labels.
  - CC200 atlas and ROI table.
  - Tangent references from the illustrative training partition: the strict cohort minus a random 10 % (`default_rng(42)`), 784 of 871.
  - These come from the cached `compute()` output of this script.
- **Image integrity:**
  - The matrices, vectors and traces are the real computed values, displayed at symmetric 98th-percentile limits: ±0.7, ±0.22 and ±0.096.
  - The dot cloud is drawn at about 1 dot per 10 participants (40 filled, 47 open), which the caption states.
  - The brain is a surface projection of the CC200 atlas (loaded with nilearn) drawn by matplotlib.
- **QA gates:**
  - `vs.check_layout` checks size floor, word count, overlaps, page bounds and vermilion text.
  - `audit_pdf_text.py --min-pt 6` checks the exported PDF.
  - The PNG is inspected at full size after every render.
  - Panel-alignment auditor: NOT APPLICABLE by construction. There is one canvas axes, and alignment is enforced by shared mm coordinates in code.
  - Collision audit (`audit_figure_collisions.py`): NOT AUDITABLE, because PyMuPDF is not installed and may not be installed.
