# Figure contract: fig_designs (Methods Figure 3, "Evaluation and analysis designs")

Source of truth: `ASD-journal-I/sections/method.tex` (sec:m-4-1 to m-4-3, m-6-3, m-6-7, m-6-9,
m-7-1 to m-7-4, m-8-1, m-8-3). Redesign brief section 2 "Figure 3: designs" governs style.
Backend: Python (matplotlib + nilearn via `viz_style.py`) only. No results are shown.

## 1. Core message (5-second read)
Every learned quantity sits inside the blue training lane, and each secondary analysis
changes exactly one element of that design.

Results-level question it serves: how are the primary model and the four contribution
analyses evaluated without held-out information entering any fitted step?

What could overturn the claim: a fitted quantity (reference, scaler, selector, C, classifier)
drawn outside the training lane without a leakage mark; a design that changes more than one
element without saying so. Both are checked panel by panel against method.tex.

## 2. Evidence chain (panel roles)
| Panel | Question | Role | Decisive visual comparison |
|---|---|---|---|
| a (hero, full width) | Where is each quantity fitted? | Setup / protocol | Grey label-free pre-split zone vs blue training lane (steps 1-5) vs vermilion held-out lane; dashed blue "fitted, then applied" arrows cross only downward |
| b | What changes when selection uses all labels? | Perturbation (leakage) | Two strips identical except which glyphs sit left of the split line; leaky strip carries the dashed vermilion "labels" arrow; region-level scope chip training-fold / all labels plus k ticks |
| c | Do group contrasts survive motion matching? | Control / sensitivity design | Complete cohort flanked by the two FD-matched samples, one shared nuisance model and d_e, two outputs (36-pair system triangle, SMN parcels), one inference tag |
| d | Is discrimination confounded by IQ? | Stratification design | Same participants on one shared IQ axis under three partitions (quartiles, fixed comparison group, within-site matching) plus score-IQ glyph |
| e | Does classifier family matter under fixed inputs? | Comparator family | Shared input (features, scaling, folds) forks into 5 linear vs 3 tree-ensemble glyphs; only the logistic model carries the tuning mark |

No panel redraws another: (a) defines the grammar (split line, lanes, icons) that (b) and (e)
reuse without relabelling.

## 3. Archetype
Schematic-led composite: one full-width hero protocol panel (a), then a row of two
equal-width design panels (b, c), then two full-width strip panels (d, e). Real-data
elements: strict-cohort site sizes (LOSO fold widths), CC200 atlas + system table (brain
glyph, system strips), cohort class counts (1:10 dot cloud).

## 4. Backend
Python 3 (`/Users/reshadsazid/ASD-forever/.venv/bin/python`), matplotlib + nilearn through
`viz_style.py`; no R, no AI images, no external APIs.

## 5. Journal / export contract
- Computers in Biology and Medicine, cas-sc single column: 165 mm wide; height about 170 mm
  (limit 195 mm) so that figure plus caption fit one 224 mm text page.
- Arial; labels 7 pt, panel letters 8 pt bold lowercase, sub/superscripts 6 pt (6 pt floor
  checked on the PDF with `audit_pdf_text.py --min-pt 6`).
- Lines: keylines 0.5 pt, arrows 0.75 pt, lanes 1.0 pt, nothing below 0.25 pt.
- Colour roles: blue #0072B2 training, vermilion #D55E00 held-out/leakage only (never text),
  member colours only as tabs/strands, system colours only on strips and the brain.
- <= 4 words per label, <= ~40 words per panel, no sentences; explanatory text -> caption,
  keyed to panel letters and step numbers 1-5.
- Export: `bbox_inches=None`, 600 dpi, PDF (fonttype 42) + SVG (fonttype none, named
  `<g id>` groups) + PNG to results/latest_study/figures, copied to ASD-journal-I/figures.
- QA: `viz_style.check_layout` (size, words, overlap, clipping, vermilion text), visual
  inspection of the full-size PNG after every render (max 6 iterations), audit_pdf_text.py.
  Collision audit: NOT AUDITABLE (PyMuPDF absent). Panel-alignment gate: NOT APPLICABLE in
  its axes form (one full-page canvas axes); alignment is enforced by the 12-column mm grid
  (panel letters at x = 0, 84 mm; panel tops on shared rows).
- Data-integrity note: the (a) dot cloud draws 40 filled + 47 open markers, one per about
  ten participants (403 / 468 -> 40 / 47); the caption states the scale. Markers in (c)-(e)
  are schematic.
