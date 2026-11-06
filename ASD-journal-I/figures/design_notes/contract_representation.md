# Figure contract: Figure 2, connectivity representations (`fig_representation`)

## 1. Core message
**Question (Methods, Section 3.2):** how does one scan become the static and windowed tangent-space features that enter the ensemble?

**Figure-level claim (5-second message):** a reference fitted only on the training partition flattens each Ledoit–Wolf covariance into a 19,900-dimensional tangent vector. The windowed member does the same for six windows and then averages them.

**What could overturn the claim (the reader's worry):** the reference seeing held-out participants. The blue training lane, the padlock and the dashed blue "fitted, then applied" arrow answer this worry in the figure itself.

## 2. Evidence chain (panel roles)

| Panel | Question | Evidence role | Unique inference | Destination |
|---|---|---|---|---|
| a | How is one whole-scan covariance turned into tangent features? | Setup schematic built from a representative real example (NYU_0050952) | Order of the operations (Ledoit–Wolf ① → training anchor ② → whitening and log ③ → vecu ④) and where the reference comes from (training lane, 784/871). Also shows that the correlation member branches off the same Σ̂_i and that r = 0 is the primary value. | Main figure |
| b | How does the windowed member differ? | Same grammar applied to windows | Placement of six windows of L = 50 (⑤), per-window steps ① and ③ at a dynamic reference fitted in the training lane, and averaging of the six tangent matrices into m_i | Main figure |

Both panels are needed. Panel (a) cannot show window placement or the separate dynamic reference, and panel (b) cannot show the static chain or the correlation branch. The shared grammar (chain on top, blue lane underneath, the same x-columns) lets (b) be read as "(a) again, for each window".

Displaced material goes to the caption, keyed to circled steps ①–⑤: equation numbers, colour ranges, window starts, stride Δ_i, the ≤ 10 fixed-point updates, per-fold refitting and the planned-comparison role of r. Derivations stay in Methods.

## 3. Archetype
**Schematic-led composite** built from real-data thumbnails: one real 200 × 200 hero matrix (Σ̂_i), smaller real matrices and decks, an SPD-dome glyph, network-coloured traces, vector bars, and a tinted training lane with a padlock. There are no quantitative axes and no results.

## 4. Backend
Python only (matplotlib, with nilearn available through `viz_style`), run with `/Users/reshadsazid/ASD-forever/.venv/bin/python`. No R, no AI image generation, no external API.

## 5. Journal / export contract
- **Size:** 165 mm × about 130 mm at final printed size (CBM `cas-sc` single column, text width 16.5 cm). The figure plus caption must fit on one 22.4 cm page. `bbox_inches=None`.
- **Type:** Arial, labels 7 pt, sub/superscripts at least 6 pt, panel letters 8 pt bold lowercase. At most 4 words per label, about 40 words per panel, no sentences.
- **Lines and colour:** arrows 0.75 pt, keylines 0.5 pt, lane border 1.0 pt, nothing below 0.25 pt. Colour tokens come from `viz_style.py`: blue = training role, member tabs green (static) / orange (correlation) / purple (windowed), matrices `RdBu_r` with one shared "− 0 +" key, network colours only on strips and chips. Vermilion is not needed here because no held-out path is drawn.
- **Export:** `fig_representation.{pdf,svg,png}`, 600 dpi, `svg.fonttype='none'`, `pdf.fonttype=42`, `image.composite_image=False`, named `<g id>` groups. Written to `results/latest_study/figures` and copied to `ASD-journal-I/figures`.
- **QA:**
  - `viz_style.check_layout` (6 pt floor, word limit, text overlaps, page clipping).
  - `audit_pdf_text.py --min-pt 6`.
  - Visual inspection of the full-size PNG after every render (at most 6 iterations).
  - The collision audit is NOT AUDITABLE (PyMuPDF absent).
  - The panel-alignment gate does not apply: there is a single mm canvas axes, so alignment is controlled by the shared x-columns and checked visually.
- **Source data:** CC200 time series of NYU_0050952 (176 volumes). Reference matrices are cached in `results/latest_study/cache/fig_representation_refs.npz`, computed from the illustrative training partition (strict cohort minus a random 10 %, `default_rng(42)`, 784 of 871). Deck cards are Ledoit–Wolf covariances of the first three training participants.
