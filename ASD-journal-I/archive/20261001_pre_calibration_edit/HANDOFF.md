# HANDOFF — ASD / ABIDE-I manuscript

Written 2026-10-01. Picks up a session that ran from the label-renaming work through the
reviewer-response experiments. Read this top to bottom before touching anything.

---

## 1. What this project is

A manuscript for **Computers in Biology and Medicine** (Elsevier CAS single-column,
`cas-sc.cls`), reporting a retrospective analysis of **ABIDE-I** resting-state fMRI.

**The paper is not a new method.** It uses a standard pipeline (Ledoit–Wolf covariance →
tangent-space projection → ℓ2-logistic ensemble of three connectivity views) and its
contribution is **six findings about what drives reported performance**. Keep that framing;
several past edits drifted toward advertising a method.

The six contributions, in the order the Results announce them:

1. Cognitive ability limits detection — AUROC 0.873 (IQ < 85) → 0.719 (IQ ≥ 115)
2. Somatomotor hypoconnectivity **strengthens** under within-site motion matching (−0.215 → −0.249; all 21 regions)
3. Connection-level feature selection before the split inflates AUROC 0.756 → 0.987 (region-level barely moves)
4. Linear classifiers (0.735–0.757) beat fixed-configuration tree ensembles (0.611–0.624)
5. Classifier scores reflect group membership, not autism severity (SRS ρ 0.438 across groups → 0.050 within autism)
6. The windowed-connectivity advantage is explained by covariance shrinkage (0.756 vs 0.767, p = 0.28)

Headline numbers: **AUROC 0.776** pooled, **0.774** leave-one-site-out, **accuracy 0.706** in the
1,032-participant wider cohort (matching published 0.700–0.708).

---

## 2. Paths

| What | Where |
|---|---|
| Manuscript | `/Users/reshadsazid/ASD-forever/ASD-journal-I` |
| Main file | `main.tex` (renamed from `cas-sc-template.tex`) |
| Bibliography | `references.bib` (renamed from `cas-refs.bib`), 48 entries |
| Sections | `sections/{introduction,related_work,method,result,discussion,conclusion}.tex` |
| Figures | `figures/` — `fig_overview`, `fig_representation`, `fig_designs`, `fig_graphical_abstract` |
| Analysis root | `/Users/reshadsazid/ASD-forever` |
| Results | `results/latest_study/{tables,raw,figures,review}` |
| Python | `.venv/bin/python` (**always use this**, not system python) |

---

## 3. Build

`soul.sty` is **not installed locally** (the user's preamble uses it; Overleaf has it). Builds
here are done in a scratch copy with a stub:

```bash
SP=/tmp/build && rm -rf $SP && mkdir -p $SP && cp -R ASD-journal-I/. $SP/
cd $SP
printf '%s\n' '\NeedsTeXFormat{LaTeX2e}' '\ProvidesPackage{soul}[2026/09/22 v0.0 stub]' \
  '\newcommand{\sethlcolor}[1]{}' '\newcommand{\hl}[1]{#1}' > soul.sty
for i in 1 2 3 4; do pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1
  [ $i -eq 1 ] && bibtex main >/dev/null 2>&1; done
```

**`elsarticle-num-names.bst` is also missing locally** (minimal TeX Live). For local builds
substitute `unsrtnat` — same unsorted-numeric mechanism. Overleaf has the real one.

Last verified build: **0 errors, 0 LaTeX warnings, 49 pages.**

There is **one overfull hbox (117 pt)** in the CAS front matter. It is **pre-existing** —
verified present with the graphical abstract removed entirely. Not caused by any figure.

---

## 4. Reviewer tags — 15 of 17 resolved

Tags are `\reviewhl{[AUTHORS TO RUN/VALIDATE: ...]}`. `\reviewhl` is defined in `main.tex` as
coloured italic (**not** `soul`'s `\hl`, which breaks on the math inside several tags).

### Remaining (2)

| Where | What | Status |
|---|---|---|
| `sections/method.tex:753` | Learn-then-test finite-sample guarantee — claim it precisely, or describe as an empirical threshold experiment | **Judgement call needed** |
| `main.tex:58` | (abstract text, see §7 — not a tag, matched by grep) | Stale abstract |

### Resolved, and how

- **Site-clustered bootstrap** replaces/supplements DeLong–Wilson (method §paired-comparisons, results §performance)
- **Calibration** beyond ECE: slope, calibration-in-the-large, Brier, binning sensitivity
- **Diagnosis-by-IQ interaction** tested directly, BH across the IQ family
- **Cohort flow** reconciled 1,112 → 1,035 → 871
- **Freedman–Lane** residual permutation for system and regional contrasts
- **Vectorisation convention**: 19,900 = 200·199/2, strict upper triangle, diagonal dropped, **not** Frobenius-isometric — now stated
- **Reproducibility manifest**: versions, seed 42, atlas, participant-ID SHA-256, fold recipe
- **Nesting scope**: qualified as nested at the outer level only
- **Prediction interval**: now descriptive; site estimates are not independent studies
- **Harvard–Oxford**: `ho` = ABIDE Preprocessed `rois_ho.1D`, 111 regions → cited to Craddock 2013. The old **Desikan citation was wrong** (different parcellation)
- **Back matter** written: Data/Code availability, Ethics, Funding (none), Competing interests (none), Acknowledgements, generative-AI declaration

Also present: **~95 inline `\reviewhl{}` wording edits** across six files (not tags). Untouched.

---

## 5. Experiments run this session

Scripts created in `/Users/reshadsazid/ASD-forever`:

### `review_experiments.py`
Site-clustered bootstrap, calibration, IQ interaction, cohort flow, manifest.
Outputs → `results/latest_study/review/`.

Key results:
- Bootstrap AUROC pooled **0.776 [0.733, 0.807]** vs DeLong [0.745, 0.807] — parametric intervals
  are anti-conservative **at the lower bound**
- Calibration slope **1.117 [0.866, 1.321]** (includes 1 → well dispersed); Brier 0.194;
  **calibration-in-the-large −0.162 [−0.282, −0.061]** (excludes 0 → slight over-prediction)
- ECE depends on binning: 0.037 (5 bins) / 0.046 (10) / 0.059 (20, some empty)
- Diagnosis×IQ interaction **negative**, survives BH for FIQ (q = 0.025) and VIQ (q = 0.025)
  pooled, both q = 0.033 LOSO; **PIQ does not survive** (q = 0.082)
- Cohort flow: 1,112 records → 1,035 with derivative → 871 primary (403 autistic / 468 not)

### `exp_freedman_lane.py`
Proper Freedman–Lane: permutes **reduced-model residuals** within site and refits with the
diagnosis vector in its true position, so diagnosis keeps its covariate association.
5,000 permutations, seed 42.

**Conclusions hold and strengthen.** System pairs negative with q<0.05: 14 → **27 of 36** after
motion matching. Somatomotor pair −0.226 → **−0.294**. **All 21 regions significant** in both.

### `exp_region_selection_nested.py` — COMPLETE
Reranks regions with the nuisance model fitted **on training rows only**.

**Conclusion unchanged.** Mean |Δ| against the published training-fold condition = **0.0035**,
max 0.0085. Ordering across list lengths preserved; every restricted model still below all-196
regions (0.781). `method.tex:794` resolved with these numbers; results written into
`sections/result.tex`.

---

## 6. TRAPS — read before computing anything

**Stored OOF predictions are NOT in `metadata.csv` order.** This silently produced a
*sign-flipped* IQ interaction before it was caught.

- **LOSO** order = metadata sorted by site (`np.argsort(site, kind='stable')`)
- **Pooled** order = concatenated test folds of `StratifiedKFold(10, shuffle=True, random_state=42)`

Both were verified by checking stored labels **and** sites match 1.000. **Always verify before
joining OOF predictions to phenotype.** The validation target: within-autistic score–FIQ
ρ = −0.186 and non-autistic +0.042 (paper reports −0.18 / +0.05).

**Region effects are computed WITHIN the somatomotor system** (21×21 submatrix, 210
connections), not over all incident connections in the 196-region matrix. Using all incident
connections gives −0.079 instead of −0.215 and collapses significance from 21/21 to 1/21.

**Merge key** between `metadata.csv` and the phenotypic CSV is the numeric `SUB_ID`, not `FILE_ID`.

**`main.tex` has reverted from Overleaf at least twice**, losing the abstract rewrite and the
bibliography style. Check both before trusting the file.

---

## 7. Open decisions (author's call)

1. **Title.** Currently `CLEAR-ASD: Calibrated and Interpretable Autism Classification from
   Functional Connectomes`. "Calibrated" is hard to defend: the Results say of that whole block
   *"none of them is claimed as a contribution"*, and calibration-in-the-large excludes zero.
   Either promote calibration back to a claimed contribution (the numbers now exist to support
   it) or retitle. Suggested, preserving the acronym:
   **`CLEAR-ASD: Cohort- and Leakage-Aware Evaluation of Resting-state connectomes for Autism classification`**

2. **The abstract in `main.tex` is STALE.** It still opens *"ASD classification … remains
   difficult because…"*, sells calibration and selective prediction, and quotes 0.79/0.72 —
   values the paper does not report. A corrected 250-word version was written earlier in the
   session and lost to a revert. **It must be rewritten before submission.**

3. **CRediT** roles for authors 2–6 — skeleton left in `main.tex` with a comment; author 1 filled.

4. **Generative-AI declaration** in the back matter is written as "none used" — amend to match
   reality. Note Elsevier **prohibits** AI-generated artwork; the graphical abstract was drawn by
   the author, which is fine.

5. **Atlas/fusion claim.** The Discussion says "three atlases perform no better than one". This
   is now supported (Methods §par:atlas-fusion + Results), backed by `tables/atlas.csv`.

---

## 8. Style rules (standing, from the author)

- Present simple, passive, academic. **No bullet lists in subsections** (the Introduction
  contribution list is the sole exception).
- Sentences connected with connecting words; long sentences split in two.
- Beginner-readable; explain *why* each step is done.
- **Every equation, table and figure cited in text.** No backward references ("as mentioned in
  §3.1").
- **Results belong in the Results section**, never in Methods. Methods describes procedure only.
- Labels are descriptive: `sec:`, `subsec:`, `subsubsec:`, `par:`, `eq:`, `fig:`, `tab:` +
  lowercase hyphenated names. 113 labels, all resolving.
- Cohort naming: **primary** (871) / **wider quality-control** (1,032). "Strict" was renamed out.
- Journals in `references.bib` are **ISO 4 abbreviated**; proper nouns in titles are
  brace-protected (the style sentence-cases otherwise, which silently lowercased
  "alzheimer's", "bayes", "gnn-lstm").

---

## 9. Figures

Generated by `viz_method_overview.py`, `viz_method_representation.py`, `viz_method_designs.py`,
`viz_graphical_abstract.py`, all importing `viz_style.py`. Run with `.venv/bin/python`; each
prints `layout check: N texts, 0 problem(s)` — **0 problems is the gate**.

- Times New Roman throughout; verify with `vs.pdf_fonts(...)` (no DejaVu fallback)
- Arrows are **bold** (`LW_ARROW = 1.3`, heads 2.4×1.9) and semantically colour-coded: grey data
  flow, dashed blue fitted-then-applied, vermilion held-out labels, purple matching, green
  aggregation, teal/brown linear/tree branches
- Graphical abstract: 210 × 96 mm (2.2:1), 4960 × 2267 px, placed via the CAS
  `graphicalabstract` environment on page 1, and also the standalone Elsevier upload
- Every number in the figures is read from `results/latest_study/tables/` at render time

---

## 10. Suggested next steps

1. Decide the title; rewrite the stale abstract to match
2. Resolve `method.tex:753` (learn-then-test: claim precisely or reframe as empirical)
3. Fill CRediT
4. Work through the ~95 inline `\reviewhl{}` wording edits
5. Still genuinely blocked: **external validation on ABIDE-II** (different preprocessing
   pipeline; ~1 TB of BOLD), and **rerunning preprocessing inside every inner CV split**. Both
   are declared as outstanding in the Discussion — keep that declaration honest.

---

## 11. PENDING TASK — calibration and interpretation as contributions (opened 2026-10-01)

**Goal.** Make calibration and interpretability into claimed contributions of the paper, which
needs experiments the paper does not yet contain. The title is decided later and must not mention
leakage. **Nothing in the manuscript has been edited for this yet.**

**State at handover.** A sequential ladder is running in the background
(`/Users/reshadsazid/ASD-forever/run_review_ladder.py`, 16 stages, run under `caffeinate`). It halts
instead of cascading: a stage starts only when the previous one exited 0 **and** its output
validator passed. Status, logs and events are in `results/latest_study/review/ladder/`
(`status.json`, `ladder.log`, `events.jsonl`, `logs/NN_stage.log`; `RUN_REPORT.md` and
`outputs_manifest.json` are written at the end). Watch it with
`.venv/bin/python ladder_watch.py --replay`. Resume after any interruption with
`.venv/bin/python run_review_ladder.py` (finished stages are re-validated, not re-run;
`--from <stage>` forces a stage; `--dry-run` lists every stage and whether it is already valid).

| stage | what it produces |
|---|---|
| pooled_refit, loso_refit | refit of the ensemble on every outer fold with 5-fold inner cross-fitting, weights, structure coefficients, held-out features (`review/nested/`) |
| pooled_preview_* | pooled-only preview of both analyses (`review/preview_pooled/`) |
| null_permutations, splithalf | 100 within-site label permutations; 40 independent split-halves with 30 permutations each |
| fully_nested_pooled / _loso | tangent reference and scaler refitted inside every inner split when choosing C, plus a grid extended to 1e-5 (`review/fully_nested/`) |
| final_calibration | `calibration_nested*.csv`, reliability bins, per-site calibration, recalibration parameters |
| final_interpretation, null_analysis, splithalf_analysis | `interpretation_*.csv` |
| figures | `fig_calibration`, `fig_interpretation` in `results/latest_study/figures/` |

**Decisions already fixed (do not revisit without cause).**
- The **intercept-only** recalibration is the pre-specified primary map; logistic and isotonic are
  sensitivity analyses; the unpenalised-intercept refit is a diagnostic, not a recalibration.
- The native calibration offset comes from liblinear penalising the intercept (fitted intercept
  about -0.016 against about -0.25 unpenalised). The fully nested penalty search and its extended
  grid test whether the published C = 0.001 (the smallest value of the grid, chosen in every
  outer fold) is a boundary artefact.
- These analyses were specified **after** the primary results. The manuscript's evidential
  framework (Methods, analysis status) must label them secondary, not primary descriptive.

**Code changes made for this task** (all compile; the original is kept at
`scratchpad/gpu_tangent.py.orig` for the tangent module): `gpu_tangent.py` now falls back to
float64 when `torch.linalg.eigh` fails to converge (it did, once, in an inner split of pooled fold
8; the four non-failing splits were verified bit-identical to the original). New scripts:
`exp_nested_outer.py`, `exp_calibration_nested.py`, `exp_interpretation.py`,
`exp_interpretation_null.py`, `exp_interpretation_splithalf.py`, `exp_fully_nested_c.py`,
`viz_calibration_interpretation.py`, `run_review_ladder.py`, `ladder_watch.py`.

**Next steps once the ladder reports COMPLETE.**
1. Read `review/calibration_nested*.csv` and `review/interpretation_*.csv`; state only what the
   numbers support (the interpretation results may be weaker than hoped; report them as found).
2. Apply the surgical edit plan (see section 12 when it exists): Introduction contributions and
   scope, a recalibration subsubsection and an interpretation subsection in Methods, two new
   Results subsections with two figures, Discussion and Conclusion, abstract, highlights, keywords,
   title.
3. Fix the inconsistencies found in the audit of "resolved" reviewer tags (section 12).

---

## 12. AUDIT OF THE "RESOLVED" REVIEWER TAGS AND OF TABLES/FIGURES (2026-10-01)

Method: each pasted reviewer tag was traced to the text, and every numeric cell of the 12 LaTeX
tables was compared with `results/latest_study/tables` (strict rounding): **192 of 192 cells match**.
The four Results figures were read panel by panel against the tables and the text.

| item | status | where / what is left |
|---|---|---|
| Vectorisation convention (19,900) | resolved | method.tex:332-346 (strict upper triangle, no sqrt 2, not an isometry, diagonal omitted); Fig. 2 shows `vecu` and 19,900; code agrees |
| Fully nested inner-split fitting | **open, qualified** | method.tex:563-567, discussion.tex:179-183. Feasible (about 30 min); now run by the ladder (`fully_nested_*`). The earlier belief that it is blocked was wrong |
| LOSO overlap, random-effects interval | mostly resolved | **still contradictory:** method.tex:622-624 ("expected range of AUROC at a new site"); Fig. 3 caption (b) ("prediction interval for a site not yet seen") and the "95% PI" text drawn inside panel b |
| ECE plus slope, CITL, Brier, intervals, binning | resolved in the body | abstract and title still sell calibration (tag main.tex:58); bin counts (86/17/0) are in the CSV, not the text |
| Learn-then-test guarantee | **open** | tag at method.tex:753; Results already treat it empirically, so the smallest fix is to reframe it and state the exchangeability assumption |
| Region selection, strictly nested | resolved | method.tex:794, result.tex:514-523; **discussion.tex:183-184 still says it is not nested (stale)** |
| Freedman-Lane | resolved | method.tex:968, result.tex:425-433; **discussion.tex:184-186 still says it has not been substituted (stale)** |
| Site-aware uncertainty | resolved | result.tex:66-77; Table 3 and Fig. 3a still show only DeLong/Wilson intervals |
| External validation (ABIDE-II) | open, blocked | research-cohort framing is kept; no clinical claim (discussion.tex:136, 186-189) |
| Diagnosis-by-IQ test, multiplicity | resolved for FIQ and VIQ | IQ-balanced independent sample is future work; **stale inline tag discussion.tex:57** |
| Cohort counts | partly resolved | counts agree everywhere, but there is no flow table, and "1,035" names two different sets (records with a preprocessed derivative vs rater-3 functional OK; overlap 1,033; `review/cohort_flow.csv` row says 1,033) |

Other inconsistencies found: the Discussion paragraph "Four requirements nevertheless remain open"
lists two items that the Results have already closed; Fig. 3d and `fig_guarantee` (b) show the
same quantities with the colours for "cases answered" and "splits meeting target" swapped;
result.tex:428-432 reports -0.226 to -0.294 both for the somatomotor pair and for the mean regional
effect, on the regression-coefficient scale and in the 512-participant sample, whereas the primary
regional mean is -0.215 to -0.249 in 560 participants; method.tex:8 says "nested cross-validation"
without the outer-level qualifier; abstract values 0.79 and 0.71 do round correctly from the wider
cohort (0.789, 0.706) but the abstract does not say which cohort.

---

## 13. LADDER COMPLETE (2026-10-01 06:25): results in one place

All 16 stages DONE and validated (`results/latest_study/review/ladder/RUN_REPORT.md`; 168 output files hashed
in `outputs_manifest.json`). The refits reproduce the stored predictions to 2.1e-07 (pooled) and 1.5e-07
(LOSO); contributions add to the score to 2e-15. No manuscript file has been edited yet.

- **Calibration** (`review/calibration_nested*.csv`): native CITL -0.162 [-0.287, -0.048] pooled and -0.181
  [-0.389, 0.019] LOSO; intercept recalibration (primary) -0.010 [-0.136, 0.103] and +0.023 [-0.186, 0.235];
  AUROC change -0.0001 and -0.001; Brier, slope and ECE unchanged within sampling error; sensitivity falls
  (0.705 to 0.655 pooled, 0.742 to 0.625 LOSO) and specificity rises; site-level miscalibration persists under
  LOSO (I2 63%, tau 0.37). The paper's old slope and CITL intervals used 400 resamples, the new ones 2,000.
- **Fully nested penalty** (`review/fully_nested_c_summary.csv`): C changed in 2 of 30 (pooled) and 2 of 60
  (LOSO) cases; AUROC 0.7760 to 0.7756 and 0.7745 to 0.7746. C = 0.001 is not a grid-boundary artefact.
- **Interpretation** (`review/interpretation_*.csv`): default-mode pairs carry the largest share of the group
  separation; per connection, somatomotor-somatomotor is largest; 13 of 36 pairs pass a familywise permutation
  test; agreement between overlapping training partitions (0.91-0.98) is almost as high under permuted labels
  (0.87-0.89), so it is uninformative; independent split-halves replicate at system-pair level (rho 0.60-0.70,
  null <= 0.08) and weakly at connection level (0.18-0.53); deleting the 12 top-ranked pairs lowers AUROC to
  0.707 against 0.757 for random pairs.
