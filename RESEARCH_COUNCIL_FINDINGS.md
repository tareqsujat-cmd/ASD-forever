# Research council findings — novelty analysis for the ABIDE-I paper

Produced by a seven-agent survey: top-venue literature scan (2025–26), codebase asset audit,
adjacent-field transfer scout, NeurIPS/AAAI methodology transfer, conformal-under-shift
specialist, and two dedicated novelty-verification agents (Bayes-error prior art;
site-effect / domain-generalisation prior art; IQ/sex disparity prior art).

**Status of the numbers below.** Findings marked *(measured)* were computed by an agent
against this repository's own data during the survey and have **not been independently
reproduced**. They are promising, not banked. See §7.

Companion artifact (same content, formatted):
https://claude.ai/code/artifact/7c3889e6-935e-4f43-b995-cb541e2cad6d

---

## 0. What survived testing

Every *(measured)* figure below was subsequently recomputed directly. This section records
the outcome, because several of the agents' numbers did not reproduce and the ones that
failed were consistently the more publishable ones.

| claim | agent figure | recomputed | verdict |
|---|---|---|---|
| Windowed vs whole-scan is a regularisation artefact | +0.0124, p = 0.0039 | **+0.0127, p = 0.0036**; reverses to −0.0025 (p = 0.709) at matched shrinkage | **confirmed, and stronger** |
| Label-noise ceiling | 0.926 on 275 participants | **0.9535 on 213**, 15.0% instrument disagreement | **confirmed, corrected upward** |
| Severity correlations are a pooling artefact | — | SRS **0.455 pooled → 0.061 within-autism** | **confirmed** |
| Subjects saturate at n ≈ 400 while duration does not | monotone rise to T = 232, flat in n | **both axes rise equally** (ρ = +0.93 vs +0.89, p = 0.003 / 0.007) | **falsified** |
| Homogeneous training data transfers better | 172 within-site > 825 pooled | **+0.0016, p = 0.939** — the gap is easier evaluation (+0.079), not data value | **falsified** |
| Multicalibration subgroup audit (IQ, sex) | sensitivity 0.754 → 0.621, p = 0.032 | computed on the transductive rank score, on strata contaminated by 37 `-9999` sentinels; the trend vanishes at matched predicted-positive rate | **withdrawn** |

Two further defects were found in this repository during verification, both of which had
been producing published-looking numbers:

- **The adopted ensemble score was transductive.** Ranks were taken within the held-out
  fold, which under leave-one-site-out is an entire site. Correcting it to rank against the
  training distribution costs **0.020–0.028 AUROC**, and the corrected headline is the
  averaged probability at 0.775 / 0.707 (871) and 0.791 / 0.707 (1,032). The matched-cohort
  accuracy that placed this work above the published results was that artefact.
- **The abstention guarantee was circular.** The threshold was fitted and its bound reported
  on the same participants. Evaluated honestly, the cohort-wide rule holds its target in
  82% of splits against a nominal 90%, while the per-site rule holds it in 91%.

**Net position: two contributions, not four.** The regularisation correction and the
label-noise ceiling. The scaling law and every subgroup claim did not survive.

---

## 1. The diagnosis

Accuracy on ABIDE-I has sat at 0.70–0.75 for a decade across autoencoders, population-graph
GNNs, transformers and state-space models. Flatness across model classes that different is
the signature of a **data limit, not a modelling limit** — and no published paper has tested
that hypothesis.

Reporting has degraded to the point where the field's headline results are being withdrawn:

| Count | Finding |
|---|---|
| 0 | of ~20 surveyed top-venue papers report calibration — no ECE, no Brier, no reliability diagram. Europe PMC full-text search on ABIDE returns nothing. |
| 0 | report a risk–coverage curve, or a confidence interval on the headline number. |
| 13 | JBHI / MedIA 2025–26 abstracts report *no absolute accuracy at all* — only "outperforms state-of-the-art" or a bare delta. |
| 2 | ABIDE papers retracted in 2025–26; the 98.2% eClinicalMedicine paper carries an Expression of Concern. |
| 146 | teams in the IMPAC challenge converged at AUC ≈ 0.80 — the plateau is collective. |

Accuracy bands, and the band predicts the rigour:

- **0.70–0.78** — credible protocols. Trifocal Transformer 71.2% (JBHI 2026), MSPPGT 72.2%
  (JBHI 2026), DGAT 71.99% (nested CV), VarCoNet AUC 72.6% (HBM 2026).
- **0.82–0.88** — protocols get vague, CV strategy often unstated.
- **0.95–0.99** — every instance is on a tiny subsample, under editorial concern, or retracted.

Your 0.716 sits squarely inside the credible band. A 2026 JBHI paper calls **72.2%
"state-of-the-art."**

---

## 2. The four contributions

One argument in sequence: the ceiling is not where you think it is, the axis that still pays
is measurement rather than modelling, so what a deployable system owes you is a guarantee
rather than a number — and the number everyone reports conceals who the system fails.

### C1 — Accuracy is bought with scan minutes, not subjects or architectures

**Status: open. Strongest of the four.**

Import the Rosenfeld et al. two-axis scaling form `ε(m,n) ≈ a·n^(−α) + b·m^(−β) + c_∞`,
replacing model size with **scan duration**.

*(measured)* Within-subject truncation, n = 321 with T ≥ 232 — the same people at every
point, so nothing is confounded with who was scanned:

| frames retained | 40 | 60 | 90 | 120 | 160 | 200 | 232 |
|---|---|---|---|---|---|---|---|
| AUROC | 0.716 | 0.724 | 0.723 | 0.751 | 0.763 | 0.766 | **0.770** |

Still rising at the longest scans available. The subject axis does the opposite — it
**saturates by n ≈ 400**: 0.634 (n=100) → 0.748 (400) → 0.746 (550) → 0.754 (700) → 0.736 (825).

The two axes separate cleanly across sites: per-site LOSO AUROC tracks site median scan
length (Spearman ρ = 0.461, p = 0.041) but **not** site sample size (ρ = 0.143, p = 0.55).
Joint OLS: β_T = 0.039/SD vs β_logn = 0.006/SD.

**Report the derivative, not the asymptote.** Seven noisy points over a 6× range cannot
identify a ceiling; the naive power-law fit is degenerate, driving c_∞→0 and implying
AUROC 1.0. The defensible claim is the slope and the contrast between axes.

**Confound to close before publishing:** tune the regulariser per T and use shrinkage /
tangent-space covariance at every T, so the effect cannot be dismissed as estimator
conditioning. Run the subject-axis sweep within NYU (n=172) separately, since "more
subjects" currently also means "more site heterogeneity."

**Contribution sentence.** *"Using within-subject truncation of the BOLD series as a
controlled measurement-budget axis, we show that ASD classification on ABIDE-I saturated in
subjects by n ≈ 400 and is invariant to model class, but has not saturated in scan duration:
a sixfold increase in retained frames buys +5.4 AUROC points, and per-site performance tracks
scan length rather than cohort size."*

**Cite / differentiate:**
- Rosenfeld, Rosenfeld, Belinkov & Shavit, ICLR 2020 — https://arxiv.org/abs/1909.12673
- Kiessner et al., *Comput Biol Med* 178:108681 (2024), EEG dataset/model scaling — doi:10.1016/j.compbiomed.2024.108681
- **Closest threat:** Horien et al., *Nature Mental Health* 2026, "Optimizing functional
  connectivity scanning conditions for predicting autistic traits" — doi:10.1038/s44220-026-00623-7.
  Dimensional traits, not diagnosis. Must be cited and distinguished explicitly.

### C2 — A certified ceiling from ABIDE's two diagnostic instruments

**Status: open in fMRI. Verify first. EEG precedent must be conceded.**

The obvious explanation for a decade-long plateau is noisy labels. It is testable, and it is
**false** — which is the more useful result.

*(measured)* ABIDE ships **two independent gold-standard instruments per case**. 275 ASD
participants have both a complete ADI-R algorithm and ADOS_TOTAL. At standard cutoffs
(ADI-R A≥10, BV≥8, C≥3, D≥1; ADOS≥7):

| Agreement pattern | Share |
|---|---|
| Both positive | 86.5% |
| ADI-R only | 5.1% |
| ADOS only | 8.4% |
| Neither | 0% |
| **Disagreement** | **13.5%** |

That is a two-rater design, and it is what a label-noise estimator consumes. Ishida's
plug-in on the ASD arm gives β̂ = 0.067; control arm ρ₀ ≈ 0.017 (SRS ≥ 60 in 3 of 175
measured controls). Hence:

- **Certified ceiling: 1 − (π₀ρ₀ + π₁ρ₁) = 0.926 accuracy / 0.924 AUROC**
- **Noise-corrected AUROC: 0.749 → 0.794** (0.761 → 0.808 on the alternative baseline)

So label noise explains **~4–5 AUROC points, roughly a fifth of the gap — not the plateau.**

A second, independent route reached the same place: SRS is the only instrument covering both
groups (n = 300 within the 871 cohort), separates them at AUROC 0.973 with 8% discordance,
and classifier error on discordant subjects is not significantly elevated (Fisher p = 0.26).
Two different methods converging near 0.92 is reassuring.

**Publish the decomposition, never "the labels are bad."** SRS at 0.973 and SCQ at 0.977 mean
the diagnostic label is *highly recoverable from behaviour* — it is not noise-dominated.

**Caveat to state plainly:** SRS is parent-reported and often collected during the diagnostic
workup, so it is not independent of diagnosis. 0.973 is inflated and 8% is a *lower bound* on
true noise.

**Positioning — the ceiling framing is established in EEG and must be conceded:**
- Gemein et al., *NeuroImage* 220:117021 (2020) — doi:10.1016/j.neuroimage.2020.117021.
  *"accuracies ... could saturate near 90% due to the imperfect inter-rater agreement of the
  clinical labels."*
- Kiessner et al., *Comput Biol Med* 178:108681 (2024), "Reaching the ceiling?" — saturation at 85–87%.
- Poziomska et al., arXiv:2411.17709 (2024) — asymptotic AUC ceiling ≈ 91%, proposes fuzzy labels.

**Never ported to fMRI:** all 22 papers citing Gemein 2020 that mention autism/fMRI are EEG
papers. PubMed for an ASD ceiling from ADOS/ADI-R reliability returns **zero**. OpenAlex for
(ABIDE OR autism) ∧ ("Bayes error" OR "irreducible error" OR "accuracy ceiling") returns
**exactly one** work, about ironic prosody perception.

**Adjacent work to cite, not claim:**
- Schulz et al., *Cell Reports* 43(1):113597 (2024) — irreducible error via learning curves,
  UK Biobank, continuous phenotypes, not diagnosis. ⚠️ Full text unverified — read before citing.
- Gell et al., *Nat Commun* 15:10678 (2024) — measurement noise limits brain-behaviour prediction.
- Rokham/Calhoun line (BPS:CNNI 2020; Xing et al. *iScience* 2024) — detects label noise in
  psychiatric MRI but *cleans* it, never bounds achievable accuracy.
- Plateau conjecture you would be quantifying: Dong et al., *HBM* 46(5):e70190 (2025);
  Traut et al., *NeuroImage* 255:119171 (2022).
- Menon et al., ICML 2015 — https://proceedings.mlr.press/v37/menon15.pdf
- Ishida et al., ICLR 2023 — https://arxiv.org/abs/2202.00395
- Ushio, Ishida & Sugiyama, arXiv:2609.02304 (2026) — extension to BER/AUC.

### C3 — Certify performance at an unseen scanner without labels

**Status: open. Has an actual theorem.**

Leave-one-site-out **destroys subject-level exchangeability**, so every conformal guarantee
calibrated on subjects is void at a held-out site. The repair: move the exchangeable unit.
Treat the 20 sites as i.i.d. draws from a hyper-distribution over sites — the assumption you
are *already implicitly making* by claiming LOSO estimates generalisation to new hospitals.

Apply **Learn-then-Test to per-site Clopper–Pearson bounds** on selective error, walking a
threshold grid strict-to-lenient under fixed-sequence FWER control:

```
P_calib( P_{S~Q}[ P(error | answered, S; λ̂) > α ] ≤ γ ) ≥ 1 − δ
```

*With confidence 1−δ over our 20 calibration sites, at most a fraction γ of future hospitals
will exceed selective error α.* With K = 20 the smallest achievable γ is
**1/(K+1) ≈ 0.048** — quote that as the resolution limit twenty sites impose, itself an
honest finding.

Cost: 20 × 100 binomial tails. Milliseconds, on LOSO scores already stored. No retraining.

**The Riemannian shift score is rehabilitated** — demoted from fudge factor to two
legitimate roles:

1. **Site-level out-of-scope test.**
   `p = (1 + #{k : d_R(Σ_k, ref) ≥ d_R(Σ_new, ref)}) / (K+1)` is *exactly valid* under site
   exchangeability. If p ≤ 0.05, refuse the whole site rather than inflating a threshold.
2. **Conditional-coverage covariate.** Feed Φ = [1, d_R, d_R²] as a finite-dimensional shift
   class into Gibbs–Cherian–Candès for exact finite-sample coverage across that span.

Pair with a label-free *point* estimate (average-thresholded-confidence) so a new hospital
gets both what to expect and what is guaranteed. ⚠️ ATC's authors report the effect is
NN-specific — a real risk for linear models, and worth reporting either way.

**Concede these; claim only what remains:**
- Your Clopper–Pearson rule **is** Learn-then-Test — Angelopoulos et al., https://arxiv.org/abs/2110.01052
- Per-site Mondrian: Vovk et al. 2003; Shahid, arXiv:2606.20115 (2026) — 20 hospitals,
  explicitly does **not** handle held-out sites.
- Weighted conformal: Tibshirani, Barber, Candès & Ramdas, NeurIPS 2019 —
  https://arxiv.org/abs/1904.06019. **Estimated weights forfeit the finite-sample guarantee
  entirely** — there is no theorem for the estimated-weight case.
- Unseen-site machinery: Park et al., NeurIPS 2022 (PAC prediction sets for meta-learning);
  Gibbs, Cherian & Candès, JRSS-B 2025 — https://arxiv.org/abs/2305.12616
- Federated/multi-site: Lu et al., ICML 2023 — https://arxiv.org/abs/2305.17564
- ATC: Garg et al., ICLR 2022 — https://arxiv.org/abs/2201.04234

**Load-bearing words in any claim: *unseen*, *finite-sample*, *rs-fMRI*.**

### C4 — The reported metric conceals who the model fails

**Status: mixed. Lead with calibration, not sensitivity. Verify first.**

*(measured)* On stored predictions:

| Quantity | Value |
|---|---|
| Sensitivity, male vs female | **0.692 vs 0.565** |
| AUROC, male vs female | 0.746 vs 0.748 — **identical** |
| Sensitivity across IQ bands | 0.754 (FIQ≤95) → 0.684 → **0.621** (110–125) → 0.642 (>125) |
| Trend test | Cochran–Armitage z = −2.15, **p = 0.032** |
| Calibration gap by IQ | **−0.091** (FIQ<99) to **+0.063** (FIQ 109–119) |

**Ordering matters — novelty check came back mixed:**

1. **IQ-stratified calibration — genuinely novel.** Nobody has audited subgroup calibration
   by IQ in any psychiatric or neuroimaging classifier. Strongest single claim here.
2. **Sex sensitivity gap — novel in this setting, borrowed pattern.** Nielsen 2013 tested sex
   on ABIDE but reported *accuracy only*, so would have missed exactly this. The
   AUC-parity-conceals-a-TPR-gap phenomenon is famous elsewhere.
3. **IQ → sensitivity decline — direction already published.** Nielsen 2013, on this dataset:
   *"as social function, verbal IQ, and daily living skills increased, a subject was more
   likely to be classified as a control."* Their effect was pooled, VIQ-only, and effectively
   null (r = −0.07, p = 0.05). Your within-diagnosis quantification is new; the observation
   is not. **Cite Nielsen; do not claim first observation.**

**The package is the contribution.** A full subgroup fairness *and* calibration audit of an
ABIDE classifier has no published equivalent, even though each individual finding has a near
neighbour.

**Framing:** the two groups the model fails on — cognitively able, and female — are precisely
the two documented as under-diagnosed. A screening tool with these properties would amplify
existing diagnostic inequity, and the field's chosen metric is structurally blind to it.

#### ⚠️ The severity null must be reframed — highest risk item in the dossier

*(measured)* Classifier score is uncorrelated with symptom severity **within** the autism
group: ADOS total ρ = +0.013 (p=0.81), Gotham severity ρ = +0.047 (p=0.47), SRS ρ = +0.098
(p=0.20), ADI-R social ρ = −0.054 (p=0.31). This matches `symptom_axes.csv`: 0 of 120 tests
survive FDR, best q = 0.113.

**But four papers report positive classifier–severity associations:**

| Study | Finding |
|---|---|
| Nielsen 2013 (same dataset) | ADOS soc+comm r = 0.13 (p=0.01); SRS r = 0.17 (p=0.002) |
| Plitt 2015 | P(ASD\|fMRI) vs SRS sum r = 0.51 (p = 2.79×10⁻⁸) |
| Yahata 2016, *Nat Commun* | ADOS communication r = 0.44 (n=58, within-ASD) |
| Uddin 2013, *JAMA Psychiatry* | salience-network classifier explicitly linked to severity |

**The nuance that saves it:** Nielsen's and Plitt's correlations are computed on the **pooled
ASD+TD sample**. When controls sit at floor on ADOS and SRS, *any* working classifier yields a
positive pooled correlation — arithmetic from group separation, not severity tracking.

**Required action:** reproduce the pooled correlation on your own data, recover Nielsen's
~0.13–0.17, then show it collapses within-diagnosis. That contrast is the publishable finding.
A bare "zero correlation" claim is dead on arrival with any reviewer who knows Nielsen 2013.

**Cite as motivation, never as contribution:**
- Oakden-Rayner et al., CHIL 2020, hidden stratification — https://pubmed.ncbi.nlm.nih.gov/33196064/
- Seyyed-Kalantari et al., *Nature Medicine* 2021 — https://pubmed.ncbi.nlm.nih.gov/34893776/
- Hébert-Johnson et al., ICML 2018, multicalibration — https://arxiv.org/abs/1711.08513
- Barda et al., *JAMIA* 2021, clinical instantiation — https://pubmed.ncbi.nlm.nih.gov/33236066/
- Kim et al., *PNAS* 2022, universal adaptability — doi:10.1073/pnas.2108097119 (bridges to C3)

**Direct opponent to engage:** *PNAS* 2023, doi:10.1073/pnas.2211613120 — audits ASD imaging
classifiers across gender, age and race, concludes largely **unbiased**. But audits AUC and
fairness metrics, **not calibration**, and never uses IQ as a subgroup axis. That omission is
your wedge, and a reviewer will raise it.

**Statistical caution:** ABIDE-I leaves roughly 60–70 autistic females; 0.692 vs 0.565 on that
n has a wide CI (Nielsen's null was on 51). The IQ calibration swing sits at ~250/stratum
where ECE is noisy. Report CIs and formal tests or both findings get killed in review.

**Novelty check cleared:** Floris et al.'s program concerns sex differences in brain *features*
and sex-*prediction* models, not classifier sensitivity disparity. No overlap.

---

## 3. The strongest reserve — displaces any of the four if one fails

### Is the site effect covariate shift, or a change in P(Y|X)?

**Nobody has tested it.** The community *implicitly assumes* pure covariate shift — that
assumption is literally what ComBat's location/scale model on X encodes — and never checks.

- Europe PMC: `"site effects" ∧ "covariate shift" ∧ (MRI ∨ neuroimaging)` → **1 hit, unrelated.**
- Bayer et al., *Front Neurol* 13:923988 (2022), the field's standard review of site effects,
  **does not use the framing at all.**

Your 0.584–0.899 per-site spread is consistent with *either*: some sites simply have worse SNR
and shorter runs (pure covariate shift), or different sites recruited different autistic
phenotypes (genuine ΔP(Y|X)). Those imply opposite remedies, and the field has been applying
the covariate-shift remedy without checking.

**The test: a per-site pairwise transfer matrix.** Train on site *i*, test on site *j*, all
pairs. Pure covariate shift predicts low-rank, roughly symmetric structure driven by site
quality. A genuine change in the decision boundary predicts clustered block structure driven
by cohort composition. Twenty sites, one small sweep.

**Nearest prior work:** Yamashita et al., *PLoS Biology* 17(4):e3000042 (2019) — travelling-
subject design (9 subjects × 12 sites, 411 sessions) decomposes site variance into engineering
**measurement bias** (pure X-shift) and biological **sampling bias** (shifts P(Y) and P(X|Y)).
Effect-size SDs: participant 0.0662 > **measurement bias 0.0411** > SCZ 0.0377 > MDD 0.0328 >
**ASD 0.0297**. Site measurement bias on FC *exceeds* the ASD effect. Strongest evidence that
ABIDE site effects are not pure covariate shift — but never stated in shift terminology, and
not on ABIDE.

---

## 4. Collisions — must be addressed in paragraph one, not related work

### Clarke et al., *GigaScience* 14:giaf091 (2025) — PMID 40899917

**Already published conformal prediction with abstention on ABIDE.** Transductive CP,
ABIDE-I (n=452, 10 sites) → ABIDE-II (n=424, 16 sites), males only, MIST_20 atlas, Pearson FC.
Retained 10/424 (2.4% coverage), PPV 90%, sensitivity 4.2%, specificity 99.5%.

Four openings it disclaims in its own text:

1. **Transductive** — no valid marginal coverage guarantee in the split-conformal sense.
2. **One extreme operating point.** No risk–coverage curve exists for ABIDE.
3. **No calibration analysis** of any kind.
4. **No ceiling, no site handling** — states outright they were "unable to account for site
   effects," and that no performance ceiling is described. Male-only, which C4 makes pointed.

### Other papers to watch

- **BrainLinear**, arXiv:2608.15266 (Aug 2026) — Fréchet-mean tangent space + Top-K ROI-pair
  selection + shallow MLP on ABIDE. Your nearest methodological neighbour, weeks old. No
  uncertainty, calibration, abstention or statistical testing. That absence is where you separate.
- **Nouretdinov et al.**, *NeuroImage* 56(2):809–813 (2011), PMID 20483379 — founding CP paper
  in psychiatric neuroimaging (depression). Does not address site shift.
- **Ingalhalikar et al.**, *IEEE TBME* 68(12):3628–3637 (2021), doi:10.1109/TBME.2021.3080259 —
  claims ComBat **improves** ABIDE FC classification. **The paper to rebut.** ⚠️ Paywalled; no
  agent could confirm whether ComBat was fitted before or after the train/test split. If
  before, Serra 2023 + Marzi 2024 give you the leakage argument.
- **Han et al.**, npj AI 2026, arXiv:2501.17207 — message passing *consistently degrades*
  connectome performance. Corroborates your linear-beats-deep result. The field's flagship
  negative result.
- **Abraham et al.**, *NeuroImage* 147:736–745 (2017) — N=871, 16 sites, **your exact cohort**.
  Inter-site 66.8% ± 5.4%.

---

## 5. Prior art that removes claims you might have made

### Your ComBat negative result is not a discovery

- **Serra et al.**, *Brain Informatics* 10(1):32 (2023), doi:10.1186/s40708-023-00210-x —
  ABIDE I+II, structural *and* connectivity. *"non-harmonized data and harmonization using only
  the train set showed similar results ... improved performance with the entire dataset may be
  ascribed to data leakage."* **The closest published match to your finding.**
- **Reardon et al.**, *Front Comput Neurosci* 15:762781 (2021) — on ABIDE FC, GLM and ComBat
  *decreased* ASD-vs-TD effect size by up to 11.4%.
- **Bayer et al.**, *NeuroImage* 264:119699 (2022) — on ABIDE (570 subjects), ComBat-style
  correction lost **>90% of the original variance**.
- **Spisak**, *GigaScience* 11:giac082 (2022) — runs on **ABIDE FC, N=866, tangent correlation,
  L2 logistic regression** (essentially your cohort and pipeline). ComBat leaves significant
  residual centre bias: R² = 0.05, **P = 0.009**.
- **Nieto et al.**, *Neurocomputing* 680:133146 (2026), arXiv:2410.19643 — the mechanism:
  *"ComBat may eliminate variance related to the target in such cases of site-target
  dependence."* Frames it as a dilemma with no leakage-free classical escape.
- **Nygaard, Rødland & Hovig**, *Biostatistics* 17(1):29–39 (2016) — the mirror image: with
  the group covariate *in* the model and an unbalanced design, ComBat inflates F-statistics.

### "Site is decodable from FC" is not a finding

- **Serra 2023** reports site discrimination on ABIDE connectivity at **AUC 0.98–0.99** (run on
  TD subjects only so diagnosis cannot leak in — a design detail worth copying).
- Site-prediction-as-efficacy-metric is canonical: Wachinger et al., *Med Image Anal* 67:101879
  (2021); Chen et al., *HBM* 43(4):1179–1195 (2022) (CovBat); Marzi et al., *Sci Data* 11:115 (2024).

### What *does* remain novel from your site analysis

*(measured)* The variance decomposition: between-site share of diagnosis variance = **1.56%**;
the 20-dim site-mean subspace carries **36.9%** of total feature variance but only **0.77%** of
the diagnosis weight norm; projecting out all 19 site directions costs only **0.749 → 0.732**.

**Site dominates variance while being nearly orthogonal to diagnosis** — so ComBat's damage is
estimation variance, not signal removal. No paper reports a PC1/site decomposition for ABIDE
FC. The *number* appears novel; the *conclusion* is published (above).

Also apparently novel: **no ABIDE paper publishes a full per-site AUROC table.** The field
reports pooled LOSO AUC (0.72–0.80) or fold SD. Bootstrap yours before publishing — several
sites have n ≈ 20.

### Genuinely unclaimed in neuroimaging (but predicted to fail)

IRM, V-REx, GroupDRO, Fish and IGA have **never been applied to any neuroimaging data** —
0 hits across OpenAlex, Europe PMC, PubMed and arXiv full-text. Site-as-environment with ~20
environments is a *better* fit than DomainBed's typical 2–4. But every piece of theory and
every clinical benchmark predicts they will not beat ERM at n≈871:

- Gulrajani & Lopez-Paz, ICLR 2021 — https://arxiv.org/abs/2007.01434 — no algorithm beats ERM
  by more than one point under matched conditions.
- Zhang et al., CHIL 2021, doi:10.1145/3450439.3451878 — ClinicalDG implements
  `ERMID ERM IRM VREx RVP IGA CORAL MLDG GroupDRO`; DG does not consistently beat ERM on real
  medical imaging. **Best citation for "we expected DG to fail and it did."**
- Rosenfeld, Ravikumar & Risteski, ICLR 2021 — https://arxiv.org/abs/2010.05761 — IRM "can fail
  catastrophically" precisely in the regime it targets.
- Guo et al., *Sci Rep* 12:2726 (2022) — DG and UDA failed to beat ERM on MIMIC-IV.

Frame as a **pre-registered negative result**, not a hoped-for win.

---

## 6. Killed — do not spend time here

| Candidate | Why it fails |
|---|---|
| Feature-side Bayes-error estimation | Benchmarks need ≥1000 samples/class at d=1, 2500/class at d=4. You have **540/class at d=19,900**. In high dimension the MST estimator is biased toward *spurious separation* — would report a falsely high ceiling. |
| Menon-style noise-rate estimation from η̂ extrema | *(measured)* Cross-validated η̂ spans [0.000, 0.999] at AUROC 0.721, implying noise rate ≈ 0. The extremum estimator is destroyed by tail outliers. |
| Domain-adaptation lower bounds (Zhao et al., ICML 2019) | *(measured)* Site ASD prevalence spans 0.371–0.598 → worst-case joint-error bound **0.0054** against observed joint error ≈ 0.60. Two orders of magnitude too weak. |
| Label-shift / prior correction | *(measured)* Tested and it **hurts**: Brier 0.248 → 0.268 (p=0.017), NLL 0.821 → 0.971 (p=0.003). Prevalence estimates correlate with truth at r = 0.14. |
| Error-subgroup / slice discovery | *(measured)* Errors unpredictable from phenotype, QC and site — error-prediction AUROC **0.557**, site alone **0.503**. |
| Heterogeneity / subtyping / mixture-of-experts | Comprehensively precedented: Katuwal et al. 2016 (subgroup-specific ABIDE classifiers); HYDRA run on ABIDE twice (Liu 2022, Wang 2025); normative modelling owned by Zabihi/Marquand. ~378-paper space. |
| Riemannian conformal weights | Unclaimed but worthless — tangent embedding makes it ordinary feature engineering, density ratio unidentifiable at ~50 subjects/site, estimated weights carry no theorem. |
| PU learning on controls | *(measured)* Only **1.1%** of controls reach the ASD 25th percentile on SRS. Too little contamination to model. |
| Normative modelling | Saturated in ASD (Zabihi et al.; Bethlehem et al.). |

---

## 7. Verification checklist — nothing below is banked

1. **Reproduce the scan-length curve with per-T regularisation tuning.** C1 is the headline and
   rests entirely on it. Tune C at every truncation, use shrinkage covariance throughout, so the
   effect cannot be dismissed as estimator conditioning. Run the subject-axis sweep within NYU
   (n=172) separately.
2. **Resolve a genuine disagreement between two agents on per-site variation.** One found the
   spread indistinguishable from noise (I² = 20.2%, Cochran Q p = 0.20, simulated-vs-observed
   range P = 0.56, 95% PI for a new site [0.678, 0.858]). The other found per-site AUROC tracking
   scan length at ρ = 0.46, p = 0.041. Both can hold — a weak systematic component inside
   mostly-noise — but the paper cannot assert both without reconciling them. **Note the
   meta-analysis caveat: LOSO per-site AUROCs share training data and are not independent,
   which typically deflates apparent heterogeneity. Bootstrap or permutation check required.**
3. **Reproduce the instrument-disagreement and subgroup analyses on stored predictions.** All 85
   out-of-fold sets are exactly linkable to participants (see §8), so both are cheap.
4. **Put confidence intervals on every C4 subgroup number.** ~60–70 autistic females;
   ~250/stratum for the IQ calibration swing.
5. **Reproduce the pooled severity correlation, then show it collapses within-diagnosis.**
   Without that contrast, the severity null is dead on arrival.
6. **Read Ingalhalikar et al. 2021 and establish whether ComBat preceded the split.**
7. **Read Schulz et al. 2024 in full before citing.** If they discuss label noise anywhere,
   C2's positioning needs adjusting.
8. **Fix `run_statistics.py:collect()`.** It keys by condition name across all `raw/*.json` read
   alphabetically, so `voting_pheno.json` shadows `voting_func.json`. The 1,032-cohort
   comparisons you think you have are unreachable.

---

## 8. Codebase assets that make this feasible

**The unlock:** all **85 stored out-of-fold prediction sets are exactly linkable to
individual participants.** No subject IDs are stored, but the fold-concatenation order is
deterministic; both cohorts and both split schemes were reconstructed and matched on
`oof_true` *and* `oof_site` simultaneously, 85/85. So participant-level analysis costs zero
model re-runs.

**Phenotype coverage** (strict n=871 / func n=1032):

| Variable | Strict | Func | Usable for |
|---|---|---|---|
| Motion & QC block (`func_mean_fd`, `perc_fd`, `dvars`, `gsr`, `anat_snr` …) | **871/871** | 1031/1032 | C4, motion confound |
| `EYE_STATUS_AT_SCAN` | **871/871** | **1032/1032** | subgroup analysis |
| SEX (727 M / 144 F) | 871/871 | 1032/1032 | C4 |
| FIQ | 811 | 963 | C4 IQ strata |
| ADOS_TOTAL (ASD) | 282 | 350 | C2 |
| ADI-R subscales (ASD) | ~276–277 | ~356 | C2 |
| SRS_RAW_TOTAL | 300 | 326 | C2 cross-check |
| SRS subscales | — | **63 in the entire CSV** | unusable |

**Compute:** Apple M2, 8 cores, 16 GB, running on **CPU not MPS**. ~4 min per static condition
× 10 folds; ~7 min per dynamic condition; ~7 hours for a full pass.

**Hard infeasible:** no 4D NIfTI anywhere (ROI time series only) — no voxel analysis, no
re-preprocessing, no GSR comparison. `abide2/ts/` is empty; ABIDE-II is labels-only.

---

## 9. The fifth, minor contribution

Leakage audit, negative results, benchmarking — kept as methodological hygiene, not claimed as
novelty. One evaluation-theory import worth making there: **Bates, Hastie & Tibshirani**,
arXiv:2104.00673 — standard k-fold intervals undercover by 2–3×.

Note the leakage finding (feature selection outside the split alone taking AUROC 0.756 → 0.987)
lands differently now that two ABIDE papers have been retracted and a third carries an
Expression of Concern. That exclusion decision was made under different assumptions.

---

## 10. Conflicts with the current plan

`publication_plan.md` now contradicts its own evidence:

- **N1** (dynamic-FC trajectories) predicted +0.02–0.04 AUROC. Measured: dispersion −0.026,
  velocity −0.027, both −0.048, fused −0.070. The surviving `mean` block is not statistically
  separable from static tangent (DeLong p = 0.281).
- **N2** (low-rank spectral) — the weight matrix needs **131 components for 90% of spectral
  mass**; r=50 *matches* rather than beats dense (DeLong p = 0.428) while decisions differ
  significantly (McNemar p = 0.032). Keep as parameter efficiency, not a contribution.
- **N3** (site-conditional conformal) — the one existing novelty that survives review, and it
  upgrades cleanly into C3.
- The **0.79–0.82 AUROC target** belongs to exactly the leaderboard framing that C1 and C2
  argue against.
