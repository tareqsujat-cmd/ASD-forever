"""
Build the interpreted results summary, with the figures inline.

Each section states what was compared, what came out, and what it means, in that order.
Interpretations are kept to a few sentences and avoid restating the numbers already in
the table above them. Sections whose inputs are missing are listed as outstanding rather
than dropped, so the document always reflects the actual state of the study.

Usage:
    python make_summary.py                       # -> RESULTS_SUMMARY.md
    python make_summary.py --study results/<run> --out docs/results.md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(".")


def newest(pattern: str):
    hits = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(hits[0]) if hits else None


def jload(p):
    try:
        return json.loads(Path(p).read_text()) if p and Path(p).exists() else None
    except Exception:
        return None


def cload(p):
    try:
        return pd.read_csv(p) if p and Path(p).exists() and Path(p).stat().st_size else None
    except Exception:
        return None


FIGDIR: Path | None = None          # every figure is copied here and referenced from here
_SEQ = [0]


def set_figdir(p: Path):
    """Point collection at a directory and clear what a previous run collected into it.

    Collected copies are numbered in the order the figures are discussed, so the numbers
    shift whenever a figure is added or removed and copies from an earlier run are left
    behind referring to nothing. Those stale copies are a hazard rather than clutter: a
    figure drawn from a superseded model keeps sitting in the directory a reader draws
    figures from. Numbering starts at 01, so anything the scripts themselves write as
    fig00_* is a source and is left alone.
    """
    global FIGDIR
    FIGDIR = Path(p)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for f in list(FIGDIR.glob("fig[0-9][0-9]_*.png")) + \
             list(FIGDIR.glob("fig[0-9][0-9]_*.pdf")):
        if not f.name.startswith("fig00_"):
            f.unlink()
    _SEQ[0] = 0


def fig(src, caption, note="", name=None):
    """Copy a figure into the single figure directory and reference it from there.

    Figures are produced by several scripts into different run directories; collecting
    them here means the document refers to one place and the numbering follows the order
    in which they are discussed.
    """
    import shutil
    src = ROOT / str(src)
    if not src.exists() or FIGDIR is None:
        return []
    _SEQ[0] += 1
    stem = name or src.stem
    for pre in ("fig", "fig0"):
        if stem.startswith(pre) and stem[len(pre):len(pre) + 1].isdigit():
            stem = stem.split("_", 1)[1] if "_" in stem else stem
            break
    target = FIGDIR / f"fig{_SEQ[0]:02d}_{stem}.png"
    try:
        if src.resolve() != target.resolve():
            shutil.copyfile(src, target)
            pdf = src.with_suffix(".pdf")
            if pdf.exists():
                shutil.copyfile(pdf, target.with_suffix(".pdf"))
    except Exception:
        return []
    rel = target.as_posix()
    sep = "" if caption.rstrip().endswith(("?", "!", ".")) else "."
    return [f"![{caption}]({rel})", "", f"*{caption}{sep} {note}*".rstrip(), ""]


def tbl(df, cols, headers, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float) and v != v:      # a value that was not computed
                cells.append("not computed")
            elif isinstance(v, float):
                cells.append(format(v, fmt.get(c, ".3f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description="Interpreted results summary")
    ap.add_argument("--study", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="RESULTS_SUMMARY.md")
    args = ap.parse_args()

    study = Path(args.study) if args.study else newest("results/*_full_study")
    model = Path(args.model) if args.model else newest("results/*_final")
    T = None
    if study:
        for s in ("tables", "ablations"):
            if (study / s).is_dir():
                T = study / s
                break
    F = f"{study}/figures" if study else ""
    M = f"{model}/model/figures" if model else ""
    if study:
        set_figdir(Path(study) / "figures")

    L = ["# Results summary", "",
         "Autism classification from resting-state functional connectivity on ABIDE-I "
         "(871 participants: 403 autism, 468 control, about twenty sites). Everything the "
         "model learns is fitted inside the training data only, and results are given for "
         "two splits: **pooled**, where participants are divided at random, and "
         "**leave-one-site-out**, where a whole site is held back so the model is tested "
         "on a scanner it has never seen.", "",
         "---", "",
         "## Summary of findings", "",
         "**Two findings concern autism.**", "",
         "1. **Detection is weaker in autistic participants with higher cognitive ability.** "
         "Holding the comparison group "
         "fixed at all 468 non-autistic participants and stratifying only the autistic side, "
         "discrimination falls monotonically from 0.873 below full-scale IQ 85 to 0.719 at "
         "IQ 115 and above, under both protocols. The same gradient holds within the autistic "
         "group, where nothing separates the groups to confound it (rho = -0.18 against "
         "full-scale IQ, p = 0.0004, pooled scores), and is absent in controls (+0.05, "
         "p = 0.32) — a null "
         "that makes the association a property of how autism presents in this measurement "
         "rather than a general effect of ability. A connectivity-based screening aid is "
         "therefore least effective in cognitively able autistic people. Section 1.2.", "",
         "2. **A somatomotor difference that strengthens under motion control.** In 256 "
         "within-site pairs matched for head motion at the system level, the group motion "
         "difference is removed (p = 7.6 × 10⁻⁷ before matching, 0.59 after) and significant "
         "system pairs rise from 10 of 36 to 26 of 36; in 280 pairs matched with the wider "
         "regional caliper (p = 0.28 after matching), the somatomotor block moves from "
         "-0.215 to -0.249 and all 21 of its regions remain significant. The reduction runs "
         "against the direction of the confound: mean framewise displacement correlates "
         "positively with mean connection strength (Spearman rho = +0.51 across "
         "participants, +0.45 within site), so motion raises rather than lowers "
         "connectivity in this pipeline. The direction agrees with the somatomotor "
         "involvement reported by Chen et al. (2015) and the somatomotor reductions reported "
         "by Holiga et al. (2019) and Ilioska et al. (2023); here it is obtained in "
         "within-site pairs matched for head motion. Sections 6 and S6.", "",
         "**Two concern what actually determines measured performance on this dataset.**", "",
         "3. **Feature selection outside the training partition is worth 0.231 AUROC, and "
         "the inflation depends on feature granularity.** Selecting 2,000 of 19,900 connections before "
         "rather than inside cross-validation takes AUROC from 0.756 to 0.987 and accuracy "
         "from 0.692 to 0.948 on identical folds; the same selection over 10 to 100 of 196 "
         "regions changes AUROC by 0.006 on average in absolute value (0.001 at 10 regions, "
         "at most 0.012), with inconsistent "
         "sign. This is the largest effect in "
         "the study, so a pipeline that selects connections before the split can report "
         "accuracies well above those attainable without that step. Without any such step "
         "our accuracy is "
         "competitive with those figures — 0.706 (95% CI 0.678-0.733, participant-pooled, "
         "n = 1,032) against "
         "0.700, 0.703 and 0.708 — so matching them does not require the shortcut. "
         "Sections 5 and 1.3.", "",
         "4. **The linear classifiers perform alike, and fixed-configuration tree ensembles "
         "are markedly lower.** In participant-pooled out-of-fold AUROC, penalised logistic "
         "regression reaches 0.757 against random forest 0.611, XGBoost 0.617 and "
         "CatBoost 0.624 (DeLong p from 6.3 × 10⁻¹⁵ to 2.4 × 10⁻¹²); ridge (0.755) and linear SVM "
         "(0.753) are close to it, while shrinkage LDA (0.741) and the rank-20 spectral "
         "model (0.735) are lower. The tree ensembles used fixed configurations rather than "
         "nested tuning, so the gap describes these configurations rather than the best "
         "attainable member of each family. Sections S3 and S1.", "",
         "Approaches evaluated and not adopted are collected under Limitations, and are "
         "reported as limitations on what these data can support rather than as findings.", "",
         "---", ""]

    # ---------------- 1. headline ----------------
    vot = cload(T / "voting.csv") if T else None
    sota = jload(model / "model" / "sota_report.json") if model else None
    ens_sel = jload(study / "selective_ensemble" / "selective_report.json") if study else None
    sel = ens_sel or (jload(model / "model" / "selective" / "selective_report.json")
                      if model else None)
    sel_is_ensemble = ens_sel is not None
    SD = (f"{study}/selective_ensemble" if sel_is_ensemble
          else f"{model}/model/selective")
    L += fig(f"{F}/fig00_pipeline.png", "Pipeline overview",
         "The whole procedure, from the released scans to a decision.")
    S_voting, S_stats, S_protocol, S_repr, S_classifier, S_design, S_leakage, S_weights, S_severity, S_ridge, S_ceiling, S_network, S_somato, S_regionsel, S_embed, S_compare, S_intervals, S_limits, S_panels = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    S_voting += ["## 1. Classification performance and transfer to an unseen site", ""]
    S_voting += fig(f"{F}/figS_performance.png",
                    "Discrimination, per-site transfer, calibration and abstention",
                    "(a) Discrimination with 95% intervals under both protocols. (b) Every "
                    "site with the model trained without it, marker area proportional to n, "
                    "with the random-effects mean and the prediction interval for a site not "
                    "yet seen. (c) Predicted probability against observed frequency. "
                    "(d) What the abstention rule answers, against the target it promises.")
    if vot is not None:
        d = vot.copy()
        d["split"] = ["unseen site" if "[loso]" in c else "pooled" for c in d["condition"]]
        d["configuration"] = [c.replace(" [loso]", "") for c in d["condition"]]
        d["adopted"] = ["yes" if c == "ensemble (probability mean)" else ""
                        for c in d["configuration"]]
        # the rank score is not a probability, so it has no 0.5 threshold and no accuracy
        d["accuracy"] = d["accuracy"].astype(object)
        d.loc[d["configuration"] == "ensemble (rank mean)", "accuracy"] = "—"
        d = d.sort_values(["split", "auroc"], ascending=[True, False])
        S_voting += tbl(d, ["configuration", "split", "auroc", "auroc_std", "accuracy", "adopted"],
                 ["configuration", "split", "AUROC", "sd", "accuracy", "adopted"])
        S_voting += ["The averaged probability is the configuration carried forward. The "
              "alternative, averaging each member's *rank*, places each held-out "
              "participant's score within the distribution of that member's training-set "
              "scores (Methods Section 3.4), so every participant is scored independently "
              "of the others. It is reported for comparison only; under this construction "
              "it scores below the best single description (Section S1). Because the rank "
              "score is not a probability, it has no 0.5 decision threshold, and no "
              "accuracy is reported for it.", ""]
        S_voting += ["**What this means.** Three different ways of describing the same scans were "
              "each given their own classifier and their answers averaged. Under both splits "
              "the averaged probability has the highest AUROC of the connectivity-only "
              "configurations in the table, but its gain over the best single description, "
              "the windowed mean, is small (+0.009 in participant-pooled AUROC) and is not "
              "distinguished from chance (DeLong p = 0.066; Section S1), so it is not "
              "presented as an established improvement. Performance on an unseen site is "
              "close to performance within the cohort, so the combination is not relying on "
              "knowing the scanner.", "",
              "**On the demographic variants.** Age, sex and IQ alone reach an AUROC of "
              "about 0.62, so they do carry some signal, and adding them to the ensemble "
              "raises participant-pooled AUROC by about 0.004 (p = 0.041 unadjusted, "
              "q = 0.11 across the comparisons in Section S1). The gain was not adopted "
              "because it is small; the q-value is given for reference only (Methods Section "
              "8.1). The connectivity "
              "features appear to capture the same information already.", ""]
    if sota:
        by = {r["protocol"]: r for r in sota}
        se = {r["tag"]: r for r in (sel or [])}
        S_voting += ["For reference, the three-atlas model with demographic variables gave the "
          "following under both splits. It is not the primary ensemble, and the abstention "
          "results below are not computed on it. The ROC curve in Section S7 and the "
          "connectome, network-matrix and SHAP figures in Section S5 are drawn from this "
          "model:", "",
              "| split | AUROC | accuracy | sensitivity | specificity |",
              "|---|---|---|---|---|"]
        for p, name in (("pooled", "Pooled"), ("loso", "Leave-one-site-out")):
            m = by.get(p, {}).get("fold_mean", {})
            if m:
                S_voting.append(f"| {name} | {m['auroc']:.3f} | {m['accuracy']:.3f} | "
                         f"{m.get('sensitivity', 0):.3f} | {m.get('specificity', 0):.3f} |")
        S_voting += ["", "**What this means.** Performance barely changes when a whole site is "
              "held back, so the model is not relying on knowing the scanner. Sensitivity "
              "and specificity are close to each other, so it is not simply favouring one "
              "answer.", ""]
    S_panels += fig(f"{M}/roc_pooled.png", "ROC curve",
             "How well the model ranks autism cases above controls, at every possible "
             "cut-off. Further above the diagonal is better.")
    S_panels += fig(f"{F}/probability_distribution_ensemble.png" if sel_is_ensemble
             else f"{F}/fig8_probability_distribution.png",
             "Predicted probabilities by group",
             "The two groups separate but overlap in the middle. Participants in that "
             "overlap are the ones the model is unsure about.")

    # ---------------- 2. is the difference real ----------------
    st = cload(T / "statistics.csv") if T else None
    S_stats += ["## S1. All model comparisons (unadjusted DeLong p-values)", "",
                "p-values are unadjusted, as stated in Methods Section 8.1; "
                "Benjamini-Hochberg q-values across the comparisons in this table are given "
                "for reference.", ""]
    if st is not None:
        from scipy.stats import false_discovery_control
        d = st.copy()
        d["q_bh"] = false_discovery_control(d["delong_p"].to_numpy())
        d["verdict"] = ["yes" if p < 0.05 else "no" for p in d["delong_p"]]
        S_stats += tbl(d, ["comparison", "difference", "delong_p", "q_bh", "verdict"],
                 ["comparison", "difference in AUROC", "p (unadjusted)", "BH q", "p < 0.05"],
                 {"difference": "+.4f", "delong_p": ".4f", "q_bh": ".4f"})
        S_stats += ["**What this means.** A small gap between two numbers may be nothing more "
              "than which participants happened to land in which fold, so each comparison "
              "is tested. The linear model is separated from the fixed-configuration tree "
              "ensembles by a wide margin. The gain from combining the three descriptions is "
              "not separated from chance: under the probability-averaging rule used "
              "throughout, p = 0.066 in the strict cohort and p = 0.055 in the wider cohort, "
              "and the rank-averaging rule, reported for comparison, scores below the best "
              "single description. Several "
              "smaller differences are *not* distinguishable from chance, including the "
              "gain from windowed averaging on its own and the gap between using one atlas "
              "and three — those should not be presented as improvements.", ""]
    else:
        S_stats += ["*Outstanding.*", ""]

    # ---------------- 3. abstention ----------------
    S_protocol += ["## 2. Calibration, and the cost of a guarantee", "",
          ("Computed on the ensemble, so this section describes the same model as the "
           "headline above." if sel_is_ensemble else
           "Computed on the three-atlas model with demographic variables."), "",
          "The figures here are computed over all participants' held-out predictions "
          "pooled together, whereas Section 1 averages each fold's score. The two agree "
          "closely under the pooled split. They differ more under leave-one-site-out, "
          "where averaging folds gives every site equal weight regardless of size while "
          "pooling gives every participant equal weight; the accuracies quoted below are "
          "therefore slightly higher than the per-site averages in Section 1.", ""]
    if sel:
        se = {r["tag"]: r for r in sel}
        S_protocol += ["| split | accuracy, all cases | calibration error | accuracy on the more "
              "confident half* |",
              "|---|---|---|---|"]
        for t, name in (("pooled", "Pooled"), ("loso", "Leave-one-site-out")):
            e = se.get(t, {})
            if e:
                S_protocol.append(f"| {name} | {e['acc_full']:.3f} | {e['ece']:.3f} | "
                         f"{e['acc_at_50pct']:.3f} |")
        S_protocol += ["", "\\* This last column is reported for completeness and should not be read "
              "as an operating point. The half is defined by the median confidence *of the "
              "held-out set itself*, which is a transductive quantity: it cannot be computed "
              "for a single incoming participant; it corresponds to the exploratory "
              "risk–coverage analysis of Methods Section 5.2. The valid version of this analysis — a threshold fixed on "
              "calibration data and applied to participants who took no part in choosing it "
              "— is reported below and gives a substantially less favourable picture.", ""]
        S_protocol += ["**What this means.** The calibration is sound: where the model reports "
              "seventy per cent it is correct about seventy per cent of the time, with an "
              "expected calibration error near 0.04 under both protocols. Whether that "
              "calibration can be converted into a guarantee on the cases the model chooses "
              "to answer is a separate question, addressed next.", ""]
    S_panels += fig(f"{SD}/calibration_pooled.png" if sel_is_ensemble
             else f"{M}/calibration_pooled.png", "Are the stated probabilities honest?",
             "Predicted probability against how often the model was actually right. "
             "Closer to the diagonal is better.")
    S_panels += fig(f"{SD}/coverage_pooled.png", "Accuracy when answering fewer cases",
             "Moving left means answering only the more confident participants.")
    S_protocol += fig(f"{F}/figS_guarantee.png",
                      "A guarantee, and what it is worth when checked honestly",
                      "(a) The error bound each rule reports on the participants that "
                      "calibrated it, against the error those rules actually incur on "
                      "held-out cases. (b) What each rule costs to obtain that bound: the "
                      "share of cases it answers, and the share of held-out splits in which "
                      "it meets its target.")
    S_protocol += ["**Is the guarantee real?** A threshold chosen so that a bound passes, then scored "
          "on the same participants, passes by construction and says nothing about a future "
          "case. Splitting the data — one half to choose the threshold, the other to be "
          "scored with it — is what tests it. Under that test the single cohort-wide "
          "threshold does not hold its target, whereas the per-site variant does but answers "
          "few cases: with the threshold set on the whole sample it answers 170 of 871 "
          "participants (19.5%), 143 of them at a single site, and its mean coverage across "
          "held-out half-splits is 13.7%. It is therefore reported as a measurement rather "
          "than as a deployable configuration.", ""]

    # ---------------- 4. what to feed the model ----------------
    rep = cload(T / "representation.csv") if T else None
    S_repr += ["## S2. Connectivity representations compared", ""]
    if rep is not None:
        S_repr += tbl(rep.sort_values("auroc", ascending=False),
                 ["condition", "auroc", "auroc_std", "accuracy"],
                 ["description", "AUROC", "sd", "accuracy"])
        S_repr += ["**What this means.** The windowed mean reaches a slightly higher AUROC than "
              "the whole-scan estimate (0.767 against 0.756, fold means), but the difference "
              "is not distinguished from chance (p = 0.28; Section S1), and it disappears "
              "when the same extra shrinkage is applied to the whole-scan estimate (0.767 "
              "against 0.769, p = 0.71; Section 4). It therefore reflects the regularisation "
              "applied to the windows rather than a benefit of splitting the scan. Adding how "
              "much connectivity varies during the scan, or how fast it changes, gave lower "
              "AUROCs (0.697 to 0.741 across the six combined conditions, against 0.767 for "
              "the windowed mean alone); these differences were not formally tested and are "
              "reported as a negative finding.", ""]
    S_repr += fig(f"{F}/fig3_representation.png", "Ways of describing connectivity", "")

    # ---------------- 5. classifier ----------------
    clf = cload(T / "classifier.csv") if T else None
    rank = cload(T / "rank.csv") if T else None
    S_classifier += ["## S3. Classifier family, and compressing the weight matrix", ""]
    if clf is not None:
        clf = clf.assign(condition=clf["condition"].str.replace("(r=20)", "(rank 20)",
                                                                regex=False))
        S_classifier += tbl(clf.sort_values("auroc_pooled_oof", ascending=False),
                 ["condition", "auroc_pooled_oof", "auroc", "auroc_std", "accuracy"],
                 ["classifier", "AUROC (participant-pooled)", "AUROC (fold mean)",
                  "sd (folds)", "accuracy (fold mean)"])
        S_classifier += ["Participant-pooled AUROC is computed over all 871 out-of-fold "
              "predictions together; it is the value quoted in the summary and tested with "
              "DeLong's test. Fold mean and sd summarise the ten outer folds.", ""]
        S_classifier += ["**What this means.** The linear and spectral methods land close "
              "together, and the three tree ensembles land far below. Only the logistic model "
              "was tuned; the tree ensembles used fixed configurations, so the contrast "
              "describes these configurations rather than the best attainable member of each "
              "model family.", ""]
    S_classifier += fig(f"{F}/fig7_critical_difference.png", "Ranking with significance",
             "Methods joined by a bar cannot be told apart. The linear group is separated "
             "from the fixed-configuration tree ensembles.")
    if rank is not None:
        S_classifier += tbl(rank.sort_values("auroc", ascending=False),
                 ["condition", "auroc", "auroc_std"], ["components kept", "AUROC", "sd"])
        S_classifier += ["**What this means.** The model can be rewritten using a small number of "
              "whole-brain patterns instead of twenty thousand separate weights. Keeping "
              "fifty patterns leaves the ranking of participants statistically "
              "indistinguishable from the full model, which makes it far easier to "
              "inspect. The individual decisions do shift, however, so the compression "
              "is not entirely free even where the summary score is unchanged. Keeping "
              "twenty is a genuine step down.", ""]
    S_classifier += fig(f"{F}/fig4_spectrum.png", "How much can be compressed",
             "Accuracy against the number of patterns retained.")

    # ---------------- 6. design choices ----------------
    atl = cload(T / "atlas.csv") if T else None
    har = cload(T / "harmonisation.csv") if T else None
    S_design += ["## S4. Atlas, fusion and harmonisation", ""]
    if atl is not None:
        S_design += tbl(atl.sort_values("auroc", ascending=False),
                 ["condition", "auroc", "auroc_std"], ["atlas combination", "AUROC", "sd"])
        S_design += ["**What this means.** Using three different maps of the brain instead of one "
              "did not help; a single map is as good. The three maps apparently capture much "
              "the same information, so averaging them adds little.", ""]
    ai = cload(T / "atlas_investigation.csv") if T else None
    if ai is not None:
        S_design += ["A published meta-analysis reports the opposite ordering — studies using the "
              "AAL atlas performing better than studies using CC200 — so the comparison was "
              "repeated under both connectivity descriptions, holding participants, "
              "preprocessing, classifier and folds fixed. This is a separate five-fold "
              "comparison with its own regularisation sweep, so its values are not "
              "directly comparable with the nested-CV table above; only the ordering "
              "within it should be read:", ""]
        S_design += tbl(ai, ["condition", "regions", "features", "auroc_correlation",
                      "auroc_tangent"],
                 ["parcellation", "regions", "features", "AUROC (correlation)",
                  "AUROC (tangent)"], {"regions": ".0f", "features": ".0f"})
        S_design += ["**What this means.** The ordering here is the reverse of the meta-analytic "
              "one, and it holds under both connectivity descriptions, so it is not an "
              "artefact of the tangent embedding. It is not a size effect either: "
              "Harvard-Oxford has fewer measurements than AAL yet performs better. Two "
              "observations bear on it. CC200, with the most regions, gives the highest AUROC "
              "under both descriptions, although AUROC does not rise monotonically with region "
              "count (Harvard-Oxford, with 111 regions, exceeds the 116-region AAL atlas). And "
              "the 26 cerebellar regions of the full AAL atlas are at chance in this cohort "
              "(tangent AUROC 0.513); removing them raises the tangent AUROC from 0.688 to 0.695. The "
              "meta-analytic result compares whole studies rather than "
              "holding conditions fixed, and that same analysis found smaller studies "
              "reporting higher performance, so atlas choice there may stand in for study "
              "size.", ""]
    if har is not None:
        S_design += tbl(har, ["condition", "auroc", "auroc_std", "accuracy"],
                 ["site correction", "AUROC", "sd", "accuracy"])
        S_design += ["**What this means.** Statistically removing scanner-related differences made "
              "results worse, and the drop is larger than chance. The correction appears to "
              "remove some of the signal along with the scanner effect, so it is not used.", ""]
    S_design += fig(f"{F}/fig2_ablations.png", "Remaining design comparisons", "")

    # ---------------- 7. protocol and sample ----------------
    leak = cload(T / "leakage_audit.csv") if T else None
    qc = cload(T / "qc_cohort.csv") if T else None
    S_leakage += ["## 5. Feature-selection leakage depends on granularity", ""]
    if leak is not None:
        S_leakage += tbl(leak.sort_values("auroc"), ["variant", "auroc", "auroc_std"],
                 ["how the pipeline was fitted", "AUROC", "sd"])
        S_leakage += ["**What this means.** Fitting the tangent reference, or the tangent reference "
              "and the feature scaler, on all participants before splitting changed AUROC by "
              "at most 0.003 (0.756 and 0.753 against 0.756 for the reference protocol), and "
              "choosing the regularisation constant on the test fold (0.758) or adding "
              "acquisition site as a predictor (0.756) changed it by a similar amount. "
              "Selecting the 2,000 connections with an F-test on every participant's diagnosis "
              "before splitting raised AUROC from 0.756 to 0.987, and applying every shortcut "
              "together gave 0.988. Only the step that sees the labels inflates the score, so "
              "figures from pipelines that select connections before cross-validation are not "
              "comparable with estimates in which selection is nested.", ""]
    if qc is not None:
        S_leakage += tbl(qc, ["cohort", "n", "auroc", "auroc_std", "accuracy"],
                 ["sample definition", "n", "AUROC", "sd", "accuracy"],
                 {"n": ".0f"})
        S_leakage += ["**What this means.** Studies differ in how strictly they discard poor-quality "
              "scans, and that choice changes the score. Allowing the lower-rated scans back "
              "in raises performance slightly. Since the excluded scans are mostly those with "
              "head motion or failed processing, and are not evenly split between the two "
              "groups, part of that gain may reflect data quality rather than brain "
              "differences. Both definitions are reported so comparisons with other studies "
              "are like for like.", ""]

    # ---------------- 8. what the model looks at ----------------
    shap = jload(model / "analyses" / "shap" / "shap_report.json") if model else None
    sym = cload(T / "symptom_axes.csv") if T else None
    S_weights += ["## S5. Where the weight map falls", ""]
    S_weights += fig(f"{model}/analyses/brain/connectome.png", "The connections that matter most",
             "Drawn on a see-through brain, viewed from three sides.")
    S_weights += fig(f"{model}/analyses/brain/network_matrix.png", "Grouped into brain systems",
             "The same information summarised over the seven major brain networks.")
    S_weights += ["**What this means.** The weights fall on connections within recognised "
          "functional systems rather than scattering at random, so the weight map has "
          "anatomical structure. No Haufe transform (Haufe et al., 2014) is applied, so this "
          "describes how the "
              "decision is formed and not which measurements carry the group difference. "
              "Which systems, and how strongly, is "
          "answered directly in Section 6 rather than read off this figure: the difference "
          "is largest within the somatomotor system, with the default-mode network also "
          "affected but less strongly. An earlier version of this document attributed the "
          "decision mainly to the default-mode network, on a superseded model and without "
          "the network-level test that now settles it.", ""]
    S_weights += fig(f"{model}/analyses/shap/shap_subject_waterfalls.png",
             "Why one person was classified this way",
             "Three participants: two clear cases and one the model was unsure about.")
    if shap:
        S_weights += [f"**What this means.** The weights say which connections matter on average; "
              "this says which mattered for one person, which is the form an explanation "
              "has to take if it is to accompany an individual result. For the uncertain "
              "participant the evidence points both ways and cancels out, which is "
              "exactly why the model hesitates.", "",
              f"For a linear model this attribution has a closed form — the weight "
              f"multiplied by how far the participant departs from the average — so it "
              f"re-expresses the fitted weights per participant rather than providing "
              f"independent evidence about them. The agreement with the coefficients "
              f"({shap['shap_vs_coef_corr']:.3f}) is therefore expected by construction, "
              "and is reported as a check that the implementation is correct, not as "
              "corroboration of the pattern itself.", ""]
    if sym is not None:
        n_sig = int((sym["q_fdr"] < 0.05).sum())
        S_weights += [f"**Do these patterns track how severe the autism is?** Across "
              f"{len(sym)} tests against standard symptom scales, none survived correction "
              "for multiple testing.", ""]
    S_severity += ["## 3. The classifier score does not track symptom severity", ""]
    S_severity += fig(f"{F}/figS_corrections.png",
                      "The four effects that change when their control is applied",
                      "(a) Severity correlation, pooled against within-group; rows marked "
                      "with a dagger are instruments given only to autistic participants, "
                      "for which the two scopes must coincide. (b) Whole-scan connectivity "
                      "across shrinkage levels, with the windowed representation placed at "
                      "the shrinkage it applies to its own windows. (c) Inflation from "
                      "selecting outside the training split, against the number of "
                      "candidates. (d) Observed accuracy against the interval attainable "
                      "given the diagnostic labels.")
    sev = cload(T / "severity_pooled_vs_within.csv") if T else None
    if sev is not None and len(sev):
        sev1 = sev[sev["protocol"] == sev["protocol"].iloc[0]]
        piv = sev1.pivot_table(index="instrument", columns="scope",
                               values=["rho", "p", "n", "q_bh"]).reset_index()
        piv.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in piv.columns]
        cols = ["instrument", "n_pooled", "rho_pooled", "q_bh_pooled",
                "n_within-autism", "rho_within-autism", "q_bh_within-autism"]
        cols = [c for c in cols if c in piv.columns]
        S_severity += ["Several published studies report that a classifier's score correlates with "
              "symptom severity. Those correlations are computed across both groups. "
              "Non-autistic participants who complete these instruments score at the low end "
              "of the scale (in this cohort 19 have an ADOS total and 137 an SRS total; the "
              "ADI-R was administered only to autistic participants), so any model that "
              "separates the groups produces a pooled correlation largely by construction. "
              "Computing it both ways separates the two readings:", ""]
        S_severity += tbl(piv, cols,
                 ["instrument", "n (pooled)", "rho (pooled)", "q (pooled)",
                  "n (within autism)", "rho (within autism)", "q (within autism)"][:len(cols)],
                 {"n_pooled": ".0f", "n_within-autism": ".0f"})
        S_severity += ["Scores are out-of-fold ensemble probabilities from the pooled validation "
              "protocol; 'pooled' in the column headers refers to the correlation scope (both "
              "diagnostic groups). q is the Benjamini-Hochberg-adjusted p-value within each "
              "scope's six-instrument family.", ""]
        pool = sev1[sev1.scope == "pooled"]; wit = sev1[sev1.scope == "within-autism"]
        S_severity += [f"**What this means.** Pooled across groups, "
              f"{int((pool['q_bh'] < 0.05).sum())} of {len(pool)} associations survive "
              f"Benjamini-Hochberg correction within the six-instrument family. Within the "
              f"autism group, where the floor effect is absent, "
              f"{int((wit['q_bh'] < 0.05).sum())} of {len(wit)} do. The largest pooled value "
              "reproduces the magnitude reported in the literature and then all but vanishes "
              "once controls are removed, which is the signature of a group-separation "
              "artefact rather than a graded measurement. The instruments administered only "
              "to autistic participants show the same correlation and the same uncorrected "
              "p-value in both scopes, as they must, because both scopes contain the same "
              "people; their q-values differ only because each scope is corrected as its own "
              "six-instrument family. This is a check that the comparison is doing what it "
              "claims. The model "
              "distinguishes autistic from non-autistic participants; it carries little "
              "information about how autistic someone is.", ""]
        S_panels += fig(f"{F}/fig_severity_pooled_vs_within.png",
                 "The same correlations computed two ways",
                 "Asterisks mark associations reaching p < 0.05.")

    # ---------------- 9. the data at each stage ----------------
    # ---------------- 9. regularisation, duration, ceiling ----------------
    S_ridge += ["## 4. The windowed advantage is a difference in shrinkage", ""]
    rs = cload((T / "ridge_sweep.csv")) if T else None
    if rs is None and study:
        rs = cload(Path(study) / "ablations" / "ridge_sweep.csv")
    if rs is not None and len(rs):
        rs = rs.assign(condition=rs["condition"].str.replace("ridge=", "r=", regex=False))
        S_ridge += tbl(rs.sort_values(["representation", "ridge"]),
                 ["condition", "representation", "ridge", "auroc", "auroc_std"],
                 ["condition", "representation", "extra shrinkage", "AUROC", "sd"],
                 {"ridge": ".2f"})
        rt = cload(T / "ridge_sweep_tests.csv") if T else None
        if rt is None and study:
            rt = cload(Path(study) / "ablations" / "ridge_sweep_tests.csv")
        if rt is not None and len(rt):
            rt = rt.assign(meaning=rt["meaning"].str.replace("unridged", "unshrunk", regex=False)
                                                .str.replace("ridge alone", "shrinkage alone",
                                                             regex=False))
            S_ridge += tbl(rt, ["meaning", "difference", "delong_p"],
                     ["what is compared", "difference in AUROC", "p"])
        S_ridge += ["**What this means.** A covariance estimated from a short window is far noisier "
              "than one estimated from a whole scan, so the windowed representation applies an "
              "extra shrinkage towards the identity that the whole-scan representation never "
              "receives. Every comparison between the two therefore changes the amount of "
              "regularisation at the same time as the representation. Applying the same "
              "shrinkage to the whole-scan estimate, and changing nothing else, recovers the "
              "difference that was attributed to windowing. The effect is a property of the "
              "estimator, not evidence that connectivity changes during the scan.", ""]
    S_panels += fig(f"{F}/fig_ridge_sweep.png",
             "Whole-scan connectivity across shrinkage levels",
             "The dashed line is the windowed representation at its own shrinkage level.")

    S_ceiling += ["## 7. What the diagnostic label can and cannot explain", "",
                  "*Reported for completeness; not claimed as a contribution.*", ""]
    lc = jload((Path(study) / "raw" / "label_ceiling.json")) if study else None
    if lc:
        n = lc["n_dual_instrument"]
        both, adi, ados = lc["both_positive"], lc["adi_only"], lc["ados_only"]
        po = (both + 0) / n
        pe = ((both + adi) / n) * ((both + ados) / n) + (1 - (both + adi) / n) * (1 - (both + ados) / n)
        kappa = (po - pe) / (1 - pe)
        S_ceiling += [f"Of the autistic participants, {n} have both a complete ADI-R algorithm and an "
              f"ADOS total. The two disagree about whether the participant meets criteria in "
              f"**{lc['instrument_disagreement']*100:.1f}%** of those cases ({adi} on the "
              f"interview only, {ados} on the observation only, {both} on both, and none on "
              f"neither).", ""]
        _cor = {c["protocol"]: c for c in lc.get("corrected", [])}.get("pooled", {})
        _gap = lc.get("ceiling_range_min", float("nan")) - _cor.get("accuracy", float("nan"))
        S_ceiling += ["Because at least one instrument is wrong wherever they disagree, that rate "
              "estimates the noise in the label a classifier is trained against, and with it "
              "an upper bound on attainable accuracy. Under the stated assumptions the bound "
              f"lies between approximately **{lc.get('ceiling_range_min', 0.85):.2f} and "
              f"{lc.get('ceiling_range_max', 0.99):.2f}**. Correcting the observed figures "
              f"for label noise at the central rates moves accuracy from "
              f"{_cor.get('accuracy', float('nan')):.3f} to "
              f"{_cor.get('accuracy_noise_corrected', float('nan')):.3f} and AUROC from "
              f"{_cor.get('auroc', float('nan')):.3f} to "
              f"{_cor.get('auroc_noise_corrected', float('nan')):.3f}, while observed accuracy "
              f"lies {_gap:.2f} below even the lower end of that range. The shortfall is "
              "therefore not explained by uncertainty in the diagnosis alone, although at the "
              "most pessimistic rates the correction is larger, so the size of the remaining "
              "gap depends on assumptions these data cannot settle.", ""]
        S_ceiling += ["A point estimate is not reported, for three reasons. The two instruments are "
              "not independent raters of a common quantity: one queries developmental "
              "history through a caregiver, the other observes current behaviour. Agreement "
              f"beyond chance is in fact slightly negative (Cohen's kappa = {kappa:+.3f}), "
              "which is inconsistent with the two-rater model the correction assumes. And "
              "two instruments applied to a single population do not identify their error "
              "rates whatever the cell counts, because the 2 x 2 table supplies three "
              "independent proportions against five unknowns (Hui & Walter, 1980); here, in "
              "addition, no participant was negative on both instruments. The correlation "
              "between instrument errors is therefore not estimable from these data rather "
              "than merely imprecise.", ""]
        S_ceiling += [f"*{lc['caveat']}*", ""]

    # ---------------- brain systems, and what the difference looks like ----------------
    S_network += ["## 6. Group differences between functional systems", ""]
    S_network += fig(f"{F}/figS_brain.png",
                     "Group differences, and what motion matching does to them",
                     "(a, b) Standardised difference for every pair of systems, on all "
                     "participants and on within-site pairs matched for head motion, on a "
                     "shared colour scale; asterisks mark pairs surviving correction. "
                     "(c) Each somatomotor region's effect on all participants plotted against its "
                     "effect in motion-matched pairs; points "
                     "below the diagonal are stronger once motion is matched. (d) "
                     "Discrimination by full-scale IQ band among autistic participants (below 85, "
                     "85–99, 100–114, 115 and above), each against all 468 non-autistic "
                     "participants, under both protocols.")
    nc = cload(T / "network_contrast.csv") if T else None
    if nc is not None and len(nc):
        top = nc.reindex(nc["effect"].abs().sort_values(ascending=False).index).head(12)
        top = top.assign(system_a=top["system_a"].replace("sensorimotor", "somatomotor"),
                         system_b=top["system_b"].replace("sensorimotor", "somatomotor"))
        S_network += ["There are too many individual connections (19,110) to test one at a time "
              "with useful power, and their ordering was checked only inside the somatomotor "
              "system (Section S6), where the fifty strongest agree across random halves at a "
              "correlation of 0.34. Grouping the regions into the seven cortical systems of "
              "Yeo et al. plus a subcortical and cerebellar class reduces the comparisons to "
              "thirty-six system pairs, each averaging over 21 to 1,462 connections (median "
              "441). Every connection is first adjusted for acquisition site, head "
              "motion, age and sex, and significance comes from permuting diagnosis within "
              "each site. The permutation certifies the p-value on residualised connections "
          "but not the residualisation itself. Within-site pairs matched for head motion "
          "(panels a–c, and Section S6) therefore address the possibility that a difference "
          "reflects autistic participants being scanned elsewhere or moving more; age is "
          "handled by adjustment alone.", ""]
        S_network += tbl(top, ["system_a", "system_b", "n_edges", "effect", "z", "q_fdr"],
                 ["system", "system", "connections", "difference", "z", "q"],
                 {"n_edges": ".0f"})
        n_sig = int((nc["q_fdr"] < 0.05).sum())
        n_neg = int((nc["effect"] < 0).sum())
        S_network += [f"**What this means.** {n_sig} of {len(nc)} system pairs differ after "
              f"correction, and **every one of them is lower in autism** "
              f"({n_neg} of {len(nc)} pairs are negative overall). Nothing is significantly "
              "raised. The largest difference is within the somatomotor system, and the "
              "limbic system is the most widely involved, differing from most others. A "
              "reduction spread across many systems rather than concentrated in one is the "
              "pattern long proposed for autism, recovered here from the same data the "
              "classifier reads, with the acquisition differences that usually confound it "
              "removed.", ""]
    S_panels += fig(f"{F}/fig_networks.png", "The systems, and where they differ",
             "Regions coloured by the system they belong to; the matrix gives the "
             "difference between autistic and non-autistic participants for each pair, "
             "with asterisks on the pairs that survive correction.")

    S_somato += ["## S6. Inside the somatomotor system (exploratory)", ""]
    sm = cload(T / "sensorimotor_regions.csv") if T else None
    smm = cload(T / "sensorimotor_regions_matched.csv") if T else None
    smj = jload((Path(study) / "raw" / "sensorimotor_regions.json")) if study else None
    smmj = jload((Path(study) / "raw" / "sensorimotor_regions_matched.json")) if study else None
    if sm is not None and len(sm):
        S_somato += ["Everything in this section is exploratory: the system was chosen because it "
              "carried the largest difference in Section 6, so the regional ordering inside it is "
              "a description of that result, not an independent test of it.", ""]
        S_somato += tbl(sm.head(10), ["region", "hemisphere", "effect", "z", "q_fdr"],
                 ["region", "hemisphere", "difference", "z", "q"])
        if smj:
            nb = smj.get("null_block_rho_median")
            S_somato += [f"**Does the ordering reproduce?** Splitting the participants in half "
                  f"repeatedly, refitting the nuisance adjustment inside each half and "
                  f"comparing the two orderings gives a correlation of "
                  f"**{smj['split_half_rho']:.2f}** across the twenty-one regions. Measured "
                  f"the same way in the same splits, the ordering of the "
                  f"{'210' if smj.get('split_half_rho_all_edges') else ''} individual "
                  f"connections inside the system reproduces at "
                  f"{smj.get('split_half_rho_all_edges', float('nan')):.2f} and its top fifty "
                  f"at {smj.get('split_half_rho_top50_edges', float('nan')):.2f}. Region-level "
                  f"reporting is therefore preferred on multiplicity — twenty-one comparisons "
                  f"rather than {len(sm) and '19,110'} — and modestly on stability, not "
                  f"because connection-level effects are unreproducible.", ""]
            if nb is not None:
                S_somato += [f"That figure only means something against a comparator of the same "
                      f"size. Drawing {smj.get('n_null_blocks', 100)} random blocks of "
                      f"twenty-one regions from the whole brain and scoring them identically "
                      f"gives a median of **{nb:.2f}** (range "
                      f"{smj['null_block_rho_min']:.2f}–{smj['null_block_rho_max']:.2f}); "
                      f"{smj['null_block_rho_pct_ge'] * 100:.0f}% of them reach this system's "
                      f"value. The ordering is more stable than most equally sized blocks, "
                      f"but not categorically so.", ""]
        S_somato += ["The regions at the top are the postcentral gyrus, the paracentral lobule and "
              "the precentral gyrus — primary somatosensory and motor cortex, on both sides "
              "and to a similar degree. The effect is spread across the system rather than "
              "sitting in one place: every one of the twenty-one regions differs, and the "
              "strongest accounts for only a small share of the total.", ""]
        if smm is not None and len(smm) and smmj:
            npair = smmj.get("n_participants", 0) // 2
            S_somato += [f"**The motion check.** These regions lie along the central sulcus, where "
                  f"head movement writes its largest artefact, and autistic participants "
                  f"move more — so the analysis was repeated on within-site pairs matched on "
                  f"mean framewise displacement. In {npair} matched pairs "
                  f"({smmj.get('n_participants', 0)} participants) the group difference in "
                  f"motion is removed (p = 7.6 × 10⁻⁷ before matching, p = 0.28 after), and "
                  f"**{int((smm['q_fdr'] < 0.05).sum())} of {len(smm)} regions remain "
                  f"significant** with the ordering preserved (rank correlation 0.89 against "
                  f"the unmatched result). The effects are *larger* after matching "
                  f"({sm['effect'].mean():.3f} → {smm['effect'].mean():.3f} on average), as "
                  f"they were at the system level. Residual motion is not a sufficient "
                  f"explanation for this difference; it is not thereby excluded as a "
                  f"contributor.", ""]
            S_somato += tbl(smm.head(5), ["region", "hemisphere", "effect", "z", "q_fdr"],
                     ["region (motion-matched)", "hemisphere", "difference", "z", "q"])
        S_somato += ["*Limitations of this section.* This is a univariate contrast of residualised "
              "connections, computed without reference to the classifier, and licenses no "
              "statement about what the classifier uses. Hemisphere is assigned from the parcel "
              "centroid while the anatomical name comes from the atlas label of greatest "
              "overlap; for parcels straddling the midline the two can disagree. Results are "
              "reported under a single denoising choice.", ""]

    S_regionsel += ["### 5.2 Region-level selection, in-fold against all-data", ""]
    rs = cload(T / "region_selection.csv") if T else None
    rt = cload(T / "region_selection_tests.csv") if T else None
    if rs is not None and len(rs):
        base = rs[rs["selection"] == "none"]
        reduced = rs[rs["selection"] != "none"]
        piv = reduced.pivot_table(index="k", columns="selection", values="auroc").reset_index()
        piv.columns = [str(c) for c in piv.columns]
        cols = ["k"] + [c for c in piv.columns if c != "k"]
        S_regionsel += tbl(piv[cols], cols,
                 ["regions kept", "chosen from everyone", "chosen inside each fold"],
                 {"k": ".0f"})
        if len(base):
            S_regionsel += [f"Using **all {int(base['k'].iloc[0])} regions** gives "
                  f"**{base['auroc'].iloc[0]:.3f}**, higher than any reduced set above.", ""]
        if rt is not None and len(rt):
            worst = rt.sort_values("vs_all_regions").iloc[0]
            S_regionsel += [f"**What this means.** Keeping only the most discriminating regions makes "
                  f"screening worse, and steadily so: the smallest set loses "
                  f"{abs(worst['vs_all_regions']):.3f} of area under the curve against using "
                  f"every region. There is no subset that improves on the whole, which says "
                  f"the difference is spread across the cortex rather than concentrated "
                  f"where it is strongest. The somatomotor result of Section 6 "
                  f"identifies where the effect is largest, not where it lives, and a "
                  f"reduced set of regions is not a route to a better screening test.", ""]
        S_regionsel += ["**A second reading.** The regions were chosen twice: once from every "
              "participant, which uses the answer before measuring it, and once inside each "
              "training partition only. The two differ by 0.006 on average in absolute value "
              "(at most 0.012, at 100 regions), and the difference does not have a consistent "
              "sign, so selection at this granularity gives no consistent advantage. That "
              "contrasts sharply with the same procedure applied to "
              "individual connections, which inflates the score enormously (Section 5). The "
              "risk of a selection step depends on how fine-grained the choice is: ten of 196 "
              "regions is too coarse a decision to fit noise, whereas two thousand of 19,900 "
              "connections is not.", ""]

    S_embed += ["## 14. `embedding_flow` — what the data look like at each stage", ""]
    S_embed += fig(f"{F}/fig_embedding_flow.png",
             "The same participants at successive stages of the pipeline",
             "Every stage is a change of representation, shown along its three "
             "strongest directions. Panel D is built from out-of-fold scores, so no "
             "participant contributed to their own position on that axis; the other "
             "panels are unsupervised.")
    S_embed += ["**What this means.** Four things are visible, and the first is the most "
          "important. The two groups do not form separate clouds. They sit on top of one "
          "another at every stage, which is what a task at this accuracy has to look "
          "like; a figure showing two clean clusters would not be describing these data. "
          "Second, the strongest directions of variation track the scanner, not the "
          "diagnosis — one site occupies its own region of the space. Third, the group "
          "difference is real but small and does not lie along those strongest "
          "directions: it appears only once the decision score is used as an axis. "
          "Fourth, the eight strongest directions together account for a few per cent of "
          "the total variation, so the signal is spread thinly across many dimensions "
          "rather than concentrated in a few. This is the reason a linear model over the "
          "full feature set does better here than methods that compress first.", ""]
    S_embed += ["The trajectories show something separate. Each participant's scan is cut into "
          "six windows, and the six points from one scan stay close together relative to "
          "the distance between participants. Within-scan variation is therefore small "
          "next to between-participant variation, which is consistent with the windowed "
          "average helping while the dispersion and velocity descriptions did not.", ""]

    # ---------------- 9. against published work ----------------
    # Each row is taken from one configuration only, and the model it refers to is
    # named, because pairing a figure from one model with one from another would not
    # describe anything that was actually run.  The last three columns are properties
    # the comparison studies do not report; they are shown because they, rather than
    # accuracy alone, are where this work differs.
    prot = cload(T / "protocol.csv") if T else None
    vot_all = cload(T / "voting.csv") if T else None
    se2 = {r["tag"]: r for r in (sel or [])}
    cross = calib = confid = "not reported"
    if vot_all is not None:
        vp = vot_all.set_index("condition")
        if "ensemble (probability mean)" in vp.index and "ensemble (probability mean) [loso]" in vp.index:
            a_p = vp.loc["ensemble (probability mean)", "auroc_pooled_oof"]
            a_l = vp.loc["ensemble (probability mean) [loso]", "auroc_pooled_oof"]
            cross = f"{a_l:.3f} ({a_l - a_p:+.3f})"
    if se2.get("pooled"):
        calib = f"{se2['pooled']['ece']:.3f}"
        confid = f"{se2['pooled']['acc_at_50pct']:.3f}"

    # participant-pooled accuracy with a Wilson interval for the matched sample, computed
    # from the stored out-of-fold predictions exactly as in compile_results.py
    acc_ci = ""
    vfj = jload(Path(study) / "raw" / "voting_func.json") if study else None
    if vfj:
        import numpy as np
        from exp_intervals import wilson
        e ={r["condition"]: r for r in vfj}.get("ensemble (probability mean)")
        if e:
            yv = np.asarray(e["oof_true"]); pv_ = np.asarray(e["oof_prob"], dtype=float)
            k = int(((pv_ >= 0.5).astype(int) == yv).sum()); wl, wh = wilson(k, len(yv))
            acc_ci = (f" For the matched sample, the participant-pooled accuracy is "
                      f"{k / len(yv):.3f} (95% Wilson interval {wl:.3f}–{wh:.3f}, "
                      f"n = {len(yv):,}).")
    S_compare += ["### 1.3 Placement against published work on ABIDE-I", "",
          "All entries use ABIDE-I resting-state functional connectivity. Sample size and "
          "testing procedure are listed because both change what the number means. Values "
          "for the other studies are as reported by their authors and were not recomputed "
          "here. The matched-sample rows use the same quality-control definition as those "
          "studies, which admits 1,035 participants; three of them have no usable time "
          "series in the release, so 1,032 are analysed.", "",
          "**The accuracy comparison in this table is uninformative and is included only "
          "because its absence would be conspicuous.** None of the comparison studies "
          "reports a confidence interval, the intervals implied by these sample sizes are "
          "roughly +/- 0.03, and the three published values span 0.008. The last two columns "
          "record properties those studies do not report: how much accuracy survives on a "
          "scanner never seen in training, and whether the stated probabilities match "
          "observed frequencies. Those are the columns on which this work differs. AUROC "
          "values for this work are participant-pooled out-of-fold estimates; accuracy is "
          "the fold mean, which agrees with the pooled value to three decimals." + acc_ci, "",
          "| study | accuracy | AUROC | sample | procedure | AUROC on unseen site | calibration error |",
          "|---|---|---|---|---|---|---|",
          "| Heinsfeld et al. 2018 | 0.700 | not reported | 1,035 | 10-fold | not reported | not reported |",
          "| Eslami et al. 2019 (ASD-DiagNet) | 0.703 | not reported | 1,035 | 10-fold | not reported | not reported |",
          "| Almuqhim & Saeed 2021 (ASD-SAENet) | 0.708 | not reported | 1,035 | 10-fold | not reported | not reported |"]

    rows_ours = []
    if vot_all is not None:
        vp = vot_all.set_index("condition")
        if "ensemble (probability mean)" in vp.index:
            b = vp.loc["ensemble (probability mean)"]
            rows_ours.append(("**This work — ensemble**", b["accuracy"], b["auroc_pooled_oof"],
                              "871", "nested 10-fold", cross, calib))
    vf = cload(T / "voting_func.csv") if T else None
    if vf is not None:
        vfp = vf.set_index("condition")
        if "ensemble (probability mean)" in vfp.index:
            b = vfp.loc["ensemble (probability mean)"]
            cf = "not run"
            if "ensemble (probability mean) [loso]" in vfp.index:
                al = vfp.loc["ensemble (probability mean) [loso]", "auroc_pooled_oof"]
                cf = f"{al:.3f} ({al - b['auroc_pooled_oof']:+.3f})"
            # calibration and the confident-half figure for this cohort, when computed
            sf = jload(study / "selective_ensemble_func" / "selective_report.json")
            sfp = {r["tag"]: r for r in (sf or [])}.get("pooled", {})
            ca_f = f"{sfp['ece']:.3f}" if sfp else "—"
            cf_f = f"{sfp['acc_at_50pct']:.3f}" if sfp else "—"
            rows_ours.append(("**This work — ensemble, matched sample**",
                              b["accuracy"], b["auroc_pooled_oof"], "1,032", "nested 10-fold",
                              cf, ca_f))
    if qc is not None and "func" in set(qc["cohort"]):
        f_ = qc.set_index("cohort").loc["func"]
        rows_ours.append(("This work — tangent model, matched sample",
                          f_["accuracy"],
                          f_["auroc_pooled_oof"] if "auroc_pooled_oof" in f_.index else f_["auroc"],
                          f"{int(f_['n']):,}",
                          "nested 10-fold", "—", "—"))
    for nm, acc, auc, n, proc, cs, ca in rows_ours:
        S_compare.append(f"| {nm} | {acc:.3f} | {auc:.3f} | {n} | {proc} | {cs} | {ca} |")

    S_compare += ["", "**What this means.** Accuracy is in the same range as the established "
          "results on this dataset, reached under a stricter procedure in which nothing is "
          "fitted before the data are split. Higher figures reported elsewhere could not be "
          "checked against their sources and are therefore not listed.", "",
          "**Where the difference actually lies.** The accuracy column compares models on "
          "all cases at a fixed cut-off, which is the least of what matters for a screening "
          "instrument. The two columns beside it are the substantive difference: "
          "performance is essentially unchanged on an unseen scanner, and the stated "
          "probabilities can be taken at face value. Abstention was also examined: a "
          "per-site rule held its error target on held-out half-splits but answered only "
          "170 of 871 participants (19.5%), 143 of them at a single site, and the cohort-wide "
          "rule answered none, so abstention is reported as a measurement rather than as a "
          "feature of the model (Section 2).", ""]
    S_panels += fig(f"{F}/per_site_loso.png", "Performance site by site",
             "Each bar is a site the model had never seen during training. Computed from "
             "the adopted ensemble's own stored predictions.")
    ps = cload(T / "per_site_loso.csv") if T else None
    if ps is not None and len(ps):
        ps = ps.sort_values("auroc")
        weak = ps[ps["auroc"] < 0.65]
        lo, hi = ps.iloc[0], ps.iloc[-1]
        chance = int((ps["auroc"] <= 0.5).sum())
        psi = cload(T / "per_site_intervals.csv") if T else None
        inc = psi[psi["auroc_lo"] <= 0.5]["site"].tolist() if psi is not None else []
        ci_txt = (f"The 95% intervals of {len(inc)} sites ({', '.join(inc)}) include 0.5. "
                  if inc else "")
        S_compare += [f"**The main limitation.** Performance is uneven across sites: it ranges "
              f"from {lo['auroc']:.3f} at {lo['site']} to {hi['auroc']:.3f} at "
              f"{hi['site']}. "
              + (f"No site's point estimate falls to chance, but {len(weak)} of {len(ps)} "
                 f"sit below 0.65 ({', '.join(weak['site'])}). " if chance == 0 else
                 f"{chance} of {len(ps)} sit at or below chance. ")
              + ci_txt
              + "Whether that spread reflects real differences between scanners or the "
              "small size of most sites is not settled by the range alone: the median site "
              "contributes about thirty participants, so a wide interval is expected even if "
              "every site performed identically. It is reported as a range, not as evidence "
              "of site-dependent failure.", ""]
    else:
        S_compare += ["**The main limitation.** Performance is uneven across sites, and there is "
              "currently no way to tell in advance which scanners the model will do "
              "poorly on. This is the clearest open problem in the work.", ""]

    # ---------------- 10. outstanding ----------------
    want = ["representation", "voting", "classifier", "rank", "atlas", "fusion",
            "harmonisation", "protocol", "statistics", "leakage_audit", "qc_cohort",
            "symptom_axes"]
    missing = [w for w in want if not (T and (T / f"{w}.csv").exists()
                                       and (T / f"{w}.csv").stat().st_size)]
    S_intervals += ["### 1.2 Intervals, per-site variation, and a prediction interval", ""]
    iv = cload(T / "intervals.csv") if T else None
    ivj = jload((Path(study) / "raw" / "intervals.json")) if study else None
    sg = cload(T / "subgroup_auroc.csv") if T else None
    if iv is not None and len(iv):
        S_intervals += ["Headline figures with intervals. Areas under the curve carry DeLong intervals; "
              "accuracy, sensitivity and specificity carry Wilson intervals. Both condition on "
              "the fitted model and treat dependent out-of-fold predictions as independent, so "
              "they are the narrowest defensible intervals rather than a full accounting of "
              "uncertainty.", ""]
        rowsL = ["| split | AUROC | accuracy | sensitivity | specificity |", "|---|---|---|---|---|"]
        for _, r in iv.iterrows():
            rowsL.append(
                f"| {r['protocol']} | {r['auroc']:.3f} ({r['auroc_lo']:.3f}-{r['auroc_hi']:.3f}) "
                f"| {r['accuracy']:.3f} ({r['accuracy_lo']:.3f}-{r['accuracy_hi']:.3f}) "
                f"| {r['sensitivity']:.3f} ({r['sensitivity_lo']:.3f}-{r['sensitivity_hi']:.3f}) "
                f"| {r['specificity']:.3f} ({r['specificity_lo']:.3f}-{r['specificity_hi']:.3f}) |")
        S_intervals += rowsL + [""]
    if ivj and ivj.get("meta_analysis"):
        m = ivj["meta_analysis"]
        S_intervals += [f"Combining the {m['k_sites']} per-site estimates under a random-effects model "
              f"gives a mean of **{m['random_effects_mean']:.3f}**, with between-site "
              f"heterogeneity not distinguishable from chance (Q = {m['Q']:.1f} on {m['df']} df, "
              f"p = {m['Q_p']:.2f}; I-squared = {m['I2']:.0f}%). The 95% prediction interval for "
              f"a site not yet seen is **{m['prediction_interval_lo']:.3f} to "
              f"{m['prediction_interval_hi']:.3f}** — the quantity a reader deciding whether to "
              f"use this instrument at their own centre actually needs, and one no comparison "
              f"study reports. Because leave-one-site-out estimates share training data, the "
              f"heterogeneity is if anything understated.", ""]
    if sg is not None and len(sg):
        band = sg[sg["variable"].str.contains("FIQ band")]
        if len(band):
            S_intervals += ["**Detection is weaker at higher cognitive ability.** Holding the "
                  "comparison group fixed at every non-autistic participant and stratifying "
                  "only the autistic side, each stratum is scored against the same reference, "
                  "so differences between strata do not arise from changes in the "
                  "comparison group. Discrimination declines monotonically under both "
                  "protocols. The autistic strata still differ from the reference group in "
                  "full-scale IQ; that separation is addressed by the IQ-matched analysis "
                  "that follows.", ""]
            pv = band.pivot_table(index="group", columns="protocol", values="auroc")
            nn = band[band.protocol == "pooled"].set_index("group")["n"]
            order = ["FIQ < 85", "FIQ 85-99", "FIQ 100-114", "FIQ >= 115"]
            S_intervals += ["| autistic participants | n | pooled | site withheld |",
                            "|---|---|---|---|"]
            for gname in order:
                if gname in pv.index:
                    S_intervals.append(
                        f"| {gname} | {int(nn.get(gname, 0))} | "
                        f"{pv.loc[gname].get('pooled', float('nan')):.3f} | "
                        f"{pv.loc[gname].get('loso', float('nan')):.3f} |")
            S_intervals += [""]
        QA, QM = "FIQ (site-centred)", "FIQ (site-centred, IQ-matched pairs)"
        qa, qm = sg[sg["variable"] == QA], sg[sg["variable"] == QM]
        if len(qa) and len(qm):
            def cell(dd, pr, g, col="auroc"):
                v = dd[(dd["protocol"] == pr) & (dd["group"] == g)][col]
                return v.iloc[0] if len(v) else float("nan")
            Q = ("Q1", "Q2", "Q3", "Q4")
            npair = {pr: int(qm[qm["protocol"] == pr]["n"].sum()) // 2 for pr in ("pooled", "loso")}
            up = {pr: [cell(qm, pr, g) for g in Q[1:]] for pr in ("pooled", "loso")}
            S_intervals += [f"**Stratifying both groups by IQ.** Dividing all participants into "
                  f"quartiles of full-scale IQ centred within site reproduces the gradient, from "
                  f"{cell(qa, 'pooled', 'Q1'):.3f} in the lowest quartile to "
                  f"{cell(qa, 'pooled', 'Q4'):.3f} in the highest (pooled). Because autistic "
                  "participants have lower full-scale IQ, the lower quartiles are also those in "
                  "which the groups differ most on IQ. Matching each autistic participant to a "
                  "non-autistic participant at the same site whose full-scale IQ differs by at "
                  f"most five points ({npair['pooled']} pairs pooled, {npair['loso']} with the "
                  "site withheld; Methods Section 6.9) removes that separation. Discrimination "
                  f"then remains highest in the lowest quartile ({cell(qm, 'pooled', 'Q1'):.3f} "
                  f"pooled, {cell(qm, 'loso', 'Q1'):.3f} site withheld) but no longer declines "
                  f"across the upper three ({min(up['pooled']):.3f}–{max(up['pooled']):.3f} "
                  f"pooled; {min(up['loso']):.3f}–{max(up['loso']):.3f} site withheld). Part "
                  "of the full-range gradient therefore reflects the group difference in IQ; "
                  "the within-group association below involves no between-group comparison "
                  "and is the direct evidence.", ""]
            S_intervals += ["| site-centred FIQ quartile | n (all) | all, pooled | all, site "
                            "withheld | n matched (pooled / site withheld) | matched, pooled | "
                            "matched, site withheld |", "|---|---|---|---|---|---|---|"]
            for g in Q:
                S_intervals.append(
                    f"| {g} | {int(cell(qa, 'pooled', g, 'n'))} | {cell(qa, 'pooled', g):.3f} | "
                    f"{cell(qa, 'loso', g):.3f} | {int(cell(qm, 'pooled', g, 'n'))} / "
                    f"{int(cell(qm, 'loso', g, 'n'))} | {cell(qm, 'pooled', g):.3f} | "
                    f"{cell(qm, 'loso', g):.3f} |")
            S_intervals += [""]
        rho = sg[sg["variable"].str.startswith("score vs")]
        if len(rho):
            S_intervals += ["The same gradient appears within the autistic group, where there "
                  "is no group separation to confound it, and is absent in non-autistic "
                  "participants. Scores are residualised on site, age and head motion before "
                  "Spearman correlation; values are given for both validation protocols, as "
                  "rho (p).", ""]
            S_intervals += ["| ability measure | autistic, pooled | autistic, site withheld | "
                            "non-autistic, pooled | non-autistic, site withheld |",
                            "|---|---|---|---|---|"]
            for meas in ("FIQ", "VIQ", "PIQ"):
                cells = []
                for grp in ("autistic", "control"):
                    for proto in ("pooled", "loso"):
                        d = rho[rho["variable"].str.contains(f"vs {meas},")
                                & (rho.protocol == proto) & (rho.group == grp)]
                        if len(d):
                            r0 = d.iloc[0]
                            pv = r0["p"] if "p" in r0.index else float("nan")
                            cells.append(f"{r0['auroc']:+.3f} ({pv:.4f}), n = {int(r0['n'])}")
                        else:
                            cells.append("—")
                S_intervals.append(f"| {meas} | " + " | ".join(cells) + " |")
            S_intervals += ["", "The control-group null is what makes this interpretable: the "
                  "association is a property of how autism presents in the connectivity "
                  "measurement, not a general relationship between ability and the score. "
                  "Nielsen et al. (2013) reported that classifier scores covaried with verbal "
                  "IQ on this dataset, so the association is anticipated; what is added here "
                  "is the control-group comparison and the graded loss of detectability.", ""]
        sx = sg[sg["variable"] == "sex"]
        if len(sx):
            S_intervals += ["**Discrimination by sex shows no deficit.** Area under the curve is somewhat "
                  "higher in female participants — " +
                  "; ".join(f"{r['group']} {r['auroc']:.3f} ({r['auroc_lo']:.3f}-"
                            f"{r['auroc_hi']:.3f}, n = {int(r['n'])}) under {r['protocol']}"
                            for _, r in sx.iterrows()) +
                  " — but with 144 female participants the intervals overlap and no difference "
                  "is claimed.", ""]
        S_intervals += ["*Linkage note.* Stored out-of-fold predictions are written in fold order and "
              "carried no participant identifier, so the fold order was replayed and the "
              "reconstruction verified element-wise against both the stored labels and the "
              "stored site codes for all 871 participants before any phenotype was joined. "
              "Sentinel-coded missing values are excluded rather than imputed. Subsequent runs "
              "record identifiers directly.", ""]

    S_limits += ["## Limitations", ""]
    if missing:
        S_limits += ["Still outstanding: " + ", ".join(missing), ""]
    S_limits += ["**Scope of the cohort.** Participants range from 6.5 to 58 years with a median "
          "of 14.8 and only one below seven, so nothing here bears on early identification. "
          "The sample is 727 male to 144 female, the comparison group is typically "
          "developing rather than clinically referred, and every label is a diagnosis "
          "obtained before the scan; the results therefore describe discrimination within an "
          "aggregated research cohort, not performance in a differential-diagnostic "
          "setting.", ""]
    S_limits += ["**Denoising and interpretation.** All results are obtained under a single "
          "preprocessing strategy, without global signal regression, so the direction of any "
          "group difference is conditional on that choice. Group differences are univariate "
          "contrasts of residualised connections computed without reference to the "
          "classifier, and therefore support statements about how the groups differ but "
          "none about which measurements the classifier uses.", ""]
    S_limits += ["**Abstention has limited reach on these data.** The per-site rule holds its "
          "target but answers few cases — 170 of 871 (19.5%) with the threshold set on the "
          "whole sample, 143 of them at a single site, and 13.7% on average across held-out "
          "half-splits — and seventeen of twenty sites answer no one. This characterises the "
          "cohort rather than the method.", ""]
    S_limits += ["**Approaches evaluated and not adopted.** Site harmonisation showed no evidence "
          "of benefit; within-scan variability and rate of change lowered accuracy; three "
          "atlases performed no better than one; a rank constraint on the weight matrix "
          "matched the dense model without improving on it, and the premise motivating it "
          "does not hold; demographic variables added about 0.004 AUROC; restricting the "
          "model to the most discriminating regions lowered performance at every list "
          "length; the extracted components do not track symptom severity within the autism "
          "group; and neither the scan-duration nor the sample-size axis shows evidence of "
          "saturation, so the two cannot be separated on these data. Each is reported "
          "because a negative result on a public dataset bounds what the next study can "
          "expect.", ""]

    L += ["# Main findings", ""]
    L += S_voting + S_intervals + S_compare
    L += S_protocol + S_severity + S_ridge
    L += S_leakage + S_regionsel + S_network + S_ceiling
    L += ["", "---", "", "# Supplementary results", "",
          "These sections record the comparisons behind the design choices made above. None "
          "carries a claim of its own; each is included so that a choice reported in the "
          "main results rests on evidence rather than on preference.", ""]
    L += S_stats + S_repr + S_classifier + S_design + S_weights + S_somato
    L += ["## S7. Individual panels", "",
          "Single-panel versions of figures that appear merged in the main results, kept "
          "because a reader checking one number should not have to read a composite.", ""]
    L += S_panels
    L += ["", "---", ""]
    L += S_limits

    Path(args.out).write_text("\n".join(L))
    n_fig = sum(1 for x in L if x.startswith("!["))
    print(f"wrote {args.out}  ({n_fig} figures)")
    if missing:
        print("outstanding:", ", ".join(missing))


if __name__ == "__main__":
    main()
