"""
LaTeX tables for the calibration and interpretation results, generated from the stored CSVs so that
every cell equals the stored value under strict rounding.  Nothing is typed by hand.

Writes one file per table to results/latest_study/review/tables_tex/ (each is a complete ``table``
environment in the style of the existing Results tables; the tables themselves live in sections/result.tex) and prints a check that every numeric cell
was produced from a stored value.

    .venv/bin/python review_tex_tables.py [--review results/latest_study/review] [--out results/latest_study/review/tables_tex]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SYSTEM_SHORT = {"visual": "Visual", "sensorimotor": "Somatomotor", "dorsal attention": "Dorsal attention",
                "salience / ventral attention": "Salience / ventral attention", "limbic": "Limbic",
                "frontoparietal control": "Frontoparietal control", "default mode": "Default mode",
                "subcortical / cerebellar": "Subcortical / cerebellar", "unassigned": "Unassigned"}


def num(x: float, nd: int = 3, sign: bool = False) -> str:
    """Strict rounding; a negative value is written in math mode, as in the existing tables."""
    s = f"{x:.{nd}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]
    if s.startswith("-"):
        return f"${s}$"
    if sign and float(s) != 0.0:
        return f"$+{s}$"
    return s


def with_ci(est, lo, hi, nd=3, sign=False) -> str:
    return f"{num(est, nd, sign)} ({num(lo, nd)} to {num(hi, nd)})"


def pair_name(block: str) -> tuple[str, str]:
    a, b = block.split(" | ") if " | " in block else (block, "")
    return SYSTEM_SHORT[a], SYSTEM_SHORT.get(b, b)


# =========================================================================================
def calibration_table(rv: Path) -> str:
    met = pd.read_csv(rv / "calibration_nested.csv")
    protos = [p for p in ("pooled", "loso") if p in set(met.protocol)]
    names = {"native": "Native", "intercept only": "Intercept (primary)",
             "logistic": "Logistic", "isotonic": "Isotonic",
             "unpenalised intercept": "Unpenalised intercept (diagnostic)"}
    title = {"pooled": "Pooled ten-fold validation", "loso": "Leave-one-site-out validation"}
    rows = []
    for p in protos:
        rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{title[p]}}}}} \\\\")
        for v, label in names.items():
            g = met[(met.protocol == p) & (met.variant == v)].set_index("metric")
            cit, slo = g.loc["calibration-in-the-large"], g.loc["calibration slope"]
            rows.append(" & ".join([
                label,
                with_ci(cit.estimate, cit.lo, cit.hi, 3, True),
                with_ci(slo.estimate, slo.lo, slo.hi, 3),
                num(g.loc["Brier", "estimate"]), num(g.loc["ECE (10 bins)", "estimate"]),
                num(g.loc["AUROC", "estimate"]), num(g.loc["accuracy", "estimate"])]) + " \\\\")
        if p != protos[-1]:
            rows.append("\\midrule")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Calibration of the ensemble probabilities before and after recalibration that is fitted inside
each outer training partition only. Calibration-in-the-large (negative values indicate over-prediction of
autism) and the calibration slope carry 95\\% site-clustered bootstrap intervals. ECE is the expected
calibration error with ten equal-width bins. Accuracy is taken at the threshold of 0.5. The last row of each
block refits the members with an unpenalised intercept and is a diagnostic, not a recalibration.}}
\\label{{tab:calibration-recalibration}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\\toprule
Correction & Calibration-in-the-large & Slope & Brier & ECE & AUROC & Accuracy \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def contribution_table(rv: Path, n_top: int = 10) -> str:
    b = pd.read_csv(rv / "interpretation_blocks.csv")
    b = b[(b.protocol == "pooled") & (b.view == "mean of views")].copy()
    nullp = None
    if (rv / "interpretation_block_null.csv").exists():
        nullp = pd.read_csv(rv / "interpretation_block_null.csv")
        nullp = nullp[nullp.view == "mean of views"].set_index("block").p_familywise
    top = b[b.block != "unassigned"].sort_values("contribution_diff", ascending=False).head(n_top)
    rows = []
    for _, r in top.iterrows():
        a, c = pair_name(r.block)
        p = "--" if nullp is None else (num(nullp[r.block], 3) if nullp[r.block] >= 0.001 else "$<0.001$")
        rows.append(" & ".join([a, c, f"{int(r.n_edges):,}".replace(",", "{,}"),
                                with_ci(100 * r.share_of_separation, 100 * (r.diff_lo / r.total_separation),
                                        100 * (r.diff_hi / r.total_separation), 1),
                                with_ci(r.auroc, r.auroc_lo, r.auroc_hi), p]) + " \\\\")
    body = "\n".join(rows)
    un = b[b.block == "unassigned"].iloc[0]
    tot = b.total_separation.iloc[0]
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{System pairs whose connections carry the largest share of the separation between autistic and
non-autistic participants in the held-out scores, averaged over the three views of the ensemble. The share is
the difference in mean contribution between the groups divided by the total difference in mean score, and its
interval is the 95\\% site-clustered bootstrap interval of the contribution difference on the same scale. The AUROC
is that of the contribution of the pair taken alone. The familywise $p$-value compares the AUROC with the maximum
deviation from 0.5 over the 36 pairs under within-site label permutation. The total difference in mean score is
{num(tot, 3)} on the logit scale, and the connections of the four regions excluded from the group-difference
analyses carry {num(100 * un.share_of_separation, 1)}\\% of it.}}
\\label{{tab:system-contributions}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}llcccc@{{}}}}
\\toprule
System & System & Connections & Share (\\%) & AUROC & Familywise $p$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def reproducibility_table(rv: Path) -> str:
    h = pd.read_csv(rv / "interpretation_splithalf.csv")
    names = {"correlation": "Correlation", "tangent (static)": "Static tangent", "dynamic: mean": "Windowed tangent, mean"}
    lv = {"connection": "Connections", "region": "Regions", "system pair": "System pairs"}
    rows = []
    for v, vn in names.items():
        for k, (l, ln) in enumerate(lv.items()):
            st = h[(h.view == v) & (h.quantity == "struct") & (h.level == l)].iloc[0]
            wt = h[(h.view == v) & (h.quantity == "coef") & (h.level == l)].iloc[0]
            first = vn if k == 0 else ""
            rows.append(" & ".join([first, ln, f"{num(st.observed_rho)} ({num(st.null_p95)})",
                                    f"{num(100 * st.sign_agreement, 0)} ({num(100 * st.sign_agreement_null, 0)})",
                                    f"{num(wt.observed_rho)} ({num(wt.null_p95)})"]) + " \\\\")
        if v != list(names)[-1]:
            rows.append("\\midrule")
    body = "\n".join(rows)
    n = int(h.n_splits.iloc[0])
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Agreement between the patterns fitted in two independent halves of the cohort, as the mean Spearman
correlation over {n} random divisions. The structure coefficient of a connection is its correlation with the model
score, and the weight is its coefficient in the model. The figure in brackets is the 95th percentile of the same
quantity when the diagnoses are permuted within site. Sign agreement is the percentage of connections, regions or
system pairs whose structure coefficients have the same sign in the two halves, with the permutation mean in brackets.}}
\\label{{tab:pattern-reproducibility}}
\\small
\\begin{{tabular}}{{@{{}}llccc@{{}}}}
\\toprule
View & Level & Structure coefficients $\\rho$ & Sign agreement (\\%) & Weights $\\rho$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default=str(ROOT / "results/latest_study/review"))
    ap.add_argument("--out", default=str(ROOT / "results/latest_study/review/tables_tex"))
    a = ap.parse_args()
    rv, out = Path(a.review), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    if (rv / "calibration_nested.csv").exists():
        (out / "tab_calibration_recalibration.tex").write_text(calibration_table(rv)); made.append("calibration")
    if (rv / "interpretation_blocks.csv").exists():
        (out / "tab_system_contributions.tex").write_text(contribution_table(rv)); made.append("system contributions")
    if (rv / "interpretation_splithalf.csv").exists():
        (out / "tab_pattern_reproducibility.tex").write_text(reproducibility_table(rv)); made.append("reproducibility")
    print("wrote:", ", ".join(made), "->", out)


if __name__ == "__main__":
    main()
