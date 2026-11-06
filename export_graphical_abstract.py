"""
Graphical abstract (version 3): check the Design artboard against the study tables, render the
journal files, and install them.

The artboard is a Claude Design canvas (https://claude.ai/artifact/LwLTExeiVHYkQKaqA2doUr); its source is
kept in results/<study>/figures/graphical_abstract_design/Main.dc.html (2000 x 800 px, 2.5:1, the ratio
of the ScienceDirect window and of Elsevier's 1328 x 531 px minimum).  Every number printed on it, and the
pixel position of every dot, is recomputed here from the study tables and must appear in the artboard
source; the script stops if one does not.

    .venv/bin/python export_graphical_abstract.py [--study results/latest_study] [--no-copy]

Outputs (study figures folder): fig_graphical_abstract.png (4000 x 1600 px), .pdf (vector, fonts embedded),
.tiff (LZW, 600 dpi) and .sources.json (values, tables of origin, alt text); png and pdf are copied to
ASD-journal-I/figures/.  Rendering uses headless Google Chrome (no network, nothing installed).
Fonts: Arial (one of the four Elsevier asks for); text is 32 px or more on the 2000 px artboard, which is
5.9 pt at the 13 cm width ScienceDirect uses.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
STEM = "fig_graphical_abstract"
W, H, DPR = 2000, 800, 2
MINUS = "−"
JOURNAL_FIG = ROOT / "ASD-journal-I" / "figures"
X0, PLOT_W = 200.0, 232.0                      # plot area of every mini chart in the artboard (px)


def sgn(v: float, nd: int) -> str:
    s = f"{abs(v):.{nd}f}"
    return MINUS + s if v < 0 and float(s) != 0 else s


def scale(lo: float, hi: float):
    return lambda a: X0 + (float(a) - lo) * PLOT_W / (hi - lo)


def evidence(study: Path) -> dict:
    import viz_graphical_abstract as old                      # table readers of the previous version
    d, _ = old.evidence(study)
    sd = pd.read_csv(study / "review" / "interpretation_splithalf_sitedisjoint.csv")
    sd = sd[(sd.variant == "raw") & (sd.quantity == "struct")]
    order = ["correlation", "tangent (static)", "dynamic: mean"]            # amber, green, pink in the artboard
    for lvl, key in (("system pair", "pair"), ("connection", "conn")):
        s = sd[sd.level == lvl].set_index("view").loc[order]
        d[f"sd_{key}"], d[f"sd_{key}_p95"] = s.observed_rho.to_numpy(float), float(s.null_p95.max())
    return d


def expected_text(d: dict) -> dict[str, str]:
    """Strings the artboard must contain (displayed numbers)."""
    return {
        "participants": f"{d['n']} participants",
        "auroc ten-fold": f"{d['auroc']:.3f}",
        "auroc loso": f"{d['auroc_loso']:.3f}",
        "iq": f"AUROC {d['iq'][0]:.3f} → {d['iq'][3]:.3f}",
        "calibration": f"Offset {sgn(d['citl_nat_pooled'][0], 3)} → {sgn(d['citl_rec_pooled'][0], 3)}",
        "linear": f"Linear {d['lin'].min():.3f}–{d['lin'].max():.3f}",
        "trees": f"trees {d['tree'].min():.3f}–{d['tree'].max():.3f}",
        "site-disjoint": f"ρ {d['sd_pair'].min():.2f}–{d['sd_pair'].max():.2f}",
        "chance": f"chance ≤ {d['sd_pair_p95']:.2f}",
        "somatomotor aria": f"all participants mean {sgn(d['sm_all'].mean(), 3).replace(MINUS, 'minus ')}",
        "somatomotor aria 2": f"motion-matched pairs mean {sgn(d['sm_matched'].mean(), 3).replace(MINUS, 'minus ')}",
    }


def expected_positions(d: dict) -> dict[str, list[float]]:
    f1, f2 = scale(-0.35, 0.15), scale(0.0, 0.8)
    f3, f4, f5 = scale(-0.35, -0.05), scale(0.5, 1.0), scale(0.55, 0.80)
    est = lambda k: d[f"citl_{k}_pooled"]
    return {
        "iq dots": [f4(v) for v in d["iq"]],
        "somatomotor all (min, mean, max)": [f3(d["sm_all"].min()), f3(d["sm_all"].mean()), f3(d["sm_all"].max())],
        "somatomotor matched (min, mean, max)": [f3(d["sm_matched"].min()), f3(d["sm_matched"].mean()), f3(d["sm_matched"].max())],
        "linear dots": [f5(v) for v in d["lin"]],
        "tree dots": [f5(v) for v in d["tree"]],
        "native (lo, est, hi)": [f1(est("nat")[1]), f1(est("nat")[0]), f1(est("nat")[2])],
        "recalibrated (lo, square left edge = est - 9, hi)": [f1(est("rec")[1]), f1(est("rec")[0]) - 9.0, f1(est("rec")[2])],
        "zero line": [f1(0.0)],
        "system-pair dots": [f2(v) for v in d["sd_pair"]],
        "connection dots": [f2(v) for v in d["sd_conn"]],
        "chance bands (widths)": [f2(d["sd_pair_p95"]) - X0, f2(d["sd_conn_p95"]) - X0],
    }


def check(html: str, d: dict) -> list[str]:
    problems = []
    for name, s in expected_text(d).items():
        if s not in html:
            problems.append(f"text not found ({name}): {s!r}")
    for name, xs in expected_positions(d).items():
        for x in xs:
            if f'"{x:.1f}"' not in html:
                problems.append(f"position not found ({name}): {x:.1f}")
    # the unmatched cohort split drawn as dots: 15 of 32 filled is the nearest to the autistic share
    share = d["n_asd"] / d["n"]
    if abs(15 / 32 - share) > 0.02:
        problems.append(f"dot cloud 15/32 = {15 / 32:.3f} differs from the autistic share {share:.3f}")
    return problems


def render(html_path: Path, out: Path, work: Path) -> None:
    html = html_path.read_text()
    html = html.replace('<script src="./support.js"></script>',
                        f"<style>@page{{size:{W}px {H}px;margin:0}}html,body{{margin:0;background:#fff}}</style>")
    page = work / "ga.html"
    page.write_text(html)
    base = [str(CHROME), "--headless=new", "--disable-gpu", "--hide-scrollbars"]
    subprocess.run(base + [f"--force-device-scale-factor={DPR}", f"--window-size={W},{H}",
                           f"--screenshot={out / (STEM + '.png')}", page.as_uri()],
                   check=True, capture_output=True, timeout=180)
    subprocess.run(base + ["--no-pdf-header-footer", f"--print-to-pdf={out / (STEM + '.pdf')}", page.as_uri()],
                   check=True, capture_output=True, timeout=180)
    im = Image.open(out / (STEM + ".png")).convert("RGB")
    assert im.size == (W * DPR, H * DPR), im.size
    im.save(out / (STEM + ".tiff"), compression="tiff_lzw", dpi=(600, 600))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default=str(ROOT / "results/latest_study"))
    ap.add_argument("--no-copy", action="store_true")
    a = ap.parse_args()
    study = Path(a.study).resolve()
    out = study / "figures"
    src = out / "graphical_abstract_design" / "Main.dc.html"
    html = src.read_text()
    d = evidence(study)
    problems = check(html, d)
    for p in problems:
        print("CHECK", p)
    if problems:
        raise SystemExit(f"{len(problems)} mismatch(es) between the artboard and the study tables")
    print("artboard numbers and dot positions match the study tables")
    work = out / "graphical_abstract_design" / "_render"
    work.mkdir(exist_ok=True)
    render(src, out, work)
    shutil.rmtree(work)
    im = Image.open(out / (STEM + ".tiff"))
    print(f"{STEM}.png/.pdf/.tiff written: {im.size[0]} x {im.size[1]} px, ratio {im.size[0] / im.size[1]:.2f}:1, "
          f"{im.info.get('dpi')} dpi (Elsevier minimum 1328 x 531 px, 300 dpi)")
    if not a.no_copy:
        for ext in ("png", "pdf"):
            shutil.copy2(out / f"{STEM}.{ext}", JOURNAL_FIG / f"{STEM}.{ext}")
        print("copied png and pdf to", JOURNAL_FIG)
    manifest = {
        "figure": STEM, "artboard_px": [W, H], "ratio": W / H, "design": "Claude Design canvas, Main.dc.html (graphical_abstract_design/)",
        "min_text_px": 32, "min_text_pt_at_13cm": round(32 / W * 130 / 25.4 * 72, 1), "font": "Arial",
        "sources": {"discrimination": "tables/intervals.csv", "calibration": "review/calibration_nested.csv",
                    "cognitive ability": "tables/subgroup_auroc.csv", "severity": "tables/severity_pooled_vs_within.csv",
                    "somatomotor": ["tables/sensorimotor_regions.csv", "tables/sensorimotor_regions_matched.csv"],
                    "classifiers": "tables/classifier.csv",
                    "site-disjoint halves": "review/interpretation_splithalf_sitedisjoint.csv (raw variant, structure coefficients)"},
        "values": {"n": d["n"], "n_autistic": d["n_asd"], "auroc_ten_fold": round(d["auroc"], 4), "auroc_loso": round(d["auroc_loso"], 4),
                   "citl_native": round(d["citl_nat_pooled"][0], 4), "citl_recalibrated": round(d["citl_rec_pooled"][0], 4),
                   "iq_bands": [round(float(v), 4) for v in d["iq"]], "somatomotor_mean_all": round(float(d["sm_all"].mean()), 4),
                   "somatomotor_mean_matched": round(float(d["sm_matched"].mean()), 4),
                   "linear": [round(float(v), 4) for v in d["lin"]], "trees": [round(float(v), 4) for v in d["tree"]],
                   "site_disjoint_system_pair": [round(float(v), 4) for v in d["sd_pair"]],
                   "site_disjoint_connection": [round(float(v), 4) for v in d["sd_conn"]],
                   "chance_p95_system_pair": round(d["sd_pair_p95"], 4)},
        "positions_px": {k: [round(float(x), 1) for x in v] for k, v in expected_positions(d).items()},
        "alt_text": (
            "Graphical abstract. Title: Beyond discrimination, toward trustworthy autism classification from connectomes. "
            "Left, cohort and method: 871 participants from ABIDE-I, three connectivity views, and logistic models fitted inside training data only, "
            f"giving AUROC {d['auroc']:.3f} with ten-fold validation and {d['auroc_loso']:.3f} leave-one-site-out. "
            f"Right, five findings. 1: AUROC falls from {d['iq'][0]:.3f} to {d['iq'][3]:.3f} across unmatched IQ bands of autistic participants, "
            "and no symptom association within autism survives correction. 2: the somatomotor reduction strengthens after within-site motion matching (exploratory). "
            f"3: linear models reach AUROC {d['lin'].min():.3f} to {d['lin'].max():.3f} against {d['tree'].min():.3f} to {d['tree'].max():.3f} for tree ensembles, "
            "and the windowing gain disappears with matched shrinkage. "
            f"4: recalibration moves the calibration offset from {sgn(d['citl_nat_pooled'][0], 3)} to {sgn(d['citl_rec_pooled'][0], 3)} without changing AUROC, "
            "and site offsets vary with the autistic proportion. "
            f"5: default-mode pairs carry the largest attribution shares and system-pair patterns reproduce across halves that share no site "
            f"(rho {d['sd_pair'].min():.2f} to {d['sd_pair'].max():.2f}, chance up to {d['sd_pair_p95']:.2f}), more than single connections. "
            "Conclusion: reliable connectome-based autism classification requires evaluating discrimination, calibration, cohort characteristics, "
            "analytical design and explanation reproducibility."),
    }
    (out / f"{STEM}.sources.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print("wrote", out / f"{STEM}.sources.json")


if __name__ == "__main__":
    main()
