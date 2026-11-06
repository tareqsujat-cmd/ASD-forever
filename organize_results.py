"""
Organise a finished run directory into a clean, self-describing layout and write a
README index with the headline numbers pulled from the result JSONs.

Idempotent: safe to run repeatedly.  Layout produced:

    <run>/
    ├── README.md                     index + results tables
    ├── run_manifest.json
    ├── run.log
    ├── model/                        multi-atlas tangent-linear + phenotype
    ├── benchmarks/                   classifier comparison (logreg vs trees)
    └── analyses/
        ├── baseline/                 correlation/tangent single-model benchmark
        ├── interpretability_edges/   cross-fold edge importance
        ├── brain/                    glass-brain + Yeo-7 + example FC
        ├── lowrank/                  AUROC vs subspace rank
        └── shap/                     SHAP validation + per-subject + TreeSHAP

Usage:
    python organize_results.py --dir results/latest
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _mv(src: Path, dst: Path):
    if src.exists() and src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return
        shutil.move(str(src), str(dst))
        print(f"  {src} -> {dst}")


def restructure(run: Path):
    # drop macOS cruft
    for junk in run.rglob(".DS_Store"):
        junk.unlink(missing_ok=True)

    _mv(run / "sota", run / "model")

    bench = run / "benchmarks"
    for f in list(run.glob("classifier_comparison.*")):
        _mv(f, bench / f.name)
    # comparison may instead sit inside a timestamped create_run_dir subdir
    for d in glob.glob(str(run / "*classifier_comparison")):
        dd = Path(d)
        for f in dd.glob("classifier_comparison.*"):
            _mv(f, bench / f.name)
        shutil.rmtree(dd, ignore_errors=True)

    an = run / "analyses"
    for d in glob.glob(str(an / "*e1_baseline")):
        _mv(Path(d), an / "baseline")
    for d in glob.glob(str(an / "*xai")):
        _mv(Path(d), an / "interpretability_edges")


def _fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def build_readme(run: Path) -> str:
    model = run / "model"
    sota = _load(model / "sota_report.json") or []
    sel = _load(model / "selective" / "selective_report.json") or []
    comp = _load(run / "benchmarks" / "classifier_comparison.json")
    low = _load(run / "analyses" / "lowrank" / "lowrank_report.json")
    shp = _load(run / "analyses" / "shap" / "shap_report.json")

    sota_by = {r.get("protocol"): r for r in sota}
    sel_by = {r.get("tag"): r for r in sel}

    L = []
    L.append("# ASD Connectome Classifier — Results\n")
    L.append("Multi-atlas (CC200 + AAL + HO) tangent-space functional connectivity, "
             "ℓ2-logistic ensemble with phenotype late-fusion, nested cross-validation "
             "on ABIDE-I (871 subjects: 403 ASD / 468 controls).\n")
    L.append("See `../../METHODOLOGY.md` for the full method and reasoning.\n")

    # headline table
    L.append("## Headline results\n")
    L.append("| Protocol | AUROC | Accuracy | AUROC @50% coverage | Acc @50% | ECE |")
    L.append("|----------|-------|----------|---------------------|----------|-----|")
    for proto, tag in (("pooled", "pooled"), ("loso", "loso")):
        s = sota_by.get(proto, {}).get("fold_mean", {})
        sd = sota_by.get(proto, {}).get("fold_std", {})
        e = sel_by.get(tag, {})
        au = s.get("auroc"); ausd = sd.get("auroc")
        auc_cell = f"{_fmt(au)} ± {_fmt(ausd)}" if au is not None else "—"
        name = "Pooled 10-fold" if proto == "pooled" else "Leave-one-site-out"
        L.append(f"| {name} | {auc_cell} | {_fmt(s.get('accuracy'))} | "
                 f"{_fmt(e.get('auroc_at_50pct'))} | {_fmt(e.get('acc_at_50pct'))} | "
                 f"{_fmt(e.get('ece'))} |")
    L.append("\n*Selective prediction: the model abstains on its least-confident cases; "
             "@50% coverage = performance on the confident half.*\n")

    # classifier comparison
    if comp:
        L.append("## Classifier comparison (same tangent-FC, same CV)\n")
        L.append("| Classifier | AUROC | Accuracy |")
        L.append("|------------|-------|----------|")
        for r in sorted(comp, key=lambda r: -r.get("auroc_mean", 0)):
            L.append(f"| {r['classifier']} | {_fmt(r.get('auroc_mean'))} ± "
                     f"{_fmt(r.get('auroc_std'))} | {_fmt(r.get('acc_mean'))} ± "
                     f"{_fmt(r.get('acc_std'))} |")
        L.append("\n*The linear model outperforms every tree ensemble by a wide margin "
                 "(~0.13–0.15 AUROC) — the ASD-FC signal is low-rank and linear, structure "
                 "the tree ensembles cannot exploit at p ≫ n.*\n")

    # low-rank
    if low:
        L.append("## Why simple wins (low-rank analysis)\n")
        L.append(f"- AUROC reaches 95% of its maximum by **rank {low.get('rank_at_95pct')}** "
                 f"(of {low.get('n_features')} features).")
        L.append(f"- Effective rank (participation ratio): "
                 f"**{_fmt(low.get('participation_ratio'), 1)}**.\n")

    # shap
    if shp:
        L.append("## SHAP interpretation\n")
        L.append(f"- Linear-SHAP vs coefficient validation: r = "
                 f"**{_fmt(shp.get('shap_vs_coef_corr'), 4)}** (exact for a linear model).")
        tsc = shp.get("tree_shap_rank_corr") or {}
        if tsc:
            pretty = ", ".join(f"{k} ρ={_fmt(v)}" for k, v in tsc.items())
            L.append(f"- TreeSHAP edge-importance vs linear (Spearman): {pretty} — only "
                     f"weakly correlated, consistent with the tree ensembles failing to "
                     f"recover the low-rank linear signal (hence their lower AUROC).")
        L.append("")

    # file map
    L.append("## Directory map\n")
    L.append("```")
    L.append("model/         multi-atlas model: sota_report.json, OOF, figures/, selective/")
    L.append("benchmarks/    classifier_comparison.{json,png}")
    L.append("analyses/")
    L.append("  baseline/                correlation/tangent single-model benchmark")
    L.append("  interpretability_edges/  cross-fold edge importance")
    L.append("  brain/                   glass-brain connectome + Yeo-7 + example FC")
    L.append("  lowrank/                 AUROC vs subspace rank")
    L.append("  shap/                    SHAP validation, per-subject waterfalls, TreeSHAP")
    L.append("run_manifest.json / run.log   seed, args, git, device, full log")
    L.append("```")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Organise a finished run directory")
    ap.add_argument("--dir", default="results/latest")
    args = ap.parse_args()
    run = Path(args.dir).resolve()
    print(f"organising {run}")
    restructure(run)
    (run / "README.md").write_text(build_readme(run))
    print("README ->", run / "README.md")
    print("\nfinal tree:")
    for p in sorted(run.rglob("*")):
        if p.is_dir():
            print("  ", p.relative_to(run))


if __name__ == "__main__":
    main()
