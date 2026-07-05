"""
Automated HTML experiment report for the ASD detection framework.

Produces a single self-contained HTML file — all figures embedded as
base64 PNG, all CSS inline, zero external dependencies — that covers
every section an IEEE reviewer or lab member needs to inspect:

  1.  Summary — key metric cards at a glance
  2.  Reproducibility manifest — environment, seeds, data fingerprint, git hash
  3.  Dataset — size, prevalence, sites, validation report
  4.  Training — CV metrics table with mean ± std and per-fold breakdown
  5.  Evaluation — full metric suite with bootstrap CIs
  6.  Statistical tests — pairwise comparison table (if available)
  7.  Error analysis — failure modes, hard examples summary
  8.  Robustness — perturbation degradation table, ΔAUC heatmap-style rows
  9.  Computational profile — FLOPs, latency, throughput, per-layer top-10
  10. Model export — ONNX / TorchScript equivalence validation
  11. Paper figures — embedded figures from paper_figures/ directory
  12. Configuration — full YAML config dump

Usage
-----
    from reporting.report_generator import ReportGenerator

    gen = ReportGenerator()
    gen.generate(bundle, out_path=out_dir / "experiment_report.html")

Bundle keys (all optional — missing sections are omitted gracefully)
--------------------------------------------------------------------
  cfg            : Config object (has .__dict__ or to_dict())
  dataset        : Dataset object (has __len__, .labels attribute)
  train_results  : dict from run_training()
  eval_bundle    : dict from run_evaluation()
  prof_report    : ProfilingReport from computational profiling
  export_report  : ExportReport from model export
  rob_report     : RobustnessReport from robustness evaluation
  out_dir        : Path — root results dir (to discover figure PNGs and JSONs)
  seed           : int
  device         : torch.device
  n_folds        : int
  git_hash       : str (auto-detected if absent)
  torch_version  : str (auto-detected)
  python_version : str (auto-detected)
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSS — minimal, IEEE-inspired, dark-header/light-body theme
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', Arial, sans-serif;
  font-size: 13px;
  background: #f4f6f9;
  color: #222;
  line-height: 1.5;
}
nav {
  position: fixed; top: 0; left: 0; width: 200px; height: 100vh;
  background: #1a2540; color: #cdd6e3; overflow-y: auto;
  padding: 16px 0; z-index: 100;
}
nav h3 { padding: 0 14px 10px; color: #7eb3d8; font-size: 11px;
          text-transform: uppercase; letter-spacing: 0.05em; }
nav a  { display: block; padding: 5px 14px; color: #b8cfe0; text-decoration: none;
         font-size: 12px; }
nav a:hover { background: #263358; color: #fff; }
main { margin-left: 200px; padding: 24px 32px; max-width: 1200px; }
h1   { font-size: 20px; color: #1a2540; margin-bottom: 6px; }
h2   { font-size: 15px; color: #1a2540; border-bottom: 2px solid #3a6bb5;
       padding-bottom: 4px; margin: 28px 0 12px; }
h3   { font-size: 13px; color: #34558a; margin: 16px 0 6px; }
.subtitle { color: #666; font-size: 12px; margin-bottom: 20px; }
/* Key metric cards */
.cards { display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0; }
.card  { background: #fff; border: 1px solid #dde3ee; border-radius: 6px;
         padding: 12px 18px; min-width: 110px; text-align: center; }
.card .val  { font-size: 22px; font-weight: bold; color: #1a2540; }
.card .ci   { font-size: 10px; color: #888; }
.card .lbl  { font-size: 11px; color: #555; margin-top: 2px; }
.card.good  .val { color: #1a7a3c; }
.card.warn  .val { color: #c06000; }
.card.bad   .val { color: #b72020; }
/* Tables */
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 12px; }
th    { background: #263358; color: #fff; padding: 6px 10px; text-align: left; }
td    { padding: 5px 10px; border-bottom: 1px solid #e4e8f0; }
tr:nth-child(even) td { background: #f8fafc; }
tr:hover td { background: #eef2fb; }
td.good { color: #1a7a3c; font-weight: 600; }
td.warn { color: #c06000; }
td.bad  { color: #b72020; font-weight: 600; }
td.num  { text-align: right; font-variant-numeric: tabular-nums; }
/* Details / collapsible */
details { margin: 10px 0; background: #fff; border: 1px solid #dde3ee;
           border-radius: 5px; }
summary { padding: 8px 14px; cursor: pointer; font-weight: 600;
           color: #263358; list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "▶  "; font-size: 10px; }
details[open] summary::before { content: "▼  "; }
.det-body { padding: 10px 16px 14px; }
/* Code / pre */
pre { background: #1e2733; color: #c8d8e8; padding: 12px 16px; font-size: 11px;
      border-radius: 4px; overflow-x: auto; white-space: pre-wrap;
      word-break: break-word; margin: 8px 0; }
/* Badges */
.badge { display: inline-block; padding: 2px 7px; border-radius: 10px;
         font-size: 11px; font-weight: 600; }
.badge.pass { background: #d4edda; color: #1a7a3c; }
.badge.fail { background: #f8d7da; color: #b72020; }
.badge.warn { background: #fff3cd; color: #856404; }
/* Figures */
.figures { display: flex; flex-wrap: wrap; gap: 14px; margin: 14px 0; }
.fig-box { background: #fff; border: 1px solid #dde3ee; border-radius: 5px;
            padding: 8px; max-width: 340px; }
.fig-box img { max-width: 100%; height: auto; display: block; }
.fig-box .fig-cap { font-size: 10px; color: #666; margin-top: 5px; text-align: center; }
/* Delta cells in robustness table */
.delta-neg { color: #b72020; font-weight: 600; }
.delta-pos { color: #1a7a3c; }
/* Section anchors */
.section-anchor { scroll-margin-top: 16px; }
/* Alert box */
.alert { padding: 10px 14px; border-radius: 4px; margin: 8px 0; font-size: 12px; }
.alert.info    { background: #d1ecf1; border-left: 4px solid #0c5460; color: #0c5460; }
.alert.warning { background: #fff3cd; border-left: 4px solid #856404; color: #533f03; }
.alert.error   { background: #f8d7da; border-left: 4px solid #721c24; color: #721c24; }
"""

# ---------------------------------------------------------------------------
# HTML primitives
# ---------------------------------------------------------------------------

def _esc(s: Any) -> str:
    """HTML-escape a value."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _section(title: str, anchor: str, body: str, open_: bool = True) -> str:
    open_attr = " open" if open_ else ""
    return (
        f'<details{open_attr} id="{anchor}" class="section-anchor">'
        f'<summary>{_esc(title)}</summary>'
        f'<div class="det-body">{body}</div>'
        f'</details>\n'
    )


def _table(
    headers: List[str],
    rows: List[List[Any]],
    col_classes: Optional[List[str]] = None,
) -> str:
    """Build an HTML table from headers and rows."""
    th_html = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    rows_html = ""
    for row in rows:
        tds = ""
        for c_idx, cell in enumerate(row):
            cls = ""
            if col_classes and c_idx < len(col_classes):
                cls = col_classes[c_idx]
            # Auto-colour delta columns
            if isinstance(cell, float) and not isinstance(cell, bool):
                cls_extra = ""
                if "delta" in (col_classes[c_idx] if col_classes and c_idx < len(col_classes) else "").lower():
                    if cell < -0.05:
                        cls_extra = " bad"
                    elif cell < 0:
                        cls_extra = " warn"
                    elif cell > 0:
                        cls_extra = " good"
                cls = (cls + cls_extra).strip()
                cell_str = f"{cell:+.4f}" if "delta" in cls.lower() else f"{cell:.4f}"
            else:
                cell_str = _esc(cell)
            tds += f'<td class="{cls}">{cell_str}</td>'
        rows_html += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{th_html}</tr></thead><tbody>{rows_html}</tbody></table>"


def _card(label: str, value: str, ci: str = "", rating: str = "") -> str:
    cls = f"card {rating}".strip()
    ci_html = f'<div class="ci">{_esc(ci)}</div>' if ci else ""
    return (
        f'<div class="{cls}">'
        f'<div class="val">{_esc(value)}</div>'
        f'{ci_html}'
        f'<div class="lbl">{_esc(label)}</div>'
        f'</div>'
    )


def _badge(text: str, kind: str = "pass") -> str:
    return f'<span class="badge {kind}">{_esc(text)}</span>'


def _embed_image(path: Path) -> str:
    """Return an <img> tag with base64-encoded PNG."""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}" loading="lazy" />'
    except Exception:
        return f'<p style="color:#888;font-size:11px">[Image unavailable: {path.name}]</p>'


def _fmt(v: Any, decimals: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float) and (v != v):    # NaN
        return "NaN"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _rating(v: float, good: float = 0.80, warn: float = 0.65) -> str:
    if v >= good:
        return "good"
    if v >= warn:
        return "warn"
    return "bad"


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def _git_hash(cwd: Optional[Path] = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=str(cwd) if cwd else None, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def _library_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for pkg in ["torch", "numpy", "sklearn", "scipy", "matplotlib",
                "optuna", "nibabel", "pandas"]:
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    return versions


def _sha256_of_json(path: Path) -> str:
    """SHA-256 of a JSON file for integrity tracking."""
    try:
        data = path.read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_summary(bundle: dict) -> str:
    eval_b   = bundle.get("eval_bundle", {})
    report   = eval_b.get("report")
    cv_rep   = eval_b.get("cv_report", {})

    cards_html = '<div class="cards">'

    if report is not None:
        m = report.metrics if hasattr(report, "metrics") else {}
        pairs: List[Tuple[str, str, str]] = [
            ("AUC",         "auc",         ""),
            ("Sensitivity", "sensitivity", ""),
            ("Specificity", "specificity", ""),
            ("F1",          "f1",          ""),
            ("MCC",         "mcc",         ""),
            ("Kappa",       "kappa",       ""),
        ]
        for label, key, _ in pairs:
            val = float(m.get(key, float("nan")))
            ci_lo = float(m.get(f"{key}_ci_lo", float("nan")))
            ci_hi = float(m.get(f"{key}_ci_hi", float("nan")))
            ci_str = (
                f"[{ci_lo:.3f}, {ci_hi:.3f}]"
                if not (ci_lo != ci_lo or ci_hi != ci_hi)
                else ""
            )
            rtg = _rating(val) if key in ("auc", "f1", "sensitivity") else ""
            cards_html += _card(label, _fmt(val, 3), ci_str, rtg)
    elif cv_rep:
        for key in ("auc", "f1", "sensitivity", "specificity"):
            mci = cv_rep.get(key)
            if mci is not None:
                val  = getattr(mci, "value", float("nan"))
                lo   = getattr(mci, "ci_lower", float("nan"))
                hi   = getattr(mci, "ci_upper", float("nan"))
                rtg  = _rating(val) if key in ("auc", "f1", "sensitivity") else ""
                ci_str = f"[{lo:.3f}, {hi:.3f}]" if not (lo != lo) else ""
                cards_html += _card(key.upper(), _fmt(val, 3), ci_str, rtg)

    cards_html += "</div>"
    return cards_html


def _build_reproducibility(bundle: dict) -> str:
    seed    = bundle.get("seed", "?")
    device  = str(bundle.get("device", "?"))
    git_h   = bundle.get("git_hash") or _git_hash(bundle.get("out_dir"))
    libs    = _library_versions()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    fingerprint = "?"
    out_dir: Optional[Path] = bundle.get("out_dir")
    if out_dir:
        dvr_path = out_dir / "data_validation_report.json"
        if dvr_path.exists():
            try:
                dvr = json.loads(dvr_path.read_text())
                fingerprint = dvr.get("fingerprint", "?")
            except Exception:
                pass

    rows = [
        ["Generated at",     now_str],
        ["Python",           sys.version.split()[0]],
        ["Platform",         platform.platform()],
        ["Git hash",         git_h],
        ["Random seed",      str(seed)],
        ["Device",           device],
        ["Data fingerprint", fingerprint],
    ]
    for lib, ver in libs.items():
        rows.append([lib, ver])

    return _table(["Property", "Value"], rows)


def _build_dataset(bundle: dict) -> str:
    dataset = bundle.get("dataset")
    n_folds = bundle.get("n_folds", "?")
    out_dir: Optional[Path] = bundle.get("out_dir")

    parts: List[str] = []

    # Basic counts
    if dataset is not None:
        n_total = len(dataset)
        if hasattr(dataset, "labels"):
            labels = np.asarray(dataset.labels)
            n_asd  = int(labels.sum())
            n_tc   = n_total - n_asd
            prev   = n_asd / n_total
            rows   = [
                ["Total subjects",  n_total],
                ["ASD",             n_asd],
                ["TC",              n_tc],
                ["Prevalence",      f"{prev:.1%}"],
                ["CV folds",        n_folds],
            ]
            if hasattr(dataset, "sites"):
                sites = np.asarray(dataset.sites)
                rows.append(["Sites", len(np.unique(sites))])
            parts.append(_table(["Property", "Value"], rows))

    # Data validation report
    if out_dir:
        dvr_path = out_dir / "data_validation_report.json"
        if dvr_path.exists():
            try:
                dvr = json.loads(dvr_path.read_text())
                status_badge = (
                    _badge("PASSED", "pass")
                    if dvr.get("passed", False)
                    else _badge("FAILED", "fail")
                )
                parts.append(f"<h3>Data Validation {status_badge}</h3>")
                issues = dvr.get("issues", [])
                if issues:
                    issue_rows = [
                        [i.get("severity", "?"),
                         i.get("category", "?"),
                         i.get("message", "?")[:120]]
                        for i in issues[:20]
                    ]
                    parts.append(
                        _table(["Severity", "Category", "Message"], issue_rows)
                    )
                    if len(issues) > 20:
                        parts.append(f"<p>… and {len(issues)-20} more issues</p>")
                else:
                    parts.append('<div class="alert info">No issues recorded.</div>')
            except Exception as exc:
                parts.append(f'<div class="alert warning">Could not load validation report: {_esc(exc)}</div>')

    return "\n".join(parts) if parts else "<p>No dataset info available.</p>"


def _build_training(bundle: dict) -> str:
    train = bundle.get("train_results", {})
    parts: List[str] = []

    mean_m = train.get("mean_metrics", {})
    std_m  = train.get("std_metrics",  {})

    if mean_m:
        rows = []
        for k, v in sorted(mean_m.items()):
            std = std_m.get(k, 0.0)
            rows.append([k, f"{v:.4f}", f"± {std:.4f}"])
        parts.append("<h3>Cross-Validation Mean ± Std</h3>")
        parts.append(_table(["Metric", "Mean", "Std"], rows))

    fold_results = train.get("fold_results", [])
    if fold_results:
        parts.append("<h3>Per-Fold Breakdown</h3>")
        all_keys = sorted({k for f in fold_results for k in f.keys()})
        headers  = ["Fold"] + all_keys
        rows = []
        for i, f in enumerate(fold_results):
            rows.append([f"Fold {i+1}"] + [_fmt(f.get(k, float("nan"))) for k in all_keys])
        parts.append(_table(headers, rows))

    return "\n".join(parts) if parts else "<p>No training results available.</p>"


def _build_evaluation(bundle: dict) -> str:
    eval_b = bundle.get("eval_bundle", {})
    parts: List[str] = []

    # Full metrics table
    report = eval_b.get("report")
    if report is not None and hasattr(report, "metrics"):
        m = report.metrics
        rows = []
        for k, v in sorted(m.items()):
            if k.startswith("per_class") or not isinstance(v, (int, float)):
                continue
            rows.append([k, _fmt(float(v))])
        if rows:
            parts.append("<h3>Full Metric Suite</h3>")
            parts.append(_table(["Metric", "Value"], rows))

        # Per-class breakdown
        pc = m.get("per_class", {})
        if pc:
            parts.append("<h3>Per-Class Metrics</h3>")
            headers = ["Class"] + list(next(iter(pc.values())).keys())
            pc_rows = [[cls] + [_fmt(v2) for v2 in mv.values()]
                       for cls, mv in pc.items()]
            parts.append(_table(headers, pc_rows))

    # CV report with bootstrap CIs
    cv_rep = eval_b.get("cv_report", {})
    if cv_rep:
        parts.append("<h3>Cross-Validated Metrics (t-distribution CI)</h3>")
        rows = []
        for k, mci in sorted(cv_rep.items()):
            val  = getattr(mci, "value",    float("nan"))
            lo   = getattr(mci, "ci_lower", float("nan"))
            hi   = getattr(mci, "ci_upper", float("nan"))
            rows.append([k, _fmt(val), f"[{_fmt(lo)}, {_fmt(hi)}]"])
        parts.append(_table(["Metric", "Value", "95% CI"], rows))

    return "\n".join(parts) if parts else "<p>No evaluation results available.</p>"


def _build_error_analysis(bundle: dict) -> str:
    eval_b  = bundle.get("eval_bundle", {})
    err_rep = eval_b.get("err_report")
    parts:  List[str] = []

    if err_rep is None:
        out_dir: Optional[Path] = bundle.get("out_dir")
        if out_dir:
            err_path = out_dir / "evaluation" / "error_analysis" / "error_analysis.json"
            if err_path.exists():
                try:
                    data = json.loads(err_path.read_text())
                    rows = [
                        ["Total errors",     data.get("total_errors", "?")],
                        ["False positives",  data.get("n_fp", "?")],
                        ["False negatives",  data.get("n_fn", "?")],
                        ["Hard FP",          data.get("n_hard_fp", "?")],
                        ["Hard FN",          data.get("n_hard_fn", "?")],
                        ["Uncertain",        data.get("n_uncertain", "?")],
                        ["Clinical cost",    _fmt(data.get("clinical_cost", float("nan")))],
                    ]
                    parts.append("<h3>Error Summary</h3>")
                    parts.append(_table(["Property", "Value"], rows))
                    fmodes = data.get("failure_modes", [])
                    if fmodes:
                        parts.append("<h3>Detected Failure Modes</h3>")
                        parts.append("<ul>" + "".join(f"<li>{_esc(m)}</li>" for m in fmodes) + "</ul>")
                    recs = data.get("recommendations", [])
                    if recs:
                        parts.append("<h3>Recommendations</h3>")
                        parts.append("<ul>" + "".join(f"<li>{_esc(r)}</li>" for r in recs) + "</ul>")
                except Exception as exc:
                    parts.append(f'<div class="alert warning">Could not load error analysis: {_esc(exc)}</div>')
        return "\n".join(parts) if parts else "<p>No error analysis available.</p>"

    # Live report object
    rows = [
        ["Total errors",    getattr(err_rep, "total_errors",   "?")],
        ["False positives", getattr(err_rep, "n_fp",           "?")],
        ["False negatives", getattr(err_rep, "n_fn",           "?")],
        ["Hard FP (≥75%)",  getattr(err_rep, "n_hard_fp",      "?")],
        ["Hard FN (≥75%)",  getattr(err_rep, "n_hard_fn",      "?")],
        ["Uncertain",       getattr(err_rep, "n_uncertain",    "?")],
        ["Clinical cost",   _fmt(getattr(err_rep, "clinical_cost", float("nan")))],
    ]
    parts.append("<h3>Error Summary</h3>")
    parts.append(_table(["Property", "Value"], rows))

    fmodes = getattr(err_rep, "failure_modes", [])
    if fmodes:
        parts.append("<h3>Failure Modes</h3>")
        parts.append("<ul>" + "".join(f"<li>{_esc(m)}</li>" for m in fmodes) + "</ul>")

    recs = getattr(err_rep, "recommendations", [])
    if recs:
        parts.append("<h3>Recommendations</h3>")
        parts.append("<ul>" + "".join(f"<li>{_esc(r)}</li>" for r in recs) + "</ul>")

    return "\n".join(parts)


def _build_robustness(bundle: dict) -> str:
    rob = bundle.get("rob_report")
    parts: List[str] = []

    if rob is None:
        out_dir: Optional[Path] = bundle.get("out_dir")
        if out_dir:
            rob_path = out_dir / "robustness" / "robustness_report.json"
            if rob_path.exists():
                try:
                    data   = json.loads(rob_path.read_text())
                    bl     = data.get("baseline", {})
                    bl_auc = bl.get("metrics", {}).get("auc", float("nan"))
                    parts.append(
                        f'<div class="alert info">Baseline AUC: {_fmt(bl_auc)}'
                        f"  |  n={data.get('n_subjects','?')}"
                        f"  |  threshold={data.get('threshold','?')}</div>"
                    )
                    conds = data.get("conditions", [])
                    if conds:
                        rows = []
                        for c in conds:
                            dauc = c.get("delta_auc", 0.0)
                            rows.append([
                                c.get("condition", "?"),
                                c.get("n_subjects", "?"),
                                _fmt(c.get("metrics", {}).get("auc", float("nan"))),
                                dauc,
                                c.get("delta_f1", 0.0),
                                c.get("delta_sens", 0.0),
                                c.get("delta_spec", 0.0),
                            ])
                        parts.append(_table(
                            ["Condition", "N", "AUC", "ΔAUC", "ΔF1", "ΔSens", "ΔSpec"],
                            rows,
                            col_classes=["", "num", "num", "delta num", "delta num",
                                         "delta num", "delta num"],
                        ))
                except Exception as exc:
                    parts.append(f'<div class="alert warning">Could not load robustness report: {_esc(exc)}</div>')
        return "\n".join(parts) if parts else "<p>No robustness report available.</p>"

    # Live report object
    bl = rob.baseline
    if bl:
        bl_auc = bl.metrics.get("auc", float("nan"))
        parts.append(
            f'<div class="alert info">Baseline AUC: {_fmt(bl_auc)}'
            f"  |  n={rob.n_subjects}"
            f"  |  threshold={rob.threshold}</div>"
        )

    rows = []
    for c in rob.conditions:
        rows.append([
            c.condition,
            c.n_subjects,
            _fmt(c.metrics.get("auc", float("nan"))),
            c.delta_auc,
            c.delta_f1,
            c.delta_sens,
            c.delta_spec,
        ])
    if rows:
        parts.append(_table(
            ["Condition", "N", "AUC", "ΔAUC", "ΔF1", "ΔSens", "ΔSpec"],
            rows,
            col_classes=["", "num", "num", "delta num", "delta num",
                         "delta num", "delta num"],
        ))
    return "\n".join(parts) if parts else "<p>No robustness conditions recorded.</p>"


def _build_compute_profile(bundle: dict) -> str:
    prof = bundle.get("prof_report")
    parts: List[str] = []

    if prof is None:
        out_dir: Optional[Path] = bundle.get("out_dir")
        if out_dir:
            cp_path = out_dir / "computational_profile" / "computational_profile.json"
            if cp_path.exists():
                try:
                    data = json.loads(cp_path.read_text())
                    summary_rows = [
                        ["Parameters (total)",     _fmt(data.get("total_params", 0), 0)],
                        ["Parameters (trainable)", _fmt(data.get("trainable_params", 0), 0)],
                        ["FP32 size (MB)",          _fmt(data.get("size_fp32_mb", 0.0))],
                        ["GFLOPs",                  _fmt(data.get("gflops", 0.0))],
                        ["GMACs",                   _fmt(data.get("gmacs", 0.0))],
                        ["FLOPs source",            data.get("flops_source", "?")],
                        ["GPU",                     data.get("gpu_name", "CPU")],
                    ]
                    parts.append("<h3>Architecture & Complexity</h3>")
                    parts.append(_table(["Property", "Value"], summary_rows))

                    lat = data.get("latency", {})
                    if lat:
                        parts.append("<h3>Latency & Throughput</h3>")
                        lat_rows = []
                        for bs, ls in sorted(lat.items(), key=lambda x: int(x[0])):
                            lat_rows.append([
                                bs,
                                f"{ls.get('mean_ms',0):.2f} ± {ls.get('std_ms',0):.2f}",
                                f"{ls.get('p95_ms',0):.2f}",
                                f"{ls.get('throughput_samples_per_sec',0):.0f}",
                                f"{ls.get('peak_gpu_memory_mb',0):.0f}",
                            ])
                        parts.append(_table(
                            ["Batch size", "Latency ms (mean±std)", "P95 ms",
                             "Throughput (s/s)", "Peak GPU MB"],
                            lat_rows,
                        ))

                    layers = data.get("top_layers_by_time", [])
                    if layers:
                        parts.append("<h3>Top-10 Layers by CPU Time</h3>")
                        layer_rows = [
                            [i+1, l.get("name","?")[:55],
                             f"{l.get('cpu_time_ms',0):.3f}",
                             f"{l.get('cuda_time_ms',0):.3f}",
                             l.get("n_calls","?")]
                            for i, l in enumerate(layers[:10])
                        ]
                        parts.append(_table(
                            ["#", "Layer / Op", "CPU ms", "CUDA ms", "Calls"],
                            layer_rows,
                        ))
                except Exception as exc:
                    parts.append(f'<div class="alert warning">Could not load profile: {_esc(exc)}</div>')
        return "\n".join(parts) if parts else "<p>No computational profile available.</p>"

    # Live ProfilingReport object
    summary_rows = [
        ["Parameters (total)",     f"{prof.total_params:,}"],
        ["Parameters (trainable)", f"{prof.trainable_params:,}"],
        ["FP32 size (MB)",          _fmt(prof.size_fp32_mb)],
        ["GFLOPs",                  _fmt(prof.gflops)],
        ["GMACs",                   _fmt(prof.gmacs)],
        ["FLOPs source",            prof.flops_source],
        ["GPU",                     prof.gpu_name or "CPU"],
    ]
    parts.append("<h3>Architecture & Complexity</h3>")
    parts.append(_table(["Property", "Value"], summary_rows))

    if prof.latency:
        parts.append("<h3>Latency & Throughput</h3>")
        lat_rows = []
        for bs, ls in sorted(prof.latency.items()):
            lat_rows.append([
                bs,
                f"{ls.mean_ms:.2f} ± {ls.std_ms:.2f}",
                f"{ls.p95_ms:.2f}",
                f"{ls.throughput_samples_per_sec:.0f}",
                f"{ls.peak_gpu_memory_mb:.0f}",
            ])
        parts.append(_table(
            ["Batch size", "Latency ms (mean±std)", "P95 ms",
             "Throughput (s/s)", "Peak GPU MB"],
            lat_rows,
        ))

    if prof.top_layers_by_time:
        parts.append("<h3>Top-10 Layers by CPU Time</h3>")
        layer_rows = [
            [i+1, l.get("name","?")[:55],
             f"{l.get('cpu_time_ms',0):.3f}",
             f"{l.get('cuda_time_ms',0):.3f}",
             l.get("n_calls","?")]
            for i, l in enumerate(prof.top_layers_by_time[:10])
        ]
        parts.append(_table(
            ["#", "Layer / Op", "CPU ms", "CUDA ms", "Calls"],
            layer_rows,
        ))

    return "\n".join(parts)


def _build_export(bundle: dict) -> str:
    exp = bundle.get("export_report")
    parts: List[str] = []

    def _render_results(results_list: list) -> str:
        rows = []
        for r in results_list:
            validated = r.get("validated", False) if isinstance(r, dict) else r.validated
            exported  = r.get("exported",  False) if isinstance(r, dict) else r.exported
            fmt       = r.get("format",    "?")   if isinstance(r, dict) else r.format
            size_mb   = r.get("size_mb",   0.0)   if isinstance(r, dict) else r.size_mb
            max_diff  = r.get("max_diff",  float("nan")) if isinstance(r, dict) else r.max_diff
            err_msg   = r.get("error_message", "") if isinstance(r, dict) else r.error_message

            if not exported:
                status = _badge("ERROR", "fail")
            elif validated:
                status = _badge("PASS", "pass")
            else:
                status = _badge("FAIL", "fail")

            row = [fmt, status,
                   f"{size_mb:.1f}" if exported else "—",
                   f"{max_diff:.2e}" if not (max_diff != max_diff) else "NaN"]
            if err_msg:
                row.append(err_msg[:80])
            else:
                row.append("")
            rows.append(row)
        return _table(["Format", "Status", "Size MB", "Max diff", "Error"], rows)

    if exp is None:
        out_dir: Optional[Path] = bundle.get("out_dir")
        if out_dir:
            ep = out_dir / "model_export" / "export_report.json"
            if ep.exists():
                try:
                    data = json.loads(ep.read_text())
                    overall = _badge("ALL PASS", "pass") if data.get("all_passed") else _badge("SOME FAILED", "fail")
                    parts.append(f"<p>Overall: {overall}  |  opset={data.get('onnx_opset','?')}  "
                                 f"|  tolerance={data.get('tolerance','?')}</p>")
                    parts.append(_render_results(data.get("results", [])))
                except Exception as exc:
                    parts.append(f'<div class="alert warning">Could not load export report: {_esc(exc)}</div>')
        return "\n".join(parts) if parts else "<p>No model export report available.</p>"

    overall = _badge("ALL PASS", "pass") if exp.all_passed() else _badge("SOME FAILED", "fail")
    parts.append(f"<p>Overall: {overall}  |  opset={exp.onnx_opset}  "
                 f"|  tolerance={exp.tolerance}  |  ORT={exp.ort_available}</p>")
    parts.append(_render_results(exp.results))
    return "\n".join(parts)


def _build_figures(bundle: dict) -> str:
    out_dir: Optional[Path] = bundle.get("out_dir")
    if out_dir is None:
        return "<p>No output directory specified.</p>"

    fig_dir = out_dir / "paper_figures"
    if not fig_dir.exists():
        return "<p>Paper figures directory not found.</p>"

    pngs = sorted(fig_dir.glob("*.png"))
    if not pngs:
        return "<p>No PNG figures found in paper_figures/.</p>"

    html = '<div class="figures">'
    for png in pngs:
        img_tag = _embed_image(png)
        html += (
            f'<div class="fig-box">'
            f'{img_tag}'
            f'<div class="fig-cap">{_esc(png.stem)}</div>'
            f'</div>'
        )
    html += "</div>"
    return html


def _build_config(bundle: dict) -> str:
    cfg = bundle.get("cfg")
    if cfg is None:
        return "<p>No configuration available.</p>"
    try:
        if hasattr(cfg, "to_dict"):
            cfg_str = json.dumps(cfg.to_dict(), indent=2, default=str)
        elif hasattr(cfg, "__dict__"):
            cfg_str = json.dumps(
                {k: (v.__dict__ if hasattr(v, "__dict__") else str(v))
                 for k, v in cfg.__dict__.items()},
                indent=2, default=str,
            )
        else:
            cfg_str = str(cfg)
        return f"<pre>{_esc(cfg_str)}</pre>"
    except Exception as exc:
        return f"<p>Could not serialise config: {_esc(exc)}</p>"


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Generates a self-contained HTML experiment report.

    All figures are embedded as base64 PNG; no external CSS or JS.
    Sections use ``<details>/<summary>`` for collapsible display.
    """

    def generate(
        self,
        bundle:   Dict[str, Any],
        out_path: Path,
    ) -> None:
        """
        Write the HTML report to ``out_path``.

        Parameters
        ----------
        bundle   : dict with keys described in the module docstring
        out_path : destination file path (created with parent dirs)
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        now_str  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        title    = "ASD Detection — Experiment Report"
        subtitle = f"Generated {now_str}"

        # Navigation links
        nav_links = [
            ("summary",        "Summary"),
            ("reproducibility","Reproducibility"),
            ("dataset",        "Dataset"),
            ("training",       "Training"),
            ("evaluation",     "Evaluation"),
            ("error_analysis", "Error Analysis"),
            ("robustness",     "Robustness"),
            ("compute",        "Compute Profile"),
            ("export",         "Model Export"),
            ("figures",        "Paper Figures"),
            ("config",         "Configuration"),
        ]
        nav_html = "<nav><h3>Navigation</h3>"
        for anchor, label in nav_links:
            nav_html += f'<a href="#{anchor}">{_esc(label)}</a>'
        nav_html += "</nav>"

        # Content sections
        sections = [
            ("Summary",                "summary",        _build_summary(bundle),          True),
            ("Reproducibility",        "reproducibility", _build_reproducibility(bundle), False),
            ("Dataset",                "dataset",        _build_dataset(bundle),          True),
            ("Training",               "training",       _build_training(bundle),         True),
            ("Evaluation",             "evaluation",     _build_evaluation(bundle),       True),
            ("Error Analysis",         "error_analysis", _build_error_analysis(bundle),   False),
            ("Robustness Evaluation",  "robustness",     _build_robustness(bundle),       False),
            ("Computational Profile",  "compute",        _build_compute_profile(bundle),  False),
            ("Model Export",           "export",         _build_export(bundle),           False),
            ("Paper Figures",          "figures",        _build_figures(bundle),          False),
            ("Configuration",          "config",         _build_config(bundle),           False),
        ]

        main_html = f"<h1>{_esc(title)}</h1><p class='subtitle'>{_esc(subtitle)}</p>"
        for sec_title, anchor, body, open_ in sections:
            main_html += _section(sec_title, anchor, body, open_)

        html = textwrap.dedent(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8"/>
          <meta name="viewport" content="width=device-width, initial-scale=1"/>
          <title>{_esc(title)}</title>
          <style>{_CSS}</style>
        </head>
        <body>
          {nav_html}
          <main>{main_html}</main>
        </body>
        </html>
        """).strip()

        out_path.write_text(html, encoding="utf-8")
        size_kb = out_path.stat().st_size // 1024
        logger.info("HTML report written → %s  (%d KB)", out_path, size_kb)
