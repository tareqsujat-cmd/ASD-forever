"""
Figures that report what the standard summaries leave out.

Three additions, each answering a question the bar charts cannot.

*Decision curve.* Reporting guidance for clinical prediction models asks for a measure of
clinical usefulness alongside discrimination and calibration. The decision curve plots
net benefit against the threshold probability at which one would act, and compares the
model with the two trivial policies of assessing everyone and assessing no one. A model
is useful over the range of thresholds where its curve lies above both.

*Critical difference diagram.* Comparing several classifiers by their mean scores does not
say which differences are larger than chance. Following Demsar (2006), the classifiers are
ranked within each fold, a Friedman test is applied to the ranks, and a Nemenyi critical
difference is drawn: methods joined by a bar are not separated at the chosen level.

*Predicted-probability distributions.* The distribution of predicted probabilities within
each group shows where the two classes overlap, which a single threshold-based score
hides.

Usage:
    python viz_advanced.py --study results/latest_study
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ACCENT, BASE, MUTED = "#c0392b", "#4a7ab5", "#9aa5b1"

# Nemenyi critical values (alpha = 0.05), indexed by number of methods
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
       8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style():
    from visualization.style import ieee_style
    return ieee_style


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    _mpl().close(fig)
    print(f"  {name}.png")


# --------------------------------------------------------------------------
# decision curve
# --------------------------------------------------------------------------

def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray):
    """Net benefit of acting on the model, of acting on everyone, and of acting on none."""
    n = len(y)
    nb_model, nb_all = [], []
    prev = float(np.mean(y))
    for t in thresholds:
        act = p >= t
        tp = float(np.sum(act & (y == 1)))
        fp = float(np.sum(act & (y == 0)))
        w = t / (1.0 - t) if t < 1 else np.inf
        nb_model.append(tp / n - (fp / n) * w)
        nb_all.append(prev - (1.0 - prev) * w)
    return np.array(nb_model), np.array(nb_all)


def fig_decision_curve(oof: dict, out: Path, name="fig6_decision_curve"):
    plt = _mpl(); ieee_style = _style()
    thr = np.linspace(0.05, 0.75, 60)
    with ieee_style():
        fig, axes = plt.subplots(1, len(oof), figsize=(4.6 * len(oof), 3.5), squeeze=False)
        for ax, (tag, (y, p)) in zip(axes[0], oof.items()):
            nb, nb_all = net_benefit(np.asarray(y), np.asarray(p), thr)
            ax.plot(thr, nb, lw=2.4, color=ACCENT, label="model", zorder=3)
            ax.plot(thr, nb_all, lw=1.6, color=MUTED, ls="--", label="assess everyone")
            ax.axhline(0.0, lw=1.6, color="black", ls=":", label="assess no one")
            lo = min(0.0, float(np.nanmin(nb))) - 0.02
            ax.set_ylim(max(lo, -0.15), float(np.nanmax(nb)) + 0.05)
            ax.set_xlabel("threshold probability")
            ax.set_ylabel("net benefit")
            ax.set_title(f"Decision curve — {tag}", fontsize=9.5)
            ax.legend(fontsize=7.5, loc="upper right")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(lw=0.5, alpha=0.3)
        fig.tight_layout()
        _save(fig, out, name)


# --------------------------------------------------------------------------
# critical difference diagram
# --------------------------------------------------------------------------

def per_fold_scores(records, y, sites, n_folds, seed, protocol="pooled"):
    """Recover each condition's per-fold AUROC from its stored out-of-fold predictions."""
    from run_benchmark import _outer_splits
    from sklearn.metrics import roc_auc_score
    splits, _ = _outer_splits(protocol, y, sites, n_folds, seed)
    scores = {}
    for cond, (yt, pp) in records.items():
        yt, pp = np.asarray(yt), np.asarray(pp)
        if len(yt) != sum(len(te) for _, te in splits):
            continue
        vals, start = [], 0
        ok = True
        for _, te in splits:
            sl = slice(start, start + len(te)); start += len(te)
            if len(np.unique(yt[sl])) < 2:
                ok = False; break
            vals.append(roc_auc_score(yt[sl], pp[sl]))
        if ok:
            scores[cond] = np.array(vals)
    return scores


def fig_critical_difference(scores: dict, out: Path, alpha=0.05,
                            name="fig7_critical_difference"):
    from scipy.stats import friedmanchisquare
    plt = _mpl(); ieee_style = _style()
    names = list(scores)
    M = np.vstack([scores[n] for n in names])            # (k methods, N folds)
    k, N = M.shape
    if k < 3 or N < 3:
        print("  [skip] critical difference needs at least three methods and folds")
        return
    ranks = np.vstack([pd.Series(-M[:, j]).rank().to_numpy() for j in range(N)]).T
    avg = ranks.mean(axis=1)
    stat, p = friedmanchisquare(*[M[i] for i in range(k)])
    cd = Q05.get(k, 3.2) * np.sqrt(k * (k + 1) / (6.0 * N))

    order = np.argsort(avg)
    names_s = [names[i] for i in order]; avg_s = avg[order]

    # Layout: the rank axis runs along the top with the best rank on the left. Half the
    # methods are labelled on the left and half on the right so that the connectors do
    # not cross, and the bars joining statistically indistinguishable methods occupy a
    # band immediately beneath the axis.
    lo, hi = float(avg_s.min()), float(avg_s.max())
    pad = max(0.6, 0.08 * (hi - lo))
    left_n = (k + 1) // 2
    n_bars = 0
    groups = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (avg_s[j + 1] - avg_s[i]) <= cd:
            j += 1
        if j > i:
            groups.append((avg_s[i], avg_s[j]))
            n_bars += 1
        i += 1
    bar_band = 0.30 * max(n_bars, 1) + 0.35
    row_h = 0.42
    depth = bar_band + row_h * max(left_n, k - left_n) + 0.5

    with ieee_style():
        fig, ax = plt.subplots(figsize=(8.0, 0.30 * k + 2.6))
        ax.set_xlim(lo - pad - 0.9, hi + pad + 0.9)
        ax.set_ylim(-depth, 0.9)
        ax.axis("off")
        ax.plot([lo - pad, hi + pad], [0, 0], color="black", lw=1.3)
        for t in np.arange(np.floor(lo), np.ceil(hi) + 1):
            ax.plot([t, t], [0, 0.12], color="black", lw=1.0)
            ax.text(t, 0.22, f"{t:.0f}", ha="center", fontsize=8)

        for idx, (nm, a) in enumerate(zip(names_s, avg_s)):
            left = idx < left_n
            row = idx if left else (k - 1 - idx)
            yy = -(bar_band + row_h * row + row_h)
            edge = (lo - pad - 0.15) if left else (hi + pad + 0.15)
            ax.plot([a, a], [0, yy], color="#7a7a7a", lw=0.9)
            ax.plot([a, edge], [yy, yy], color="#7a7a7a", lw=0.9)
            ax.text(edge + (-0.12 if left else 0.12), yy, f"{nm}  ({a:.2f})",
                    va="center", ha="right" if left else "left", fontsize=8,
                    color=ACCENT if idx == 0 else "black")

        for bi, (a0, a1) in enumerate(groups):
            yy = -(0.30 * bi + 0.28)
            ax.plot([a0 - 0.04, a1 + 0.04], [yy, yy], color=ACCENT, lw=3.4,
                    solid_capstyle="round")

        ax.text((lo + hi) / 2, -depth + 0.15,
                f"Friedman p = {p:.2e}    critical difference = {cd:.2f}    "
                f"({k} methods, {N} folds)", fontsize=7.5, ha="center")
        ax.set_title("Average rank across folds — lower is better; "
                     "methods joined by a bar are not separated", fontsize=9.5, pad=14)
        fig.tight_layout()
        _save(fig, out, name)
    return {"friedman_p": float(p), "critical_difference": float(cd),
            "avg_rank": {n: float(a) for n, a in zip(names, avg)}}


# --------------------------------------------------------------------------
# predicted-probability distributions
# --------------------------------------------------------------------------

def fig_probability_distribution(oof: dict, out: Path, name="fig8_probability_distribution"):
    plt = _mpl(); ieee_style = _style()
    with ieee_style():
        fig, axes = plt.subplots(1, len(oof), figsize=(4.4 * len(oof), 3.2), squeeze=False)
        for ax, (tag, (y, p)) in zip(axes[0], oof.items()):
            y, p = np.asarray(y), np.asarray(p)
            bins = np.linspace(0, 1, 26)
            ax.hist(p[y == 0], bins=bins, alpha=0.65, color=BASE, label="control",
                    edgecolor="white", linewidth=0.4)
            ax.hist(p[y == 1], bins=bins, alpha=0.65, color=ACCENT, label="autism",
                    edgecolor="white", linewidth=0.4)
            ax.axvline(0.5, color="black", ls=":", lw=1.2)
            ax.set_xlabel("predicted probability"); ax.set_ylabel("participants")
            ax.set_title(f"Predicted probabilities — {tag}", fontsize=9.5)
            ax.legend(fontsize=7.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.tight_layout()
        _save(fig, out, name)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Additional reporting figures")
    ap.add_argument("--study", required=True)
    ap.add_argument("--model_oof", default=None,
                    help="sota_oof.json from the headline model run")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    study = Path(args.study).resolve()
    raw = study / "raw" if (study / "raw").is_dir() else study / "ablations"
    out = Path(args.out) if args.out else study / "figures"
    print("figures:")

    # --- decision curve and probability distributions from the headline model ---
    oof_path = Path(args.model_oof) if args.model_oof else None
    if oof_path is None:
        cand = sorted(Path("results").glob("*_final/model/sota_oof.json"))
        oof_path = cand[-1] if cand else None
    oof = {}
    if oof_path and oof_path.exists():
        for r in json.loads(oof_path.read_text()):
            oof[r["protocol"]] = (r["oof_true"], r["oof_prob"])
    if oof:
        fig_decision_curve(oof, out)
        fig_probability_distribution(oof, out)
    else:
        print("  [skip] no model out-of-fold predictions found")

    # --- critical difference over the classifier comparison ---
    f = raw / "classifier.json"
    if f.exists():
        import warnings
        warnings.filterwarnings("ignore")
        from run_benchmark import load_dataset
        recs = {r["condition"]: (r["oof_true"], r["oof_prob"])
                for r in json.loads(f.read_text())}
        ts, y, sites, _ = load_dataset(Path("./abide_processed"),
                                       Path("./abide_raw/ABIDE_pcp/cpac/filt_noglobal"))
        scores = per_fold_scores(recs, y, sites, args.n_folds, args.seed)
        if len(scores) >= 3:
            info = fig_critical_difference(scores, out)
            if info:
                (raw / "critical_difference.json").write_text(json.dumps(info, indent=2))
        else:
            print("  [skip] could not recover per-fold scores")
    else:
        print("  [skip] classifier.json not found")


if __name__ == "__main__":
    main()
