"""Composite publication figures, one per claim group, in a consistent house style.

Each figure merges several stored conditions into a single panel set, so that a claim and the
controls that support it are read together rather than three figures apart.  Styling comes from
scienceplots ('science' + 'no-latex', so no TeX installation is required).

    python viz_summary.py --study results/latest_study
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASD, TD, ACC, GREY = "#B4442E", "#2E6FB4", "#2A9D8F", "#8A8A8A"


def style():
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12,
        "legend.fontsize": 10.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=6)
    ax.text(-0.08, 1.06, letter, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="right")


def load(t, name):
    p = t / name
    return pd.read_csv(p) if p.exists() else None


def fig_performance(t, raw, out):
    """Where the model stands: discrimination, per-site spread, calibration, abstention."""
    iv, ps = load(t, "intervals.csv"), load(t, "per_site_intervals.csv")
    meta = json.loads((raw / "intervals.json").read_text()).get("meta_analysis", {}) \
        if (raw / "intervals.json").exists() else {}
    conf = json.loads((raw / "conformal_ensemble.json").read_text()) \
        if (raw / "conformal_ensemble.json").exists() else {}

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.9), layout="constrained")

    ax = axes[0, 0]
    if iv is not None:
        m = ["auroc", "accuracy", "sensitivity", "specificity"]
        lbl = ["AUROC", "accuracy", "sensitivity", "specificity"]
        w, xs = 0.36, np.arange(len(m))
        for k, (proto, col) in enumerate([("pooled", TD), ("loso", ASD)]):
            r = iv[iv.protocol == proto]
            if not len(r):
                continue
            r = r.iloc[0]
            v = [r[c] for c in m]
            lo = [r[c] - r[f"{c}_lo"] for c in m]
            hi = [r[f"{c}_hi"] - r[c] for c in m]
            ax.bar(xs + (k - 0.5) * w, v, w, color=col, alpha=.85,
                   label="pooled" if proto == "pooled" else "site withheld",
                   yerr=[lo, hi], capsize=2.5, error_kw=dict(lw=.8))
        ax.axhline(.5, color=GREY, ls=":", lw=.8)
        ax.set_xticks(xs); ax.set_xticklabels(lbl, rotation=12)
        ax.set_ylim(.4, .95); ax.legend(frameon=False, loc="upper right")
    panel(ax, "a", "Discrimination, 95% DeLong and Wilson intervals")

    ax = axes[0, 1]
    if ps is not None and len(ps):
        d = ps.sort_values("auroc").reset_index(drop=True)
        yy = np.arange(len(d))
        ax.hlines(yy, d.auroc_lo, d.auroc_hi, color=GREY, lw=1)
        ax.scatter(d.auroc, yy, s=np.clip(d.n / 3, 6, 60), color=TD, zorder=3)
        ax.set_yticks(yy); ax.set_yticklabels(d.site, fontsize=8.5)
        if meta:
            mu = meta["random_effects_mean"]
            ax.axvline(mu, color=ASD, lw=1.2)
            ax.axvspan(meta["prediction_interval_lo"], meta["prediction_interval_hi"],
                       color=ASD, alpha=.10, lw=0)
            ax.text(.98, .03, f"random-effects {mu:.3f}\n"
                              f"descriptive 95% range [{meta['prediction_interval_lo']:.2f}, "
                              f"{meta['prediction_interval_hi']:.2f}]\n"
                              f"$I^2$ = {meta['I2']:.0f}%",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5)
        ax.axvline(.5, color=GREY, ls=":", lw=.8)
        ax.set_xlabel("AUROC at a site withheld from training")
    panel(ax, "b", "Every site, marker area $\\propto$ n")

    ax = axes[1, 0]
    # reliability computed from the stored predictions with ten equal-width bins, the same
    # binning as the expected calibration error (Methods, secondary probability assessment)
    oofp = raw.parent / "ensemble_oof.json"
    selp = raw.parent / "selective_ensemble" / "selective_report.json"
    # the calibration error quoted here is the stored one the text also quotes, so the figure
    # and the prose cannot disagree; only the curve is recomputed
    stored_ece = {r["tag"]: r["ece"] for r in json.loads(selp.read_text())} \
        if selp.exists() else {}
    if oofp.exists():
        recs = {r["protocol"]: r for r in json.loads(oofp.read_text())}
        for proto, col, nm in [("pooled", TD, "pooled"), ("loso", ASD, "site withheld")]:
            r = recs.get(proto)
            if not r:
                continue
            yv = np.asarray(r["oof_true"], dtype=float)
            pv = np.asarray(r["oof_prob"], dtype=float)
            edges = np.linspace(0.0, 1.0, 11)
            xs, ys = [], []
            for a_, b_ in zip(edges[:-1], edges[1:]):
                m = (pv >= a_) & (pv < b_ if b_ < 1.0 else pv <= b_)
                if not m.any():          # empty bins contribute no term
                    continue
                xs.append(pv[m].mean()); ys.append(yv[m].mean())
            e = stored_ece.get(proto)
            ax.plot(xs, ys, "o-", ms=3.5, lw=1.2, color=col,
                    label=f"{nm}" + (f" (ECE {e:.3f})" if e is not None else ""))
        ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=.8)
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel(ax, "c", "Calibration")

    ax = axes[1, 1]
    if conf:
        names, cov, met = [], [], []
        for mode in ("marginal", "mondrian"):
            if mode in conf:
                names.append("cohort-wide" if mode == "marginal" else "per site")
                cov.append(conf[mode]["coverage"] * 100)
                h = conf[mode].get("held_out", {})
                met.append(h.get("fraction_target_met", np.nan) * 100)
        xs = np.arange(len(names)); w = .36
        ax.bar(xs - w / 2, cov, w, color=ACC, alpha=.85, label="cases answered (%)")
        ax.bar(xs + w / 2, met, w, color=TD, alpha=.85, label="splits meeting target (%)")
        ax.axhline(90, color=ASD, ls="--", lw=1)
        # label sits in the empty gap between the two bar groups so it touches no bar
        ax.text((xs[0] + xs[-1]) / 2, 91.5, "nominal 90%", fontsize=9.5, color=ASD,
                ha="center", va="bottom")
        for i, c in enumerate(cov):
            if c < 1:     # vertical, inside its own bar column, clear of the neighbour
                ax.text(xs[i] - w / 2, 2, "answers nothing", rotation=90, ha="center",
                        va="bottom", fontsize=9, color=ASD)
            else:
                ax.text(xs[i] - w / 2, c + 1.5, f"{c:.0f}%", ha="center", va="bottom",
                        fontsize=9, color="black")
        ax.set_xticks(xs); ax.set_xticklabels(names)
        ax.set_ylim(0, 112); ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    panel(ax, "d", "Abstention: cases answered and target met")

    fig.savefig(out / "figS_performance.png"); plt.close(fig)
    return "figS_performance.png"


def fig_corrections(t, raw, out):
    """The four effects that change when their control is applied."""
    sev, rs, rt = load(t, "severity_pooled_vs_within.csv"), load(t, "ridge_sweep.csv"), \
        load(t, "ridge_sweep_tests.csv")
    lk, sel = load(t, "leakage_audit.csv"), load(t, "region_selection_tests.csv")
    lc = json.loads((raw / "label_ceiling.json").read_text()) \
        if (raw / "label_ceiling.json").exists() else {}

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.9), layout="constrained")

    ax = axes[0, 0]
    if sev is not None:
        d = sev[sev.protocol == "pooled"]
        inst = list(dict.fromkeys(d.instrument))
        yy = np.arange(len(inst))
        # pooled drawn as a larger open ring so that rows where the two coincide -- the
        # instruments given only to autistic participants -- still show both markers
        for scope, col, mk, s_, fc in [("pooled", ASD, "o", 62, "none"),
                                       ("within-autism", TD, "s", 24, None)]:
            v = [d[(d.instrument == i) & (d.scope == scope)].rho.mean() for i in inst]
            ax.scatter(v, yy, marker=mk, s=s_, label={"pooled": "both groups", "within-autism": "within autism"}[scope], zorder=3,
                       facecolors=(fc if fc else col), edgecolors=col, linewidths=1.1)
        for k, i in enumerate(inst):
            a = d[(d.instrument == i) & (d.scope == "pooled")].rho.mean()
            b = d[(d.instrument == i) & (d.scope == "within-autism")].rho.mean()
            ax.plot([a, b], [k, k], color=GREY, lw=.8, zorder=1)
        ax.axvline(0, color=GREY, ls=":", lw=.8)
        ctrl = {i for i in inst
                if abs(d[(d.instrument == i) & (d.scope == "pooled")].rho.mean()
                       - d[(d.instrument == i) & (d.scope == "within-autism")].rho.mean()) < 1e-9}
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{i} †" if i in ctrl else i for i in inst], fontsize=9.5)
        ax.set_xlabel("Spearman $\\rho$ with classifier score\n"
                      "† given only to autistic participants: the two scopes hold\n"
                      "the same people and coincide",
                      fontsize=9.5)
        ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    panel(ax, "a", "Severity correlation by scope")

    ax = axes[0, 1]
    if rs is not None:
        for rep, col, nm in [("static", TD, "whole scan"), ("windowed", ASD, "windowed")]:
            d = rs[rs.representation == rep].sort_values("ridge")
            if len(d):
                ax.plot(d.ridge, d.auroc, "o-", ms=3, lw=1.2, color=col, label=nm)
        st = rs[rs.representation == "static"]
        if len(st):
            b = st[st.ridge == 0.0]
            if len(b):
                ax.scatter(b.ridge, b.auroc, s=60, facecolors="none",
                           edgecolors=GREY, lw=1.2, zorder=4)
                ax.annotate("unregularised\nbaseline",
                            (b.ridge.iloc[0], b.auroc.iloc[0]), textcoords="offset points",
                            xytext=(14, 2), fontsize=8.5, color=GREY, ha="left",
                            arrowprops=dict(arrowstyle="-", color=GREY, lw=.6))
        ax.set_xlabel("shrinkage $r$"); ax.set_ylabel("AUROC")
        ax.legend(frameon=False, fontsize=9.5, loc="lower right",
                  bbox_to_anchor=(1.0, 0.0))
        if rt is not None:
            mm = rt[rt.meaning.str.contains("matched static", case=False, na=False)]
            un = rt[rt.meaning.str.contains("unridged static", case=False, na=False)]
            bits = []
            if len(un):
                bits.append(f"vs unregularised: {un.iloc[0].difference:+.4f} "
                            f"(p = {un.iloc[0].delong_p:.3f})")
            if len(mm):
                bits.append(f"vs matched: {mm.iloc[0].difference:+.4f} "
                            f"(p = {mm.iloc[0].delong_p:.3f})")
            if bits:
                ax.set_xlabel("shrinkage $r$\nwindowed " + "\n".join(bits), fontsize=10)
    panel(ax, "b", "Shrinkage and the windowed representation")

    ax = axes[1, 0]
    if lk is not None and sel is not None:
        hon = float(lk.set_index("variant").loc["honest", "auroc"])
        fs = float(lk.set_index("variant").loc["fs_outside", "auroc"])
        ks, diffs = [], []
        for _, r in sel.iterrows():
            if "chosen from everyone" in r.condition:
                k = r.condition.split()[1]
                other = sel[sel.condition == f"top {k} (chosen in-fold)"]
                if len(other):
                    ks.append(int(k)); diffs.append(r.auroc - other.iloc[0].auroc)
        if ks:
            ax.bar(range(len(ks)), diffs, .55, color=TD, alpha=.85)
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([f"top {k}\nof 196\nregions" for k in ks], fontsize=8.5)
        ax.bar([len(ks) + .6], [fs - hon], .55, color=ASD, alpha=.9)
        ax.text(len(ks) + .6, (fs - hon) * 1.02, f"+{fs-hon:.3f}", ha="center",
                va="bottom", fontsize=10, color=ASD, fontweight="bold")
        ax.set_ylim(min(diffs + [0]) * 1.6 if diffs else -0.02, (fs - hon) * 1.16)
        ax.set_xticks(list(range(len(ks))) + [len(ks) + .6])
        ax.set_xticklabels([f"top {k}\nregions" for k in ks] +
                           ["2,000\nconnections\nof 19,900"], fontsize=8.5)
        ax.axhline(0, color="k", lw=.8)
        ax.set_ylabel("AUROC inflation from selecting\noutside the training partition", fontsize=10.5)
    panel(ax, "c", "Selection outside the training partition")

    ax = axes[1, 1]
    if lc:
        obs = 0.707
        ax.axvspan(0.85, 0.99, color=ACC, alpha=.16, lw=0)
        ax.axvline(obs, color=ASD, lw=1.6)
        ax.text(obs, .78, f"  observed {obs:.3f}", color=ASD, fontsize=10, va="center")
        ax.text(.92, .30, "attainable against\nthese labels\n[0.85, 0.99]", ha="center",
                fontsize=9.5, color="#1F6F63")
        ax.annotate("", xy=(0.85, .55), xytext=(obs, .55),
                    arrowprops=dict(arrowstyle="<->", color=GREY, lw=.9))
        ax.text((obs + .85) / 2, .60, "gap $\\geq$ 0.14", ha="center", fontsize=9.5, color=GREY)
        ax.set_xlim(.6, 1.0); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_xlabel("accuracy")
        ax.text(.02, .04, f"instruments disagree in "
                          f"{lc['instrument_disagreement']*100:.1f}% of "
                          f"{lc['n_dual_instrument']} cases; $\\kappa$ = -0.078",
                transform=ax.transAxes, fontsize=8.5, color=GREY)
    panel(ax, "d", "Observed and attainable accuracy")

    fig.savefig(out / "figS_corrections.png"); plt.close(fig)
    return "figS_corrections.png"


def fig_brain(t, out):
    """The group difference, and what motion matching does to it."""
    nc, ncm = load(t, "network_contrast.csv"), load(t, "network_contrast_motion_matched.csv")
    sm, smm = load(t, "sensorimotor_regions.csv"), load(t, "sensorimotor_regions_matched.csv")
    sg = load(t, "subgroup_auroc.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2), layout="constrained")

    # one colour scale across both matrices, or the matched panel looks stronger merely
    # because it is scaled differently
    vshared = max([np.abs(x.effect).max() for x in (nc, ncm) if x is not None] or [1.0])

    def mat(ax, d, title, letter):
        if d is None:
            return
        # canonical system order and the codes defined in Figure 1, as in the other figures
        order = ["visual", "sensorimotor", "dorsal attention", "salience / ventral attention",
                 "limbic", "frontoparietal control", "default mode", "subcortical / cerebellar"]
        codes = dict(zip(order, ["VIS", "SMN", "DAN", "SAL", "LIM", "FPN", "DMN", "SUB"]))
        present = set(d.system_a) | set(d.system_b)
        sysz = [s for s in order if s in present]
        M = np.full((len(sysz), len(sysz)), np.nan)
        for _, r in d.iterrows():
            i, j = sysz.index(r.system_a), sysz.index(r.system_b)
            M[i, j] = M[j, i] = r.effect
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vshared, vmax=vshared)
        for _, r in d.iterrows():
            if r.q_fdr < .05:
                i, j = sysz.index(r.system_a), sysz.index(r.system_b)
                ax.text(j, i, "*", ha="center", va="center", fontsize=10.5)
                ax.text(i, j, "*", ha="center", va="center", fontsize=10.5)
        short = [codes[s] for s in sysz]
        ax.set_xticks(range(len(sysz))); ax.set_xticklabels(short, fontsize=9)
        ax.set_yticks(range(len(sysz))); ax.set_yticklabels(short, fontsize=8.5)
        ax.grid(False)
        n_sig = int((d.q_fdr < .05).sum())
        ax.set_xlabel(f"{n_sig} of {len(d)} pairs significant", fontsize=9.5)
        panel(ax, letter, title)
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03).set_label(
            "standardised difference", fontsize=9.5)

    mat(axes[0, 0], nc, "All participants", "a")
    mat(axes[0, 1], ncm, "Motion-matched pairs", "b")

    ax = axes[1, 0]
    if sm is not None and smm is not None:
        m = sm[["roi_id", "region", "effect"]].merge(
            smm[["roi_id", "effect"]], on="roi_id", suffixes=("_un", "_mat"))
        ax.scatter(m.effect_un, m.effect_mat, s=18, color=ASD, alpha=.85, zorder=3)
        lo = min(m.effect_un.min(), m.effect_mat.min()) - .02
        hi = max(m.effect_un.max(), m.effect_mat.max()) + .02
        ax.plot([lo, hi], [lo, hi], ls=":", color=GREY, lw=.9)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("effect, all participants")
        ax.set_ylabel("effect, motion-matched")
        from scipy.stats import spearmanr
        rho = spearmanr(m.effect_un, m.effect_mat).statistic
        ax.text(.04, .93, f"21 of 21 significant in both\n"
                          f"rank agreement $\\rho$ = {rho:.2f}\n"
                          f"mean {m.effect_un.mean():.3f} $\\to$ {m.effect_mat.mean():.3f}",
                transform=ax.transAxes, fontsize=9.5, va="top")
        ax.text(.5, .04, "below the diagonal:\nstronger after matching",
                transform=ax.transAxes, fontsize=8.5, color=GREY, ha="center")
    panel(ax, "c", "Somatomotor regions under motion matching")

    ax = axes[1, 1]
    if sg is not None:
        order = ["FIQ < 85", "FIQ 85-99", "FIQ 100-114", "FIQ >= 115"]
        band = sg[sg.variable.str.contains("FIQ band")]
        for proto, col, nm in [("pooled", TD, "pooled"), ("loso", ASD, "site withheld")]:
            d = band[band.protocol == proto].set_index("group").reindex(order).dropna(subset=["auroc"])
            if not len(d):
                continue
            xs = np.arange(len(d))
            ax.errorbar(xs, d.auroc, yerr=[d.auroc - d.auroc_lo, d.auroc_hi - d.auroc],
                        fmt="o-", ms=5, lw=1.4, capsize=3, color=col, label=nm)
        d0 = band[band.protocol == "pooled"].set_index("group").reindex(order)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace("FIQ ", "").replace(">= ", "\u2265")
                            + f"\nn={int(d0.loc[o,'n'])}" if pd.notna(d0.loc[o, "n"]) else o
                            for o in order], fontsize=9.5)
        ax.set_ylabel("AUROC"); ax.axhline(.5, color=GREY, ls=":", lw=.8)
        ax.legend(frameon=False, fontsize=9.5, loc="lower left")
        ax.set_xlabel("autistic participants by full-scale IQ;\ncomparison group held fixed "
                      "at all 468 non-autistic participants", fontsize=10)
    panel(ax, "d", "Discrimination by full-scale IQ band")

    fig.savefig(out / "figS_brain.png"); plt.close(fig)
    return "figS_brain.png"


def fig_guarantee(raw, out):
    """A bound reported on its own calibration data, against the same bound tested honestly."""
    p = raw / "conformal_ensemble.json"
    if not p.exists():
        return None
    conf = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), layout="constrained")

    modes = [m for m in ("marginal", "mondrian") if m in conf]
    names = {"marginal": "one threshold\nfor the whole cohort",
             "mondrian": "a threshold\nper site"}

    ax = axes[0]
    xs, w = np.arange(len(modes)), .34
    insample = [conf[m]["risk_bound"] for m in modes]
    honest = [conf[m].get("held_out", {}).get("held_out_risk_mean", np.nan) for m in modes]
    alpha = conf[modes[0]]["alpha"]
    ax.bar(xs - w / 2, insample, w, color=GREY, alpha=.75,
           label="bound reported on the data that set it")
    ax.bar(xs + w / 2, honest, w, color=TD, alpha=.9,
           label="error on held-out cases")
    ax.axhline(alpha, color=ASD, ls="--", lw=1.2)
    ax.text(.98, alpha + .04, f"target $\\alpha$ = {alpha:g}", transform=ax.get_yaxis_transform(),
            color=ASD, fontsize=9.5, ha="right")
    for i, v in enumerate(insample):
        ax.text(xs[i] - w / 2, v + .035, f"{v:.2f}", ha="center", fontsize=9.5, color="#555")
    for i, v in enumerate(honest):
        if np.isfinite(v):
            ax.text(xs[i] + w / 2, v + .035, f"{v:.2f}", ha="center", fontsize=9.5, color=TD)
    ax.set_xticks(xs); ax.set_xticklabels([names[m] for m in modes], fontsize=10)
    ax.set_ylabel("error among answered cases", fontsize=10.5)
    ax.set_ylim(0, 1.22)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=1)
    panel(ax, "a", "Reported bound and held-out error")

    ax = axes[1]
    met = [conf[m].get("held_out", {}).get("fraction_target_met", np.nan) * 100 for m in modes]
    cov = [conf[m]["coverage"] * 100 for m in modes]
    ax.bar(xs - w / 2, cov, w, color=ACC, alpha=.9, label="cases answered")
    ax.bar(xs + w / 2, met, w, color=TD, alpha=.9, label="splits meeting the target")
    ax.axhline(90, color=ASD, ls="--", lw=1.2)
    ax.text(len(modes) - .45, 91.5, "nominal 90%", color=ASD, fontsize=9.5, ha="right")
    for i, v in enumerate(cov):
        ax.text(xs[i] - w / 2, v + 1.5, "answers\nnothing" if v < 1 else f"{v:.0f}%",
                ha="center", fontsize=9, color=(ASD if v < 1 else "black"))
    for i, v in enumerate(met):
        if np.isfinite(v):
            ax.text(xs[i] + w / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=TD)
    ax.set_xticks(xs); ax.set_xticklabels([names[m] for m in modes], fontsize=10)
    ax.set_ylabel("per cent", fontsize=10.5); ax.set_ylim(0, 118)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    panel(ax, "b", "Cases answered and target met")

    fig.savefig(out / "figS_guarantee.png"); plt.close(fig)
    return "figS_guarantee.png"


def main() -> None:
    ap = argparse.ArgumentParser(description="Composite summary figures")
    ap.add_argument("--study", required=True)
    ap.add_argument("--out", default=None,
                    help="output directory (default: <study>/figures)")
    args = ap.parse_args()
    style()
    study = Path(args.study).resolve()
    t, raw = study / "tables", study / "raw"
    out = Path(args.out).resolve() if args.out else study / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = [x for x in (fig_performance(t, raw, out), fig_guarantee(raw, out),
                        fig_corrections(t, raw, out), fig_brain(t, out)) if x]
    for m in made:
        print("->", out / m)


if __name__ == "__main__":
    main()
