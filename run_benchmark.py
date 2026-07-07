"""
Leakage-free connectivity benchmark for ABIDE-I ASD classification.

This is the honest reference harness for the publication (experiments E1.x):
it computes functional connectivity **inside each CV fold** (no transform ever
sees validation/test data) and reports both the pooled and leave-one-site-out
numbers with confidence intervals.

Connectivity metrics
--------------------
- ``correlation``  : Pearson correlation (Fisher-z), the naive baseline.
- ``tangent``      : Ledoit-Wolf covariance projected to the Riemannian tangent
                     space at the *training-fold* group mean — the best-evidenced
                     FC parametrization (Dadi et al. 2019).  The tangent reference
                     is fit on train subjects only, then applied to val/test.

Classifier: ℓ2-regularized logistic regression, with the inverse-regularization
strength ``C`` tuned by an **inner** cross-validation (nested CV).

Protocols
---------
- ``pooled`` : subject-independent stratified K-fold (headline protocol; matches
               Heinsfeld 2018 / METAFormer).
- ``loso``   : leave-one-site-out (generalization to unseen scanners).

Usage
-----
    python run_benchmark.py --protocol both --metrics correlation tangent \
        --n_folds 10 --n_perm 0

Outputs a JSON report + console summary under ``results/run_N/benchmark/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

N_ROIS = 200  # CC200


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _index_timeseries(ts_dir: Path, atlas: str = "rois_cc200") -> Dict[int, Path]:
    """Map integer SUB_ID -> ROI-timeseries .1D path (site prefixes vary)."""
    pat = re.compile(r"(\d+)_%s\.1D$" % re.escape(atlas))
    out: Dict[int, Path] = {}
    for p in ts_dir.rglob(f"*_{atlas}.1D"):
        m = pat.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def load_dataset(
    processed_dir: Path,
    ts_dir: Path,
    atlas: str = "rois_cc200",
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    """
    Return (timeseries_list, y, site_ids, subject_ids) aligned by subject.

    Each timeseries is (T, N_ROIS).  Uses the metadata.csv produced by
    data/preprocess_abide.py for labels/sites, and the raw .1D files for the
    time series (needed for tangent-space FC).
    """
    meta = pd.read_csv(processed_dir / "mri" / "metadata.csv")
    ts_index = _index_timeseries(ts_dir, atlas)

    ts_list: List[np.ndarray] = []
    y: List[int] = []
    sites: List[str] = []
    sids: List[str] = []
    n_missing = 0

    for _, row in meta.iterrows():
        sid = int(row["subject_id"])
        path = ts_index.get(sid)
        if path is None:
            n_missing += 1
            continue
        ts = np.loadtxt(str(path), dtype=np.float64)
        if ts.ndim != 2 or ts.shape[1] != N_ROIS:
            # some files transpose or include a header column; coerce/skip
            if ts.ndim == 2 and ts.shape[0] == N_ROIS:
                ts = ts.T
            else:
                n_missing += 1
                continue
        ts_list.append(ts)
        y.append(int(row["label"]))
        sites.append(str(row["site"]))
        sids.append(str(sid))

    if n_missing:
        logger.warning("%d subjects had no usable timeseries and were skipped", n_missing)
    logger.info("Loaded %d subjects | ASD=%d TC=%d | sites=%d",
                len(y), int(np.sum(y)), int(len(y) - np.sum(y)), len(set(sites)))
    return ts_list, np.array(y), np.array(sites), sids


# ---------------------------------------------------------------------------
# Connectivity (fit-on-train)
# ---------------------------------------------------------------------------

def _connectivity(kind: str):
    """Build a fresh nilearn ConnectivityMeasure (fit on train, transform others)."""
    from nilearn.connectome import ConnectivityMeasure
    from sklearn.covariance import LedoitWolf
    return ConnectivityMeasure(
        kind=kind,                       # 'correlation' | 'tangent' | 'partial correlation'
        cov_estimator=LedoitWolf(store_precision=False),
        vectorize=True,
        discard_diagonal=True,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, matthews_corrcoef,
        balanced_accuracy_score, average_precision_score,
    )
    y_pred = (y_prob >= thr).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": sens,
        "specificity": spec,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
    }


def _bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n: int = 1000,
                  seed: int = 42) -> Dict[str, Tuple[float, float]]:
    from sklearn.metrics import roc_auc_score, accuracy_score
    rng = np.random.default_rng(seed)
    aucs, accs = [], []
    idx = np.arange(len(y_true))
    y_pred = (y_prob >= 0.5).astype(int)
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y_true[b])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[b], y_prob[b]))
        accs.append(accuracy_score(y_true[b], y_pred[b]))
    def ci(a):
        return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a else (float("nan"),) * 2
    return {"auroc": ci(aucs), "accuracy": ci(accs)}


# ---------------------------------------------------------------------------
# Nested CV over one protocol
# ---------------------------------------------------------------------------

def _outer_splits(protocol: str, y: np.ndarray, sites: np.ndarray,
                  n_folds: int, seed: int):
    from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
    idx = np.arange(len(y))
    if protocol == "pooled":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(skf.split(idx, y)), None
    elif protocol == "loso":
        logo = LeaveOneGroupOut()
        groups = sites
        return list(logo.split(idx, y, groups=groups)), groups
    raise ValueError(protocol)


def run_protocol(
    ts_list: List[np.ndarray],
    y: np.ndarray,
    sites: np.ndarray,
    kind: str,
    protocol: str,
    n_folds: int,
    seed: int,
    inner_folds: int = 3,
    C_grid: Optional[List[float]] = None,
) -> Dict:
    """Nested CV for one (metric, protocol). Returns metrics + OOF predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    C_grid = C_grid or [0.001, 0.01, 0.1, 1.0]
    splits, _ = _outer_splits(protocol, y, sites, n_folds, seed)

    oof_true, oof_prob, oof_site = [], [], []
    fold_metrics: List[Dict[str, float]] = []
    t0 = time.time()

    for k, (tr, te) in enumerate(splits):
        cm = _connectivity(kind)
        Xtr = cm.fit_transform([ts_list[i] for i in tr])   # reference fit on TRAIN only
        Xte = cm.transform([ts_list[i] for i in te])

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=2000)),
        ])
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
        gs = GridSearchCV(pipe, {"clf__C": C_grid}, scoring="roc_auc", cv=inner, n_jobs=-1)
        gs.fit(Xtr, y[tr])

        prob = gs.predict_proba(Xte)[:, 1]
        oof_true.extend(y[te].tolist())
        oof_prob.extend(prob.tolist())
        oof_site.extend(sites[te].tolist())
        fm = _metrics(y[te], prob)
        fm["best_C"] = float(gs.best_params_["clf__C"])
        fold_metrics.append(fm)
        logger.info("  [%s/%s] fold %2d/%d  AUROC=%.3f acc=%.3f (C=%.3g)",
                    kind, protocol, k + 1, len(splits),
                    fm["auroc"], fm["accuracy"], fm["best_C"])

    oof_true = np.array(oof_true)
    oof_prob = np.array(oof_prob, dtype=np.float64)
    keys = [k for k in fold_metrics[0] if k != "best_C"]
    mean = {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in keys}
    std = {k: float(np.nanstd([f[k] for f in fold_metrics])) for k in keys}
    pooled_oof = _metrics(oof_true, oof_prob)      # metrics on pooled OOF predictions
    ci = _bootstrap_ci(oof_true, oof_prob, seed=seed)

    return {
        "metric": kind, "protocol": protocol, "n_folds": len(splits),
        "fold_mean": mean, "fold_std": std,
        "pooled_oof": pooled_oof, "ci": ci,
        "elapsed_s": round(time.time() - t0, 1),
        "oof_true": oof_true.tolist(), "oof_prob": oof_prob.tolist(),
        "oof_site": oof_site,
    }


def run_protocol_torch(
    ts_list: List[np.ndarray],
    y: np.ndarray,
    sites: np.ndarray,
    kind: str,
    protocol: str,
    n_folds: int,
    seed: int,
    device: str = "cpu",
    epochs: int = 60,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    d_model: int = 128,
    patience: int = 12,
) -> Dict:
    """
    Nested CV for the Connectome Transformer on in-fold tangent/correlation FC.

    Per outer fold: fit the connectivity reference on TRAIN only, standardize on
    TRAIN, carve an inner validation split for early stopping, train the
    transformer, and score the held-out fold.  Leakage-free by construction.
    """
    import torch
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from models.connectome_transformer import ConnectomeTransformer

    dev = torch.device(device)
    splits, _ = _outer_splits(protocol, y, sites, n_folds, seed)
    oof_true, oof_prob, oof_site = [], [], []
    fold_metrics: List[Dict[str, float]] = []
    t0 = time.time()
    n_rois = ts_list[0].shape[1]

    def _train_one(Xtr, ytr, Xva, yva):
        torch.manual_seed(seed)
        model = ConnectomeTransformer(n_rois=n_rois, d_model=d_model,
                                      n_heads=4, n_layers=2, dropout=0.3).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        crit = torch.nn.CrossEntropyLoss()
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.long)
        Xva_t = torch.tensor(Xva, dtype=torch.float32, device=dev)
        from sklearn.metrics import roc_auc_score
        best_auc, best_state, bad = -1.0, None, 0
        n = len(Xtr_t)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                bi = perm[i:i + batch_size]
                xb = Xtr_t[bi].to(dev); yb = ytr_t[bi].to(dev)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pv = torch.softmax(model(Xva_t), dim=-1)[:, 1].cpu().numpy()
            auc = roc_auc_score(yva, pv) if len(set(yva)) > 1 else 0.5
            if auc > best_auc:
                best_auc, best_state, bad = auc, {k: v.detach().cpu().clone()
                                                  for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    for k, (tr, te) in enumerate(splits):
        cm = _connectivity(kind)
        Xtr_full = cm.fit_transform([ts_list[i] for i in tr])
        Xte = cm.transform([ts_list[i] for i in te])
        scaler = StandardScaler().fit(Xtr_full)
        Xtr_full = scaler.transform(Xtr_full); Xte = scaler.transform(Xte)

        # inner split for early stopping
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        itr, iva = next(inner.split(Xtr_full, y[tr]))
        model = _train_one(Xtr_full[itr], y[tr][itr], Xtr_full[iva], y[tr][iva])

        model.eval()
        with torch.no_grad():
            prob = torch.softmax(
                model(torch.tensor(Xte, dtype=torch.float32, device=dev)), dim=-1
            )[:, 1].cpu().numpy()
        oof_true.extend(y[te].tolist()); oof_prob.extend(prob.tolist())
        oof_site.extend(sites[te].tolist())
        fm = _metrics(y[te], prob)
        fold_metrics.append(fm)
        logger.info("  [transformer/%s/%s] fold %2d/%d  AUROC=%.3f acc=%.3f",
                    kind, protocol, k + 1, len(splits), fm["auroc"], fm["accuracy"])

    oof_true = np.array(oof_true); oof_prob = np.array(oof_prob, dtype=np.float64)
    keys = list(fold_metrics[0].keys())
    mean = {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in keys}
    std = {k: float(np.nanstd([f[k] for f in fold_metrics])) for k in keys}
    return {
        "metric": f"transformer+{kind}", "protocol": protocol, "n_folds": len(splits),
        "fold_mean": mean, "fold_std": std,
        "pooled_oof": _metrics(oof_true, oof_prob),
        "ci": _bootstrap_ci(oof_true, oof_prob, seed=seed),
        "elapsed_s": round(time.time() - t0, 1),
        "oof_true": oof_true.tolist(), "oof_prob": oof_prob.tolist(), "oof_site": oof_site,
    }


def permutation_pvalue(
    ts_list, y, sites, kind, protocol, n_folds, seed, n_perm: int,
) -> Optional[float]:
    """Permutation test: fraction of label-shuffled runs whose AUROC >= observed."""
    if n_perm <= 0:
        return None
    logger.info("Permutation test (%d shuffles) for %s/%s …", n_perm, kind, protocol)
    observed = run_protocol(ts_list, y, sites, kind, protocol, n_folds, seed)["pooled_oof"]["auroc"]
    rng = np.random.default_rng(seed)
    ge = 1  # +1 for the observed (standard correction)
    for i in range(n_perm):
        yp = rng.permutation(y)
        auc = run_protocol(ts_list, yp, sites, kind, protocol, n_folds, seed + 1 + i)["pooled_oof"]["auroc"]
        ge += int(auc >= observed)
    return ge / (n_perm + 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Leakage-free ABIDE-I FC benchmark")
    p.add_argument("--processed_dir", default="./abide_processed")
    p.add_argument("--ts_dir", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    p.add_argument("--atlas", default="rois_cc200")
    p.add_argument("--metrics", nargs="+", default=["correlation", "tangent"],
                   choices=["correlation", "tangent", "partial correlation"])
    p.add_argument("--protocol", default="both", choices=["pooled", "loso", "both"])
    p.add_argument("--model", default="logreg", choices=["logreg", "transformer"],
                   help="logreg = linear baseline; transformer = connectome transformer")
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps (transformer only)")
    p.add_argument("--epochs", type=int, default=60, help="transformer epochs/fold")
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--n_perm", type=int, default=0, help="permutation shuffles (0=skip)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--run_name", default=None)
    args = p.parse_args()

    from utilities.reproducibility import seed_everything
    from utilities.run_dir import create_run_dir, attach_file_logger, write_manifest
    seed_everything(args.seed)

    run_dir = create_run_dir(args.out_dir, run_name=args.run_name or "benchmark")
    attach_file_logger(run_dir)
    write_manifest(run_dir, seed=args.seed, device="cpu", args=args, mode="benchmark")
    bench_dir = run_dir / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    ts_list, y, sites, sids = load_dataset(
        Path(args.processed_dir), Path(args.ts_dir), args.atlas)

    protocols = ["pooled", "loso"] if args.protocol == "both" else [args.protocol]
    results = []
    for kind in args.metrics:
        for proto in protocols:
            logger.info("=== %s | %s | %s ===", args.model, kind, proto)
            if args.model == "transformer":
                res = run_protocol_torch(ts_list, y, sites, kind, proto, args.n_folds,
                                         args.seed, device=args.device, epochs=args.epochs)
            else:
                res = run_protocol(ts_list, y, sites, kind, proto, args.n_folds, args.seed)
                res["permutation_p"] = permutation_pvalue(
                    ts_list, y, sites, kind, proto, args.n_folds, args.seed, args.n_perm)
            res.setdefault("permutation_p", None)
            results.append(res)

    # --- Report ---
    table_rows = []
    for r in results:
        m, s, c = r["fold_mean"], r["fold_std"], r["ci"]
        table_rows.append(
            f"{r['metric']:>12s} | {r['protocol']:>6s} | "
            f"AUROC {m['auroc']:.3f}±{s['auroc']:.3f} "
            f"[{c['auroc'][0]:.3f},{c['auroc'][1]:.3f}] | "
            f"acc {m['accuracy']:.3f}±{s['accuracy']:.3f} | "
            f"sens {m['sensitivity']:.3f} spec {m['specificity']:.3f} | "
            f"p={r['permutation_p']}")
    header = f"{'metric':>12s} | {'proto':>6s} | AUROC (mean±std [95% CI]) | accuracy | sens/spec | perm-p"
    summary = "\n".join([header, "-" * len(header)] + table_rows)
    logger.info("\n===== BENCHMARK SUMMARY =====\n%s", summary)

    # strip large OOF arrays from the compact report; keep a full copy separately
    compact = [{k: v for k, v in r.items() if not k.startswith("oof_")} for r in results]
    (bench_dir / "benchmark_report.json").write_text(json.dumps(compact, indent=2))
    (bench_dir / "benchmark_oof.json").write_text(json.dumps(
        [{"metric": r["metric"], "protocol": r["protocol"],
          "oof_true": r["oof_true"], "oof_prob": r["oof_prob"], "oof_site": r["oof_site"]}
         for r in results], indent=2))
    (bench_dir / "benchmark_summary.txt").write_text(summary + "\n")
    logger.info("Report → %s", bench_dir / "benchmark_report.json")


if __name__ == "__main__":
    main()
