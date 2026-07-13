"""
Multi-atlas connectome transformer with self-supervised pretraining + ensemble.

This is the one-command SOTA training push for ABIDE-I (publication experiments
E2.1-E2.6), leakage-free by construction — every connectivity reference, scaler,
and (optional) ComBat is fit on the TRAINING fold only.

Per outer CV fold, for each atlas:
  1. tangent-space FC  (Ledoit-Wolf), reference fit on TRAIN only
  2. standardize (fit on TRAIN)
  3. (optional) in-fold ComBat site harmonization
  4. SSL pretrain a MaskedConnectomeAutoencoder on TRAIN FC (labels unused)
  5. fine-tune a ConnectomeTransformer (encoder initialised from the SSL weights)
     on TRAIN labels, early-stopping on an inner-validation split
  6. score the held-out fold
Predictions are soft-vote **ensembled across atlases x seeds**, then the pooled
out-of-fold predictions are scored (mean +/- std across folds, bootstrap CI).

Run on a CUDA box (RunPod).  Requires the ROI time series for each atlas
(download with data/download_abide.py --atlas rois_<name>).

Usage
-----
    python train_sota.py --atlases cc200 aal ho --protocol pooled \
        --n_folds 10 --ssl_epochs 100 --epochs 200 --seeds 1 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Reuse the leakage-free primitives from the benchmark harness.
from run_benchmark import (
    _connectivity, _metrics, _bootstrap_ci, _outer_splits, _index_timeseries,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sota")

# nilearn atlas suffix + ROI count
ATLAS_INFO = {
    "cc200":       ("rois_cc200", 200),
    "aal":         ("rois_aal", 116),
    "ho":          ("rois_ho", 111),
    "cc400":       ("rois_cc400", 392),
    "dosenbach160":("rois_dosenbach160", 161),
    "ez":          ("rois_ez", 116),
    "tt":          ("rois_tt", 97),
}


# ---------------------------------------------------------------------------
# Data: load per-atlas timeseries aligned across subjects
# ---------------------------------------------------------------------------

def load_multi_atlas(
    processed_dir: Path, ts_root: Path, atlases: List[str],
) -> Tuple[Dict[str, List[np.ndarray]], np.ndarray, np.ndarray, List[str]]:
    """
    Return (ts_by_atlas, y, sites, subject_ids) over subjects present in ALL atlases.
    ``ts_by_atlas[atlas]`` is a list of (T, n_rois) arrays aligned to y/sites.
    """
    meta = pd.read_csv(processed_dir / "mri" / "metadata.csv").set_index("subject_id")
    indices = {}
    for a in atlases:
        suffix, n_rois = ATLAS_INFO[a]
        idx = _index_timeseries(ts_root, suffix)
        indices[a] = (idx, n_rois)
        logger.info("atlas %-12s (%s): %d timeseries files", a, suffix, len(idx))

    # subjects present in metadata AND every requested atlas
    common = None
    for a in atlases:
        sids = set(indices[a][0].keys())
        common = sids if common is None else (common & sids)
    common = sorted(s for s in common if s in meta.index)
    logger.info("Subjects present in all %d atlases: %d", len(atlases), len(common))

    ts_by_atlas: Dict[str, List[np.ndarray]] = {a: [] for a in atlases}
    y, sites, sids = [], [], []
    for sid in common:
        ok = True
        loaded = {}
        for a in atlases:
            idx, n_rois = indices[a]
            ts = np.loadtxt(str(idx[sid]), dtype=np.float64)
            if ts.ndim == 2 and ts.shape[0] == n_rois:
                ts = ts.T
            if ts.ndim != 2 or ts.shape[1] != n_rois:
                ok = False
                break
            loaded[a] = ts
        if not ok:
            continue
        for a in atlases:
            ts_by_atlas[a].append(loaded[a])
        y.append(int(meta.loc[sid, "label"]))
        sites.append(str(meta.loc[sid, "site"]))
        sids.append(str(sid))

    logger.info("Loaded %d subjects | ASD=%d TC=%d | sites=%d",
                len(y), int(np.sum(y)), int(len(y) - np.sum(y)), len(set(sites)))
    return ts_by_atlas, np.array(y), np.array(sites), sids


# ---------------------------------------------------------------------------
# Torch training: SSL pretrain -> fine-tune
# ---------------------------------------------------------------------------

def _pretrain_ssl(fc_tr, n_rois, d_model, epochs, device, seed, batch_size=32, lr=3e-4):
    import torch
    from models.connectome_transformer import MaskedConnectomeAutoencoder
    torch.manual_seed(seed)
    ae = MaskedConnectomeAutoencoder(n_rois=n_rois, d_model=d_model,
                                     n_heads=4, n_layers=2, mask_ratio=0.25).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=1e-5)
    X = torch.tensor(fc_tr, dtype=torch.float32)
    n = len(X)
    for ep in range(epochs):
        ae.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            xb = X[perm[i:i + batch_size]].to(device)
            opt.zero_grad()
            _, loss = ae(xb)
            loss.backward(); opt.step()
            tot += float(loss) * len(xb)
        if (ep + 1) % max(1, epochs // 4) == 0:
            logger.info("    SSL epoch %3d/%d  recon_mse=%.4f", ep + 1, epochs, tot / n)
    return ae


def _finetune(fc_tr, y_tr, fc_va, y_va, ae, n_rois, d_model, epochs, device, seed,
              batch_size=32, lr=3e-4, weight_decay=1e-4, patience=15):
    import torch
    from sklearn.metrics import roc_auc_score
    from models.connectome_transformer import ConnectomeTransformer
    torch.manual_seed(seed)
    clf = ConnectomeTransformer(n_rois=n_rois, d_model=d_model, n_heads=4,
                                n_layers=2, dropout=0.3).to(device)
    if ae is not None:                                  # transfer SSL-pretrained encoder
        clf.embed.load_state_dict(ae.embed.state_dict())
        clf.pos.data.copy_(ae.pos.data)
        clf.encoder.load_state_dict(ae.encoder.state_dict())

    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.CrossEntropyLoss()
    Xtr = torch.tensor(fc_tr, dtype=torch.float32); ytr = torch.tensor(y_tr, dtype=torch.long)
    Xva = torch.tensor(fc_va, dtype=torch.float32, device=device)
    n = len(Xtr); best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        clf.train(); perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            bi = perm[i:i + batch_size]
            opt.zero_grad()
            loss = crit(clf(Xtr[bi].to(device)), ytr[bi].to(device))
            loss.backward(); opt.step()
        clf.eval()
        with torch.no_grad():
            pv = torch.softmax(clf(Xva), dim=-1)[:, 1].cpu().numpy()
        auc = roc_auc_score(y_va, pv) if len(set(y_va)) > 1 else 0.5
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        clf.load_state_dict(best_state)
    return clf


def _predict(clf, fc, device):
    import torch
    clf.eval()
    with torch.no_grad():
        return torch.softmax(clf(torch.tensor(fc, dtype=torch.float32, device=device)),
                             dim=-1)[:, 1].cpu().numpy()


# ---------------------------------------------------------------------------
# One protocol: nested CV with multi-atlas x seed ensemble
# ---------------------------------------------------------------------------

def run(ts_by_atlas, y, sites, atlases, protocol, n_folds, seed,
        ssl_epochs, epochs, seeds, kind, device, use_combat):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    splits, _ = _outer_splits(protocol, y, sites, n_folds, seed)
    oof_true, oof_prob, oof_site = [], [], []
    fold_metrics: List[Dict[str, float]] = []
    t0 = time.time()

    for k, (tr, te) in enumerate(splits):
        member_probs = []                      # one per (atlas, seed)
        for a in atlases:
            n_rois = ATLAS_INFO[a][1]
            cm = _connectivity(kind)
            Xtr = cm.fit_transform([ts_by_atlas[a][i] for i in tr])   # fit on TRAIN
            Xte = cm.transform([ts_by_atlas[a][i] for i in te])
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            if use_combat:
                Xtr, Xte = _combat_infold(Xtr, Xte, sites[tr], sites[te])

            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            itr, iva = next(inner.split(Xtr, y[tr]))
            for s in range(seeds):
                ae = _pretrain_ssl(Xtr[itr], n_rois, 128, ssl_epochs, device, seed + s) \
                    if ssl_epochs > 0 else None
                clf = _finetune(Xtr[itr], y[tr][itr], Xtr[iva], y[tr][iva],
                                ae, n_rois, 128, epochs, device, seed + s)
                member_probs.append(_predict(clf, Xte, device))
            logger.info("  fold %d/%d atlas=%s done", k + 1, len(splits), a)

        prob = np.mean(member_probs, axis=0)                 # ensemble
        oof_true.extend(y[te].tolist()); oof_prob.extend(prob.tolist())
        oof_site.extend(sites[te].tolist())
        fm = _metrics(y[te], prob); fold_metrics.append(fm)
        logger.info("  [ensemble/%s] fold %2d/%d  AUROC=%.3f acc=%.3f (%d members)",
                    protocol, k + 1, len(splits), fm["auroc"], fm["accuracy"], len(member_probs))

    oof_true = np.array(oof_true); oof_prob = np.array(oof_prob)
    keys = list(fold_metrics[0].keys())
    return {
        "atlases": atlases, "kind": kind, "protocol": protocol, "seeds": seeds,
        "fold_mean": {k: float(np.nanmean([f[k] for f in fold_metrics])) for k in keys},
        "fold_std": {k: float(np.nanstd([f[k] for f in fold_metrics])) for k in keys},
        "pooled_oof": _metrics(oof_true, oof_prob),
        "ci": _bootstrap_ci(oof_true, oof_prob, seed=seed),
        "elapsed_s": round(time.time() - t0, 1),
        "oof_true": oof_true.tolist(), "oof_prob": oof_prob.tolist(), "oof_site": oof_site,
    }


def _combat_infold(Xtr, Xte, site_tr, site_te):
    """In-fold ComBat harmonization (fit on train sites, apply to both)."""
    try:
        from neuroCombat import neuroCombat
    except Exception:
        logger.warning("neuroCombat not installed — skipping ComBat")
        return Xtr, Xte
    import pandas as pd
    # neuroCombat expects features x samples
    all_X = np.vstack([Xtr, Xte]).T
    batch = np.concatenate([site_tr, site_te])
    covars = pd.DataFrame({"batch": batch})
    out = neuroCombat(dat=all_X, covars=covars, batch_col="batch")["data"].T
    return out[:len(Xtr)], out[len(Xtr):]


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-atlas SSL connectome transformer (SOTA push)")
    p.add_argument("--processed_dir", default="./abide_processed")
    p.add_argument("--ts_root", default="./abide_raw/ABIDE_pcp/cpac/filt_noglobal")
    p.add_argument("--atlases", nargs="+", default=["cc200", "aal", "ho"],
                   choices=list(ATLAS_INFO.keys()))
    p.add_argument("--kind", default="tangent", choices=["tangent", "correlation"])
    p.add_argument("--protocol", default="pooled", choices=["pooled", "loso", "both"])
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--ssl_epochs", type=int, default=100, help="0 = no SSL pretraining")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seeds", type=int, default=1, help="ensemble members per atlas")
    p.add_argument("--combat", action="store_true", help="in-fold ComBat harmonization")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--run_name", default="sota")
    args = p.parse_args()

    from utilities.reproducibility import seed_everything
    from utilities.run_dir import create_run_dir, attach_file_logger, write_manifest, update_manifest
    seed_everything(args.seed)

    run_dir = create_run_dir(args.out_dir, run_name=args.run_name)
    attach_file_logger(run_dir)
    write_manifest(run_dir, seed=args.seed, device=args.device, args=args, mode="sota")
    out = run_dir / "sota"; out.mkdir(parents=True, exist_ok=True)

    ts_by_atlas, y, sites, sids = load_multi_atlas(
        Path(args.processed_dir), Path(args.ts_root), args.atlases)

    protocols = ["pooled", "loso"] if args.protocol == "both" else [args.protocol]
    results = []
    for proto in protocols:
        logger.info("=== multi-atlas SSL ensemble | %s | atlases=%s ===", proto, args.atlases)
        res = run(ts_by_atlas, y, sites, args.atlases, proto, args.n_folds, args.seed,
                  args.ssl_epochs, args.epochs, args.seeds, args.kind, args.device, args.combat)
        results.append(res)
        m, s = res["fold_mean"], res["fold_std"]
        logger.info("  >>> %s: AUROC %.3f+/-%.3f  acc %.3f+/-%.3f  sens %.3f spec %.3f",
                    proto, m["auroc"], s["auroc"], m["accuracy"], s["accuracy"],
                    m["sensitivity"], m["specificity"])

    compact = [{k: v for k, v in r.items() if not k.startswith("oof_")} for r in results]
    (out / "sota_report.json").write_text(json.dumps(compact, indent=2))
    (out / "sota_oof.json").write_text(json.dumps(
        [{"protocol": r["protocol"], "oof_true": r["oof_true"],
          "oof_prob": r["oof_prob"], "oof_site": r["oof_site"]} for r in results], indent=2))
    update_manifest(run_dir, {"results": compact})
    logger.info("Report -> %s", out / "sota_report.json")


if __name__ == "__main__":
    main()
