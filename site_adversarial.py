"""
C1 — Learned site-adversarial harmonization (publication contribution).

Multi-site fMRI models can cheat by learning *site* instead of biology, which is
why pooled accuracy collapses under leave-one-site-out.  Instead of harmonizing
with ComBat as a preprocessing step, we make the learned representation
**site-invariant but diagnosis-preserving, end-to-end**: a site-discriminator is
attached to the shared features through a **gradient-reversal layer** (Ganin &
Lempitsky 2015), so the encoder is pushed to produce features from which site
cannot be decoded, while the diagnosis head still separates ASD/TC.

Status (honest, from ``validate_site_adversarial``):
  - The gradient-reversal MECHANISM is implemented correctly (``grl_unit_test``
    passes: the layer negates the input gradient).
  - BUT on synthetic site-confounded data, naive adversarial invariance is
    *finicky*: it reduces linear site-decodability only modestly and does NOT
    reliably improve unseen-site generalization when a site has a spurious
    label-shortcut (marginal-distribution alignment cannot remove conditional
    shortcuts — a well-known DANN limitation).
  => Do NOT assume C1 beats ComBat.  Its real value must be measured on ABIDE
     leave-one-site-out; the honest fallback is "learned harmonization ≈ ComBat".
     A more stable variant (IRM / conditional alignment) is a future upgrade.

This module provides the reusable pieces (``grad_reverse``, ``SiteAdversary``)
plus the validation.  Integration into the connectome-transformer training is a
thin add-on: add ``lambda_adv * CE(site_logits, GRL(features))`` to the loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambd * grad_out, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class SiteAdversary(nn.Module):
    """Site classifier applied through a gradient-reversal layer.

    forward(features, lambd) returns site logits; because of the reversal, the
    encoder receives *negated* gradients and thus learns site-invariant features.
    """

    def __init__(self, feat_dim: int, n_sites: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(hidden, n_sites),
            )
        else:                                   # linear adversary -> enforces
            self.net = nn.Linear(feat_dim, n_sites)   # *linear* site-invariance

    def forward(self, features: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(features, lambd))


def adv_lambda_schedule(step: int, total: int, gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """Ganin's schedule: ramp lambda 0->max over training so the adversary does
    not destabilise early optimisation."""
    p = min(1.0, step / max(1, total))
    return float(max_lambda * (2.0 / (1.0 + np.exp(-gamma * p)) - 1.0))


# ---------------------------------------------------------------------------
# Mechanism validation on synthetic site-confounded data
# ---------------------------------------------------------------------------

def grl_unit_test() -> bool:
    """The gradient-reversal layer must negate the gradient w.r.t. its input."""
    x = torch.ones(3, requires_grad=True)
    grad_reverse(x, lambd=2.0).sum().backward()
    ok = torch.allclose(x.grad, torch.full((3,), -2.0))
    print(f"GRL unit test: grad={x.grad.tolist()} (expect -2)  ->  {'OK' if ok else 'FAIL'}")
    return ok


def validate_site_adversarial(device: str = "cpu", seed: int = 0) -> bool:
    """
    Proper test of what C1 is *for* — cross-site generalization (the LOSO gap).

    Each site has the true diagnosis signal PLUS a per-site **shortcut** feature
    that spuriously equals the label in the training sites but is random in a
    held-out site.  A naive model exploits the shortcut and fails on the unseen
    site.  A domain-adversarial model (using the *unlabeled* held-out site to
    align feature distributions) is pushed to ignore the site-specific shortcut
    and rely on the real signal — so it should GENERALIZE better to the unseen
    site.  Metric: held-out-site AUROC.
    """
    from sklearn.metrics import roc_auc_score

    ok_grl = grl_unit_test()
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    dev = torch.device(device)
    n_per, n_sites, d = 300, 6, 30
    holdout = n_sites - 1

    Xs, ys, ds = [], [], []
    for s in range(n_sites):
        yy = rng.integers(0, 2, n_per)
        bio = yy[:, None] * 1.0 + rng.normal(0, 1.0, (n_per, 5))          # real signal
        sc_lbl = rng.integers(0, 2, n_per) if s == holdout else yy       # shortcut
        shortcut = sc_lbl[:, None] * 2.0 * np.ones((1, 5)) + rng.normal(0, 0.3, (n_per, 5))
        X = np.concatenate([bio, shortcut, rng.normal(0, 1, (n_per, d - 10))], 1)
        Xs.append(X); ys.append(yy); ds.append(np.full(n_per, int(s == holdout)))
    X = np.vstack(Xs).astype(np.float32); y = np.concatenate(ys); dom = np.concatenate(ds)
    tr = dom == 0; ho = dom == 1                                          # labeled train / unlabeled holdout

    Xt = torch.tensor(X, device=dev); yt = torch.tensor(y, device=dev)
    dt = torch.tensor(dom, device=dev)

    def train(adversarial: bool):
        torch.manual_seed(seed)
        enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16)).to(dev)
        clf = nn.Linear(16, 2).to(dev)
        adv = SiteAdversary(16, 2, hidden=32).to(dev)          # domain (train vs holdout)
        opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters())
                               + list(adv.parameters()), lr=1e-3, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()
        E = 300
        tr_i = torch.tensor(np.where(tr)[0], device=dev)
        for ep in range(E):
            enc.train()
            loss = ce(clf(enc(Xt[tr_i])), yt[tr_i])            # diagnosis on labelled train
            if adversarial:
                lam = adv_lambda_schedule(ep, E, max_lambda=1.0)
                loss = loss + ce(adv(enc(Xt), lam), dt)        # align train vs holdout (all X)
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval()
        with torch.no_grad():
            prob = torch.softmax(clf(enc(Xt)), -1)[:, 1].cpu().numpy()
        return roc_auc_score(y[tr], prob[tr]), roc_auc_score(y[ho], prob[ho])

    base_tr, base_ho = train(False)
    adv_tr, adv_ho = train(True)
    print(f"{'':14s}{'train-site AUROC':>18s}{'HELD-OUT-site AUROC':>22s}")
    print(f"{'baseline':14s}{base_tr:>18.3f}{base_ho:>22.3f}")
    print(f"{'+ adversary':14s}{adv_tr:>18.3f}{adv_ho:>22.3f}")
    ok = ok_grl and (adv_ho > base_ho + 0.05)
    print(f"held-out generalization: {base_ho:.3f} -> {adv_ho:.3f}")
    print("VALIDATION:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    validate_site_adversarial()
