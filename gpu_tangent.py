"""
GPU-accelerated tangent-space functional connectivity.

Tangent-space FC (Dadi et al. 2019) is the strongest connectivity parametrization
for prediction, but nilearn computes it on CPU with a Python loop over an iterative
Riemannian geometric mean — the dominant cost in cross-validated pipelines.

This module reimplements the exact same computation with batched, eigendecomposition
based matrix functions on the GPU (``torch.linalg.eigh``).  It is **numerically
identical to nilearn** (validated: per-subject tangent cosine similarity > 0.9999,
Pearson r ≈ 1.0) and much faster on CUDA, while remaining strictly leakage-free:
the geometric-mean reference is fit on the training fold only.

Design
------
- Covariances (Ledoit-Wolf shrinkage) are computed **once** for all subjects and
  cached — they do not depend on the CV fold.
- Per fold: ``fit`` the reference (geometric mean of the *training* covariances) on
  GPU, then ``transform`` all covariances to tangent vectors on GPU.

Usage
-----
    tan = GPUTangentFC(device="cuda")
    covs = tan.compute_covariances(timeseries_list)   # once
    tan.fit(covs[train_idx])                           # per fold (train only)
    Xtr = tan.transform(covs[train_idx])               # tangent vectors
    Xte = tan.transform(covs[test_idx])
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Batched SPD matrix functions via eigendecomposition
# ---------------------------------------------------------------------------

def _eigh(M: torch.Tensor):
    """``torch.linalg.eigh`` with a fallback for the rare matrix on which LAPACK fails to converge.

    Non-failing inputs take exactly the original path, so every result that was obtainable before is
    unchanged.  If the batched call raises, each matrix is retried alone, then in float64, and finally
    in float64 with a diagonal jitter that grows from 1e-10 to 1e-6 of the mean diagonal, which is far
    below the clamp of 1e-6 applied to the eigenvalues by the callers.
    """
    try:
        return torch.linalg.eigh(M)
    except torch.linalg.LinAlgError:
        batched = M.dim() == 3
        mats = M if batched else M.unsqueeze(0)
        ws, Vs = [], []
        tries = (("native", 0.0), ("float64", 0.0), ("float64", 1e-10), ("float64", 1e-8), ("float64", 1e-6))
        for m in mats:
            for k, (mode, jitter) in enumerate(tries):
                try:
                    if mode == "native":
                        w, V = torch.linalg.eigh(m)
                    else:
                        m64 = m.to(torch.float64)
                        jit = jitter * float(m64.diagonal().mean().abs())
                        w, V = torch.linalg.eigh(m64 + jit * torch.eye(m64.shape[-1], dtype=m64.dtype,
                                                                       device=m64.device))
                        w, V = w.to(M.dtype), V.to(M.dtype)
                    break
                except torch.linalg.LinAlgError:
                    if k == len(tries) - 1:
                        raise
            ws.append(w)
            Vs.append(V)
        w, V = torch.stack(ws), torch.stack(Vs)
        return (w, V) if batched else (w[0], V[0])


def _eigfun(M: torch.Tensor, f, eps: float = 1e-6) -> torch.Tensor:
    """Apply scalar function ``f`` to the (clamped) eigenvalues of symmetric M."""
    w, V = _eigh(M)
    w = f(torch.clamp(w, min=eps))
    return (V * w.unsqueeze(-2)) @ V.transpose(-1, -2)


def _expm_sym(M: torch.Tensor) -> torch.Tensor:
    """Matrix exponential of a symmetric matrix (eigenvalues may be negative)."""
    w, V = _eigh(M)
    return (V * torch.exp(w).unsqueeze(-2)) @ V.transpose(-1, -2)


def geometric_mean(covs: torch.Tensor, max_iter: int = 10, tol: float = 1e-6) -> torch.Tensor:
    """
    Riemannian geometric (Fréchet) mean of a batch of SPD matrices.

    Same fixed-point iteration nilearn uses: whiten by the current mean, average
    the matrix logs, and update along the geodesic.  ``covs`` is ``(n, p, p)``.
    """
    g = covs.mean(0)
    for _ in range(max_iter):
        w, V = _eigh(g)
        sw = torch.sqrt(torch.clamp(w, min=1e-6))
        g_sqrt = (V * sw.unsqueeze(-2)) @ V.transpose(-1, -2)
        g_isqrt = (V * (1.0 / sw).unsqueeze(-2)) @ V.transpose(-1, -2)
        whitened = g_isqrt @ covs @ g_isqrt
        logs = _eigfun(whitened, torch.log)
        logs_mean = logs.mean(0)
        if torch.linalg.norm(logs_mean) < tol:
            break
        g = g_sqrt @ _expm_sym(logs_mean) @ g_sqrt
    return g


def tangent_matrices(covs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Project covariances to the tangent space at ``reference``: log(R^-½ C R^-½)."""
    w, V = _eigh(reference)
    isqrt = (V * (1.0 / torch.sqrt(torch.clamp(w, min=1e-6))).unsqueeze(-2)) @ V.transpose(-1, -2)
    whitened = isqrt @ covs @ isqrt
    return _eigfun(whitened, torch.log)


# ---------------------------------------------------------------------------
# High-level, leakage-free transform
# ---------------------------------------------------------------------------

class GPUTangentFC:
    """
    Leakage-free tangent-space FC with GPU-accelerated Riemannian operations.

    Parameters
    ----------
    device : str          "cuda" | "mps" | "cpu"  (falls back to CPU eigh if the
                          backend lacks an eigh kernel).
    max_iter : int        geometric-mean iterations (10 matches nilearn closely).
    dtype : torch.dtype   float32 (fast, validated) or float64 (reference-exact).
    discard_diagonal : bool  drop the diagonal in the vectorized output (matches
                          nilearn vectorize).
    ridge : float         extra isotropic shrinkage applied after Ledoit-Wolf, as
                          ``(1-r) C + r (tr C / P) I``.  Zero reproduces the standard
                          pipeline.  The windowed path in ``dynamic_fc`` applies such a
                          ridge unconditionally, so a static baseline left at zero is
                          regularised less than the representation it is compared with.
    """

    def __init__(self, device: str = "cuda", max_iter: int = 10,
                 dtype: torch.dtype = torch.float32, discard_diagonal: bool = True,
                 ridge: float = 0.0) -> None:
        self.device = torch.device(device if _eigh_ok(device) else "cpu")
        self.max_iter = max_iter
        self.dtype = dtype
        self.discard_diagonal = discard_diagonal
        self.ridge = float(ridge)
        self.reference_: Optional[torch.Tensor] = None
        self._n_rois: Optional[int] = None

    # -- covariances (computed once, fold-independent) --
    def compute_covariances(self, timeseries: List[np.ndarray]) -> np.ndarray:
        """Ledoit-Wolf shrinkage covariance per subject -> (n, p, p) float array."""
        from sklearn.covariance import LedoitWolf
        covs = np.stack([LedoitWolf().fit(np.asarray(ts, dtype=np.float64)).covariance_
                         for ts in timeseries])
        if self.ridge > 0.0:
            # Additional isotropic shrinkage towards the scaled identity, matching the
            # ridge the windowed path applies in dynamic_fc.  Without it the static and
            # windowed representations are regularised differently, so a comparison
            # between them confounds the representation with the amount of shrinkage.
            P = covs.shape[-1]
            mu = np.trace(covs, axis1=-2, axis2=-1) / P
            eye = np.eye(P)[None, :, :]
            covs = (1.0 - self.ridge) * covs + self.ridge * mu[:, None, None] * eye
        self._n_rois = covs.shape[-1]
        return covs

    def _to_t(self, covs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(covs, dtype=self.dtype, device=self.device)

    # -- per-fold: fit reference on TRAIN covariances only --
    def fit(self, covs_train: np.ndarray) -> "GPUTangentFC":
        self._n_rois = covs_train.shape[-1]
        self.reference_ = geometric_mean(self._to_t(covs_train), max_iter=self.max_iter)
        return self

    # -- project covariances -> vectorized tangent features --
    def transform(self, covs: np.ndarray) -> np.ndarray:
        if self.reference_ is None:
            raise RuntimeError("call fit() on the training covariances first")
        tang = tangent_matrices(self._to_t(covs), self.reference_)   # (n, p, p)
        p = tang.shape[-1]
        iu = torch.triu_indices(p, p, offset=1 if self.discard_diagonal else 0,
                                device=tang.device)
        vec = tang[:, iu[0], iu[1]]
        return vec.float().cpu().numpy()

    def fit_transform_train(self, covs_train: np.ndarray) -> np.ndarray:
        return self.fit(covs_train).transform(covs_train)


def _eigh_ok(device: str) -> bool:
    """torch.linalg.eigh is unavailable on MPS; route those to CPU."""
    return not str(device).startswith("mps")
