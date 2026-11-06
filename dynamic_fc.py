"""
Riemannian dynamic functional connectivity (Novelty N1).

Static connectivity summarises a whole scan with one covariance matrix and therefore
discards how connectivity *reconfigures* over time.  Conventional dynamic-FC pipelines
recover some of that by sliding a window, computing Pearson correlations and clustering
the windows into discrete "states", but they operate in Euclidean space and so give up
the manifold geometry that the static part of this pipeline relies on.

Here each participant is instead treated as a **trajectory on the SPD manifold**.  Window
covariances are estimated with Ledoit-Wolf shrinkage and projected to the tangent space
at a reference fitted on the training participants only, giving a sequence of tangent
matrices T_i(1..K).  Three geometry-native statistics summarise the trajectory:

    mean        (1/K) sum_t T_i(t)                    -- where connectivity sits
    dispersion  std_t T_i(t)                          -- how variable it is
    velocity    (1/(K-1)) sum_t |T_i(t+1) - T_i(t)|   -- how fast it reconfigures

The mean block reproduces the static representation, so the other two blocks isolate the
dynamic contribution and can be ablated independently.  Dispersion and velocity are
motivated by reports of atypical network flexibility in autism.

All statistics are per-edge, so each block has the same P(P-1)/2 dimensionality as the
static features and drops straight into the existing cross-validation.

Usage:
    dfc = DynamicTangentFC(window=60, step=30)
    covs = dfc.compute_window_covariances(timeseries)   # once, fold-independent
    dfc.fit(covs, train_idx)                            # reference on TRAIN only
    blocks = dfc.transform(covs, idx)                   # {"mean":..., "dispersion":..., "velocity":...}
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

from gpu_tangent import geometric_mean, tangent_matrices


class DynamicTangentFC:
    """Sliding-window tangent-space trajectory features.

    Parameters
    ----------
    window : int   window length in volumes.
    step   : int   stride between consecutive windows.
    min_windows : int
        Participants yielding fewer windows than this fall back to a single
        whole-scan window, so short scans are not dropped.
    """

    def __init__(self, window: int = 50, n_windows: int = 6, step: int | None = None,
                 min_windows: int = 3, device: str = "cpu", max_iter: int = 10,
                 ridge: float = 0.15) -> None:
        self.window = window
        self.n_windows = n_windows
        self.step = step                       # None -> adaptive (recommended)
        self.min_windows = min_windows
        self.device = torch.device(device if not str(device).startswith("mps") else "cpu")
        self.max_iter = max_iter
        self.ridge = ridge
        self.reference_: torch.Tensor | None = None

    # -- window placement --
    def _starts(self, T: int):
        """Window start indices for a scan of length ``T``.

        With a fixed stride the number of windows scales with scan length, and ABIDE
        scan lengths range from 78 to 316 volumes and differ systematically by site.
        Statistics computed over those windows would then carry a different amount of
        estimation noise per site, which is a site effect rather than a biological one.
        The stride is therefore chosen per participant so that every scan yields the
        same number of windows ``n_windows`` of the same length ``window``; only the
        overlap varies, and the velocity of Eq. (10) is expressed per volume so that it
        remains comparable across different overlaps.
        """
        L = min(self.window, T)
        if self.step is not None:                              # fixed-stride (legacy)
            starts = list(range(0, max(T - L + 1, 1), self.step))
            return (starts if len(starts) >= self.min_windows else [0]), L, float(self.step)
        K = self.n_windows
        if T <= L or K <= 1:
            return [0], L, float(max(T, 1))
        stride = (T - L) / (K - 1)
        starts = [int(round(k * stride)) for k in range(K)]
        starts = [min(s, T - L) for s in starts]
        return starts, L, max(stride, 1e-6)

    # -- window covariances (computed once; independent of the CV fold) --
    def compute_window_covariances(self, timeseries: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Per participant -> (K_i, P, P) float64 array of window covariances.

        A window is much shorter than the number of regions (e.g. 60 volumes for 200
        ROIs), so the sample covariance is rank deficient.  Ledoit-Wolf shrinkage alone
        leaves the matrix too ill-conditioned for the matrix logarithm, so an explicit
        ridge towards the scaled identity is applied on top.  Matrices are kept in
        float64 because the eigendecomposition is the numerically delicate step.
        """
        from sklearn.covariance import LedoitWolf
        out: List[np.ndarray] = []
        self.strides_ = []
        for ts in timeseries:
            ts = np.asarray(ts, dtype=np.float64)
            T, P = ts.shape
            starts, win, stride = self._starts(T)
            covs = []
            for s in starts:
                C = LedoitWolf().fit(ts[s:s + win]).covariance_
                mu = np.trace(C) / P
                C = (1.0 - self.ridge) * C + self.ridge * mu * np.eye(P)
                covs.append(C)
            out.append(np.stack(covs).astype(np.float64))
            self.strides_.append(stride)
        return out

    # -- reference from the training participants only --
    def fit(self, window_covs: Sequence[np.ndarray], train_idx: Sequence[int]) -> "DynamicTangentFC":
        """Fit the tangent reference as the geometric mean of training window covariances.

        Each training participant contributes the arithmetic mean of their own windows,
        so participants with many windows do not dominate the reference.
        """
        per_subject = np.stack([window_covs[i].mean(0) for i in train_idx])
        t = torch.as_tensor(per_subject, dtype=torch.float64, device=self.device)
        self.reference_ = geometric_mean(t, max_iter=self.max_iter)
        return self

    # -- trajectory statistics in the tangent space --
    def transform(self, window_covs: Sequence[np.ndarray],
                  idx: Sequence[int] | None = None,
                  blocks: Sequence[str] = ("mean", "dispersion", "velocity"),
                  batch: int = 32) -> Dict[str, np.ndarray]:
        if self.reference_ is None:
            raise RuntimeError("call fit() with the training indices first")
        idx = list(range(len(window_covs))) if idx is None else list(idx)
        p = window_covs[0].shape[-1]
        iu = np.triu_indices(p, k=1)
        acc: Dict[str, List[np.ndarray]] = {b: [] for b in blocks}

        for start in range(0, len(idx), batch):
            for i in idx[start:start + batch]:
                covs = torch.as_tensor(window_covs[i], dtype=torch.float64, device=self.device)
                tang = self._tangent_safe(covs)                         # (K, P, P)
                V = tang[:, iu[0], iu[1]].cpu().numpy()                 # (K, d)
                if "mean" in acc:
                    acc["mean"].append(V.mean(0))
                if "dispersion" in acc:
                    acc["dispersion"].append(V.std(0) if V.shape[0] > 1 else np.zeros(V.shape[1], np.float32))
                if "velocity" in acc:
                    # per-volume rate: dividing by the participant's stride makes the
                    # measure comparable when overlaps differ (see ``_starts``)
                    stride = getattr(self, "strides_", None)
                    dt = float(stride[i]) if stride is not None and i < len(stride) else 1.0
                    acc["velocity"].append(np.abs(np.diff(V, axis=0)).mean(0) / max(dt, 1e-6)
                                           if V.shape[0] > 1 else np.zeros(V.shape[1], np.float32))
        return {b: np.stack(v).astype(np.float64) for b, v in acc.items()}

    def _tangent_safe(self, covs: torch.Tensor, tries: int = 4) -> torch.Tensor:
        """Tangent projection with escalating jitter if the eigendecomposition fails.

        Short-window covariances occasionally still defeat ``eigh``; adding a small
        multiple of the identity restores convergence without materially changing the
        projection.
        """
        p = covs.shape[-1]
        eye = torch.eye(p, dtype=covs.dtype, device=covs.device)
        scale = torch.diagonal(covs, dim1=-2, dim2=-1).mean()
        for k in range(tries):
            try:
                jitter = (0.0 if k == 0 else (10.0 ** (-6 + k)) * scale)
                return tangent_matrices(covs + jitter * eye, self.reference_)
            except Exception:
                if k == tries - 1:
                    raise
        raise RuntimeError("unreachable")


def concat_blocks(blocks: Dict[str, np.ndarray], use: Sequence[str]) -> np.ndarray:
    """Concatenate the requested trajectory blocks into one feature matrix."""
    missing = [u for u in use if u not in blocks]
    if missing:
        raise KeyError(f"blocks not computed: {missing}")
    return np.hstack([blocks[u] for u in use])
