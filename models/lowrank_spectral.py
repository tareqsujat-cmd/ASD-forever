"""
Low-rank spectral classifier for tangent-space functional connectivity (Novelty N2).

A linear model on vectorised tangent features learns one weight per edge and discards
the fact that those edges are the upper triangle of a *symmetric matrix*.  Writing the
decision as a matrix inner product makes the structure explicit,

    f(S) = <W, S> + b,      W symmetric,

and constraining W to be low rank,

    W = sum_k lambda_k u_k u_k^T   ->   f(S) = sum_k lambda_k (u_k^T S u_k) + b,

has three consequences.  The parameter count falls from P(P-1)/2 (19,900 for CC200) to
about rP, which is the appropriate regime when p >> n; each u_k is a whole-brain
*eigen-connectome* component that can be projected onto the cortex and interpreted; and
the rank r is exactly the quantity our low-rank analysis measured (AUROC saturates near
rank 50, participation ratio ~12), so the model is matched to the measured structure of
the signal rather than to a guess.

Two estimators are provided:

``LowRankSpectralClassifier``
    Two-stage, convex and deterministic.  Stage 1 fits an l2-logistic model on the
    vectorised features and folds the weights into the symmetric matrix W.  Stage 2
    eigendecomposes W, keeps the r components with the largest |lambda|, projects every
    subject onto them (z_ik = u_k^T S_i u_k) and refits the r coefficients.  Nothing is
    fit outside the training fold.

``JointLowRankClassifier``
    Optimises U and lambda directly against the logistic loss by gradient descent
    (optional, torch).  Slower, but does not inherit stage 1's full-rank solution.

Both are scikit-learn estimators: they accept the same vectorised tangent matrix used
everywhere else in the pipeline, so they drop into the existing cross-validation.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


# --------------------------------------------------------------------------
# vec <-> symmetric-matrix helpers
# --------------------------------------------------------------------------

def n_rois_from_dim(d: int) -> int:
    """Recover P from the length of a strict-upper-triangle vector."""
    p = int(round((1 + np.sqrt(1 + 8 * d)) / 2))
    if p * (p - 1) // 2 != d:
        raise ValueError(f"length {d} is not P(P-1)/2 for integer P")
    return p


def vec_to_sym(v: np.ndarray, p: int | None = None, halve: bool = True) -> np.ndarray:
    """Strict-upper-triangle vector -> symmetric matrix.

    With ``halve`` the off-diagonal entries are set to v/2, so that <W, S> equals the
    plain dot product w . x between the two vectorised triangles.
    """
    v = np.asarray(v, dtype=np.float64)
    p = p or n_rois_from_dim(v.shape[-1])
    iu = np.triu_indices(p, k=1)
    M = np.zeros((p, p), dtype=np.float64)
    M[iu] = v / 2.0 if halve else v
    return M + M.T


def batch_vec_to_sym(X: np.ndarray, p: int | None = None) -> np.ndarray:
    """(n, d) vectorised tangent features -> (n, P, P) symmetric matrices."""
    X = np.asarray(X, dtype=np.float64)
    p = p or n_rois_from_dim(X.shape[1])
    n = X.shape[0]
    iu = np.triu_indices(p, k=1)
    S = np.zeros((n, p, p), dtype=np.float64)
    S[:, iu[0], iu[1]] = X
    return S + np.transpose(S, (0, 2, 1))


def project_components(S: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Quadratic-form projection z_ik = u_k^T S_i u_k -> (n, r)."""
    return np.einsum("nij,ik,jk->nk", S, U, U, optimize=True)


# --------------------------------------------------------------------------
# Two-stage spectral estimator
# --------------------------------------------------------------------------

class LowRankSpectralClassifier(BaseEstimator, ClassifierMixin):
    """Rank-constrained symmetric linear classifier on tangent-space FC.

    Parameters
    ----------
    rank : int
        Number of eigen-connectome components retained.
    C_dense : float
        Inverse regularisation for the stage-1 (full-rank) logistic fit.
    C_head : float
        Inverse regularisation for the stage-2 fit on the r component scores.
    n_rois : int, optional
        P; inferred from the feature dimension when omitted.
    """

    def __init__(self, rank: int = 20, C_dense: float = 0.01, C_head: float = 1.0,
                 n_rois: int | None = None, max_iter: int = 2000):
        self.rank = rank
        self.C_dense = C_dense
        self.C_head = C_head
        self.n_rois = n_rois
        self.max_iter = max_iter

    # -- fit -------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        p = self.n_rois or n_rois_from_dim(X.shape[1])
        self.n_rois_ = p

        # stage 1: dense linear model -> symmetric weight matrix
        dense = LogisticRegression(C=self.C_dense, solver="liblinear",
                                   max_iter=self.max_iter).fit(X, y)
        W = vec_to_sym(dense.coef_.ravel(), p, halve=True)

        # spectral truncation: keep the r components with the largest |eigenvalue|
        w_eig, V = np.linalg.eigh(W)                       # ascending
        order = np.argsort(np.abs(w_eig))[::-1][:self.rank]
        self.components_ = V[:, order]                     # (P, r) eigen-connectomes
        self.eigenvalues_ = w_eig[order]                   # (r,) stage-1 weights

        # stage 2: refit the component coefficients
        Z = self.transform(X)
        self.head_ = LogisticRegression(C=self.C_head, solver="liblinear",
                                        max_iter=self.max_iter).fit(Z, y)
        self.coef_components_ = self.head_.coef_.ravel()   # (r,) lambda_k
        return self

    # -- component scores -------------------------------------------------
    def transform(self, X) -> np.ndarray:
        """Vectorised tangent features -> (n, r) eigen-connectome scores."""
        check_is_fitted(self, "components_")
        S = batch_vec_to_sym(np.asarray(X, dtype=np.float64), self.n_rois_)
        return np.einsum("nij,ik,jk->nk", S, self.components_, self.components_,
                         optimize=True)

    # -- prediction -------------------------------------------------------
    def decision_function(self, X):
        return self.head_.decision_function(self.transform(X))

    def predict_proba(self, X):
        return self.head_.predict_proba(self.transform(X))

    def predict(self, X):
        return self.head_.predict(self.transform(X))

    # -- interpretation ---------------------------------------------------
    def weight_matrix(self) -> np.ndarray:
        """Reconstruct the rank-r symmetric weight matrix W = sum_k lambda_k u_k u_k^T."""
        check_is_fitted(self, "components_")
        U, lam = self.components_, self.coef_components_
        return (U * lam[None, :]) @ U.T

    def edge_weights(self) -> np.ndarray:
        """W as a strict-upper-triangle vector, comparable to a dense model's coef_."""
        W = self.weight_matrix()
        iu = np.triu_indices(self.n_rois_, k=1)
        return W[iu]


# --------------------------------------------------------------------------
# Jointly optimised estimator (optional, torch)
# --------------------------------------------------------------------------

class JointLowRankClassifier(BaseEstimator, ClassifierMixin):
    """Directly optimise W = U diag(lambda) U^T against the logistic loss."""

    def __init__(self, rank: int = 20, l2: float = 1e-3, epochs: int = 300,
                 lr: float = 5e-2, n_rois: int | None = None, seed: int = 42,
                 device: str = "cpu"):
        self.rank = rank
        self.l2 = l2
        self.epochs = epochs
        self.lr = lr
        self.n_rois = n_rois
        self.seed = seed
        self.device = device

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        p = self.n_rois or n_rois_from_dim(X.shape[1])
        self.n_rois_ = p

        dev = torch.device(self.device)
        S = torch.as_tensor(batch_vec_to_sym(X, p), dtype=torch.float32, device=dev)
        t = torch.as_tensor((y == self.classes_[1]).astype(np.float32), device=dev)

        U = torch.randn(p, self.rank, device=dev) * (1.0 / np.sqrt(p))
        lam = torch.zeros(self.rank, device=dev)
        b = torch.zeros(1, device=dev)
        for prm in (U, lam, b):
            prm.requires_grad_(True)

        opt = torch.optim.Adam([U, lam, b], lr=self.lr)
        lossf = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            z = torch.einsum("nij,ik,jk->nk", S, U, U)      # (n, r)
            logits = z @ lam + b
            loss = lossf(logits, t) + self.l2 * (U.pow(2).sum() + lam.pow(2).sum())
            loss.backward()
            opt.step()

        self.components_ = U.detach().cpu().numpy()
        self.coef_components_ = lam.detach().cpu().numpy()
        self.intercept_ = float(b.detach().cpu().numpy()[0])
        return self

    def decision_function(self, X):
        check_is_fitted(self, "components_")
        S = batch_vec_to_sym(np.asarray(X, dtype=np.float64), self.n_rois_)
        z = np.einsum("nij,ik,jk->nk", S, self.components_, self.components_, optimize=True)
        return z @ self.coef_components_ + self.intercept_

    def predict_proba(self, X):
        d = self.decision_function(X)
        pos = 1.0 / (1.0 + np.exp(-d))
        return np.column_stack([1 - pos, pos])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) >= 0).astype(int)]

    def weight_matrix(self) -> np.ndarray:
        U, lam = self.components_, self.coef_components_
        return (U * lam[None, :]) @ U.T
