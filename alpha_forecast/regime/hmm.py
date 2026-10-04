"""Gaussian hidden Markov model for market regime detection.

Markets alternate between calm, trending periods and turbulent, high-volatility
ones. A hidden Markov model with Gaussian emissions recovers those regimes from
observable series (returns, realised volatility, ...).

Implemented in NumPy (scaled Baum-Welch, diagonal covariances) so it installs
anywhere. States are re-ordered after fitting by the variance of the first
column, so state 0 is always the calmest regime.

Look-ahead matters here too: :meth:`GaussianHMM.filter` returns
``P(state_t | x_1..x_t)`` and is safe for trading decisions, whereas
:meth:`GaussianHMM.smooth` conditions on the whole sample and is only for
in-sample analysis.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LOG_2PI = np.log(2.0 * np.pi)


def _as_2d(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    return arr[:, None] if arr.ndim == 1 else arr


class GaussianHMM:
    def __init__(
        self,
        n_states: int = 2,
        *,
        n_iter: int = 50,
        tol: float = 1e-4,
        min_var: float = 1e-6,
        stickiness: float = 0.95,
    ) -> None:
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.min_var = min_var
        self.stickiness = stickiness
        self.startprob_ = np.zeros(0)
        self.transmat_ = np.zeros((0, 0))
        self.means_ = np.zeros((0, 0))
        self.vars_ = np.zeros((0, 0))
        self.loglik_: float = float("nan")
        self.n_iter_: int = 0

    def _check_fitted(self) -> None:
        if self.means_.size == 0:
            raise RuntimeError("GaussianHMM is not fitted; call fit() first")

    # ------------------------------------------------------------------
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.means_[None, :, :]
        return -0.5 * (
            _LOG_2PI * X.shape[1]
            + np.log(self.vars_).sum(axis=1)[None, :]
            + (diff**2 / self.vars_[None, :, :]).sum(axis=2)
        )

    @staticmethod
    def _scaled_emission(log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        offset = log_b.max(axis=1, keepdims=True)
        return np.exp(log_b - offset), offset.ravel()

    def _forward(self, B: np.ndarray, init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A = self.transmat_
        T, K = B.shape
        alpha = np.empty((T, K))
        scale = np.empty(T)
        a = init * B[0]
        for t in range(T):
            if t > 0:
                a = (a @ A) * B[t]
            s = a.sum()
            if s <= 0:
                a = np.full(K, 1.0 / K)
                s = 1.0
            else:
                a = a / s
            alpha[t] = a
            scale[t] = s
        return alpha, scale

    def _backward(self, B: np.ndarray, scale: np.ndarray) -> np.ndarray:
        A = self.transmat_
        T, K = B.shape
        beta = np.empty((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = A @ (B[t + 1] * beta[t + 1]) / scale[t + 1]
        return beta

    # ------------------------------------------------------------------
    def fit(self, X) -> GaussianHMM:
        X = _as_2d(X)
        T, d = X.shape
        K = self.n_states
        if T < 2 * K:
            raise ValueError(f"Need at least {2 * K} observations, got {T}")

        # Initialise means at spread-out quantiles of the data and a sticky
        # transition matrix (regimes persist for many days).
        qs = np.linspace(0.2, 0.8, K) if K > 1 else np.array([0.5])
        self.means_ = np.quantile(X, qs, axis=0).reshape(K, d)
        base_var = np.maximum(X.var(axis=0), self.min_var)
        self.vars_ = np.tile(base_var, (K, 1)) * np.linspace(0.5, 2.0, K)[:, None]
        off = (1.0 - self.stickiness) / (K - 1) if K > 1 else 0.0
        self.transmat_ = np.full((K, K), off) + np.eye(K) * (self.stickiness - off)
        self.startprob_ = np.full(K, 1.0 / K)

        prev = -np.inf
        for it in range(1, self.n_iter + 1):
            B, offset = self._scaled_emission(self._log_emission(X))
            alpha, scale = self._forward(B, self.startprob_)
            beta = self._backward(B, scale)
            loglik = float(np.log(scale).sum() + offset.sum())

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = (
                alpha[:-1, :, None]
                * self.transmat_[None, :, :]
                * (B[1:] * beta[1:])[:, None, :]
                / scale[1:, None, None]
            ).sum(axis=0)

            self.startprob_ = gamma[0] / gamma[0].sum()
            self.transmat_ = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
            weight = np.maximum(gamma.sum(axis=0), 1e-12)[:, None]
            self.means_ = gamma.T @ X / weight
            sq = gamma.T @ (X**2) / weight - self.means_**2
            self.vars_ = np.maximum(sq, self.min_var)

            self.loglik_ = loglik
            self.n_iter_ = it
            if abs(loglik - prev) < self.tol * max(1.0, abs(loglik)):
                break
            prev = loglik

        self._order_states()
        return self

    def _order_states(self) -> None:
        order = np.argsort(self.vars_[:, 0])
        self.means_ = self.means_[order]
        self.vars_ = self.vars_[order]
        self.startprob_ = self.startprob_[order]
        self.transmat_ = self.transmat_[np.ix_(order, order)]

    # ------------------------------------------------------------------
    def filter(self, X, init: np.ndarray | None = None) -> np.ndarray:
        """Causal state probabilities ``P(state_t | x_1..x_t)``, shape (T, K).

        ``init`` is the filtered distribution *before* the first row (e.g. the
        last row of a previous :meth:`filter` call); it is propagated one step
        through the transition matrix.
        """
        self._check_fitted()
        X = _as_2d(X)
        B, _ = self._scaled_emission(self._log_emission(X))
        start = self.startprob_ if init is None else np.asarray(init) @ self.transmat_
        alpha, _ = self._forward(B, start)
        return alpha

    def smooth(self, X) -> np.ndarray:
        """In-sample state probabilities conditioned on the full sample."""
        self._check_fitted()
        X = _as_2d(X)
        B, _ = self._scaled_emission(self._log_emission(X))
        alpha, scale = self._forward(B, self.startprob_)
        beta = self._backward(B, scale)
        gamma = alpha * beta
        return gamma / gamma.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """Most likely current state per row, using causal filtering."""
        return self.filter(X).argmax(axis=1)


def regime_features(returns: pd.Series, vol_window: int = 10) -> pd.DataFrame:
    """Default observation matrix: daily return and trailing realised vol."""
    r = returns.astype(float)
    return pd.DataFrame(
        {"ret": r, "vol": r.rolling(vol_window).std()}, index=returns.index
    ).dropna()


def rolling_regime_probabilities(
    returns: pd.Series,
    *,
    n_states: int = 2,
    initial_train: int = 252,
    step: int = 21,
    vol_window: int = 10,
) -> pd.DataFrame:
    """Walk-forward regime probabilities with no look-ahead.

    Every ``step`` rows the HMM is refit on all data observed so far, then
    filtered forward through the next ``step`` rows. Columns are
    ``regime_0`` (calmest) .. ``regime_{K-1}`` (most volatile).
    """
    obs = regime_features(returns, vol_window)
    Z = obs.to_numpy()
    n = len(Z)
    if n <= initial_train:
        raise ValueError(f"Not enough data: have {n} rows, need > {initial_train}")

    rows: list[np.ndarray] = []
    idx: list = []
    start = initial_train
    while start < n:
        end = min(start + step, n)
        hmm = GaussianHMM(n_states).fit(Z[:start])
        probs = hmm.filter(Z[:end])[start:end]
        rows.append(probs)
        idx.extend(obs.index[start:end])
        start = end
    cols = [f"regime_{k}" for k in range(n_states)]
    return pd.DataFrame(np.vstack(rows), index=pd.Index(idx, name=obs.index.name), columns=cols)
