"""Portfolio construction from expected returns and a covariance matrix.

Three allocation schemes, all long-only and fully invested (weights sum to 1):

    * equal_weight        -- naive 1/N baseline
    * mean_variance_weights -- maximise expected return per unit of risk
                               (tangency-style, with optional risk aversion)
    * risk_parity_weights -- each asset contributes equal risk

Implemented with numpy only (no cvxpy dependency) using projected-gradient /
iterative schemes, so the module installs anywhere. These are the workhorse
methods a quant is expected to know cold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_arrays(expected_returns, cov):
    mu = np.asarray(expected_returns, dtype=float).ravel()
    sigma = np.asarray(cov, dtype=float)
    if sigma.shape != (len(mu), len(mu)):
        raise ValueError("cov must be square and match len(expected_returns)")
    return mu, sigma


def _simplex_projection(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``v`` onto the probability simplex.

    Produces long-only weights that sum to 1 (Duchi et al., 2008).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def equal_weight(assets) -> pd.Series:
    """1/N portfolio over the given asset labels."""
    assets = list(assets)
    n = len(assets)
    return pd.Series(np.full(n, 1.0 / n), index=assets, name="weight")


def mean_variance_weights(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    *,
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> pd.Series:
    """Mean-variance optimal weights.

    Maximises ``mu @ w - 0.5 * risk_aversion * w' Σ w``.

    * If ``long_only`` (default) weights are projected onto the simplex via a
      short projected-gradient ascent (sum to 1, no shorts).
    * Otherwise returns the closed-form unconstrained solution renormalised to
      sum to 1 (allows shorts).
    """
    labels = list(expected_returns.index)
    mu, sigma = _as_arrays(expected_returns, cov)
    sigma = sigma + 1e-8 * np.eye(len(mu))  # numerical stability

    if not long_only:
        w = np.linalg.solve(risk_aversion * sigma, mu)
        s = w.sum()
        w = w / s if s != 0 else np.full_like(w, 1.0 / len(w))
        return pd.Series(w, index=labels, name="weight")

    # Projected gradient ascent on the simplex.
    w = np.full(len(mu), 1.0 / len(mu))
    lr = 1.0 / (np.linalg.norm(sigma, 2) * risk_aversion + 1.0)
    for _ in range(500):
        grad = mu - risk_aversion * sigma @ w
        w = _simplex_projection(w + lr * grad)
    return pd.Series(w, index=labels, name="weight")


def risk_parity_weights(
    cov: pd.DataFrame,
    *,
    iterations: int = 1000,
    tol: float = 1e-8,
) -> pd.Series:
    """Equal risk-contribution (risk parity) weights, long-only, sum to 1.

    Uses the classic iterative fixed-point update on the risk contributions.
    """
    labels = list(cov.index) if hasattr(cov, "index") else list(range(len(cov)))
    sigma = np.asarray(cov, dtype=float)
    n = sigma.shape[0]
    sigma = sigma + 1e-8 * np.eye(n)

    # Spinu's cyclical coordinate descent. For each asset i we solve the
    # scalar quadratic that sets its risk contribution to the common target
    # (1/n of the budget). Weights stay strictly positive throughout, so there
    # are no sqrt-of-negative issues regardless of the covariance structure.
    b = 1.0 / n  # equal risk budget per asset
    w = np.full(n, 1.0 / n)
    for _ in range(iterations):
        w_prev = w.copy()
        for i in range(n):
            # Risk from all other assets w.r.t. asset i.
            others = sigma[i] @ w - sigma[i, i] * w[i]
            # Solve sigma_ii * w_i^2 + others * w_i - b = 0 for w_i > 0.
            disc = others**2 + 4.0 * sigma[i, i] * b
            w[i] = (-others + np.sqrt(disc)) / (2.0 * sigma[i, i])
        if np.max(np.abs(w - w_prev)) < tol:
            break
    w = np.maximum(w, 0.0)
    w /= w.sum()
    return pd.Series(w, index=labels, name="weight")
