"""Tests for the portfolio optimizer, neural model, and portfolio backtest.

All run offline on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_forecast.data import load_prices
from alpha_forecast.features import build_features
from alpha_forecast.models import MODEL_REGISTRY, NeuralForecaster
from alpha_forecast.portfolio import (
    equal_weight,
    mean_variance_weights,
    risk_parity_weights,
)
from alpha_forecast.portfolio.backtest import portfolio_backtest


def _cov_mu(n=4, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    cov = pd.DataFrame(a @ a.T / n + np.eye(n) * 0.01)
    mu = pd.Series(rng.normal(0.05, 0.02, n), index=cov.index)
    return mu, cov


def test_neural_in_registry():
    assert "neural" in MODEL_REGISTRY


def test_equal_weight_sums_to_one():
    w = equal_weight(["A", "B", "C"])
    assert np.isclose(w.sum(), 1.0)
    assert (w == 1 / 3).all()


def test_mean_variance_long_only_valid():
    mu, cov = _cov_mu()
    w = mean_variance_weights(mu, cov, risk_aversion=5.0)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-9).all()  # long-only


def test_mean_variance_tilts_toward_high_return():
    # Diagonal cov so risk is equal; weight should favour the highest mu.
    cov = pd.DataFrame(np.eye(3) * 0.04, index=["A", "B", "C"], columns=["A", "B", "C"])
    mu = pd.Series([0.01, 0.02, 0.10], index=["A", "B", "C"])
    w = mean_variance_weights(mu, cov, risk_aversion=1.0)
    assert w.idxmax() == "C"


def test_risk_parity_valid_and_balanced():
    mu, cov = _cov_mu()
    w = risk_parity_weights(cov)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert (w > 0).all()
    # Risk contributions should be roughly equal.
    sigma = cov.to_numpy()
    rc = w.to_numpy() * (sigma @ w.to_numpy())
    assert rc.std() / rc.mean() < 0.1


def test_neural_fits_and_predicts():
    prices = load_prices("TEST", start="2019-01-01", end="2022-12-31")
    X, y = build_features(prices)
    m = NeuralForecaster(epochs=5).fit(X, y)
    p = m.predict(X)
    assert len(p) == len(X)
    assert np.isfinite(p).all()
    assert m.backend in ("torch_lstm", "sklearn_mlp", "linear")


def test_portfolio_backtest_runs():
    out = portfolio_backtest(
        ["AAA", "BBB", "CCC"],
        scheme="mean_variance",
        model="drift",
        initial_train=252,
        step=21,
        cov_window=60,
        rebalance=21,
    )
    assert "sharpe" in out
    assert -1.0 <= out["max_drawdown"] <= 0.0
    assert len(out["returns"]) > 0
    assert np.isclose(out["equity"].iloc[-1], (1 + out["returns"]).prod(), rtol=1e-6)


def test_portfolio_schemes_all_work():
    for scheme in ("equal", "risk_parity", "mean_variance"):
        out = portfolio_backtest(
            ["AAA", "BBB"], scheme=scheme, model="drift", cov_window=40, rebalance=21
        )
        assert np.isfinite(out["ann_return"])
