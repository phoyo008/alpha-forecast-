"""Tests for VaR/ES, the Gaussian HMM, the regime forecaster, and Black-Litterman."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_forecast.data import load_prices
from alpha_forecast.features import build_features
from alpha_forecast.models import RegimeSwitchingForecaster
from alpha_forecast.portfolio import black_litterman, mean_variance_weights
from alpha_forecast.portfolio.backtest import portfolio_backtest
from alpha_forecast.regime import GaussianHMM, rolling_regime_probabilities
from alpha_forecast.risk import (
    historical_es,
    historical_var,
    kupiec_test,
    parametric_es,
    parametric_var,
    rolling_var_forecast,
)


def _returns(n=1000, seed=0, vol=0.01):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, vol, n), index=pd.bdate_range("2018-01-01", periods=n))


def _two_regime(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < 0.98 else 1 - states[t - 1]
    vols = np.where(states == 0, 0.005, 0.03)
    r = pd.Series(rng.normal(0, vols), index=pd.bdate_range("2015-01-01", periods=n))
    return r, states


# --- VaR / ES -----------------------------------------------------------------

def test_parametric_var_es_match_normal_quantiles():
    assert np.isclose(parametric_var(1.0, 0.05), 1.6448536, atol=1e-6)
    assert np.isclose(parametric_es(1.0, 0.05), 2.0627128, atol=1e-6)
    assert parametric_es(0.02, 0.01) > parametric_var(0.02, 0.01) > 0


def test_historical_var_es_ordering():
    r = _returns()
    assert historical_es(r, 0.01) >= historical_var(r, 0.01) > 0


def test_kupiec_accepts_correct_and_rejects_understated_risk():
    r = _returns(n=2000, seed=5, vol=0.01)
    true_var = pd.Series(parametric_var(0.01, 0.01), index=r.index)
    assert kupiec_test(r, true_var, 0.01)["p_value"] > 0.05
    assert kupiec_test(r, true_var / 2, 0.01)["p_value"] < 0.01


@pytest.mark.parametrize("method", ["garch", "historical"])
def test_rolling_var_forecast_is_positive_and_out_of_sample(method):
    r = _returns(n=600)
    out = rolling_var_forecast(r, alpha=0.01, method=method, initial_train=252)
    assert (out["var"] > 0).all() and (out["es"] >= out["var"] - 1e-12).all()
    assert out.index.min() >= r.index[252]


# --- HMM ----------------------------------------------------------------------

def test_hmm_recovers_volatility_regimes():
    r, states = _two_regime()
    hmm = GaussianHMM(2).fit(r.to_numpy())
    assert hmm.vars_[0, 0] < hmm.vars_[1, 0]  # state 0 is the calm one
    acc = (hmm.predict(r.to_numpy()) == states).mean()
    assert acc > 0.85
    assert np.allclose(hmm.transmat_.sum(axis=1), 1.0)


def test_hmm_filter_is_causal():
    r, _ = _two_regime(n=400)
    hmm = GaussianHMM(2).fit(r.to_numpy())
    full = hmm.filter(r.to_numpy())
    head = hmm.filter(r.to_numpy()[:250])
    np.testing.assert_allclose(full[:250], head)


def test_rolling_regime_probabilities_valid():
    r, _ = _two_regime(n=500)
    probs = rolling_regime_probabilities(r, initial_train=252, step=63)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert list(probs.columns) == ["regime_0", "regime_1"]
    assert probs.index.min() > r.index[252]


def test_regime_forecaster_fits_and_predicts():
    X, y = build_features(load_prices("TEST", "2019-01-01", "2022-12-31"))
    m = RegimeSwitchingForecaster().fit(X.iloc[:500], y.iloc[:500])
    p = m.predict(X.iloc[500:])
    assert len(p) == len(X) - 500
    assert np.isfinite(p).all()


# --- Black-Litterman ----------------------------------------------------------

def _cov():
    labels = ["A", "B", "C"]
    a = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    return pd.DataFrame(a, index=labels, columns=labels)


def test_black_litterman_without_views_returns_equilibrium():
    cov = _cov()
    w_mkt = pd.Series([0.5, 0.3, 0.2], index=cov.index)
    mu, cov_post = black_litterman(cov, market_weights=w_mkt, risk_aversion=3.0)
    np.testing.assert_allclose(mu.to_numpy(), 3.0 * cov.to_numpy() @ w_mkt.to_numpy())
    w = mean_variance_weights(mu, cov_post, risk_aversion=3.0, long_only=False)
    np.testing.assert_allclose(w.to_numpy(), w_mkt.to_numpy(), atol=1e-8)


def test_black_litterman_confidence_pulls_toward_views():
    cov = _cov()
    views = pd.Series({"A": 0.30})
    prior, _ = black_litterman(cov)
    weak, _ = black_litterman(cov, views=views, view_confidence=0.1)
    strong, _ = black_litterman(cov, views=views, view_confidence=100.0)
    assert prior["A"] < weak["A"] < strong["A"] < 0.30 + 1e-9
    assert abs(strong["A"] - 0.30) < abs(weak["A"] - 0.30)


def test_black_litterman_relative_view():
    cov = _cov()
    P = pd.DataFrame([[1.0, -1.0, 0.0]], columns=cov.index)
    prior, _ = black_litterman(cov)
    post, _ = black_litterman(cov, P=P, Q=np.array([0.10]))
    assert (post["A"] - post["B"]) > (prior["A"] - prior["B"])


def test_black_litterman_portfolio_scheme_runs():
    out = portfolio_backtest(
        ["AAA", "BBB", "CCC"], scheme="black_litterman", model="drift",
        start="2019-01-01", end="2022-12-31",
    )
    assert np.isfinite(out["sharpe"])
    for _, w in out["weights_history"]:
        assert np.isclose(w.sum(), 1.0, atol=1e-6)
