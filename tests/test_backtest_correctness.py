"""Regression tests for look-ahead, overlap, cost, and timing bugs."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from alpha_forecast.backtest import walk_forward_backtest
from alpha_forecast.data import load_prices
from alpha_forecast.evaluation import (
    directional_accuracy,
    strategy_performance,
    strategy_returns,
)
from alpha_forecast.features import build_features
from alpha_forecast.models.base import Forecaster
from alpha_forecast.portfolio.backtest import portfolio_backtest

START, END = "2019-01-01", "2022-12-31"


class _SpyForecaster(Forecaster):
    """Records the last training date seen in each fold."""

    name = "spy"
    train_ends: list = []

    def fit(self, X, y):
        _SpyForecaster.train_ends.append(X.index[-1])
        return self

    def predict(self, X):
        return np.zeros(len(X))


@pytest.mark.parametrize("horizon", [1, 5, 10])
def test_walk_forward_purges_overlapping_targets(horizon):
    X, y = build_features(load_prices("TEST", START, END), horizon=horizon)
    _SpyForecaster.train_ends = []
    res = walk_forward_backtest(_SpyForecaster, X, y, initial_train=252, step=21, horizon=horizon)

    positions = {d: i for i, d in enumerate(X.index)}
    test_starts = list(range(252, len(X), 21))
    assert len(_SpyForecaster.train_ends) == len(test_starts)
    for train_end, test_start in zip(_SpyForecaster.train_ends, test_starts, strict=True):
        # Target of the last training row is realised horizon rows later, which
        # must be no later than the first test row's feature date.
        assert positions[train_end] + horizon <= test_start
    assert len(res.predictions) == len(X) - 252


def test_walk_forward_rejects_bad_horizon():
    X, y = build_features(load_prices("TEST", START, END))
    with pytest.raises(ValueError):
        walk_forward_backtest(_SpyForecaster, X, y, horizon=0)


def _toy_series(values, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_strategy_costs_charge_entry_and_flips():
    y = _toy_series([0.01, 0.01, 0.01, 0.01])
    pred = _toy_series([1, 1, -1, -1])
    net = strategy_returns(y, pred, cost_bps=10)
    cost = 10 / 1e4
    expected = [0.01 - cost, 0.01, -0.01 - 2 * cost, -0.01]
    np.testing.assert_allclose(net.to_numpy(), expected)


def test_strategy_uses_non_overlapping_periods_for_multi_day_horizon():
    rng = np.random.default_rng(0)
    y = _toy_series(rng.normal(0, 0.02, 100))
    pred = _toy_series(rng.normal(0, 1, 100))
    net = strategy_returns(y, pred, cost_bps=0, horizon=5)
    assert len(net) == 20
    assert net.index.equals(y.index[::5])
    perf = strategy_performance(y, pred, cost_bps=0, horizon=5)
    assert perf["n_periods"] == 20
    expected_ann = net.mean() * 252 / 5
    assert np.isclose(perf["ann_return"], expected_ann)


def test_equity_compounds_log_returns():
    y = _toy_series([np.log(1.1), np.log(1.1)])
    pred = _toy_series([1, 1])
    perf = strategy_performance(y, pred, cost_bps=0)
    assert np.isclose(perf["total_return"], 1.21 - 1)


def test_directional_accuracy_treats_zero_forecast_as_long():
    y = _toy_series([0.01, -0.01, 0.02, 0.03])
    assert directional_accuracy(y, y * 0) == 0.75


def test_portfolio_earns_next_day_return():
    symbols = ["AAA", "BBB"]
    out = portfolio_backtest(
        symbols, scheme="equal", model="drift", start=START, end=END,
        rebalance=1, cost_bps=0.0,
    )
    fwd = pd.DataFrame(
        {s: load_prices(s, START, END)["close"].pct_change().shift(-1) for s in symbols}
    )
    expected = fwd.loc[out["returns"].index].mean(axis=1)
    np.testing.assert_allclose(out["returns"].to_numpy(), expected.to_numpy(), atol=1e-12)


def test_portfolio_costs_reduce_returns_by_turnover():
    kwargs = dict(scheme="mean_variance", model="drift", start=START, end=END, rebalance=21)
    free = portfolio_backtest(["AAA", "BBB", "CCC"], cost_bps=0.0, **kwargs)
    costly = portfolio_backtest(["AAA", "BBB", "CCC"], cost_bps=25.0, **kwargs)
    assert costly["ann_return"] < free["ann_return"]
    assert free["ann_turnover"] > 0


def test_portfolio_respects_date_range():
    out = portfolio_backtest(
        ["AAA", "BBB"], scheme="equal", model="drift", start="2020-01-01", end="2022-06-30"
    )
    assert out["returns"].index.min() >= pd.Timestamp("2020-01-01")
    assert out["returns"].index.max() <= pd.Timestamp("2022-06-30")


def test_portfolio_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        portfolio_backtest(["AAA"], scheme="magic", model="drift")


def test_synthetic_prices_identical_across_processes():
    code = (
        "from alpha_forecast.data import load_prices;"
        "print(repr(float(load_prices('XYZ','2020-01-01','2020-12-31')['close'].iloc[-1])))"
    )
    outs = set()
    for seed in ("1", "2"):
        env = {"PYTHONHASHSEED": seed, "ALPHA_FORECAST_OFFLINE": "1",
               "ALPHA_FORECAST_CACHE_DIR": ""}
        res = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        outs.add(res.stdout.strip())
    assert len(outs) == 1
