"""Tests that run fully offline using the synthetic data fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_forecast.backtest import walk_forward_backtest
from alpha_forecast.data import load_prices
from alpha_forecast.evaluation import (
    directional_accuracy,
    forecast_error_metrics,
    strategy_performance,
)
from alpha_forecast.features import build_features
from alpha_forecast.models import MODEL_REGISTRY, NaiveForecaster
from alpha_forecast.pipeline import run_comparison


def _synthetic_prices():
    # Force synthetic path with a fixed window for determinism.
    return load_prices("TEST", start="2019-01-01", end="2023-12-31")


def test_loader_shape_and_columns():
    df = _synthetic_prices()
    assert not df.empty
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert df["close"].gt(0).all()


def test_loader_is_reproducible():
    a = load_prices("REPRO", start="2020-01-01", end="2021-01-01")
    b = load_prices("REPRO", start="2020-01-01", end="2021-01-01")
    pd.testing.assert_frame_equal(a, b)


def test_features_have_no_nan_and_align():
    df = _synthetic_prices()
    X, y = build_features(df, horizon=1)
    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert X.index.equals(y.index)


def test_features_are_causal_no_lookahead():
    # Truncating future rows must not change earlier feature values.
    df = _synthetic_prices()
    X_full, _ = build_features(df, horizon=1)
    X_trunc, _ = build_features(df.iloc[: len(df) // 2 + 60], horizon=1)
    common = X_full.index.intersection(X_trunc.index)
    # overlap on the earlier portion should match exactly
    common = common[: len(common) // 2]
    pd.testing.assert_frame_equal(
        X_full.loc[common], X_trunc.loc[common], check_exact=False, rtol=1e-9
    )


def test_naive_predicts_zero():
    df = _synthetic_prices()
    X, y = build_features(df)
    m = NaiveForecaster().fit(X, y)
    assert np.allclose(m.predict(X), 0.0)


def test_walk_forward_is_out_of_sample():
    df = _synthetic_prices()
    X, y = build_features(df)
    res = walk_forward_backtest(MODEL_REGISTRY["drift"], X, y, initial_train=252, step=21)
    # Every OOS prediction index must be at or after initial_train.
    assert len(res.predictions) == len(X) - 252
    assert res.predictions.index.equals(res.actuals.index)


def test_metrics_sane():
    df = _synthetic_prices()
    X, y = build_features(df)
    res = walk_forward_backtest(MODEL_REGISTRY["drift"], X, y)
    err = forecast_error_metrics(res.actuals, res.predictions)
    assert err["rmse"] >= 0
    da = directional_accuracy(res.actuals, res.predictions)
    assert 0.0 <= da <= 1.0
    strat = strategy_performance(res.actuals, res.predictions)
    assert -1.0 <= strat["max_drawdown"] <= 0.0
    assert 0.0 <= strat["hit_rate"] <= 1.0


def test_full_comparison_leaderboard():
    board = run_comparison("TEST", start="2019-01-01", end="2023-12-31")
    assert set(board["model"]) == set(MODEL_REGISTRY)
    assert board["sharpe"].notna().any()
    # leaderboard sorted by sharpe descending
    assert board["sharpe"].is_monotonic_decreasing
