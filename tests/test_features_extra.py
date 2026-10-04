"""Tests for factor, macro, volatility, and extra-feature integration.

All run offline using synthetic fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_forecast.data import load_prices
from alpha_forecast.features import build_features, load_fama_french
from alpha_forecast.features.factors import FACTOR_COLUMNS
from alpha_forecast.features.macro import load_macro
from alpha_forecast.models.volatility import GARCHVolForecaster, rolling_vol_forecast
from alpha_forecast.pipeline import run_comparison


def _prices():
    return load_prices("TEST", start="2019-01-01", end="2023-12-31")


def test_fama_french_synthetic_shape():
    ff = load_fama_french("2020-01-01", "2021-12-31")
    assert not ff.empty
    assert set(ff.columns).issubset(set(FACTOR_COLUMNS))
    # Decimal units: daily factor magnitudes should be small.
    assert ff.abs().mean().mean() < 0.05


def test_macro_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert load_macro("2020-01-01", "2021-01-01") is None


def test_extra_features_are_lagged_and_joined():
    prices = _prices()
    ff = load_fama_french(
        prices.index.min().date().isoformat(),
        prices.index.max().date().isoformat(),
    )
    X_base, _ = build_features(prices, horizon=1)
    X_ext, y_ext = build_features(prices, horizon=1, extra=ff)
    # Extra columns are prefixed and present.
    ext_cols = [c for c in X_ext.columns if c.startswith("ext_")]
    assert len(ext_cols) >= 1
    # Base price features are still there.
    assert "rsi_14" in X_ext.columns
    assert len(X_ext) == len(y_ext)
    assert not X_ext.isna().any().any()


def test_garch_forecast_positive():
    prices = _prices()
    ret = np.log(prices["close"]).diff().dropna()
    m = GARCHVolForecaster().fit(ret)
    v = m.predict_next()
    assert v > 0
    assert m.backend in ("arch", "ewma")


def test_rolling_vol_forecast_aligned():
    prices = _prices()
    ret = np.log(prices["close"]).diff().dropna()
    vf = rolling_vol_forecast(ret, initial_train=252, step=21)
    assert (vf > 0).all()
    assert len(vf) == len(ret) - 252


def test_pipeline_with_factors_runs():
    board = run_comparison(
        "TEST", start="2019-01-01", end="2023-12-31", use_factors=True
    )
    assert not board.empty
    assert board["sharpe"].is_monotonic_decreasing
