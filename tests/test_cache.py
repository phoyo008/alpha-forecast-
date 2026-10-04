"""Tests for the on-disk price cache and offline mode."""

from __future__ import annotations

import pandas as pd

from alpha_forecast.data import loader


def _fake_prices():
    idx = pd.bdate_range("2021-01-01", periods=5, name="date")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": [1.0, 1.1, 1.2, 1.1, 1.3],
         "volume": 100},
        index=idx,
    )


def test_provider_results_are_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHA_FORECAST_OFFLINE", "0")
    monkeypatch.setenv("ALPHA_FORECAST_CACHE_DIR", str(tmp_path))
    calls = []

    def provider(symbol, start, end):
        calls.append(symbol)
        return _fake_prices()

    monkeypatch.setattr(loader, "_from_openbb", provider)
    monkeypatch.setattr(loader, "_from_yfinance", lambda *a: None)

    first = loader.load_prices("FAKE", "2021-01-01", "2021-01-08")
    second = loader.load_prices("FAKE", "2021-01-01", "2021-01-08")
    pd.testing.assert_frame_equal(first, second)
    assert calls == ["FAKE"]
    assert list((tmp_path / "prices").glob("FAKE_*.pkl"))

    loader.load_prices("FAKE", "2021-01-01", "2021-01-08", use_cache=False)
    assert calls == ["FAKE", "FAKE"]


def test_synthetic_data_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHA_FORECAST_OFFLINE", "1")
    monkeypatch.setenv("ALPHA_FORECAST_CACHE_DIR", str(tmp_path))
    loader.load_prices("SYN", "2021-01-01", "2021-06-30")
    assert not (tmp_path / "prices").exists()


def test_offline_mode_skips_providers(monkeypatch):
    monkeypatch.setenv("ALPHA_FORECAST_OFFLINE", "1")

    def boom(*args):
        raise AssertionError("network provider called in offline mode")

    monkeypatch.setattr(loader, "_from_openbb", boom)
    monkeypatch.setattr(loader, "_from_yfinance", boom)
    df = loader.load_prices("OFF", "2021-01-01", "2021-03-31")
    assert not df.empty
