"""Crypto / on-chain / holdings tests. Network is never touched: the suite runs offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_forecast.data import crypto
from alpha_forecast.data.crypto import (
    chain_for_symbol,
    daily_onchain,
    is_crypto_symbol,
    network_stats,
    onchain_features,
    onchain_frame,
)
from alpha_forecast.evaluation.metrics import CRYPTO_DAYS, TRADING_DAYS, strategy_performance
from alpha_forecast.features.onchain import load_onchain
from alpha_forecast.pipeline import compare_models, days_per_year_for
from alpha_forecast.portfolio.holdings import (
    aggregate_by_symbol,
    load_holdings,
    quote_symbol,
    value_holdings,
)


def test_symbol_classification():
    assert chain_for_symbol("BTC-USD") == "bitcoin"
    assert chain_for_symbol("eth-usd") == "ethereum"
    assert chain_for_symbol("AAPL") is None
    assert is_crypto_symbol("BTC-USD")
    assert is_crypto_symbol("DOGE")
    assert is_crypto_symbol("SOL-USD")  # pair syntax even if Blockchair lacks the chain
    assert not is_crypto_symbol("AAPL")
    assert days_per_year_for("BTC-USD") == CRYPTO_DAYS
    assert days_per_year_for("AAPL") == TRADING_DAYS


def test_onchain_frame_parses_blockchair_rows():
    rows = [
        {"date": "2026-10-02", "count()": 141, "sum(transaction_count)": 593742,
         "sum(fee_total_usd)": 326947.49, "sum(output_total_usd)": 1.15e11,
         "avg(difficulty)": 1.3e14},
        {"date": "2026-10-01", "count()": 136, "sum(transaction_count)": 535739,
         "sum(fee_total_usd)": 354312.09, "sum(output_total_usd)": 7.4e10,
         "avg(difficulty)": 1.3e14},
    ]
    df = onchain_frame(rows, "bitcoin")
    assert list(df.columns) == ["blocks", "tx_count", "fees_usd", "volume_usd", "difficulty"]
    assert df.index.is_monotonic_increasing
    assert df.loc["2026-10-02", "tx_count"] == 593742


def test_onchain_frame_ethereum_drops_unsupported():
    rows = [{"date": "2026-10-02", "count()": 7000, "sum(transaction_count)": 1.5e6,
             "sum(fee_total_usd)": 1e6, "avg(difficulty)": 0}]
    df = onchain_frame(rows, "ethereum")
    assert "volume_usd" not in df.columns
    assert onchain_frame([], "ethereum").empty


def test_onchain_features_are_backward_looking():
    idx = pd.date_range("2026-01-01", periods=60, freq="D")
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({
        "blocks": rng.integers(130, 160, 60),
        "tx_count": rng.integers(5e5, 8e5, 60),
        "fees_usd": rng.uniform(1e5, 4e5, 60),
        "volume_usd": rng.uniform(5e10, 1.5e11, 60),
        "difficulty": np.linspace(1.3e14, 1.35e14, 60),
    }, index=idx)
    feats = onchain_features(raw)
    assert {"tx_growth_7", "fee_per_tx_usd", "volume_z_30", "difficulty_chg_14",
            "blocks_vs_30"} == set(feats.columns)
    # Changing the future must not change today's features.
    bumped = raw.copy()
    bumped.iloc[-1] *= 10
    pd.testing.assert_frame_equal(onchain_features(bumped).iloc[:-1], feats.iloc[:-1])
    # Proof-of-stake: difficulty is 0, so that feature must be dropped, not all-NaN.
    pos = raw.assign(difficulty=0.0).drop(columns=["volume_usd"])
    pos_feats = onchain_features(pos)
    assert "difficulty_chg_14" not in pos_feats.columns
    assert "volume_z_30" not in pos_feats.columns
    assert pos_feats.iloc[30:].notna().all().all()


def test_onchain_skips_network_offline_or_without_key(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network called")

    monkeypatch.setattr(crypto, "_get", boom)
    monkeypatch.setenv("ALPHA_FORECAST_OFFLINE", "1")
    assert daily_onchain("bitcoin", "2026-01-01", "2026-02-01") is None
    assert network_stats("bitcoin") is None
    monkeypatch.setenv("ALPHA_FORECAST_OFFLINE", "0")
    monkeypatch.delenv("BLOCKCHAIR_API_KEY", raising=False)
    assert daily_onchain("bitcoin", "2026-01-01", "2026-02-01") is None
    assert load_onchain("BTC-USD", "2026-01-01", "2026-02-01") is None
    assert load_onchain("AAPL", "2026-01-01", "2026-02-01") is None


def test_daily_onchain_uses_key_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHA_FORECAST_OFFLINE", "0")
    monkeypatch.setenv("BLOCKCHAIR_API_KEY", "test-key")
    monkeypatch.setenv("ALPHA_FORECAST_CACHE_DIR", str(tmp_path))
    calls: list[tuple[str, dict]] = []

    def fake_get(path, params=None):
        calls.append((path, dict(params or {})))
        return {"data": [{"date": "2021-01-02", "count()": 1, "sum(transaction_count)": 2,
                          "sum(fee_total_usd)": 3.0, "sum(output_total_usd)": 4.0,
                          "avg(difficulty)": 5.0}],
                "context": {"code": 200}}

    monkeypatch.setattr(crypto, "_get", fake_get)
    a = daily_onchain("bitcoin", "2021-01-01", "2021-01-03")
    b = daily_onchain("bitcoin", "2021-01-01", "2021-01-03")
    assert a is not None and b is not None
    pd.testing.assert_frame_equal(a, b)
    assert len(calls) == 1
    assert calls[0][0] == "bitcoin/blocks"
    assert calls[0][1]["q"] == "time(2021-01-01..2021-01-03)"
    assert (tmp_path / "onchain").exists()

    feats = load_onchain("BTC-USD", "2021-01-01", "2021-01-03")
    assert feats is not None
    assert all(c.startswith("onchain_") for c in feats.columns)


def test_strategy_performance_annualises_with_calendar():
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(1)
    y = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    pred = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    a = strategy_performance(y, pred, days_per_year=TRADING_DAYS)
    b = strategy_performance(y, pred, days_per_year=CRYPTO_DAYS)
    assert b["ann_return"] == pytest.approx(a["ann_return"] * CRYPTO_DAYS / TRADING_DAYS)
    assert b["sharpe"] == pytest.approx(a["sharpe"] * np.sqrt(CRYPTO_DAYS / TRADING_DAYS))


def test_compare_models_crypto_symbol_offline():
    comp = compare_models("BTC-USD", start="2020-01-01", end="2022-12-31",
                          models=["naive", "drift"], use_onchain=True, n_boot=50)
    assert comp.days_per_year == CRYPTO_DAYS
    assert comp.extra_features == []  # offline: nothing loaded, pipeline still runs
    assert not comp.board.empty


def test_holdings_roundtrip(tmp_path):
    path = tmp_path / "holdings.csv"
    path.write_text(
        "account,asset_class,symbol,quantity,avg_cost\n"
        "6050,equity,msft,0.4,380\n"
        "2192,equity,MSFT,0.25,390\n"
        "6050,crypto,btc,0.002,70000\n"
        "6050,crypto,SOL,0.01,\n"
        "6050,equity,ZERO,0,10\n"
    )
    h = load_holdings(path)
    assert len(h) == 4  # zero-quantity row dropped
    assert list(h["symbol"]) == ["MSFT", "MSFT", "BTC", "SOL"]
    assert quote_symbol("BTC", "crypto") == "BTC-USD"
    assert quote_symbol("MSFT", "equity") == "MSFT"

    quotes = pd.DataFrame({
        "symbol": ["MSFT", "BTC-USD"], "price": [400.0, 80000.0],
        "previous_close": [390.0, 79000.0], "change": [0.0, 0.0],
        "as_of": ["2026-10-02"] * 2, "source": ["yfinance"] * 2,
    })
    v = value_holdings(h, quotes)
    msft = v[v["symbol"] == "MSFT"]
    assert msft["market_value"].sum() == pytest.approx(0.65 * 400)
    assert v.loc[v["symbol"] == "SOL", "price"].isna().all()
    assert v.loc[v["symbol"] == "SOL", "weight"].iloc[0] == 0.0
    assert v["weight"].sum() == pytest.approx(1.0)

    agg = aggregate_by_symbol(v)
    assert agg.iloc[0]["quote_symbol"] == "MSFT"
    assert agg.loc[agg["quote_symbol"] == "MSFT", "quantity"].iloc[0] == pytest.approx(0.65)


def test_missing_holdings_file_is_empty(tmp_path):
    h = load_holdings(tmp_path / "nope.csv")
    assert h.empty
    assert value_holdings(h, pd.DataFrame()).empty
