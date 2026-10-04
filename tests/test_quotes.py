"""Live quote parsing. Network is mocked; offline mode must not call it."""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from alpha_forecast.data.quotes import (
    describe_sources,
    latest_quotes,
    quotes_from_download,
    read_snapshot,
)


def _multiindex_download() -> pd.DataFrame:
    idx = pd.to_datetime(["2026-10-01", "2026-10-02"])
    cols = pd.MultiIndex.from_tuples(
        [
            ("Close", "AAPL"),
            ("Close", "MSFT"),
            ("High", "AAPL"),
            ("High", "MSFT"),
        ],
        names=["Price", "Ticker"],
    )
    return pd.DataFrame(
        [[100.0, 200.0, 101.0, 201.0], [110.0, 190.0, 111.0, 191.0]],
        index=idx,
        columns=cols,
    )


def test_quotes_from_multiindex_download():
    out = quotes_from_download(_multiindex_download(), ["AAPL", "MSFT"])
    assert list(out["symbol"]) == ["AAPL", "MSFT"]
    assert out.loc[0, "price"] == 110.0
    assert out.loc[0, "previous_close"] == 100.0
    assert out.loc[0, "change"] == pytest.approx(0.1)
    assert out.loc[1, "change"] == pytest.approx(190.0 / 200.0 - 1.0)
    assert out.loc[0, "as_of"] == "2026-10-02"
    assert set(out["source"]) == {"yfinance"}


def test_quotes_from_flat_download():
    idx = pd.to_datetime(["2026-10-01", "2026-10-02"])
    df = pd.DataFrame({"Close": [10.0, 10.5], "Volume": [1, 1]}, index=idx)
    out = quotes_from_download(df, ["AAPL"])
    assert out.loc[0, "price"] == 10.5
    assert out.loc[0, "symbol"] == "AAPL"


def test_latest_quotes_skips_network_when_offline(monkeypatch):
    monkeypatch.setenv("ALPHA_FORECAST_OFFLINE", "1")

    def boom(symbols):
        raise AssertionError(f"download called for {symbols}")

    monkeypatch.setattr("alpha_forecast.data.quotes._download", boom)
    assert latest_quotes(["aapl", " AAPL ", ""]).empty


def test_latest_quotes_uses_download_when_online(monkeypatch):
    monkeypatch.setenv("ALPHA_FORECAST_OFFLINE", "0")
    monkeypatch.setattr(
        "alpha_forecast.data.quotes._download",
        lambda symbols: _multiindex_download(),
    )
    out = latest_quotes(["aapl", "msft"])
    assert list(out["symbol"]) == ["AAPL", "MSFT"]


SNAPSHOT = (
    "symbol,price,previous_close,as_of,source\n"
    "MSFT,517.19,512.80,2026-10-02 19:59 ET,robinhood\n"
    "btc-usd,85340.50,84831.65,2026-10-04 12:45 ET,robinhood\n"
)


def test_snapshot_preferred_and_yahoo_fills_the_rest(monkeypatch, tmp_path):
    path = tmp_path / "quotes.csv"
    path.write_text(SNAPSHOT)
    monkeypatch.setenv("ALPHA_FORECAST_QUOTES", str(path))
    monkeypatch.setenv("ALPHA_FORECAST_OFFLINE", "0")
    asked: list[list[str]] = []

    def fake_download(symbols):
        asked.append(symbols)
        return _multiindex_download()

    monkeypatch.setattr("alpha_forecast.data.quotes._download", fake_download)
    out = latest_quotes(["AAPL", "BTC-USD", "MSFT"])
    assert asked == [["AAPL"]]
    assert list(out["symbol"]) == ["AAPL", "BTC-USD", "MSFT"]
    assert list(out["source"]) == ["yfinance", "robinhood", "robinhood"]
    msft = out[out["symbol"] == "MSFT"].iloc[0]
    assert msft["price"] == 517.19
    assert msft["change"] == pytest.approx(517.19 / 512.80 - 1.0)
    text = describe_sources(out)
    assert "Robinhood as of 2026-10-02 19:59 ET to 2026-10-04 12:45 ET" in text
    assert "Yahoo Finance, delayed as of 2026-10-02" in text


def test_snapshot_used_offline(monkeypatch, tmp_path):
    path = tmp_path / "quotes.csv"
    path.write_text(SNAPSHOT)
    monkeypatch.setenv("ALPHA_FORECAST_QUOTES", str(path))
    monkeypatch.setenv("ALPHA_FORECAST_OFFLINE", "1")
    out = latest_quotes(["MSFT", "AAPL"])
    assert list(out["symbol"]) == ["MSFT"]


def test_stale_or_malformed_snapshot_is_ignored(tmp_path):
    path = tmp_path / "quotes.csv"
    path.write_text(SNAPSHOT)
    assert len(read_snapshot(path)) == 2
    old = time.time() - 48 * 3600
    os.utime(path, (old, old))
    assert read_snapshot(path).empty
    bad = tmp_path / "bad.csv"
    bad.write_text("symbol,price\nMSFT,1\n")
    assert read_snapshot(bad).empty
    assert read_snapshot(tmp_path / "missing.csv").empty
