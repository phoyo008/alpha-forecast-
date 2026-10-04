"""Headless smoke tests for the Streamlit dashboard (skipped without streamlit)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
TIMEOUT = 300


def _app(mode: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.radio[0].set_value(mode).run()
    at.date_input[0].set_value(date(2019, 1, 1))
    at.date_input[1].set_value(date(2022, 12, 31))
    return at


def test_dashboard_idle_state():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("Run backtest" in i.value for i in at.info)


def test_dashboard_single_asset_run():
    at = _app("Single asset")
    at.text_input[0].set_value("TEST")
    at.multiselect[0].set_value(["naive", "drift", "regime"])
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("synthetic" in w.value for w in at.warning)
    assert len(at.tabs) == 5
    assert at.dataframe  # leaderboard, Kupiec table, regime stats


def test_dashboard_portfolio_run():
    at = _app("Portfolio")
    at.text_input[0].set_value("AAA, BBB, CCC")
    at.selectbox[1].set_value("drift")
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert [m.label for m in at.metric][:2] == ["Sharpe", "Ann. return"]


def test_dashboard_crypto_symbol_has_onchain_toggle_and_network_tab(monkeypatch):
    monkeypatch.delenv("BLOCKCHAIR_API_KEY", raising=False)
    at = _app("Single asset")
    at.text_input[0].set_value("BTC-USD").run()
    assert not at.exception, at.exception
    labels = [t.label for t in at.toggle]
    assert any("Blockchair" in lbl for lbl in labels)
    at.multiselect[0].set_value(["naive", "drift"])
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6  # Network tab appears for crypto
    assert any("BLOCKCHAIR_API_KEY" in i.value for i in at.info)


def test_dashboard_holdings_mode_without_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHA_FORECAST_HOLDINGS", str(tmp_path / "none.csv"))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.radio[0].set_value("Holdings").run()
    assert not at.exception, at.exception
    assert any("No holdings file" in i.value for i in at.info)


def test_dashboard_holdings_mode_with_file(monkeypatch, tmp_path):
    path = tmp_path / "holdings.csv"
    path.write_text(
        "account,asset_class,symbol,quantity,avg_cost\n"
        "6050,equity,AAA,1,10\n6050,equity,BBB,2,20\n6050,crypto,CCC,0.5,\n"
    )
    quotes = tmp_path / "quotes.csv"
    quotes.write_text(
        "symbol,price,previous_close,as_of,source\n"
        "AAA,11,10,2026-10-02 16:00 ET,robinhood\n"
        "BBB,25,24,2026-10-02 16:00 ET,robinhood\n"
        "CCC-USD,100,99,2026-10-04 12:00 ET,robinhood\n"
    )
    monkeypatch.setenv("ALPHA_FORECAST_HOLDINGS", str(path))
    monkeypatch.setenv("ALPHA_FORECAST_QUOTES", str(quotes))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    at.radio[0].set_value("Holdings").run()
    assert not at.exception, at.exception
    assert [m.label for m in at.metric][:2] == ["Market value", "Open P&L"]
    # 11 + 50 + 50 in value; P&L counts only AAA and BBB, which have a cost basis.
    assert at.metric[0].value == "$111.00"
    assert at.metric[1].value == "$+11.00"
    assert any("Robinhood as of" in c.value for c in at.caption)
    assert at.dataframe
    # Backtest the holdings universe (synthetic prices offline; quotes are not needed).
    at.selectbox[1].set_value("drift")
    at.button[0].click().run()
    assert not at.exception, at.exception
    labels = [m.label for m in at.metric]
    assert "Sharpe" in labels and "Market value" in labels
