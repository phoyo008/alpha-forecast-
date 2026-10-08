from datetime import date

from app.ingest.normalize import (
    extract_ticker,
    parse_amount_range,
    parse_date,
    parse_tx_type,
)
from app.models import TxType


def test_amount_ranges():
    assert parse_amount_range("$1,001 - $15,000") == (1001, 15000)
    assert parse_amount_range("$1,001 – $15,000") == (1001, 15000)  # en-dash
    assert parse_amount_range("$50,000,000 +") == (50000000, None)
    assert parse_amount_range(None) == (None, None)


def test_tx_type():
    assert parse_tx_type("Purchase") == TxType.BUY
    assert parse_tx_type("P") == TxType.BUY
    assert parse_tx_type("Sale (Full)") == TxType.SELL
    assert parse_tx_type("Exchange") == TxType.EXCHANGE


def test_extract_ticker():
    assert extract_ticker("Apple Inc. (AAPL) [ST]") == "AAPL"
    assert extract_ticker("Some Bond (ST)") is None  # asset-type code, not a ticker
    assert extract_ticker("No ticker here") is None


def test_parse_date():
    assert parse_date("01/15/2026") == date(2026, 1, 15)
    assert parse_date("2026-01-15") == date(2026, 1, 15)
    assert parse_date("garbage") is None
