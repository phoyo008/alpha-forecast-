"""Helpers to clean messy disclosure text into normalized fields."""

from __future__ import annotations

import re
from datetime import date, datetime

from app.models import TxType

# Congressional disclosures report a $ *range*, not an exact amount.
# These are the standard STOCK Act brackets.
AMOUNT_RANGES: dict[str, tuple[float, float]] = {
    "$1,001 - $15,000": (1001, 15000),
    "$15,001 - $50,000": (15001, 50000),
    "$50,001 - $100,000": (50001, 100000),
    "$100,001 - $250,000": (100001, 250000),
    "$250,001 - $500,000": (250001, 500000),
    "$500,001 - $1,000,000": (500001, 1000000),
    "$1,000,001 - $5,000,000": (1000001, 5000000),
    "$5,000,001 - $25,000,000": (5000001, 25000000),
    "$25,000,001 - $50,000,000": (25000001, 50000000),
    "$50,000,000 +": (50000000, None),
}

_TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)")


def parse_amount_range(text: str | None) -> tuple[float | None, float | None]:
    if not text:
        return None, None
    # Normalize whitespace/dashes before matching against the known brackets.
    cleaned = re.sub(r"\s+", " ", text).replace("–", "-").replace("—", "-").strip()
    for label, (low, high) in AMOUNT_RANGES.items():
        if cleaned == label:
            return low, high
    nums = [int(n.replace(",", "")) for n in re.findall(r"[\d,]+", cleaned)]
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), None
    return None, None


def parse_tx_type(text: str | None) -> TxType:
    t = (text or "").strip().lower()
    if t.startswith("p") or "purchase" in t or "buy" in t:
        return TxType.BUY
    if "exchange" in t:
        return TxType.EXCHANGE
    # Covers "sale", "sale (partial)", "sale (full)", "s".
    return TxType.SELL


def extract_ticker(text: str | None) -> str | None:
    """Pull a ticker from strings like 'Apple Inc. (AAPL) [ST]'."""
    if not text:
        return None
    m = _TICKER_RE.search(text)
    if m:
        candidate = m.group(1)
        # Filter obvious non-tickers occasionally captured in parentheses.
        if candidate not in {"ST", "OP", "PS", "RP", "N", "A"}:
            return candidate
    return None


def parse_date(text: str | None) -> date | None:
    if not text:
        return None
    text = text.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None
