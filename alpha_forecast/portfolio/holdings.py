"""Brokerage holdings from a local CSV, valued at the latest quotes.

Robinhood exposes account data to AI agents through its Trading MCP server
(OAuth in the editor), not through a library this app can import. The bridge
is a small CSV that the agent writes and this module reads. It lives outside
the repository (see ``ALPHA_FORECAST_HOLDINGS`` in :mod:`alpha_forecast.config`)
so personal positions are never committed.

CSV columns::

    account,asset_class,symbol,quantity,avg_cost

``account`` is a free label (last four digits is plenty). ``asset_class`` is
``equity`` or ``crypto``. ``avg_cost`` may be blank when the broker has no
cost basis for the units.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from alpha_forecast.config import holdings_path

logger = logging.getLogger(__name__)

HOLDINGS_COLUMNS = ["account", "asset_class", "symbol", "quantity", "avg_cost"]
VALUED_COLUMNS = HOLDINGS_COLUMNS + [
    "quote_symbol", "price", "market_value", "cost_basis", "pnl", "pnl_pct", "weight",
]


def quote_symbol(symbol: str, asset_class: str) -> str:
    """Ticker to request from the quote provider (crypto needs a ``-USD`` pair)."""
    s = symbol.strip().upper()
    if asset_class.strip().lower() == "crypto" and "-" not in s:
        return f"{s}-USD"
    return s


def load_holdings(path: Path | None = None) -> pd.DataFrame:
    """Read the holdings CSV; an empty frame with the right columns if absent."""
    path = path or holdings_path()
    if not path.exists():
        return pd.DataFrame(columns=HOLDINGS_COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read holdings file %s: %s", path, exc)
        return pd.DataFrame(columns=HOLDINGS_COLUMNS)
    missing = [c for c in HOLDINGS_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Holdings file %s is missing columns %s", path, missing)
        return pd.DataFrame(columns=HOLDINGS_COLUMNS)
    df = df[HOLDINGS_COLUMNS].copy()
    df["account"] = df["account"].astype(str)
    df["asset_class"] = df["asset_class"].astype(str).str.lower().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")
    df = df.dropna(subset=["quantity"])
    return df[df["quantity"] > 0].reset_index(drop=True)


def value_holdings(holdings: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    """Attach prices, market value, P&L and portfolio weight to each holding.

    ``quotes`` is the frame from :func:`alpha_forecast.data.latest_quotes`.
    Rows without a quote keep NaN values and zero weight.
    """
    if holdings.empty:
        return pd.DataFrame(columns=VALUED_COLUMNS)
    out = holdings.copy()
    out["quote_symbol"] = [
        quote_symbol(s, a) for s, a in zip(out["symbol"], out["asset_class"], strict=True)
    ]
    price_by_symbol = (
        quotes.set_index("symbol")["price"] if not quotes.empty else pd.Series(dtype=float)
    )
    out["price"] = out["quote_symbol"].map(price_by_symbol).astype(float)
    out["market_value"] = out["price"] * out["quantity"]
    out["cost_basis"] = out["avg_cost"] * out["quantity"]
    out["pnl"] = out["market_value"] - out["cost_basis"]
    out["pnl_pct"] = out["pnl"] / out["cost_basis"].where(out["cost_basis"] > 0)
    total = out["market_value"].sum(skipna=True)
    out["weight"] = (out["market_value"] / total).fillna(0.0) if total > 0 else 0.0
    return out[VALUED_COLUMNS]


def aggregate_by_symbol(valued: pd.DataFrame) -> pd.DataFrame:
    """Combine the same symbol across accounts (for backtesting the universe)."""
    if valued.empty:
        return valued
    grouped = (
        valued.groupby(["quote_symbol", "asset_class"], as_index=False)
        .agg(quantity=("quantity", "sum"), market_value=("market_value", "sum"))
    )
    total = grouped["market_value"].sum(skipna=True)
    grouped["weight"] = grouped["market_value"] / total if total > 0 else 0.0
    return grouped.sort_values("market_value", ascending=False).reset_index(drop=True)
