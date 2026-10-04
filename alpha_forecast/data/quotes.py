"""Latest traded prices.

Historical research data still comes from :func:`alpha_forecast.data.load_prices`.
This module is the small live read: the most recent price for a handful of
symbols, used by the dashboard before a backtest is run.

Two sources, in order:

1. A broker quote snapshot CSV (see ``ALPHA_FORECAST_QUOTES`` in
   :mod:`alpha_forecast.config`). Robinhood quotes reach this project only
   through its Trading MCP server in the editor, so the agent writes them to
   this file on request. Columns: ``symbol,price,previous_close,as_of,source``.
   The file is ignored once it is older than :data:`SNAPSHOT_MAX_AGE_HOURS`.
2. Yahoo Finance via ``yfinance`` (delayed daily closes) for any symbol the
   snapshot does not cover. Skipped when ``ALPHA_FORECAST_OFFLINE`` is set.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from alpha_forecast.config import is_offline, quotes_snapshot_path

logger = logging.getLogger(__name__)

QUOTE_COLUMNS = ["symbol", "price", "previous_close", "change", "as_of", "source"]
SNAPSHOT_COLUMNS = ["symbol", "price", "previous_close", "as_of", "source"]
SNAPSHOT_MAX_AGE_HOURS = 24.0
SOURCE_LABELS = {"robinhood": "Robinhood", "yfinance": "Yahoo Finance, delayed"}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=QUOTE_COLUMNS)


def _close_series(df: pd.DataFrame, symbol: str) -> pd.Series | None:
    if isinstance(df.columns, pd.MultiIndex):
        fields = df.columns.get_level_values(0)
        tickers = df.columns.get_level_values(1)
        if "Close" in fields:
            close = df["Close"]
            if isinstance(close, pd.Series):
                return close
            if symbol in close.columns:
                return close[symbol]
        if symbol in tickers:
            block = df.xs(symbol, axis=1, level=1)
            if "Close" in block.columns:
                return block["Close"]
        return None
    for name in ("Close", "close"):
        if name in df.columns:
            return df[name]
    return None


def quotes_from_download(df: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Turn a yfinance download frame into one row per symbol."""
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        close = _close_series(df, symbol)
        if close is None:
            continue
        close = close.dropna()
        if close.empty:
            continue
        price = float(close.iloc[-1])
        previous = float(close.iloc[-2]) if len(close) > 1 else float("nan")
        change = price / previous - 1.0 if previous == previous and previous != 0 else float("nan")
        as_of = pd.Timestamp(close.index[-1]).date().isoformat()
        rows.append(
            {
                "symbol": symbol,
                "price": price,
                "previous_close": previous,
                "change": change,
                "as_of": as_of,
                "source": "yfinance",
            }
        )
    if not rows:
        return _empty()
    return pd.DataFrame(rows, columns=QUOTE_COLUMNS)


def _download(symbols: list[str]) -> pd.DataFrame | None:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None
    try:
        df = yf.download(
            symbols,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Live quote download failed: %s", exc)
        return None
    if df is None or df.empty:
        return None
    return df


def read_snapshot(
    path: Path | None = None,
    *,
    max_age_hours: float = SNAPSHOT_MAX_AGE_HOURS,
) -> pd.DataFrame:
    """Read the broker quote snapshot; empty if absent, stale, or malformed.

    Staleness is judged by when the file was written, not by ``as_of``: a
    weekend snapshot legitimately holds Friday's stock prices.
    """
    path = path if path is not None else quotes_snapshot_path()
    if path is None or not path.exists():
        return _empty()
    if time.time() - path.stat().st_mtime > max_age_hours * 3600:
        logger.info("Quote snapshot %s is older than %.0fh; ignoring it", path, max_age_hours)
        return _empty()
    try:
        df = pd.read_csv(path, dtype={"as_of": str, "source": str})
    except Exception as exc:
        logger.warning("Could not read quote snapshot %s: %s", path, exc)
        return _empty()
    missing = [c for c in SNAPSHOT_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Quote snapshot %s is missing columns %s", path, missing)
        return _empty()
    df = df[SNAPSHOT_COLUMNS].copy()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["previous_close"] = pd.to_numeric(df["previous_close"], errors="coerce")
    df = df.dropna(subset=["price"]).drop_duplicates("symbol", keep="last")
    prev = df["previous_close"].where(df["previous_close"] > 0)
    df["change"] = df["price"] / prev - 1.0
    return df[QUOTE_COLUMNS].reset_index(drop=True)


def describe_sources(quotes: pd.DataFrame) -> str:
    """``"Robinhood as of 2026-10-04 12:45 ET · Yahoo Finance, delayed as of 2026-10-02"``."""
    parts = []
    for source, group in quotes.groupby("source", sort=False):
        label = SOURCE_LABELS.get(str(source), str(source))
        stamps = sorted(group["as_of"].astype(str))
        when = stamps[-1] if stamps[0] == stamps[-1] else f"{stamps[0]} to {stamps[-1]}"
        parts.append(f"{label} as of {when}")
    return " · ".join(parts)


def latest_quotes(symbols: Sequence[str]) -> pd.DataFrame:
    """Latest price for each symbol: broker snapshot first, then Yahoo Finance.

    Returns an empty frame when nothing is available. Columns are ``symbol``,
    ``price``, ``previous_close``, ``change`` (fraction vs the prior close),
    ``as_of``, and ``source``. Rows follow the order of ``symbols``.
    """
    cleaned: list[str] = []
    for raw in symbols:
        symbol = str(raw).strip().upper()
        if symbol and symbol not in cleaned:
            cleaned.append(symbol)
    if not cleaned:
        return _empty()

    frames = []
    snapshot = read_snapshot()
    if not snapshot.empty:
        frames.append(snapshot[snapshot["symbol"].isin(cleaned)])
    covered = set(frames[0]["symbol"]) if frames else set()
    remaining = [s for s in cleaned if s not in covered]
    if remaining and not is_offline():
        df = _download(remaining)
        if df is not None:
            frames.append(quotes_from_download(df, remaining))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty()
    out = pd.concat(frames, ignore_index=True)
    order = {s: i for i, s in enumerate(cleaned)}
    out = out.sort_values("symbol", key=lambda s: s.map(order)).reset_index(drop=True)
    return out[QUOTE_COLUMNS]
