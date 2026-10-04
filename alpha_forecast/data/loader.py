"""Price data loading.

Tries real data providers in order of preference:
    1. OpenBB Platform  (``obb.equity.price.historical``)
    2. yfinance

If neither is available (e.g. offline / no network), falls back to a
reproducible synthetic series generated with geometric Brownian motion so
that the rest of the pipeline is always runnable and testable.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def _from_openbb(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        from openbb import obb  # type: ignore
    except Exception:
        return None
    try:
        res = obb.equity.price.historical(symbol, start_date=start, end_date=end)
        df = res.to_dataframe()
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        logger.info("Loaded %s from OpenBB (%d rows)", symbol, len(df))
        return df
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("OpenBB load failed for %s: %s", symbol, exc)
        return None


def _from_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return None
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        df.columns = [str(c[0]).lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        logger.info("Loaded %s from yfinance (%d rows)", symbol, len(df))
        return df
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("yfinance load failed for %s: %s", symbol, exc)
        return None


def _synthetic(symbol: str, start: str, end: str, seed: int | None = None) -> pd.DataFrame:
    """Generate a reproducible GBM price series with weekday business dates."""
    start_d = pd.to_datetime(start).date()
    end_d = pd.to_datetime(end).date()
    dates = pd.bdate_range(start=start_d, end=end_d)
    n = len(dates)
    if n == 0:
        dates = pd.bdate_range(start=start_d, periods=252)
        n = len(dates)

    # Seed from the symbol so each ticker is distinct but reproducible.
    seed = seed if seed is not None else (abs(hash(symbol)) % (2**32))
    rng = np.random.default_rng(seed)

    mu = 0.08 / 252          # ~8% annual drift
    sigma = 0.20 / np.sqrt(252)  # ~20% annual vol
    shocks = rng.normal(mu - 0.5 * sigma**2, sigma, size=n)
    log_price = np.log(100.0) + np.cumsum(shocks)
    close = np.exp(log_price)

    intraday = np.abs(rng.normal(0, sigma, size=n))
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000_000, 10_000_000, size=n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    logger.info("Generated synthetic series for %s (%d rows, seed=%d)", symbol, n, seed)
    return df


def load_prices(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    *,
    allow_synthetic: bool = True,
) -> pd.DataFrame:
    """Load daily OHLCV prices for ``symbol``.

    Parameters
    ----------
    symbol:
        Ticker symbol, e.g. ``"AAPL"``.
    start, end:
        ISO date strings. Defaults to the last ~3 years.
    allow_synthetic:
        If True (default) fall back to synthetic data when no provider is
        reachable. Set False to force a failure instead.

    Returns
    -------
    DataFrame indexed by date with columns open/high/low/close/volume.
    """
    if end is None:
        end = date.today().isoformat()
    if start is None:
        start = (date.today() - timedelta(days=365 * 3)).isoformat()

    for provider in (_from_openbb, _from_yfinance):
        df = provider(symbol, start, end)
        if df is not None and not df.empty:
            return df[[c for c in PRICE_COLUMNS if c in df.columns]].dropna()

    if not allow_synthetic:
        raise RuntimeError(
            f"No data provider available for {symbol} and allow_synthetic=False"
        )
    logger.warning("Falling back to SYNTHETIC data for %s", symbol)
    return _synthetic(symbol, start, end)
