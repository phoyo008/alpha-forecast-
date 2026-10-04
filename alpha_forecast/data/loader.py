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
import time
import zlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_forecast.config import cache_dir, is_offline

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
CACHE_TTL_OPEN_RANGE = 12 * 3600  # seconds


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

    # Seed from the symbol so each ticker is distinct but reproducible across
    # processes (built-in hash() is salted per interpreter run).
    seed = seed if seed is not None else zlib.crc32(symbol.encode())
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


def _cache_path(symbol: str, start: str, end: str) -> Path | None:
    root = cache_dir()
    if root is None:
        return None
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in symbol.upper())
    return root / "prices" / f"{safe}_{start}_{end}.pkl"


def _read_cache(path: Path | None, end: str) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    # Ranges ending today (or later) are still growing; refresh them periodically.
    if pd.to_datetime(end).date() >= date.today():
        age = time.time() - path.stat().st_mtime
        if age > CACHE_TTL_OPEN_RANGE:
            return None
    try:
        return pd.read_pickle(path)
    except Exception as exc:
        logger.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None


def _write_cache(path: Path | None, df: pd.DataFrame) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)
    except OSError as exc:
        logger.warning("Could not write price cache %s: %s", path, exc)


def load_prices(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    *,
    allow_synthetic: bool = True,
    use_cache: bool = True,
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
    use_cache:
        Reuse provider downloads cached on disk (see :mod:`alpha_forecast.config`).
        Synthetic data is never cached.

    Returns
    -------
    DataFrame indexed by date with columns open/high/low/close/volume.
    ``df.attrs["source"]`` names the provider (``"openbb"``, ``"yfinance"``
    or ``"synthetic"``).
    """
    if end is None:
        end = date.today().isoformat()
    if start is None:
        start = (date.today() - timedelta(days=365 * 3)).isoformat()

    cache_file = _cache_path(symbol, start, end) if use_cache else None
    cached = _read_cache(cache_file, end)
    if cached is not None:
        logger.info("Loaded %s from cache (%d rows)", symbol, len(cached))
        return cached

    providers = () if is_offline() else (_from_openbb, _from_yfinance)
    for provider in providers:
        df = provider(symbol, start, end)
        if df is not None and not df.empty:
            df = df[[c for c in PRICE_COLUMNS if c in df.columns]].dropna()
            df.attrs["source"] = provider.__name__.removeprefix("_from_")
            _write_cache(cache_file, df)
            return df

    if not allow_synthetic:
        raise RuntimeError(
            f"No data provider available for {symbol} and allow_synthetic=False"
        )
    logger.warning("Falling back to SYNTHETIC data for %s", symbol)
    df = _synthetic(symbol, start, end)
    df.attrs["source"] = "synthetic"
    return df
