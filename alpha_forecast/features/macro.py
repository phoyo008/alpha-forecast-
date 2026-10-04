"""Macroeconomic features from FRED.

Pulls a small, curated set of macro series that are known to carry
cross-asset information:

    * DGS10 / DGS2  -> 10y and 2y Treasury yields (we derive the slope)
    * T10YIE        -> 10y breakeven inflation
    * VIXCLS        -> CBOE volatility index (risk regime)

Requires a free FRED API key, read from the ``FRED_API_KEY`` environment
variable (never hard-coded). If the key or the ``fredapi`` package is absent,
returns ``None`` so the caller can simply skip macro features.

Get a free key at: https://fredaccount.stlouisfed.org/apikeys
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from alpha_forecast.config import is_offline

logger = logging.getLogger(__name__)

DEFAULT_SERIES = {
    "DGS10": "y10",
    "DGS2": "y2",
    "T10YIE": "breakeven_10y",
    "VIXCLS": "vix",
}


def load_macro(
    start: str,
    end: str,
    *,
    series: dict[str, str] | None = None,
    api_key: str | None = None,
) -> pd.DataFrame | None:
    """Load macro features from FRED, or ``None`` if unavailable.

    Derives a ``yield_slope`` column (10y - 2y) when both are present, a
    classic recession / risk indicator.
    """
    if is_offline():
        logger.info("Offline mode; skipping macro features.")
        return None
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.info("No FRED_API_KEY set; skipping macro features.")
        return None
    try:
        from fredapi import Fred  # type: ignore
    except Exception:
        logger.info("fredapi not installed; skipping macro features.")
        return None

    series = series or DEFAULT_SERIES
    try:
        fred = Fred(api_key=api_key)
        cols = {}
        for code, name in series.items():
            s = fred.get_series(code, observation_start=start, observation_end=end)
            cols[name] = s
        df = pd.DataFrame(cols)
        df.index = pd.to_datetime(df.index)
        if "y10" in df.columns and "y2" in df.columns:
            df["yield_slope"] = df["y10"] - df["y2"]
        logger.info("Loaded %d macro series from FRED (%d rows)", len(cols), len(df))
        return df
    except Exception as exc:  # pragma: no cover - network/key dependent
        logger.warning("FRED load failed: %s", exc)
        return None
