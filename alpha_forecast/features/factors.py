"""Fama-French factor features.

Loads the daily Fama-French research factors (Mkt-RF, SMB, HML, RF, and the
momentum factor when available). These are classic drivers of cross-sectional
equity returns, and including an asset's recent exposure to them is a standard
quant technique.

Provider order:
    1. OpenBB Platform  (``obb.economy.fama_french`` / famafrench extension)
    2. pandas-datareader -> Ken French Data Library
    3. Synthetic reproducible factors (offline fallback)

Returned frame is indexed by date with factor columns in *decimal* units
(e.g. 0.0012 == 0.12%), ready to pass as ``extra`` to ``build_features``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rf"]


def _from_openbb(start: str, end: str) -> pd.DataFrame | None:
    try:
        from openbb import obb  # type: ignore
    except Exception:
        return None
    try:
        # The famafrench extension exposes factor datasets. API surface has
        # shifted across OpenBB versions, so we probe defensively.
        res = obb.economy.fama_french.factors(
            dataset="famafrench", factor="5_factors", frequency="daily",
            start_date=start, end_date=end,
        )  # type: ignore[attr-defined]
        df = res.to_dataframe()
        df.columns = [str(c).lower().replace("-", "_").strip() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        rename = {"mkt_rf": "mkt_rf", "mktrf": "mkt_rf", "mkt": "mkt_rf"}
        df = df.rename(columns=rename)
        keep = [c for c in FACTOR_COLUMNS if c in df.columns]
        if not keep:
            return None
        logger.info("Loaded Fama-French factors from OpenBB (%d rows)", len(df))
        # OpenBB returns percent; convert to decimal.
        return df[keep] / 100.0
    except Exception as exc:  # pragma: no cover - network/version dependent
        logger.warning("OpenBB Fama-French load failed: %s", exc)
        return None


def _from_datareader(start: str, end: str) -> pd.DataFrame | None:
    try:
        import pandas_datareader.data as web  # type: ignore
    except Exception:
        return None
    try:
        ff = web.DataReader(
            "F-F_Research_Data_Factors_daily", "famafrench", start, end
        )[0]
        ff.columns = [str(c).lower().replace("-", "_").strip() for c in ff.columns]
        ff.index = pd.to_datetime(ff.index)
        keep = [c for c in FACTOR_COLUMNS if c in ff.columns]
        logger.info("Loaded Fama-French factors from pandas-datareader (%d rows)", len(ff))
        return ff[keep] / 100.0
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("pandas-datareader Fama-French load failed: %s", exc)
        return None


def _synthetic(start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start=pd.to_datetime(start), end=pd.to_datetime(end))
    if len(dates) == 0:
        dates = pd.bdate_range(start=pd.to_datetime(start), periods=252)
    rng = np.random.default_rng(42)
    n = len(dates)
    df = pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.0003, 0.010, n),
            "smb": rng.normal(0.0000, 0.005, n),
            "hml": rng.normal(0.0000, 0.005, n),
            "rf": np.full(n, 0.00008),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    logger.warning("Falling back to SYNTHETIC Fama-French factors")
    return df


def load_fama_french(start: str, end: str) -> pd.DataFrame:
    """Load daily Fama-French factors in decimal units, indexed by date."""
    for provider in (_from_openbb, _from_datareader):
        df = provider(start, end)
        if df is not None and not df.empty:
            return df.dropna(how="all")
    return _synthetic(start, end)
