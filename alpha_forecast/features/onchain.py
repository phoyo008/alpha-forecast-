"""On-chain features for crypto symbols, sourced from Blockchair.

Mirrors :mod:`alpha_forecast.features.macro`: returns ``None`` whenever the
data cannot be fetched so the pipeline simply proceeds without it. The
feature builder forward-fills and lags everything one day before joining, so
a day's on-chain activity is only used from the following close.
"""

from __future__ import annotations

import logging

import pandas as pd

from alpha_forecast.data.crypto import chain_for_symbol, daily_onchain, onchain_features

logger = logging.getLogger(__name__)


def load_onchain(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Daily on-chain feature frame for ``symbol``, or ``None`` if unavailable."""
    chain = chain_for_symbol(symbol)
    if chain is None:
        logger.info("%s is not a Blockchair-supported chain; skipping on-chain features.", symbol)
        return None
    raw = daily_onchain(chain, start, end)
    if raw is None or raw.empty:
        return None
    feats = onchain_features(raw)
    feats.columns = [f"onchain_{c}" for c in feats.columns]
    return feats
