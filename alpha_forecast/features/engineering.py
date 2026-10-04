"""Feature engineering for return forecasting.

All features are computed in a strictly causal way (only past information)
so the resulting matrix can be fed to walk-forward backtesting without
look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(
    prices: pd.DataFrame,
    *,
    horizon: int = 1,
    extra: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a causal feature matrix ``X`` and forward-return target ``y``.

    The target ``y`` is the log return realised ``horizon`` days ahead:
        y_t = log(close_{t+horizon}) - log(close_t)

    Parameters
    ----------
    prices:
        OHLCV DataFrame indexed by date.
    horizon:
        Forecast horizon in trading days.
    extra:
        Optional DataFrame of additional causal features (e.g. Fama-French
        factors or macro indicators) indexed by date. Columns are forward
        filled and lagged by one day to guarantee they were observable at the
        close of day ``t`` before being joined to the price features.

    Returns ``(X, y)`` aligned on the same index, with rows containing any
    NaN (from warm-up windows or the forward shift) dropped.
    """
    close = prices["close"].astype(float)
    log_ret = np.log(close).diff()

    feats = pd.DataFrame(index=prices.index)

    # Momentum / lagged returns.
    # All computed from log_ret up to and including day t (known at close t),
    # which is strictly prior to the forward target that starts at t+1.
    feats["ret_1"] = log_ret                      # today's realised return
    feats["ret_prev_1"] = log_ret.shift(1)        # yesterday's
    for w in (3, 5, 10):
        feats[f"mom_{w}"] = log_ret.rolling(w).sum()  # cumulative momentum over last w days

    # Moving-average ratios (price relative to its own MA)
    for w in (5, 10, 20, 50):
        ma = close.rolling(w).mean()
        feats[f"ma_ratio_{w}"] = (close / ma - 1.0)

    # Realised volatility
    for w in (5, 10, 20):
        feats[f"vol_{w}"] = log_ret.rolling(w).std()

    # RSI
    feats["rsi_14"] = _rsi(close, 14)

    # Volume signal (if present)
    if "volume" in prices.columns:
        vol = prices["volume"].astype(float)
        feats["vol_z_20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()

    # Optional external features (factors / macro). Reindex to the price
    # calendar, forward-fill gaps (e.g. monthly macro series), then lag by one
    # day so a value is only used after it would have been observable.
    if extra is not None and not extra.empty:
        ext = extra.copy()
        ext.index = pd.to_datetime(ext.index)
        ext = ext.reindex(prices.index).ffill().shift(1)
        ext.columns = [f"ext_{c}" for c in ext.columns]
        feats = feats.join(ext)

    # Target: forward log return
    target = np.log(close).shift(-horizon) - np.log(close)
    target.name = f"fwd_ret_{horizon}"

    feature_cols = list(feats.columns)
    data = feats.join(target).dropna()
    X = data[feature_cols]
    y = data[target.name]
    return X, y
