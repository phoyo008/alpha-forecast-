"""Multi-asset portfolio backtest.

Ties the whole project together: forecast each asset's forward return, then at
each rebalance date allocate capital across assets using one of the optimizer
schemes, and measure realised portfolio performance.

This is deliberately a *cross-sectional* use of the forecasts (which asset to
favour), complementing the single-asset timing strategy in ``evaluation``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha_forecast.backtest import walk_forward_backtest
from alpha_forecast.data import load_prices
from alpha_forecast.evaluation.metrics import TRADING_DAYS
from alpha_forecast.features import build_features
from alpha_forecast.models import MODEL_REGISTRY
from alpha_forecast.portfolio.optimizer import (
    equal_weight,
    mean_variance_weights,
    risk_parity_weights,
)

logger = logging.getLogger(__name__)


def _oos_forecasts(symbol, model, horizon, initial_train, step):
    prices = load_prices(symbol)
    X, y = build_features(prices, horizon=horizon)
    res = walk_forward_backtest(
        MODEL_REGISTRY[model], X, y, initial_train=initial_train, step=step
    )
    rets = np.log(prices["close"]).diff()
    return res.predictions.rename(symbol), rets.rename(symbol)


def portfolio_backtest(
    symbols: list[str],
    *,
    scheme: str = "mean_variance",
    model: str = "gbm",
    horizon: int = 1,
    initial_train: int = 252,
    step: int = 21,
    rebalance: int = 21,
    cov_window: int = 60,
    risk_aversion: float = 10.0,
) -> dict:
    """Backtest a cross-sectional portfolio built from per-asset forecasts.

    Parameters
    ----------
    symbols:
        Universe of tickers.
    scheme:
        ``"equal"``, ``"mean_variance"`` or ``"risk_parity"``.
    model:
        Which forecasting model to use for expected returns.
    rebalance:
        Rebalance frequency in trading days.
    cov_window:
        Trailing window (days) for the sample covariance estimate.

    Returns a dict with the realised daily portfolio returns and summary stats.
    """
    preds, rets = {}, {}
    for s in symbols:
        p, r = _oos_forecasts(s, model, horizon, initial_train, step)
        preds[s] = p
        rets[s] = r

    pred_df = pd.DataFrame(preds).dropna()
    ret_df = pd.DataFrame(rets).reindex(pred_df.index).dropna()
    common = pred_df.index.intersection(ret_df.index)
    pred_df, ret_df = pred_df.loc[common], ret_df.loc[common]

    dates = pred_df.index
    weights_history = []
    port_ret = pd.Series(0.0, index=dates)
    current_w = equal_weight(symbols)

    for i, dt in enumerate(dates):
        if i % rebalance == 0 and i >= cov_window:
            window = ret_df.iloc[i - cov_window : i]
            cov = window.cov() * TRADING_DAYS      # annualised
            mu = pred_df.loc[dt] * TRADING_DAYS     # annualised expected return

            if scheme == "equal":
                current_w = equal_weight(symbols)
            elif scheme == "risk_parity":
                current_w = risk_parity_weights(cov)
            else:  # mean_variance
                current_w = mean_variance_weights(
                    mu, cov, risk_aversion=risk_aversion
                )
            weights_history.append((dt, current_w))

        port_ret.loc[dt] = float((current_w.reindex(symbols).fillna(0) * ret_df.loc[dt]).sum())

    ann_factor = np.sqrt(TRADING_DAYS)
    mean, std = port_ret.mean(), port_ret.std()
    sharpe = float(ann_factor * mean / std) if std > 0 else float("nan")
    equity = (1 + port_ret).cumprod()
    max_dd = float((equity / equity.cummax() - 1).min())

    return {
        "scheme": scheme,
        "model": model,
        "returns": port_ret,
        "equity": equity,
        "sharpe": sharpe,
        "ann_return": float(mean * TRADING_DAYS),
        "ann_vol": float(std * ann_factor),
        "max_drawdown": max_dd,
        "weights_history": weights_history,
    }
