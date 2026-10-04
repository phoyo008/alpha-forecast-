"""Multi-asset portfolio backtest.

Ties the whole project together: forecast each asset's forward return, then at
each rebalance date allocate capital across assets using one of the optimizer
schemes, and measure realised portfolio performance.

This is deliberately a *cross-sectional* use of the forecasts (which asset to
favour), complementing the single-asset timing strategy in ``evaluation``.

Timing convention: weights chosen at the close of day ``t`` (using the
forecast and covariance known at ``t``) earn the simple return from ``t`` to
``t + 1``. Between rebalances the holdings drift with prices; transaction
costs are charged on the turnover needed to get back to target.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha_forecast.backtest import walk_forward_backtest
from alpha_forecast.data import load_prices
from alpha_forecast.evaluation.metrics import TRADING_DAYS, max_drawdown, sharpe_ratio
from alpha_forecast.features import build_features
from alpha_forecast.models import MODEL_REGISTRY
from alpha_forecast.portfolio.optimizer import (
    black_litterman,
    equal_weight,
    mean_variance_weights,
    risk_parity_weights,
)

logger = logging.getLogger(__name__)

SCHEMES = ("equal", "mean_variance", "risk_parity", "black_litterman")


def _oos_forecasts(symbol, model, horizon, initial_train, step, start, end):
    """Out-of-sample forecasts plus realised (t-1 -> t) and forward (t -> t+1) returns."""
    prices = load_prices(symbol, start, end)
    X, y = build_features(prices, horizon=horizon)
    res = walk_forward_backtest(
        MODEL_REGISTRY[model], X, y, initial_train=initial_train, step=step, horizon=horizon
    )
    realised = prices["close"].astype(float).pct_change()
    forward = realised.shift(-1)
    return res.predictions.rename(symbol), realised.rename(symbol), forward.rename(symbol)


def _target_weights(
    scheme: str,
    symbols: list[str],
    mu: pd.Series,
    cov: pd.DataFrame | None,
    risk_aversion: float,
) -> pd.Series:
    if scheme == "equal" or cov is None:
        return equal_weight(symbols)
    if scheme == "risk_parity":
        return risk_parity_weights(cov)
    if scheme == "black_litterman":
        mu_bl, cov_bl = black_litterman(cov, views=mu, risk_aversion=risk_aversion)
        return mean_variance_weights(mu_bl, cov_bl, risk_aversion=risk_aversion)
    return mean_variance_weights(mu, cov, risk_aversion=risk_aversion)


def portfolio_backtest(
    symbols: list[str],
    *,
    scheme: str = "mean_variance",
    model: str = "gbm",
    start: str | None = None,
    end: str | None = None,
    horizon: int = 1,
    initial_train: int = 252,
    step: int = 21,
    rebalance: int = 21,
    cov_window: int = 60,
    risk_aversion: float = 10.0,
    cost_bps: float = 1.0,
) -> dict:
    """Backtest a cross-sectional portfolio built from per-asset forecasts.

    Parameters
    ----------
    symbols:
        Universe of tickers.
    scheme:
        ``"equal"``, ``"mean_variance"``, ``"risk_parity"`` or ``"black_litterman"``.
    model:
        Which forecasting model to use for expected returns.
    rebalance:
        Rebalance frequency in trading days.
    cov_window:
        Trailing window (days) for the sample covariance estimate.
    cost_bps:
        Transaction cost per unit of notional traded at each rebalance.

    Returns a dict with the realised daily portfolio returns and summary stats.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme '{scheme}'; choose from {SCHEMES}")
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model}'; choose from {list(MODEL_REGISTRY)}")

    preds, realised, forward = {}, {}, {}
    for s in symbols:
        preds[s], realised[s], forward[s] = _oos_forecasts(
            s, model, horizon, initial_train, step, start, end
        )

    pred_df = pd.DataFrame(preds).dropna()
    realised_df = pd.DataFrame(realised)
    fwd_df = pd.DataFrame(forward).reindex(pred_df.index).dropna()
    dates = pred_df.index.intersection(fwd_df.index)
    pred_df, fwd_df = pred_df.loc[dates], fwd_df.loc[dates]

    weights_history = []
    port_ret = pd.Series(0.0, index=dates, name="portfolio")
    held = pd.Series(0.0, index=symbols)
    total_turnover = 0.0
    cost_rate = cost_bps / 1e4

    for i, dt in enumerate(dates):
        turnover = 0.0
        if i % rebalance == 0:
            window = realised_df.loc[:dt].dropna().iloc[-cov_window:]
            cov = window.cov() * TRADING_DAYS if len(window) >= cov_window else None
            mu = pred_df.loc[dt] * (TRADING_DAYS / horizon)
            target = _target_weights(scheme, symbols, mu, cov, risk_aversion).reindex(symbols)
            target = target.fillna(0.0)
            turnover = float((target - held).abs().sum())
            total_turnover += turnover
            held = target
            weights_history.append((dt, target))

        asset_ret = fwd_df.loc[dt]
        gross = float((held * asset_ret).sum())
        port_ret.loc[dt] = gross - turnover * cost_rate
        grown = held * (1.0 + asset_ret)
        held = grown / grown.sum() if grown.sum() > 0 else held

    equity = (1 + port_ret).cumprod()
    n_years = len(port_ret) / TRADING_DAYS

    return {
        "scheme": scheme,
        "model": model,
        "returns": port_ret,
        "equity": equity,
        "sharpe": sharpe_ratio(port_ret),
        "ann_return": float(port_ret.mean() * TRADING_DAYS),
        "ann_vol": float(port_ret.std() * np.sqrt(TRADING_DAYS)),
        "max_drawdown": max_drawdown(equity),
        "ann_turnover": total_turnover / n_years if n_years > 0 else float("nan"),
        "weights_history": weights_history,
    }
