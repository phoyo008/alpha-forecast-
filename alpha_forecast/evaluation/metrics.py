"""Forecast and strategy evaluation metrics.

Two complementary views:
  * statistical accuracy of the forecast (RMSE/MAE/R2), and
  * economic value of trading on the forecast (Sharpe, drawdown, hit rate).

A forecast can have poor RMSE yet still be tradable if it gets the *direction*
right often enough -- so both views matter.

Targets throughout the project are forward *log* returns, so equity curves
compound with ``exp(cumsum(r))`` rather than ``cumprod(1 + r)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
CRYPTO_DAYS = 365


def forecast_error_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    err = (y_pred - y_true).to_numpy()
    y_t = y_true.to_numpy()
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_t - y_t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": len(y_true)}


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of predictions that get the sign of the return right.

    A zero forecast counts as long, matching the position taken by
    :func:`strategy_returns`.
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    predicted_sign = np.sign(y_pred[mask]).replace(0, 1.0)
    correct = predicted_sign == np.sign(y_true[mask])
    return float(correct.mean())


def sharpe_ratio(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    """Annualised Sharpe ratio of a per-period return series (zero risk-free)."""
    std = returns.std()
    if not np.isfinite(std) or std <= 0:
        return float("nan")
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def log_equity(log_returns: pd.Series) -> pd.Series:
    """Growth of $1 from a series of log returns."""
    return np.exp(log_returns.cumsum())


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def strategy_returns(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    cost_bps: float = 1.0,
    horizon: int = 1,
) -> pd.Series:
    """Net log returns of a long/short strategy driven by the forecast sign.

    Position is +1 when the model predicts a positive forward return, -1
    otherwise. For ``horizon > 1`` consecutive targets overlap, so only every
    ``horizon``-th forecast is traded and each position is held for the full
    horizon -- giving one independent return per holding period.

    ``cost_bps`` is charged per unit of notional traded: entering a position
    costs 1x, flipping from long to short costs 2x.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    y_true = y_true.iloc[::horizon]
    y_pred = y_pred.loc[y_true.index]

    position = np.sign(y_pred).replace(0, 1.0)
    traded = position.diff().abs()
    traded.iloc[0] = abs(position.iloc[0])
    gross = position * y_true
    net = gross - traded * (cost_bps / 1e4)
    net.name = "strategy"
    return net


def strategy_performance(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    cost_bps: float = 1.0,
    horizon: int = 1,
    days_per_year: float = TRADING_DAYS,
) -> dict:
    """Evaluate the sign-driven long/short strategy (see :func:`strategy_returns`).

    ``days_per_year`` is 252 for exchange-traded assets and 365 for crypto,
    which trades every calendar day.
    """
    net = strategy_returns(y_true, y_pred, cost_bps=cost_bps, horizon=horizon)
    periods_per_year = days_per_year / horizon
    equity = log_equity(net)

    return {
        "sharpe": sharpe_ratio(net, periods_per_year),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": float((net > 0).mean()),
        "ann_return": float(net.mean() * periods_per_year),
        "ann_vol": float(net.std() * np.sqrt(periods_per_year)),
        "n_periods": len(net),
    }
