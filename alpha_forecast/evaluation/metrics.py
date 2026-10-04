"""Forecast and strategy evaluation metrics.

Two complementary views:
  * statistical accuracy of the forecast (RMSE/MAE/R2), and
  * economic value of trading on the forecast (Sharpe, drawdown, hit rate).

A forecast can have poor RMSE yet still be tradable if it gets the *direction*
right often enough -- so both views matter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def forecast_error_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    err = (y_pred - y_true).to_numpy()
    y_t = y_true.to_numpy()
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_t - y_t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))}


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of predictions that get the sign of the return right."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    correct = np.sign(y_pred[mask]) == np.sign(y_true[mask])
    return float(correct.mean())


def strategy_performance(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    cost_bps: float = 1.0,
) -> dict:
    """Evaluate a simple long/short strategy driven by forecast sign.

    Position is +1 when the model predicts a positive forward return, -1
    otherwise. Realised strategy return = position * actual return, minus a
    per-trade transaction cost applied when the position flips.
    """
    position = np.sign(y_pred).replace(0, 1.0)
    gross = position * y_true

    turnover = position.diff().abs().fillna(0) / 2.0  # 1.0 == full flip
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost

    ann_factor = np.sqrt(TRADING_DAYS)
    mean, std = net.mean(), net.std()
    sharpe = float(ann_factor * mean / std) if std > 0 else float("nan")

    equity = (1 + net).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min())

    total_return = float(equity.iloc[-1] - 1.0)
    hit_rate = float((net > 0).mean())

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "ann_return": float(mean * TRADING_DAYS),
        "ann_vol": float(std * ann_factor),
    }
