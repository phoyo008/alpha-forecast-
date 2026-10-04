"""Plotting helpers for backtest results.

Uses matplotlib if available; raises a clear error otherwise. Kept separate
from the core so the engine has no hard plotting dependency.
"""

from __future__ import annotations

import pandas as pd

from alpha_forecast.evaluation.metrics import log_equity, strategy_returns


def _require_mpl():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc


def equity_curve(
    actuals: pd.Series,
    predictions: pd.Series,
    *,
    cost_bps: float = 1.0,
    horizon: int = 1,
):
    """Return a matplotlib Figure of the net strategy equity curve vs buy & hold."""
    plt = _require_mpl()
    net = strategy_returns(actuals, predictions, cost_bps=cost_bps, horizon=horizon)
    strat = log_equity(net)
    bh = log_equity(actuals.loc[net.index])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strat.index, strat.values, label="Forecast strategy")
    ax.plot(bh.index, bh.values, label="Buy & hold", alpha=0.7)
    ax.set_title("Out-of-sample equity curve")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def prediction_scatter(actuals: pd.Series, predictions: pd.Series):
    """Scatter of predicted vs actual forward returns."""
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actuals.values, predictions.values, s=8, alpha=0.4)
    lim = max(abs(actuals).max(), abs(predictions).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", alpha=0.5)
    ax.set_xlabel("Actual forward return")
    ax.set_ylabel("Predicted forward return")
    ax.set_title("Prediction vs actual")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
