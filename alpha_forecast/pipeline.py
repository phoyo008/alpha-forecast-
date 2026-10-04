"""End-to-end forecasting pipeline: data -> features -> models -> backtest -> report."""

from __future__ import annotations

import logging

import pandas as pd

from alpha_forecast.backtest import walk_forward_backtest
from alpha_forecast.data import load_prices
from alpha_forecast.evaluation import (
    directional_accuracy,
    forecast_error_metrics,
    strategy_performance,
)
from alpha_forecast.features import build_features
from alpha_forecast.models import MODEL_REGISTRY

logger = logging.getLogger(__name__)


def run_comparison(
    symbol: str,
    *,
    start: str | None = None,
    end: str | None = None,
    horizon: int = 1,
    models: list[str] | None = None,
    initial_train: int = 252,
    step: int = 21,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Run every requested model through walk-forward backtesting and compare.

    Returns a leaderboard DataFrame (one row per model) sorted by Sharpe.
    """
    models = models or list(MODEL_REGISTRY)
    prices = load_prices(symbol, start, end)
    X, y = build_features(prices, horizon=horizon)
    logger.info("Prepared %d samples with %d features for %s", len(X), X.shape[1], symbol)

    rows = []
    for name in models:
        if name not in MODEL_REGISTRY:
            logger.warning("Unknown model '%s' skipped", name)
            continue
        cls = MODEL_REGISTRY[name]
        result = walk_forward_backtest(
            cls, X, y, initial_train=initial_train, step=step
        )
        err = forecast_error_metrics(result.actuals, result.predictions)
        da = directional_accuracy(result.actuals, result.predictions)
        strat = strategy_performance(result.actuals, result.predictions, cost_bps=cost_bps)
        rows.append(
            {
                "model": name,
                "rmse": err["rmse"],
                "r2": err["r2"],
                "dir_acc": da,
                "sharpe": strat["sharpe"],
                "ann_return": strat["ann_return"],
                "max_dd": strat["max_drawdown"],
                "n_oos": err["n"],
            }
        )

    board = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return board
