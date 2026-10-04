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


def _collect_extra_features(
    prices: pd.DataFrame,
    *,
    use_factors: bool,
    use_macro: bool,
) -> pd.DataFrame | None:
    """Assemble optional Fama-French factor and FRED macro features."""
    frames = []
    start = prices.index.min().date().isoformat()
    end = prices.index.max().date().isoformat()

    if use_factors:
        try:
            from alpha_forecast.features.factors import load_fama_french

            frames.append(load_fama_french(start, end))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load Fama-French factors: %s", exc)

    if use_macro:
        try:
            from alpha_forecast.features.macro import load_macro

            m = load_macro(start, end)
            if m is not None:
                frames.append(m)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load macro features: %s", exc)

    if not frames:
        return None
    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(f, how="outer")
    return combined


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
    use_factors: bool = False,
    use_macro: bool = False,
) -> pd.DataFrame:
    """Run every requested model through walk-forward backtesting and compare.

    Returns a leaderboard DataFrame (one row per model) sorted by Sharpe.
    """
    models = models or list(MODEL_REGISTRY)
    prices = load_prices(symbol, start, end)
    extra = _collect_extra_features(prices, use_factors=use_factors, use_macro=use_macro)
    X, y = build_features(prices, horizon=horizon, extra=extra)
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
