"""End-to-end forecasting pipeline: data -> features -> models -> backtest -> report."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from alpha_forecast.backtest import BacktestResult, walk_forward_backtest
from alpha_forecast.data import is_crypto_symbol, load_prices
from alpha_forecast.evaluation import (
    bootstrap_sharpe_ci,
    diebold_mariano,
    directional_accuracy,
    forecast_error_metrics,
    strategy_performance,
    strategy_returns,
)
from alpha_forecast.evaluation.metrics import CRYPTO_DAYS, TRADING_DAYS
from alpha_forecast.features import build_features
from alpha_forecast.models import MODEL_REGISTRY

logger = logging.getLogger(__name__)

BASELINE = "naive"


@dataclass
class Comparison:
    """Everything produced by :func:`compare_models`."""

    symbol: str
    horizon: int
    cost_bps: float
    board: pd.DataFrame
    prices: pd.DataFrame
    results: dict[str, BacktestResult] = field(default_factory=dict)
    strategy: dict[str, pd.Series] = field(default_factory=dict)
    days_per_year: float = TRADING_DAYS
    extra_features: list[str] = field(default_factory=list)


def days_per_year_for(symbol: str) -> float:
    """365 for crypto pairs (traded every day), 252 otherwise."""
    return CRYPTO_DAYS if is_crypto_symbol(symbol) else TRADING_DAYS


def _collect_extra_features(
    prices: pd.DataFrame,
    *,
    use_factors: bool,
    use_macro: bool,
    use_onchain: bool = False,
    symbol: str = "",
) -> pd.DataFrame | None:
    """Assemble optional Fama-French, FRED macro, and Blockchair on-chain features."""
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

    if use_onchain:
        try:
            from alpha_forecast.features.onchain import load_onchain

            oc = load_onchain(symbol, start, end)
            if oc is not None:
                frames.append(oc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not load on-chain features: %s", exc)

    if not frames:
        return None
    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(f, how="outer")
    return combined


def compare_models(
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
    use_onchain: bool = False,
    n_boot: int = 1000,
) -> Comparison:
    """Run every requested model through walk-forward backtesting and compare.

    The leaderboard (sorted by Sharpe) includes a block-bootstrap 95% interval
    for the Sharpe ratio and the one-sided Diebold-Mariano p-value of each
    model's squared error against the naive baseline.

    Crypto pairs (``BTC-USD``) are annualised over 365 days; ``use_onchain``
    adds Blockchair on-chain activity features for supported chains.
    """
    models = models or list(MODEL_REGISTRY)
    prices = load_prices(symbol, start, end)
    extra = _collect_extra_features(
        prices, use_factors=use_factors, use_macro=use_macro,
        use_onchain=use_onchain, symbol=symbol,
    )
    X, y = build_features(prices, horizon=horizon, extra=extra)
    days_per_year = days_per_year_for(symbol)
    logger.info("Prepared %d samples with %d features for %s", len(X), X.shape[1], symbol)

    def run(name: str) -> BacktestResult:
        return walk_forward_backtest(
            MODEL_REGISTRY[name], X, y, initial_train=initial_train, step=step, horizon=horizon
        )

    results: dict[str, BacktestResult] = {}
    for name in models:
        if name not in MODEL_REGISTRY:
            logger.warning("Unknown model '%s' skipped", name)
            continue
        results[name] = run(name)

    baseline = results[BASELINE] if BASELINE in results else run(BASELINE)
    periods_per_year = days_per_year / horizon

    rows = []
    strategy: dict[str, pd.Series] = {}
    for name, result in results.items():
        err = forecast_error_metrics(result.actuals, result.predictions)
        da = directional_accuracy(result.actuals, result.predictions)
        strat = strategy_performance(
            result.actuals, result.predictions, cost_bps=cost_bps, horizon=horizon,
            days_per_year=days_per_year,
        )
        net = strategy_returns(
            result.actuals, result.predictions, cost_bps=cost_bps, horizon=horizon
        )
        strategy[name] = net
        ci = bootstrap_sharpe_ci(net, periods_per_year=periods_per_year, n_boot=n_boot)
        dm = diebold_mariano(
            result.actuals, result.predictions, baseline.predictions, horizon=horizon
        )
        rows.append(
            {
                "model": name,
                "rmse": err["rmse"],
                "r2": err["r2"],
                "dir_acc": da,
                "sharpe": strat["sharpe"],
                "sharpe_lo": ci["lo"],
                "sharpe_hi": ci["hi"],
                "ann_return": strat["ann_return"],
                "max_dd": strat["max_drawdown"],
                "dm_pvalue": dm["p_value"],
                "n_oos": err["n"],
            }
        )

    board = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return Comparison(
        symbol=symbol,
        horizon=horizon,
        cost_bps=cost_bps,
        board=board,
        prices=prices,
        results=results,
        strategy=strategy,
        days_per_year=days_per_year,
        extra_features=[c for c in X.columns if c.startswith("ext_")],
    )


def run_comparison(symbol: str, **kwargs) -> pd.DataFrame:
    """Leaderboard-only convenience wrapper around :func:`compare_models`."""
    return compare_models(symbol, **kwargs).board
