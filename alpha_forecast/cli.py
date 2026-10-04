"""Command-line interface for alpha-forecast.

Example:
    python -m alpha_forecast.cli --symbol AAPL --horizon 1
    python -m alpha_forecast.cli --symbol MSFT --models naive gbm neural --cost-bps 2
    python -m alpha_forecast.cli --portfolio AAPL MSFT GOOG --scheme black_litterman
    python -m alpha_forecast.cli --symbol BTC-USD --onchain -v
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from alpha_forecast.models import MODEL_REGISTRY
from alpha_forecast.pipeline import run_comparison
from alpha_forecast.portfolio.backtest import SCHEMES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alpha-forecast",
        description="Multi-model equity return forecasting & backtesting.",
    )
    p.add_argument(
        "--portfolio",
        nargs="+",
        default=None,
        metavar="SYMBOL",
        help="Run a multi-asset portfolio backtest over these tickers instead "
        "of a single-symbol leaderboard.",
    )
    p.add_argument(
        "--scheme",
        default="mean_variance",
        choices=list(SCHEMES),
        help="Portfolio allocation scheme (with --portfolio)",
    )
    p.add_argument("--symbol", default="AAPL", help="Ticker symbol (default: AAPL)")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon in days")
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY),
        choices=list(MODEL_REGISTRY),
        help="Which models to compare (portfolio mode uses the first)",
    )
    p.add_argument("--initial-train", type=int, default=252)
    p.add_argument("--step", type=int, default=21)
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument(
        "--factors",
        action="store_true",
        help="Add Fama-French factor features (free, no API key)",
    )
    p.add_argument(
        "--macro",
        action="store_true",
        help="Add FRED macro features (requires FRED_API_KEY env var)",
    )
    p.add_argument(
        "--onchain",
        action="store_true",
        help="Add Blockchair on-chain features for crypto symbols such as BTC-USD "
        "(requires BLOCKCHAIR_API_KEY env var)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Skip network data providers and use synthetic data",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the on-disk price cache",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _format_board(board):
    out = board.copy()
    for col in ("r2", "dir_acc", "ann_return", "max_dd"):
        out[col] = (out[col] * 100).round(2).astype(str) + "%"
    out["rmse"] = out["rmse"].round(5)
    out["sharpe"] = out["sharpe"].round(2)
    out["sharpe_95ci"] = (
        "[" + out.pop("sharpe_lo").round(2).astype(str)
        + ", " + out.pop("sharpe_hi").round(2).astype(str) + "]"
    )
    out["dm_pvalue"] = out["dm_pvalue"].round(3)
    cols = ["model", "rmse", "r2", "dir_acc", "sharpe", "sharpe_95ci",
            "ann_return", "max_dd", "dm_pvalue", "n_oos"]
    return out[cols]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.offline:
        os.environ["ALPHA_FORECAST_OFFLINE"] = "1"
    if args.no_cache:
        os.environ["ALPHA_FORECAST_CACHE_DIR"] = ""

    # Multi-asset portfolio mode
    if args.portfolio:
        from alpha_forecast.portfolio.backtest import portfolio_backtest

        model = args.models[0] if args.models else "gbm"
        out = portfolio_backtest(
            args.portfolio,
            scheme=args.scheme,
            model=model,
            start=args.start,
            end=args.end,
            horizon=args.horizon,
            initial_train=args.initial_train,
            step=args.step,
            cost_bps=args.cost_bps,
        )
        print(
            f"\n=== portfolio backtest: {', '.join(args.portfolio)} "
            f"[{args.scheme}, model={model}] ===\n"
        )
        print(f"  Sharpe       : {out['sharpe']:.2f}")
        print(f"  Ann. return  : {out['ann_return'] * 100:.2f}%")
        print(f"  Ann. vol     : {out['ann_vol'] * 100:.2f}%")
        print(f"  Max drawdown : {out['max_drawdown'] * 100:.2f}%")
        print(f"  Ann. turnover: {out['ann_turnover']:.2f}x")
        print(f"  Rebalances   : {len(out['weights_history'])}\n")
        return 0

    board = run_comparison(
        args.symbol,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        models=args.models,
        initial_train=args.initial_train,
        step=args.step,
        cost_bps=args.cost_bps,
        use_factors=args.factors,
        use_macro=args.macro,
        use_onchain=args.onchain,
    )
    print(f"\n=== alpha-forecast leaderboard: {args.symbol} (horizon={args.horizon}d) ===\n")
    print(_format_board(board).to_string(index=False))
    print(
        "\nsharpe_95ci: block-bootstrap interval. dm_pvalue: one-sided Diebold-Mariano test "
        "that the model's squared error beats the naive baseline.\n"
        "A Sharpe interval spanning 0 or dm_pvalue > 0.05 is not evidence of real alpha.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
