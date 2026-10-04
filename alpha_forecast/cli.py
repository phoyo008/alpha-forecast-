"""Command-line interface for alpha-forecast.

Example:
    python -m alpha_forecast.cli --symbol AAPL --horizon 1
    python -m alpha_forecast.cli --symbol MSFT --models naive gbm --cost-bps 2
"""

from __future__ import annotations

import argparse
import logging
import sys

from alpha_forecast.models import MODEL_REGISTRY
from alpha_forecast.pipeline import run_comparison


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alpha-forecast",
        description="Multi-model equity return forecasting & backtesting.",
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
        help="Which models to compare",
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
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
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
    )
    print(f"\n=== alpha-forecast leaderboard: {args.symbol} (horizon={args.horizon}d) ===\n")
    with_pct = board.copy()
    for col in ("r2", "dir_acc", "ann_return", "max_dd"):
        with_pct[col] = (with_pct[col] * 100).round(2).astype(str) + "%"
    with_pct["rmse"] = with_pct["rmse"].round(5)
    with_pct["sharpe"] = with_pct["sharpe"].round(2)
    print(with_pct.to_string(index=False))
    print("\nNote: positive Sharpe that fails to beat the 'naive' baseline is not real alpha.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
