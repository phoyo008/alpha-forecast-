"""Smoke tests for the CLI entry point and plotting helpers."""

from __future__ import annotations

import pytest

from alpha_forecast.backtest import walk_forward_backtest
from alpha_forecast.cli import main
from alpha_forecast.data import load_prices
from alpha_forecast.features import build_features
from alpha_forecast.models import MODEL_REGISTRY

ARGS = ["--offline", "--start", "2019-01-01", "--end", "2022-12-31"]


def test_cli_leaderboard(capsys):
    assert main([*ARGS, "--symbol", "TEST", "--models", "naive", "drift"]) == 0
    out = capsys.readouterr().out
    assert "leaderboard: TEST" in out
    assert "sharpe_95ci" in out and "dm_pvalue" in out


def test_cli_portfolio(capsys):
    argv = [*ARGS, "--portfolio", "AAA", "BBB", "--scheme", "black_litterman",
            "--models", "drift", "--cost-bps", "2"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "black_litterman" in out and "Ann. turnover" in out


def test_plots_render():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from alpha_forecast.evaluation.plots import equity_curve, prediction_scatter

    X, y = build_features(load_prices("TEST", "2019-01-01", "2022-12-31"), horizon=5)
    res = walk_forward_backtest(MODEL_REGISTRY["drift"], X, y, horizon=5)
    assert equity_curve(res.actuals, res.predictions, horizon=5).axes
    assert prediction_scatter(res.actuals, res.predictions).axes
