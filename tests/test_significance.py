"""Tests for Diebold-Mariano, bootstrap Sharpe intervals, and the leaderboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_forecast.evaluation import bootstrap_sharpe_ci, diebold_mariano
from alpha_forecast.pipeline import compare_models


def _series(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_dm_detects_better_forecast():
    rng = np.random.default_rng(1)
    y = _series(rng.normal(0, 0.01, 500))
    good = y + rng.normal(0, 0.002, 500)
    bad = y + rng.normal(0, 0.02, 500)
    assert diebold_mariano(y, good, bad)["p_value"] < 0.01
    assert diebold_mariano(y, bad, good)["p_value"] > 0.99


def test_dm_equal_forecasts_is_not_significant():
    rng = np.random.default_rng(2)
    y = _series(rng.normal(0, 0.01, 500))
    a = _series(rng.normal(0, 0.01, 500))
    b = _series(rng.normal(0, 0.01, 500))
    res = diebold_mariano(y, a, b, horizon=5)
    assert 0.01 < res["p_value_two_sided"] <= 1.0


def test_dm_identical_forecasts_is_nan():
    y = _series(np.linspace(-0.01, 0.01, 50))
    p = _series(np.zeros(50))
    assert np.isnan(diebold_mariano(y, p, p)["stat"])


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(3)
    r = _series(rng.normal(0.0005, 0.01, 750))
    ci = bootstrap_sharpe_ci(r, n_boot=500)
    assert ci["lo"] < ci["sharpe"] < ci["hi"]
    assert 0.0 <= ci["p_le_zero"] <= 1.0


def test_bootstrap_ci_excludes_zero_for_strong_signal():
    rng = np.random.default_rng(4)
    r = _series(rng.normal(0.003, 0.01, 750))
    assert bootstrap_sharpe_ci(r, n_boot=500)["lo"] > 0


def test_bootstrap_ci_degenerate_input_is_nan():
    assert np.isnan(bootstrap_sharpe_ci(_series(np.zeros(50)))["sharpe"])


def test_compare_models_reports_significance():
    comp = compare_models(
        "TEST", start="2019-01-01", end="2022-12-31", models=["naive", "drift"], n_boot=200
    )
    board = comp.board
    assert {"sharpe_lo", "sharpe_hi", "dm_pvalue"}.issubset(board.columns)
    assert set(comp.results) == {"naive", "drift"}
    naive = board.set_index("model").loc["naive"]
    assert np.isnan(naive["dm_pvalue"])
    drift = board.set_index("model").loc["drift"]
    assert 0.0 <= drift["dm_pvalue"] <= 1.0
    assert drift["sharpe_lo"] <= drift["sharpe"] <= drift["sharpe_hi"]
    assert comp.strategy["drift"].index.equals(comp.results["drift"].predictions.index)
