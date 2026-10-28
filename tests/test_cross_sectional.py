"""Tests for cross-sectional (sector-relative) momentum features.

All run offline on hand-built or synthetic price panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_forecast.data import load_prices
from alpha_forecast.features import build_features, cross_sectional_momentum
from alpha_forecast.portfolio.backtest import portfolio_backtest


def _closes(n=40, seed=0, symbols=("A", "B", "C", "D")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    rets = rng.normal(0.0005, 0.01, size=(n, len(symbols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=list(symbols))


def test_returns_one_frame_per_symbol_with_expected_columns():
    closes = _closes()
    feats = cross_sectional_momentum(closes, windows=(5, 10))
    assert set(feats) == set(closes.columns)
    expected = {f"{k}_{w}" for k in ("xs_rank", "xs_z", "sec_rel") for w in (5, 10)}
    for frame in feats.values():
        assert set(frame.columns) == expected
        assert frame.index.equals(closes.index)


def test_rank_and_zscore_are_cross_sectionally_consistent():
    feats = cross_sectional_momentum(_closes(), windows=(5,))
    dt = feats["A"].dropna().index[-1]
    ranks = pd.Series({s: f.loc[dt, "xs_rank_5"] for s, f in feats.items()})
    zs = pd.Series({s: f.loc[dt, "xs_z_5"] for s, f in feats.items()})
    assert sorted(ranks) == [0.25, 0.5, 0.75, 1.0]
    assert np.isclose(zs.mean(), 0.0, atol=1e-12)
    assert np.isclose(zs.std(ddof=0), 1.0)
    # Same ordering for both measures.
    assert list(ranks.sort_values().index) == list(zs.sort_values().index)


def test_sector_relative_uses_leave_one_out_sector_mean():
    closes = _closes()
    sectors = {"A": "tech", "B": "tech", "C": "tech", "D": "energy"}
    feats = cross_sectional_momentum(closes, windows=(5,), sectors=sectors)
    mom = np.log(closes).diff(5)
    dt = closes.index[-1]
    # A is compared with the mean of its sector peers B and C only.
    expected_a = mom.loc[dt, "A"] - mom.loc[dt, ["B", "C"]].mean()
    assert np.isclose(feats["A"].loc[dt, "sec_rel_5"], expected_a)
    # D has no sector peer, so it falls back to the rest of the universe.
    expected_d = mom.loc[dt, "D"] - mom.loc[dt, ["A", "B", "C"]].mean()
    assert np.isclose(feats["D"].loc[dt, "sec_rel_5"], expected_d)


def test_without_sectors_sector_relative_is_universe_relative():
    closes = _closes()
    feats = cross_sectional_momentum(closes, windows=(5,))
    mom = np.log(closes).diff(5)
    dt = closes.index[-1]
    expected = mom.loc[dt, "B"] - mom.loc[dt, ["A", "C", "D"]].mean()
    assert np.isclose(feats["B"].loc[dt, "sec_rel_5"], expected)


def test_features_are_causal():
    # Changing prices after a date must not change features on or before it.
    closes = _closes(n=60)
    cut = closes.index[40]
    shocked = closes.copy()
    shocked.loc[shocked.index > cut, "A"] *= 3.0
    base = cross_sectional_momentum(closes, windows=(5, 10))
    after = cross_sectional_momentum(shocked, windows=(5, 10))
    for s in closes.columns:
        pd.testing.assert_frame_equal(base[s].loc[:cut], after[s].loc[:cut])


def test_price_gaps_are_forward_filled():
    # An equity holiday in a mixed universe must not knock C out of the cross-section.
    closes = _closes(n=40)
    gappy = closes.copy()
    gappy.iloc[20:23, gappy.columns.get_loc("C")] = np.nan
    feats = cross_sectional_momentum(gappy, windows=(5,))
    filled = cross_sectional_momentum(gappy.ffill(), windows=(5,))
    for s in closes.columns:
        assert feats[s].iloc[5:].notna().all().all()
        pd.testing.assert_frame_equal(feats[s], filled[s])


def test_symbol_before_its_first_price_is_left_out():
    closes = _closes(n=40)
    closes.iloc[:15, closes.columns.get_loc("D")] = np.nan  # D lists on day 15
    feats = cross_sectional_momentum(closes, windows=(5,))
    mom = np.log(closes).diff(5)
    dt = closes.index[10]
    assert feats["D"].loc[dt].isna().all()
    ranks = sorted(feats[s].loc[dt, "xs_rank_5"] for s in "ABC")
    assert np.allclose(ranks, [1 / 3, 2 / 3, 1.0])
    expected = mom.loc[dt, "A"] - mom.loc[dt, ["B", "C"]].mean()
    assert np.isclose(feats["A"].loc[dt, "sec_rel_5"], expected)


def test_sector_without_live_peers_falls_back_to_universe():
    closes = _closes(n=40)
    closes.iloc[:15, closes.columns.get_loc("B")] = np.nan  # A's only sector peer
    sectors = {"A": "tech", "B": "tech", "C": "energy", "D": "energy"}
    feats = cross_sectional_momentum(closes, windows=(5,), sectors=sectors)
    mom = np.log(closes).diff(5)
    early, late = closes.index[10], closes.index[-1]
    assert np.isclose(feats["A"].loc[early, "sec_rel_5"],
                      mom.loc[early, "A"] - mom.loc[early, ["C", "D"]].mean())
    assert np.isclose(feats["A"].loc[late, "sec_rel_5"],
                      mom.loc[late, "A"] - mom.loc[late, "B"])


def test_sector_keys_match_symbols_case_insensitively():
    closes = _closes()
    upper = cross_sectional_momentum(closes, windows=(5,), sectors={"A": "t", "B": "t"})
    mixed = cross_sectional_momentum(closes, windows=(5,), sectors={"a": "t", "b": "t"})
    pd.testing.assert_frame_equal(upper["A"], mixed["A"])


def test_features_reach_the_model_lagged_one_day():
    prices = load_prices("AAA", start="2020-01-01", end="2021-12-31")
    other = load_prices("BBB", start="2020-01-01", end="2021-12-31")
    closes = pd.DataFrame({"AAA": prices["close"], "BBB": other["close"]})
    feats = cross_sectional_momentum(closes, windows=(5,))
    X, _ = build_features(prices, extra=feats["AAA"])
    assert {"ext_xs_rank_5", "ext_xs_z_5", "ext_sec_rel_5"} <= set(X.columns)
    dt = X.index[10]
    prev = prices.index[prices.index.get_loc(dt) - 1]
    assert np.isclose(X.loc[dt, "ext_sec_rel_5"], feats["AAA"].loc[prev, "sec_rel_5"])


def test_requires_at_least_two_symbols():
    with pytest.raises(ValueError, match="at least two"):
        cross_sectional_momentum(_closes(symbols=("A",)), windows=(5,))


def test_portfolio_backtest_accepts_cross_sectional_features():
    common = dict(scheme="mean_variance", model="gbm", cov_window=40, rebalance=21)
    base = portfolio_backtest(["AAA", "BBB", "CCC"], **common)
    xs = portfolio_backtest(
        ["AAA", "BBB", "CCC"],
        cross_sectional=True,
        sectors={"AAA": "x", "BBB": "x", "CCC": "y"},
        **common,
    )
    assert np.isfinite(xs["sharpe"])
    assert len(xs["returns"]) > 0
    # Extra features must change the model's forecasts, hence the returns.
    assert not np.allclose(
        base["returns"].reindex(xs["returns"].index).fillna(0).to_numpy(),
        xs["returns"].to_numpy(),
    )
