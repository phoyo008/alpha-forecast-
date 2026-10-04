"""Tests for the options pricing, Greeks, and implied-volatility module.

These are deterministic (no data download) and check against analytic
identities and well-known reference values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alpha_forecast.options import (
    bs_price,
    call_price,
    delta,
    gamma,
    greeks,
    implied_volatility,
    put_price,
    rho,
    theta,
    vega,
)

# A standard reference contract.
S, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


def test_atm_call_reference_value():
    # Known Black-Scholes value for S=K=100, r=5%, sigma=20%, T=1y ~ 10.4506
    c = call_price(S, K, T, R, SIGMA)
    assert c == pytest.approx(10.4506, abs=1e-3)


def test_atm_put_reference_value():
    p = put_price(S, K, T, R, SIGMA)
    # From put-call parity: p = c - S + K e^{-rT}
    assert p == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    c = call_price(S, K, T, R, SIGMA, Q)
    p = put_price(S, K, T, R, SIGMA, Q)
    lhs = c - p
    rhs = S * math.exp(-Q * T) - K * math.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_price_at_expiry_is_intrinsic():
    assert call_price(120, 100, 0.0, R, SIGMA) == pytest.approx(20.0)
    assert put_price(80, 100, 0.0, R, SIGMA) == pytest.approx(20.0)
    assert call_price(80, 100, 0.0, R, SIGMA) == pytest.approx(0.0)


def test_prices_accept_arrays():
    spots = np.array([80.0, 100.0, 120.0])
    out = call_price(spots, K, T, R, SIGMA)
    assert out.shape == (3,)
    assert np.all(np.diff(out) > 0)  # call price increases with spot


def test_delta_bounds_and_signs():
    cd = delta(S, K, T, R, SIGMA, Q, "call")
    pd_ = delta(S, K, T, R, SIGMA, Q, "put")
    assert 0.0 < cd < 1.0
    assert -1.0 < pd_ < 0.0
    # delta_call - delta_put == e^{-qT} (== 1 when q=0)
    assert cd - pd_ == pytest.approx(math.exp(-Q * T), abs=1e-9)


def test_gamma_vega_positive_and_shared():
    assert gamma(S, K, T, R, SIGMA) > 0
    assert vega(S, K, T, R, SIGMA) > 0
    # gamma is identical for calls and puts (no option_type arg)
    g = gamma(S, K, T, R, SIGMA)
    assert g == pytest.approx(gamma(S, K, T, R, SIGMA))


def test_theta_negative_for_long_call():
    assert theta(S, K, T, R, SIGMA, Q, "call") < 0


def test_rho_sign():
    assert rho(S, K, T, R, SIGMA, Q, "call") > 0
    assert rho(S, K, T, R, SIGMA, Q, "put") < 0


def test_greeks_dict_complete():
    g = greeks(S, K, T, R, SIGMA, Q, "call")
    assert set(g) == {"delta", "gamma", "vega", "theta", "rho"}


def test_vega_matches_finite_difference():
    h = 1e-4
    analytic = vega(S, K, T, R, SIGMA)
    fd = (bs_price(S, K, T, R, SIGMA + h, Q, "call")
          - bs_price(S, K, T, R, SIGMA - h, Q, "call")) / (2 * h)
    assert analytic == pytest.approx(fd, rel=1e-4)


def test_delta_matches_finite_difference():
    h = 1e-4
    analytic = delta(S, K, T, R, SIGMA, Q, "call")
    fd = (bs_price(S + h, K, T, R, SIGMA, Q, "call")
          - bs_price(S - h, K, T, R, SIGMA, Q, "call")) / (2 * h)
    assert analytic == pytest.approx(fd, rel=1e-4)


@pytest.mark.parametrize("true_sigma", [0.1, 0.2, 0.35, 0.8])
@pytest.mark.parametrize("opt", ["call", "put"])
def test_implied_vol_roundtrip(true_sigma, opt):
    price = bs_price(S, K, T, R, true_sigma, Q, opt)
    iv = implied_volatility(price, S, K, T, R, Q, opt)
    assert iv == pytest.approx(true_sigma, abs=1e-4)


def test_implied_vol_otm_roundtrip():
    # Out-of-the-money call where Newton's vega is flatter.
    price = bs_price(100, 140, 0.5, R, 0.3, Q, "call")
    iv = implied_volatility(price, 100, 140, 0.5, R, Q, "call")
    assert iv == pytest.approx(0.3, abs=1e-4)


def test_implied_vol_unattainable_price_is_nan():
    # Price above the no-arbitrage upper bound.
    assert math.isnan(implied_volatility(1e6, S, K, T, R, Q, "call"))
    assert math.isnan(implied_volatility(-1.0, S, K, T, R, Q, "call"))
