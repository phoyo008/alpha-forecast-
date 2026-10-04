"""Analytic Black-Scholes-Merton Greeks for European options.

Sign and scaling conventions follow common desk usage:
    * delta : ∂V/∂S
    * gamma : ∂²V/∂S²   (same for calls and puts)
    * vega  : ∂V/∂σ, reported per 1.00 (100%) change in vol
    * theta : ∂V/∂t as time decay PER YEAR (negative for long options)
    * rho   : ∂V/∂r per 1.00 (100%) change in rate

Divide ``vega``/``rho`` by 100 for a per-1%-point sensitivity, and ``theta``
by 365 for per-calendar-day decay.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from alpha_forecast.options.black_scholes import _d1_d2


def _validate(option_type: str) -> str:
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    return option_type


def delta(S, K, t, r, sigma, q=0.0, option_type: str = "call"):
    option_type = _validate(option_type)
    d1, _ = _d1_d2(S, K, t, r, sigma, q)
    disc_q = np.exp(-q * np.asarray(t, dtype=float))
    if option_type == "call":
        out = disc_q * norm.cdf(d1)
    else:
        out = disc_q * (norm.cdf(d1) - 1.0)
    return out.item() if np.ndim(out) == 0 else out


def gamma(S, K, t, r, sigma, q=0.0):
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1, _ = _d1_d2(S, K, t, r, sigma, q)
    disc_q = np.exp(-q * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = disc_q * norm.pdf(d1) / (S * sigma * np.sqrt(t))
    return out.item() if np.ndim(out) == 0 else out


def vega(S, K, t, r, sigma, q=0.0):
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    d1, _ = _d1_d2(S, K, t, r, sigma, q)
    disc_q = np.exp(-q * t)
    out = disc_q * S * norm.pdf(d1) * np.sqrt(t)
    return out.item() if np.ndim(out) == 0 else out


def theta(S, K, t, r, sigma, q=0.0, option_type: str = "call"):
    option_type = _validate(option_type)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1, d2 = _d1_d2(S, K, t, r, sigma, q)
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)

    term1 = -disc_q * S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(t))
    if option_type == "call":
        out = (
            term1
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
    else:
        out = (
            term1
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )
    return out.item() if np.ndim(out) == 0 else out


def rho(S, K, t, r, sigma, q=0.0, option_type: str = "call"):
    option_type = _validate(option_type)
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)
    _, d2 = _d1_d2(S, K, t, r, sigma, q)
    disc_r = np.exp(-r * t)
    if option_type == "call":
        out = K * t * disc_r * norm.cdf(d2)
    else:
        out = -K * t * disc_r * norm.cdf(-d2)
    return out.item() if np.ndim(out) == 0 else out


def greeks(S, K, t, r, sigma, q=0.0, option_type: str = "call") -> dict:
    """Return all Greeks in a single dict for convenience."""
    option_type = _validate(option_type)
    return {
        "delta": delta(S, K, t, r, sigma, q, option_type),
        "gamma": gamma(S, K, t, r, sigma, q),
        "vega": vega(S, K, t, r, sigma, q),
        "theta": theta(S, K, t, r, sigma, q, option_type),
        "rho": rho(S, K, t, r, sigma, q, option_type),
    }
