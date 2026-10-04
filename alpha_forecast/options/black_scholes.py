"""Black-Scholes-Merton European option pricing.

All functions accept scalars or NumPy arrays and support a continuous dividend
yield ``q`` (set ``q=0`` for non-dividend-paying underlyings).

Conventions
-----------
S : spot price of the underlying
K : strike price
t : time to expiry in years
r : continuously-compounded risk-free rate (annual, decimal)
sigma : annualised volatility (decimal)
q : continuous dividend yield (annual, decimal)
option_type : "call" or "put"
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, t, r, sigma, q=0.0):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Guard against zero time / zero vol to avoid division warnings; callers
    # that hit expiry should use the intrinsic value (handled in bs_price).
    vol_sqrt_t = sigma * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
    return d1, d2


def bs_price(S, K, t, r, sigma, q=0.0, option_type: str = "call"):
    """Black-Scholes-Merton price for a European call or put.

    At expiry (``t == 0``) or zero volatility the intrinsic value is returned.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    d1, d2 = _d1_d2(S, K, t, r, sigma, q)
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)

    if option_type == "call":
        price = disc_q * S * norm.cdf(d1) - disc_r * K * norm.cdf(d2)
        intrinsic = np.maximum(S - K, 0.0)
    else:
        price = disc_r * K * norm.cdf(-d2) - disc_q * S * norm.cdf(-d1)
        intrinsic = np.maximum(K - S, 0.0)

    # Where time or vol is non-positive, fall back to intrinsic value.
    expired = (t <= 0) | (sigma <= 0)
    price = np.where(expired, intrinsic, price)
    return price.item() if price.ndim == 0 else price


def call_price(S, K, t, r, sigma, q=0.0):
    return bs_price(S, K, t, r, sigma, q, "call")


def put_price(S, K, t, r, sigma, q=0.0):
    return bs_price(S, K, t, r, sigma, q, "put")
