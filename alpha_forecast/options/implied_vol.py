"""Implied volatility: invert Black-Scholes for sigma given a market price.

Uses Newton-Raphson (fast quadratic convergence via vega) with a bracketed
bisection fallback for the cases where Newton misbehaves -- flat vega deep
in/out of the money, or a starting guess that overshoots. This hybrid is the
standard robust approach used on trading desks.
"""

from __future__ import annotations

import numpy as np

from alpha_forecast.options.black_scholes import bs_price
from alpha_forecast.options.greeks import vega


def _no_arbitrage_bounds(S, K, t, r, q, option_type):
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)
    if option_type == "call":
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r
    return lower, upper


def implied_volatility(
    price: float,
    S: float,
    K: float,
    t: float,
    r: float,
    q: float = 0.0,
    option_type: str = "call",
    *,
    tol: float = 1e-8,
    max_iter: int = 100,
    lo: float = 1e-6,
    hi: float = 5.0,
) -> float:
    """Return the implied volatility, or ``nan`` if the price is unattainable.

    Parameters
    ----------
    price:
        Observed option price to match.
    S, K, t, r, q, option_type:
        Standard Black-Scholes inputs (see ``black_scholes``).
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if t <= 0 or price <= 0:
        return float("nan")

    lower, upper = _no_arbitrage_bounds(S, K, t, r, q, option_type)
    # Allow a tiny tolerance for rounding at the no-arbitrage boundaries.
    if price < lower - 1e-10 or price > upper + 1e-10:
        return float("nan")

    def objective(sig):
        return bs_price(S, K, t, r, sig, q, option_type) - price

    # --- Newton-Raphson -------------------------------------------------
    sigma = 0.2  # sensible starting guess
    for _ in range(max_iter):
        diff = objective(sigma)
        if abs(diff) < tol:
            return float(sigma)
        v = vega(S, K, t, r, sigma, q)
        if v < 1e-10:  # vega too flat; hand off to bisection
            break
        step = diff / v
        sigma_new = sigma - step
        if sigma_new <= 0 or sigma_new > hi:  # left the sensible region
            break
        sigma = sigma_new

    # --- Bisection fallback (guaranteed to converge in the bracket) -----
    a, b = lo, hi
    fa = objective(a)
    fb = objective(b)
    if fa * fb > 0:
        return float("nan")  # root not bracketed -> price unattainable
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = objective(m)
        if abs(fm) < tol or (b - a) < tol:
            return float(m)
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return float(0.5 * (a + b))
