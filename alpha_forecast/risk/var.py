"""Value-at-Risk and Expected Shortfall.

Turns the GARCH/EWMA volatility forecasts into one-day risk numbers and
backtests them. VaR and ES are reported as *positive losses* in return units
(``0.02`` == a 2% loss).

* :func:`parametric_var` / :func:`parametric_es` -- normal model on a vol forecast
* :func:`historical_var` / :func:`historical_es` -- empirical quantile of past returns
* :func:`rolling_var_forecast` -- walk-forward, out-of-sample VaR/ES series
* :func:`kupiec_test` -- does the realised breach rate match ``alpha``?
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from alpha_forecast.models.volatility import rolling_vol_forecast

_NORMAL = NormalDist()


def parametric_var(vol, alpha: float = 0.01, mean: float = 0.0):
    """Normal VaR at tail probability ``alpha`` for a volatility forecast."""
    z = _NORMAL.inv_cdf(alpha)
    return -(mean + z * vol)


def parametric_es(vol, alpha: float = 0.01, mean: float = 0.0):
    """Normal Expected Shortfall (mean loss beyond the VaR)."""
    z = _NORMAL.inv_cdf(alpha)
    return -mean + vol * _NORMAL.pdf(z) / alpha


def historical_var(returns: pd.Series, alpha: float = 0.01) -> float:
    return float(-np.quantile(returns.dropna(), alpha))


def historical_es(returns: pd.Series, alpha: float = 0.01) -> float:
    r = returns.dropna().to_numpy()
    q = np.quantile(r, alpha)
    return float(-r[r <= q].mean())


def rolling_var_forecast(
    returns: pd.Series,
    *,
    alpha: float = 0.01,
    method: str = "garch",
    initial_train: int = 252,
    step: int = 21,
    window: int = 250,
) -> pd.DataFrame:
    """Out-of-sample one-day VaR and ES, aligned to the return index.

    ``method="garch"`` scales the walk-forward GARCH (EWMA fallback) volatility
    forecast by the normal quantile. ``method="historical"`` uses the empirical
    quantile of the trailing ``window`` returns strictly before each date.
    Both start after ``initial_train`` observations.
    """
    r = returns.dropna().astype(float)
    if method == "garch":
        vol = rolling_vol_forecast(r, initial_train=initial_train, step=step)
        out = pd.DataFrame(
            {"var": parametric_var(vol, alpha), "es": parametric_es(vol, alpha)}
        )
    elif method == "historical":
        past = r.shift(1)
        q = past.rolling(window, min_periods=min(window, initial_train)).quantile(alpha)
        es = past.rolling(window, min_periods=min(window, initial_train)).apply(
            lambda x: x[x <= np.quantile(x, alpha)].mean(), raw=True
        )
        out = pd.DataFrame({"var": -q, "es": -es}).iloc[initial_train:]
    else:
        raise ValueError(f"Unknown method '{method}'; use 'garch' or 'historical'")
    out.index.name = r.index.name
    return out.dropna()


def kupiec_test(returns: pd.Series, var: pd.Series, alpha: float = 0.01) -> dict:
    """Kupiec proportion-of-failures test for a VaR forecast.

    H0: the probability of a loss exceeding VaR equals ``alpha``. A small
    ``p_value`` means the model under- or over-states risk.
    """
    idx = returns.index.intersection(var.index)
    r = returns.loc[idx].to_numpy(dtype=float)
    v = var.loc[idx].to_numpy(dtype=float)
    n = len(r)
    breaches = int((r < -v).sum())
    rate = breaches / n if n else float("nan")

    def _loglik(p: float) -> float:
        ll = 0.0
        if n - breaches > 0:
            ll += (n - breaches) * math.log(1.0 - p)
        if breaches > 0:
            ll += breaches * math.log(p)
        return ll

    if n == 0:
        lr = float("nan")
    elif breaches in (0, n):
        lr = -2.0 * _loglik(alpha)
    else:
        lr = -2.0 * (_loglik(alpha) - _loglik(rate))
    p_value = math.erfc(math.sqrt(lr / 2.0)) if np.isfinite(lr) else float("nan")
    return {
        "n": n,
        "breaches": breaches,
        "breach_rate": rate,
        "expected_rate": alpha,
        "lr_stat": lr,
        "p_value": p_value,
    }
