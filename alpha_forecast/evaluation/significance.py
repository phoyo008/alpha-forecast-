"""Statistical significance of forecast and strategy performance.

A leaderboard ranks models, but with a few hundred noisy daily returns most
differences are luck. Two tools separate signal from noise:

* :func:`diebold_mariano` -- is model A's forecast loss *significantly* lower
  than model B's (typically the naive baseline)?
* :func:`bootstrap_sharpe_ci` -- how wide is the confidence interval around a
  strategy's Sharpe ratio, respecting autocorrelation via a block bootstrap?

Pure NumPy plus the standard library; no scipy dependency.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from alpha_forecast.evaluation.metrics import TRADING_DAYS

_NORMAL = NormalDist()


def _newey_west_lrv(d: np.ndarray, lags: int) -> float:
    """Long-run variance of ``d`` with Bartlett (Newey-West) weights."""
    d = d - d.mean()
    n = len(d)
    lrv = float(d @ d) / n
    for k in range(1, min(lags, n - 1) + 1):
        gamma = float(d[k:] @ d[:-k]) / n
        lrv += 2.0 * (1.0 - k / (lags + 1.0)) * gamma
    return lrv


def diebold_mariano(
    y_true: pd.Series,
    pred_a: pd.Series,
    pred_b: pd.Series,
    *,
    horizon: int = 1,
    loss: str = "mse",
) -> dict:
    """Diebold-Mariano test of equal predictive accuracy.

    The loss differential is ``d_t = L(a_t) - L(b_t)``; negative values favour
    model A. The long-run variance uses Newey-West weights with ``horizon - 1``
    lags (overlapping multi-step forecasts are autocorrelated up to that lag),
    and the statistic includes the Harvey-Leybourne-Newbold small-sample
    correction. P-values use the normal approximation.

    Returns ``stat``, ``p_value`` (one-sided, H1: A more accurate than B),
    ``p_value_two_sided`` and ``mean_loss_diff``. Identical forecasts give NaN.
    """
    idx = y_true.index.intersection(pred_a.index).intersection(pred_b.index)
    y = y_true.loc[idx].to_numpy(dtype=float)
    a = pred_a.loc[idx].to_numpy(dtype=float)
    b = pred_b.loc[idx].to_numpy(dtype=float)

    if loss == "mse":
        d = (a - y) ** 2 - (b - y) ** 2
    elif loss == "mae":
        d = np.abs(a - y) - np.abs(b - y)
    else:
        raise ValueError(f"Unknown loss '{loss}'; use 'mse' or 'mae'")

    n = len(d)
    nan = float("nan")
    result = {"stat": nan, "p_value": nan, "p_value_two_sided": nan, "mean_loss_diff": nan}
    if n < 2:
        return result
    result["mean_loss_diff"] = float(d.mean())

    lrv = _newey_west_lrv(d, horizon - 1)
    if lrv <= 0 or not np.isfinite(lrv):
        return result

    h = horizon
    hln = np.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1.0) / n)
    stat = float(d.mean() / np.sqrt(lrv / n) * hln)
    result["stat"] = stat
    result["p_value"] = _NORMAL.cdf(stat)
    result["p_value_two_sided"] = 2.0 * (1.0 - _NORMAL.cdf(abs(stat)))
    return result


def bootstrap_sharpe_ci(
    returns: pd.Series,
    *,
    periods_per_year: float = TRADING_DAYS,
    n_boot: int = 1000,
    block: int | None = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Circular block-bootstrap confidence interval for the annualised Sharpe.

    ``block`` defaults to ``n ** (1/3)``, a standard rule of thumb that keeps
    short-range autocorrelation inside each resampled block.

    Returns ``sharpe`` (point estimate), ``lo``/``hi`` (the ``1 - alpha``
    interval) and ``p_le_zero`` (bootstrap share of Sharpe <= 0).
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    nan = float("nan")
    if n < 10 or r.std() == 0:
        return {"sharpe": nan, "lo": nan, "hi": nan, "p_le_zero": nan}

    block = block or max(1, int(round(n ** (1 / 3))))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    samples = r[idx.reshape(n_boot, -1)[:, :n]]

    std = samples.std(axis=1, ddof=1)
    std[std == 0] = np.nan
    boot = np.sqrt(periods_per_year) * samples.mean(axis=1) / std
    boot = boot[np.isfinite(boot)]

    point = float(np.sqrt(periods_per_year) * r.mean() / r.std(ddof=1))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "sharpe": point,
        "lo": float(lo),
        "hi": float(hi),
        "p_le_zero": float((boot <= 0).mean()),
    }
