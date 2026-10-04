"""Volatility forecasting (GARCH family).

Volatility is far more forecastable than returns, which is why risk and
options desks model it directly. This module forecasts next-step conditional
volatility of the return series.

Backend order:
    1. ``arch`` package  -> GARCH(1,1)
    2. EWMA (RiskMetrics-style) fallback -- always available.

Unlike the return Forecasters, this predicts volatility (a positive number),
so it lives apart from the return-model registry and is exposed through its
own helper. It can still be evaluated with the standard error metrics against
realised volatility.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GARCHVolForecaster:
    name = "garch"

    def __init__(self, lam: float = 0.94) -> None:
        self.lam = lam  # EWMA decay for the fallback
        self._backend = "none"
        self._forecast = np.nan
        self._fitted = None

    def fit(self, returns: pd.Series) -> "GARCHVolForecaster":
        r = returns.dropna().astype(float)
        try:
            from arch import arch_model  # type: ignore

            # Scale to percent for numerical stability, as arch recommends.
            am = arch_model(r * 100, vol="GARCH", p=1, q=1, mean="constant", dist="normal")
            res = am.fit(disp="off")
            fc = res.forecast(horizon=1, reindex=False)
            self._forecast = float(np.sqrt(fc.variance.values[-1, 0]) / 100.0)
            self._fitted = res
            self._backend = "arch"
            return self
        except Exception as exc:
            logger.debug("arch unavailable, using EWMA: %s", exc)

        # RiskMetrics EWMA variance recursion.
        var = r.var()
        for x in r.values:
            var = self.lam * var + (1 - self.lam) * x * x
        self._forecast = float(np.sqrt(var))
        self._backend = "ewma"
        return self

    def predict_next(self) -> float:
        """One-step-ahead conditional volatility (standard deviation)."""
        return self._forecast

    @property
    def backend(self) -> str:
        return self._backend


def rolling_vol_forecast(
    returns: pd.Series,
    *,
    initial_train: int = 252,
    step: int = 21,
    lam: float = 0.94,
) -> pd.Series:
    """Walk-forward one-step volatility forecasts aligned to the return index."""
    r = returns.dropna().astype(float)
    n = len(r)
    out_idx, out_val = [], []
    start = initial_train
    while start < n:
        end = min(start + step, n)
        model = GARCHVolForecaster(lam=lam).fit(r.iloc[:start])
        f = model.predict_next()
        for i in range(start, end):
            out_idx.append(r.index[i])
            out_val.append(f)
        start = end
    return pd.Series(out_val, index=pd.Index(out_idx, name=r.index.name), name="vol_forecast")
