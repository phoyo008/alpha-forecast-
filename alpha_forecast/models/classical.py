"""Classical time-series forecaster (Exponential Smoothing / ETS).

Uses statsmodels if available. ETS models the return series directly rather
than using the engineered feature matrix, so it ignores ``X`` and works off
the target history. Falls back to an EWMA mean if statsmodels is missing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha_forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


class ETSForecaster(Forecaster):
    name = "ets"

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self._last_pred = 0.0
        self._y: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ETSForecaster":
        self._y = y.copy()
        try:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing  # type: ignore

            model = SimpleExpSmoothing(y.values, initialization_method="estimated")
            fit = model.fit(smoothing_level=self.alpha, optimized=False)
            self._last_pred = float(fit.forecast(1)[0])
        except Exception as exc:
            logger.debug("statsmodels unavailable, using EWMA fallback: %s", exc)
            self._last_pred = float(y.ewm(alpha=self.alpha).mean().iloc[-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._last_pred)
