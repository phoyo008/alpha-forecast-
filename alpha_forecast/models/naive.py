"""Baseline forecasters.

A credible forecasting project MUST include naive baselines. If a fancy model
can't beat "tomorrow looks like today", it isn't adding value. Recruiters in
quant look for exactly this kind of honesty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_forecast.models.base import Forecaster


class NaiveForecaster(Forecaster):
    """Random-walk baseline: predicted forward return is always 0.

    (Equivalent to "price tomorrow == price today".)
    """

    name = "naive"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveForecaster":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class DriftForecaster(Forecaster):
    """Drift baseline: predict the mean forward return seen in training."""

    name = "drift"

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DriftForecaster":
        self._mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)
