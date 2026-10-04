"""Machine-learning forecaster.

Prefers LightGBM, then falls back to scikit-learn's HistGradientBoosting,
then to a plain Ridge-style linear model if neither tree library is present.
This keeps the project installable in minimal environments while using a
strong model when the full stack is available.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from alpha_forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


class GradientBoostingForecaster(Forecaster):
    name = "gbm"

    def __init__(self, **params) -> None:
        self.params = params
        self._model: Any = None
        self._mean = 0.0
        self._backend = "none"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GradientBoostingForecaster:
        self._mean = float(y.mean())

        # 1. LightGBM
        try:
            import lightgbm as lgb  # type: ignore

            defaults: dict[str, Any] = dict(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                subsample_freq=1,
                n_jobs=1,  # threading overhead dominates on ~1k-row folds
                verbosity=-1,
            )
            defaults.update(self.params)
            self._model = lgb.LGBMRegressor(**defaults)
            self._model.fit(X.values, y.values)
            self._backend = "lightgbm"
            return self
        except Exception as exc:
            logger.debug("LightGBM unavailable: %s", exc)

        # 2. scikit-learn HistGradientBoosting
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore

            self._model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.03, max_depth=3
            )
            self._model.fit(X.values, y.values)
            self._backend = "sklearn_hgb"
            return self
        except Exception as exc:
            logger.debug("sklearn HGB unavailable: %s", exc)

        # 3. Linear fallback (normal equations with ridge penalty)
        Xv = np.column_stack([np.ones(len(X)), X.values])
        lam = 1e-3
        A = Xv.T @ Xv + lam * np.eye(Xv.shape[1])
        b = Xv.T @ y.values
        self._coef = np.linalg.solve(A, b)
        self._backend = "linear"
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._backend in ("lightgbm", "sklearn_hgb") and self._model is not None:
            return self._model.predict(X.values)
        if self._backend == "linear":
            Xv = np.column_stack([np.ones(len(X)), X.values])
            return Xv @ self._coef
        return np.full(len(X), self._mean)

    @property
    def backend(self) -> str:
        return self._backend
