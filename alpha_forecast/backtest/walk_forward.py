"""Walk-forward (expanding-window) backtesting.

This is the heart of credible forecasting evaluation. Instead of a single
train/test split -- which leaks information and overstates performance -- we
repeatedly:

    1. train on all data up to time t,
    2. predict the next ``step`` observations,
    3. roll the window forward,

so every prediction is strictly out-of-sample. This mirrors how a model would
actually be deployed and retrained in production.

For multi-day horizons the target of training row ``t`` is only realised at
``t + horizon``, so the last ``horizon - 1`` training rows would peek into the
test window. Those rows are *purged* from each training fold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    model_name: str
    predictions: pd.Series
    actuals: pd.Series
    extras: dict = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"y_true": self.actuals, "y_pred": self.predictions})


def walk_forward_backtest(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    initial_train: int = 252,
    step: int = 21,
    horizon: int = 1,
) -> BacktestResult:
    """Run an expanding-window walk-forward backtest.

    Parameters
    ----------
    model_factory:
        Zero-arg callable returning a fresh :class:`Forecaster` each fold.
    X, y:
        Aligned feature matrix and forward-return target.
    initial_train:
        Number of observations in the first training window (default ~1yr).
    step:
        How many observations to predict before retraining (default ~1mo).
    horizon:
        Forecast horizon of ``y`` in rows. The final ``horizon - 1`` rows of
        each training window are purged because their targets overlap the
        test window.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    purge = horizon - 1
    n = len(X)
    if n <= initial_train or initial_train - purge < 1:
        raise ValueError(
            f"Not enough data: have {n} rows, need > initial_train={initial_train} "
            f"and initial_train > horizon - 1={purge}"
        )

    preds: list[float] = []
    idx: list = []
    start = initial_train
    model_name = "unknown"

    while start < n:
        end = min(start + step, n)
        X_train, y_train = X.iloc[: start - purge], y.iloc[: start - purge]
        X_test = X.iloc[start:end]

        model: Forecaster = model_factory()
        model_name = model.name
        model.fit(X_train, y_train)
        fold_pred = np.asarray(model.predict(X_test), dtype=float)

        preds.extend(fold_pred.tolist())
        idx.extend(X_test.index.tolist())
        start = end

    pred_series = pd.Series(preds, index=pd.Index(idx, name=X.index.name))
    actual_series = y.loc[pred_series.index]
    logger.info(
        "Walk-forward complete for %s: %d OOS predictions", model_name, len(pred_series)
    )
    return BacktestResult(model_name, pred_series, actual_series)
