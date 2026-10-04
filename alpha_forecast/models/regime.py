"""Regime-switching forecaster.

Return dynamics differ between calm and turbulent markets (momentum in one,
mean reversion in the other), so a single global model averages them away.
This forecaster:

    1. fits a Gaussian HMM on (daily return, realised vol) in the training fold,
    2. fits one ridge regression per regime, weighting each training row by
       its smoothed regime probability,
    3. at prediction time blends the per-regime forecasts using *filtered*
       (causal) regime probabilities, continuing the filter from the end of
       the training window.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alpha_forecast.models.base import Forecaster
from alpha_forecast.regime.hmm import GaussianHMM

logger = logging.getLogger(__name__)

REGIME_COLUMNS = ("ret_1", "vol_5")


class RegimeSwitchingForecaster(Forecaster):
    name = "regime"

    def __init__(self, n_states: int = 2, ridge: float = 10.0, n_iter: int = 30) -> None:
        self.n_states = n_states
        self.ridge = ridge
        self.n_iter = n_iter
        self._mean = 0.0
        self._hmm: GaussianHMM | None = None
        self._coefs: list[np.ndarray] = []
        self._last_prob: np.ndarray | None = None

    def _regime_obs(self, X: pd.DataFrame) -> np.ndarray:
        cols = [c for c in REGIME_COLUMNS if c in X.columns] or [X.columns[0]]
        return (X[cols].to_numpy(dtype=float) - self._z_mean) / self._z_std

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        Xs = (X.to_numpy(dtype=float) - self._x_mean) / self._x_std
        return np.column_stack([np.ones(len(Xs)), Xs])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RegimeSwitchingForecaster:
        self._mean = float(y.mean())
        Xv = X.to_numpy(dtype=float)
        self._x_mean = Xv.mean(axis=0)
        self._x_std = np.where(Xv.std(axis=0) > 0, Xv.std(axis=0), 1.0)
        cols = [c for c in REGIME_COLUMNS if c in X.columns] or [X.columns[0]]
        Zraw = X[cols].to_numpy(dtype=float)
        self._z_mean = Zraw.mean(axis=0)
        self._z_std = np.where(Zraw.std(axis=0) > 0, Zraw.std(axis=0), 1.0)
        Z = self._regime_obs(X)

        try:
            self._hmm = GaussianHMM(self.n_states, n_iter=self.n_iter).fit(Z)
            gamma = self._hmm.smooth(Z)
            self._last_prob = self._hmm.filter(Z)[-1]
        except Exception as exc:
            logger.warning("HMM fit failed, using a single regime: %s", exc)
            self._hmm = None
            gamma = np.ones((len(X), 1))

        D = self._design(X)
        yv = y.to_numpy(dtype=float)
        penalty = self.ridge * np.eye(D.shape[1])
        penalty[0, 0] = 0.0
        self._coefs = []
        for k in range(gamma.shape[1]):
            w = gamma[:, k]
            Dw = D * w[:, None]
            self._coefs.append(np.linalg.solve(D.T @ Dw + penalty, Dw.T @ yv))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._coefs:
            return np.full(len(X), self._mean)
        D = self._design(X)
        per_regime = np.column_stack([D @ c for c in self._coefs])
        if self._hmm is None:
            return per_regime[:, 0]
        probs = self._hmm.filter(self._regime_obs(X), init=self._last_prob)
        return (per_regime * probs).sum(axis=1)
