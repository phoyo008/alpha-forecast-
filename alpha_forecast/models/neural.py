"""Neural-network forecaster.

Provides a feed-forward neural net for return forecasting with a graceful
capability ladder:

    1. PyTorch  -> a small LSTM over a lookback window of features
    2. scikit-learn MLPRegressor (feed-forward) fallback
    3. Ridge linear fallback (same as the GBM linear path) if neither is present

Standardising inputs matters a lot for neural nets, so features are z-scored
using statistics learned on the training fold only (no leakage).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from alpha_forecast.models.base import Forecaster

logger = logging.getLogger(__name__)


class _Scaler:
    """Simple train-fit standardiser (mean/std), leakage-free."""

    def __init__(self) -> None:
        self.mean_: np.ndarray = np.zeros(0)
        self.std_: np.ndarray = np.ones(0)

    def fit(self, X: np.ndarray) -> _Scaler:
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


class NeuralForecaster(Forecaster):
    name = "neural"

    def __init__(self, lookback: int = 10, epochs: int = 60, hidden: int = 32) -> None:
        self.lookback = lookback
        self.epochs = epochs
        self.hidden = hidden
        self._scaler = _Scaler()
        self._backend = "none"
        self._model: Any = None
        self._mean = 0.0
        self._y_std = 1.0
        self._coef: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> NeuralForecaster:
        self._mean = float(y.mean())
        Xv = X.to_numpy(dtype=float)
        # Daily returns are ~1e-2; nets train far better on a unit-scale target.
        self._y_std = float(y.std()) or 1.0
        yv = (y.to_numpy(dtype=float) - self._mean) / self._y_std
        self._scaler.fit(Xv)
        Xs = self._scaler.transform(Xv)

        if self._try_torch(Xs, yv):
            return self
        if self._try_sklearn(Xs, yv):
            return self
        return self._fit_linear(Xs, yv)

    def _try_torch(self, Xs: np.ndarray, yv: np.ndarray) -> bool:
        try:
            import torch  # type: ignore
            from torch import nn
        except Exception as exc:
            logger.debug("PyTorch unavailable: %s", exc)
            return False
        try:
            seqs, targets = self._make_sequences(Xs, yv, self.lookback)
            if len(seqs) < 20:  # not enough data for a sequence model
                return False
            n_features = Xs.shape[1]

            class LSTMNet(nn.Module):
                def __init__(self, n_in: int, hidden: int) -> None:
                    super().__init__()
                    self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
                    self.head = nn.Linear(hidden, 1)

                def forward(self, x):  # noqa: D401
                    out, _ = self.lstm(x)
                    return self.head(out[:, -1, :]).squeeze(-1)

            device = "cpu"
            net = LSTMNet(n_features, self.hidden).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            xt = torch.tensor(seqs, dtype=torch.float32)
            yt = torch.tensor(targets, dtype=torch.float32)
            net.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                pred = net(xt)
                loss = loss_fn(pred, yt)
                loss.backward()
                opt.step()
            net.eval()
            self._model = net
            self._torch = torch
            self._backend = "torch_lstm"
            logger.info("Trained LSTM (%d sequences)", len(seqs))
            return True
        except Exception as exc:  # pragma: no cover - torch runtime issues
            logger.warning("Torch LSTM training failed, falling back: %s", exc)
            return False

    def _try_sklearn(self, Xs: np.ndarray, yv: np.ndarray) -> bool:
        try:
            from sklearn.neural_network import MLPRegressor  # type: ignore
        except Exception as exc:
            logger.debug("sklearn MLP unavailable: %s", exc)
            return False
        self._model = MLPRegressor(
            hidden_layer_sizes=(self.hidden, self.hidden // 2),
            activation="relu",
            max_iter=400,
            early_stopping=True,
            random_state=0,
        )
        self._model.fit(Xs, yv)
        self._backend = "sklearn_mlp"
        return True

    def _fit_linear(self, Xs: np.ndarray, yv: np.ndarray) -> NeuralForecaster:
        Xb = np.column_stack([np.ones(len(Xs)), Xs])
        lam = 1e-3
        A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
        self._coef = np.linalg.solve(A, Xb.T @ yv)
        self._backend = "linear"
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _make_sequences(Xs: np.ndarray, yv: np.ndarray, lookback: int):
        seqs, targets = [], []
        for i in range(lookback, len(Xs)):
            seqs.append(Xs[i - lookback : i])
            targets.append(yv[i])
        return np.asarray(seqs, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._mean + self._y_std * self._predict_scaled(X)

    def _predict_scaled(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self._scaler.transform(X.to_numpy(dtype=float))

        if self._backend == "torch_lstm" and self._model is not None:
            torch = self._torch
            # Build rolling sequences; the first `lookback` rows can't form a
            # full window, so pad them with the mean (zero on the scaled target).
            n = len(Xs)
            preds = np.zeros(n, dtype=float)
            if n > self.lookback:
                seqs = np.stack(
                    [Xs[i - self.lookback : i] for i in range(self.lookback, n)]
                ).astype(np.float32)
                with torch.no_grad():
                    out = self._model(torch.tensor(seqs)).numpy()
                preds[self.lookback :] = out
            return preds

        if self._backend == "sklearn_mlp" and self._model is not None:
            return self._model.predict(Xs)

        if self._backend == "linear" and self._coef is not None:
            Xb = np.column_stack([np.ones(len(Xs)), Xs])
            return Xb @ self._coef

        return np.zeros(len(X))

    @property
    def backend(self) -> str:
        return self._backend
