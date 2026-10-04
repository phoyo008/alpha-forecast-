"""Common forecaster interface.

Every model predicts a *forward return* for the next ``horizon`` step given
the feature matrix. Keeping a single interface lets the backtester treat all
models uniformly and makes head-to-head comparison fair.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted forward returns, one per row of ``X``."""
        ...

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"
