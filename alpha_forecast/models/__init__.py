from alpha_forecast.models.base import Forecaster
from alpha_forecast.models.classical import ETSForecaster
from alpha_forecast.models.ml import GradientBoostingForecaster
from alpha_forecast.models.naive import DriftForecaster, NaiveForecaster
from alpha_forecast.models.neural import NeuralForecaster
from alpha_forecast.models.regime import RegimeSwitchingForecaster

MODEL_REGISTRY = {
    "naive": NaiveForecaster,
    "drift": DriftForecaster,
    "ets": ETSForecaster,
    "gbm": GradientBoostingForecaster,
    "neural": NeuralForecaster,
    "regime": RegimeSwitchingForecaster,
}

__all__ = [
    "MODEL_REGISTRY",
    "DriftForecaster",
    "ETSForecaster",
    "Forecaster",
    "GradientBoostingForecaster",
    "NaiveForecaster",
    "NeuralForecaster",
    "RegimeSwitchingForecaster",
]
