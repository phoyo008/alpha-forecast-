from alpha_forecast.models.base import Forecaster
from alpha_forecast.models.naive import NaiveForecaster, DriftForecaster
from alpha_forecast.models.classical import ETSForecaster
from alpha_forecast.models.ml import GradientBoostingForecaster

MODEL_REGISTRY = {
    "naive": NaiveForecaster,
    "drift": DriftForecaster,
    "ets": ETSForecaster,
    "gbm": GradientBoostingForecaster,
}

__all__ = [
    "Forecaster",
    "NaiveForecaster",
    "DriftForecaster",
    "ETSForecaster",
    "GradientBoostingForecaster",
    "MODEL_REGISTRY",
]
