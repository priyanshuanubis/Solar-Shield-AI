"""Model training, forecasting, and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from importlib.util import find_spec

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from solarshield_ai.preprocessing import HORIZON_STEPS, build_supervised_dataset


@dataclass
class ForecastResult:
    """Forecast output and metrics for one horizon."""

    horizon: str
    model_name: str
    prediction: float
    mae: float
    rmse: float
    r2: float
    actual: pd.Series
    predicted: pd.Series


def train_forecast_models(
    featured: pd.DataFrame,
    model_name: str = "Gradient Boosting",
    target_column: str = "electron_flux",
) -> list[ForecastResult]:
    """Train one model for each configured forecast horizon."""

    results: list[ForecastResult] = []
    for horizon, steps in HORIZON_STEPS.items():
        x, y = build_supervised_dataset(featured, steps, target_column=target_column)
        split_index = max(int(len(x) * 0.8), 1)
        if split_index >= len(x):
            raise ValueError(f"Not enough rows to train and validate the {horizon} model.")

        x_train, x_test = x.iloc[:split_index], x.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        model = _make_model(model_name)
        model.fit(x_train, y_train)
        predictions = pd.Series(model.predict(x_test), index=y_test.index, name="predicted_flux")
        latest_prediction = float(model.predict(featured.tail(1))[0])
        results.append(
            ForecastResult(
                horizon=horizon,
                model_name=model_name,
                prediction=latest_prediction,
                mae=float(mean_absolute_error(y_test, predictions)),
                rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
                r2=float(r2_score(y_test, predictions)),
                actual=y_test.rename(f"actual_{target_column}"),
                predicted=predictions,
            )
        )
    return results


def _make_model(model_name: str):
    if model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=3, n_jobs=-1)
    if model_name == "XGBoost":
        if find_spec("xgboost") is None:
            raise ImportError("xgboost is required for the XGBoost model. Install dependencies from requirements.txt.")
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, random_state=42)),
        ]
    )
