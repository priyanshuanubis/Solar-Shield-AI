"""Cleaning and feature engineering for energetic electron flux forecasting."""

from __future__ import annotations

import pandas as pd

from solarshield_ai.data import REQUIRED_COLUMNS

HORIZON_STEPS = {
    "30-45 min": 9,
    "6 hours": 72,
    "12 hours": 144,
}


def clean_space_weather_data(frame: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Normalize cadence, repair missing values, and keep required signal columns."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Input dataframe must use a DatetimeIndex.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned = frame.sort_index()[REQUIRED_COLUMNS]
    cleaned = cleaned.resample(freq).median()
    cleaned = cleaned.interpolate(method="time", limit_direction="both")
    cleaned = cleaned.ffill().bfill()
    return cleaned


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, and coupling features for forecasting models."""

    featured = frame.copy()
    for column in REQUIRED_COLUMNS:
        for lag in (1, 3, 6, 12):
            featured[f"{column}_lag_{lag}"] = featured[column].shift(lag)
        for window in (3, 12, 36):
            featured[f"{column}_roll_mean_{window}"] = featured[column].rolling(window).mean()
            featured[f"{column}_roll_std_{window}"] = featured[column].rolling(window).std()

    featured["southward_imf"] = featured["imf_bz"].clip(upper=0).abs()
    featured["solar_wind_coupling"] = featured["southward_imf"] * featured["solar_wind_speed"]
    featured["density_pressure_proxy"] = featured["proton_density"] * featured["solar_wind_speed"] ** 2
    return featured.dropna()


def build_supervised_dataset(featured: pd.DataFrame, horizon_steps: int) -> tuple[pd.DataFrame, pd.Series]:
    """Create X/y data where y is the future energetic electron flux."""

    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive.")
    dataset = featured.copy()
    dataset["target_electron_flux"] = dataset["electron_flux"].shift(-horizon_steps)
    dataset = dataset.dropna()
    x = dataset.drop(columns=["target_electron_flux"])
    y = dataset["target_electron_flux"]
    return x, y
