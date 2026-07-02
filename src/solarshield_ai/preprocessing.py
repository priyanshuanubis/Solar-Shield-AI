"""Cleaning and feature engineering for energetic electron flux forecasting."""

from __future__ import annotations

import pandas as pd

from solarshield_ai.data import SAMPLE_REQUIRED_COLUMNS, numeric_columns

HORIZON_STEPS = {
    "30-45 min": 9,
    "6 hours": 72,
    "12 hours": 144,
}


def clean_space_weather_data(
    frame: pd.DataFrame,
    freq: str = "5min",
    selected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Normalize cadence, repair missing values, and keep selected numeric signal columns."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Input dataframe must use a DatetimeIndex.")

    if selected_columns is None:
        selected_columns = SAMPLE_REQUIRED_COLUMNS if set(SAMPLE_REQUIRED_COLUMNS).issubset(frame.columns) else numeric_columns(frame)
    missing_columns = [column for column in selected_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if not selected_columns:
        raise ValueError("At least one numeric column is required for preprocessing.")

    cleaned = frame.sort_index()[selected_columns]
    cleaned = cleaned.resample(freq).median(numeric_only=True)
    cleaned = cleaned.interpolate(method="time", limit_direction="both")
    cleaned = cleaned.ffill().bfill()
    return cleaned.dropna(axis=1, how="all")


def build_features(frame: pd.DataFrame, target_column: str = "electron_flux") -> pd.DataFrame:
    """Create lag, rolling, and plasma-coupling features for forecasting models."""

    if target_column not in frame.columns:
        raise ValueError(f"Target column {target_column!r} is not available in the cleaned data.")

    featured = frame.copy()
    source_columns = list(frame.columns)
    for column in source_columns:
        for lag in (1, 3, 6, 12):
            featured[f"{column}_lag_{lag}"] = featured[column].shift(lag)
        for window in (3, 12, 36):
            featured[f"{column}_roll_mean_{window}"] = featured[column].rolling(window).mean()
            featured[f"{column}_roll_std_{window}"] = featured[column].rolling(window).std()

    if {"imf_bz", "solar_wind_speed"}.issubset(featured.columns):
        featured["southward_imf"] = featured["imf_bz"].clip(upper=0).abs()
        featured["solar_wind_coupling"] = featured["southward_imf"] * featured["solar_wind_speed"]
    if {"proton_density", "solar_wind_speed"}.issubset(featured.columns):
        featured["density_pressure_proxy"] = featured["proton_density"] * featured["solar_wind_speed"] ** 2
    _add_vector_magnitude(
        featured,
        ["U_eGSE_0", "U_eGSE_1", "U_eGSE_2"],
        "wind_speed_magnitude",
    )
    _add_vector_magnitude(
        featured,
        ["electron_velocity_gse_x", "electron_velocity_gse_y", "electron_velocity_gse_z"],
        "wind_speed_magnitude",
    )
    _add_vector_magnitude(
        featured,
        ["P_eGSE_0", "P_eGSE_1", "P_eGSE_2"],
        "pressure_vector_magnitude",
    )
    _add_vector_magnitude(
        featured,
        ["electron_pressure_gse_x", "electron_pressure_gse_y", "electron_pressure_gse_z"],
        "pressure_vector_magnitude",
    )

    return featured.dropna()


def build_supervised_dataset(featured: pd.DataFrame, horizon_steps: int, target_column: str = "electron_flux") -> tuple[pd.DataFrame, pd.Series]:
    """Create X/y data where y is the future value of the selected target column."""

    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive.")
    if target_column not in featured.columns:
        raise ValueError(f"Target column {target_column!r} is not available in the feature table.")
    dataset = featured.copy()
    dataset["forecast_target"] = dataset[target_column].shift(-horizon_steps)
    dataset = dataset.dropna()
    x = dataset.drop(columns=["forecast_target"])
    y = dataset["forecast_target"]
    return x, y


def _add_vector_magnitude(frame: pd.DataFrame, components: list[str], output_column: str) -> None:
    if set(components).issubset(frame.columns) and output_column not in frame.columns:
        frame[output_column] = sum(frame[component] ** 2 for component in components) ** 0.5
