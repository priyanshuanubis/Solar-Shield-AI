"""Data loading utilities for SolarShield AI."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["electron_flux", "solar_wind_speed", "proton_density", "imf_bz"]


@dataclass(frozen=True)
class DataSourceSummary:
    """Metadata describing a loaded dataset."""

    source_name: str
    rows: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def generate_sample_space_weather_data(periods: int = 7 * 24 * 12, freq: str = "5min", seed: int = 42) -> pd.DataFrame:
    """Generate deterministic sample GOES/Wind-like data for demos and tests."""

    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=periods, freq=freq, name="timestamp")
    hours = np.arange(periods) / 12

    solar_wind_speed = 430 + 55 * np.sin(hours / 9) + rng.normal(0, 18, periods)
    proton_density = 6 + 1.5 * np.cos(hours / 7) + rng.normal(0, 0.45, periods)
    imf_bz = -0.8 + 3.2 * np.sin(hours / 5) + rng.normal(0, 0.8, periods)
    geomagnetic_drive = np.clip(-imf_bz, 0, None) * solar_wind_speed / 500

    flux_base = 1.8e3 + 180 * np.sin(hours / 4)
    delayed_drive = pd.Series(geomagnetic_drive).rolling(9, min_periods=1).mean().to_numpy()
    electron_flux = flux_base + 850 * delayed_drive + rng.normal(0, 95, periods)
    electron_flux = np.clip(electron_flux, 50, None)

    return pd.DataFrame(
        {
            "electron_flux": electron_flux,
            "solar_wind_speed": solar_wind_speed,
            "proton_density": proton_density,
            "imf_bz": imf_bz,
        },
        index=index,
    )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a timestamp-indexed CSV file."""

    frame = pd.read_csv(path)
    timestamp_column = _first_present(frame.columns, ["timestamp", "time", "datetime", "date"])
    if timestamp_column is None:
        raise ValueError("CSV data must include a timestamp, time, datetime, or date column.")

    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=False)
    frame = frame.set_index(timestamp_column).sort_index()
    frame.index.name = "timestamp"
    return _coerce_numeric_frame(frame)


def load_cdf(path: str | Path, variables: Iterable[str] | None = None) -> pd.DataFrame:
    """Load selected variables from a CDF file into a timestamp-indexed dataframe."""

    if find_spec("cdflib") is None:
        raise ImportError("cdflib is required to read CDF files. Install dependencies from requirements.txt.")

    import cdflib

    cdf = cdflib.CDF(str(path))
    cdf_info = cdf.cdf_info()
    variable_names = list(variables) if variables else list(cdf_info.zVariables + cdf_info.rVariables)
    time_variable = _first_present(variable_names, ["Epoch", "epoch", "Time", "time", "timestamp"])
    if time_variable is None:
        raise ValueError("CDF file must contain an Epoch, Time, or timestamp variable.")

    epoch_values = cdf.varget(time_variable)
    timestamps = pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch_values))
    data: dict[str, np.ndarray] = {}
    for variable_name in variable_names:
        if variable_name == time_variable:
            continue
        values = np.asarray(cdf.varget(variable_name))
        if values.ndim == 1 and len(values) == len(timestamps):
            data[variable_name] = values

    frame = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name="timestamp")).sort_index()
    return _coerce_numeric_frame(frame)


def summarize_source(frame: pd.DataFrame, source_name: str) -> DataSourceSummary:
    """Return a compact summary for UI display."""

    if frame.empty:
        raise ValueError("Cannot summarize an empty dataframe.")
    return DataSourceSummary(source_name, len(frame), frame.index.min(), frame.index.max())


def _coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.apply(pd.to_numeric, errors="coerce")


def _first_present(values: Iterable[str], candidates: Iterable[str]) -> str | None:
    value_set = set(values)
    return next((candidate for candidate in candidates if candidate in value_set), None)
