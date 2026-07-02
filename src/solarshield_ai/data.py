"""Data loading utilities for SolarShield AI."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_TARGET_COLUMN = "electron_flux"
SAMPLE_FEATURE_COLUMNS = ["solar_wind_speed", "proton_density", "imf_bz"]
SAMPLE_REQUIRED_COLUMNS = [DEFAULT_TARGET_COLUMN, *SAMPLE_FEATURE_COLUMNS]
CDF_TIME_CANDIDATES = ["Epoch", "epoch", "Time", "time", "Timestamp", "timestamp"]
CDF_FILL_SENTINELS = (-1.0e31, -1.0e30, 1.0e30, 1.0e31)


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
    return _clean_numeric_frame(frame)


def load_cdf(path: str | Path, variables: Iterable[str] | None = None) -> pd.DataFrame:
    """Load CDAWeb CDF variables into a timestamp-indexed dataframe.

    One-dimensional variables become one dataframe column. Multi-dimensional variables,
    such as Wind SWE vector fields (`U_eGSE`, `P_eGSE`, etc.), are flattened into
    component columns like `U_eGSE_0`, `U_eGSE_1`, and `U_eGSE_2`.
    """

    cdflib = _require_cdflib()
    cdf = cdflib.CDF(str(path))
    cdf_info = cdf.cdf_info()
    all_variables = list(getattr(cdf_info, "zVariables", [])) + list(getattr(cdf_info, "rVariables", []))
    variable_names = list(variables) if variables else all_variables
    time_variable = _first_present(variable_names, CDF_TIME_CANDIDATES) or _first_present(all_variables, CDF_TIME_CANDIDATES)
    if time_variable is None:
        raise ValueError(f"{path} must contain one of these time variables: {', '.join(CDF_TIME_CANDIDATES)}")

    timestamps = _cdf_epoch_to_datetime(cdflib, cdf.varget(time_variable))
    data: dict[str, np.ndarray] = {}
    for variable_name in variable_names:
        if variable_name == time_variable or variable_name not in all_variables:
            continue
        values = np.asarray(cdf.varget(variable_name))
        data.update(_flatten_cdf_variable(variable_name, values, len(timestamps)))

    frame = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, name="timestamp")).sort_index()
    return _clean_numeric_frame(frame)


def load_cdf_folder(folder: str | Path, pattern: str = "*.cdf") -> pd.DataFrame:
    """Load and concatenate every CDAWeb CDF file in a folder.

    This is intended for local `combined/` folders containing many daily CDAWeb
    downloads such as `wi_h5_swe_YYYYMMDD_v01.cdf`.
    """

    folder_path = Path(folder).expanduser().resolve()
    files = sorted(folder_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CDF files matching {pattern!r} were found in {folder_path}.")
    return combine_frames([load_cdf(file_path) for file_path in files])


def combine_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate timestamp-indexed frames and collapse duplicate timestamps."""

    usable_frames = [frame for frame in frames if not frame.empty]
    if not usable_frames:
        raise ValueError("No usable data frames were loaded.")
    combined = pd.concat(usable_frames, axis=0, sort=True).sort_index()
    combined = combined.groupby(level=0).mean(numeric_only=True)
    combined.index.name = "timestamp"
    return _clean_numeric_frame(combined)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Return dataframe columns with at least one numeric value."""

    return [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any()]


def default_target_column(frame: pd.DataFrame) -> str:
    """Choose a sensible default target from CDAWeb or sample-data columns."""

    preferred = ["electron_flux", "N_elec", "T_elec", "Te_pal", "Te_per"]
    available = numeric_columns(frame)
    for column in preferred:
        if column in available:
            return column
    if not available:
        raise ValueError("No numeric columns are available for forecasting.")
    return available[0]


def summarize_source(frame: pd.DataFrame, source_name: str) -> DataSourceSummary:
    """Return a compact summary for UI display."""

    if frame.empty:
        raise ValueError("Cannot summarize an empty dataframe.")
    return DataSourceSummary(source_name, len(frame), frame.index.min(), frame.index.max())


def _require_cdflib():
    if find_spec("cdflib") is None:
        raise ImportError("cdflib is required to read CDF files. Install dependencies from requirements.txt.")
    import cdflib

    return cdflib


def _cdf_epoch_to_datetime(cdflib, epoch_values: np.ndarray) -> pd.DatetimeIndex:
    converted = cdflib.cdfepoch.to_datetime(epoch_values)
    return pd.DatetimeIndex(pd.to_datetime(converted))


def _flatten_cdf_variable(variable_name: str, values: np.ndarray, expected_rows: int) -> dict[str, np.ndarray]:
    if values.ndim == 1 and len(values) == expected_rows:
        return {variable_name: values}
    if values.ndim == 2 and values.shape[0] == expected_rows:
        return {f"{variable_name}_{component}": values[:, component] for component in range(values.shape[1])}
    if values.ndim > 2 and values.shape[0] == expected_rows:
        flattened = values.reshape(expected_rows, -1)
        return {f"{variable_name}_{component}": flattened[:, component] for component in range(flattened.shape[1])}
    return {}


def _clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.apply(pd.to_numeric, errors="coerce")
    for sentinel in CDF_FILL_SENTINELS:
        cleaned = cleaned.mask(np.isclose(cleaned, sentinel, rtol=0, atol=abs(sentinel) * 1e-6))
    return cleaned.replace([np.inf, -np.inf], np.nan)


def _first_present(values: Iterable[str], candidates: Iterable[str]) -> str | None:
    value_set = set(values)
    return next((candidate for candidate in candidates if candidate in value_set), None)
