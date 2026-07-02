"""CDF processing pipeline for SolarShield AI.

Pipeline implemented here mirrors the requested flow:

CDF -> read CDF -> extract variables -> rename variables -> merge datasets -> clean_space_weather_data()
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from solarshield_ai.data import combine_frames, load_cdf

CDF_VARIABLE_RENAME_MAP = {
    "N_elec": "electron_density",
    "U_eGSE_0": "electron_velocity_gse_x",
    "U_eGSE_1": "electron_velocity_gse_y",
    "U_eGSE_2": "electron_velocity_gse_z",
    "UceGSE_0": "core_electron_velocity_gse_x",
    "UceGSE_1": "core_electron_velocity_gse_y",
    "UceGSE_2": "core_electron_velocity_gse_z",
    "P_eGSE_0": "electron_pressure_gse_x",
    "P_eGSE_1": "electron_pressure_gse_y",
    "P_eGSE_2": "electron_pressure_gse_z",
    "T_elec": "electron_temperature",
    "TcElec": "core_electron_temperature",
    "W_elec_0": "electron_heat_flux_x",
    "W_elec_1": "electron_heat_flux_y",
    "W_elec_2": "electron_heat_flux_z",
    "WcElec_0": "core_electron_heat_flux_x",
    "WcElec_1": "core_electron_heat_flux_y",
    "WcElec_2": "core_electron_heat_flux_z",
    "Te_pal": "electron_parallel_temperature",
    "Te_per": "electron_perpendicular_temperature",
    "TecPal": "core_electron_parallel_temperature",
    "TecPer": "core_electron_perpendicular_temperature",
    "Te_ani": "electron_temperature_anisotropy",
    "TecAni": "core_electron_temperature_anisotropy",
    "Gyrtrp": "electron_gyrotropy",
}


def read_cdf(path: str | Path) -> pd.DataFrame:
    """Read a single CDF file and flatten CDAWeb variables into dataframe columns."""

    return load_cdf(path)


def extract_variables(frame: pd.DataFrame, variables: Iterable[str] | None = None) -> pd.DataFrame:
    """Extract requested variables/components while keeping every available variable by default."""

    if variables is None:
        return frame.copy()
    requested = list(variables)
    available = [column for column in requested if column in frame.columns]
    if not available:
        raise ValueError("None of the requested CDF variables were found in the file.")
    return frame[available].copy()


def rename_variables(frame: pd.DataFrame, rename_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Rename CDAWeb variable names to clear model-facing feature names."""

    return frame.rename(columns=rename_map or CDF_VARIABLE_RENAME_MAP)


def merge_datasets(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge daily CDF datasets into one timestamp-indexed dataset."""

    return combine_frames(frames)


def process_cdf_file(path: str | Path, variables: Iterable[str] | None = None) -> pd.DataFrame:
    """Run read, extract, and rename steps for one CDF file."""

    cdf_frame = read_cdf(path)
    extracted = extract_variables(cdf_frame, variables=variables)
    return rename_variables(extracted)


def process_cdf_folder(folder: str | Path, variables: Iterable[str] | None = None, pattern: str = "*.cdf") -> pd.DataFrame:
    """Run the full CDF pipeline through the merge step for every file in a folder."""

    folder_path = Path(folder).expanduser().resolve()
    cdf_files = sorted(folder_path.glob(pattern))
    if not cdf_files:
        raise FileNotFoundError(f"No CDF files matching {pattern!r} were found in {folder_path}.")
    processed_files = [process_cdf_file(cdf_file, variables=variables) for cdf_file in cdf_files]
    return merge_datasets(processed_files)
