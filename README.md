# SolarShield AI

SolarShield AI is a hackathon-ready Python and Streamlit application for forecasting energetic electron/plasma variables from CDAWeb CDF downloads. It follows the development document architecture: CDF/CSV ingestion, cleaning, feature engineering, supervised dataset creation, machine-learning forecasting, evaluation, and an interactive dashboard.

## Features

- Reads a whole local `combined/` folder of CDAWeb `.cdf` files, including daily Wind SWE files such as `wi_h5_swe_20250227_v01.cdf`.
- Reads multiple uploaded CSV/CDF files and merges them into one timestamp-indexed dataset.
- Flattens CDAWeb vector/tensor variables into model-ready component columns, for example `U_eGSE_0`, `U_eGSE_1`, and `U_eGSE_2`.
- Supports Wind SWE fields shown by CDAWeb/cdflib, including `N_elec`, `U_eGSE`, `UceGSE`, `P_eGSE`, `T_elec`, `TcElec`, `W_elec`, `WcElec`, `Te_pal`, `Te_per`, `TecPal`, `TecPer`, `Te_ani`, `TecAni`, and `Gyrtrp` when present in the files.
- Lets the user choose the forecast target and model input fields directly in the dashboard, with Wind SWE columns ordered ahead of generic variables.
- Provides a deterministic built-in sample dataset for demos when mission data is unavailable.
- Cleans and resamples time-series data to a configurable cadence.
- Engineers lag, rolling-window, and plasma/vector magnitude features.
- Trains separate models for 30-45 minute, 6 hour, and 12 hour forecasts.
- Displays metrics, validation charts, loaded fields, cleaned data, and feature previews in Streamlit.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src streamlit run app.py
```

## Using CDAWeb CDF files

1. Download CDF files from CDAWeb.
2. Put all files that should be processed together into a folder named `combined/` at the repository root, or enter a different folder path in the dashboard sidebar.
3. Start the app and keep **Input mode** set to **Local combined folder**.
4. Choose the forecast target, such as `N_elec` for Wind SWE electron density or any other numeric CDAWeb variable exposed by the files.
5. Choose the input fields to use as model features and run the dashboard.

The app concatenates all `.cdf` files in the folder, sorts by `Epoch`, averages duplicate timestamps, replaces common CDF fill values with missing values, and resamples the data before feature engineering.

## CSV input schema

CSV uploads should include a timestamp-like column named `timestamp`, `time`, `datetime`, or `date`, plus any numeric columns you want to forecast from. The sample dataset uses these columns:

- `electron_flux`
- `solar_wind_speed`
- `proton_density`
- `imf_bz`

## Project layout

```text
app.py                         Streamlit dashboard
src/solarshield_ai/data.py      CSV/CDF/folder readers and sample data generation
src/solarshield_ai/preprocessing.py Cleaning, features, and target building
src/solarshield_ai/modeling.py  Forecast model training and evaluation
src/solarshield_ai/visualization.py Plotly charts
tests/                         Unit tests for the forecasting pipeline
```
