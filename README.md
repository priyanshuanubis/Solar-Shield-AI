# SolarShield AI

SolarShield AI is a hackathon-ready Python and Streamlit application for forecasting energetic electron fluxes in the radiation environment around geostationary satellites. It follows the development document architecture: CDF/CSV ingestion, cleaning, feature engineering, supervised dataset creation, machine-learning forecasting, evaluation, and an interactive dashboard.

## Features

- Reads CSV files and CDF files from GOES/Wind-style spacecraft datasets.
- Provides a deterministic built-in sample dataset for demos when mission data is unavailable.
- Cleans and resamples time-series data to a configurable cadence.
- Engineers lag, rolling-window, and solar-wind coupling features.
- Trains separate models for 30-45 minute, 6 hour, and 12 hour energetic electron flux forecasts.
- Displays metrics, validation charts, and feature previews in Streamlit.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src streamlit run app.py
```

## Expected input schema

CSV uploads should include a timestamp-like column named `timestamp`, `time`, `datetime`, or `date`, plus these numeric columns:

- `electron_flux`
- `solar_wind_speed`
- `proton_density`
- `imf_bz`

CDF uploads should include an epoch/time variable and one-dimensional variables that can be mapped into the same forecasting workflow.

## Project layout

```text
app.py                         Streamlit dashboard
src/solarshield_ai/data.py      CSV/CDF readers and sample data generation
src/solarshield_ai/preprocessing.py Cleaning, features, and target building
src/solarshield_ai/modeling.py  Forecast model training and evaluation
src/solarshield_ai/visualization.py Plotly charts
tests/                         Unit tests for the forecasting pipeline
```
