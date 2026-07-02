"""Streamlit dashboard for SolarShield AI."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from solarshield_ai.data import generate_sample_space_weather_data, load_cdf, load_csv, summarize_source
from solarshield_ai.modeling import train_forecast_models
from solarshield_ai.preprocessing import build_features, clean_space_weather_data
from solarshield_ai.visualization import flux_history_chart, prediction_comparison_chart

st.set_page_config(page_title="SolarShield AI", page_icon="🛰️", layout="wide")

st.title("🛰️ SolarShield AI")
st.caption("Forecasting energetic particle radiation environment for geostationary satellites")

with st.sidebar:
    st.header("Data source")
    uploaded_file = st.file_uploader("Upload GOES/Wind CSV or CDF", type=["csv", "cdf"])
    model_name = st.selectbox("Model", ["Gradient Boosting", "Random Forest", "XGBoost"])
    cadence = st.selectbox("Resampling cadence", ["5min", "10min", "15min"], index=0)

@st.cache_data(show_spinner=False)
def load_uploaded_file(name: str, suffix: str, content: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    if suffix.lower() == ".csv":
        return load_csv(temp_path)
    return load_cdf(temp_path)

if uploaded_file is None:
    raw_data = generate_sample_space_weather_data()
    source_name = "Built-in sample GOES/Wind-like dataset"
else:
    raw_data = load_uploaded_file(uploaded_file.name, Path(uploaded_file.name).suffix, uploaded_file.getvalue())
    source_name = uploaded_file.name

cleaned_data = clean_space_weather_data(raw_data, freq=cadence)
features = build_features(cleaned_data)
summary = summarize_source(cleaned_data, source_name)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Rows", f"{summary.rows:,}")
col_b.metric("Start", summary.start_time.strftime("%Y-%m-%d %H:%M"))
col_c.metric("End", summary.end_time.strftime("%Y-%m-%d %H:%M"))

st.plotly_chart(flux_history_chart(cleaned_data), use_container_width=True)

with st.spinner("Training forecast models..."):
    results = train_forecast_models(features, model_name=model_name)

st.subheader("Forecasts")
forecast_columns = st.columns(len(results))
for column, result in zip(forecast_columns, results, strict=True):
    column.metric(result.horizon, f"{result.prediction:,.0f}", help="Predicted energetic electron flux")
    column.write(f"MAE: {result.mae:,.1f}")
    column.write(f"RMSE: {result.rmse:,.1f}")
    column.write(f"R²: {result.r2:.3f}")

selected_horizon = st.selectbox("Validation chart horizon", [result.horizon for result in results])
selected_result = next(result for result in results if result.horizon == selected_horizon)
st.plotly_chart(prediction_comparison_chart(selected_result), use_container_width=True)

with st.expander("Feature preview"):
    st.dataframe(features.tail(50), use_container_width=True)
