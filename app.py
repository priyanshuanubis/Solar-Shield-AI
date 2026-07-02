"""Streamlit dashboard for SolarShield AI."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from solarshield_ai.data import (
    combine_frames,
    default_target_column,
    generate_sample_space_weather_data,
    load_csv,
    default_feature_columns,
    ordered_numeric_columns,
    summarize_source,
)
from solarshield_ai.modeling import train_forecast_models
from solarshield_ai.pipeline import process_cdf_file, process_cdf_folder
from solarshield_ai.preprocessing import build_features, clean_space_weather_data
from solarshield_ai.visualization import prediction_comparison_chart, target_history_chart

st.set_page_config(page_title="SolarShield AI", page_icon="🛰️", layout="wide")

st.title("🛰️ SolarShield AI")
st.caption("Forecasting energetic particle radiation environment for geostationary satellites")

with st.sidebar:
    st.header("Data source")
    data_mode = st.radio("Input mode", ["Local combined folder", "Upload files", "Sample demo data"])
    combined_folder = st.text_input("Combined CDF folder", value="combined", help="Folder containing CDAWeb .cdf files")
    uploaded_files = st.file_uploader("Upload CDAWeb CSV or CDF files", type=["csv", "cdf"], accept_multiple_files=True)
    model_name = st.selectbox("Model", ["Gradient Boosting", "Random Forest", "XGBoost"])
    cadence = st.selectbox("Resampling cadence", ["5min", "10min", "15min"], index=0)


@st.cache_data(show_spinner=False)
def load_uploaded_file(name: str, suffix: str, content: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    if suffix.lower() == ".csv":
        return load_csv(temp_path)
    return process_cdf_file(temp_path)


@st.cache_data(show_spinner=False)
def load_combined_folder(folder: str) -> pd.DataFrame:
    return process_cdf_folder(folder)


def load_dashboard_data() -> tuple[pd.DataFrame, str]:
    if data_mode == "Sample demo data":
        return generate_sample_space_weather_data(), "Built-in sample GOES/Wind-like dataset"
    if data_mode == "Upload files" and uploaded_files:
        frames = [load_uploaded_file(file.name, Path(file.name).suffix, file.getvalue()) for file in uploaded_files]
        return combine_frames(frames), f"{len(uploaded_files)} uploaded file(s)"
    if data_mode == "Upload files":
        st.info("Upload one or more CSV/CDF files, or switch to the local combined folder mode.")
        st.stop()
    return load_combined_folder(combined_folder), f"Local CDF folder: {combined_folder}"


raw_data, source_name = load_dashboard_data()
available_columns = ordered_numeric_columns(raw_data)
if not available_columns:
    st.error("No numeric CDAWeb variables were found in the selected input.")
    st.stop()

default_target = default_target_column(raw_data)
with st.sidebar:
    target_column = st.selectbox("Forecast target", available_columns, index=available_columns.index(default_target))
    default_features = default_feature_columns(raw_data, target_column)
    selected_features = st.multiselect(
        "Input fields",
        available_columns,
        default=default_features,
        help="Select CDAWeb variables/components to use as model inputs.",
    )

selected_columns = [target_column, *[column for column in selected_features if column != target_column]]
cleaned_data = clean_space_weather_data(raw_data, freq=cadence, selected_columns=selected_columns)
features = build_features(cleaned_data, target_column=target_column)
summary = summarize_source(cleaned_data, source_name)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Rows", f"{summary.rows:,}")
col_b.metric("Fields", f"{len(cleaned_data.columns):,}")
col_c.metric("Start", summary.start_time.strftime("%Y-%m-%d %H:%M"))
col_d.metric("End", summary.end_time.strftime("%Y-%m-%d %H:%M"))

with st.expander("Loaded CDAWeb fields", expanded=False):
    st.write(", ".join(available_columns))

st.plotly_chart(target_history_chart(cleaned_data, target_column), use_container_width=True)

with st.spinner("Training forecast models..."):
    results = train_forecast_models(features, model_name=model_name, target_column=target_column)

st.subheader("Forecasts")
forecast_columns = st.columns(len(results))
for column, result in zip(forecast_columns, results, strict=True):
    column.metric(result.horizon, f"{result.prediction:,.3g}", help=f"Predicted future {target_column}")
    column.write(f"MAE: {result.mae:,.3g}")
    column.write(f"RMSE: {result.rmse:,.3g}")
    column.write(f"R²: {result.r2:.3f}")

selected_horizon = st.selectbox("Validation chart horizon", [result.horizon for result in results])
selected_result = next(result for result in results if result.horizon == selected_horizon)
st.plotly_chart(prediction_comparison_chart(selected_result, target_column), use_container_width=True)

with st.expander("Cleaned data preview"):
    st.dataframe(cleaned_data.tail(50), use_container_width=True)

with st.expander("Feature preview"):
    st.dataframe(features.tail(50), use_container_width=True)
