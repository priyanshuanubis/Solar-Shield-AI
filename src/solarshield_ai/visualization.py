"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from solarshield_ai.modeling import ForecastResult


def flux_history_chart(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame.index, y=frame["electron_flux"], mode="lines", name="Electron flux"))
    figure.update_layout(title="Energetic Electron Flux History", xaxis_title="Time", yaxis_title="Flux")
    return figure


def prediction_comparison_chart(result: ForecastResult) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=result.actual.index, y=result.actual, mode="lines", name="Actual"))
    figure.add_trace(go.Scatter(x=result.predicted.index, y=result.predicted, mode="lines", name="Predicted"))
    figure.update_layout(title=f"{result.horizon} Forecast Validation", xaxis_title="Time", yaxis_title="Flux")
    return figure
