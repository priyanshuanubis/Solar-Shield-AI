"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from solarshield_ai.modeling import ForecastResult


def target_history_chart(frame: pd.DataFrame, target_column: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=frame.index, y=frame[target_column], mode="lines", name=target_column))
    figure.update_layout(title=f"{target_column} History", xaxis_title="Time", yaxis_title=target_column)
    return figure


def prediction_comparison_chart(result: ForecastResult, target_column: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=result.actual.index, y=result.actual, mode="lines", name="Actual"))
    figure.add_trace(go.Scatter(x=result.predicted.index, y=result.predicted, mode="lines", name="Predicted"))
    figure.update_layout(title=f"{result.horizon} {target_column} Forecast Validation", xaxis_title="Time", yaxis_title=target_column)
    return figure
