# SolarShield AI - Development Document

## Hackathon Project

**Forecasting Energetic Particle Radiation Environment for ISRO's Geostationary Satellites**

## Problem Statement

Develop an AI/ML system to forecast energetic electron fluxes 30-45 minutes, 6 hours, and 12 hours ahead using GOES and Wind spacecraft data.

## Objectives

- Read CDF files from GOES and Wind spacecraft data sources.
- Preprocess and clean time-series radiation environment data.
- Engineer predictive features for energetic electron flux forecasting.
- Train forecasting models for short- and medium-horizon prediction windows.
- Evaluate predictions against measured electron fluxes.
- Build an interactive dashboard for visualization and demonstration.

## Architecture

```text
CDF Files -> Reader -> Cleaning -> Feature Engineering -> Dataset Builder -> LSTM/XGBoost -> Evaluation -> Streamlit Dashboard
```

## Technology Stack

- Python
- Pandas
- NumPy
- cdflib
- Plotly
- Scikit-learn
- TensorFlow or PyTorch
- XGBoost
- Streamlit

## Forecast Horizons

- 30-45 minutes ahead
- 6 hours ahead
- 12 hours ahead

## Deliverables

- Forecasting model
- Interactive dashboard
- Documentation
- Presentation and demo
