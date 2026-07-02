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

## CDAWeb CDF Input Compatibility

The application supports CDAWeb CDF downloads stored in a local `combined/` folder. All `.cdf` files in that folder are loaded, concatenated by `Epoch`, sorted, deduplicated by timestamp, and exposed in the dashboard as selectable numeric fields.

Wind SWE vector variables such as `U_eGSE`, `UceGSE`, and `P_eGSE` are flattened into component columns such as `U_eGSE_0`, `U_eGSE_1`, and `U_eGSE_2` so they can be used directly as model inputs. Scalar fields such as `N_elec`, `T_elec`, `TcElec`, `Te_pal`, `Te_per`, `Te_ani`, `TecAni`, and `Gyrtrp` remain selectable by their CDAWeb variable names.
