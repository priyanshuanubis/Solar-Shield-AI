import pandas as pd

from solarshield_ai.data import combine_frames, default_target_column, generate_sample_space_weather_data
from solarshield_ai.modeling import train_forecast_models
from solarshield_ai.preprocessing import HORIZON_STEPS, build_features, clean_space_weather_data


def test_sample_pipeline_trains_all_forecast_horizons():
    raw = generate_sample_space_weather_data(periods=420)
    cleaned = clean_space_weather_data(raw)
    features = build_features(cleaned)

    results = train_forecast_models(features)

    assert [result.horizon for result in results] == list(HORIZON_STEPS)
    assert all(result.prediction > 0 for result in results)
    assert all(result.actual.shape == result.predicted.shape for result in results)


def test_cleaning_requires_selected_columns():
    raw = generate_sample_space_weather_data(periods=48).drop(columns=["imf_bz"])

    try:
        clean_space_weather_data(raw, selected_columns=["electron_flux", "imf_bz"])
    except ValueError as exc:
        assert "imf_bz" in str(exc)
    else:
        raise AssertionError("clean_space_weather_data should reject missing selected columns")


def test_wind_swe_style_fields_can_drive_forecast_target_selection():
    index = pd.date_range("2025-02-27", periods=420, freq="5min", name="timestamp")
    raw = pd.DataFrame(
        {
            "N_elec": range(420),
            "U_eGSE_0": range(1, 421),
            "U_eGSE_1": range(2, 422),
            "U_eGSE_2": range(3, 423),
            "T_elec": range(4, 424),
        },
        index=index,
    )

    cleaned = clean_space_weather_data(raw, selected_columns=list(raw.columns))
    features = build_features(cleaned, target_column="N_elec")
    results = train_forecast_models(features, target_column="N_elec")

    assert default_target_column(raw) == "N_elec"
    assert "wind_speed_magnitude" in features.columns
    assert [result.horizon for result in results] == list(HORIZON_STEPS)


def test_combine_frames_collapses_duplicate_timestamps():
    index = pd.date_range("2025-02-27", periods=2, freq="5min", name="timestamp")
    first = pd.DataFrame({"N_elec": [1.0, 2.0]}, index=index)
    second = pd.DataFrame({"N_elec": [3.0]}, index=index[:1])

    combined = combine_frames([first, second])

    assert len(combined) == 2
    assert combined.loc[index[0], "N_elec"] == 2.0
