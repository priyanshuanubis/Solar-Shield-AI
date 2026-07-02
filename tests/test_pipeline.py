from solarshield_ai.data import generate_sample_space_weather_data
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


def test_cleaning_requires_expected_columns():
    raw = generate_sample_space_weather_data(periods=48).drop(columns=["imf_bz"])

    try:
        clean_space_weather_data(raw)
    except ValueError as exc:
        assert "imf_bz" in str(exc)
    else:
        raise AssertionError("clean_space_weather_data should reject missing required columns")
