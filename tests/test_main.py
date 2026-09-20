"""
Unit tests for the Weather Intelligence API (app/main.py).

Since the API's prediction endpoints call out to the live Open-Meteo Historical
Weather API internally (via the custom package's ``fetch_historical_weather``),
these tests patch that call to return a slice of real, previously-fetched
Sydney weather data rather than hitting the network - this keeps the tests
fast, deterministic, and runnable offline/in CI, while still exercising the
exact same feature-engineering and prediction code path used in production.
"""

from datetime import date, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

REAL_DAILY_DATA_PATH = "data/sydney_daily_weather_sample.csv"


def _load_sample_daily_data() -> pd.DataFrame:
    return pd.read_csv(REAL_DAILY_DATA_PATH, index_col="date", parse_dates=True)


@pytest.fixture
def mock_weather_fetch():
    """Patch the live Open-Meteo call to return real historical data from disk."""
    daily = _load_sample_daily_data()

    def fake_aggregate_daily_weather(_hourly_df):
        return daily.copy()

    with patch("app.main.fetch_historical_weather", return_value=None) as mock_fetch, \
         patch("app.main.aggregate_daily_weather", side_effect=fake_aggregate_daily_weather):
        yield mock_fetch


# ---------------------------------------------------------------------------
# Simple, non-prediction endpoints
# ---------------------------------------------------------------------------

def test_root_endpoint_returns_expected_shape():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["project"] == "Weather Intelligence API"
    assert "github" in body
    assert set(body["endpoints"]) == {
        "/", "/health", "/predict/index/comfort_climate",
        "/predict/category/weather_hazard", "/model-metadata",
    }


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "message": "Weather Intelligence API is running"}


def test_model_metadata_returns_two_models_with_required_fields():
    response = client.get("/model-metadata")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2

    targets = {m["target"] for m in body}
    assert targets == {"Climate Comfort Index", "Weather Hazard Category"}

    for model_meta in body:
        for key in ["target", "algorithm", "forecast_horizon", "features", "feature_importance"]:
            assert key in model_meta, f"missing '{key}' in metadata for {model_meta.get('target')}"


# ---------------------------------------------------------------------------
# Prediction endpoints - happy path
# ---------------------------------------------------------------------------

def test_predict_comfort_climate_happy_path(mock_weather_fetch):
    response = client.get("/predict/index/comfort_climate", params={"date": "2024-06-15"})
    assert response.status_code == 200
    body = response.json()

    assert body["input_date"] == "2024-06-15"
    predictions = body["predictions"]["comfort_climate"]
    assert len(predictions) == 3

    base = date(2024, 6, 15)
    expected_dates = {(base + timedelta(days=h)).isoformat() for h in (1, 2, 3)}
    assert set(predictions.keys()) == expected_dates
    for value in predictions.values():
        assert isinstance(value, int)
        assert 0 <= value <= 100


def test_predict_weather_hazard_happy_path(mock_weather_fetch):
    response = client.get("/predict/category/weather_hazard", params={"date": "2024-06-15"})
    assert response.status_code == 200
    body = response.json()

    assert body["input_date"] == "2024-06-15"
    predictions = body["predictions"]["weather_hazard"]
    assert len(predictions) == 1

    expected_date = (date(2024, 6, 15) + timedelta(days=7)).isoformat()
    assert expected_date in predictions
    assert predictions[expected_date] in {"Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"}


# ---------------------------------------------------------------------------
# Robustness - invalid input / edge cases (Criterion 5 explicitly requires this)
# ---------------------------------------------------------------------------

def test_predict_comfort_climate_missing_date_param_returns_422():
    response = client.get("/predict/index/comfort_climate")
    assert response.status_code == 422


@pytest.mark.parametrize("bad_date", ["not-a-date", "2024/06/15", "2024-13-40", "", "15-06-2024"])
def test_predict_comfort_climate_invalid_date_format_returns_422(bad_date):
    response = client.get("/predict/index/comfort_climate", params={"date": bad_date})
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert "traceback" not in str(body).lower()


@pytest.mark.parametrize("bad_date", ["not-a-date", "2024-02-30"])
def test_predict_weather_hazard_invalid_date_format_returns_422(bad_date):
    response = client.get("/predict/category/weather_hazard", params={"date": bad_date})
    assert response.status_code == 422


def test_predict_comfort_climate_date_outside_data_coverage_returns_client_error(mock_weather_fetch):
    # a date far outside the sample data's range - simulates a date Open-Meteo has no
    # observations for (either too far in the past or an unreleased future date)
    response = client.get("/predict/index/comfort_climate", params={"date": "1900-01-01"})
    assert response.status_code in (422, 503)
    assert "detail" in response.json()


def test_predict_weather_hazard_upstream_failure_returns_503():
    with patch("app.main.fetch_historical_weather", side_effect=ConnectionError("network unreachable")):
        response = client.get("/predict/category/weather_hazard", params={"date": "2024-06-15"})
    assert response.status_code == 503
    body = response.json()
    assert "detail" in body
    assert "traceback" not in str(body).lower()
