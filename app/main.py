"""
FastAPI service for AT2 - Machine Learning as a Service (36120, Session 26SP).

Serves two trained models via REST endpoints:
- Climate Comfort Index (CCI): regression, 0-100 scale, forecast 1/2/3 days ahead.
- Weather Hazard Category (WHC): 4-class classification, forecast exactly 7 days ahead.

Both models were trained in the companion experimentation repository
(36120-at2-experiments) on historical Sydney, NSW weather data from the Open-Meteo
Historical Weather API. This service loads the saved pipelines (preprocessing +
model bundled together via sklearn.Pipeline) and, given only a calendar date,
fetches the recent weather history needed to reconstruct the same features used
at training time before calling .predict().

Author: Vignesh Selvam (25740287)
"""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from joblib import load
from starlette.responses import JSONResponse

from mykrml_25740287.data.weather import aggregate_daily_weather, fetch_historical_weather

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------

GITHUB_REPO_URL = "https://github.com/vigneshcrack/36120-at2-api"

app = FastAPI(
    title="Weather Intelligence API",
    description=(
        "Machine learning services for weather forecasting: Climate Comfort Index "
        "(regression) and Weather Hazard Category (classification) for Sydney, NSW, "
        "built on Open-Meteo historical weather data."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Load trained pipelines ONCE at module level (not inside route functions), so
# the (relatively expensive) model deserialisation happens once at startup.
# ---------------------------------------------------------------------------

MODELS_DIR = "models"

CCI_FEATURES = [
    "temperature_2m_mean", "relative_humidity_2m_mean", "wind_gusts_10m_max",
    "cloud_cover_mean", "precipitation_sum", "month_sin", "month_cos",
    "temp_7d_roll_mean", "precip_3d_roll_sum",
]
WHC_FEATURES = [
    "precipitation_sum", "wind_gusts_10m_max", "cloud_cover_mean",
    "temperature_2m_mean", "month_sin", "month_cos", "precip_3d_roll_sum",
    "wind_gusts_7d_roll_max", "cloud_3d_roll_mean",
]

# how many days of trailing history are needed to compute every rolling feature
# above (7-day windows are the longest used by either model) - we fetch a
# slightly larger buffer than strictly required as a safety margin against
# the occasional missing day in an upstream data source.
HISTORY_BUFFER_DAYS = 14

try:
    cci_models = {
        1: load(f"{MODELS_DIR}/comfort_climate_h1.joblib"),
        2: load(f"{MODELS_DIR}/comfort_climate_h2.joblib"),
        3: load(f"{MODELS_DIR}/comfort_climate_h3.joblib"),
    }
    whc_model = load(f"{MODELS_DIR}/weather_hazard_h7.joblib")
except FileNotFoundError as exc:  # pragma: no cover - fails fast at startup, not per-request
    raise RuntimeError(
        f"Could not load model artefacts from '{MODELS_DIR}/'. Make sure the models/ "
        f"directory (with comfort_climate_h1.joblib, comfort_climate_h2.joblib, "
        f"comfort_climate_h3.joblib, weather_hazard_h7.joblib) is present next to app/. "
        f"Original error: {exc}"
    ) from exc

# recorded once from the experimentation repo's held-out TEST SET evaluation
# (notebooks/comfort_climate and notebooks/weather_hazard, Section J.4) - these
# are reported here rather than recomputed live, since the API does not carry
# the test split around; the notebooks are the source of truth for these figures.
CCI_TEST_METRICS = {"rmse": 10.23, "mae": 7.78, "r2": 0.19}
WHC_TEST_METRICS = {"accuracy": 0.479, "macro_f1": 0.260, "weighted_f1": 0.488}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> date:
    """Parse and validate a YYYY-MM-DD date string.

    Parameters
    ----------
    date_str : str
        The date string supplied by the caller as a query parameter.

    Returns
    -------
    date
        The parsed date.

    Raises
    ------
    HTTPException
        422 Unprocessable Entity if the string is not a valid YYYY-MM-DD date.
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid 'date' value: {date_str!r}. Expected an ISO-8601 date in "
                f"YYYY-MM-DD format, e.g. '2025-01-01'."
            ),
        ) from exc


def _build_daily_features(input_date: date) -> pd.DataFrame:
    """Fetch recent history and engineer the feature row for ``input_date``.

    Reconstructs, from live Open-Meteo historical data, the exact same features
    used at training time in the experimentation notebooks: same-day readings,
    cyclical month encoding, and 7-day/3-day trailing rolling statistics.

    Parameters
    ----------
    input_date : date
        The calendar date to build a feature row for.

    Returns
    -------
    pandas.DataFrame
        A single-row DataFrame indexed by ``input_date`` with every feature
        required by both the CCI and WHC models.

    Raises
    ------
    HTTPException
        503 if the upstream weather data source cannot be reached.
        422 if not enough historical data is available for ``input_date`` to
        compute the rolling features (e.g. a date outside Open-Meteo's
        historical coverage, or too close to the start of its records).
    """
    start = input_date - timedelta(days=HISTORY_BUFFER_DAYS)

    try:
        hourly = fetch_historical_weather(start_date=str(start), end_date=str(input_date))
        daily = aggregate_daily_weather(hourly)
    except Exception as exc:  # noqa: BLE001 - any upstream failure becomes a clean 503
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to retrieve historical weather data needed to build a "
                f"prediction for {input_date.isoformat()}. The upstream weather data "
                f"source may be temporarily unavailable. Original error: {exc}"
            ),
        ) from exc

    daily["month"] = daily.index.month
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
    daily["temp_7d_roll_mean"] = daily["temperature_2m_mean"].rolling(7).mean()
    daily["precip_3d_roll_sum"] = daily["precipitation_sum"].rolling(3).sum()
    daily["wind_gusts_7d_roll_max"] = daily["wind_gusts_10m_max"].rolling(7).max()
    daily["cloud_3d_roll_mean"] = daily["cloud_cover_mean"].rolling(3).mean()

    target_ts = pd.Timestamp(input_date)
    if target_ts not in daily.index:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No weather observations are available for {input_date.isoformat()}. "
                f"Please choose a date within Open-Meteo's historical coverage."
            ),
        )

    row = daily.loc[[target_ts]]
    required_cols = set(CCI_FEATURES) | set(WHC_FEATURES)
    if row[list(required_cols)].isna().any(axis=None):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient trailing history to compute rolling features for "
                f"{input_date.isoformat()} (need at least {HISTORY_BUFFER_DAYS} days of "
                f"prior observations). Please choose a later date."
            ),
        )
    return row


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def read_root():
    """Project overview and self-documentation for the API."""
    return {
        "project": "Weather Intelligence API",
        "description": (
            "Machine learning services for weather forecasting: Climate Comfort "
            "Index (regression, 1-3 days ahead) and Weather Hazard Category "
            "(classification, 7 days ahead) for Sydney, NSW, built on Open-Meteo "
            "historical weather data."
        ),
        "version": "1.0.0",
        "github": GITHUB_REPO_URL,
        "endpoints": [
            "/",
            "/health",
            "/predict/index/comfort_climate",
            "/predict/category/weather_hazard",
            "/model-metadata",
        ],
    }


@app.get("/health", status_code=200)
def health_check():
    """Simple liveness check."""
    return {"status": "healthy", "message": "Weather Intelligence API is running"}


@app.get("/model-metadata")
def model_metadata():
    """Metadata describing both trained models: algorithm, features, and performance."""
    cci_model_step = cci_models[3].named_steps["model"]
    cci_importance = dict(
        zip(CCI_FEATURES, [round(float(v), 4) for v in cci_model_step.feature_importances_])
    )
    whc_model_step = whc_model.named_steps["model"]
    whc_importance = dict(
        zip(WHC_FEATURES, [round(float(v), 4) for v in whc_model_step.feature_importances_])
    )

    return [
        {
            "target": "Climate Comfort Index",
            "prediction_type": "regression",
            "algorithm": "Gradient Boosting Regressor (systematically tuned via GridSearchCV "
                         "with TimeSeriesSplit)",
            "forecast_horizon": "1, 2 and 3 days ahead",
            "features": CCI_FEATURES,
            "feature_importance": cci_importance,
            "model_performance": {
                "description": "3-day-ahead model, evaluated on the held-out 2024-2025 test set",
                **CCI_TEST_METRICS,
            },
        },
        {
            "target": "Weather Hazard Category",
            "prediction_type": "classification",
            "algorithm": "Random Forest Classifier (class_weight='balanced')",
            "forecast_horizon": "7 days ahead",
            "features": WHC_FEATURES,
            "feature_importance": whc_importance,
            "model_performance": {
                "description": "Evaluated on the held-out 2024-2025 test set",
                **WHC_TEST_METRICS,
            },
        },
    ]


@app.get("/predict/index/comfort_climate")
def predict_comfort_climate(
    date: str = Query(..., description="Reference date in YYYY-MM-DD format")
):
    """Predict the Climate Comfort Index for 1, 2 and 3 days after ``date``."""
    input_date = _parse_date(date)
    features = _build_daily_features(input_date)

    predictions = {}
    for horizon, pipe in cci_models.items():
        pred = pipe.predict(features[CCI_FEATURES])[0]
        target_date = input_date + timedelta(days=horizon)
        predictions[target_date.isoformat()] = int(round(pred))

    return JSONResponse(
        content={
            "input_date": input_date.isoformat(),
            "predictions": {"comfort_climate": predictions},
        }
    )


@app.get("/predict/category/weather_hazard")
def predict_weather_hazard(
    date: str = Query(..., description="Reference date in YYYY-MM-DD format")
):
    """Predict the Weather Hazard Category for exactly 7 days after ``date``."""
    input_date = _parse_date(date)
    features = _build_daily_features(input_date)

    pred = whc_model.predict(features[WHC_FEATURES])[0]
    target_date = input_date + timedelta(days=7)

    return JSONResponse(
        content={
            "input_date": input_date.isoformat(),
            "predictions": {"weather_hazard": {target_date.isoformat(): str(pred)}},
        }
    )
