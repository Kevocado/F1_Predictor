"""open_meteo.py — cache-or-fetch weather for a circuit/date, no key
required. Two Open-Meteo endpoints depending on whether the date is in the
past (historical training data) or the near future (pre-race forecast for
an upcoming weekend, only available up to ~16 days out):

- `archive-api.open-meteo.com/v1/archive` — any past date, used to build
  historical training features.
- `api.open-meteo.com/v1/forecast` — forecast for dates from today up to
  ~16 days ahead, used for an upcoming race's PRE_WEEKEND-tier weather
  feature (see features/session_state.py).

A race further out than the forecast horizon simply gets NaN weather
features until it's close enough — XGBoost's native missing-value handling
covers this the same way it covers unreached session-state tiers.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import requests

from ..config import OPEN_METEO_BASE_URL, OPEN_METEO_CACHE_DIR

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_DAILY_VARS = "temperature_2m_max,precipitation_sum,windspeed_10m_max"
_FORECAST_HORIZON_DAYS = 16


def _cache_key(lat: float, long: float, race_date: str) -> str:
    return f"{race_date}_{lat:.2f}_{long:.2f}"


def _fetch(url: str, params: dict) -> dict | None:
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001 - weather is a nice-to-have feature, never fail the pipeline over it
        return None


def fetch_weather(lat: float, long: float, race_date: str, force_refresh: bool = False) -> dict | None:
    """Daily weather summary for one circuit/date: {temp_max_c,
    precipitation_mm, wind_max_kph}, or None if unavailable (too far in the
    future, or a transient API failure)."""
    cache_path = OPEN_METEO_CACHE_DIR / f"{_cache_key(lat, long, race_date)}.json"
    if cache_path.exists() and not force_refresh:
        cached = json.loads(cache_path.read_text())
        return cached if cached else None

    target = date.fromisoformat(race_date)
    today = date.today()
    if target <= today:
        params = {
            "latitude": lat,
            "longitude": long,
            "start_date": race_date,
            "end_date": race_date,
            "daily": _DAILY_VARS,
            "timezone": "UTC",
        }
        data = _fetch(_ARCHIVE_URL, params)
    elif target <= today + timedelta(days=_FORECAST_HORIZON_DAYS):
        params = {
            "latitude": lat,
            "longitude": long,
            "start_date": race_date,
            "end_date": race_date,
            "daily": _DAILY_VARS,
            "timezone": "UTC",
        }
        data = _fetch(OPEN_METEO_BASE_URL, params)
    else:
        data = None

    result = None
    if data and data.get("daily", {}).get("time"):
        daily = data["daily"]
        result = {
            "temp_max_c": daily["temperature_2m_max"][0],
            "precipitation_mm": daily["precipitation_sum"][0],
            "wind_max_kph": daily["windspeed_10m_max"][0],
        }

    cache_path.write_text(json.dumps(result) if result else "null")
    return result


def weather_frame(schedule_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """One row per (season, round) with weather columns, built from a
    schedule frame's lat/long/race_datetime."""
    rows = []
    for _, race in schedule_df.iterrows():
        race_date = race["race_datetime"].date().isoformat() if pd.notna(race["race_datetime"]) else None
        weather = fetch_weather(race["lat"], race["long"], race_date, force_refresh) if race_date else None
        row = {"season": race["season"], "round": race["round"]}
        row.update(weather or {"temp_max_c": None, "precipitation_mm": None, "wind_max_kph": None})
        rows.append(row)
    return pd.DataFrame(rows)
