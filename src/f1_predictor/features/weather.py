"""weather.py — joins per-race weather onto the feature frame, sourced from
data/open_meteo.py (historical archive for past races, forecast for an
upcoming one within the ~16-day horizon). A future race outside the
forecast horizon simply carries NaN weather columns until it's close
enough — no special-casing needed, XGBoost's missing-value handling covers
it the same way it covers unreached session-state tiers.
"""

from __future__ import annotations

import pandas as pd

from ..data import open_meteo


def compute_weather_features(schedule_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    return open_meteo.weather_frame(schedule_df, force_refresh=force_refresh)
