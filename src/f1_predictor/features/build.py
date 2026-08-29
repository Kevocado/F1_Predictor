"""build.py — the single canonical (df, feature_cols) entry point: merges
race results with every features/*.py module's output into one training
frame, then tier-augments it (features/session_state.py) so
models/race_outcome.py and models/dnf.py train on exactly the shape
(including legitimately-missing columns) they'll see at prediction time.
"""

from __future__ import annotations

import pandas as pd

from ..data import jolpica
from . import circuit, elo, rolling_form, safety_car, session_state, team_strength
from . import weather as weather_features

FEATURE_COLUMNS = [
    "elo_pre_race",
    "team_strength_pre_race",
    "team_form_avg_position_3",
    "team_form_points_3",
    "form_avg_position_3",
    "form_avg_points_3",
    "form_dnf_rate_3",
    "form_avg_position_5",
    "form_avg_points_5",
    "form_dnf_rate_5",
    "form_avg_position_10",
    "form_avg_points_10",
    "form_dnf_rate_10",
    "driver_circuit_avg_position",
    "driver_circuit_avg_points",
    "constructor_circuit_avg_position",
    "circuit_dnf_rate",
    "grid",
    "quali_position",
    "quali_gap_to_pole",
    "temp_max_c",
    "precipitation_mm",
    "wind_max_kph",
]


def _parse_time_to_seconds(t) -> float | None:
    if t is None or (isinstance(t, float) and pd.isna(t)) or str(t).strip() == "":
        return None
    parts = str(t).split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def quali_features(quali_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["season", "round", "driver_id", "quali_position", "quali_gap_to_pole"]
    if quali_df.empty:
        return pd.DataFrame(columns=cols)
    df = quali_df.copy()
    df["best_time_s"] = df[["q1", "q2", "q3"]].map(_parse_time_to_seconds).min(axis=1)
    pole_time = df.groupby(["season", "round"])["best_time_s"].transform("min")
    df["quali_gap_to_pole"] = df["best_time_s"] - pole_time
    return df[cols]


def build_training_frame(
    seasons: list[int] | None = None,
    results_df: pd.DataFrame | None = None,
    schedule_df: pd.DataFrame | None = None,
    quali_df: pd.DataFrame | None = None,
    weather_force_refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Pass pre-fetched results_df/schedule_df/quali_df to avoid redundant
    refetching when called many times (e.g. from evaluate/walk_forward.py).
    Returns (df, feature_cols) — feature_cols excludes identifier/target
    columns, and df is tier-augmented (one real race becomes up to 3 rows,
    see features/session_state.py::tier_augment)."""
    seasons = seasons or jolpica.default_seasons()

    if results_df is None:
        results_df = jolpica.load_multi_season_results(seasons)
    if schedule_df is None:
        schedule_df = pd.concat([jolpica.fetch_season_schedule(s) for s in seasons], ignore_index=True)
    if quali_df is None:
        quali_df = pd.concat([jolpica.load_season_qualifying(s) for s in seasons], ignore_index=True)

    elo_hist = elo.compute_elo_history(results_df)
    team_hist = team_strength.compute_team_strength_history(results_df)
    team_form_hist = team_strength.compute_team_rolling_form(results_df)
    form_hist = rolling_form.compute_rolling_form(results_df)
    circuit_hist = circuit.compute_circuit_history(results_df, schedule_df)
    dnf_rate_hist = safety_car.compute_circuit_dnf_rate(results_df, schedule_df)
    quali_feat = quali_features(quali_df)
    weather_feat = weather_features.compute_weather_features(schedule_df, force_refresh=weather_force_refresh)

    df = results_df.merge(elo_hist, on=["season", "round", "driver_id"], how="left")
    df = df.merge(team_hist, on=["season", "round", "constructor_id"], how="left")
    df = df.merge(team_form_hist, on=["season", "round", "constructor_id"], how="left")
    df = df.merge(form_hist, on=["season", "round", "driver_id"], how="left")
    df = df.merge(circuit_hist.drop(columns=["constructor_id"]), on=["season", "round", "driver_id"], how="left")
    df = df.merge(dnf_rate_hist.drop(columns=["circuit_id"]), on=["season", "round"], how="left")
    df = df.merge(quali_feat, on=["season", "round", "driver_id"], how="left")
    df = df.merge(weather_feat, on=["season", "round"], how="left")

    df = session_state.tier_augment(df)

    return df, FEATURE_COLUMNS
