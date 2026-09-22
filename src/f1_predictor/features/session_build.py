"""session_build.py — training-frame builders for the three new session
predictors (sprint_qualifying, qualifying, sprint), sibling to
features/build.py's build_training_frame (which stays the race model's
own, unmodified builder). Reuses the exact same performance-history
feature computations build_training_frame does (Elo, team strength,
rolling form, circuit history, DNF rate, weather) — those are computed
from RACE results regardless of which session is being predicted, since
they're the underlying driver/constructor performance signal, not
session-specific.

Deliberately NOT tier_augmented (unlike build_training_frame): each of
these three session types has exactly one fixed feature-availability
point (models/session_outcome.py::SessionSpec.feature_cutoff_tier), so
there's only one training row per (season, round, driver) to build, not
one per tier.

Data-source note: jolpica has no sprint-qualifying results endpoint
(confirmed directly: GET /{season}/{round}/sprint-qualifying.json ->
400 Bad Request). Sprint qualifying's target comes from jolpica's
sprint.json `grid` column instead — a sprint's starting grid IS its
sprint-qualifying classification barring penalties, the same
caveat-tolerant proxy this codebase already accepts for the race model's
own `grid` feature (see api/routes.py::_future_feature_frame).
"""

from __future__ import annotations

import pandas as pd

from ..data import jolpica
from . import circuit, elo, rolling_form, safety_car, team_strength
from . import weather as weather_features

BASE_SESSION_FEATURES = [
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
    "temp_max_c",
    "precipitation_mm",
    "wind_max_kph",
]

SESSION_FEATURE_COLUMNS: dict[str, list[str]] = {
    "sprint_qualifying": list(BASE_SESSION_FEATURES),
    "qualifying": list(BASE_SESSION_FEATURES) + ["sprint_finish_position"],
    "sprint": list(BASE_SESSION_FEATURES) + ["sprint_quali_position"],
}

_VALID_SESSION_TYPES = set(SESSION_FEATURE_COLUMNS.keys())


def build_session_training_frame(
    session_type: str, seasons: list[int] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    if session_type not in _VALID_SESSION_TYPES:
        raise ValueError(
            f"build_session_training_frame does not support session_type={session_type!r} "
            f"(valid: {sorted(_VALID_SESSION_TYPES)}) — 'race' uses features/build.py directly"
        )
    seasons = seasons or jolpica.default_seasons()
    feature_cols = SESSION_FEATURE_COLUMNS[session_type]

    results_df = jolpica.load_multi_season_results(seasons)
    schedule_df = pd.concat([jolpica.fetch_season_schedule(s) for s in seasons], ignore_index=True)
    sprint_results_df = pd.concat([jolpica.load_season_sprints(s) for s in seasons], ignore_index=True)

    if session_type == "sprint_qualifying":
        if sprint_results_df.empty:
            target_df = pd.DataFrame(columns=["season", "round", "driver_id", "constructor_id", "sprint_quali_position"])
        else:
            target_df = sprint_results_df[["season", "round", "driver_id", "constructor_id", "grid"]].rename(
                columns={"grid": "sprint_quali_position"}
            )
    elif session_type == "qualifying":
        target_df = pd.concat([jolpica.load_season_qualifying(s) for s in seasons], ignore_index=True)
    else:  # "sprint"
        if sprint_results_df.empty:
            target_df = pd.DataFrame(
                columns=["season", "round", "driver_id", "constructor_id", "position", "dnf", "grid"]
            )
        else:
            target_df = sprint_results_df[
                ["season", "round", "driver_id", "constructor_id", "position", "dnf", "grid"]
            ].copy()

    if target_df.empty:
        return pd.DataFrame(columns=["season", "round", "driver_id", "constructor_id"] + feature_cols), feature_cols

    elo_hist = elo.compute_elo_history(results_df)
    team_hist = team_strength.compute_team_strength_history(results_df)
    team_form_hist = team_strength.compute_team_rolling_form(results_df)
    form_hist = rolling_form.compute_rolling_form(results_df)
    circuit_hist = circuit.compute_circuit_history(results_df, schedule_df)
    dnf_rate_hist = safety_car.compute_circuit_dnf_rate(results_df, schedule_df)
    weather_feat = weather_features.compute_weather_features(schedule_df)

    df = target_df.merge(elo_hist, on=["season", "round", "driver_id"], how="left")
    df = df.merge(team_hist, on=["season", "round", "constructor_id"], how="left")
    df = df.merge(team_form_hist, on=["season", "round", "constructor_id"], how="left")
    df = df.merge(form_hist, on=["season", "round", "driver_id"], how="left")
    df = df.merge(circuit_hist.drop(columns=["constructor_id"]), on=["season", "round", "driver_id"], how="left")
    df = df.merge(dnf_rate_hist.drop(columns=["circuit_id"]), on=["season", "round"], how="left")
    df = df.merge(weather_feat, on=["season", "round"], how="left")

    if session_type == "qualifying":
        if sprint_results_df.empty:
            df["sprint_finish_position"] = float("nan")
        else:
            sprint_feat = sprint_results_df[["season", "round", "driver_id", "position"]].rename(
                columns={"position": "sprint_finish_position"}
            )
            df = df.merge(sprint_feat, on=["season", "round", "driver_id"], how="left")
    elif session_type == "sprint":
        df["sprint_quali_position"] = df["grid"]

    return df, feature_cols
