"""fastf1_client.py — lap-by-lap session data via the FastF1 package,
local-cached (FastF1's own on-disk cache, `config.FASTF1_CACHE_DIR` —
"highly recommended, not optional" per FastF1's own docs, 50-100MB/session
downloaded once then reused).

`build_lap_snapshots` is the training corpus for models/live_win_prob.py
(Phase 3's live in-race engine, requirement 3): one row per (lap, driver)
with the race state at that point, used to train "given this exact
mid-race state, what's P(win)/P(podium) from here" — the standard sports
win-probability-model pattern. Telemetry/lap data only goes back to
2018 — this is deliberately NOT the live-inference feed (that's
data/openf1.py, which has ~3s latency during an actually-live session);
this is the historical replay corpus the live model trains on.
"""

from __future__ import annotations

import fastf1
import numpy as np
import pandas as pd

from ..config import FASTF1_CACHE_DIR

fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))

_SC_TRACK_STATUS_CODES = {"4", "6", "7"}  # SC, VSC, VSC ending


def _safety_car_active(track_status) -> bool:
    if pd.isna(track_status):
        return False
    return any(c in str(track_status) for c in _SC_TRACK_STATUS_CODES)


def load_race_session(season: int, round_: int):
    """A loaded FastF1 Race session — laps + results, no telemetry/weather/
    messages (not needed for lap-state snapshots, and skipping them cuts
    load time substantially)."""
    session = fastf1.get_session(season, round_, "R")
    session.load(telemetry=False, weather=False, messages=False)
    return session


def build_lap_snapshots(
    season: int,
    round_: int,
    driver_code_to_id: dict[str, str],
    elo_pre_race: dict[str, float],
    team_strength_pre_race: dict[str, float],
    driver_to_constructor: dict[str, str],
) -> pd.DataFrame:
    """One row per (lap, driver) still classified as running that lap:
    lap_number, laps_remaining, current_position, gap_to_leader_s,
    gap_ahead_s, tyre_compound, tyre_age_laps, pace_delta_last_3_laps,
    pit_stops_so_far, pitted_this_lap, safety_car_active, grid_position,
    driver_pre_race_strength, constructor_strength — plus the label columns
    `won`/`podium` (that driver's EVENTUAL result, from this session's own
    final classification), so every lap-state row is labeled with the
    outcome it's trying to predict.

    `driver_code_to_id`/`elo_pre_race`/`team_strength_pre_race` are passed
    in rather than computed here so a multi-race corpus build (models/
    live_win_prob.py::build_training_corpus) can compute Elo/team-strength
    history ONCE across all seasons instead of once per race.
    """
    session = load_race_session(season, round_)
    laps = session.laps.copy()
    laps = laps[laps["LapNumber"].notna() & laps["Position"].notna()].copy()
    laps["LapNumber"] = laps["LapNumber"].astype(int)
    total_laps = int(laps["LapNumber"].max())

    laps["lap_seconds"] = laps["Time"].dt.total_seconds()
    laps["lap_time_s"] = laps["LapTime"].dt.total_seconds()

    leader_time = laps.groupby("LapNumber")["lap_seconds"].transform("min")
    laps["gap_to_leader_s"] = laps["lap_seconds"] - leader_time

    laps = laps.sort_values(["LapNumber", "Position"])
    laps["_time_ahead"] = laps.groupby("LapNumber")["lap_seconds"].shift(1)
    laps["gap_ahead_s"] = (laps["lap_seconds"] - laps["_time_ahead"]).fillna(0.0)

    laps = laps.sort_values(["Driver", "LapNumber"])
    laps["_driver_roll3"] = laps.groupby("Driver")["lap_time_s"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    field_roll3_median = laps.groupby("LapNumber")["_driver_roll3"].transform("median")
    laps["pace_delta_last_3_laps"] = laps["_driver_roll3"] - field_roll3_median

    laps["_pit_this_lap"] = laps["PitInTime"].notna() | laps["PitOutTime"].notna()
    laps["pit_stops_so_far"] = laps.groupby("Driver")["_pit_this_lap"].cumsum()
    laps["safety_car_active"] = laps["TrackStatus"].apply(_safety_car_active)

    grid_by_abbr = session.results.set_index("Abbreviation")["GridPosition"]
    final_pos_by_abbr = session.results.set_index("Abbreviation")["Position"]

    rows = []
    for _, lap in laps.iterrows():
        code = lap["Driver"]
        driver_id = driver_code_to_id.get(code)
        if driver_id is None:
            continue
        constructor_id = driver_to_constructor.get(driver_id)
        final_pos = final_pos_by_abbr.get(code)
        rows.append(
            {
                "season": season,
                "round": round_,
                "driver_id": driver_id,
                "lap_number": int(lap["LapNumber"]),
                "laps_remaining": total_laps - int(lap["LapNumber"]),
                "current_position": float(lap["Position"]),
                "gap_to_leader_s": float(lap["gap_to_leader_s"]) if pd.notna(lap["gap_to_leader_s"]) else np.nan,
                "gap_ahead_s": float(lap["gap_ahead_s"]) if pd.notna(lap["gap_ahead_s"]) else np.nan,
                "tyre_compound": lap.get("Compound"),
                "tyre_age_laps": float(lap["TyreLife"]) if pd.notna(lap.get("TyreLife")) else np.nan,
                "pace_delta_last_3_laps": (
                    float(lap["pace_delta_last_3_laps"]) if pd.notna(lap["pace_delta_last_3_laps"]) else np.nan
                ),
                "pit_stops_so_far": int(lap["pit_stops_so_far"]),
                "pitted_this_lap": bool(lap["_pit_this_lap"]),
                "safety_car_active": bool(lap["safety_car_active"]),
                "grid_position": float(grid_by_abbr.get(code)) if pd.notna(grid_by_abbr.get(code)) else np.nan,
                "driver_pre_race_strength": elo_pre_race.get(driver_id, np.nan),
                "constructor_strength": team_strength_pre_race.get(constructor_id, np.nan),
                "won": bool(pd.notna(final_pos) and int(final_pos) == 1),
                "podium": bool(pd.notna(final_pos) and int(final_pos) <= 3),
            }
        )
    return pd.DataFrame(rows)
