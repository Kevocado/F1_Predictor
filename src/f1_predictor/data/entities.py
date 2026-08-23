"""entities.py — canonical driver/constructor reference and the
per-(season, round) driver->constructor mapping, resolved directly off race
results rather than assumed to be static for a season. A driver can change
constructors mid-season (a mid-season swap, a reserve-driver substitution) —
`features/elo.py` keeps driver and constructor ratings separate specifically
so a driver's rating survives a team change, and this module is what
supplies "which constructor was this driver actually racing for in this
specific race" to every downstream feature/model that needs it.
"""

from __future__ import annotations

import pandas as pd

from . import jolpica


def driver_constructor_map(results_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, round, driver_id) -> constructor_id, taken
    directly from that race's results. This is the ground truth for "who
    was this driver driving for" at any point in history — no separate
    entities table to keep in sync."""
    return results_df[["season", "round", "driver_id", "constructor_id"]].drop_duplicates()


def current_constructor(results_df: pd.DataFrame, driver_id: str) -> str | None:
    """The constructor a driver most recently raced for, per the results
    frame passed in (chronological order not required — this sorts)."""
    rows = results_df[results_df["driver_id"] == driver_id]
    if rows.empty:
        return None
    rows = rows.sort_values(["season", "round"])
    return rows.iloc[-1]["constructor_id"]


def driver_reference(results_df: pd.DataFrame) -> pd.DataFrame:
    """One row per driver_id seen in `results_df`, with their most recent
    constructor and driver_code — a lightweight lookup table for the API/
    frontend layer, not a separate source of truth."""
    cols = [c for c in ("driver_id", "driver_code") if c in results_df.columns]
    latest = results_df.sort_values(["season", "round"]).drop_duplicates("driver_id", keep="last")
    return latest[cols + ["constructor_id"]].reset_index(drop=True)


def grid_for_race(results_df: pd.DataFrame, season: int, round_: int) -> list[str]:
    """The list of driver_ids who actually started a given race."""
    race = results_df[(results_df["season"] == season) & (results_df["round"] == round_)]
    return race["driver_id"].tolist()


def upcoming_grid(schedule_row: pd.Series, driver_standings: pd.DataFrame) -> list[str]:
    """Best-effort grid for a race that hasn't happened yet: the drivers
    currently in the standings (i.e. who have started at least one race
    this season). Doesn't attempt to predict mid-season driver swaps beyond
    what's already reflected in `driver_standings` as of the most recent
    completed round — a deliberate scope boundary, not an oversight."""
    return driver_standings["driver_id"].tolist()
