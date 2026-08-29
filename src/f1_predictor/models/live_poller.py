"""live_poller.py — the background live-polling loop for requirement 3:
checks OpenF1's `/sessions?session_key=latest` on an interval, and while a
Race session is actually in progress (and jolpica has that race's grid
from qualifying, but not yet a final classification), builds the live
feature frame (data/openf1.py::build_live_state_frame) and serves
predict_live()'s output from an in-process cache.

Started from api/main.py's FastAPI lifespan hook. See
docs/live_engine_design.md for the full design — including what's still
genuinely untested here: this has been exercised against OpenF1's
real historical data (data/openf1.py's own tests) and against FastF1
replay (evaluate/live_replay.py), but not yet against an actual live
session, since none has occurred during this build.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd

from ..config import OPENF1_LIVE_POLL_INTERVAL_S
from ..data import jolpica, openf1
from ..features import elo, team_strength
from . import live_win_prob

_cache: dict = {"live": False}
_models: tuple | None = None


def get_cached() -> dict:
    return _cache


def _load_models_cached() -> tuple:
    global _models
    if _models is None:
        _models = live_win_prob.load_models()
    return _models


def _resolve_live_context(session: dict) -> dict | None:
    """Matches an OpenF1 Race session to its jolpica season/round and
    builds everything build_live_state_frame needs. Returns None if no
    matching race is found, qualifying hasn't happened yet (the live
    engine needs grid position — before that, the pre-race tiers in
    features/session_state.py already cover it), or the race already has
    an official jolpica classification (no longer live even if OpenF1
    still reports it as "latest")."""
    season = session["year"]
    schedule = jolpica.fetch_season_schedule(season)
    session_date = pd.Timestamp(session["date_start"]).date()
    match = schedule[schedule["race_datetime"].notna() & (schedule["race_datetime"].dt.date == session_date)]
    if match.empty:
        return None
    race_row = match.iloc[0]
    round_ = int(race_row["round"])

    if not jolpica.fetch_race_results(season, round_).empty:
        return None

    quali_df = jolpica.fetch_qualifying(season, round_)
    if quali_df.empty:
        return None

    prior_results = jolpica.load_season_results(season)
    code_map: dict[str, str] = {}
    driver_to_constructor: dict[str, str] = {}
    if not prior_results.empty:
        code_map = dict(zip(prior_results["driver_code"], prior_results["driver_id"]))
        driver_to_constructor = dict(zip(prior_results["driver_id"], prior_results["constructor_id"]))
    for _, r in quali_df.iterrows():
        driver_to_constructor.setdefault(r["driver_id"], r["constructor_id"])

    grid_positions = dict(zip(quali_df["driver_id"], quali_df["quali_position"]))

    results_history = jolpica.load_multi_season_results(live_win_prob.default_live_seasons())
    prior = results_history[
        (results_history["season"] < season)
        | ((results_history["season"] == season) & (results_history["round"] < round_))
    ]
    elo_ratings = elo.latest_ratings(prior)
    team_ratings = team_strength.latest_ratings(prior)

    return {
        "season": season,
        "round": round_,
        "race_name": race_row["race_name"],
        "circuit_id": race_row["circuit_id"],
        "code_map": code_map,
        "driver_to_constructor": driver_to_constructor,
        "elo_ratings": elo_ratings,
        "team_ratings": team_ratings,
        "grid_positions": grid_positions,
    }


def poll_once() -> dict:
    """One tick: resolve whether a race is genuinely live, and if so,
    rebuild the cached prediction. Safe to call synchronously (e.g. from a
    script) — run_poller wraps it for the async background loop."""
    global _cache

    session = openf1.fetch_latest_session()
    if session is None or session.get("session_type") != "Race" or not openf1.is_session_live(session):
        _cache = {"live": False}
        return _cache

    context = _resolve_live_context(session)
    if context is None:
        _cache = {"live": False}
        return _cache

    state_df = openf1.build_live_state_frame(
        session["session_key"],
        context["circuit_id"],
        context["code_map"],
        context["driver_to_constructor"],
        context["elo_ratings"],
        context["team_ratings"],
        context["grid_positions"],
    )
    if state_df.empty:
        _cache = {"live": False}
        return _cache

    win_model, podium_model = _load_models_cached()
    preds = live_win_prob.predict_live(win_model, podium_model, state_df)
    preds["constructor_id"] = preds["driver_id"].map(context["driver_to_constructor"])

    _cache = {
        "live": True,
        "season": context["season"],
        "round": context["round"],
        "race_name": context["race_name"],
        "lap_number": int(state_df["lap_number"].max()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": preds.to_dict(orient="records"),
    }
    return _cache


async def run_poller(interval_s: float = OPENF1_LIVE_POLL_INTERVAL_S) -> None:
    """3 OpenF1 endpoint calls per tick (positions/intervals/laps, plus
    stints/race-control) at `interval_s` ≈ 18s puts this well under
    OpenF1's 30 req/min budget even with retries. One bad tick (a
    transient network error, a mid-race data gap) shouldn't kill the loop
    — logged and retried next interval."""
    while True:
        try:
            await asyncio.to_thread(poll_once)
        except Exception as exc:  # noqa: BLE001 - see docstring
            print(f"[live_poller] tick failed: {exc}")
        await asyncio.sleep(interval_s)
