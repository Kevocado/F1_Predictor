"""public_snapshot.py — precomputes races/predictions/championship/session-
predictions for the current season so the public deployment never has to
run this project's live feature-building pipeline (Elo/team-strength
replay, ranker predict, championship Monte Carlo sim) on an actual
request. Same pattern as PL_Predictor's own public_snapshot.py.

Also does the actual "honest track record" job requirement 4's tracking
store was built for but never had a caller: before each not-yet-happened
session, records its current live prediction into tracking/store.py
(idempotent — safe to call every run, only fills in newly-reached tiers);
after a session's real result lands, reconciles every prediction of that
type against it. Without this, tracking.db accumulates nothing and Track
Record stays permanently empty — confirmed directly while building the
session predictors: no code anywhere else in this project ever called
store.record_session_predictions/reconcile_session_predictions.

Run this locally, or from .github/workflows/refresh-public-snapshot.yml:

    python -m f1_predictor.public_snapshot

Then commit + push data/public_snapshot.json AND data/tracking.db (both
force-added past .gitignore, same as the model artifacts) -- the running
deployment picks up the new public_snapshot.json within
PUBLIC_SNAPSHOT_POLL_SECONDS via api/routes.py::refresh_public_snapshot_from_remote,
no redeploy needed. tracking.db has no equivalent live-poll mechanism (SQLite,
not a small JSON blob) -- it only updates on the next deploy, which is an
accepted gap: it means Track Record can lag by up to a deploy cycle, not
that it's permanently broken. See config.py's PUBLIC_SNAPSHOT_PATH /
TRACKING_DB_PATH for the rest of both mechanisms.
"""

from __future__ import annotations

import json

import pandas as pd
from fastapi.encoders import jsonable_encoder

from . import config
from .api import routes
from .data import jolpica
from .tracking import store

# How many rounds past the current one get freshly rebuilt every run.
# Everything else reuses the previous snapshot verbatim (if that round was
# already in it) -- a race weekend is ~2 weeks apart, so unlike NFL/CFB's
# weekly cadence there's less pressure to rebuild the whole season every
# time, but a completed race's classification can still arrive late.
REBUILD_ROUNDS_AHEAD = 2
REBUILD_ROUNDS_BEHIND = 1

_ALL_SESSION_TYPES = ("sprint_qualifying", "sprint", "qualifying")

_SESSION_DATETIME_COL = {
    "sprint_qualifying": "sprint_quali_datetime",
    "sprint": "sprint_datetime",
    "qualifying": "qualifying_datetime",
    "race": "race_datetime",
}


def _current_round(races: list) -> int:
    unfinished = [r["round"] for r in races if not r["completed"]]
    if unfinished:
        return min(unfinished)
    return max((r["round"] for r in races), default=1)


def _session_types_for(race: dict) -> list[str]:
    return list(_ALL_SESSION_TYPES) if race["is_sprint_weekend"] else ["qualifying"]


def build_snapshot(previous: dict | None = None, season: int | None = None) -> dict:
    season = season or config.CURRENT_SEASON
    races = jsonable_encoder(routes._list_races_live(season))
    current_round = _current_round(races)

    previous = previous or {}
    same_season = previous.get("season") == season
    previous_predictions = previous.get("predictions", {}) if same_season else {}
    previous_session_predictions = previous.get("session_predictions", {}) if same_season else {}

    rebuild_from = current_round - REBUILD_ROUNDS_BEHIND
    rebuild_to = current_round + REBUILD_ROUNDS_AHEAD

    print(f"Season {season}, current round {current_round}; rebuilding {rebuild_from}-{rebuild_to}...")
    predictions: dict[str, dict] = {}
    session_predictions: dict[str, dict[str, dict]] = {t: {} for t in _ALL_SESSION_TYPES}
    for race in races:
        round_ = race["round"]
        key = str(round_)
        rebuild = rebuild_from <= round_ <= rebuild_to or key not in previous_predictions
        if rebuild:
            print(f"  round {round_} ({race['race_name']})")
            try:
                predictions[key] = jsonable_encoder(routes._get_race_prediction_live(season, round_))
            except Exception as exc:
                print(f"    ! skipped race prediction for round {round_}: {exc}")
        else:
            predictions[key] = previous_predictions[key]

        for session_type in _session_types_for(race):
            if rebuild or key not in previous_session_predictions.get(session_type, {}):
                try:
                    session_predictions[session_type][key] = jsonable_encoder(
                        routes._get_session_prediction_live(season, round_, session_type)
                    )
                except Exception as exc:
                    print(f"    ! skipped {session_type} prediction for round {round_}: {exc}")
            elif key in previous_session_predictions.get(session_type, {}):
                session_predictions[session_type][key] = previous_session_predictions[session_type][key]

    print("Building championship projections...")
    championship = {}
    for kind in ("drivers", "constructors"):
        try:
            championship[kind] = jsonable_encoder(routes._get_championship_live(kind, season, n_trials=2000))
        except Exception as exc:
            print(f"    ! skipped {kind} championship: {exc}")

    print("Updating track record (snapshot upcoming, reconcile completed)...")
    try:
        _snapshot_upcoming_predictions(season)
    except Exception as exc:
        print(f"    ! snapshot pass failed: {exc}")
    try:
        _reconcile_predictions(season)
    except Exception as exc:
        print(f"    ! reconcile pass failed: {exc}")

    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "season": season,
        "current_round": current_round,
        "races": races,
        "predictions": predictions,
        "session_predictions": session_predictions,
        "championship": championship,
    }


def _snapshot_upcoming_predictions(season: int) -> None:
    """Records an honest before-the-fact snapshot of every not-yet-happened
    session's CURRENT live prediction (whatever tier it's naturally at right
    now). Idempotent: record_session_predictions only inserts genuinely-new
    (season, round, session_type, driver_id, tier, market) rows, so running
    this every refresh cycle just fills in whatever tier has newly been
    reached since the last run -- it never overwrites or duplicates an
    earlier tier's already-recorded snapshot. A session that already
    happened is deliberately skipped here (nothing "before the fact" left
    to snapshot for it -- see _reconcile_predictions for the other half)."""
    schedule = jolpica.fetch_season_schedule(season)
    now = pd.Timestamp.now(tz="UTC")

    for _, race_row in schedule.iterrows():
        round_ = int(race_row["round"])
        session_types = list(_ALL_SESSION_TYPES) + ["race"] if bool(race_row["is_sprint_weekend"]) else ["qualifying", "race"]

        for session_type in session_types:
            session_dt = race_row.get(_SESSION_DATETIME_COL[session_type])
            if pd.isna(session_dt) or session_dt <= now:
                continue  # already happened (or unknown) -- reconciled instead, not snapshotted now

            try:
                if session_type == "race":
                    completed = routes._race_is_completed(season, round_, race_row["race_datetime"], now)
                    sim, tier, source = routes._race_prediction_bundle(season, round_, race_row, completed)
                else:
                    completed = routes._session_is_completed(season, round_, session_type, race_row, now)
                    sim, tier, source = routes._session_prediction_bundle(season, round_, race_row, session_type, completed)
                if source != "live":
                    continue  # already resolved/tracked -- nothing new to snapshot
                store.record_session_predictions(
                    sim, season, round_, race_row["race_name"], session_type, tier,
                    session_time=session_dt.isoformat(),
                )
            except Exception as exc:
                print(f"    ! snapshot skipped for round {round_} {session_type}: {exc}")


def _reconcile_predictions(season: int) -> None:
    """Fills in actual outcomes for every already-recorded, still-unresolved
    snapshot whose session has now produced a real result."""
    results_df = jolpica.load_season_results(season)
    if not results_df.empty:
        store.reconcile_session_predictions(results_df, "race")

    quali_df = jolpica.load_season_qualifying(season)
    if not quali_df.empty:
        store.reconcile_session_predictions(quali_df.rename(columns={"quali_position": "position"}), "qualifying")

    sprint_df = jolpica.load_season_sprints(season)
    if not sprint_df.empty:
        store.reconcile_session_predictions(sprint_df, "sprint")
        # sprint_qualifying's real classification is that same sprint's
        # starting grid (jolpica has no dedicated sprint-qualifying results
        # endpoint -- see features/session_build.py's docstring for the
        # same proxy used to build its training target).
        store.reconcile_session_predictions(sprint_df.rename(columns={"grid": "position"}), "sprint_qualifying")


def main() -> None:
    previous = json.loads(config.PUBLIC_SNAPSHOT_PATH.read_text()) if config.PUBLIC_SNAPSHOT_PATH.exists() else None
    snapshot = jsonable_encoder(build_snapshot(previous))
    config.PUBLIC_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {config.PUBLIC_SNAPSHOT_PATH} ({config.PUBLIC_SNAPSHOT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
