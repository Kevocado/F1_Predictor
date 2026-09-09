"""public_snapshot.py — precomputes races/predictions/championship for the
current season so the public deployment never has to run this project's
live feature-building pipeline (Elo/team-strength replay, ranker predict,
championship Monte Carlo sim) on an actual request. Same pattern as
PL_Predictor's own public_snapshot.py.

Run this locally, or from .github/workflows/refresh-public-snapshot.yml:

    python -m f1_predictor.public_snapshot

Then commit + push data/public_snapshot.json -- the running deployment
picks it up within PUBLIC_SNAPSHOT_POLL_SECONDS via
api/routes.py::refresh_public_snapshot_from_remote, no redeploy needed.
See config.py's PUBLIC_SNAPSHOT_PATH for the rest of that mechanism.
"""

from __future__ import annotations

import json

import pandas as pd
from fastapi.encoders import jsonable_encoder

from . import config
from .api import routes

# How many rounds past the current one get freshly rebuilt every run.
# Everything else reuses the previous snapshot verbatim (if that round was
# already in it) -- a race weekend is ~2 weeks apart, so unlike NFL/CFB's
# weekly cadence there's less pressure to rebuild the whole season every
# time, but a completed race's classification can still arrive late.
REBUILD_ROUNDS_AHEAD = 2
REBUILD_ROUNDS_BEHIND = 1


def _current_round(races: list) -> int:
    unfinished = [r["round"] for r in races if not r["completed"]]
    if unfinished:
        return min(unfinished)
    return max((r["round"] for r in races), default=1)


def build_snapshot(previous: dict | None = None, season: int | None = None) -> dict:
    season = season or config.CURRENT_SEASON
    races = jsonable_encoder(routes._list_races_live(season))
    current_round = _current_round(races)

    previous = previous or {}
    previous_predictions = previous.get("predictions", {}) if previous.get("season") == season else {}

    rebuild_from = current_round - REBUILD_ROUNDS_BEHIND
    rebuild_to = current_round + REBUILD_ROUNDS_AHEAD

    print(f"Season {season}, current round {current_round}; rebuilding {rebuild_from}-{rebuild_to}...")
    predictions: dict[str, dict] = {}
    for race in races:
        round_ = race["round"]
        key = str(round_)
        if rebuild_from <= round_ <= rebuild_to or key not in previous_predictions:
            print(f"  round {round_} ({race['race_name']})")
            try:
                predictions[key] = jsonable_encoder(routes._get_race_prediction_live(season, round_))
            except Exception as exc:
                print(f"    ! skipped prediction for round {round_}: {exc}")
        else:
            predictions[key] = previous_predictions[key]

    print("Building championship projections...")
    championship = {}
    for kind in ("drivers", "constructors"):
        try:
            championship[kind] = jsonable_encoder(routes._get_championship_live(kind, season, n_trials=2000))
        except Exception as exc:
            print(f"    ! skipped {kind} championship: {exc}")

    return {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "season": season,
        "current_round": current_round,
        "races": races,
        "predictions": predictions,
        "championship": championship,
    }


def main() -> None:
    previous = json.loads(config.PUBLIC_SNAPSHOT_PATH.read_text()) if config.PUBLIC_SNAPSHOT_PATH.exists() else None
    snapshot = jsonable_encoder(build_snapshot(previous))
    config.PUBLIC_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"Wrote {config.PUBLIC_SNAPSHOT_PATH} ({config.PUBLIC_SNAPSHOT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
