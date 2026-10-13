"""weekend_gate.py -- decides whether a frequent (race-weekend) snapshot
refresh has anything to do.

The weekend cron fires every ~20 minutes Fri-Sun. Almost all of those runs
should be a no-op: the expensive part (rebuilding predictions, championship
Monte Carlo, tracking reconcile) only matters once a *new session result* has
landed in Jolpica (qualifying, sprint, race), because that is what changes a
prediction's inputs (grid / quali_position) or reconciles a pick.

    python -m f1_predictor.weekend_gate check    # prints/sets refresh=true|false
    python -m f1_predictor.weekend_gate record   # after a successful build

`check` stores what it saw in PENDING_PATH; `record` promotes that to SEEN_PATH
(committed). Splitting the two means a failed build never marks a result as
handled. Cost of a no-op run: one cached schedule read + <=3 Jolpica requests
per round in the weekend window.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

from . import config
from .data import jolpica

SEEN_PATH = config.DATA_DIR / "session_results_seen.json"
PENDING_PATH = config.CACHE_DIR / "session_results_pending.json"  # gitignored
WINDOW = pd.Timedelta("4D")  # a round is "live" from 4d before to 4d after its race


def new_session_results(seen: set[str], available: set[str]) -> set[str]:
    """Pure: result keys ("season:round:type") present now but not yet handled."""
    return set(available) - set(seen)


def available_session_keys(season: int, now: pd.Timestamp | None = None) -> set[str]:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    schedule = jolpica.fetch_season_schedule(season)
    keys: set[str] = set()
    for _, row in schedule.iterrows():
        race_dt = row["race_datetime"]
        if pd.isna(race_dt) or not (now - WINDOW <= race_dt <= now + WINDOW):
            continue
        round_ = int(row["round"])
        fetchers = [("qualifying", jolpica.fetch_qualifying), ("race", jolpica.fetch_race_results)]
        if bool(row["is_sprint_weekend"]):
            fetchers.append(("sprint", jolpica.fetch_sprint_results))
        for name, fetch in fetchers:
            # force_refresh: a cached-empty result never exists (see
            # jolpica._cache_or_fetch), and a cached non-empty one is final.
            if not fetch(season, round_, force_refresh=True).empty:
                keys.add(f"{season}:{round_}:{name}")
    return keys


def _seen() -> set[str]:
    return set(json.loads(SEEN_PATH.read_text())) if SEEN_PATH.exists() else set()


def check(season: int) -> bool:
    available = available_session_keys(season)
    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(json.dumps(sorted(available)))
    new = new_session_results(_seen(), available)
    print(f"new session results: {sorted(new) or 'none'}")
    return bool(new)


def record() -> None:
    if PENDING_PATH.exists():
        SEEN_PATH.write_text(json.dumps(sorted(_seen() | set(json.loads(PENDING_PATH.read_text()))), indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        refresh = check(config.CURRENT_SEASON)
        print(f"refresh={'true' if refresh else 'false'}")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
                fh.write(f"refresh={'true' if refresh else 'false'}\n")
    elif sys.argv[1:] == ["record"]:
        record()
    else:
        sys.exit("usage: python -m f1_predictor.weekend_gate check|record")
