"""jolpica.py — cache-or-fetch historical F1 results/qualifying/sprint/
schedule/standings via the jolpica-f1 API (Ergast-successor,
api.jolpi.ca/ergast/f1).

Mirrors PL_Predictor's data/football_data.py cache-or-fetch idiom: check
the cache file first, otherwise fetch (with retry + rate-limit backoff),
write to cache, return. One JSON file per (endpoint, season[, round]) call
under `config.JOLPICA_CACHE_DIR` — cheap, human-readable, and lets a
season's worth of per-round calls be replayed offline after the first run.
"""

from __future__ import annotations

import json
import re
import time

import pandas as pd
import requests

from ..config import (
    CURRENT_SEASON,
    JOLPICA_BASE_URL,
    JOLPICA_CACHE_DIR,
    JOLPICA_MIN_INTERVAL_S,
    JOLPICA_USER_AGENT,
)

_last_request_at = 0.0

# A still-classified finisher who's merely behind on laps shows up as
# EITHER a literal "Lapped" string OR "+N Lap(s)" depending on season —
# confirmed directly: 2019 data uses "+1 Lap"/"+2 Laps" style, 2024+ data
# uses the literal "Lapped" instead, for the exact same real-world
# situation. Anything other than these (or "Finished") is a genuine DNF —
# mechanical failure, accident, disqualification, did-not-start, etc.
_NOT_DNF_STATUSES = {"Finished", "Lapped"}
_LAPPED_COUNT_RE = re.compile(r"^\+\d+ Laps?$")


def is_dnf(status: str) -> bool:
    if status in _NOT_DNF_STATUSES:
        return False
    return not _LAPPED_COUNT_RE.match(status)


def _rate_limit() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < JOLPICA_MIN_INTERVAL_S:
        time.sleep(JOLPICA_MIN_INTERVAL_S - elapsed)
    _last_request_at = time.monotonic()


def _get(path: str, params: dict | None = None, attempts: int = 3, backoff: float = 2.0) -> dict:
    url = f"{JOLPICA_BASE_URL}/{path}"
    headers = {"User-Agent": JOLPICA_USER_AGENT}
    last_err: Exception | None = None
    for attempt in range(attempts):
        _rate_limit()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            last_err = exc
            # Confirmed live in production: a cold container (Render's disk
            # is ephemeral) refetching a full season's worth of history can
            # burn through jolpica's hourly quota. A 429 means "come back
            # later," not "retry immediately" — respect Retry-After when
            # given, and back off much longer than the generic case below
            # rather than giving up after a few seconds and surfacing a 500.
            if exc.response is not None and exc.response.status_code == 429 and attempt < attempts - 1:
                retry_after = exc.response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 30.0 * (attempt + 1)
                time.sleep(wait)
            elif attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - retry on any other transient fetch failure
            last_err = exc
            if attempt < attempts - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts") from last_err


def _is_not_yet_available(data: dict) -> bool:
    """True for a RaceTable-shaped response with no Races yet — e.g.
    results.json fetched while that round's race is still in progress.
    Confirmed directly: this is NOT immutable the way a finished round's
    data is (jolpica publishes it later, once the race ends), so it must
    never be written to the on-disk cache below — with no TTL/staleness
    check, a round cached empty mid-race would stay "empty" forever, even
    long after jolpica actually has the real results."""
    race_table = data.get("MRData", {}).get("RaceTable")
    return race_table is not None and not race_table.get("Races")


def _cache_or_fetch(cache_name: str, path: str, params: dict | None = None, force_refresh: bool = False) -> dict:
    cache_path = JOLPICA_CACHE_DIR / f"{cache_name}.json"
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text())
    data = _get(path, params=params)
    if _is_not_yet_available(data):
        return data
    cache_path.write_text(json.dumps(data))
    return data


def _combine_dt(date: str | None, time_str: str | None) -> pd.Timestamp | None:
    if not date:
        return None
    if not time_str:
        return pd.Timestamp(date, tz="UTC")
    return pd.Timestamp(f"{date}T{time_str}")


def fetch_season_schedule(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """One row per race weekend: round, circuit reference, and every
    session's datetime (fp1-3, sprint quali/sprint if it's a sprint
    weekend, qualifying, race) — the backbone `features/session_state.py`
    uses to know which tier a weekend is currently in."""
    data = _cache_or_fetch(f"{season}_schedule", f"{season}.json", {"limit": 40}, force_refresh)
    races = data["MRData"]["RaceTable"]["Races"]
    rows = []
    for r in races:
        row = {
            "season": season,
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "circuit_id": r["Circuit"]["circuitId"],
            "circuit_name": r["Circuit"]["circuitName"],
            "lat": float(r["Circuit"]["Location"]["lat"]),
            "long": float(r["Circuit"]["Location"]["long"]),
            "locality": r["Circuit"]["Location"]["locality"],
            "country": r["Circuit"]["Location"]["country"],
            "race_datetime": _combine_dt(r.get("date"), r.get("time")),
        }
        for key, prefix in [
            ("FirstPractice", "fp1"),
            ("SecondPractice", "fp2"),
            ("ThirdPractice", "fp3"),
            ("SprintQualifying", "sprint_quali"),
            ("Sprint", "sprint"),
            ("Qualifying", "qualifying"),
        ]:
            session = r.get(key)
            row[f"{prefix}_datetime"] = _combine_dt(session["date"], session.get("time")) if session else None
        row["is_sprint_weekend"] = r.get("Sprint") is not None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def _parse_result_row(res: dict, season: int, round_: int) -> dict:
    position = res["position"]
    return {
        "season": season,
        "round": round_,
        "driver_id": res["Driver"]["driverId"],
        "driver_code": res["Driver"].get("code"),
        "constructor_id": res["Constructor"]["constructorId"],
        "grid": int(res["grid"]),
        "position": int(position) if str(position).isdigit() else None,
        "position_text": res["positionText"],
        "points": float(res["points"]),
        "laps": int(res["laps"]),
        "status": res["status"],
        "dnf": is_dnf(res["status"]),
        "fastest_lap_rank": int(res["FastestLap"]["rank"]) if "FastestLap" in res else None,
    }


def fetch_race_results(season: int, round_: int, force_refresh: bool = False) -> pd.DataFrame:
    data = _cache_or_fetch(
        f"{season}_{round_}_results", f"{season}/{round_}/results.json", {"limit": 40}, force_refresh
    )
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return pd.DataFrame()
    return pd.DataFrame([_parse_result_row(res, season, round_) for res in races[0]["Results"]])


def fetch_sprint_results(season: int, round_: int, force_refresh: bool = False) -> pd.DataFrame:
    data = _cache_or_fetch(
        f"{season}_{round_}_sprint", f"{season}/{round_}/sprint.json", {"limit": 40}, force_refresh
    )
    races = data["MRData"]["RaceTable"]["Races"]
    if not races or "SprintResults" not in races[0]:
        return pd.DataFrame()
    return pd.DataFrame([_parse_result_row(res, season, round_) for res in races[0]["SprintResults"]])


def fetch_qualifying(season: int, round_: int, force_refresh: bool = False) -> pd.DataFrame:
    data = _cache_or_fetch(
        f"{season}_{round_}_qualifying", f"{season}/{round_}/qualifying.json", {"limit": 40}, force_refresh
    )
    races = data["MRData"]["RaceTable"]["Races"]
    if not races:
        return pd.DataFrame()
    rows = []
    for res in races[0]["QualifyingResults"]:
        rows.append(
            {
                "season": season,
                "round": round_,
                "driver_id": res["Driver"]["driverId"],
                "constructor_id": res["Constructor"]["constructorId"],
                "quali_position": int(res["position"]),
                "q1": res.get("Q1"),
                "q2": res.get("Q2"),
                "q3": res.get("Q3"),
            }
        )
    return pd.DataFrame(rows)


def fetch_driver_standings(season: int, round_: int | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Standings as of `round_` (or the final/current standings if None)."""
    suffix = f"{round_}/driverStandings" if round_ else "driverStandings"
    cache_name = f"{season}_{round_ or 'latest'}_driver_standings"
    data = _cache_or_fetch(cache_name, f"{season}/{suffix}.json", {"limit": 40}, force_refresh)
    lists = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not lists:
        return pd.DataFrame()
    rows = []
    for s in lists[0]["DriverStandings"]:
        rows.append(
            {
                "season": season,
                "driver_id": s["Driver"]["driverId"],
                "constructor_id": s["Constructors"][-1]["constructorId"],
                "position": int(s["position"]),
                "points": float(s["points"]),
                "wins": int(s["wins"]),
            }
        )
    return pd.DataFrame(rows)


def fetch_constructor_standings(season: int, round_: int | None = None, force_refresh: bool = False) -> pd.DataFrame:
    suffix = f"{round_}/constructorStandings" if round_ else "constructorStandings"
    cache_name = f"{season}_{round_ or 'latest'}_constructor_standings"
    data = _cache_or_fetch(cache_name, f"{season}/{suffix}.json", {"limit": 40}, force_refresh)
    lists = data["MRData"]["StandingsTable"]["StandingsLists"]
    if not lists:
        return pd.DataFrame()
    rows = []
    for s in lists[0]["ConstructorStandings"]:
        rows.append(
            {
                "season": season,
                "constructor_id": s["Constructor"]["constructorId"],
                "position": int(s["position"]),
                "points": float(s["points"]),
                "wins": int(s["wins"]),
            }
        )
    return pd.DataFrame(rows)


def _completed(schedule: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    return schedule[schedule[dt_col].notna() & (schedule[dt_col] <= now)]


def load_season_results(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Concatenate every completed round's race results for one season."""
    schedule = fetch_season_schedule(season, force_refresh=force_refresh)
    frames = []
    for _, race in _completed(schedule, "race_datetime").iterrows():
        df = fetch_race_results(season, int(race["round"]), force_refresh=force_refresh)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_season_qualifying(season: int, force_refresh: bool = False) -> pd.DataFrame:
    schedule = fetch_season_schedule(season, force_refresh=force_refresh)
    frames = []
    for _, race in _completed(schedule, "qualifying_datetime").iterrows():
        df = fetch_qualifying(season, int(race["round"]), force_refresh=force_refresh)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_season_sprints(season: int, force_refresh: bool = False) -> pd.DataFrame:
    schedule = fetch_season_schedule(season, force_refresh=force_refresh)
    sprint_weekends = _completed(schedule, "race_datetime")
    sprint_weekends = sprint_weekends[sprint_weekends["is_sprint_weekend"]]
    frames = []
    for _, race in sprint_weekends.iterrows():
        df = fetch_sprint_results(season, int(race["round"]), force_refresh=force_refresh)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_multi_season_results(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    frames = [load_season_results(s, force_refresh=force_refresh) for s in seasons]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def default_seasons(n: int = 8, include_current: bool = True) -> list[int]:
    """The `n` most recent seasons, oldest first. Includes the in-progress
    current season by default — unlike PL_Predictor's football-data.co.uk
    equivalent, jolpica has no publishing lag: a completed round's results
    are available immediately, so there's no need for a separate fast
    current-season supplement the way pulselive.py exists for PL_Predictor."""
    latest = CURRENT_SEASON if include_current else CURRENT_SEASON - 1
    return list(range(latest - n + 1, latest + 1))
