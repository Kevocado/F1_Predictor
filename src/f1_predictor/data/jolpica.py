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
import logging
import re
import time
from dataclasses import dataclass, field

import pandas as pd
import requests

logger = logging.getLogger(__name__)

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
    last_status: int | None = None
    for attempt in range(attempts):
        _rate_limit()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            last_err = exc
            if exc.response is not None:
                last_status = exc.response.status_code
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
    # The status has to survive into the message. This error is the `reason`
    # a caller logs when it gives up on a season, and "Failed to fetch <url>
    # after 3 attempts" is the same sentence for a 429 (come back later, the
    # quota is spent) as for a 404 (that path does not exist) as for a socket
    # timeout. A warning nobody can act on is close to no warning at all.
    status = f" (last HTTP status {last_status})" if last_status is not None else ""
    raise RuntimeError(f"Failed to fetch {url} after {attempts} attempts{status}") from last_err


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
            # `raceName` is the upstream event's own name, passed through
            # verbatim -- this project does not name races, and it has no
            # circuit->name table to consult. Do not "tidy" a name that looks
            # wrong; check the record's Circuit block first, and if the two
            # disagree, the circuit block is what locates the race.
            #
            # The 2026 calendar has one name that reads like a bug:
            #
            #     round 16   "Bahrain Grand Prix in Malaysia"
            #                circuitId sepang, Sepang International Circuit,
            #                locality Kuala Lumpur, country Malaysia
            #
            # It is the official name. The FIA announced on 26.07.26 that
            # Bahrain's 2026 round moved to Malaysia and "will become the
            # Formula 1 Gulf Air Bahrain Grand Prix in Malaysia", run at
            # Sepang on 2-4 October. It was reported as a country-mapping bug
            # and "corrected" to "Bahrain Grand Prix", which would have placed
            # the race at Sakhir. tests/test_round16_race_identity.py pins it.
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
        "fastest_lap_rank": int(res["FastestLap"]["rank"]) if "rank" in res.get("FastestLap", {}) else None,
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


#: How many prior seasons to pull for feature history. Circuit form is an
#: expanding mean per (driver, circuit), and a circuit is typically visited once
#: a season — so with only the current season loaded, EVERY future race has null
#: circuit features. That is not an edge case: the 2026 calendar visits no
#: circuit twice.
#:
#: Measured: with prior seasons merged, round 17 (Marina Bay, 40 races across
#: 2024-25) goes from 0/23 drivers having circuit history to 22/23. Round 16
#: stays 0/23 and that is correct — Sepang has no race in the loaded window.
HISTORY_SEASONS = 3


@dataclass(frozen=True)
class SkippedSeason:
    """A season the model asked for and did not get, and why.

    `reason` is required rather than defaulted: a skip with no reason on the
    payload is the same silence the log line used to be, moved somewhere less
    likely to be read. The two causes are different facts — a failed fetch is
    about this run (rate limit, connection reset, timeout) and an empty frame is
    about the calendar — and the string is the only place that says which.
    """
    season: int
    reason: str


@dataclass(frozen=True)
class HistoryCoverage:
    """What the history window actually contains, for whoever asked for it.

    The third element of `load_history`'s return. It exists because the two
    frames cannot express a short window: a window missing 2024 is the same two
    frames as a complete one, just smaller, and the caller cannot tell. Circuit
    form is an expanding mean over whatever arrived, so a skipped season did not
    fail the run — it made the model quietly less informed, and the forecast it
    produces looks identical to a normal one. Anything that serves those numbers
    has to be able to say so.

    `complete` is a derived property, never a value a caller sets: it is True iff
    nothing was skipped. It cannot drift from `missing_seasons` beside it, and
    there is no path that reports a complete window after dropping a season.
    """
    seasons_requested: tuple[int, ...]
    seasons_loaded: tuple[int, ...]
    missing_seasons: tuple[SkippedSeason, ...] = field(default_factory=tuple)

    @property
    def complete(self) -> bool:
        return not self.missing_seasons


def load_history(
    season: int, force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, HistoryCoverage]:
    """Results and schedule for `season` and the seasons before it, plus what
    of that window actually arrived.

    Both frames carry a `season` column, because `features.circuit.with_circuit`
    joins on `["season", "round"]` — a schedule without it would join on a
    column that does not exist and every circuit would come out null. That is
    the shape of the bug this function exists to fix, and it is why the frames
    are stamped here rather than at the call site.

    A season that cannot be loaded is skipped rather than raised, because one
    unavailable season should not cost a run the seasons that did load — and
    `_get` already retries a 429 with bounded backoff before it gives up, so a
    transient rate limit is normally absorbed here without a skip at all. But
    "skipped" is not "silent", and it is not only a log concern: every skip is
    recorded in the returned `HistoryCoverage`, which is what the API puts on the
    response. The WARNING below is for whoever is watching the container; the
    coverage is for whoever is reading the forecast, and the second audience is
    the larger one.
    """
    seasons = [season - i for i in range(HISTORY_SEASONS) if season - i > 1950]
    results, schedules, missing = [], [], []
    for s in seasons:
        try:
            df = load_season_results(s, force_refresh=force_refresh)
            sch = fetch_season_schedule(s, force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001 - one bad season must not sink the window
            # Not the same fact as "the upstream does not carry this season
            # yet": this is a statement about this run (rate limit, connection
            # reset, timeout), and it is the case that used to cost a season
            # with an INFO line nobody sees under default logging.
            reason = f"fetch failed: {type(exc).__name__}: {exc}"
            logger.warning("skipped %s history — %s", s, reason)
            missing.append(SkippedSeason(season=s, reason=reason))
            continue
        if df.empty or sch.empty:
            # This branch had no log statement at all, so a season could leave
            # the window without a trace. Say which frame came back empty: an
            # empty schedule and empty results are different upstream problems.
            empty = "results" if df.empty else "schedule"
            reason = f"upstream returned an empty {empty} frame"
            logger.warning("skipped %s history — %s", s, reason)
            missing.append(SkippedSeason(season=s, reason=reason))
            continue
        results.append(df.assign(season=s))
        schedules.append(sch.assign(season=s))
    coverage = HistoryCoverage(
        seasons_requested=tuple(seasons),
        seasons_loaded=tuple(s for s in seasons if s not in {m.season for m in missing}),
        missing_seasons=tuple(missing),
    )
    if not results:
        return pd.DataFrame(), pd.DataFrame(), coverage
    return pd.concat(results, ignore_index=True), pd.concat(schedules, ignore_index=True), coverage


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
