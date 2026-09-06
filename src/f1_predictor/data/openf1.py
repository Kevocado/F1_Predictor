"""openf1.py — live in-weekend data via the OpenF1 API (openf1.org), free,
no key, ~3s latency during an actually-live session. This is the LIVE
inference feed for models/live_win_prob.py (Phase 3's live in-race
engine) — the training corpus is FastF1 historical replay
(data/fastf1_client.py), since OpenF1 only covers 2023+.

Verified gotcha (confirmed directly against the real API, not assumed from
docs): the `limit` query param breaks `/position`, `/intervals`, and
`/race_control` — they return `{"detail": "No results found."}` with it
set, and real data without it. Every fetcher here deliberately omits
`limit` and filters client-side instead.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from ..config import OPENF1_BASE_URL, OPENF1_MIN_INTERVAL_S

_last_request_at = 0.0

# OpenF1 race-control message categories/flags that mean the track isn't at
# green-flag racing conditions — mirrors the SC/VSC/VSC-ending vocabulary
# data/fastf1_client.py's TrackStatus codes {4,6,7} capture for the
# training corpus, but OpenF1 uses free-text category/flag fields instead
# of numeric codes, so this is a second, separate vocabulary reconciled
# onto the same `safety_car_active` boolean the model was trained on.
_SC_FLAGS = {"YELLOW", "DOUBLE YELLOW", "SAFETY CAR", "VIRTUAL SAFETY CAR", "RED"}


def _safe_float(value) -> float:
    """OpenF1's `gap_to_leader`/`interval` are numeric seconds most of the
    time, but a lapped driver gets a literal string like "+1 LAP" instead
    (confirmed directly against real live data) — not a gap in seconds at
    all. Treated as NaN rather than a parse error; XGBoost's missing-value
    handling covers it the same way an unavailable stat does elsewhere."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _rate_limit() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < OPENF1_MIN_INTERVAL_S:
        time.sleep(OPENF1_MIN_INTERVAL_S - elapsed)
    _last_request_at = time.monotonic()


def _get(path: str, params: dict | None = None, attempts: int = 2, timeout: float = 8.0) -> list[dict]:
    url = f"{OPENF1_BASE_URL}/{path}"
    last_err: Exception | None = None
    for attempt in range(attempts):
        _rate_limit()
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as exc:  # noqa: BLE001 - a live poll tick failing shouldn't crash the poller
            last_err = exc
            if attempt < attempts - 1:
                time.sleep(1.0)
    raise RuntimeError(f"Failed to fetch {url}: {last_err}") from last_err


def fetch_latest_session() -> dict | None:
    """The currently-live (or most recently completed) session, or None if
    OpenF1 genuinely has none to report. `GET /v1/sessions?session_key=latest`
    — confirmed live against the real API. Deliberately does NOT swallow a
    RuntimeError into None here: OpenF1 has since added a paywall that
    blocks ALL unauthenticated access (not just live-session endpoints,
    confirmed directly — even historical queries 401 the same way) for the
    entire duration of a live session, with a 401 pointing at a paid-key
    signup page. That failure means "we got blocked," not "no race is
    live" — collapsing the two would tell the user the opposite of what's
    actually happening. Let it propagate; the caller decides what to do."""
    sessions = _get("sessions", {"session_key": "latest"})
    return sessions[0] if sessions else None


def is_session_live(session: dict, now: pd.Timestamp | None = None) -> bool:
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    start = pd.Timestamp(session["date_start"])
    end = pd.Timestamp(session["date_end"])
    return start <= now <= end


def fetch_drivers(session_key: int) -> dict[int, str]:
    """driver_number -> name_acronym (FastF1's 3-letter code), the bridge
    back to jolpica's driver_id via the same driver_code_to_id map
    data/fastf1_client.py already builds from jolpica results."""
    drivers = _get("drivers", {"session_key": session_key})
    return {d["driver_number"]: d["name_acronym"] for d in drivers}


def fetch_positions(session_key: int) -> pd.DataFrame:
    """Latest position per driver_number. `/position` returns the full
    history — deliberately no `limit` param (see module docstring); this
    takes the last row per driver client-side instead."""
    rows = _get("position", {"session_key": session_key})
    if not rows:
        return pd.DataFrame(columns=["driver_number", "position", "date"])
    df = pd.DataFrame(rows)
    return df.sort_values("date").groupby("driver_number").tail(1).reset_index(drop=True)


def fetch_intervals(session_key: int) -> pd.DataFrame:
    """Latest gap_to_leader/interval (gap ahead) per driver_number."""
    rows = _get("intervals", {"session_key": session_key})
    if not rows:
        return pd.DataFrame(columns=["driver_number", "gap_to_leader", "interval", "date"])
    df = pd.DataFrame(rows)
    return df.sort_values("date").groupby("driver_number").tail(1).reset_index(drop=True)


def fetch_laps(session_key: int) -> pd.DataFrame:
    """Every lap logged so far this session, all drivers — used to derive
    current lap number and pit-stop counts, same shape as the training
    corpus's `pit_stops_so_far`."""
    rows = _get("laps", {"session_key": session_key})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["driver_number", "lap_number"])


def fetch_race_control_flags(session_key: int, since: pd.Timestamp | None = None) -> list[dict]:
    """Race-control messages, optionally filtered to those after `since`
    (server-side via a `date>=` filter, since `limit` doesn't work here)."""
    params: dict = {"session_key": session_key}
    if since is not None:
        params["date"] = f">={since.isoformat()}"
    return _get("race_control", params)


def safety_car_active_from_messages(messages: list[dict]) -> bool:
    """True if the most recent flag-category message indicates anything
    other than green-flag racing. Mirrors data/fastf1_client.py's
    `_safety_car_active` (TrackStatus codes {4,6,7}) but reconciled from
    OpenF1's free-text vocabulary instead — see module docstring."""
    flag_messages = [m for m in messages if m.get("category") == "Flag" and m.get("flag")]
    if not flag_messages:
        return False
    latest = max(flag_messages, key=lambda m: m["date"])
    return latest["flag"].upper() in _SC_FLAGS


def fetch_stints(session_key: int) -> pd.DataFrame:
    """Tyre stint history — `compound`/`tyre_age_at_start` per stint, the
    live counterpart to FastF1's `Compound`/`TyreLife` columns. Also the
    only reliable live pit-stop count: `/pit` returned no data on manual
    testing, but each new stint IS a pit stop by definition, so
    `len(stints) - 1` gives the same count `data/fastf1_client.py` derives
    from `PitInTime`/`PitOutTime` in the training corpus."""
    rows = _get("stints", {"session_key": session_key})
    cols = ["driver_number", "stint_number", "lap_start", "lap_end", "compound", "tyre_age_at_start"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


# Standard race distance per circuit (public, stable F1 knowledge) — used
# to compute `laps_remaining` live, since neither OpenF1's `/sessions` nor
# jolpica's schedule expose a race's planned lap count before it finishes.
# A circuit missing from this table (a new venue, or a fictional-calendar
# entry) falls back to NaN rather than a guessed number — XGBoost's native
# missing-value handling covers it the same way an unreached session-state
# tier does elsewhere in this project.
CIRCUIT_LAP_COUNTS: dict[str, int] = {
    "albert_park": 58, "shanghai": 56, "suzuka": 53, "miami": 57, "villeneuve": 70,
    "monaco": 78, "catalunya": 66, "red_bull_ring": 71, "silverstone": 52, "spa": 44,
    "hungaroring": 70, "zandvoort": 72, "monza": 53, "baku": 51, "marina_bay": 62,
    "americas": 56, "rodriguez": 71, "interlagos": 71, "vegas": 50, "losail": 57,
    "yas_marina": 58, "bahrain": 57, "jeddah": 50, "imola": 63, "sepang": 56,
}


def build_live_state_frame(
    session_key: int,
    circuit_id: str,
    driver_code_to_id: dict[str, str],
    driver_to_constructor: dict[str, str],
    elo_ratings: dict[str, float],
    team_ratings: dict[str, float],
    grid_positions: dict[str, float],
) -> pd.DataFrame:
    """One row per driver in the exact LIVE_FEATURE_COLUMNS shape
    models/live_win_prob.py trains on — the live-inference counterpart to
    data/fastf1_client.py::build_lap_snapshots's historical training rows.
    `driver_code_to_id`/`driver_to_constructor`/`elo_ratings`/
    `team_ratings`/`grid_positions` are passed in (built once per session
    by the caller, e.g. the live poller) rather than recomputed per poll
    tick."""
    drivers = fetch_drivers(session_key)
    positions = fetch_positions(session_key)
    intervals = fetch_intervals(session_key)
    laps = fetch_laps(session_key)
    stints = fetch_stints(session_key)
    sc_active = safety_car_active_from_messages(fetch_race_control_flags(session_key))
    total_laps = CIRCUIT_LAP_COUNTS.get(circuit_id)

    # KNOWN LIMITATION, not fixed here (see docs/live_engine_design.md):
    # `/position` is a sparse event log that only emits a row when a
    # driver's position CHANGES, not a periodic snapshot — confirmed
    # directly. That makes "this driver's last update is old" ambiguous
    # between two very different situations (stably holding position for
    # a long stretch — genuinely current — vs. having retired and simply
    # stopped being tracked — stale) with no way to tell them apart from
    # this endpoint alone. An earlier attempt to filter "stale" rows by
    # update age excluded the actual race leader, who simply hadn't
    # changed position in a while, so it was removed rather than shipped
    # broken. A retired driver may therefore show a misleadingly
    # competitive P(win) until a cleaner signal (e.g. cross-referencing
    # race-control retirement messages) is built.
    rows = []
    for driver_number, code in drivers.items():
        driver_id = driver_code_to_id.get(code)
        if driver_id is None:
            continue

        pos_row = positions[positions["driver_number"] == driver_number]
        if pos_row.empty:
            continue
        current_position = float(pos_row["position"].iloc[0])

        int_row = intervals[intervals["driver_number"] == driver_number]
        gap_to_leader = _safe_float(int_row["gap_to_leader"].iloc[0]) if not int_row.empty else np.nan
        gap_ahead = _safe_float(int_row["interval"].iloc[0]) if not int_row.empty else np.nan

        driver_laps = laps[laps["driver_number"] == driver_number].sort_values("lap_number")
        lap_number = int(driver_laps["lap_number"].max()) if not driver_laps.empty else 0
        laps_remaining = float(total_laps - lap_number) if total_laps is not None else np.nan

        recent = driver_laps.tail(3)["lap_duration"].dropna() if "lap_duration" in driver_laps.columns else pd.Series(dtype=float)
        driver_roll3 = float(recent.mean()) if not recent.empty else np.nan
        field_at_lap = laps[laps["lap_number"] == lap_number]["lap_duration"].dropna() if "lap_duration" in laps.columns else pd.Series(dtype=float)
        field_median = float(field_at_lap.median()) if not field_at_lap.empty else np.nan
        pace_delta = driver_roll3 - field_median if pd.notna(driver_roll3) and pd.notna(field_median) else np.nan

        driver_stints = stints[stints["driver_number"] == driver_number].sort_values("stint_number")
        pit_stops_so_far = max(0, len(driver_stints) - 1)
        pitted_this_lap = False
        tyre_age_laps = np.nan
        if not driver_stints.empty:
            current_stint_rows = driver_stints[driver_stints["lap_start"] <= max(lap_number, 1)].tail(1)
            if not current_stint_rows.empty:
                stint = current_stint_rows.iloc[0]
                tyre_age_laps = float(stint["tyre_age_at_start"] + max(0, lap_number - stint["lap_start"]))
                pitted_this_lap = bool(int(stint["lap_start"]) == lap_number and int(stint["stint_number"]) > 1)

        constructor_id = driver_to_constructor.get(driver_id)
        rows.append(
            {
                "driver_id": driver_id,
                "lap_number": lap_number,
                "laps_remaining": laps_remaining,
                "current_position": current_position,
                "gap_to_leader_s": gap_to_leader,
                "gap_ahead_s": gap_ahead,
                "tyre_age_laps": tyre_age_laps,
                "pace_delta_last_3_laps": pace_delta,
                "pit_stops_so_far": pit_stops_so_far,
                "pitted_this_lap": pitted_this_lap,
                "safety_car_active": sc_active,
                "grid_position": grid_positions.get(driver_id, np.nan),
                "driver_pre_race_strength": elo_ratings.get(driver_id, np.nan),
                "constructor_strength": team_ratings.get(constructor_id, np.nan),
                "inferred_soc": np.nan,
                "deploy_time_s": np.nan,
                "harvest_time_s": np.nan,
                "clipping_time_s": np.nan,
            }
        )

    return pd.DataFrame(rows)
