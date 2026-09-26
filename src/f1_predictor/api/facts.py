"""facts.py — the read-only F1 /facts bundle the match explainer consumes.

Read-only and suggest-only: this router never writes, never retrains and
never places anything.

The race story is told only from data this project already has: grid
position, headline chance, podium/points/DNF, and the contributors the
existing /explain endpoint already computes for the top three plus the
biggest grid-versus-predicted mover. Nothing is invented.

Two rules, both learned in Tasks 8-10:

1. **A started session is described only by its stored pre-session record.**
   The current prediction is today's model. ``honest_source`` and
   ``made_before_session`` decide whether a stored snapshot is 'tracked'
   (written before the session) or 'rebuilt'; a started session with no
   tracked record has no pick at all, so nothing is judged from hindsight.
2. **No per-request state in module globals** — these are sync endpoints and
   FastAPI runs them in a thread pool, so every id is passed explicitly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from . import routes
from ..tracking import store

router = APIRouter()
logger = logging.getLogger(__name__)

#: The session types an id may carry, in the order a race weekend runs them.
SESSION_TYPES = ("qualifying", "sprint", "race")
#: How a driver is headline-chosen: by race win chance, or by pole in quali.
_HEADLINE_KEY = {"race": "win", "sprint": "win", "qualifying": "win"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- id parsing ---------------------------------------------------------

def parse_session_id(session_id: str) -> tuple[int, int, str] | None:
    """'2026-15-race' -> (2026, 15, 'race')."""
    parts = str(session_id).split("-")
    if len(parts) != 3:
        return None
    season, round_, session = parts
    if session not in SESSION_TYPES:
        return None
    try:
        return int(season), int(round_), session
    except ValueError:
        return None


# --- helpers ------------------------------------------------------------

def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _as_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def _iso_utc(value: Any) -> str:
    stamp = _as_utc(value)
    return "" if stamp is None else stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- data access --------------------------------------------------------

def _current_prediction(season: int, round_: int, session: str) -> dict | None:
    """Today's model read (the site's own routes). None when unknown."""
    try:
        if session == "race":
            payload = routes.get_race_prediction(season, round_)
        elif session == "qualifying":
            payload = routes.get_qualifying_prediction(season, round_)
        else:
            payload = routes.get_sprint_prediction(season, round_)
    except Exception:
        logger.info("no %s prediction for %s round %s", session, season, round_)
        return None
    if payload is None:
        return None
    return routes.honest_source(season, round_, session, _as_dict(payload))


def _as_dict(payload: Any) -> dict:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    if hasattr(payload, "dict"):
        return payload.dict()
    return {}


def _stored_rows(season: int, round_: int, session: str, tier: str | None = None) -> list[dict]:
    """The pre-session tracking rows. These are the only numbers a started
    session may be described by."""
    try:
        return store.get_session_prediction(season, round_, session, tier=tier)
    except Exception:
        logger.info("no stored %s rows for %s round %s", session, season, round_)
        return []


def _contributors_for(season: int, round_: int, session: str) -> dict[str, list[dict]]:
    """driver_id -> contributor dicts, from the same explain machinery the
    /explain endpoint uses. Never computes anything new."""
    try:
        frame, feature_cols = routes._race_feature_frame_for_explain(season, round_)
    except Exception:
        logger.info("explain features unavailable for %s round %s", season, round_)
        return {}
    if frame is None or frame.empty:
        return {}
    out: dict[str, list[dict]] = {}
    try:
        from ..models import explain as explain_lib

        contrib = explain_lib.strength_contributions(frame, feature_cols)
    except Exception:
        logger.info("strength contributions unavailable for %s round %s", season, round_)
        return {}
    if contrib is None:
        return {}
    for _, row in frame.iterrows():
        driver_id = row.get("driver_id")
        if driver_id is None or driver_id not in contrib.index:
            continue
        raw = frame.loc[frame["driver_id"] == driver_id].iloc[0]
        items = explain_lib.top_contributors(contrib.loc[driver_id], raw, feature_cols, n=5)
        out[str(driver_id)] = [
            {
                "label": _label(item.get("feature")),
                "value": item.get("value"),
                "direction": "raises" if _num(item.get("contribution")) >= 0 else "lowers",
            }
            for item in items
        ]
    return out


def _label(feature: Any) -> str:
    return str(feature or "").replace("_", " ").strip().capitalize()


def _upcoming_sessions() -> list[dict]:
    """Sessions with a known start time, for the pre-generation window."""
    out: list[dict] = []
    try:
        schedule = routes.jolpica.fetch_season_schedule(routes.CURRENT_SEASON)
    except Exception:
        logger.info("no schedule available for /facts/upcoming")
        return []
    if schedule is None or getattr(schedule, "empty", True):
        return out
    for _, row in schedule.iterrows():
        round_ = row.get("round")
        when = _as_utc(row.get("race_datetime"))
        if round_ is None or when is None:
            continue
        out.append({
            "season": routes.CURRENT_SEASON,
            "round": int(round_),
            "session": "race",
            "session_time": when.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        })
    return out


# --- bundle assembly ----------------------------------------------------

def _status(prediction: dict | None, now: datetime, stored_rows: list[dict] | None = None) -> str:
    """A session that has begun and whose stored rows are all resolved is a
    final; one that has begun without resolved rows is still 'live'. Nothing
    is called final before it has actually happened."""
    when = _as_utc((prediction or {}).get("session_time"))
    if when is None or when > now:
        return "upcoming"
    rows = stored_rows or []
    if rows and all(bool(row.get("resolved")) for row in rows):
        return "final"
    return "live"


def _stored_by_driver(rows: list[dict]) -> dict[str, dict]:
    """driver_id -> {win, position} from the stored pre-session rows."""
    out: dict[str, dict] = {}
    for row in rows:
        driver_id = str(row.get("driver_id"))
        entry = out.setdefault(driver_id, {})
        if row.get("market") == "win":
            entry["win"] = _num(row.get("predicted_prob"))
        position = _num(row.get("expected_position"))
        if position is not None:
            entry["position"] = position
    return out


def _name_by_driver(prediction: dict) -> dict[str, str]:
    names = {}
    for d in prediction.get("drivers") or []:
        driver_id = d.get("driver_id")
        if driver_id is not None:
            names[str(driver_id)] = d.get("name") or str(driver_id)
    return names


def _pick_timing(prediction: dict | None, stored_tracked: bool) -> str:
    if prediction is None and not stored_tracked:
        return "none"
    source = (prediction or {}).get("source")
    if source in ("rebuilt", "backtest"):
        return "rebuilt"
    if not stored_tracked:
        return "rebuilt"
    return "pre_kickoff"


def _markets(prediction: dict | None, stored: dict[str, dict], names: dict[str, str]) -> list[dict]:
    """win/podium/points/dnf, per the spec's F1 market names. Numbers come
    from the stored record for a started session, never from today's model."""
    if prediction is None and not stored:
        return []
    source = stored or {}
    drivers = prediction.get("drivers") or [] if prediction else []
    by_id = {str(d.get("driver_id")): d for d in drivers}

    def field(market: str, key: str) -> dict:
        out = {}
        for driver_id, d in by_id.items():
            value = _num(d.get(key))
            if value is not None:
                out[names.get(driver_id, driver_id)] = value
        return out

    out: list[dict] = []
    win = field("win", "win")
    if not win:
        win = {names.get(k, k): v["win"] for k, v in source.items() if v.get("win") is not None}
    if win:
        out.append({"market": "win", "model": win})
    for market, key in (("podium", "podium"), ("points", "points"), ("dnf", "dnf")):
        values = field(market, key)
        if values:
            out.append({"market": market, "model": values})
    return out


def _drivers(
    prediction: dict | None,
    stored: dict[str, dict],
    names: dict[str, str],
    contributors: dict[str, list[dict]],
    started: bool,
) -> list[dict]:
    """Top eight by headline chance, each with grid/win/podium/points/dnf.

    For a STARTED session only the stored pre-session numbers are shown (win
    and grid); quoting today's post-session podium/points/DNF would be
    hindsight. Contributors are attached to the top three plus the biggest
    mover — the largest gap between grid position and predicted rank."""
    by_id = {str(d.get("driver_id")): d for d in (prediction or {}).get("drivers") or []}
    rows = []
    for driver_id in set(by_id) | set(stored):
        current = by_id.get(driver_id, {})
        entry = stored.get(driver_id, {})
        if started:
            win = entry.get("win")
            grid = entry.get("position")
            rows.append({
                "driver_id": driver_id,
                "name": current.get("name") or names.get(driver_id, driver_id),
                "grid": grid,
                "win": win,
            })
        else:
            rows.append({
                "driver_id": driver_id,
                "name": current.get("name") or names.get(driver_id, driver_id),
                "grid": _num(current.get("grid")) if _num(current.get("grid")) is not None else entry.get("position"),
                "win": _num(current.get("win")) if _num(current.get("win")) is not None else entry.get("win"),
                "podium": _num(current.get("podium")),
                "points": _num(current.get("points")),
                "dnf": _num(current.get("dnf")),
            })
    rows = [r for r in rows if r["win"] is not None]
    rows.sort(key=lambda r: r["win"], reverse=True)
    top = rows[:8]

    # Predicted rank is the order above; the mover is the largest
    # grid-minus-rank gap among the drivers that actually have a grid.
    mover = None
    best_gap = 0.5  # ignore trivial gaps
    for rank, row in enumerate(top, start=1):
        if row["grid"] is None:
            continue
        gap = abs(row["grid"] - rank)
        if gap > best_gap:
            best_gap = gap
            mover = row["driver_id"]

    featured = {r["driver_id"] for r in top[:3]}
    if mover:
        featured.add(mover)
    for row in top:
        if row["driver_id"] in featured and contributors.get(row["driver_id"]):
            row["contributors"] = contributors[row["driver_id"]]
    return top


def _context(prediction: dict | None) -> dict:
    prediction = prediction or {}
    context: dict[str, Any] = {}
    for key in ("circuit", "tier"):
        if prediction.get(key):
            context[key] = prediction[key]
    if prediction.get("sprint_weekend") is not None:
        context["sprint_weekend"] = bool(prediction["sprint_weekend"])
    weather = prediction.get("weather")
    if weather:
        context["weather"] = weather
    return context


def _record() -> dict | None:
    try:
        data = store.get_session_track_record() or {}
    except Exception:
        return None
    settled = int(data.get("n_resolved") or 0)
    if settled <= 0:
        return None
    return {
        "label": "Picks made before the session",
        "hits": int(data.get("n_correct") or 0),
        "settled": settled,
    }


def _result(
    stored_rows: list[dict],
    names: dict[str, str],
    status: str,
    pick_timing: str,
    pick_label: str | None,
) -> dict | None:
    """The winner, and whether the pre-session pick took it. The winner comes
    from the resolved stored rows, never from a model."""
    if status != "final" or not stored_rows:
        return None
    winner_id = None
    for row in stored_rows:
        if row.get("market") == "win" and _num(row.get("actual_outcome")):
            winner_id = str(row.get("driver_id"))
    result: dict[str, Any] = {}
    if winner_id:
        result["winner"] = names.get(winner_id, winner_id)
    if pick_timing == "pre_kickoff" and pick_label and winner_id:
        # Map the pick's display name back to its driver id via the name map.
        pick_id = next((d for d, n in names.items() if n == pick_label), pick_label)
        result["pick_won"] = bool(winner_id == pick_id)
    return result or None


@router.get("/facts/upcoming")
def get_facts_upcoming(hours: int = 72) -> dict:
    """Ids of sessions starting within the window, for pre-generation."""
    if hours < 0:
        raise HTTPException(status_code=422, detail="hours must be >= 0")
    now = _now()
    cutoff = now + timedelta(hours=hours)
    ids = []
    for entry in _upcoming_sessions():
        when = _as_utc(entry.get("session_time"))
        if when is None or when <= now or when > cutoff:
            continue
        ids.append(f"{entry['season']}-{entry['round']}-{entry['session']}")
    return {"ids": ids}


@router.get("/facts/{session_id}")
def get_facts(session_id: str) -> dict:
    now = _now()
    parsed = parse_session_id(session_id)
    if parsed is None:
        raise HTTPException(status_code=404, detail=f"Unknown session id: {session_id}")
    season, round_, session = parsed

    prediction = _current_prediction(season, round_, session)
    stored_rows = _stored_rows(season, round_, session)
    stored = _stored_by_driver(stored_rows)
    if prediction is None and not stored:
        raise HTTPException(status_code=404, detail=f"Unknown session id: {session_id}")

    status = _status(prediction, now, stored_rows)
    started = status in ("live", "final")

    # THE RULE: a started session is described only by what was stored before
    # it began. The current prediction is today's model and is discarded for
    # the pick. A stored row is 'tracked' only when every row was written
    # before the session started (store.made_before_session), which is the
    # same test routes._best_snapshot uses.
    stored_tracked = bool(stored_rows) and all(
        store.made_before_session(row.get("snapshotted_at"), row.get("session_time"))
        for row in stored_rows
    )
    if started:
        source = stored if stored else None
        source_source = {"source": "tracked"} if stored_tracked else {"source": "rebuilt"}
    else:
        source = prediction
        source_source = {"source": (prediction or {}).get("source")}

    names = _name_by_driver(prediction) if prediction else {}
    for driver_id in stored:
        names.setdefault(driver_id, driver_id)

    pick_timing = _pick_timing(source_source, stored_tracked)
    headline_key = _HEADLINE_KEY[session]
    pick = None
    for driver_id, entry in sorted(stored.items(), key=lambda kv: -(kv[1].get("win") or 0)):
        if entry.get("win") is not None:
            pick = {"label": names.get(driver_id, driver_id), "prob": entry["win"]}
            break
    if pick is None and source and not started:
        drivers = [d for d in (source.get("drivers") or []) if _num(d.get(headline_key)) is not None]
        drivers.sort(key=lambda d: -_num(d[headline_key]))
        if drivers:
            pick = {"label": drivers[0].get("name"), "prob": _num(drivers[0][headline_key])}
    if pick is None:
        pick_timing = "none"

    contributors = _contributors_for(season, round_, session) if (pick_timing != "none") else {}
    result = _result(stored_rows, names, status, pick_timing, pick["label"] if pick else None)
    if pick_timing == "rebuilt" and result:
        result.pop("pick_won", None)

    return {
        "sport": "f1",
        "id": str(session_id),
        "title": f"{((prediction or {}).get('race_name') or 'Grand Prix')} · {session.capitalize()}",
        "starts_at": _iso_utc((prediction or {}).get("session_time")),
        "status": status,
        "pick_timing": pick_timing,
        "pick": pick,
        "markets": [] if (started and not stored) else _markets(source, stored, names),
        "drivers": _drivers(prediction, stored, names, contributors, started),
        "context": _context(prediction),
        "players": [],
        "record": _record(),
        "result": result,
    }
