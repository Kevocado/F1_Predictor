"""Tests for the read-only F1 /facts bundle the match explainer consumes.

Offline throughout: the prediction bundle, the tracking store and the
contributor source are all injected, so nothing reaches jolpica, the
weather feed or the model files.

The race story comes only from data this project already has: grid, headline
chance, podium/points/DNF, and the existing /explain contributors for the
top three plus the biggest grid-vs-predicted mover.

The rule pinned hardest here: once a session has STARTED, the pick comes only
from the stored pre-session record. A current prediction is today's model and
may not stand in for it.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, model_validator

from f1_predictor.api import facts as facts_mod
from f1_predictor.api.main import app

# The contract, copied from predictor-hub/services/explainer/explainer/facts.py.
class Market(BaseModel):
    market: str
    model_config = ConfigDict(extra="allow")


class Facts(BaseModel):
    sport: Literal["pl", "f1", "nfl", "cfb", "nba"]
    id: str
    title: str
    starts_at: str
    status: Literal["upcoming", "live", "final"]
    pick_timing: Literal["pre_kickoff", "rebuilt", "none"]
    pick: dict | None = None
    markets: list[Market] = []
    drivers: list[dict] = []
    context: dict = {}
    players: list[dict] = []
    record: dict | None = None
    result: dict | None = None

    @model_validator(mode="after")
    def _rebuilt_never_won(self) -> "Facts":
        if self.pick_timing == "rebuilt" and self.result and "pick_won" in self.result:
            raise ValueError("a rebuilt pick cannot carry result.pick_won")
        return self


NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
SEASON, ROUND = 2026, 15
SESSION_TIME = "2026-09-13T14:00:00Z"


def _driver(driver_id, name, win, grid=None, podium=None, points=None, dnf=None):
    return {
        "driver_id": driver_id,
        "name": name,
        "constructor": driver_id.split("_")[0] if "_" in driver_id else driver_id,
        "grid": grid,
        "win": win,
        "podium": podium,
        "points": points,
        "dnf": dnf,
    }


DRIVERS = [
    _driver("ver_1", "Max Verstappen", 0.31, grid=1, podium=0.72, points=22.4, dnf=0.06),
    _driver("nor_1", "Lando Norris", 0.27, grid=2, podium=0.66, points=20.9, dnf=0.08),
    _driver("rus_1", "George Russell", 0.19, grid=9, podium=0.51, points=16.2, dnf=0.11),
    _driver("ham_1", "Lewis Hamilton", 0.11, grid=3, podium=0.38, points=12.1, dnf=0.09),
    _driver("lec_1", "Charles Leclerc", 0.06, grid=4, podium=0.22, points=8.4, dnf=0.12),
]


def _prediction(drivers=None, source="live", tier="pre_qualifying", **over):
    prediction = {
        "season": SEASON,
        "round": ROUND,
        "session_type": "race",
        "race_name": "Italian Grand Prix",
        "circuit": "Monza",
        "sprint_weekend": False,
        "tier": tier,
        "source": source,
        "session_time": SESSION_TIME,
        "weather": {"air_temp_c": 28.0, "rain_probability": 0.1},
        "drivers": drivers if drivers is not None else [dict(d) for d in DRIVERS],
    }
    prediction.update(over)
    return prediction


def _stored(drivers=None, snapshotted_at="2026-09-13T10:00:00Z", session_time=SESSION_TIME):
    """The pre-session tracking rows: driver_id -> market -> predicted_prob."""
    rows = []
    for d in (drivers if drivers is not None else DRIVERS):
        rows.append({"driver_id": d["driver_id"], "market": "win", "predicted_prob": d["win"],
                     "expected_position": d["grid"], "session_time": session_time,
                     "snapshotted_at": snapshotted_at, "tier": "pre_qualifying", "backfilled": 0,
                     "resolved": 0, "actual_outcome": None})
    return rows


def _contributors():
    return {
        "ver_1": [{"label": "Grid position", "value": 1.0, "direction": "raises"},
                  {"label": "Team strength", "value": 0.92, "direction": "raises"}],
        "nor_1": [{"label": "Grid position", "value": 2.0, "direction": "raises"}],
        "rus_1": [{"label": "Form, last 3 races", "value": 2.4, "direction": "lowers"},
                  {"label": "Circuit average position", "value": 3.8, "direction": "lowers"}],
        "ham_1": [{"label": "Team strength", "value": 0.8, "direction": "lowers"}],
    }


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(facts_mod, "_now", lambda: NOW)
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda season, rnd, session: _prediction())
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda season, rnd, session, tier=None: _stored())
    monkeypatch.setattr(facts_mod, "_contributors_for", lambda season, rnd, session: _contributors())
    monkeypatch.setattr(
        facts_mod.store, "get_session_track_record",
        lambda season=None, session_type=None: {"n_resolved": 40, "n_correct": 24, "pct_correct": 0.6},
    )
    return TestClient(app)


# --- the contract -------------------------------------------------------

def test_bundle_validates_against_the_contract(api):
    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    Facts(**body)
    assert body["sport"] == "f1"
    assert body["id"] == f"{SEASON}-{ROUND}-race"
    assert body["title"] == "Italian Grand Prix · Race"
    assert body["starts_at"] == SESSION_TIME
    assert body["status"] == "upcoming"
    assert body["pick"] == {"label": "Max Verstappen", "prob": 0.31}
    assert body["pick_timing"] == "pre_kickoff"


def test_context_carries_circuit_sprint_tier_and_weather(api):
    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    assert body["context"]["circuit"] == "Monza"
    assert body["context"]["sprint_weekend"] is False
    assert body["context"]["tier"] == "pre_qualifying"
    assert "28" in str(body["context"]["weather"])


def test_qualifying_session_is_titled_pole_and_picks_by_pole(api, monkeypatch):
    quali = _prediction(
        session_type="qualifying", race_name="Italian Grand Prix",
        drivers=[_driver("a_1", "Pole sitter", 0.40, grid=1), _driver("b_1", "Second", 0.30, grid=2)],
    )
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: quali)
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda s, r, x, tier=None: [])

    body = api.get(f"/facts/{SEASON}-{ROUND}-qualifying").json()

    assert body["title"] == "Italian Grand Prix · Qualifying"
    assert body["pick"]["label"] == "Pole sitter"


def test_drivers_are_the_top_eight_with_grid_and_markets(api, monkeypatch):
    many = [_driver(f"d{i}", f"Driver {i}", round(0.4 - i * 0.03, 3), grid=i + 1,
                    podium=0.5, points=10.0, dnf=0.1) for i in range(10)]
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: _prediction(drivers=many))
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda s, r, x, tier=None: [])

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    assert len(body["drivers"]) == 8
    assert body["drivers"][0]["name"] == "Driver 0"
    assert body["drivers"][0]["grid"] == 1
    assert body["drivers"][0]["win"] == pytest.approx(0.4)
    assert "podium" in body["drivers"][0] and "points" in body["drivers"][0] and "dnf" in body["drivers"][0]


def test_win_podium_points_and_dnf_markets_are_present(api):
    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()
    by_market = {m["market"]: m for m in body["markets"]}

    # The win market carries every driver's headline chance.
    assert by_market["win"]["model"]["Max Verstappen"] == pytest.approx(0.31)
    assert by_market["win"]["model"]["Lando Norris"] == pytest.approx(0.27)
    assert by_market["podium"]["model"]["Max Verstappen"] == pytest.approx(0.72)
    assert by_market["points"]["model"]["Max Verstappen"] == pytest.approx(22.4)
    assert by_market["dnf"]["model"]["Max Verstappen"] == pytest.approx(0.06)


def test_contributors_present_for_the_top_three_and_the_biggest_mover(api):
    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()
    by_name = {d["name"]: d for d in body["drivers"]}

    # Top three by win chance.
    for name in ("Max Verstappen", "Lando Norris", "George Russell"):
        assert by_name[name]["contributors"], name
    # The biggest mover: Russell starts 9th but is predicted 3rd.
    assert by_name["George Russell"]["grid"] == 9
    assert by_name["George Russell"]["contributors"]


def test_the_biggest_mover_gets_contributors_even_outside_the_top_three(api, monkeypatch):
    drivers = [
        _driver("a_1", "Front", 0.30, grid=1),
        _driver("b_1", "Second", 0.25, grid=2),
        _driver("c_1", "Third", 0.20, grid=3),
        _driver("d_1", "Mover", 0.15, grid=20),  # 20th on the grid, predicted 4th
    ]
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: _prediction(drivers=drivers))
    monkeypatch.setattr(
        facts_mod, "_contributors_for", lambda s, r, x: {"d_1": [{"label": "Grid position", "value": 20.0, "direction": "lowers"}]},
    )
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda s, r, x, tier=None: [])

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()
    by_name = {d["name"]: d for d in body["drivers"]}

    assert by_name["Mover"]["contributors"][0]["label"] == "Grid position"
    # A driver outside the top three and not the mover gets none.
    assert not by_name["Third"].get("contributors")


def test_record_reports_pre_session_hits(api):
    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    assert body["record"] == {"label": "Picks made before the session", "hits": 24, "settled": 40}


# --- pick_timing --------------------------------------------------------

def test_pick_timing_is_none_when_the_session_has_no_prediction_rows(api, monkeypatch):
    # The session is known (it has a prediction) but carries no driver rows,
    # so there is nothing to claim a pick from.
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: _prediction(drivers=[]))
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda s, r, x, tier=None: [])

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    assert body["pick_timing"] == "none"
    assert body["pick"] is None


def test_pick_timing_is_rebuilt_for_a_rebuilt_source(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: _prediction(source="rebuilt"))

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    assert body["pick_timing"] == "rebuilt"


def test_pick_timing_is_rebuilt_for_a_backtest_source(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: _prediction(source="backtest"))

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    assert body["pick_timing"] == "rebuilt"


def test_pick_timing_is_rebuilt_when_the_stored_snapshot_was_taken_late(api, monkeypatch):
    monkeypatch.setattr(
        facts_mod, "_stored_rows",
        lambda s, r, x, tier=None: _stored(snapshotted_at="2026-09-13T15:00:00Z"),  # after the session
    )
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: _prediction(source="tracked"))

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    assert body["pick_timing"] == "rebuilt"


# --- THE RULE: a started session uses the stored pre-session record -------

def test_started_session_uses_the_stored_record_not_the_current_prediction(api, monkeypatch):
    started = _prediction(session_time="2026-09-01T14:00:00Z", source="live")
    # The stored pre-session record disagrees with today's model.
    stored = _stored(
        drivers=[_driver("nor_1", "Lando Norris", 0.44, grid=2), _driver("ver_1", "Max Verstappen", 0.30, grid=1)],
        snapshotted_at="2026-08-31T10:00:00Z", session_time="2026-09-01T14:00:00Z",
    )
    for row in stored:
        row["resolved"] = 1
        row["actual_outcome"] = 0
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: started)
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda s, r, x, tier=None: stored)

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    assert body["status"] == "final"
    # The stored record's Norris-at-0.44 pick, not the current model's Verstappen.
    assert body["pick"] == {"label": "Lando Norris", "prob": 0.44}
    assert body["pick_timing"] == "pre_kickoff"


def test_started_session_still_lists_its_drivers_from_the_stored_record(api, monkeypatch):
    started = _prediction(session_time="2026-09-01T14:00:00Z", source="live")
    stored = _stored(
        drivers=[_driver("nor_1", "Lando Norris", 0.44, grid=2), _driver("ver_1", "Max Verstappen", 0.30, grid=1)],
        snapshotted_at="2026-08-31T10:00:00Z", session_time="2026-09-01T14:00:00Z",
    )
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: started)
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda s, r, x, tier=None: stored)

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    by_name = {d["name"]: d for d in body["drivers"]}
    assert by_name["Lando Norris"]["win"] == pytest.approx(0.44)  # the stored chance
    assert by_name["Lando Norris"]["grid"] == 2
    # A completed session shows only what was stored before it: today's
    # post-session podium/points/DNF would be hindsight.
    assert "podium" not in by_name["Lando Norris"]
    assert by_name["Max Verstappen"]["win"] == pytest.approx(0.30)


def test_started_session_with_no_stored_record_has_no_pick(api, monkeypatch):
    started = _prediction(session_time="2026-09-01T14:00:00Z", source="live")
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: started)
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda s, r, x, tier=None: [])

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    # A session that has started with nothing stored before it: no pick at
    # all, so nothing can be judged from today's model.
    assert body["pick"] is None
    assert body["pick_timing"] == "none"
    assert body["markets"] == []


def test_started_session_judges_the_stored_pick_on_the_real_result(api, monkeypatch):
    started = _prediction(session_time="2026-09-01T14:00:00Z", source="live")
    stored = _stored(
        drivers=[_driver("nor_1", "Lando Norris", 0.44, grid=2), _driver("ver_1", "Max Verstappen", 0.30, grid=1)],
        snapshotted_at="2026-08-31T10:00:00Z", session_time="2026-09-01T14:00:00Z",
    )
    # Resolved session: Verstappen actually won, so the stored Norris pick lost.
    for row in stored:
        row["resolved"] = 1
        row["actual_outcome"] = 1 if row["driver_id"] == "ver_1" else 0
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: started)
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda s, r, x, tier=None: stored)

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    assert body["pick"] == {"label": "Lando Norris", "prob": 0.44}
    assert body["result"]["winner"] == "Max Verstappen"
    assert body["result"]["pick_won"] is False
    Facts(**body)


def test_started_session_omits_pick_won_for_a_rebuilt_record(api, monkeypatch):
    started = _prediction(session_time="2026-09-01T14:00:00Z", source="rebuilt")
    stored = _stored(snapshotted_at="2026-09-01T15:00:00Z", session_time="2026-09-01T14:00:00Z")
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: started)
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda s, r, x, tier=None: stored)

    body = api.get(f"/facts/{SEASON}-{ROUND}-race").json()

    assert body["pick_timing"] == "rebuilt"
    assert "pick_won" not in (body["result"] or {})


# --- /facts/upcoming ----------------------------------------------------

def test_upcoming_lists_only_sessions_inside_the_window(api, monkeypatch):
    soon = _prediction(session_time=(NOW + timedelta(hours=10)).isoformat().replace("+00:00", "Z"))
    later = _prediction(session_time=(NOW + timedelta(hours=100)).isoformat().replace("+00:00", "Z"))
    past = _prediction(session_time=(NOW - timedelta(hours=10)).isoformat().replace("+00:00", "Z"))
    monkeypatch.setattr(facts_mod, "_upcoming_sessions", lambda: [
        {"season": SEASON, "round": 1, "session": "race", "session_time": soon["session_time"]},
        {"season": SEASON, "round": 2, "session": "race", "session_time": later["session_time"]},
        {"season": SEASON, "round": 3, "session": "race", "session_time": past["session_time"]},
    ])

    body = api.get("/facts/upcoming?hours=72").json()

    assert body["ids"] == [f"{SEASON}-1-race"]


def test_unknown_session_is_404(api, monkeypatch):
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda s, r, x: None)
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda s, r, x, tier=None: [])

    assert api.get("/facts/1999-1-race").status_code == 404
