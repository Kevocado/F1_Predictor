"""The race-prediction API must carry a display name for every driver, not
just the driver_id.

The hub's F1 teaser reads `driver_name` off the same payload the site reads.
The API only ever had `driver_id` (e.g. "russell"), so every consumer had to
invent a display name: the F1 site title-cases the id
(frontend/src/lib/teamColors.ts:46), which turns "russell" into "Russell" and
drops the first name. With the name on the API, hub and site share one string
instead of two derivations that can drift.

The same change widens the expected_* floats to Optional, and these tests
hold that half too: expected_position/expected_points are NaN in 287 of the
515 race rows of the committed snapshot (measured: data/public_snapshot.json
— rounds 1-12 and 15, the tracked-completed path's NaN fallback), pydantic
2.13.5 serialises NaN to null on the way out, and the snapshot-sanitising
task in this plan starts writing literal null where the NaN used to be. A
non-optional `float` rejects None on re-validation (measured in this repo's
venv), which would 500 every endpoint that serves those rows through a
response_model.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from f1_predictor.api import routes
from f1_predictor.data import jolpica

_SEASON = 2026
_ROUND = 1


def _fake_schedule() -> pd.DataFrame:
    """The race-prediction fixture test_race_prediction_caching.py already
    uses: one far-future round, so the race is never "completed" and the
    route takes the upcoming-race path (monkeypatched sim, no network, no
    backtest)."""
    return pd.DataFrame(
        [
            {
                "round": 1,
                "race_name": "Fake Grand Prix",
                "circuit_name": "Fake Circuit",
                "race_datetime": pd.Timestamp("2099-01-01", tz="UTC"),  # far future — never "completed"
                "is_sprint_weekend": False,
            }
        ]
    )


def _fake_sim() -> pd.DataFrame:
    """The same fixture shape (same columns) as test_race_prediction_caching
    .py's _fake_sim, plus a second driver: "driver_1" is an id no name source
    knows, "max_verstappen" has the multi-part shape real driver ids use —
    so the assertions below cover both title-case shapes the fallback
    produces."""
    return pd.DataFrame(
        [
            {
                "driver_id": "driver_1",
                "constructor_id": "team_1",
                "p_win": 0.4,
                "p_podium": 0.7,
                "p_points_finish": 0.9,
                "p_dnf": 0.05,
                "expected_position": 2.0,
                "expected_points": 15.0,
                "actual_position": None,
                "actual_dnf": None,
            },
            {
                "driver_id": "max_verstappen",
                "constructor_id": "team_2",
                "p_win": 0.3,
                "p_podium": 0.6,
                "p_points_finish": 0.8,
                "p_dnf": 0.1,
                "expected_position": 3.0,
                "expected_points": 12.0,
                "actual_position": None,
                "actual_dnf": None,
            },
        ]
    )


@pytest.fixture(autouse=True)
def _fake_race_prediction(monkeypatch):
    """Route the live race-prediction path through the fixtures above — the
    same monkeypatch shape test_race_prediction_caching.py already uses, so
    this file invents no new fixture. Autouse so the tests can call
    _prediction_payload_for_test() with no arguments of their own."""
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule())

    def fake_predict_upcoming_race(season, round_, race_row):
        # The third value is the history window this prediction was built on;
        # nothing was skipped, because nothing was fetched.
        return _fake_sim(), "pre_qualifying", jolpica.HistoryCoverage(
            seasons_requested=(2026, 2025, 2024), seasons_loaded=(2026, 2025, 2024)
        )

    monkeypatch.setattr(routes, "_predict_upcoming_race", fake_predict_upcoming_race)


def _prediction_payload_for_test() -> dict:
    """The race-prediction payload, exactly as the route builds it."""
    return routes.get_race_prediction(_SEASON, _ROUND).model_dump()


def _client() -> TestClient:
    """The routes mounted the way test_facts_real_schema.py mounts its
    routers: a bare FastAPI app with no lifespan — so no snapshot poller, no
    live poller, no network. Needed because response_model re-validation
    happens in the ASGI layer, not on a direct route call — that is where
    the 500s these tests guard against would occur."""
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_race_prediction_carries_a_driver_name():
    """The API returns driver_id only, so the hub shows a raw id like
    'russell'. The site title-cases it; the API should just say the name."""
    payload = _prediction_payload_for_test()
    rows = payload["predictions"]
    assert rows, "no driver rows in the fixture"
    assert all(r.get("driver_name") for r in rows), [r for r in rows if not r.get("driver_name")]


def test_driver_name_falls_back_to_the_id_when_unknown():
    rows = _prediction_payload_for_test()["predictions"]
    assert all(isinstance(r["driver_name"], str) and r["driver_name"] for r in rows)


def test_driver_name_matches_the_sites_title_casing():
    """teamColors.ts:46 driverName is the one algorithm for turning a bare id
    into a display name, and the API's fallback mirrors it verbatim so the
    hub and the site can never disagree: "max_verstappen" -> "Max
    Verstappen", "driver_1" -> "Driver 1"."""
    names = {r["driver_id"]: r["driver_name"] for r in _prediction_payload_for_test()["predictions"]}
    assert names["max_verstappen"] == "Max Verstappen"
    assert names["driver_1"] == "Driver 1"


def test_title_case_changes_only_the_first_letter_of_each_part():
    """Pins the one subtle way the mirror could drift from the site: the
    site's driverName upper-cases charAt(0) of each "_"-separated part and
    leaves the rest verbatim, while str.capitalize would also lower-case the
    rest ("aBC" -> "Abc"). Also pins that a single-part id gains no trailing
    space (a naive "first part + ' ' + rest" join would emit one)."""
    assert routes._title_case("aBC_deF") == "ABC DeF"
    assert routes._title_case("russell") == "Russell"
    assert routes._title_case("") == ""


def test_public_mode_serves_a_snapshot_row_without_a_name_or_an_expected_position(monkeypatch):
    """The two regressions this change guards, end to end through the
    response_model (a direct route call never re-validates; only the ASGI
    layer does):

    1. A snapshot generated before driver_name existed — the committed one —
       has rows carrying only driver_id. A non-optional driver_name would
       reject the route's own response, so the route fills the same fallback
       the live path uses into rows that lack the field. A row that already
       carries a name keeps it (the fill is per-row, gaps only — asserted on
       the second row below, so a future "helpfully rewrite every row"
       regression fails here).
    2. After the snapshot-sanitising task in this plan lands, the NaNs that
       used to sit in expected_position/expected_points become literal null.
       A non-optional float rejects None, so the widened Optional fields
       have to accept it.
    """
    snap = {
        "season": _SEASON,
        "predictions": {
            str(_ROUND): {
                "season": _SEASON,
                "round": _ROUND,
                "race_name": "Fake Grand Prix",
                "tier": "post_qualifying",
                "source": "live",
                "predictions": [
                    {
                        "driver_id": "russell",
                        "constructor_id": "mercedes",
                        "p_win": 0.15,
                        "p_podium": 0.4,
                        "p_points_finish": 0.85,
                        "p_dnf": 0.09,
                        "expected_position": None,
                        "expected_points": None,
                        "actual_position": 1,
                        "actual_dnf": False,
                    },
                    {
                        "driver_id": "verstappen",
                        # Already carries a name — the backfill must keep it.
                        "driver_name": "Max Verstappen",
                        "constructor_id": "red_bull",
                        "p_win": 0.3,
                        "p_podium": 0.6,
                        "p_points_finish": 0.8,
                        "p_dnf": 0.1,
                        "expected_position": 2.5,
                        "expected_points": 14.0,
                        "actual_position": 2,
                        "actual_dnf": False,
                    },
                ],
            }
        },
    }
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: snap)

    res = _client().get(f"/api/races/{_SEASON}/{_ROUND}/prediction")

    assert res.status_code == 200, res.text
    rows = res.json()["predictions"]
    served = rows[0]
    assert served["driver_name"] == "Russell"
    assert served["expected_position"] is None
    assert served["expected_points"] is None
    assert rows[1]["driver_name"] == "Max Verstappen"


def test_session_prediction_accepts_a_null_expected_position(monkeypatch):
    """The session half of the same widening: the three session-prediction
    endpoints serve SessionDriverPrediction rows from the same snapshot, and
    a driver with no stored expected_position gets float('nan') today (the
    routes' own fallback), which serialises to null exactly like the race
    fields above — so its schema has to accept None too or those endpoints
    500 once the snapshot carries literal nulls."""
    snap = {
        "season": _SEASON,
        "session_predictions": {
            "qualifying": {
                str(_ROUND): {
                    "season": _SEASON,
                    "round": _ROUND,
                    "race_name": "Fake Grand Prix",
                    "session_type": "qualifying",
                    "tier": "pre_qualifying",
                    "source": "live",
                    "predictions": [
                        {
                            "driver_id": "norris",
                            "expected_position": None,
                            "p_pole": 0.2,
                            "p_top_3": 0.5,
                            "p_top_10": 0.9,
                        }
                    ],
                }
            }
        },
    }
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: snap)

    res = _client().get(f"/api/races/{_SEASON}/{_ROUND}/qualifying-prediction")

    assert res.status_code == 200, res.text
    assert res.json()["predictions"][0]["expected_position"] is None
