"""/facts must read the payload the real routes actually produce.

`facts.py` read `prediction["drivers"]` with per-driver `name`/`grid`/`win`/
`podium`/`points`/`dnf`. The routes return `RacePredictionResponse`, whose
per-driver list is **`predictions`** and whose fields are `p_win`/`p_podium`/
`p_points_finish`/`p_dnf`/`expected_position` — and no `name`, no `grid`, no
`session_time`, no `circuit`, no `weather`.

So every read missed. In production the panel rendered "no pick yet" for a race
the model has a real pick for, with an empty `starts_at` and no context.

These build the payload with the app's OWN schema class and drive the real
route, so they cannot drift from what `routes.py` emits. That is the gap the
hand-shaped fixture in test_facts.py could not see: its fixture was shaped like
what facts.py wanted, not like what the API returns.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from f1_predictor.api import facts as facts_mod
from f1_predictor.api.schemas import DriverPrediction, RacePredictionResponse


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(facts_mod.router)
    return TestClient(app)


def _real_prediction(source: str = "tracked") -> dict:
    """Exactly what routes.py builds, via the response model."""
    return RacePredictionResponse(
        season=2026, round=12, race_name="Italian Grand Prix",
        tier="post_qualifying", source=source,
        predictions=[
            # driver_name is required since the API grew the field: the
            # route fills _title_case(driver_id), so the fixture mirrors
            # that (the title-cased id) rather than a hand-invented name.
            DriverPrediction(driver_id="max_verstappen", driver_name="Max Verstappen",
                             p_win=0.31, p_podium=0.72,
                             p_points_finish=0.66, p_dnf=0.12,
                             expected_position=1.4, expected_points=9.1),
            DriverPrediction(driver_id="lando_norris", driver_name="Lando Norris",
                             p_win=0.27, p_podium=0.70,
                             p_points_finish=0.64, p_dnf=0.14,
                             expected_position=3.9, expected_points=8.4),
            DriverPrediction(driver_id="george_russell", driver_name="George Russell",
                             p_win=0.19, p_podium=0.61,
                             p_points_finish=0.58, p_dnf=0.16,
                             expected_position=2.2, expected_points=7.9),
        ],
    ).model_dump()


@pytest.fixture
def live_prediction(monkeypatch):
    """The route reads the live prediction and the stored pre-session rows."""
    monkeypatch.setattr(facts_mod, "_current_prediction",
                        lambda season, round_, session: _real_prediction())
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda season, round_, session: [])
    return _real_prediction()


def test_the_real_race_payload_yields_a_pick(client, live_prediction):
    """The regression. Against the real schema the bundle came back
    `pick: null` / `pick_timing: "none"`, so the panel said "no pick yet" for a
    race the model has a 31% pick for."""
    res = client.get("/facts/2026-12-race")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["pick"] is not None, "the real payload must produce a pick"
    assert body["pick"]["prob"] == pytest.approx(0.31)
    assert body["pick_timing"] in ("pre_kickoff", "none")


def test_the_real_payload_keeps_the_story_markets(client, live_prediction):
    """Podium / points / DNF are three of the spec'd F1 markets. All were
    dropped because the field names did not match."""
    body = client.get("/facts/2026-12-race").json()
    assert body["drivers"], f"the real payload produced no drivers: {body}"
    top = body["drivers"][0]
    for field in ("win", "podium", "points", "dnf"):
        assert top.get(field) is not None, f"driver row lost {field!r}: {top}"


def test_grid_is_not_the_models_expected_finishing_position(client, live_prediction):
    """`expected_position` is a Monte-Carlo mean finishing order — a model
    output, and fractional (1.4, 3.9, 2.2 in the fixture above).

    Publishing it as `grid` told readers a driver "starts from 1.4", and
    reported George Russell (actual grid 9) as 2.2. The spec ties the whole F1
    narrative to the grid, and the biggest-mover selection compares grid with
    predicted rank — so a phantom grid corrupts which driver gets the "who can
    move up" story, and the panel tells a reader a false fact about the grid.
    """
    body = client.get("/facts/2026-12-race").json()
    for driver in body["drivers"]:
        grid = driver.get("grid")
        if grid is not None:
            assert float(grid).is_integer(), (
                f"{driver.get('name')} has grid {grid!r}: a grid slot is a whole "
                f"number, so a fractional value here is expected_position"
            )
            assert float(grid) >= 1, "a grid slot is at least 1"
