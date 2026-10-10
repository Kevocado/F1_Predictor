"""context.form_rows: recent form / quali pace / track history, built from the
committed jolpica cache (real frames, offline -- any network call fails)."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from f1_predictor.api import facts as facts_mod
from f1_predictor.api.schemas import DriverPrediction, RacePredictionResponse
from f1_predictor.data import jolpica

SEASON, ROUND = 2025, 20

#: The display name each constructor id must render as. These are the names
#: the sport uses, not the ids: the id is jolpica's `constructorId` (verified
#: against the committed 2025 results cache in `test_constructor_names_cover_
#: every_id_in_the_real_cache`), and the display name is what a reader expects
#: to see. `mclaren`, `rb` and `sauber` are the three the title-cased id got
#: wrong; the rest exist so the map is the whole current grid, not a patch list.
EXPECTED_TEAM_NAMES = {
    "alpine": "Alpine",
    "aston_martin": "Aston Martin",
    "ferrari": "Ferrari",
    "haas": "Haas",
    "mclaren": "McLaren",
    "mercedes": "Mercedes",
    "rb": "Racing Bulls",
    "red_bull": "Red Bull",
    "sauber": "Sauber",
    "williams": "Williams",
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network call in test")
    monkeypatch.setattr(jolpica, "_get", boom)


@pytest.fixture
def frames():
    results, schedule, _ = jolpica.load_history(SEASON)
    return results, schedule, jolpica.load_season_qualifying(SEASON)


def _field(results):
    race = results[(results.season == SEASON) & (results["round"] == ROUND)].sort_values("grid")
    return [{"driver_id": r.driver_id, "constructor_id": r.constructor_id,
             "name": r.driver_id, "win": 0.5 / (i + 1)} for i, r in enumerate(race.itertuples())]


def test_rows_use_only_prior_sessions_and_rank_in_field(frames):
    results, schedule, quali = frames
    field = _field(results)
    rows = facts_mod._form_rows(field, SEASON, ROUND, results, schedule, quali)
    top = field[0]["driver_id"]
    by_id = {r["id"]: r for r in rows}
    form = by_id[f"driver:{top}:form"]
    prior = results[(results.season == SEASON) & (results["round"] < ROUND) & (results.driver_id == top)].tail(5)
    expect = prior["position"].fillna(20).mean()
    assert form["value"] == f"avg finish P{expect:.1f} (last {len(prior)})"
    assert 1 <= form["rank"] <= form["n"] <= len(field)
    assert {"form", "quali", "track"} <= {r["id"].split(":")[2] for r in rows}
    assert any(r["id"].startswith("team:") for r in rows)
    for r in rows:
        assert set(r) == {"id", "subject", "label", "value", "rank", "n"}
        assert isinstance(r["value"], str) and r["value"]


def test_no_history_no_rows(frames):
    results, schedule, quali = frames
    assert facts_mod._form_rows(_field(results), 2019, 1, results, schedule, quali) == []


def test_facts_route_emits_form_rows(frames, monkeypatch):
    results, schedule, quali = frames
    field = _field(results)
    pred = RacePredictionResponse(
        season=SEASON, round=ROUND, race_name="Mexico City Grand Prix", tier="post_qualifying",
        source="tracked", session_datetime="2099-09-06T13:00:00Z",
        predictions=[DriverPrediction(driver_id=f["driver_id"], driver_name=f["driver_id"],
                                      constructor_id=f["constructor_id"], p_win=f["win"],
                                      p_podium=0.5, p_points_finish=0.5, p_dnf=0.1) for f in field],
    ).model_dump()
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda *a: pred)
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda *a, **k: [])
    monkeypatch.setattr(facts_mod, "_contributors_for", lambda *a: {})
    monkeypatch.setattr(facts_mod, "_record", lambda: None)
    app = FastAPI(); app.include_router(facts_mod.router)
    body = TestClient(app).get(f"/facts/{SEASON}-{ROUND}-race").json()
    rows = body["context"]["form_rows"]
    assert rows and all(r["subject"] for r in rows)
    print(*rows, sep="\n")


def test_constructor_names_cover_every_id_in_the_real_cache(frames):
    """The committed 2025 jolpica cache for the CURRENT SEASON contains exactly
    these constructor ids (the current grid).

    Historical seasons bring in retired constructors (alfa, alphatauri); this
    test checks only the constructors active in the 2025 season itself.
    """
    from f1_predictor.data import jolpica
    results, _, _ = frames
    # Only 2025-season results (not historical)
    current_season = results[results["season"] == SEASON]
    seen = {r.constructor_id for r in current_season.itertuples()}
    assert set(EXPECTED_TEAM_NAMES) == seen, (
        f"2025 cache has new/removed constructor ids: got {sorted(seen)}, expected {sorted(EXPECTED_TEAM_NAMES)}"
    )


def test_form_rows_constructor_subject_uses_display_name_not_id(frames):
    results, schedule, quali = frames
    field = _field(results)
    # seed team form so the team rows exist
    rows = facts_mod._form_rows(field, SEASON, ROUND, results, schedule, quali)
    team_rows = [r for r in rows if r["id"].startswith("team:")]
    assert team_rows, "at least one team row expected"
    for r in team_rows:
        assert r["subject"] == EXPECTED_TEAM_NAMES[r["id"].split(":")[1]], (
            f"{r['id']} subject {r['subject']!r} != expected {EXPECTED_TEAM_NAMES[r['id'].split(':')[1]]!r}"
        )
