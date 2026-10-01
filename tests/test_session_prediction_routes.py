# tests/test_session_prediction_routes.py
import pandas as pd
import pytest
from fastapi import HTTPException

from f1_predictor.api import routes
from f1_predictor.data import jolpica
from f1_predictor.public_snapshot import _session_order_for

# The race-prediction stubs below stand in for `_predict_upcoming_race`, which
# hands back the history window it loaded alongside the prediction. "Everything
# loaded" is the honest value for a stub: nothing was skipped, because nothing
# was fetched. Built per call rather than at import so a run against code
# without the coverage type fails the tests rather than erroring the collection.
def _complete_history():
    return jolpica.HistoryCoverage(
        seasons_requested=(2026, 2025, 2024), seasons_loaded=(2026, 2025, 2024)
    )


def _fake_schedule(is_sprint_weekend: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "round": 5,
                "race_name": "Fake GP",
                "circuit_id": "fake",
                "circuit_name": "Fake Circuit",
                "race_datetime": pd.Timestamp("2099-01-01", tz="UTC"),
                "qualifying_datetime": pd.Timestamp("2099-01-01", tz="UTC") - pd.Timedelta(days=1),
                "sprint_datetime": pd.Timestamp("2099-01-01", tz="UTC") - pd.Timedelta(days=2) if is_sprint_weekend else None,
                "sprint_quali_datetime": pd.Timestamp("2099-01-01", tz="UTC") - pd.Timedelta(days=3) if is_sprint_weekend else None,
                "fp1_datetime": pd.Timestamp("2099-01-01", tz="UTC") - pd.Timedelta(days=4),
                "fp2_datetime": None,
                "fp3_datetime": None,
                "is_sprint_weekend": is_sprint_weekend,
            }
        ]
    )


def _fake_schedule_past(is_sprint_weekend: bool) -> pd.DataFrame:
    """A PAST-dated schedule (unlike _fake_schedule's far-future 2099 dates)
    so _session_is_completed actually evaluates the real completed-session
    (tracked/backtest) path instead of short-circuiting on session_dt > now."""
    return pd.DataFrame(
        [
            {
                "season": 2021,
                "round": 10,
                "race_name": "Past GP",
                "circuit_id": "past",
                "circuit_name": "Past Circuit",
                "race_datetime": pd.Timestamp("2021-07-04", tz="UTC"),
                "qualifying_datetime": pd.Timestamp("2021-07-03", tz="UTC"),
                "sprint_datetime": pd.Timestamp("2021-07-03", tz="UTC") - pd.Timedelta(hours=6) if is_sprint_weekend else None,
                "sprint_quali_datetime": pd.Timestamp("2021-07-02", tz="UTC") if is_sprint_weekend else None,
                "fp1_datetime": pd.Timestamp("2021-07-01", tz="UTC"),
                "fp2_datetime": None,
                "fp3_datetime": None,
                "is_sprint_weekend": is_sprint_weekend,
            }
        ]
    )


def _fake_sim():
    return pd.DataFrame(
        [
            {
                "driver_id": "norris",
                "constructor_id": "mclaren",
                "p_win": 0.3,
                "p_podium": 0.6,
                "p_points_finish": 0.9,
                "p_dnf": 0.0,
                "expected_position": 3.0,
                "expected_points": 10.0,
            }
        ]
    )


def test_qualifying_prediction_endpoint_returns_pole_markets(monkeypatch):
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(False))
    monkeypatch.setattr(routes, "_predict_upcoming_session", lambda season, round_, race_row, session_type: (_fake_sim(), "post_practice"))

    result = routes.get_qualifying_prediction(2024, 5)

    assert result.session_type == "qualifying"
    assert result.predictions[0].p_pole == pytest.approx(0.3)
    assert result.predictions[0].p_win is None


def test_sprint_prediction_endpoint_404s_on_non_sprint_weekend(monkeypatch):
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(False))

    with pytest.raises(HTTPException) as exc_info:
        routes.get_sprint_prediction(2024, 5)
    assert exc_info.value.status_code == 404


def test_sprint_prediction_endpoint_returns_race_shaped_markets_on_sprint_weekend(monkeypatch):
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(True))
    monkeypatch.setattr(routes, "_predict_upcoming_session", lambda season, round_, race_row, session_type: (_fake_sim(), "post_sprint_qualifying"))

    result = routes.get_sprint_prediction(2024, 5)

    assert result.session_type == "sprint"
    assert result.predictions[0].p_win == pytest.approx(0.3)
    assert result.predictions[0].p_pole is None


def test_sprint_prediction_completed_path_no_columns_empty_sprint_results_does_not_keyerror(monkeypatch):
    """Regression test for I1: before a season's first sprint weekend
    completes, jolpica.load_season_sprints (and fetch_sprint_results, for
    a still-pending round) return a DataFrame with NO columns at all
    (matching the real jolpica shape), not just no rows. Indexing such a
    frame by ["round"] raises KeyError. Uses a PAST race_datetime/
    sprint_datetime (unlike every other test in this file, which uses
    far-future 2099 dates or mocks _predict_upcoming_session directly) so
    _session_is_completed actually reaches the real sprint-results check
    instead of short-circuiting on "session hasn't happened yet"."""
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule_past(True))
    # Real jolpica shape for "no results yet": zero rows AND zero columns.
    monkeypatch.setattr(routes.jolpica, "fetch_sprint_results", lambda season, round_, force_refresh=False: pd.DataFrame())
    monkeypatch.setattr(
        routes, "_predict_upcoming_session", lambda season, round_, race_row, session_type: (_fake_sim(), "post_sprint_qualifying")
    )

    result = routes.get_sprint_prediction(2021, 10)

    # With no completed sprint results available, the session must be
    # treated as not-yet-completed, falling through to the live prediction
    # path (via the mocked _predict_upcoming_session) rather than raising.
    assert result.session_type == "sprint"
    assert result.predictions[0].p_win == pytest.approx(0.3)


def test_qualifying_prediction_carries_session_datetime(monkeypatch):
    """session_datetime is populated from the schedule row and parses."""
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(False))
    monkeypatch.setattr(routes, "_predict_upcoming_session", lambda *a: (_fake_sim(), "post_practice"))

    result = routes.get_qualifying_prediction(2024, 5)

    assert result.session_datetime is not None
    assert result.session_datetime.endswith("Z")


def test_sprint_prediction_carries_session_datetime(monkeypatch):
    """Sprint prediction also carries session_datetime."""
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(True))
    monkeypatch.setattr(routes, "_predict_upcoming_session", lambda *a: (_fake_sim(), "post_sprint_qualifying"))

    result = routes.get_sprint_prediction(2024, 5)

    assert result.session_datetime is not None


def test_race_prediction_carries_session_datetime(monkeypatch):
    """Race prediction carries session_datetime from race_datetime."""
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(False))
    monkeypatch.setattr(routes, "_predict_upcoming_race", lambda *a: (_fake_sim(), "pre_weekend", _complete_history()))

    result = routes.get_race_prediction(2024, 5)

    assert result.session_datetime is not None
    assert result.session_datetime.endswith("Z")


def test_race_prediction_missing_session_datetime_validates(monkeypatch):
    """Regression guard: a race prediction with a missing or None
    session_datetime must NOT 500 — the field is nullable. The
    committed snapshot predates the field and has no session_datetime."""
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(False))
    monkeypatch.setattr(routes, "_predict_upcoming_race", lambda *a: (_fake_sim(), "pre_weekend", _complete_history()))

    # Even with the field missing from the source data (simulating
    # an old snapshot dict), validation succeeds because the field
    # is nullable with a None default.
    snap = {"season": 2024, "predictions": {"5": {"season": 2024, "round": 5, "race_name": "Fake GP",
        "tier": "pre_weekend", "source": "live", "predictions": [], "session_datetime": None}}}
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: snap)

    result = routes.get_race_prediction(2024, 5)
    # In PUBLIC_MODE the function returns a dict (FastAPI validates it
    # against response_model). Verify the dict has the field.
    assert isinstance(result, dict)
    assert result.get("session_datetime") is None


def test_race_prediction_without_session_datetime_key_in_snapshot(monkeypatch):
    """Even when the snapshot dict has no session_datetime key at all
    (old committed snapshot), the endpoint still validates and serves
    — the field is optional with a None default on the schema."""
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(False))
    monkeypatch.setattr(routes, "_predict_upcoming_race", lambda *a: (_fake_sim(), "pre_weekend", _complete_history()))

    # Old snapshot dict without session_datetime key
    snap = {"season": 2024, "predictions": {"5": {"season": 2024, "round": 5, "race_name": "Fake GP",
        "tier": "pre_weekend", "source": "live", "predictions": []}}}
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: snap)

    result = routes.get_race_prediction(2024, 5)
    assert isinstance(result, dict)
    # pydantic uses the None default for the missing key
    assert result.get("session_datetime") is None


def test_session_order_helper_sprint_weekend():
    """A sprint weekend returns sprint_qualifying → sprint → qualifying → race."""
    sprint_race = {"is_sprint_weekend": True}
    assert _session_order_for(sprint_race) == ["sprint_qualifying", "sprint", "qualifying", "race"]


def test_session_order_helper_normal_weekend():
    """A normal weekend returns qualifying → race."""
    normal_race = {"is_sprint_weekend": False}
    assert _session_order_for(normal_race) == ["qualifying", "race"]
