# tests/test_session_prediction_routes.py
import pandas as pd
import pytest
from fastapi import HTTPException

from f1_predictor.api import routes


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
