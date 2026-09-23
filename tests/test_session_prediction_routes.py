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
