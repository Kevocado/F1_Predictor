"""get_race_prediction's caching — confirmed directly (not assumed) while
porting PL_Predictor's PUBLIC_MODE pattern here that this endpoint rebuilt
its whole prediction (elo/team-strength replay for an upcoming race, or a
full model retrain via evaluate/backtest.py for a completed one with no
tracked snapshot) on every single request, uncached. This is exactly the
per-request-rebuild shape that caused PL_Predictor's own OOM before its
equivalent fix. This test locks in that the fix actually avoids the
rebuild on a repeated request."""

import pandas as pd

from f1_predictor.api import routes


def _fake_schedule() -> pd.DataFrame:
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
            }
        ]
    )


def test_upcoming_race_prediction_is_cached_across_requests(monkeypatch):
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule())

    calls = []

    def fake_predict_upcoming_race(season, round_, race_row):
        calls.append((season, round_))
        return _fake_sim(), "pre_qualifying"

    monkeypatch.setattr(routes, "_predict_upcoming_race", fake_predict_upcoming_race)

    routes.get_race_prediction(2026, 1)
    routes.get_race_prediction(2026, 1)

    assert len(calls) == 1, "second request should hit the cache, not rebuild the prediction"
