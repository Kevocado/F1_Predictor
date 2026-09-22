import pandas as pd
import pytest

from f1_predictor.evaluate import backtest


def test_backtest_session_race_delegates_to_backtest_race(monkeypatch):
    calls = []

    def fake_backtest_race(season, round_, seasons=None):
        calls.append((season, round_, seasons))
        return pd.DataFrame([{"driver_id": "a", "p_win": 1.0}])

    monkeypatch.setattr(backtest, "backtest_race", fake_backtest_race)
    result = backtest.backtest_session(2024, 5, session_type="race")

    assert calls == [(2024, 5, None)]
    assert result.iloc[0]["driver_id"] == "a"


def test_backtest_session_qualifying_no_lookahead(monkeypatch):
    from f1_predictor.models import session_outcome

    rows = []
    for rnd in range(1, 4):
        for i, driver in enumerate(["a", "b", "c"], start=1):
            rows.append(
                {
                    "season": 2024,
                    "round": rnd,
                    "driver_id": driver,
                    "constructor_id": f"team_{i % 2}",
                    "quali_position": i,
                    "elo_pre_race": 1600 - i * 10,
                    "team_strength_pre_race": 1500.0,
                }
            )
    fake_df = pd.DataFrame(rows)
    feature_cols = ["elo_pre_race", "team_strength_pre_race"]

    monkeypatch.setattr(
        backtest.session_build, "build_session_training_frame", lambda session_type, seasons=None: (fake_df, feature_cols)
    )

    result = backtest.backtest_session(2024, 3, session_type="qualifying")

    assert set(result["driver_id"]) == {"a", "b", "c"}
    assert "position" in result.columns  # actual quali_position, renamed for comparison


def test_backtest_session_raises_when_no_data_for_round(monkeypatch):
    fake_df = pd.DataFrame(
        [{"season": 2024, "round": 1, "driver_id": "a", "constructor_id": "t", "quali_position": 1, "elo_pre_race": 1600.0}]
    )
    monkeypatch.setattr(
        backtest.session_build, "build_session_training_frame", lambda session_type, seasons=None: (fake_df, ["elo_pre_race"])
    )
    with pytest.raises(ValueError, match="No qualifying data found"):
        backtest.backtest_session(2024, 99, session_type="qualifying")
