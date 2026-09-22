import pandas as pd

from f1_predictor.features import session_build


def _fake_results(season=2024, rounds=(1, 2, 3)):
    rows = []
    for rnd in rounds:
        for i, driver in enumerate(["a", "b", "c"], start=1):
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "driver_id": driver,
                    "driver_code": driver.upper(),
                    "constructor_id": f"team_{i % 2}",
                    "grid": i,
                    "position": i,
                    "points": 0.0,
                    "laps": 50,
                    "status": "Finished",
                    "dnf": False,
                    "fastest_lap_rank": None,
                }
            )
    return pd.DataFrame(rows)


def _fake_schedule(season=2024, rounds=(1, 2, 3)):
    rows = []
    for rnd in rounds:
        rows.append(
            {
                "season": season,
                "round": rnd,
                "race_name": f"Round {rnd}",
                "circuit_id": "fake_circuit",
                "circuit_name": "Fake Circuit",
                "lat": 0.0,
                "long": 0.0,
                "locality": "Nowhere",
                "country": "Nowhere",
                "race_datetime": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=rnd),
                "is_sprint_weekend": False,
            }
        )
    return pd.DataFrame(rows)


def test_build_session_training_frame_qualifying_excludes_target_leakage(monkeypatch):
    results = _fake_results()
    schedule = _fake_schedule()
    quali = results.rename(columns={"position": "quali_position"}).drop(columns=["grid", "dnf", "status"])

    monkeypatch.setattr(session_build.jolpica, "load_multi_season_results", lambda seasons: results)
    monkeypatch.setattr(session_build.jolpica, "fetch_season_schedule", lambda season: schedule)
    monkeypatch.setattr(session_build.jolpica, "load_season_qualifying", lambda season: quali)
    monkeypatch.setattr(session_build.jolpica, "load_season_sprints", lambda season: pd.DataFrame())
    monkeypatch.setattr(
        session_build.weather_features, "compute_weather_features", lambda schedule_df, **kw: pd.DataFrame(
            columns=["season", "round", "temp_max_c", "precipitation_mm", "wind_max_kph"]
        )
    )

    df, feature_cols = session_build.build_session_training_frame("qualifying", seasons=[2024])

    assert "quali_position" not in feature_cols, "target column must never appear in feature_cols"
    assert "sprint_finish_position" in feature_cols
    assert len(df) == 9  # 3 rounds x 3 drivers
    assert df["sprint_finish_position"].isna().all(), "no sprint data faked -> all NaN, not fabricated"


def test_build_session_training_frame_sprint_uses_grid_as_sprint_quali_position(monkeypatch):
    sprints = _fake_results()
    schedule = _fake_schedule()

    monkeypatch.setattr(session_build.jolpica, "load_multi_season_results", lambda seasons: _fake_results())
    monkeypatch.setattr(session_build.jolpica, "fetch_season_schedule", lambda season: schedule)
    monkeypatch.setattr(session_build.jolpica, "load_season_sprints", lambda season: sprints)
    monkeypatch.setattr(
        session_build.weather_features, "compute_weather_features", lambda schedule_df, **kw: pd.DataFrame(
            columns=["season", "round", "temp_max_c", "precipitation_mm", "wind_max_kph"]
        )
    )

    df, feature_cols = session_build.build_session_training_frame("sprint", seasons=[2024])

    assert "sprint_quali_position" in feature_cols
    assert (df["sprint_quali_position"] == df["grid"]).all()
    assert "position" in df.columns  # the sprint's own finishing position, the target
