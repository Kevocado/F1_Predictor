import sqlite3

import pandas as pd
import pytest

from f1_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_tracking.db"
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield db_path


def _sim_table():
    return pd.DataFrame(
        [
            {
                "driver_id": "norris",
                "constructor_id": "mclaren",
                "p_win": 0.3,
                "p_podium": 0.6,
                "p_points_finish": 0.9,
                "p_dnf": 0.05,
            },
            {
                "driver_id": "leclerc",
                "constructor_id": "ferrari",
                "p_win": 0.15,
                "p_podium": 0.4,
                "p_points_finish": 0.8,
                "p_dnf": 0.05,
            },
        ]
    )


def test_record_and_get_race_session_prediction():
    n = store.record_session_predictions(
        _sim_table(), 2024, 5, "Fake GP", "race", "post_qualifying", "2024-05-01T00:00:00Z"
    )
    assert n == 8  # 2 drivers x 4 markets (win/podium/points_finish/dnf)

    rows = store.get_session_prediction(2024, 5, "race")
    markets = {r["market"] for r in rows}
    assert markets == {"win", "podium", "points_finish", "dnf"}


def test_record_and_get_qualifying_session_prediction_uses_quali_markets():
    n = store.record_session_predictions(
        _sim_table(), 2024, 5, "Fake GP", "qualifying", "post_practice", "2024-05-01T00:00:00Z"
    )
    assert n == 6  # 2 drivers x 3 markets (pole/top_3/top_10)

    rows = store.get_session_prediction(2024, 5, "qualifying")
    markets = {r["market"] for r in rows}
    assert markets == {"pole", "top_3", "top_10"}
    assert all(r["actual_dnf"] is None for r in rows)


def test_reconcile_session_predictions_fills_actual_outcome():
    store.record_session_predictions(
        _sim_table(), 2024, 5, "Fake GP", "qualifying", "post_practice", "2024-05-01T00:00:00Z"
    )
    results = pd.DataFrame(
        [
            {"season": 2024, "round": 5, "driver_id": "norris", "position": 1, "dnf": False},
            {"season": 2024, "round": 5, "driver_id": "leclerc", "position": 4, "dnf": False},
        ]
    )
    updated = store.reconcile_session_predictions(results, "qualifying")
    assert updated == 6

    rows = store.get_session_prediction(2024, 5, "qualifying")
    pole_rows = {r["driver_id"]: r for r in rows if r["market"] == "pole"}
    assert pole_rows["norris"]["actual_outcome"] == 1
    assert pole_rows["leclerc"]["actual_outcome"] == 0
    assert pole_rows["norris"]["actual_position"] == 1


def test_migration_preserves_existing_race_predictions_rows(tmp_path, monkeypatch):
    # Simulate a pre-migration database: create the OLD race_predictions
    # shape by hand, then verify _connect()'s migration step carries every
    # row over into session_predictions with session_type='race' and
    # actual_finish_position renamed to actual_position.
    db_path = tmp_path / "legacy_tracking.db"
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE race_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL, round INTEGER NOT NULL, race_name TEXT NOT NULL,
            driver_id TEXT NOT NULL, constructor_id TEXT, tier TEXT NOT NULL, market TEXT NOT NULL,
            predicted_prob REAL NOT NULL, session_time TEXT NOT NULL, snapshotted_at TEXT NOT NULL,
            model_trained_at TEXT, resolved INTEGER NOT NULL DEFAULT 0, actual_outcome INTEGER,
            actual_finish_position INTEGER, actual_dnf INTEGER, resolved_at TEXT,
            backfilled INTEGER NOT NULL DEFAULT 0,
            UNIQUE(season, round, driver_id, tier, market)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO race_predictions
            (season, round, race_name, driver_id, constructor_id, tier, market, predicted_prob,
             session_time, snapshotted_at, resolved, actual_outcome, actual_finish_position, actual_dnf)
        VALUES (2023, 1, 'Old GP', 'hamilton', 'mercedes', 'post_qualifying', 'win', 0.2,
                '2023-01-01T00:00:00Z', '2023-01-01T00:00:00Z', 1, 1, 1, 0)
        """
    )
    conn.commit()
    conn.close()

    rows = store.get_session_prediction(2023, 1, "race")
    assert len(rows) == 1
    assert rows[0]["session_type"] == "race"
    assert rows[0]["actual_position"] == 1
    assert rows[0]["driver_id"] == "hamilton"
