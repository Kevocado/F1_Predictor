import pandas as pd

from f1_predictor.tracking import store


def test_reconcile_session_predictions_qualifying_and_sprint_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")

    sim = pd.DataFrame(
        [{"driver_id": "norris", "constructor_id": "mclaren", "p_win": 0.3, "p_podium": 0.6, "p_points_finish": 0.9, "p_dnf": 0.0}]
    )
    store.record_session_predictions(sim, 2024, 1, "Fake GP", "qualifying", "post_practice", "2024-01-01T00:00:00Z")
    store.record_session_predictions(sim, 2024, 1, "Fake GP", "sprint", "post_sprint_qualifying", "2024-01-01T00:00:00Z")

    quali_results = pd.DataFrame([{"season": 2024, "round": 1, "driver_id": "norris", "position": 1, "dnf": False}])
    n_updated = store.reconcile_session_predictions(quali_results, "qualifying")
    assert n_updated == 3  # pole/top_3/top_10

    sprint_rows = store.get_session_prediction(2024, 1, "sprint")
    assert all(r["resolved"] == 0 for r in sprint_rows), "reconciling qualifying must not touch sprint's rows"
