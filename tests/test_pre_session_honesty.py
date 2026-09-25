"""Only predictions snapshotted before their session count toward the track
record. A row written after the session (a seeded or back-filled snapshot)
is still shown, labelled rebuilt, and never judged."""
import pandas as pd
import pytest

from f1_predictor.tracking import store

FUTURE = "2099-01-01T13:00:00+00:00"  # snapshot now < session: made before
PAST = "2020-01-01T13:00:00+00:00"  # snapshot now > session: rebuilt


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")


def _sim(winner_prob=0.6):
    return pd.DataFrame(
        [
            {"driver_id": "norris", "constructor_id": "mclaren", "p_win": winner_prob, "p_podium": 0.9, "p_points_finish": 0.95, "p_dnf": 0.05},
            {"driver_id": "leclerc", "constructor_id": "ferrari", "p_win": 1 - winner_prob, "p_podium": 0.8, "p_points_finish": 0.9, "p_dnf": 0.05},
        ]
    )


def _results(round_):
    return pd.DataFrame(
        [
            {"season": 2026, "round": round_, "driver_id": "norris", "position": 1, "dnf": False},
            {"season": 2026, "round": round_, "driver_id": "leclerc", "position": 2, "dnf": False},
        ]
    )


def _record_and_resolve(round_, session_time):
    store.record_session_predictions(_sim(), 2026, round_, f"GP {round_}", "race", "post_qualifying", session_time)
    store.reconcile_session_predictions(_results(round_), "race")


def test_made_before_session_parses_times_and_never_counts_garbage():
    assert store.made_before_session("2026-03-07T10:00:00+00:00", "2026-03-08 04:00:00+00:00") is True
    assert store.made_before_session("2026-08-23T22:57:33+00:00", "2026-03-08 04:00:00+00:00") is False
    assert store.made_before_session("2026-03-08T04:00:00Z", "2026-03-08T04:00:00Z") is False
    assert store.made_before_session("nonsense", "2026-03-08T04:00:00Z") is False


def test_track_record_counts_only_pre_session_snapshots():
    _record_and_resolve(1, FUTURE)
    _record_and_resolve(2, PAST)

    record = store.get_session_track_record(session_type="race")

    win = next(r for r in record["by_market"] if r["market"] == "win")
    assert win["n"] == 2  # two drivers from round 1 only
    assert record["n_resolved"] == 8  # round 1: 2 drivers x 4 markets
    assert record["n_rebuilt_sessions"] == 1


def test_race_accuracy_flags_rebuilt_sessions():
    _record_and_resolve(1, FUTURE)
    _record_and_resolve(2, PAST)

    by_round = {r["round"]: r for r in store.get_session_accuracy(session_type="race")}

    assert by_round[1]["rebuilt"] is False
    assert by_round[2]["rebuilt"] is True


def test_completed_race_serves_a_late_snapshot_as_rebuilt_not_tracked():
    from f1_predictor.api import routes

    _record_and_resolve(2, PAST)

    _, _, source = routes._completed_race_prediction(2026, 2)

    assert source == "rebuilt"


def test_completed_race_serves_a_pre_race_snapshot_as_tracked():
    from f1_predictor.api import routes

    _record_and_resolve(1, FUTURE)

    _, _, source = routes._completed_race_prediction(2026, 1)

    assert source == "tracked"


def test_stored_late_prediction_is_relabelled_rebuilt():
    """Public snapshots store predictions verbatim (and build_snapshot reuses
    older rounds); one saved as 'tracked' before this fix must not keep it,
    whether it is reused at build time or served as-is."""
    from f1_predictor.api import routes

    _record_and_resolve(2, PAST)
    _record_and_resolve(1, FUTURE)
    stale = {"source": "tracked", "tier": "post_qualifying", "predictions": []}

    assert routes.honest_source(2026, 2, "race", dict(stale))["source"] == "rebuilt"
    assert routes.honest_source(2026, 1, "race", dict(stale))["source"] == "tracked"
    live = {"source": "live", "predictions": []}
    assert routes.honest_source(2026, 3, "race", dict(live))["source"] == "live"
