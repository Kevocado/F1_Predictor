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


def test_completed_race_serves_an_earlier_tier_snapshot_that_beat_the_session():
    """The cron may miss the qualifying-to-race gap: a pre-weekend snapshot
    made in time is still the honest pick, and beats a late post-qualifying one."""
    from f1_predictor.api import routes

    store.record_session_predictions(_sim(0.7), 2026, 4, "GP 4", "race", "pre_weekend", FUTURE)
    store.record_session_predictions(_sim(0.2), 2026, 4, "GP 4", "race", "post_qualifying", PAST)

    sim, tier, source = routes._completed_race_prediction(2026, 4)

    assert (tier, source) == ("pre_weekend", "tracked")
    assert sim.set_index("driver_id").loc["norris", "p_win"] == 0.7


def test_completed_sprint_weekend_qualifying_finds_its_post_sprint_snapshot():
    from f1_predictor.api import routes

    store.record_session_predictions(_sim(), 2026, 6, "GP 6", "qualifying", "post_sprint", FUTURE)

    _, tier, source = routes._completed_session_prediction(2026, 6, "qualifying")

    assert (tier, source) == ("post_sprint", "tracked")


def test_a_session_with_any_late_row_is_rebuilt_everywhere():
    """Calibration and the per-race table agree: one late insert makes the
    whole session rebuilt, not half-counted."""
    store.record_session_predictions(_sim(), 2026, 7, "GP 7", "race", "post_qualifying", FUTURE)
    late = pd.DataFrame([{"driver_id": "new_driver", "constructor_id": "haas", "p_win": 0.01, "p_podium": 0.02, "p_points_finish": 0.1, "p_dnf": 0.1}])
    with store._connect() as conn:
        conn.execute("UPDATE session_predictions SET session_time = ? WHERE round = 7", (PAST,))
        conn.execute("UPDATE session_predictions SET snapshotted_at = ? WHERE round = 7", ("2019-12-31T00:00:00+00:00",))
    store.record_session_predictions(late, 2026, 7, "GP 7", "race", "post_qualifying", PAST)
    store.reconcile_session_predictions(pd.concat([_results(7), pd.DataFrame([{"season": 2026, "round": 7, "driver_id": "new_driver", "position": 20, "dnf": False}])]), "race")

    record = store.get_session_track_record(session_type="race")

    assert record["n_resolved"] == 0
    assert record["n_rebuilt_sessions"] == 1


def test_a_stored_tracked_label_with_nothing_to_verify_it_is_downgraded():
    from f1_predictor.api import routes

    out = routes.honest_source(2026, 9, "race", {"source": "tracked", "tier": "post_qualifying", "predictions": []})

    assert out["source"] == "rebuilt"


def test_public_mode_serving_relabels_a_stale_snapshot(monkeypatch):
    from f1_predictor.api import routes

    _record_and_resolve(2, PAST)
    snap = {"season": 2026, "predictions": {"2": {"season": 2026, "round": 2, "race_name": "GP 2", "tier": "post_qualifying", "source": "tracked", "predictions": []}}}
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_public_snapshot", lambda: snap)

    assert routes.get_race_prediction(2026, 2)["source"] == "rebuilt"
