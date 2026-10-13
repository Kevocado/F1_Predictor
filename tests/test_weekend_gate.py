"""weekend_gate: the cheap "did a new session result land?" decision that lets
the every-20-minute weekend cron be a no-op. Offline: jolpica reads come from
the seeded cache (tests/conftest.py), so the real 2026 schedule/results shapes
are used."""

import json

import pandas as pd

from f1_predictor import weekend_gate as wg


def test_new_results_is_set_difference():
    assert wg.new_session_results({"2026:14:race"}, {"2026:14:race", "2026:15:qualifying"}) == {"2026:15:qualifying"}


def test_nothing_new_when_seen_covers_available():
    assert wg.new_session_results({"2026:14:race", "2026:14:qualifying"}, {"2026:14:race"}) == set()


def test_everything_new_on_first_run():
    assert wg.new_session_results(set(), {"2026:1:race"}) == {"2026:1:race"}


def test_available_keys_use_real_shapes(monkeypatch):
    # Round 14 (race 2026-09-13): real schedule + real cached race results.
    now = pd.Timestamp("2026-09-14T06:00:00+00:00")
    cached_read = wg.jolpica.fetch_race_results  # force_refresh would hit the network; read the seeded cache
    monkeypatch.setattr(wg.jolpica, "fetch_race_results", lambda s, r, force_refresh=False: cached_read(s, r))
    monkeypatch.setattr(wg.jolpica, "fetch_qualifying", lambda s, r, force_refresh=False: pd.DataFrame())
    keys = wg.available_session_keys(2026, now)
    assert keys == {"2026:14:race"}  # quali stubbed empty; no sprint that weekend


def test_rounds_outside_the_weekend_window_are_not_fetched(monkeypatch):
    now = pd.Timestamp("2026-09-30T06:00:00+00:00")  # round 14 raced 17 days earlier
    rounds = set()
    for fn in ("fetch_race_results", "fetch_qualifying", "fetch_sprint_results"):
        monkeypatch.setattr(wg.jolpica, fn, lambda s, r, force_refresh=False: rounds.add(r) or pd.DataFrame())
    assert wg.available_session_keys(2026, now) == set()
    assert 14 not in rounds and rounds  # round 16 (2026-10-04) is in the window


def test_check_then_record_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(wg, "SEEN_PATH", tmp_path / "seen.json")
    monkeypatch.setattr(wg, "PENDING_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(wg, "available_session_keys", lambda season, now=None: {"2026:14:race"})
    assert wg.check(2026) is True            # first run: nothing seen
    assert wg.check(2026) is True            # still not recorded (build may have failed)
    wg.record()
    assert json.loads((tmp_path / "seen.json").read_text()) == ["2026:14:race"]
    assert wg.check(2026) is False           # nothing new -> no-op
    monkeypatch.setattr(wg, "available_session_keys", lambda season, now=None: {"2026:14:race", "2026:15:qualifying"})
    assert wg.check(2026) is True
