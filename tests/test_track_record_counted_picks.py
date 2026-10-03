"""The track record counts every recorded pick, and says when each was made.

Kevin, 2026-10-01, in `predictor-hub`
`docs/superpowers/specs/2026-10-01-track-record-counts-every-pick.md` (merged as
predictor-hub #66), verbatim:

  "i dont really care about picks made after kickoff because im always re
   running the models ... with every model change it will stop tracking ...
   make it that whats recorded remains recorded and then just use every
   prediction we make for the track record stuff."

**This repo's word is SESSION, not kickoff.** A grand prix is a weekend: a
session has a start, and the decision doc says the disclosure label and its
comparison are about that start. So the flag is `made_before_session`, the
secondary figure is `pre_session`, and nothing here borrows kickoff language to
describe a session.

Before the reversal, `get_session_track_record` judged a session only if EVERY
one of its rows was snapshotted before it started; a session with any late row
was dropped whole and reported as `n_rebuilt_sessions`. That is the "stops
tracking" failure, and on this repo's own `data/tracking.db` it was severe:
measured at blob `0071bc32` on 2026-10-03 (a DATED figure -- the store is
refreshed by an automated commit, so these totals have moved since and must be
re-measured before use), 12 of 17 recorded sessions were dropped and 1,056 of
1,584 graded picks reached no published figure, with one market
(`race`/`post_qualifying`/`dnf`) published at n=22 when 286 were recorded.

What replaced it:

  1. Recorded stays recorded. `INSERT OR IGNORE`, nothing overwritten.
  2. One counted pick per key, the EARLIEST recorded. A later rerun is history:
     it neither replaces the counted pick nor counts a second time -- otherwise
     re-running until the model was right would be free.
  3. The headline counts every counted pick, whenever it was made. `pre_session`
     sits beside it with its own n, its own per-market rows and its own
     reconciliation.
  4. `made_before_session` is DERIVED on every read from the pick's own
     `snapshotted_at` against its own `session_time`, compared as UTC INSTANTS,
     failing closed to False. Never stored, never read from the `backfilled`
     column that exists and is not consulted.

**The counting key is (session, driver, market)** -- `session_predictions`' own
UNIQUE key -- because a driver pick is one pick per DRIVER per session. F1's
analogue of PL's player-props trap: collapsing to session level would delete
every driver but one. Pinned by
`test_the_counted_key_carries_the_driver_so_two_drivers_on_one_market_both_survive`.

**`get_session_accuracy` is deliberately NOT relaxed** (see
tests/test_pre_session_honesty.py): it is the per-session graded distribution
the page reads as "snapshotted before the session", not the headline, and its
all-or-nothing rule is the thing that keeps the by-race table agreeing with
itself. Documented at its own definition.
"""
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import pytest

from f1_predictor.tracking import store

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Never the shipped `data/tracking.db`."""
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")


def _sim(**probs):
    norris, leclerc = probs.get("norris", 0.6), probs.get("leclerc", 0.4)
    return pd.DataFrame([
        {"driver_id": "norris", "constructor_id": "mclaren", "p_win": norris,
         "p_podium": 0.9, "p_points_finish": 0.95, "p_dnf": 0.05},
        {"driver_id": "leclerc", "constructor_id": "ferrari", "p_win": leclerc,
         "p_podium": 0.8, "p_points_finish": 0.9, "p_dnf": 0.05},
    ])


def _results(round_, norris_position=1):
    return pd.DataFrame([
        {"season": 2026, "round": round_, "driver_id": "norris", "position": norris_position, "dnf": False},
        {"season": 2026, "round": round_, "driver_id": "leclerc", "position": 2, "dnf": False},
    ])


def _record_and_resolve(round_, session_time, **probs):
    store.record_session_predictions(
        _sim(**probs), 2026, round_, f"GP {round_}", "race", "post_qualifying", session_time,
    )
    store.reconcile_session_predictions(_results(round_), "race")


def _market(record, market="win", session_type="race", tier="post_qualifying"):
    return next(r for r in record["by_market"]
                if r["market"] == market and r["session_type"] == session_type and r["tier"] == tier)


# --- rules 3 and 4: headline over all counted picks, pre-session beside it ----

def test_the_headline_counts_a_session_whose_only_snapshot_was_taken_after_it_ran():
    """The swap, and the shape that made this rule necessary.

    The old rule, in `tests/test_pre_session_honesty.py::test_track_record_
    counts_only_pre_session_snapshots`, asserted `win["n"] == 2` and
    `n_resolved == 8` for two sessions -- and counted the second only because
    the FIRST was pre-session. Round 2, recorded entirely after it ran, was
    dropped whole: `n_rebuilt_sessions == 1` and no market row for it at all.

    Now both sessions count: `n == 4` on the win market and `n_resolved == 16`,
    and `pre_session` beside it is exactly the old figure -- 2 picks, the one
    session that was snapshotted in time. Neither figure replaces the other,
    and `n_rebuilt_sessions` becomes the reconciliation rather than a count of
    what was withheld.
    """
    _record_and_resolve(1, "2099-01-01T13:00:00+00:00", norris=0.6)
    _record_and_resolve(2, "2020-01-01T13:00:00+00:00", norris=0.9)

    record = store.get_session_track_record(session_type="race")

    win = _market(record)
    assert win["n"] == 4, "a recorded session was dropped from the headline"
    assert record["n_resolved"] == 16, "2 sessions x 2 drivers x 4 markets"
    # The secondary figure: the pre-session subset, with its own n.
    assert record["n_pre_session"] == 8
    pre_win = _market(record["pre_session"])
    assert pre_win["n"] == 2
    # The reconciliation between the two figures. Same field name as before,
    # new meaning: it no longer counts sessions that were withheld.
    assert record["n_rebuilt_sessions"] == 1
    assert record["n_resolved"] == record["n_pre_session"] + record["n_post_session_picks"]
    assert record["n_post_session_picks"] == 8


def test_the_counted_pick_is_the_earliest_recorded_one_per_session_driver_and_market():
    """Rule 2, and the write path cannot be relied on to enforce it.

    `record_session_predictions` is INSERT OR IGNORE on the table's UNIQUE key,
    so a rerun of the same session does not land -- and the read side implements
    the rule anyway, because a rule enforced only by a primary key stops being
    enforced the moment the key changes, and the failure when it does is
    invisible: the Brier score just improves, silently, and a rerun of the
    model gets to grade a second time.

    Driven on the helper directly, because the table cannot hold the duplicate.
    """
    rows = pd.DataFrame([
        {"season": 2026, "round": 3, "session_type": "race", "tier": "post_qualifying",
         "driver_id": "norris", "market": "win", "predicted_prob": 0.10,
         "actual_outcome": 1, "snapshotted_at": "2026-03-07T10:00:00+00:00", "session_time": "2026-03-08T15:00:00+00:00"},
        # Same instant as the row above, written as +09:00. A tie; first in wins.
        {"season": 2026, "round": 3, "session_type": "race", "tier": "post_qualifying",
         "driver_id": "norris", "market": "win", "predicted_prob": 0.20,
         "actual_outcome": 1, "snapshotted_at": "2026-03-07T19:00:00+09:00", "session_time": "2026-03-08T15:00:00+00:00"},
        # 09:30Z -- EARLIER than both, and listed last. This is the counted pick.
        {"season": 2026, "round": 3, "session_type": "race", "tier": "post_qualifying",
         "driver_id": "norris", "market": "win", "predicted_prob": 0.05,
         "actual_outcome": 1, "snapshotted_at": "2026-03-07T18:30:00+09:00", "session_time": "2026-03-08T15:00:00+00:00"},
        # A different market on the same driver and session is a DIFFERENT pick.
        {"season": 2026, "round": 3, "session_type": "race", "tier": "post_qualifying",
         "driver_id": "norris", "market": "podium", "predicted_prob": 0.7,
         "actual_outcome": 1, "snapshotted_at": "2026-03-07T11:00:00+00:00", "session_time": "2026-03-08T15:00:00+00:00"},
        # A different TIER is a different snapshot of the same session, and the
        # published by_market rows are grouped by tier, so it cannot collapse in.
        {"season": 2026, "round": 3, "session_type": "race", "tier": "pre_weekend",
         "driver_id": "norris", "market": "win", "predicted_prob": 0.8,
         "actual_outcome": 1, "snapshotted_at": "2026-03-06T11:00:00+00:00", "session_time": "2026-03-08T15:00:00+00:00"},
        # A stamp that cannot be read: kept, but it cannot claim to be earliest.
        {"season": 2026, "round": 4, "session_type": "race", "tier": "post_qualifying",
         "driver_id": "norris", "market": "win", "predicted_prob": 0.9,
         "actual_outcome": 1, "snapshotted_at": "not a time", "session_time": "2026-03-08T15:00:00+00:00"},
        {"season": 2026, "round": 4, "session_type": "race", "tier": "post_qualifying",
         "driver_id": "norris", "market": "win", "predicted_prob": 0.4,
         "actual_outcome": 1, "snapshotted_at": "2026-03-08T09:00:00+00:00", "session_time": "2026-03-08T15:00:00+00:00"},
    ])

    counted = store._earliest_recorded(rows).to_dict("records")

    # One row per (season, round, session_type, tier, driver, market). The
    # winner per key is the earliest UTC INSTANT even where a string sort would
    # have ranked it last: the 0.05 row is listed THIRD in the input and wins.
    by_key = {(r["round"], r["tier"], r["market"]): r["predicted_prob"] for r in counted}
    assert by_key == {
        (3, "post_qualifying", "win"): 0.05,     # 09:30Z, listed last, wins
        (3, "post_qualifying", "podium"): 0.7,   # a different market is a different pick
        (3, "pre_weekend", "win"): 0.8,         # a different tier cannot be displaced
        (4, "post_qualifying", "win"): 0.4,      # beats the unreadable stamp
    }, "the counted pick is not the earliest recorded one per key"
    # The tie inside key (3, post_qualifying, win) went to the row listed first,
    # stably: 0.05 at 09:30Z, then 0.10 and 0.20 both at 10:00Z.
    assert len(counted) == 4, "a rerun was counted a second time"


def test_the_counted_key_carries_the_driver_so_two_drivers_on_one_market_both_survive():
    """The F1 form of PL's player-props trap, and the reason the key is the
    table's own UNIQUE key rather than (session, market).

    The decision doc's "(game, market)" applied to a per-driver market would
    keep one row per session and throw the rest away. A `win` call is one call
    per DRIVER per session: 20 drivers, 20 picks. Pinned directly, because the
    collapse is silent -- the Brier score would still be a number, just computed
    over one driver.
    """
    _record_and_resolve(1, "2099-01-01T13:00:00+00:00")

    rows = pd.DataFrame([
        {"season": 2026, "round": 1, "session_type": "race", "tier": "post_qualifying",
         "driver_id": driver, "market": "win", "predicted_prob": 0.5, "actual_outcome": 0,
         "snapshotted_at": "2026-01-01T00:00:00+00:00", "session_time": "2099-01-01T13:00:00+00:00"}
        for driver in ("norris", "leclerc", "russell", "max_verstappen")
    ])

    counted = store._earliest_recorded(rows).to_dict("records")

    assert sorted(r["driver_id"] for r in counted) == [
        "leclerc", "max_verstappen", "norris", "russell",
    ], "collapsing to (session, market) deleted three drivers' records"


def test_every_counted_pick_carries_its_own_timestamp_and_the_pre_session_label():
    """Rule 4, asserted on the published rows rather than the aggregate.

    A per-pick list carrying the label without the timestamp could not be
    audited by a reader, which is the whole point of disclosing per pick.
    """
    _record_and_resolve(1, "2099-01-01T13:00:00+00:00")
    _record_and_resolve(2, "2020-01-01T13:00:00+00:00")

    rows = store.get_session_track_record(session_type="race")["per_pick"]

    assert rows, "the record published no per-pick rows to disclose anything on"
    for row in rows:
        assert row["snapshotted_at"], "a pick row was published without its own timestamp"
        assert row["session_time"], "a pick row was published without the start it is compared to"
        assert isinstance(row["made_before_session"], bool)
        assert row["counted"] is True
    pre = [r for r in rows if r["round"] == 1]
    post = [r for r in rows if r["round"] == 2]
    assert pre and all(r["made_before_session"] is True for r in pre)
    assert post and all(r["made_before_session"] is False for r in post)


def test_a_pick_whose_timing_cannot_be_proven_counts_and_is_never_labelled_pre_session():
    """Disclosure fails closed; exclusion no longer exists.

    A `snapshotted_at` that does not parse is still a recorded pick, so it
    counts. It may never be labelled `made_before_session`: the rule is never
    to backfill a `true` the timestamps do not prove. And the `backfilled`
    column exists in this table and is NOT consulted -- a stored flag is exactly
    what rule 4 rules out.
    """
    with store._connect() as conn:
        conn.execute(
            """INSERT INTO session_predictions
                 (season, round, race_name, session_type, driver_id, constructor_id, tier, market,
                  predicted_prob, session_time, snapshotted_at, resolved, actual_outcome, backfilled)
               VALUES (2026, 1, 'GP 1', 'race', 'norris', 'mclaren', 'post_qualifying', 'win',
                       0.9, '2026-03-08T15:00:00+00:00', 'not a time', 1, 1, 0)"""
        )
        # And a row whose `backfilled` flag says 1 while its own timestamps
        # prove it was made before the session: the flag is not the authority.
        conn.execute(
            """INSERT INTO session_predictions
                 (season, round, race_name, session_type, driver_id, constructor_id, tier, market,
                  predicted_prob, session_time, snapshotted_at, resolved, actual_outcome, backfilled)
               VALUES (2026, 1, 'GP 1', 'race', 'leclerc', 'ferrari', 'post_qualifying', 'win',
                       0.1, '2026-03-08T15:00:00+00:00', '2026-03-07T10:00:00+00:00', 1, 0, 1)"""
        )

    record = store.get_session_track_record(session_type="race")

    assert _market(record)["n"] == 2, "a recorded pick was dropped from the headline"
    assert record["n_pre_session"] == 1, "the unreadable stamp was labelled pre-session"
    labels = {r["driver_id"]: r["made_before_session"] for r in record["per_pick"]}
    assert labels == {"norris": False, "leclerc": True}, labels


# --- `made_before_session` is derived from UTC instants, not wall clocks -----

def test_made_before_session_is_derived_from_utc_instants_not_wall_clock_strings():
    """Two straddles where comparing the strings gives the opposite answer.

    A `session_time` can arrive with any offset, so the wall clocks on the two
    sides can read in one order while the instants read in the other. Each case
    asserts the STRING order first, so the test cannot quietly stop straddling:
    if the offsets were ever normalised upstream, the precondition fails and
    says so instead of the assertion passing for the wrong reason.

    Case A: `23:30:00+09:00` is 14:30Z against a `15:00:00+00:00` session --
            before. The strings read "23:30" against "15:00", which is after.
    Case B: `23:30:00-11:00` is 10:30Z against a `23:45:00+14:00` session
            (09:45Z) -- after. The strings read "23:30" against "23:45", which
            is before. +14:00 against -11:00 is the widest gap the zones allow,
            so this pair is forced rather than picked.
    """
    early = "2026-09-12T15:00:00+00:00"
    late = "2026-09-12T23:45:00+14:00"

    # The wall-clock reading, stated as a precondition.
    assert "2026-09-12T23:30:00+09:00" > early  # would say "after"
    assert "2026-09-12T23:30:00-11:00" < late  # would say "before"

    assert store.made_before_session("2026-09-12T23:30:00+09:00", early) is True
    assert store.made_before_session("2026-09-12T23:30:00-11:00", late) is False


def test_made_before_session_ignores_the_machine_timezone():
    """The derivation reads the offsets in the data, never the host's clock.

    The same two instants under four host zones must agree; a derivation that
    dropped the offset and read the machine's zone would not.
    """
    stamp, session = "2026-09-12T23:30:00+09:00", "2026-09-12T15:00:00+00:00"
    original = os.environ.get("TZ")
    try:
        answers = []
        for zone in ("UTC", "America/Los_Angeles", "Asia/Tokyo", "Pacific/Kiritimati"):
            os.environ["TZ"] = zone
            time.tzset()
            answers.append(store.made_before_session(stamp, session))
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()

    assert answers == [True, True, True, True], answers
    # Equal instants are not "before", whichever way each side is written:
    # 2026-09-13T00:00+09:00 is 2026-09-12T15:00Z, the session start itself.
    assert store.made_before_session("2026-09-12T15:00:00+00:00", session) is False
    assert store.made_before_session("2026-09-13T00:00:00+09:00", session) is False
    assert store.made_before_session("2026-09-12T15:00:01+00:00", session) is False


def test_the_pre_session_subset_is_never_labelled_with_a_stored_flag():
    """`backfilled` exists in the table and is not consulted by anything here.

    Two rows, one flagged `backfilled = 1` and made before its session, one
    flagged `backfilled = 0` and made after: the labels follow the timestamps,
    not the flag. Pinned because a stored flag is the one thing rule 4 exists to
    rule out, and a column that looks authoritative will be read eventually.
    """
    with store._connect() as conn:
        conn.executemany(
            """INSERT INTO session_predictions
                 (season, round, race_name, session_type, driver_id, constructor_id, tier, market,
                  predicted_prob, session_time, snapshotted_at, resolved, actual_outcome, backfilled)
               VALUES (2026, 2, 'GP 2', 'race', ?, 'x', 'post_qualifying', 'win', 0.5,
                       '2026-03-08T15:00:00+00:00', ?, 1, 1, ?)""",
            [("flagged_late", "2026-09-20T08:00:00+00:00", 0),
             ("unflagged_early", "2026-03-07T10:00:00+00:00", 1)],
        )

    labels = {r["driver_id"]: r["made_before_session"]
              for r in store.get_session_track_record(session_type="race")["per_pick"]}

    assert labels == {"flagged_late": False, "unflagged_early": True}, labels


# --- the payload the site actually reads --------------------------------------

def test_the_endpoint_serves_both_figures_and_the_per_pick_disclosure():
    """The API is where the swap becomes visible, so it is asserted on the wire.

    `n_resolved` and `by_market` keep their names; `pre_session`, `n_pre_session`
    and `n_post_session_picks` are added, and `per_pick` carries the disclosure.
    `session_type` is still dropped from each row, because `by_market` is
    grouped on it and a row's own key already says which session type it is.
    """
    from fastapi.testclient import TestClient

    from f1_predictor.api.main import app

    _record_and_resolve(1, "2099-01-01T13:00:00+00:00", norris=0.6)
    _record_and_resolve(2, "2020-01-01T13:00:00+00:00", norris=0.9)

    body = TestClient(app).get("/api/track-record?session_type=race").json()

    assert body["n_resolved"] == 16
    assert len(body["by_market"]) == 4, "one row per market, not per session"
    assert all("session_type" not in row for row in body["by_market"])
    assert body["pre_session"]["n_resolved"] == 8
    assert len(body["pre_session"]["by_market"]) == 4
    assert body["n_resolved"] == body["n_pre_session"] + body["n_post_session_picks"]
    assert body["n_rebuilt_sessions"] == 1
    # And the disclosure rides along: one row per recorded pick, with both
    # timestamps and the derived label, so the comparison is auditable.
    assert len(body["per_pick"]) == 16
    picks = {(r["round"], r["driver_id"]): r for r in body["per_pick"]}
    assert picks[(1, "norris")]["made_before_session"] is True
    assert picks[(2, "norris")]["made_before_session"] is False
    for row in body["per_pick"]:
        assert row["snapshotted_at"] and row["session_time"]
        assert "backfilled" not in row, "the stored flag must not be published as the label"


# --- the shipped state: is the headline empty over a populated record? --------
# Count-free on purpose: `data/tracking.db` refreshes itself, so the assertions
# below are properties of the rule, not totals. See the test's docstring.

def test_the_shipped_headline_covers_every_recorded_pick_and_pre_session_is_a_subset():
    """PL shipped `n_resolved_fixtures: 0, pct_correct_overall: null` with 50
    graded picks in the same payload. This repo had the same DISEASE at a
    different size.

    Before the reversal, on this repo's own `data/tracking.db`, a session was
    published only if EVERY one of its rows was snapshotted before it started;
    12 of the 17 recorded sessions were dropped whole and 1,056 graded picks
    reached no published figure at all. The worst single market,
    `race`/`post_qualifying`/`dnf`, was published at n=22 when 286 were
    recorded.

    **Why this asserts PROPERTIES and not counts.** `data/tracking.db` is
    refreshed by an automated commit (the "Refresh public snapshot + track
    record" job), so its row totals move on their own. This version of the test
    pinned 1,452 / 396 / 1,056 / 286, which is a landmine: the store refreshed to
    1,584 resolved rows and the suite went red for a reason that has nothing to
    do with the code, and would go red again on the next refresh. Re-baselining
    to 1,584 would re-arm the same landmine one refresh later, which is why it
    is not done here.

    What the test is actually FOR survives the refresh, because it is a property
    of the RULE rather than a total:

      * the headline covers every resolved pick the store holds, so whole-session
        dropping cannot come back;
      * every published market carries the FULL count recorded for it, so no
        market can be hidden again (this is the property the single `dnf` figure
        used to stand in for, and it is strictly stronger);
      * `pre_session` is a non-empty STRICT subset beside the headline -- not
        empty, and not the whole record in disguise;
      * the two partition the headline exactly.

    The dated figures that used to live here are not lost: they are re-measured
    and published in `tests/test_trust_signal.py::SNAPSHOT_BANDS`, against a named
    blob SHA, where a refresh trips a test that SAYS it is a snapshot. Keeping one
    dated snapshot of this store in one place, and leaving every other test
    count-free, is the whole arrangement.

    One earlier claim here was false against the data and is corrected: the
    pre-session subset is NOT "the figure the old headline published". The old
    rule dropped sessions whole; `pre_session` is a per-pick subset of the FULL
    counted record. The two were never the same set, so the subset is asserted
    only for what it must be -- a strict, non-empty subset.
    """
    db = REPO / "data" / "tracking.db"
    if not db.exists():
        pytest.skip("no shipped tracking database in this checkout")

    # Every count below is read out of the shipped file at run time. Nothing is
    # hardcoded, so a refresh moves the numbers without moving the assertions.
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        resolved = conn.execute(
            "SELECT COUNT(*) FROM session_predictions WHERE resolved = 1"
        ).fetchone()[0]
        recorded_by_market = {
            (session_type, tier, market): n
            for session_type, tier, market, n in conn.execute(
                "SELECT session_type, tier, market, COUNT(*) FROM session_predictions "
                "WHERE resolved = 1 GROUP BY session_type, tier, market"
            )
        }

    from f1_predictor import config

    original = store.TRACKING_DB_PATH
    try:
        store.TRACKING_DB_PATH = config.TRACKING_DB_PATH  # the shipped file
        record = store.get_session_track_record()
    finally:
        store.TRACKING_DB_PATH = original

    assert record["n_resolved"] == resolved, (
        f"the headline counts {record['n_resolved']} picks but the store holds "
        f"{resolved} resolved ones, so whole sessions are being dropped again"
    )

    published_by_market = {
        (row["session_type"], row["tier"], row["market"]): row["n"]
        for row in record["by_market"]
    }
    assert published_by_market == recorded_by_market, (
        "the published markets are not the recorded markets, so a market is "
        f"hidden again; under-published: "
        f"{ {k: (recorded_by_market[k], published_by_market.get(k)) for k in recorded_by_market if published_by_market.get(k) != recorded_by_market[k]} }"
    )

    assert 0 < record["n_pre_session"] < record["n_resolved"], (
        f"pre_session is {record['n_pre_session']} of {record['n_resolved']} picks; "
        "it must be a non-empty strict subset of the headline"
    )
    assert record["n_resolved"] == record["n_pre_session"] + record["n_post_session_picks"], (
        "the pre-session subset and the post-session picks must partition the headline"
    )
    assert record["n_post_session_picks"] > 0, (
        "no pick is recorded after its session started, so this store no longer "
        "exercises the rule this file exists to protect"
    )
