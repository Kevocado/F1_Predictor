"""`/facts` for F1: three claims that are wrong in the live payload.

The live response for `2026-16-race` was:

    title        "Bahrain Grand Prix in Malaysia · Race"
    starts_at    ""
    pick         {"label": null, "prob": 0.1479}
    pick_timing  "pre_kickoff"

Three separate defects, and the middle one explains the third.

**1. `pick.label` is null.** The panel therefore said "There is no pick for this
one yet" on a race where the model had Max Verstappen at 14.8%. `schemas.py`
names the field `driver_name` and says outright that it is never null; this
module read `name`, which no response carries. `.get()` on an absent key returns
None rather than raising, so the miss was silent — the worst kind.

**2. `starts_at` is empty.** `SessionPredictionResponse` carries
`session_datetime`; this module read `session_time`, which is the STORED-ROW
field. So the session start was never found, and `starts_at` shipped as `""`
while a scheduled session had a real time in the same response.

**3. `pick_timing` says `pre_kickoff` when it cannot be verified.** This is the
consequence of (2), and it is the one that matters. The timing is derived from
when the session starts; with no start time, "pre-session" is not a fact anyone
can check — and the live panel would show a pick's timing as verified when the
response carried no evidence at all. An explicit unknown is the honest answer,
and the explainer is told to treat it as not-verifiable.

None of these raise. A null label renders as "no pick", an empty `starts_at`
renders as nothing, and an unverifiable timing renders as a confident claim —
three wrong answers that all look like working code.
"""
import pytest

from f1_predictor.api import facts as facts_mod
from f1_predictor.api.facts import _driver_rows, get_facts


def _row(driver_id="max_verstappen", name="Max Verstappen", **kw):
    row = {
        "driver_id": driver_id,
        "driver_name": name,
        "p_win": 0.1479,
        "p_podium": 0.5,
        "p_points_finish": 0.9,
        "p_dnf": 0.05,
    }
    row.update(kw)
    return row


def _prediction(**kw):
    payload = {
        "season": 2026,
        "round": 16,
        "race_name": "Bahrain Grand Prix",
        "session_type": "race",
        "tier": "post_qualifying",
        "source": "live",
        "predictions": [_row()],
        "session_datetime": "2026-11-22T15:00:00Z",
    }
    payload.update(kw)
    return payload


# --- 1. the pick's label ---------------------------------------------------------

def test_driver_rows_expose_the_name_the_schema_actually_carries():
    """The boundary normalises it, so every reader below sees one vocabulary.

    Asserted on `_driver_rows` rather than on the payload: the bug was a field
    name mismatch, and there are two readers of it. Fixing the reader twice is
    how one of them gets missed next time.
    """
    rows = _driver_rows(_prediction())
    assert rows[0]["name"] == "Max Verstappen"


def test_the_pick_is_named_not_null(monkeypatch):
    """The live symptom, reproduced.

    Before the fix this produced `{"label": null, "prob": 0.1479}` and the
    panel said there was no pick on a race where the model had one.
    """
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda *a, **k: _prediction())
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda *a, **k: [])
    monkeypatch.setattr(facts_mod, "_stored_by_driver", lambda rows: {})

    out = get_facts("2026-16-race")

    assert out["pick"] is not None, "a session with a 14.8% driver is a session with a pick"
    assert out["pick"]["label"] == "Max Verstappen", (
        f"pick label is {out['pick']['label']!r}; the panel renders null as 'no pick yet'"
    )
    assert out["pick"]["prob"] == pytest.approx(0.1479)


# --- 2. the session start --------------------------------------------------------

def test_starts_at_comes_from_the_field_the_response_carries(monkeypatch):
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda *a, **k: _prediction())
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda *a, **k: [])

    out = get_facts("2026-16-race")

    assert out["starts_at"] == "2026-11-22T15:00:00Z", (
        f"starts_at is {out['starts_at']!r}. The response carried session_datetime; this "
        f"module read session_time, which is the stored-row field, so a scheduled "
        f"session shipped with no start time at all."
    )


def test_a_missing_start_time_is_empty_rather_than_a_guessed_time(monkeypatch):
    """The family convention, and why the honesty lives in `pick_timing` instead.

    `null` was tried and rejected: the shared contract types `starts_at: str`
    and PL emits the same empty string, so one sport sending a different type
    for the same field is worse than a shared convention. What matters is that
    nothing is invented — which the next test pins.
    """
    monkeypatch.setattr(facts_mod, "_current_prediction",
                        lambda *a, **k: _prediction(session_datetime=None))
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda *a, **k: [])

    out = get_facts("2026-16-race")

    assert out["starts_at"] == "", f"starts_at is {out['starts_at']!r}, not empty"


# --- 3. timing honesty -----------------------------------------------------------

def test_pick_timing_is_not_pre_kickoff_when_the_start_time_is_unknown(monkeypatch):
    """The consequence, and the one that matters.

    The timing says when the pick was made relative to the session start. With
    no start time there is no evidence either way, and the live payload claimed
    `pre_kickoff` anyway — a pick whose timing a reader would take as verified.
    """
    monkeypatch.setattr(facts_mod, "_current_prediction",
                        lambda *a, **k: _prediction(session_datetime=None))
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda *a, **k: [])

    out = get_facts("2026-16-race")

    assert out["pick_timing"] != "pre_kickoff", (
        "pick_timing is pre_kickoff with no session start in the response. That is a "
        "claim about provenance with nothing behind it, and the panel shows it as fact."
    )
    assert out["pick_timing"] == "unknown", (
        f"expected an explicit unknown, got {out['pick_timing']!r}"
    )


def test_a_known_start_time_does_still_report_pre_kickoff(monkeypatch):
    """The control. Without it, "never pre_kickoff" would satisfy the test above."""
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda *a, **k: _prediction())
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda *a, **k: [])

    out = get_facts("2026-16-race")

    assert out["starts_at"] is not None
    assert out["pick_timing"] == "pre_kickoff"


# --- 4. the title, which is a data problem and must be visible -------------------

def test_the_title_names_this_races_name_and_nothing_else(monkeypatch):
    """Live: "Bahrain Grand Prix in Malaysia · Race" for round 16.

    The "in Malaysia" half is the circuit's location attached to a different
    race's name, so the title describes a meeting that does not exist. The
    source is the schedule row rather than this module, so the assertion is that
    the title is built from the race name the response carries — and the
    schedule's own contents are a separate question, recorded here rather than
    fixed silently.
    """
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda *a, **k: _prediction())
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda *a, **k: [])

    out = get_facts("2026-16-race")

    assert out["title"].startswith("Bahrain Grand Prix · Race"), (
        f"title is {out['title']!r}"
    )
