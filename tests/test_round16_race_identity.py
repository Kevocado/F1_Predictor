"""Round 16 of the 2026 season is the Bahrain Grand Prix *in Malaysia*.

The name reads like a bug — "Bahrain" and "Malaysia" are two different
countries in one string — and it was filed as one ("the circuit-to-country
mapping is wrong; it should say Bahrain, not Malaysia"). It is not a bug.
It is the official name of the round.

What actually happened, per the FIA's own announcement of 26 July 2026
("FIA and FOM confirm that Malaysia will join the 2026 calendar, as host
venue for the Bahrain Grand Prix"):

  * Bahrain's round on the 2026 calendar was moved to Malaysia, and
  * the event "will become the Formula 1 Gulf Air Bahrain Grand Prix in
    Malaysia", held at Sepang International Circuit on 2-4 October 2026.

So round 16 is a Bahrain-named race at a Malaysian circuit. Both halves of
the string are load-bearing: "Bahrain" is the race's name, "in Malaysia" is
where it is being run. Rewriting it to "Bahrain Grand Prix" would tell a
reader the race is at Sakhir, which is the opposite of the truth, and
would contradict the circuit name this very snapshot carries.

This file exists so the next person who trips over the name does not
"fix" it a second time. It pins three things:

  1. the served title is the official name,
  2. the committed snapshot's round-16 row is self-consistent, and
  3. the app never invents or remaps a race name — it renders upstream's
     `raceName` verbatim, so the string's content is not this repo's
     to rewrite.
"""

import json

import pytest

from f1_predictor.api import facts as facts_mod
from f1_predictor.config import PUBLIC_SNAPSHOT_PATH
from f1_predictor.data import jolpica

#: The official name, and the circuit it is run at. Source: the FIA
#: announcement of 26.07.26 (quoted in this module's docstring) and the
#: 2026 schedule served by this repo's own data source
#: (`config.JOLPICA_BASE_URL`), whose round 16 is:
#:
#:     raceName    "Bahrain Grand Prix in Malaysia"
#:     circuitId   "sepang"
#:     circuitName "Sepang International Circuit"
#:     locality    "Kuala Lumpur"
#:     country     "Malaysia"
#:     date        "2026-10-04"
#:
#: The circuit fields are the ones that locate the race, and they agree
#: with the FIA. The `raceName` is the only field in the whole 2019-2026
#: span that contains " in " — it is the official name, not a template.
OFFICIAL_RACE_NAME = "Bahrain Grand Prix in Malaysia"
OFFICIAL_CIRCUIT_NAME = "Sepang International Circuit"


# --- 1. the served title -------------------------------------------------------

def _prediction(**kw):
    payload = {
        "season": 2026,
        "round": 16,
        "race_name": OFFICIAL_RACE_NAME,
        "session_type": "race",
        "tier": "pre_weekend",
        "source": "live",
        "predictions": [],
        "session_datetime": "2026-10-04T07:00:00Z",
    }
    payload.update(kw)
    return payload


def test_the_round16_title_is_the_official_race_name(monkeypatch):
    """The symptom as it renders: `/facts` for 2026-16-race.

    `facts.py` composes the title as f"{race_name} · {session}", so this
    asserts the whole thing a reader sees. The country half is the part
    that invites the "fix" — it is asserted literally so a future change
    has to argue with this test rather than quietly drop the words.
    """
    monkeypatch.setattr(facts_mod, "_current_prediction", lambda *a, **k: _prediction())
    monkeypatch.setattr(facts_mod, "_stored_rows", lambda *a, **k: [])

    out = facts_mod.get_facts("2026-16-race")

    assert out["title"] == f"{OFFICIAL_RACE_NAME} · Race", (
        f"title is {out['title']!r}. Round 16 is officially the "
        f"{OFFICIAL_RACE_NAME!r}, run at {OFFICIAL_CIRCUIT_NAME!r} — the FIA "
        f"moved Bahrain's 2026 round to Malaysia and kept the name. Dropping "
        f"'in Malaysia' would place the race at Sakhir, which is not where it is."
    )


def test_dropping_the_country_would_describe_a_different_race(monkeypatch):
    """The control, and the reason this is not a cosmetic string choice.

    Asserted as a fact about the data, not as a preference: the name and
    the circuit come from the same upstream record, and they have to
    agree. If a change made the title say "Bahrain Grand Prix" while the
    row still names Sepang, the page would be describing a meeting that
    does not exist — which is the complaint this file was opened to close,
    pointed the other way.
    """
    row = _schedule_row(monkeypatch)

    assert row["circuit_name"] == OFFICIAL_CIRCUIT_NAME
    assert row["country"] == "Malaysia"
    assert "Bahrain" in row["race_name"] and "Malaysia" in row["race_name"], (
        f"the record's own race name is {row['race_name']!r}; the circuit is "
        f"{row['circuit_name']!r} in {row['country']!r}. Those are not the same "
        f"place, which is exactly why the name carries both."
    )


# --- 2. the committed snapshot -------------------------------------------------

def _snapshot():
    if not PUBLIC_SNAPSHOT_PATH.exists():
        pytest.skip("no committed snapshot in this checkout")
    return json.loads(PUBLIC_SNAPSHOT_PATH.read_text())


def test_the_committed_snapshots_round16_row_is_self_consistent():
    """The file the deployment actually serves, checked on disk.

    `public_snapshot.json` is the artifact behind every public read, and
    the round-16 name arrives in it from the schedule loader verbatim.
    Pinning the committed row means the two fields that must agree — the
    race's name and the circuit it is run at — cannot drift apart in a
    refresh without a test noticing.
    """
    row = next(r for r in _snapshot()["races"] if r["round"] == 16)

    assert row["race_name"] == OFFICIAL_RACE_NAME, f"snapshot race_name is {row['race_name']!r}"
    assert row["circuit_name"] == OFFICIAL_CIRCUIT_NAME, f"snapshot circuit_name is {row['circuit_name']!r}"


def test_no_other_round_in_the_snapshot_carries_a_country_in_its_name():
    """One anomalous-looking name, and it is the official one.

    Every other round's name is a plain "<something> Grand Prix". Round
    16 is the only one with " in " in it, which is what made it look
    like a template accident. This asserts that the anomaly is confined
    to the round the FIA named that way — so if one ever appears on a
    round nobody sanctioned, this is the test that catches it.
    """
    with_in = sorted(
        r["round"] for r in _snapshot()["races"] if " in " in r["race_name"]
    )

    assert with_in == [16], (
        f"these rounds have ' in ' in their race name: {with_in}. Only round 16 "
        f"is known to be an official name of that shape; anywhere else it is a bug."
    )


# --- 3. the app does not remap names -------------------------------------------

def _schedule_row(monkeypatch=None, **overrides):
    """One schedule row, built from a single upstream-shaped record.

    Stands in for `jolpica.fetch_season_schedule`'s per-race loop, so the
    assertion below is about the loader's own field handling rather than
    about what any particular season happens to say.
    """
    record = {
        "round": "16",
        "raceName": OFFICIAL_RACE_NAME,
        "date": "2026-10-04",
        "time": "15:00:00Z",
        "Circuit": {
            "circuitId": "sepang",
            "circuitName": OFFICIAL_CIRCUIT_NAME,
            "Location": {
                "lat": "2.76083",
                "long": "101.738",
                "locality": "Kuala Lumpur",
                "country": "Malaysia",
            },
        },
    }
    record.update(overrides)
    payload = {"MRData": {"RaceTable": {"Races": [record]}}}
    if monkeypatch is not None:
        monkeypatch.setattr(jolpica, "_cache_or_fetch", lambda *a, **k: payload)
    return jolpica.fetch_season_schedule(2026).iloc[0].to_dict()


def test_the_schedule_loader_passes_the_race_name_through_verbatim(monkeypatch):
    """The invariant that makes this file's conclusion unavoidable.

    `fetch_season_schedule` copies `raceName` off the upstream record and
    nothing else. There is no circuit->name table, no template, and no
    country lookup anywhere in this project — `country` is read once, into
    a column, and no string is ever built from it (`grep -rn country src/`
    returns exactly one hit). So the text a reader sees is upstream's, and
    correcting it here would be rewriting a fact this repo does not own.
    """
    row = _schedule_row(monkeypatch)

    assert row["race_name"] == OFFICIAL_RACE_NAME
    assert row["circuit_id"] == "sepang"
    assert row["circuit_name"] == OFFICIAL_CIRCUIT_NAME
    assert row["country"] == "Malaysia"


def test_a_race_name_naming_a_different_circuit_is_still_passed_through(monkeypatch):
    """The one case worth being explicit about, so this is not mistaken
    for blind trust.

    A record whose `raceName` disagrees with its own `Circuit` block is
    possible upstream — that is exactly what a 2026 round hosted at a
    different circuit looks like before anyone reconciles it. The loader
    does not and should not guess: the circuit fields are the ones the
    feature pipeline keys on (`features/circuit.py` joins on circuit_id),
    and a name rewritten here would diverge from the circuit the model is
    actually predicting for. Pinned so that if this ever changes, it is a
    deliberate decision with a test attached.
    """
    row = _schedule_row(monkeypatch, raceName="Bahrain Grand Prix")

    assert row["race_name"] == "Bahrain Grand Prix"
    assert row["circuit_id"] == "sepang", "the circuit the model keys on is unchanged either way"
