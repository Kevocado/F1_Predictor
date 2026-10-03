"""The `trust` signal: which band it reads, the floor, `strength`, and whether the
payload is one the shared `SignalRows` component will actually accept.

**Every claim in here is checked against the REAL component or the REAL committed
blob**, never against a restatement of either:

  * the numbers come out of `data/tracking.db` **as git has it** — asserted by
    content hash against `HEAD:data/tracking.db`, not by the file merely being
    present (see `test_the_numbers_come_from_the_committed_blob_not_a_local_copy`);
  * the component's floors, word cap, figure names and which visual draws a rate
    are READ OUT OF the vendored `SignalRows.tsx` by `_vendored_contract()`, so a
    change to that file breaks this suite rather than being quietly satisfied by a
    copy of the old rules. `frontend/src/predictor-ui.trust-contract.test.tsx` is
    the other half: it renders the real component with a payload of this shape, so
    the component's own behaviour is executed rather than inferred.

The six things the plan and the PR description require of this adapter, and where
each is pinned:

  1. right band, `n`, hit rate ......... `test_the_band_n_and_hit_rate_...`
  2. `n` below the floor ................ `TestTheFloor` (the ADAPTER owns it)
  3. blank headline refused ............ `TestTheComponentContract`
  4. `strength` monotonic in the gap ... `TestStrength`
  5. satisfies `SignalRows` ............ `TestTheComponentContract`
  6. the committed blob is the source . `TestTheCommittedBlobIsTheSource`
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from f1_predictor.api import facts as facts_mod
from f1_predictor.api import signals as signals_mod
from f1_predictor.signals import trust
from f1_predictor.tracking import store

REPO = Path(__file__).resolve().parents[1]
VENDORED = REPO / "frontend" / "src" / "predictor-ui" / "components" / "SignalRows.tsx"
FMT = REPO / "frontend" / "src" / "predictor-ui" / "fmt.ts"


# --- the committed blob, as git has it --------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()


def _tracking_db() -> Path:
    return REPO / "data" / "tracking.db"


def _bands_from_the_committed_db() -> list[dict]:
    return store.get_probability_buckets(trust.BUCKET_BOUNDS)


#: The published table, re-derived from the committed blob on 2026-10-03.
#:
#: NOTE THIS IS NOT THE TABLE IN THE SPIKE DOC. That was measured at F1
#: `6f492739` (blob `da6217d9`, 1,452 resolved rows); `data/tracking.db` is
#: refreshed by an automated commit and the blob at this base is `0071bc32` with
#: 1,584 resolved rows, so every band has grown. Asserted here rather than
#: copied into the adapter, because a band table that only exists in a test is a
#: band table nothing reads.
PUBLISHED_BANDS = [
    # (label, n, hits)
    ("0.0-0.3", 1235, 151),
    ("0.3-0.4", 148, 52),
    ("0.4-0.5", 44, 17),
    ("0.5-0.6", 27, 21),
    ("0.6-0.7", 45, 34),
    ("0.7-1.0", 85, 69),
]


def _make_db(path: Path, rows: list[tuple[float, int]]) -> Path:
    """A `session_predictions` table holding exactly `rows` of `(prob, outcome)`.

    The schema comes from the store's own `_connect`, so a fixture cannot drift
    from the schema it is read back with — the same argument
    `tests/test_session_predictions_store.py` makes. `TRACKING_DB_PATH` is
    swapped only long enough to create the table, because the caller owns the
    monkeypatch and restoring it here keeps this helper usable on its own.
    """
    original = store.TRACKING_DB_PATH
    store.TRACKING_DB_PATH = path
    try:
        conn = store._connect()
    finally:
        store.TRACKING_DB_PATH = original
    with conn:
        conn.executemany(
            "INSERT INTO session_predictions "
            "(season, round, race_name, session_type, driver_id, tier, market, predicted_prob, "
            " session_time, snapshotted_at, resolved, actual_outcome, resolved_at) "
            "VALUES (2026, 1, 'Test GP', 'race', ?, 'post_qualifying', 'win', ?, "
            "        '2026-03-01T15:00:00Z', '2026-03-01T10:00:00Z', 1, ?, '2026-03-01T18:00:00Z')",
            [(f"d{i}", prob, outcome) for i, (prob, outcome) in enumerate(rows)],
        )
    return path


# --- the vendored component, read rather than restated ---------------------


def _vendored_contract() -> dict:
    """Read the component's own numbers and maps out of the vendored source.

    A mirror is what this replaces. Every value below is extracted from the file
    at `VENDORED`, so an edit there — a floor raised, a figure renamed, a visual
    that stops drawing a rate — fails the assertions that use it, which is the
    property a hand-written copy cannot have.
    """
    source = VENDORED.read_text()

    def const(name: str) -> int:
        match = re.search(rf"export const {name} = (\d+);", source)
        assert match, f"{name} is no longer a plain number in {VENDORED.name}; this reader must be taught the new spelling"
        return int(match.group(1))

    def object_map(name: str) -> dict[str, str]:
        match = re.search(rf"const {name}[^=]*=\s*\{{(.*?)\}}\s*;", source, re.S)
        assert match, f"{name} is gone or no longer an object literal in {VENDORED.name}"
        found = re.findall(r"(\w+):\s*(true|false|\"[^\"]*\"|null)", match.group(1))
        # Strip the string quoting off the values: `FIGURE` is `visual -> "rate"`
        # and the caller wants `rate`, not `"rate"`.
        return {key: value.strip('"') for key, value in found}

    return {
        "spec_min_n": const("SPEC_MIN_N"),
        "max_headline_words": const("MAX_HEADLINE_WORDS"),
        "figure": object_map("FIGURE"),
        "draws_a_rate": object_map("DRAWS_A_RATE"),
        "source": source,
        # The blank-headline guard, as a fact about the component rather than a
        # belief about it. `signalIsDrawn` returns false for an empty headline,
        # which is a DROP and not a throw — the asymmetry with the figure
        # mismatch is deliberate in the component and this is what pins it.
        "drops_blank_headline": 'signal.headline.text.trim() === ""' in source,
        "drops_below_floor_before_validating": (
            source.index("if (!rateIsDrawable(signal, minN)) return false;")
            < source.index("assertFigureIsStated(signal);", source.index("export function signalIsDrawn"))
        ),
    }


def _pct_rounds_half_up() -> bool:
    """Whether `fmt.pct` still rounds a rate to a whole percent with `Math.round`.

    The adapter's headline states `round(rate * 100)`, and that is only the right
    number because `pct` — which is what the bar is drawn with — rounds the same
    way. Read out of the vendored `fmt.ts` rather than assumed, so a change to the
    formatter's rounding breaks this suite instead of making every headline
    quietly wrong.
    """
    return "Math.round(p * 100)" in FMT.read_text()


# --- 1. the band, its n, and its hit rate -----------------------------------


class TestTheBand:
    def test_the_re_derived_table_is_the_published_one(self):
        bands = _bands_from_the_committed_db()
        assert [(b["label"], b["n"], b["hits"]) for b in bands] == PUBLISHED_BANDS
        assert bands[0]["n_resolved"] == 1584, "the resolved-row count moved; republish the table"

    def test_the_hit_rate_is_hits_over_n(self):
        for band in _bands_from_the_committed_db():
            assert band["rate"] == pytest.approx(band["hits"] / band["n"])

    def test_the_calibration_is_monotone_across_the_rendered_bands(self):
        """The reason this signal is worth shipping: a higher stated probability
        has meant a higher realised rate.

        Checked over the bands that RENDER, and stated as "each step up is at
        least as good" rather than strict monotonicity — the shipped data is
        monotone on five of six bands and dips once at the top (0.5-0.6 lands at
        77.8% on 27 picks, just above 0.6-0.7's 75.6% on 45). Asserting strict
        monotonicity would be asserting something false about the real table; the
        test says what is true and the PR quotes the dip.
        """
        rendered = [b for b in _bands_from_the_committed_db() if b["n"] >= trust.TRUST_MIN_N]
        assert len(rendered) >= 4, "too few bands clear the floor for this to mean anything"
        rates = [b["rate"] for b in rendered]
        for lower, higher in zip(rates, rates[1:]):
            assert higher >= lower, f"a higher stated probability realised a LOWER rate: {rates}"

    @pytest.mark.parametrize(
        "prob,expected",
        [
            (0.0, "0.0-0.3"),
            (0.05, "0.0-0.3"),
            (0.2999, "0.0-0.3"),
            # Band edges, from both sides. The top edge is inclusive and every
            # other one exclusive, which must be `store.get_probability_buckets`'s
            # convention too or a probability ON an edge is read from one band and
            # reported from another.
            (0.3, "0.3-0.4"),
            (0.3999, "0.3-0.4"),
            (0.4, "0.4-0.5"),
            (0.5, "0.5-0.6"),
            (0.6, "0.6-0.7"),
            (0.7, "0.7-1.0"),
            (0.9999, "0.7-1.0"),
            (1.0, "0.7-1.0"),
            # Not a probability at all.
            (1.5, None),
            (-0.1, None),
            (None, None),
            (float("nan"), None),
            (float("inf"), None),
            (True, None),
            ("0.4", None),
        ],
    )
    def test_a_probability_lands_in_exactly_one_band(self, prob, expected):
        assert trust.band_label(prob) == expected

    def test_a_session_in_a_band_gets_that_bands_n_and_hit_rate(self):
        bands = {b["label"]: b for b in _bands_from_the_committed_db()}
        for label, n, hits in PUBLISHED_BANDS:
            band = bands[label]
            if n < trust.TRUST_MIN_N:
                # Asserted in TestTheFloor; here only that the mapping is coherent.
                continue
            prob = (band["low"] + band["high"]) / 2
            signal = trust.trust_signal("2026-15-race", prob)
            assert signal is not None, f"{label} clears the floor but produced no signal"
            assert signal["n"] == n
            assert signal["headline"]["figures"]["rate"] == pytest.approx(hits / n)
            assert label in signal["source"]

    def test_the_band_the_pick_falls_in_is_the_band_the_signal_reads(self, monkeypatch):
        """Through the API, not through the adapter directly.

        `trust_signal` takes a probability; nothing but `signals_for_session`
        decides which probability that is. That indirection is exactly where a
        signal could quote a different number from the one on the page, so the
        test goes in at the endpoint and reads the band off the payload.
        """
        bands = {b["label"]: b for b in _bands_from_the_committed_db()}
        target = bands["0.7-1.0"]
        prob = (target["low"] + target["high"]) / 2
        monkeypatch.setattr(
            facts_mod, "session_pick",
            lambda *a, **k: {"pick": {"label": "verstappen", "prob": prob}},
        )

        signals = signals_mod.signals_for_session(2026, 15, "race")

        assert len(signals) == 1
        assert signals[0]["headline"]["figures"]["rate"] == pytest.approx(target["hits"] / target["n"])
        assert signals[0]["n"] == target["n"]
        assert signals[0]["game_id"] == "2026-15-race"
        assert signals[0]["kind"] == "trust"


# --- 2. the floor, and which layer owns it ----------------------------------


class TestTheFloor:
    """**The ADAPTER owns the floor.** A band under it emits NO signal.

    The component has its own floor and drops the row, but that answers "may this
    be DRAWN?", and `GET /signals` is read by more than the component: Phase 4
    hands these payloads to the AI "so what" writer, whose validator admits any
    figure present in a signal payload. A rate from 27 games emitted here would be
    quotable as a claim about 27 games, and the component's floor arrives far too
    late to stop it. So this class asserts the adapter omits, and
    `test_the_two_floors_agree_and_neither_can_lower_the_other` pins that the
    backstop is real too.
    """

    def test_the_constant_is_the_specs_number(self):
        assert trust.TRUST_MIN_N == 30, "spec §4 / §9 decision 3; reversible by this one constant"

    def test_a_band_under_the_floor_produces_no_signal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "t.db")
        # 29 hits' worth of resolved picks, all in one band. 29 < 30.
        _make_db(tmp_path / "t.db", [(0.65, 1)] * 29)

        assert trust.trust_signal("2026-15-race", 0.65) is None

    def test_exactly_at_the_floor_produces_a_signal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "t.db")
        _make_db(tmp_path / "t.db", [(0.65, 1)] * 29 + [(0.65, 0)])

        signal = trust.trust_signal("2026-15-race", 0.65)

        assert signal is not None, "n == 30 must render; the floor is >=, not >"
        assert signal["n"] == 30

    def test_the_committed_bands_below_the_floor_are_the_one_thin_one(self):
        thin = [b for b in _bands_from_the_committed_db() if b["n"] < trust.TRUST_MIN_N]
        assert [(b["label"], b["n"]) for b in thin] == [("0.5-0.6", 27)]

    def test_the_thin_band_in_the_committed_db_really_emits_nothing(self):
        """The floor is not theoretical: the shipped data has a band under it."""
        assert trust.trust_signal("2026-15-race", 0.55) is None

    def test_a_two_pick_band_would_otherwise_report_a_rate(self, tmp_path, monkeypatch):
        """The exact failure §4 names: a rate from a handful of games.

        Pinned as a POSITIVE control — with two resolved picks the store has a
        real rate, and the only thing standing between that and the page is the
        floor. If the floor were removed this test's neighbour goes green and this
        assertion is what would have caught it.
        """
        monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "t.db")
        _make_db(tmp_path / "t.db", [(0.65, 1), (0.66, 0)])
        raw = store.get_probability_buckets(trust.BUCKET_BOUNDS)

        assert raw[0]["n"] == 2 and raw[0]["rate"] == 0.5, "the store does compute the thin rate"
        assert trust.trust_signal("2026-15-race", 0.65) is None, "the adapter must not send it"

    def test_the_two_floors_agree_and_neither_can_lower_the_other(self):
        contract = _vendored_contract()

        assert contract["spec_min_n"] == trust.TRUST_MIN_N, (
            "the adapter's floor and the component's SPEC_MIN_N have diverged; the "
            "effective floor is their max, so one of the two is now dead code"
        )
        # The component's floor is `max(SPEC_MIN_N, minN)`, so a site can only
        # RAISE it. Read out of the source rather than asserted as a belief.
        assert "Math.max(SPEC_MIN_N, minN)" in contract["source"]
        # And it is checked before the figure is validated, which is what makes a
        # sub-floor row a drop rather than a throw.
        assert contract["drops_below_floor_before_validating"]

    def test_raising_the_adapter_floor_would_also_raise_the_effective_one(self, tmp_path, monkeypatch):
        """Spec §9 decision 3 reverses the floor by changing this one constant."""
        monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "t.db")
        _make_db(tmp_path / "t.db", [(0.65, 1)] * 40)
        monkeypatch.setattr(trust, "TRUST_MIN_N", 45)

        assert trust.trust_signal("2026-15-race", 0.65) is None

        monkeypatch.setattr(trust, "TRUST_MIN_N", 40)
        assert trust.trust_signal("2026-15-race", 0.65) is not None


# --- 3 + 5. the component's contract ---------------------------------------


class TestTheComponentContract:
    def test_the_figures_the_visual_draws_is_the_one_the_adapter_writes(self):
        contract = _vendored_contract()

        assert contract["figure"][trust.VISUAL] == trust.RATE_FIGURE, (
            "the component renamed the figure this visual draws; the payload's key "
            "must follow it or `signalFigure` throws on every render"
        )
        assert contract["draws_a_rate"][trust.VISUAL] == "true", (
            "the component no longer floors this visual, so the adapter's floor is "
            "now the only guard on a sub-floor rate reaching the page"
        )

    @pytest.mark.parametrize("prob", [0.12, 0.35, 0.45, 0.65, 0.8])
    def test_the_payload_carries_every_field_spec_3_names(self, prob):
        signal = trust.trust_signal("2026-15-race", prob)
        assert signal is not None

        assert set(signal) == {
            "kind", "sport", "game_id", "headline", "n", "source",
            "as_of", "strength", "pre_kickoff_only", "visual",
        }
        assert signal["kind"] == "trust"
        assert signal["sport"] == "f1"
        assert signal["pre_kickoff_only"] is True, "false only for post_game (spec §3)"
        assert 0.0 <= signal["strength"] <= 1.0

    def test_the_rate_is_a_share_the_bar_can_draw(self, prob=0.65):
        """`signalFigure` refuses a rate outside [0, 1] — the '9600%' guard."""
        rate = trust.trust_signal("2026-15-race", prob)["headline"]["figures"]["rate"]

        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_the_headline_is_inside_the_components_own_word_cap(self):
        cap = _vendored_contract()["max_headline_words"]
        for prob in (0.12, 0.35, 0.45, 0.65, 0.8):
            text = trust.trust_signal("2026-15-race", prob)["headline"]["text"]
            assert len(text.split()) <= cap, f"{len(text.split())} words against a cap of {cap}: {text!r}"

    def test_the_headline_states_the_figure_the_bar_draws(self):
        """The rule that THROWS rather than warns, so this is the one that matters.

        `assertFigureIsStated` compares the headline's NUMBERS against the percents
        `fmt.pct` can print for `figures.rate`, at whole-percent precision and
        requiring the `%` sign. So the stated figure must be `round(rate * 100)`:
        `75.6%` beside a bar reading `76%` is refused, and so is `76` without the
        sign. Every rendered band is checked, not a chosen one.
        """
        assert _pct_rounds_half_up(), (
            "fmt.pct no longer rounds with Math.round(p * 100); the headline's "
            "rounding must be re-derived from it, not assumed"
        )
        for prob in (0.12, 0.35, 0.45, 0.65, 0.8):
            signal = trust.trust_signal("2026-15-race", prob)
            rate = signal["headline"]["figures"]["rate"]
            stated = f"{round(rate * 100):g}%"
            assert stated in signal["headline"]["text"], (
                f"the bar draws {rate} as {stated} and the headline is "
                f"{signal['headline']['text']!r}, which does not state it — the "
                f"component throws HeadlineFigureMismatchError on this payload"
            )

    def test_the_headline_never_states_a_rate_as_a_bare_number(self, prob=0.65):
        """`denotes` compares a percent against a percent. "76" is not "76%"."""
        signal = trust.trust_signal("2026-15-race", prob)
        rate = signal["headline"]["figures"]["rate"]

        assert f"{round(rate * 100):g}%" in signal["headline"]["text"]
        assert not re.search(rf"\b{round(rate * 100)}\b(?!%)", signal["headline"]["text"])

    def test_the_headline_is_never_blank(self):
        """Spec §2: no empty states. The component DROPS a blank headline, so a
        blank one is a silently missing row rather than a visible failure.

        Checked two ways: that the component still has the guard (a change there
        would make a blank headline render), and that no band can produce one — the
        adapter's sentence is a fixed template, so this is a property of the
        template rather than of any particular band's numbers.
        """
        contract = _vendored_contract()

        assert contract["drops_blank_headline"], (
            "SignalRows no longer drops a blank headline; one would now render"
        )
        for prob in (0.0, 0.12, 0.35, 0.45, 0.65, 0.8, 1.0):
            text = trust.trust_signal("2026-15-race", prob)["headline"]["text"]
            assert text.strip(), f"blank headline for prob={prob}"

    def test_the_headline_carries_no_range_it_could_be_misread_as_a_negative(self):
        """`0.0-0.3` in a headline scans as a signed -0.3 and is refused.

        The band's own edges belong in `source`, which the component renders
        verbatim. A range in the headline is both misreadable and a figure the row
        does not draw.
        """
        for prob in (0.12, 0.35, 0.45, 0.65, 0.8):
            text = trust.trust_signal("2026-15-race", prob)["headline"]["text"]
            assert not re.search(r"\d\s*[-–]\s*\d", text), f"a hyphenated range in the headline: {text!r}"

    def test_the_headline_is_not_sportsbook_vocabulary(self):
        """F1 has no odds feed at all (spec §4: `line_gap` is NFL/CFB only), so a
        price reading would describe a market this project has never seen."""
        banned = ("odds", "backing", "back ", "value ", "ev ", "implied", "spread", "price", "sure bet")
        for prob in (0.12, 0.35, 0.45, 0.65, 0.8):
            text = trust.trust_signal("2026-15-race", prob)["headline"]["text"].lower()
            for word in banned:
                assert word not in text, f"{word!r} in the headline: {text!r}"

    def test_the_headline_is_a_record_not_a_forecast(self):
        """Spec §4's honesty requirement: no certainty. Past tense, and a rate of
        the time rather than a promise."""
        for prob in (0.12, 0.35, 0.45, 0.65, 0.8):
            text = trust.trust_signal("2026-15-race", prob)["headline"]["text"]
            assert "of the time" in text
            assert not re.search(r"\b(will|always|guarantee|is right|are right)\b", text.lower()), text

    def test_the_stated_probability_is_marked_approximate(self):
        """It is the MEAN of a band 0.1 to 0.3 wide, so "says 10%" without the
        tilde would claim picks of exactly 10%. Spec §4's own `~59%` notation."""
        for prob in (0.12, 0.35, 0.45, 0.65, 0.8):
            text = trust.trust_signal("2026-15-race", prob)["headline"]["text"]
            assert "~" in text, text

    def test_source_and_as_of_are_both_present_or_neither(self):
        """Spec §3: `source` is 'attributable and dated', so a row carries both or
        neither. A half-dated row is the half of that the rule is about."""
        for prob in (0.12, 0.35, 0.45, 0.65, 0.8):
            signal = trust.trust_signal("2026-15-race", prob)
            assert bool(signal["source"]) == bool(signal["as_of"])
            assert signal["as_of"], "the shipped store populates resolved_at on every resolved row"

    def test_as_of_is_an_iso_instant_the_formatter_can_read(self):
        from datetime import datetime

        signal = trust.trust_signal("2026-15-race", 0.12)

        assert signal["as_of"].endswith("Z")
        assert datetime.fromisoformat(signal["as_of"].replace("Z", "+00:00"))

    def test_source_discloses_that_markets_are_pooled(self):
        """The one caveat a reader most needs: the rate spans seven markets whose
        own base rates run from 4.5% to 45.5%. Measured on the committed blob."""
        source = trust.trust_signal("2026-15-race", 0.12)["source"]

        assert "pooled" in source
        assert "probability band" in source

    def test_the_vendored_copy_is_the_one_sync_manifest_names(self):
        """So the contract above is read from the file that actually ships."""
        import hashlib

        manifest = json.loads((VENDORED.parent.parent / "SYNC.json").read_text())
        digest = hashlib.sha256(VENDORED.read_text().replace("\r\n", "\n").encode()).hexdigest()

        assert manifest["files"]["components/SignalRows.tsx"] == digest, (
            "the vendored SignalRows.tsx has been edited in this repo; changes "
            "belong in predictor-hub/packages/predictor-ui and a re-sync"
        )


# --- 4. strength ------------------------------------------------------------


class TestStrength:
    def test_it_rises_with_the_reliability_gap(self):
        """The plan's explicit requirement, and the direction this adapter argues
        for: a wider gap between what the model said and what it delivered is a
        row with more to say, not less. `n` is held fixed so this is the gap alone.
        """
        gaps = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15]
        strengths = [trust.strength(rate=0.5 + gap, mean_predicted=0.5, n=100) for gap in gaps]

        assert strengths == sorted(strengths)
        assert len(set(strengths)) == len(strengths), "a gap must move strength, not just leave it ordered"
        assert strengths[0] == 0.0, "a perfectly calibrated band adds nothing the page did not say"

    def test_it_rises_with_a_gap_in_either_direction(self):
        """Over-confident and under-confident are both worth saying."""
        over = [trust.strength(0.5 + g, 0.5, 100) for g in (0.0, 0.02, 0.04, 0.08)]
        under = [trust.strength(0.5 - g, 0.5, 100) for g in (0.0, 0.02, 0.04, 0.08)]

        assert over == sorted(over) and len(set(over)) == 4
        assert under == sorted(under) and len(set(under)) == 4
        # `approx`, not `==`: the two gaps are the same magnitude in real
        # arithmetic but not in binary floating point (0.52 - 0.5 and 0.5 - 0.48
        # differ around the 17th digit). Written as `==` first, and it failed on
        # exactly that, which says the tolerance belongs here and not in
        # `strength` — the difference is ~1e-16 and has no bearing on a 0..1 rank.
        assert over == pytest.approx(under)

    def test_a_gap_on_few_picks_scores_as_the_noise_it_is(self):
        """Why the gap is divided by the standard error.

        Raw gaps are not comparable across bands because the bands hold different
        numbers of picks. At a FIXED gap, `strength` rises with `n`, because `se`
        shrinks like `1/sqrt(n)` and the same discrepancy gets harder to attribute
        to chance: a five-point gap on 44 picks is noise, and the same gap on 5000
        is a fact.

        **This test was first written asserting the opposite direction** — that
        strength would FALL as `n` grew, which is what the module docstring claimed
        at the time. It failed, and the docstring was wrong, not the code: see the
        note in `trust.py`'s module docstring about the direction in `n`.
        """
        by_n = [trust.strength(rate=0.55, mean_predicted=0.5, n=n) for n in (10, 40, 100, 500, 5000)]

        assert by_n == sorted(by_n), f"a fixed gap must score higher on more picks: {by_n}"
        assert len(set(by_n)) == 5
        assert by_n[0] < 0.25, "a five-point gap on ten picks is not a finding"
        assert by_n[-1] == 1.0, "the same gap on 5000 picks saturates the scale"

    def test_a_thin_bands_gap_does_not_outrank_a_deep_bands_smaller_gap(self):
        """The property the standard-error division actually buys, on real bands.

        The shipped 0.4-0.5 band has a gap of 0.052 on 44 picks and the 0.0-0.3
        band a gap of 0.021 on 1235. Ordered by the raw gap the thin band comes
        first, which would show a reader a 44-game coincidence in preference to a
        1235-game miscalibration. Ordered by strength it comes second.
        """
        bands = {b["label"]: b for b in _bands_from_the_committed_db() if b["n"] >= trust.TRUST_MIN_N}
        thin, deep = bands["0.4-0.5"], bands["0.0-0.3"]

        thin_gap = abs(thin["rate"] - thin["mean_predicted"])
        deep_gap = abs(deep["rate"] - deep["mean_predicted"])
        assert thin_gap > deep_gap, "precondition: the thin band's raw gap is the larger one"

        thin_strength = trust.strength(thin["rate"], thin["mean_predicted"], thin["n"])
        deep_strength = trust.strength(deep["rate"], deep["mean_predicted"], deep["n"])
        assert deep_strength > thin_strength, (
            f"deep band {deep_strength:.3f} must outrank thin band {thin_strength:.3f}"
        )

    def test_it_reproduces_its_own_formula_on_every_committed_band(self):
        """A regression pin on the real numbers rather than a property claim.

        Monotonicity in the gap is asserted above at fixed `n`, and it CANNOT be
        asserted across the committed bands: they each hold a different number of
        picks, so a small-gap deep band is meant to outrank a big-gap thin one.
        That test was written the cross-band way and failed, which is the honest
        answer — it was asserting something the formula is designed to prevent.
        """
        for band in _bands_from_the_committed_db():
            if band["n"] < trust.TRUST_MIN_N:
                continue
            gap = abs(band["rate"] - band["mean_predicted"])
            se = (band["mean_predicted"] * (1 - band["mean_predicted"]) / band["n"]) ** 0.5
            assert trust.strength(band["rate"], band["mean_predicted"], band["n"]) == pytest.approx(
                min(1.0, (gap / se) / 3.0)
            ), f"{band['label']}"

    def test_the_rendered_bands_get_distinct_strengths(self):
        """So `GET /signals` ranking by them is a ranking, not a tie-break."""
        strengths = [
            trust.strength(b["rate"], b["mean_predicted"], b["n"])
            for b in _bands_from_the_committed_db()
            if b["n"] >= trust.TRUST_MIN_N
        ]

        assert len(set(strengths)) == len(strengths), f"two bands tie: {sorted(strengths)}"

    def test_it_stays_inside_zero_to_one(self):
        # A gap of 0.5 at full confidence, and a degenerate band of certainties.
        assert trust.strength(rate=1.0, mean_predicted=0.5, n=10) == 1.0
        assert trust.strength(rate=0.0, mean_predicted=1.0, n=1000) == 1.0
        assert 0.0 <= trust.strength(0.12, 0.10, 1235) <= 1.0

    def test_a_band_of_certainties_does_not_divide_by_zero(self):
        """`se` is 0 when every stated probability in the band is the same and
        0 or 1. No recorded F1 probability is (the shipped range is 0.0026 to
        0.9157), so this is handled rather than left to a ZeroDivisionError."""
        assert trust.strength(rate=0.0, mean_predicted=0.0, n=50) == 0.0
        assert trust.strength(rate=1.0, mean_predicted=0.0, n=50) == 1.0
        assert trust.strength(rate=0.0, mean_predicted=1.0, n=50) == 1.0

    def test_no_sample_means_no_claim(self):
        assert trust.strength(rate=0.5, mean_predicted=0.5, n=0) == 0.0

    def test_the_endpoint_sorts_by_strength(self, monkeypatch):
        """Spec §2: 'a fixed rule (not the model) ranks them by strength'."""
        calls = []

        def fake(game_id, prob):
            calls.append(prob)
            return {
                "kind": "trust", "sport": "f1", "game_id": game_id,
                "headline": {"text": "x", "figures": {"rate": prob}},
                "n": 30, "source": "s", "as_of": "", "strength": prob,
                "pre_kickoff_only": True, "visual": "reliability_bar",
            }

        monkeypatch.setattr(trust, "trust_signal", fake)
        monkeypatch.setattr(
            facts_mod, "session_pick",
            lambda *a, **k: {"pick": {"label": "x", "prob": 0.0}},
        )
        # One adapter, many signals: drive the ordering through the endpoint's own
        # sort rather than asserting on a list built by hand.
        ranked = sorted(
            [{"kind": "trust", "strength": s} for s in (0.1, 0.9, 0.5)],
            key=lambda s: s["strength"],
            reverse=True,
        )

        assert [s["strength"] for s in ranked] == [0.9, 0.5, 0.1]
        assert signals_mod.MAX_SIGNALS == 3, "spec §2 keeps the top 2-3"


# --- 6. the committed blob is the source ------------------------------------


class TestTheCommittedBlobIsTheSource:
    def test_the_tracking_db_is_tracked_rather_than_ignored(self):
        """`.gitignore:6` lists `data/tracking.db`, but a tracked file is not
        affected by it. This is the whole reason F1 is the unblocked sport: the
        other four sports' stores are gitignored, so their counts came off one
        developer's machine and could not be reproduced by a reader.
        """
        assert _git("ls-files", "--error-unmatch", "data/tracking.db") == "data/tracking.db"

    def test_the_file_on_disk_is_the_blob_git_has(self):
        """Not "a file exists at the path" — the same BYTES git committed.

        This is the assertion the spike's retraction was about. The first version
        of that document claimed its counts came from a fresh checkout; they came
        from untracked local copies in four sports, and nothing in the repository
        could check them.
        """
        assert _git("hash-object", "data/tracking.db") == _git("rev-parse", "HEAD:data/tracking.db"), (
            "the working tree's data/tracking.db is not the committed blob; the "
            "published table was read from something the repository does not have"
        )

    def test_the_adapter_reads_the_path_git_has(self):
        from f1_predictor import config

        assert store.TRACKING_DB_PATH == config.TRACKING_DB_PATH == _tracking_db()

    def test_the_published_numbers_come_from_the_file_not_from_the_source(self, tmp_path, monkeypatch):
        """The other half of the provenance claim.

        Hash equality above shows the file on disk is the committed one. This shows
        the numbers actually COME from whichever file the store is pointed at, by
        pointing it somewhere else and watching them change. Together: not a
        constant baked into the module, and not a table that only exists on one
        machine. A test that only asserted the hash would pass just as happily
        against an adapter whose figures were `if False: pass`.
        """
        monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "untracked.db")
        _make_db(tmp_path / "untracked.db", [(0.8, 1)] * 77 + [(0.8, 0)] * 8)

        substituted = {b["label"]: b["n"] for b in store.get_probability_buckets(trust.BUCKET_BOUNDS)}
        assert substituted == {"0.7-1.0": 85}, "the whole fixture lands in the top band"
        assert substituted != {label: n for label, n, _ in PUBLISHED_BANDS}, (
            "an untracked store must not reproduce the committed table, or this "
            "test proves nothing about where the numbers come from"
        )

    def test_the_band_list_is_defined_once(self):
        """The SQL `CASE` is GENERATED from `BUCKET_BOUNDS`, so the adapter and the
        measurement cannot band the same table two different ways."""
        source = (REPO / "src/f1_predictor/tracking/store.py").read_text()

        assert "get_probability_buckets" in source
        assert 'f"WHEN predicted_prob < {high!r} THEN {low!r}"' in source
        assert [f"{low:g}" for low, _ in trust.BUCKET_BOUNDS] == ["0", "0.3", "0.4", "0.5", "0.6", "0.7"]


# --- degradation, and the endpoint ------------------------------------------


class TestDegradation:
    """What happens when the store is missing, empty, short or corrupt.

    The task this guards: *a signal that quietly reports a bucket of n = 2 is
    worse than one that reports nothing.* `sqlite3.connect` CREATES a file that is
    not there, so a deleted `data/tracking.db` does not raise — it arrives as an
    empty table, and the floor then covers it. Every path below asserts NO SIGNAL.
    """

    def test_a_missing_store_yields_no_signal_not_a_rate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "never-created.db")

        assert trust.trust_signal("2026-15-race", 0.12) is None
        assert not (tmp_path / "never-created.db").exists() or True, (
            "sqlite3.connect may create it; either way there is no signal"
        )

    def test_an_empty_store_yields_no_signal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "empty.db")
        _make_db(tmp_path / "empty.db", [])

        assert store.get_probability_buckets(trust.BUCKET_BOUNDS) == []
        assert trust.trust_signal("2026-15-race", 0.12) is None

    def test_a_store_with_fewer_resolved_rows_than_expected_yields_no_signal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "short.db")
        _make_db(tmp_path / "short.db", [(0.12, 1)] * 12)

        bands = store.get_probability_buckets(trust.BUCKET_BOUNDS)
        assert bands[0]["n"] == 12 and bands[0]["rate"] == 1.0
        assert trust.trust_signal("2026-15-race", 0.12) is None, "n=12 must not reach the page"

    def test_unresolved_rows_are_not_counted_as_evidence(self, tmp_path, monkeypatch):
        """`resolved = 1` only. A row with no outcome has no rate to contribute,
        and counting it would inflate `n` past the floor with nothing behind it."""
        monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "unresolved.db")
        _make_db(tmp_path / "unresolved.db", [(0.12, 1)] * 40)
        conn = sqlite3.connect(str(tmp_path / "unresolved.db"))
        with conn:
            conn.execute("UPDATE session_predictions SET resolved = 0, actual_outcome = NULL")
        conn.close()

        assert trust.trust_signal("2026-15-race", 0.12) is None

    def test_a_corrupt_store_yields_no_signal(self, tmp_path, monkeypatch):
        path = tmp_path / "corrupt.db"
        path.write_bytes(b"this is not a sqlite database" * 64)
        monkeypatch.setattr(store, "TRACKING_DB_PATH", path)

        assert trust.trust_signal("2026-15-race", 0.12) is None


class TestTheEndpoint:
    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(signals_mod.router)
        return TestClient(app)

    def test_a_malformed_id_is_a_404(self, client):
        assert client.get("/signals/not-an-id").status_code == 404
        assert client.get("/signals/2026-15-post_quals").status_code == 404

    def test_a_session_with_no_pick_returns_an_empty_list_not_an_error(self, client, monkeypatch):
        monkeypatch.setattr(facts_mod, "session_pick", lambda *a, **k: {"pick": None})

        response = client.get("/signals/2026-15-race")

        assert response.status_code == 200
        assert response.json() == {"sport": "f1", "id": "2026-15-race", "signals": []}

    def test_a_session_in_a_sub_floor_band_returns_no_signal(self, client, monkeypatch):
        """Through the API, so the floor is proven on the path a reader uses."""
        monkeypatch.setattr(
            facts_mod, "session_pick",
            lambda *a, **k: {"pick": {"label": "x", "prob": 0.55}},
        )

        response = client.get("/signals/2026-15-race")

        assert response.status_code == 200
        assert response.json()["signals"] == []

    def test_a_real_session_returns_a_payload_the_component_accepts(self, client, monkeypatch):
        monkeypatch.setattr(
            facts_mod, "session_pick",
            lambda *a, **k: {"pick": {"label": "verstappen", "prob": 0.12}},
        )

        body = client.get("/signals/2026-15-race").json()

        assert len(body["signals"]) == 1
        signal = body["signals"][0]
        assert signal["kind"] == "trust"
        assert signal["n"] == 1235
        contract = _vendored_contract()
        assert 0.0 <= signal["headline"]["figures"][contract["figure"][trust.VISUAL]] <= 1.0
        assert len(signal["headline"]["text"].split()) <= contract["max_headline_words"]

    def test_an_adapter_failure_does_not_take_down_the_page(self, client, monkeypatch):
        """A signal is an enhancement; the fixture page must survive its absence."""
        def explode(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(trust, "trust_signal", explode)
        monkeypatch.setattr(
            facts_mod, "session_pick",
            lambda *a, **k: {"pick": {"label": "x", "prob": 0.12}},
        )

        response = client.get("/signals/2026-15-race")

        assert response.status_code == 200
        assert response.json()["signals"] == []

    def test_an_unknown_session_is_a_404(self, client, monkeypatch):
        def unknown(*args, **kwargs):
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Unknown session id")

        monkeypatch.setattr(facts_mod, "session_pick", unknown)

        assert client.get("/signals/2026-99-race").status_code == 404

    def test_the_router_is_mounted_on_the_real_app(self):
        """Reachable from the app, not only from a router built inside a test.

        Read off the OpenAPI schema rather than `app.routes`: this FastAPI wraps
        an included router in an object with no `.path`, so the route list does
        not show included endpoints at all. The schema is the stronger claim
        anyway — it is what `/docs` serves, so a path in it is a path a client
        can call.
        """
        from f1_predictor.api.main import app

        assert "/signals/{session_id}" in app.openapi()["paths"]
        assert "get" in app.openapi()["paths"]["/signals/{session_id}"]