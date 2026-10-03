"""trust.py — the `trust` signal: how often this model has been right at the
confidence it is quoting.

Spec `2026-10-01-fixture-signals-design.md` §4, in its own words: *"When this
model says ~59% on a home favourite it has been right X% of the time (n
games)."* The probability is the session's own headline pick; `X`, `n` and the
band come from the committed `data/tracking.db`, bucketed on the same
`predicted_prob` the pick was stored with.

## THE FLOOR, and which layer owns it. Read this before changing `TRUST_MIN_N`.

Spec §4: *"Renders only at n >= 30; below that the row says nothing, never a
rate from a handful of games."* **This adapter owns it: a band under the floor
produces NO signal, not a signal with a small `n`.**

The shared component also enforces a floor — `SPEC_MIN_N` in `SignalRows.tsx`,
and `signalIsDrawn` DROPS a sub-floor rate rather than throwing it. That is not a
second opinion on the same question, because the two answer different ones:

  * The component's floor answers **"may this be DRAWN?"** Its own file header
    says so: *"the rule is about what may be drawn, not about what may be
    computed. An adapter can honestly hold a bucket of 12 games and report it;
    what it may not do is put a rate built from it on the page."*
  * This adapter's floor answers **"may this be SENT?"** — and `GET /signals`
    is read by more than the component. Phase 4 feeds the payloads to the AI
    "so what" writer and its validator admits a figure that appears in a signal
    payload. A sub-floor rate emitted here would therefore be quotable as a
    claim about 27 games, which is the thing §4 forbids, and the component's
    floor would not catch it because by then the payload is already gone.

So the adapter is the primary and the component is the backstop. They agree
today (both 30) and neither can lower the other's guard: the component clamps
`floorOf = max(SPEC_MIN_N, minN)`, so a caller can only ever RAISE it, and
`TRUST_MIN_N` is this adapter's own constant. Effective floor is therefore
`max(TRUST_MIN_N, SPEC_MIN_N)` = 30.

## `TRUST_MIN_N` is named for where the number comes from.

It is **not** a shared identifier and appears nowhere else in this repo. The
plan (`2026-10-02-fixture-signals-phases-1-4.md`, Phase 1) says "Constants per
sport: `TRUST_MIN_N = 30`", and the number is spec §4's "Renders only at
n >= 30", which §9 decision 3 lists as reversible by "one constant per sport".
The name is the plan's, kept so the adapter and the component can be read
against the same sentence.

## `strength` — the formula and WHY it points that way.

    gap      = |hit_rate - mean_predicted_prob|          the reliability gap
    se       = sqrt(mean_predicted_prob * (1 - mean_predicted_prob) / n)
    z        = gap / se                                 miscalibration in SEs
    strength = min(1.0, z / 3.0)

**It RISES with the reliability gap, and that is the direction the signal wants.**

The temptation is to rank by confidence — a 70%-said bucket feels more important
than a 12%-said one — but that is the wrong quantity here. Spec §1's complaint is
that everything the page says is a paraphrase of the model's own output: the
page already shows the model saying 12%. What it cannot show is **whether the
model's 12% has ever been worth 12%**, and the entire information content of this
row is the distance between the two. A perfectly calibrated band (`gap == 0`)
says nothing the page did not already say, so it scores `strength == 0` and sorts
last — which is the right answer, not a defect: §2's rule is "a signal exists
only if it adds something the page does not already show", and zero surprise is
exactly that.

**The gap is divided by the standard error so that a gap is scored against how
surprising it is, not how big it looks.** Raw gaps are not comparable across
bands because the bands hold different numbers of picks. The 0.4-0.5 band has a
gap of 0.052 on 44 picks — under one standard error, which is to say nothing a
coin could not produce. The 0.0-0.3 band has a smaller gap of 0.021 but on 1235
picks, over two. Scored on the raw gap the thin band would outrank the deep one and
a reader would be shown a 44-game coincidence in preference to a 1235-game
miscalibration; scored on `z`, the order reverses, which is the correct one.

**Note the direction in `n`, because it is easy to state backwards: at a fixed
gap, `strength` RISES as `n` grows.** `se` shrinks like `1/sqrt(n)`, so the same
discrepancy becomes steadily harder to attribute to chance — a five-point gap on
44 picks is noise and the same gap on 5000 picks is a fact. This comment
originally said the opposite ("strength falls as evidence accumulates"), which
the code does not do; `test_a_gap_on_few_picks_ranks_below_the_same_gap_on_many`
was written to pin the claim and failed, which is how the sentence was caught.

`z / 3.0` puts a full-strength row at three standard errors, which is the
conventional "this would happen by chance about 0.3% of the time" line. It is a
scale, not a threshold: nothing is gated on it, and it saturates rather than
clipping, so two bands can never tie at an arbitrary ceiling.

Measured on the committed blob (only the bands that clear the floor, since
`0.5-0.6` does not): gap 0.021 -> 0.81, gap 0.008 -> 0.07, gap 0.052 -> 0.23,
gap 0.101 -> 0.47, gap 0.017 -> 0.13.

## THE HEADLINE, and why its wording is load-bearing.

`SignalRows` THROWS `HeadlineFigureMismatchError` when a row's words do not
state the figure its bar draws, so an over-precise headline ("74.4%") is a 500
on the page, not a warning. The bar draws `headline.figures.rate` at
`fmt.pct`'s whole-percent rounding, so `headline()` states that same rounded
percent and nothing finer.

The wording avoids two failure modes on purpose:

  * **a sportsbook price.** "Backing picks near 30%", "value at 0.35", anything
    with odds vocabulary. F1 has no odds feed at all (`line_gap` is NFL/CFB
    only, spec §4), so a price reading would be describing a market this project
    has never seen.
  * **a certainty.** "is right 74% of the time" in the present tense would
    promise a future rate. The headline is in the past tense and names the
    sample it is drawn from, which is what makes it a record rather than a
    forecast.

It also does NOT restate the model's current pick, for spec §4's own reason: the
page already shows that number, and a signal that repeats it adds nothing.
"""

from __future__ import annotations

import math

from ..tracking import store

#: Spec §4's floor, per sport (spec §9 decision 3; plan Phase 1). See the module
#: docstring for which layer owns it and how it interacts with the component's
#: own `SPEC_MIN_N`.
TRUST_MIN_N = 30

#: The probability bands, as `(low, high)` with the text the signal quotes.
#: Half-open `[low, high)` except the last, which is closed — see
#: `store.get_probability_buckets`, which GENERATES its SQL from this list.
#:
#: These are the bands the committed-blob figures were re-derived on, and they
#: are not tuned to look good: 0.3 is the only cut wide enough to hold the F1 win
#: and pole markets (whose probabilities top out at 0.182), and 0.4/0.5/0.6/0.7
#: are the standard tenths above it. The one boundary that hurts is 0.5-0.6,
#: which lands at n=27 and is therefore never rendered at all.
#:
#: KEPT AS ONE LIST ON PURPOSE. `store.get_probability_buckets` builds its `CASE`
#: from these pairs, and the test that re-derives the published table from the
#: committed blob is handed the same list. A band edge cannot be widened in the
#: adapter and left narrow in the measurement.
BUCKET_BOUNDS: list[tuple[float, float]] = [
    (0.0, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 1.0),
]

#: `visual` for this signal, and therefore which figure in `headline.figures`
#: the component draws. `reliability_bar` draws `figures.rate`, which
#: `SignalRows.signalFigure` requires to be a share in [0, 1].
VISUAL = "reliability_bar"

#: Figure name `VISUAL` draws. Read out of the component's own `FIGURE` map by
#: the tests rather than trusted from here.
RATE_FIGURE = "rate"


def band_bounds(prob: float) -> tuple[float, float] | None:
    """The `(low, high)` band a probability falls in, or None when it is not one.

    The upper edge is INCLUSIVE on the last band and exclusive elsewhere, which is
    `store.get_probability_buckets`'s own convention — the two must agree or a
    probability sitting on an edge is reported from one band and read from
    another.

    Returns the band's EDGES rather than its label, so the caller joins the two on
    numbers. Matching on the formatted string would make a display format part of
    the data path, where a change to it would silently stop finding bands.
    """
    if prob is None or isinstance(prob, bool) or not isinstance(prob, (int, float)):
        return None
    if not math.isfinite(prob):
        return None
    for index, (low, high) in enumerate(BUCKET_BOUNDS):
        last = index == len(BUCKET_BOUNDS) - 1
        if low <= prob < high:
            return (low, high)
        # The top edge is inclusive so the highest band is closed. The `low` test
        # is repeated rather than assumed: written as `last and prob <= high` it
        # also swallowed every probability BELOW the band -- a -0.1 was reported as
        # the top band, which would have put a 0.7-1.0 record under a negative
        # probability. Both edges are checked on every band.
        if last and low <= prob == high:
            return (low, high)
    return None


def band_label(prob: float) -> str | None:
    """The band's own display text, or None when `prob` is in no band."""
    bounds = band_bounds(prob)
    return None if bounds is None else f"{bounds[0]:.1f}-{bounds[1]:.1f}"


def strength(rate: float, mean_predicted: float, n: int) -> float:
    """How much a reader should care about this band's calibration. 0..1.

    The formula and its direction are in the module docstring. In short: it rises
    with the reliability gap, and at a fixed gap it rises with `n` as well — a
    discrepancy measured on more picks is harder to write off as chance. The point
    of dividing by the standard error is that a large gap on few picks is scored as
    the noise it is, instead of outranking a smaller gap measured properly.

    `se` is zero only when `mean_predicted` is exactly 0 or 1, which no recorded
    F1 probability is (the shipped range is 0.0026 to 0.9157). It is handled
    rather than left to divide by zero: a band whose stated probabilities are all
    one thing has no spread to measure against, so a gap there is as stark as a
    gap can get.
    """
    if n <= 0:
        return 0.0
    gap = abs(rate - mean_predicted)
    variance = mean_predicted * (1.0 - mean_predicted)
    if variance <= 0.0:
        return 1.0 if gap > 0.0 else 0.0
    se = math.sqrt(variance / n)
    if se <= 0.0:
        return 1.0 if gap > 0.0 else 0.0
    return min(1.0, (gap / se) / 3.0)


def headline(rate: float, mean_predicted: float) -> str:
    """The one line, at most 12 words, stating the rate the bar draws.

    `SignalRows.assertFigureIsStated` compares NUMBERS: the stated figure must
    equal one of the percents `fmt.pct` can print for `rate`, at whole-percent
    precision and with a `%` sign. So the percent here is `round(rate * 100)`,
    never `rate * 100` unrounded — `0.738` beside a bar reading `74%` would be
    refused, and `74.4%` would be refused as a figure the bar does not draw.

    The comparison is against the WHOLE percent only, so the model-side figure is
    rounded the same way before it is quoted. Two bands can therefore print the
    same stated probability and different realised rates; that is the truth at
    this precision, and it is why the record's own `n` is what the reader is
    given alongside it.

    **`~` on the stated figure is load-bearing, not decoration.** `mean_predicted`
    is the MEAN of a whole band, and the bands are 0.1 wide at the top and 0.3
    wide at the bottom — so "When the model says 10%" would be claiming picks of
    exactly 10% when the row actually covers everything from 0.0 to 0.3. The
    tilde is spec §4's own notation for exactly this ("When this model says ~59%"),
    and it keeps the sentence true of the band. The band's own edges are in
    `source`, which the component renders verbatim.

    Two things deliberately NOT in the headline: a range like "0.0-0.3" (the
    component's figure scanner reads `0.0-0.3` as a NEGATIVE 0.3, and the row
    would be refused for stating a figure it never drew), and the page's own
    current pick (spec §2 — a signal that repeats what the page already shows
    adds nothing).

    Wording rules, and the reason for each, are in the module docstring: no
    sportsbook vocabulary (F1 has no odds feed), no present-tense certainty (this
    is a record, not a forecast).
    """
    stated = f"{round(mean_predicted * 100):g}"
    landed = f"{round(rate * 100):g}"
    return f"When the model says ~{stated}%, its picks landed {landed}% of the time"


def trust_signal(game_id: str, prob: float) -> dict | None:
    """The `trust` signal for a session whose headline pick carries `prob`, or
    None when the honest answer is that there is no signal.

    None is returned, rather than a signal with a thin `n`, when:

      * `prob` is not a finite number, or falls outside every band;
      * the tracking store cannot be read at all (missing file that
        `sqlite3.connect` would silently CREATE empty, a corrupt file, a
        `session_predictions` table that is not there); or
      * the band `prob` falls in holds fewer than `TRUST_MIN_N` resolved picks.

    That last one is the floor, and it is why a missing or emptied
    `data/tracking.db` yields no signal rather than a rate from two games: the
    bands come out empty, every one of them is under the floor, and the same
    guard covers both. A silent degradation to "n = 2, right 50% of the time"
    is the specific outcome §4 forbids, and it is unreachable from here.
    """
    label = band_bounds(prob)
    if label is None:
        return None
    low, high = label

    try:
        bands = store.get_probability_buckets(BUCKET_BOUNDS)
    except Exception:
        # sqlite3.connect CREATES a missing file and _connect then creates the
        # table, so a deleted or unreadable store does not raise — it arrives
        # here as zero rows, which the floor below already handles. This catch is
        # for the case that DOES raise: a file that is not a database, or one
        # locked by another writer. Either way there is no honest signal.
        return None

    band = next((b for b in bands if b["low"] == low and b["high"] == high), None)
    if band is None or band["n"] < TRUST_MIN_N or band["rate"] is None:
        return None

    rate, mean_predicted = band["rate"], band["mean_predicted"]
    return {
        "kind": "trust",
        "sport": "f1",
        "game_id": str(game_id),
        "headline": {
            "text": headline(rate, mean_predicted),
            # Only the figure the bar draws, plus the model-side probability the
            # sentence quotes. §3 leaves `figures` open and the component reads
            # the one its `visual` names, so carrying both is allowed and
            # carrying a THIRD unnamed number is not what the field is for.
            "figures": {RATE_FIGURE: rate, "stated_prob": mean_predicted},
        },
        "n": int(band["n"]),
        "source": (
            f"{band['n_resolved']:,} resolved picks in this project's tracking.db, "
            f"{band['label']} probability band, all markets pooled"
        ),
        "as_of": _iso(band["as_of"]),
        "strength": strength(rate, mean_predicted, int(band["n"])),
        "pre_kickoff_only": True,
        "visual": VISUAL,
    }


def _iso(value) -> str:
    """A stored timestamp as UTC ISO-8601, or "" when it cannot be read.

    `resolved_at` is written by `reconcile_session_predictions` as an aware UTC
    ISO string, so this is a normalisation rather than a parse; the `fromisoformat`
    fallback is for the older spellings that carry an offset instead of a `Z`.
    An unreadable stamp yields "" rather than a guessed date — §3 says a row
    carries a source and a date "or neither", and this component will render a
    blank date as a dash rather than as "Jan 1 1970".
    """
    if not value:
        return ""
    text = str(value)
    if text.endswith("Z"):
        return text
    from datetime import datetime, timezone

    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")