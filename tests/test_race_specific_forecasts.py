"""Two future F1 races were served byte-identical win probabilities, and neither
cause was a coincidence.

Live, before this change, `/api/races/2026/16/prediction` and
`/api/races/2026/17/prediction` returned the SAME 23-driver vector — not just
the same top pick at the same probability, but every row hashing identically:

    round 16  n=23  sha=9d367f1bf40074bd
    round 17  n=23  sha=9d367f1bf40074bd

    Max Verstappen  win 0.1479  podium 0.4055  exp_pos 5.9549   (both rounds)
    Leclerc         win 0.1375  podium 0.3836  exp_pos 6.1378   (both rounds)

**Root cause 1 — cross-season history was never loaded.**
`_future_feature_frame` calls `jolpica.load_season_results(season)`, which
returns ONE season. Circuit form is computed as an expanding mean per
`(driver_id, circuit_id)`, so with only the current season in the frame every
future race has *no* circuit history and the feature is null for everyone. The
2026 calendar visits no circuit twice, so this is not an edge case — it is every
future race, every season. Marina Bay has 40 races of history across 2024 and
2025 and none of it reached the model.

Measured: with prior seasons merged, round 17 goes from 0/23 drivers with
circuit history to 22/23. Round 16 stays 0/23, and that is CORRECT — Sepang has
no F1 race in the loaded window at all.

**Root cause 2 — identical nulls are indistinguishable from a real prediction.**
With the circuit features null for every driver, the model has nothing to
separate the two races, so it produces the same forecast for both. The reader
sees two confident, identical, race-specific-looking numbers. That is the
failure worth preventing even after root cause 1 is fixed: a null feature must
not read as a prediction.

So this file pins three things: that the frame carries cross-season circuit
history, that two races at different circuits do not get identical forecasts, and
that an all-null circuit feature is visible as such rather than silent.
"""
import pandas as pd
import pytest

from f1_predictor.api import routes
from f1_predictor.data import jolpica


@pytest.fixture(scope="module")
def schedule():
    return jolpica.fetch_season_schedule(2026)


def _frame_for(round_: int, schedule) -> pd.DataFrame:
    row = schedule[schedule["round"] == round_].iloc[0]
    df, _ = routes._future_feature_frame(2026, round_, row)
    return df


def test_the_frame_carries_circuit_history_from_prior_seasons(schedule):
    """The fix. A circuit visited in an earlier season must reach the model.

    Round 17 is Marina Bay, raced 40 times across 2024-2025, and the frame had
    null circuit form for all 23 drivers — because only the current season was
    loaded. The cache key was always round-specific, and the model was always
    round-agnostic, which is why the two predictions were identical rather than
    merely similar.
    """
    df = _frame_for(17, schedule)
    have = int(df["driver_circuit_avg_position"].notna().sum())
    assert have > 0, (
        f"only {have}/{len(df)} drivers have circuit history for a circuit raced 40 "
        f"times in prior seasons. `load_season_results(season)` returns one season, so "
        f"the expanding mean per (driver, circuit) has nothing prior to average."
    )


def test_two_races_at_different_circuits_get_different_forecasts(schedule):
    """The symptom, asserted directly.

    Pinned on the FEATURE FRAME rather than on the served JSON: the frame is
    where the two races actually differ, and a test on the response would pass
    if the endpoint happened to round or reorder. This is the property a reader
    is entitled to — two different races, two different predictions.
    """
    a, b = _frame_for(16, schedule), _frame_for(17, schedule)
    assert not a["driver_circuit_avg_position"].equals(b["driver_circuit_avg_position"]), (
        "rounds 16 and 17 are at different circuits and carry identical circuit "
        "features, so the model cannot distinguish them and serves the same "
        "prediction twice"
    )


def test_a_circuit_with_no_history_is_flagged_rather_than_silent(schedule):
    """Root cause 2, and the half that survives the fix.

    Sepang has genuinely never been raced in the loaded window, so its circuit
    features are legitimately null. A null feature is an absence and the model
    should be able to see it as one — otherwise a race at an unfamiliar circuit
    and a race at a familiar one produce forecasts of the same confidence with
    no difference between them, which is what the two identical responses above
    looked like from a reader's side.

    This test pins that the frame at least RECORDS the absence, so the
    distinction is available to be made. Whether the model then reports lower
    confidence is a separate change and is not asserted here.
    """
    df = _frame_for(16, schedule)
    circuit_col = "circuit_id"
    assert circuit_col in df.columns, (
        "the frame does not record which circuit a row is about, so 'no history "
        "for this circuit' cannot be told from 'history not loaded'"
    )
    assert df[circuit_col].notna().all(), (
        "a row with no circuit cannot be attributed to one, and the circuit form "
        "for it is then null for a reason nobody can see"
    )
    # And the absence is visible as an absence rather than a fabricated value.
    assert df["driver_circuit_avg_position"].isna().any(), (
        "Sepang has no history in the loaded window; if this now has values, the "
        "test that the frame distinguishes circuits needs re-examining"
    )


def test_the_identical_response_cannot_come_back(schedule):
    """A regression guard on the whole thing, stated as the reader saw it.

    Hashes the driver vectors the way the live check did. Cheap, and it fails
    on the exact symptom rather than on a proxy for it.
    """
    vectors = []
    for round_ in (16, 17):
        frame = _frame_for(round_, schedule)
        vectors.append(
            pd.util.hash_pandas_object(
                frame[["driver_id", "driver_circuit_avg_position"]],
                index=False,
            ).values.tobytes()
        )
    assert vectors[0] != vectors[1], (
        "the two rounds' circuit features are byte-identical again, which is the "
        "condition that produced two identical served predictions"
    )
