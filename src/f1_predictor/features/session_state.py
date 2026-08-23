"""session_state.py — the "how much of this weekend do we know yet" tiering
requirement 4 asks for. Deliberately ONE trained model (see
models/race_outcome.py), not four: a race weekend's prediction sharpens by
progressively unmasking columns as real data arrives, using XGBoost's
native missing-value handling rather than swapping between separately
trained models per stage.

The trick that makes this actually work is `tier_augment`: historical
training data always has grid/qualifying columns eventually populated, so
without augmentation the model would never see what "not known yet"
legitimately looks like. `tier_augment` replays every historical row once
per tier with not-yet-known columns masked to NaN, so the model learns the
real pattern instead of only ever seeing complete rows.

LIVE is deliberately not a tier here — the moment a race goes green, this
module hands off entirely to models/live_win_prob.py (Phase 3), a
different model trained on a different data shape (lap-by-lap state, not
weekend-level features).
"""

from __future__ import annotations

import pandas as pd

TIER_PRE_WEEKEND = "pre_weekend"
TIER_POST_PRACTICE = "post_practice"
TIER_POST_QUALIFYING = "post_qualifying"

TIER_ORDER = [TIER_PRE_WEEKEND, TIER_POST_PRACTICE, TIER_POST_QUALIFYING]

# Columns newly unlocked at each tier (cumulative — a later tier keeps
# everything an earlier one had). Phase 1 has no FastF1-derived practice
# session columns yet (data/fastf1_client.py is Phase 3), so POST_PRACTICE
# currently unlocks nothing beyond PRE_WEEKEND — the mechanism is ready for
# Phase 3 to slot fp_* columns in here without touching the rest of this
# module or models/race_outcome.py.
TIER_COLUMNS: dict[str, list[str]] = {
    TIER_PRE_WEEKEND: [],
    TIER_POST_PRACTICE: [],  # Phase 3: fp_best_lap_rank, fp_long_run_pace_delta, fp_tyre_deg_estimate
    TIER_POST_QUALIFYING: ["grid", "quali_position", "quali_gap_to_pole"],
}


def all_tier_gated_columns() -> list[str]:
    return [c for cols in TIER_COLUMNS.values() for c in cols]


def columns_known_by(tier: str) -> list[str]:
    """All tier-gated columns unlocked at or before `tier`."""
    known: list[str] = []
    for t in TIER_ORDER:
        known += TIER_COLUMNS[t]
        if t == tier:
            break
    return known


def tier_augment(df: pd.DataFrame, tier_gated_columns: list[str] | None = None) -> pd.DataFrame:
    """Replay every row once per tier, masking not-yet-known columns to NaN
    for that tier. The target/outcome columns are untouched — every replay
    keeps the same real outcome, since this only changes what the model is
    told it knew going in, not what happened."""
    tier_gated_columns = tier_gated_columns or all_tier_gated_columns()
    frames = []
    for tier in TIER_ORDER:
        known = set(columns_known_by(tier))
        copy = df.copy()
        for col in tier_gated_columns:
            if col not in known and col in copy.columns:
                # A scalar float NaN (not pd.NA) so numeric columns stay
                # float64 instead of being upcast to object dtype, which
                # XGBoost's DMatrix construction rejects outright.
                copy[col] = float("nan")
        copy["tier"] = tier
        frames.append(copy)
    return pd.concat(frames, ignore_index=True)


def current_tier(schedule_row: pd.Series, now: pd.Timestamp | None = None) -> str:
    """Which tier an upcoming race weekend is currently in, from wall-clock
    time against that weekend's session datetimes."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    quali_dt = schedule_row.get("qualifying_datetime")
    if pd.notna(quali_dt) and now >= quali_dt:
        return TIER_POST_QUALIFYING

    practice_cols = ["fp1_datetime", "fp2_datetime", "fp3_datetime", "sprint_quali_datetime"]
    practice_dts = [schedule_row.get(c) for c in practice_cols if pd.notna(schedule_row.get(c))]
    if practice_dts and now >= min(practice_dts):
        return TIER_POST_PRACTICE

    return TIER_PRE_WEEKEND


def mask_to_tier(feature_row: pd.Series, tier: str, tier_gated_columns: list[str] | None = None) -> pd.Series:
    """Prediction-time counterpart to tier_augment: given a fully-known
    feature row and a target tier, mask whatever wouldn't have been known
    yet at that tier. Used by evaluate/backtest.py to replay a real,
    already-completed race "as of" an earlier tier."""
    tier_gated_columns = tier_gated_columns or all_tier_gated_columns()
    known = set(columns_known_by(tier))
    row = feature_row.copy()
    for col in tier_gated_columns:
        if col not in known and col in row.index:
            row[col] = float("nan")
    return row
