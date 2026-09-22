import pandas as pd

from f1_predictor.features import session_state


def _row(**overrides) -> pd.Series:
    base = {
        "fp1_datetime": pd.Timestamp("2024-04-19T03:30:00Z"),
        "fp2_datetime": None,
        "fp3_datetime": None,
        "sprint_quali_datetime": pd.Timestamp("2024-04-19T07:30:00Z"),
        "sprint_datetime": pd.Timestamp("2024-04-20T03:00:00Z"),
        "qualifying_datetime": pd.Timestamp("2024-04-20T07:00:00Z"),
        "race_datetime": pd.Timestamp("2024-04-21T00:00:00Z"),
    }
    base.update(overrides)
    return pd.Series(base)


def test_current_session_tier_pre_weekend_before_any_session():
    now = pd.Timestamp("2024-04-19T00:00:00Z")
    assert session_state.current_session_tier(_row(), now) == session_state.TIER_PRE_WEEKEND


def test_current_session_tier_post_practice_after_fp1_before_sprint_quali():
    now = pd.Timestamp("2024-04-19T05:00:00Z")
    assert session_state.current_session_tier(_row(), now) == session_state.TIER_POST_PRACTICE


def test_current_session_tier_post_sprint_qualifying_after_sprint_quali_before_sprint():
    now = pd.Timestamp("2024-04-19T08:00:00Z")
    assert session_state.current_session_tier(_row(), now) == session_state.TIER_POST_SPRINT_QUALIFYING


def test_current_session_tier_post_sprint_after_sprint_before_qualifying():
    now = pd.Timestamp("2024-04-20T04:00:00Z")
    assert session_state.current_session_tier(_row(), now) == session_state.TIER_POST_SPRINT


def test_current_session_tier_post_qualifying_after_qualifying():
    now = pd.Timestamp("2024-04-20T08:00:00Z")
    assert session_state.current_session_tier(_row(), now) == session_state.TIER_POST_QUALIFYING


def test_current_session_tier_non_sprint_weekend_skips_sprint_tiers():
    # A regular weekend has no sprint_quali_datetime/sprint_datetime at all.
    row = _row(sprint_quali_datetime=None, sprint_datetime=None)
    now = pd.Timestamp("2024-04-19T05:00:00Z")
    assert session_state.current_session_tier(row, now) == session_state.TIER_POST_PRACTICE
