"""Test for quali_history features."""

import pandas as pd
import pytest

from f1_predictor.features.quali_history import compute_quali_features


def _fake_quali_frame():
    """Create a fake qualifying DataFrame with two drivers per team."""
    rows = []
    # 2023 season, 3 rounds
    for rnd in range(1, 4):
        # Team 1: ver (faster) and per (slower)
        rows.append({
            "season": 2023, "round": rnd, "driver_id": "ver", "constructor_id": "red_bull",
            "quali_position": 1, "q1": "1:30.000", "q2": "1:29.500", "q3": "1:29.000"
        })
        rows.append({
            "season": 2023, "round": rnd, "driver_id": "per", "constructor_id": "red_bull",
            "quali_position": 2, "q1": "1:30.500", "q2": "1:30.000", "q3": "1:29.500"
        })
        # Team 2: ham (faster) and rus (slower)
        rows.append({
            "season": 2023, "round": rnd, "driver_id": "ham", "constructor_id": "mercedes",
            "quali_position": 3, "q1": "1:31.000", "q2": "1:30.500", "q3": "1:30.000"
        })
        rows.append({
            "season": 2023, "round": rnd, "driver_id": "rus", "constructor_id": "mercedes",
            "quali_position": 4, "q1": "1:31.500", "q2": "1:31.000", "q3": "1:30.500"
        })
    return pd.DataFrame(rows)


def test_compute_quali_features_basic():
    """Test that quali features are computed correctly."""
    quali_df = _fake_quali_frame()
    result = compute_quali_features(quali_df)

    # Check columns (quali_position is not included to avoid merge conflicts)
    expected_cols = [
        "season", "round", "driver_id", "constructor_id",
        "quali_avg_position_3", "quali_avg_gap_to_pole_3", "teammate_quali_gap"
    ]
    assert list(result.columns) == expected_cols

    # Check shape: 2 teams * 2 drivers * 3 rounds = 12 rows
    assert len(result) == 12

    # Check that round 1 has NaN for rolling features (no prior data)
    round1 = result[result["round"] == 1]
    assert round1["quali_avg_position_3"].isna().all()
    assert round1["quali_avg_gap_to_pole_3"].isna().all()

    # Check that round 2 has values for rolling features (1 prior round)
    round2 = result[result["round"] == 2].set_index("driver_id")
    # ver: position 1 in round 1, so avg should be 1.0
    assert round2.loc["ver", "quali_avg_position_3"] == 1.0
    # per: position 2 in round 1, so avg should be 2.0
    assert round2.loc["per", "quali_avg_position_3"] == 2.0

    # Check teammate gap: ver is faster than per by 0.5s
    # ver's teammate gap should be negative (faster than teammate)
    assert round2.loc["ver", "teammate_quali_gap"] < 0
    # per's teammate gap should be positive (slower than teammate)
    assert round2.loc["per", "teammate_quali_gap"] > 0

    # Check round 3: average of rounds 1 and 2
    round3 = result[result["round"] == 3].set_index("driver_id")
    assert round3.loc["ver", "quali_avg_position_3"] == 1.0  # (1+1)/2 = 1
    assert round3.loc["per", "quali_avg_position_3"] == 2.0  # (2+2)/2 = 2


def test_compute_quali_features_empty():
    """Test empty DataFrame handling."""
    empty_df = pd.DataFrame(columns=["season", "round", "driver_id", "constructor_id", "quali_position", "q1", "q2", "q3"])
    result = compute_quali_features(empty_df)
    expected_cols = [
        "season", "round", "driver_id",
        "quali_avg_position_3", "quali_avg_gap_to_pole_3", "teammate_quali_gap"
    ]
    assert list(result.columns) == expected_cols
    assert len(result) == 0


def test_compute_quali_features_missing_times():
    """Test handling of missing qualifying times."""
    rows = [
        {"season": 2023, "round": 1, "driver_id": "ver", "constructor_id": "red_bull", "quali_position": 1, "q1": None, "q2": None, "q3": None},
        {"season": 2023, "round": 1, "driver_id": "per", "constructor_id": "red_bull", "quali_position": 2, "q1": "1:30.000", "q2": "1:29.500", "q3": "1:29.000"},
    ]
    quali_df = pd.DataFrame(rows)
    result = compute_quali_features(quali_df)

    # Should still compute teammate gap for the driver with times
    # but gap for ver should be NaN since no time
    round1 = result[result["round"] == 1].set_index("driver_id")
    # The function should handle missing times gracefully
    assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])