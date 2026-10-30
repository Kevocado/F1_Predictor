"""Guards for features/grid_interaction.py (F1-2). Pure functions, no network."""

import math

import pandas as pd
import pytest

from f1_predictor.features import grid_interaction as gi


def _race(season, rnd, circuit, rows):
    return pd.DataFrame(
        [{"season": season, "round": rnd, "circuit_id": circuit, "driver_id": f"d{g}",
          "grid": g, "position": p} for g, p in rows]
    )


def test_loo_removes_own_result_from_counts(monkeypatch):
    monkeypatch.setattr(gi, "_fit_k", lambda cells, prior: 5.0)  # hold k fixed across both calls
    # Two races at circuit "a". The 2021 pole sitter wins; the 2022 pole sitter finishes 4th.
    train = pd.concat([
        _race(2021, 1, "a", [(1, 1), (2, 2), (3, 3)]),
        _race(2022, 1, "a", [(1, 4), (2, 1), (3, 2)]),
    ], ignore_index=True)
    own_row = train.iloc[[0]]  # 2021 pole, a win
    with_loo = gi.grid_features(train, own_row, loo=True)
    # Without the own race, the only (a, pole) cell is the 2022 pole, which did not win.
    without_own = gi.grid_features(train.iloc[1:], own_row, loo=False)
    assert with_loo.loc[0, "grid_curve_win"] == pytest.approx(without_own.loc[0, "grid_curve_win"])
    assert with_loo.loc[0, "grid_base_win"] == pytest.approx(without_own.loc[0, "grid_base_win"])
    # And the own-result-included value must differ (proves the subtraction happens).
    included = gi.grid_features(train, own_row, loo=False)
    assert included.loc[0, "grid_curve_win"] != pytest.approx(with_loo.loc[0, "grid_curve_win"])


def test_sparse_cell_is_shrunk_toward_global_curve():
    # Many circuits with a pole that always wins, plus one circuit with a single race
    # where the pole did NOT win. The sparse cell must not sit at its own rate (0).
    rows = [_race(2021, r, f"c{r}", [(1, 1), (2, 5)]) for r in range(1, 9)]
    rows.append(_race(2021, 9, "solo", [(1, 4), (2, 1)]))
    train = pd.concat(rows, ignore_index=True)
    target = train[train["circuit_id"] == "solo"]
    f = gi.grid_features(train, target, loo=False)
    curve, base = f.iloc[0]["grid_curve_win"], f.iloc[0]["grid_base_win"]
    assert 0.0 < curve < base  # pulled down from the global pole rate, but not to its own 0


def test_missing_grid_falls_back_to_global():
    train = _race(2021, 1, "a", [(1, 1), (2, 2)])
    target = pd.DataFrame({"circuit_id": ["a"], "grid": [float("nan")], "position": [1.0]}, index=[7])
    f = gi.grid_features(train, target, loo=False)
    assert list(f.index) == [7]
    assert not math.isnan(f.loc[7, "grid_curve_top3"])


def test_quali_gap_pct_and_teammate_gap():
    base = {"season": 2024, "round": 1, "q2": None, "q3": None}
    q = pd.DataFrame([
        {**base, "driver_id": "a", "constructor_id": "x", "q1": "1:30.000"},
        {**base, "driver_id": "b", "constructor_id": "x", "q1": "1:31.000"},
        {**base, "driver_id": "c", "constructor_id": "y", "q1": "1:32.000"},
    ])
    out = gi.quali_gap_features(q).set_index("driver_id")
    assert out.loc["a", "quali_gap_pct"] == pytest.approx(0.0)
    assert out.loc["c", "quali_gap_pct"] == pytest.approx(2.0 / 90.0 * 100)
    assert out.loc["a", "teammate_gap_pct"] == pytest.approx(-1.0 / 90.0 * 100)
    assert out.loc["b", "teammate_gap_pct"] == pytest.approx(1.0 / 90.0 * 100)
    # a constructor with only one driver has no teammate gap
    assert math.isnan(out.loc["c", "teammate_gap_pct"])


def test_flag_default_off():
    assert gi.GRID_INTERACTION_ENABLED is False
