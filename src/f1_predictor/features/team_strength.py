"""team_strength.py — constructor ("car") strength rating, computed
separately from driver Elo (features/elo.py) so the model can separate
"this driver is good" from "this car is fast" as distinct signals, and so a
driver's rating doesn't reset when they change teams.

Same pairwise round-robin Elo mechanism as driver Elo, but compared on
constructor points scored in a race (sum of both cars) rather than
finishing position — points naturally handle a double-DNF (0 points, no
special-casing needed) and reflect that a constructor fields two cars per
race, not one.
"""

from __future__ import annotations

import pandas as pd

from .elo import INITIAL_ELO, expected_score

CONSTRUCTOR_K_FACTOR = 16.0  # softer than driver K: a two-car sample per race is noisier


def _race_constructor_points(race_results: pd.DataFrame) -> pd.Series:
    return race_results.groupby("constructor_id")["points"].sum()


def _update_race(ratings: dict[str, float], race_results: pd.DataFrame, k: float = CONSTRUCTOR_K_FACTOR) -> None:
    points = _race_constructor_points(race_results)
    constructors = points.index.tolist()
    n = len(constructors)
    if n < 2:
        return
    for c in constructors:
        ratings.setdefault(c, INITIAL_ELO)

    deltas = {c: 0.0 for c in constructors}
    for i, ci in enumerate(constructors):
        for cj in constructors[i + 1 :]:
            expected_i = expected_score(ratings[ci], ratings[cj])
            if points[ci] == points[cj]:
                actual_i = 0.5
            else:
                actual_i = 1.0 if points[ci] > points[cj] else 0.0
            delta = k * (actual_i - expected_i)
            deltas[ci] += delta
            deltas[cj] -= delta

    for c in constructors:
        ratings[c] += deltas[c] / (n - 1)


def _iter_races(results_df: pd.DataFrame):
    race_keys = results_df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    for _, key in race_keys.iterrows():
        season, round_ = int(key["season"]), int(key["round"])
        yield season, round_, results_df[(results_df["season"] == season) & (results_df["round"] == round_)]


def compute_team_strength_history(results_df: pd.DataFrame) -> pd.DataFrame:
    """Per (season, round, constructor_id): constructor strength rating
    entering that race (no-lookahead, same discipline as
    elo.compute_elo_history)."""
    ratings: dict[str, float] = {}
    rows = []
    for season, round_, race in _iter_races(results_df):
        for c in race["constructor_id"].unique():
            rows.append(
                {
                    "season": season,
                    "round": round_,
                    "constructor_id": c,
                    "team_strength_pre_race": ratings.get(c, INITIAL_ELO),
                }
            )
        _update_race(ratings, race)
    return pd.DataFrame(rows)


def latest_ratings(results_df: pd.DataFrame) -> dict[str, float]:
    ratings: dict[str, float] = {}
    for _, _, race in _iter_races(results_df):
        _update_race(ratings, race)
    return ratings
