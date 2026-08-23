"""elo.py — driver Elo ratings via per-race round-robin pairwise
comparisons, keyed on driver_id so a rating survives a driver changing
constructors mid-career (constructor/car strength is tracked separately in
features/team_strength.py). Standard adaptation of Elo to a field of N
simultaneous competitors, following the approach used by public F1-Elo
prior art (matthewperron/f1-elo, joemarlo/F1-Elo, cbowdon/F1Ranking): a
race is scored as if every pair of drivers played a 1v1 match, decided by
relative finishing position, and each driver's rating moves by K times the
gap between their actual round-robin score and what their rating predicted.
"""

from __future__ import annotations

import pandas as pd

INITIAL_ELO = 1500.0
DRIVER_K_FACTOR = 24.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _update_race(ratings: dict[str, float], race_results: pd.DataFrame, k: float = DRIVER_K_FACTOR) -> None:
    """Mutates `ratings` in place from one race's classified finishing
    order (lower `position` = better; unclassified drivers are dropped from
    this update). O(N^2) pairwise comparisons per race — N is at most ~24
    drivers, negligible even replayed across a full multi-season history."""
    race_results = race_results.dropna(subset=["position"]).sort_values("position")
    driver_ids = race_results["driver_id"].tolist()
    positions = dict(zip(driver_ids, race_results["position"]))
    n = len(driver_ids)
    if n < 2:
        return

    for d in driver_ids:
        ratings.setdefault(d, INITIAL_ELO)

    deltas = {d: 0.0 for d in driver_ids}
    for i, di in enumerate(driver_ids):
        for dj in driver_ids[i + 1 :]:
            expected_i = expected_score(ratings[di], ratings[dj])
            actual_i = 1.0 if positions[di] < positions[dj] else 0.0
            delta = k * (actual_i - expected_i)
            deltas[di] += delta
            deltas[dj] -= delta

    # Average the pairwise deltas so K stays comparable across field sizes
    # (a 20-driver round-robin has ~10x the pairwise updates of a 1v1 match).
    for d in driver_ids:
        ratings[d] += deltas[d] / (n - 1)


def _iter_races(results_df: pd.DataFrame):
    race_keys = results_df[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    for _, key in race_keys.iterrows():
        season, round_ = int(key["season"]), int(key["round"])
        yield season, round_, results_df[(results_df["season"] == season) & (results_df["round"] == round_)]


def compute_elo_history(results_df: pd.DataFrame) -> pd.DataFrame:
    """The core no-lookahead feature builder: for every (season, round,
    driver_id) row, the driver's Elo rating *entering* that race (before
    this race's own result updates it), processed in strict chronological
    order so a training row never sees information from its own outcome.

    Returns columns: season, round, driver_id, elo_pre_race.
    """
    ratings: dict[str, float] = {}
    rows = []
    for season, round_, race in _iter_races(results_df):
        for d in race["driver_id"]:
            rows.append(
                {"season": season, "round": round_, "driver_id": d, "elo_pre_race": ratings.get(d, INITIAL_ELO)}
            )
        _update_race(ratings, race)
    return pd.DataFrame(rows)


def latest_ratings(results_df: pd.DataFrame) -> dict[str, float]:
    """Final Elo rating per driver after replaying all of `results_df` in
    order — used to seed predictions for an upcoming race."""
    ratings: dict[str, float] = {}
    for _, _, race in _iter_races(results_df):
        _update_race(ratings, race)
    return ratings
