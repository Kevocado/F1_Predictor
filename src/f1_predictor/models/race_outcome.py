"""race_outcome.py — requirement 1's core: per-driver win/podium/points-
finish probability for a single race.

Two candidate *strength* scorers are race-off'd against each other (see
evaluate/walk_forward.py, models/manifest.py::train_all) — whichever wins
on held-out log-loss is what actually gets served:
  - "elo": features/elo.py's driver Elo + features/team_strength.py's
    constructor rating, summed. No training needed at prediction time —
    strengths are already-computed feature columns.
  - "xgb_ranker": an XGBoost learning-to-rank model (`rank:pairwise`) on
    the full session-state-tiered feature frame.

Either candidate's per-driver strengths feed the SAME Plackett-Luce Monte
Carlo simulator (`simulate_race`): repeatedly softmax-and-remove over the
field to draw a full finishing order, many times, so p_win/p_podium/
p_points_finish are internally consistent by construction (derived from one
coherent distribution) rather than three separately-fit numbers that could
contradict each other. DNF probability (models/dnf.py) is injected into
each trial before the order is drawn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..features.elo import INITIAL_ELO

POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

RANKER_PARAMS = dict(
    objective="rank:pairwise",
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    tree_method="hist",
)


# ---------------------------------------------------------------------------
# Candidate A: Elo + team strength
# ---------------------------------------------------------------------------


def elo_strengths(race_df: pd.DataFrame) -> dict[str, float]:
    """Driver Elo + constructor strength, already-computed columns from
    features/build.py — no training/fitting needed, just a lookup."""
    g = race_df.set_index("driver_id")
    combined = g["elo_pre_race"].fillna(INITIAL_ELO) + g["team_strength_pre_race"].fillna(INITIAL_ELO)
    return combined.to_dict()


def theta_from_elo_strength(strengths: dict[str, float]) -> dict[str, float]:
    """Converts Elo-scale strengths into Plackett-Luce weights using base
    10 / divisor 400 — the same constants features/elo.py's
    expected_score() uses, so the resulting pairwise win probability
    theta_i / (theta_i + theta_j) exactly matches the Elo model's own
    expected_score(i, j). Subtracting the race's max strength before
    exponentiating is a numerically-stable no-op: scaling every theta by
    the same positive constant doesn't change Plackett-Luce's normalized
    draw probabilities."""
    max_s = max(strengths.values())
    return {d: 10 ** ((s - max_s) / 400.0) for d, s in strengths.items()}


# ---------------------------------------------------------------------------
# Candidate B: XGBoost learning-to-rank
# ---------------------------------------------------------------------------


def train_ranker(train_df: pd.DataFrame, feature_cols: list[str], hyperparams: dict | None = None) -> xgb.XGBRanker:
    """`rank:pairwise` on the tier-augmented frame, grouped per (season,
    round, tier) so the model learns within-race relative order — the same
    race replayed at a different tier is treated as its own group, since
    the point of tiering is that fewer columns are known there.
    `hyperparams` overrides RANKER_PARAMS (e.g. from
    evaluate/tune_hyperparams.py's Optuna search) — omit for production
    defaults."""
    df = train_df.sort_values(["season", "round", "tier"]).reset_index(drop=True)
    group_keys = ["season", "round", "tier"]
    field_size = df.groupby(group_keys)["driver_id"].transform("count")
    label = (field_size - df["position"].fillna(field_size)).clip(lower=0)
    group_sizes = df.groupby(group_keys, sort=False).size().values

    params = {**RANKER_PARAMS, **(hyperparams or {})}
    model = xgb.XGBRanker(**params)
    model.fit(df[feature_cols], label, group=group_sizes)
    return model


def xgb_scores_for_race(ranker: xgb.XGBRanker, race_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    scores = ranker.predict(race_df[feature_cols])
    return dict(zip(race_df["driver_id"], scores))


def theta_from_xgb_scores(scores: dict[str, float]) -> dict[str, float]:
    max_s = max(scores.values())
    return {d: float(np.exp(s - max_s)) for d, s in scores.items()}


# ---------------------------------------------------------------------------
# Plackett-Luce Monte Carlo simulation (shared by both candidates)
# ---------------------------------------------------------------------------


def draw_classification(
    theta: dict[str, float], dnf_prob: dict[str, float], rng: np.random.Generator
) -> tuple[list[str], set[str]]:
    """One Monte Carlo trial: sample DNFs, then Plackett-Luce-sample the
    finishing order of survivors via sequential softmax-and-remove, then
    append DNF'd drivers at the tail (their relative order among each other
    doesn't materially affect any of the reported markets, so it's
    shuffled for symmetry rather than modeled). Returns (ordered
    driver_ids, set of driver_ids who DNF'd this trial)."""
    drivers = list(theta.keys())
    dnfd = {d for d in drivers if rng.random() < dnf_prob.get(d, 0.0)}
    survivors = [d for d in drivers if d not in dnfd]

    order: list[str] = []
    remaining = {d: theta[d] for d in survivors}
    while remaining:
        keys = list(remaining.keys())
        weights = np.array([remaining[k] for k in keys])
        probs = weights / weights.sum()
        choice = rng.choice(keys, p=probs)
        order.append(choice)
        del remaining[choice]

    dnf_list = list(dnfd)
    rng.shuffle(dnf_list)
    order.extend(dnf_list)
    return order, dnfd


def simulate_race(
    theta: dict[str, float],
    dnf_prob: dict[str, float] | None = None,
    n_trials: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Runs `draw_classification` n_trials times and aggregates into one
    row per driver: p_win, p_podium, p_points_finish (top 10), p_dnf,
    expected_position, expected_points — requirement 1's four targets (DNF
    reported both here and standalone from models/dnf.py directly)."""
    dnf_prob = dnf_prob or {}
    rng = np.random.default_rng(seed)
    drivers = list(theta.keys())
    win = {d: 0 for d in drivers}
    podium = {d: 0 for d in drivers}
    points_finish = {d: 0 for d in drivers}
    dnf_ct = {d: 0 for d in drivers}
    pos_sum = {d: 0 for d in drivers}
    points_sum = {d: 0.0 for d in drivers}

    for _ in range(n_trials):
        order, dnfd = draw_classification(theta, dnf_prob, rng)
        for pos, d in enumerate(order, start=1):
            pos_sum[d] += pos
            if pos == 1:
                win[d] += 1
            if pos <= 3:
                podium[d] += 1
            if pos <= 10:
                points_finish[d] += 1
                points_sum[d] += POINTS_TABLE[pos]
        for d in dnfd:
            dnf_ct[d] += 1

    rows = [
        {
            "driver_id": d,
            "p_win": win[d] / n_trials,
            "p_podium": podium[d] / n_trials,
            "p_points_finish": points_finish[d] / n_trials,
            "p_dnf": dnf_ct[d] / n_trials,
            "expected_position": pos_sum[d] / n_trials,
            "expected_points": points_sum[d] / n_trials,
        }
        for d in drivers
    ]
    return pd.DataFrame(rows).sort_values("p_win", ascending=False).reset_index(drop=True)
