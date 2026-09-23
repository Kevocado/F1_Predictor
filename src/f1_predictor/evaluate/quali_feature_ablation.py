"""quali_feature_ablation.py — answers the question this whole session-
predictors build started from: does a predicted-qualifying-position
feature improve the RACE model's PRE_WEEKEND/POST_PRACTICE-tier
predictions (the two tiers where quali_position is still genuinely
unknown, i.e. NaN)? See
docs/superpowers/specs/2026-09-22-session-predictors-design.md §6.

Trains the race model twice via a simple walk-forward split (same
no-lookahead discipline evaluate/backtest.py uses): once on the existing
feature set, once with an added `predicted_quali_position` column (from
the qualifying predictor's own out-of-fold predictions, never overriding
the real quali_position once POST_QUALIFYING data exists), and compares
Brier score across win/podium/points_finish/dnf on PRE_WEEKEND/
POST_PRACTICE rows only — the two tiers a predicted proxy could plausibly
help, since POST_QUALIFYING already has the real value.

This is a standalone evaluation tool (mirrors tune_hyperparams.py) — not
imported by manifest.py or api/routes.py. Run manually; if
result["recommendation"] == "add", add "predicted_quali_position" to
features/build.py::FEATURE_COLUMNS and wire the qualifying predictor's
output into build_training_frame as a follow-up (not done automatically
by this script — a human decision point, consistent with how this
project treats every other model-selection choice).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data import jolpica
from ..features import build as build_features
from ..features import session_build
from ..features import session_state
from ..models import dnf as dnf_model
from ..models import race_outcome
from ..models import session_outcome

_PRE_QUALI_TIERS = (session_state.TIER_PRE_WEEKEND, session_state.TIER_POST_PRACTICE)
_MARKETS = [("win", "p_win"), ("podium", "p_podium"), ("points_finish", "p_points_finish"), ("dnf", "p_dnf")]


def _brier_by_market(sim: pd.DataFrame, actual: pd.DataFrame) -> dict[str, float]:
    """actual must have columns driver_id/position/dnf, already restricted
    to the rows being scored (a single race/tier group, so driver_id is
    unique)."""
    merged = sim.merge(actual, on="driver_id", how="inner")
    scores = {}
    for market, prob_col in _MARKETS:
        if market == "dnf":
            outcome = merged["dnf"].astype(int)
        elif market == "win":
            outcome = (merged["position"] == 1).astype(int)
        elif market == "podium":
            outcome = (merged["position"] <= 3).astype(int)
        else:
            outcome = (merged["position"] <= 10).astype(int)
        scores[market] = float(((merged[prob_col] - outcome) ** 2).mean())
    return scores


def _predicted_quali_positions(seasons: list[int]) -> pd.DataFrame:
    """Out-of-fold-ish predicted quali position per (season, round,
    driver_id): for each round, train the qualifying predictor on every
    STRICTLY EARLIER round only (no-lookahead), predict that round. Rounds
    with no earlier training data (the first few of the earliest season)
    are skipped — left NaN downstream, which XGBoost already tolerates."""
    quali_spec = session_outcome.SESSION_SPECS["qualifying"]
    df, feature_cols = session_build.build_session_training_frame("qualifying", seasons=seasons)
    if df.empty:
        return pd.DataFrame(columns=["season", "round", "driver_id", "predicted_quali_position"])

    rows = []
    for (season, round_), _ in df.groupby(["season", "round"]):
        before = (df["season"] < season) | ((df["season"] == season) & (df["round"] < round_))
        train_df = df[before]
        if train_df.empty:
            continue
        target_df = df[(df["season"] == season) & (df["round"] == round_)]
        ranker = session_outcome.train_session_ranker(train_df, feature_cols, quali_spec)
        sim = session_outcome.predict_session(ranker, target_df, feature_cols, quali_spec, n_trials=2000)
        for _, r in sim.iterrows():
            rows.append(
                {"season": season, "round": round_, "driver_id": r["driver_id"], "predicted_quali_position": r["expected_position"]}
            )
    return pd.DataFrame(rows)


def run_ablation(seasons: list[int] | None = None) -> dict:
    seasons = seasons or jolpica.default_seasons()
    df, feature_cols = build_features.build_training_frame(seasons=seasons)
    predicted_quali = _predicted_quali_positions(seasons)

    df_with_feature = df.merge(predicted_quali, on=["season", "round", "driver_id"], how="left")
    df_with_feature["predicted_quali_position"] = df_with_feature["predicted_quali_position"].where(
        df_with_feature["quali_position"].isna(), df_with_feature["quali_position"]
    )
    feature_cols_with = feature_cols + ["predicted_quali_position"]

    race_keys = sorted(df[["season", "round"]].drop_duplicates().itertuples(index=False, name=None))
    if len(race_keys) < 2:
        nan_scores = {market: float("nan") for market, _ in _MARKETS}
        return {"with_feature": nan_scores, "without_feature": nan_scores, "recommendation": "no_change"}
    split_idx = max(1, len(race_keys) * 3 // 4)
    train_keys = set(race_keys[:split_idx])
    test_keys = set(race_keys[split_idx:]) or {race_keys[-1]}

    scores_without = _score_variant(df, feature_cols, train_keys, test_keys)
    scores_with = _score_variant(df_with_feature, feature_cols_with, train_keys, test_keys)

    total_without = sum(scores_without.values())
    total_with = sum(scores_with.values())
    recommendation = "add" if total_with < total_without else "no_change"

    return {"with_feature": scores_with, "without_feature": scores_without, "recommendation": recommendation}


def _score_variant(df: pd.DataFrame, feature_cols: list[str], train_keys: set, test_keys: set) -> dict[str, float]:
    post_quali = df[df["tier"] == session_state.TIER_POST_QUALIFYING]
    train_df = post_quali[[(s, r) in train_keys for s, r in zip(post_quali["season"], post_quali["round"])]]
    if train_df.empty:
        return {market: float("nan") for market, _ in _MARKETS}

    ranker = race_outcome.train_ranker(train_df, feature_cols)
    dnf_clf = dnf_model.train_dnf_model(train_df, feature_cols)

    pre_quali = df[df["tier"].isin(_PRE_QUALI_TIERS)]
    pre_quali_test = pre_quali[[(s, r) in test_keys for s, r in zip(pre_quali["season"], pre_quali["round"])]]
    if pre_quali_test.empty:
        return {market: float("nan") for market, _ in _MARKETS}

    per_market_scores: dict[str, list[float]] = {market: [] for market, _ in _MARKETS}
    for (_season, _round, _tier), race_test in pre_quali_test.groupby(["season", "round", "tier"]):
        scores = race_outcome.xgb_scores_for_race(ranker, race_test, feature_cols)
        theta = race_outcome.theta_from_xgb_scores(scores)
        dnf_prob = dnf_model.predict_dnf_prob(dnf_clf, race_test, feature_cols)
        sim = race_outcome.simulate_race(theta, dnf_prob=dnf_prob, n_trials=5000, seed=0)
        actual = race_test[["driver_id", "position", "dnf"]]
        race_scores = _brier_by_market(sim, actual)
        for market, _prob_col in _MARKETS:
            per_market_scores[market].append(race_scores[market])

    return {
        market: (float(np.mean(vals)) if vals else float("nan"))
        for market, vals in per_market_scores.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Does a predicted-qualifying feature improve the race model?")
    parser.add_argument("--seasons", type=int, nargs="*", default=None)
    args = parser.parse_args()

    result = run_ablation(seasons=args.seasons)
    print("Brier score by market (lower is better), PRE_WEEKEND/POST_PRACTICE rows only:")
    print(f"{'market':<15}{'without':>12}{'with':>12}")
    for market, _ in _MARKETS:
        print(f"{market:<15}{result['without_feature'][market]:>12.4f}{result['with_feature'][market]:>12.4f}")
    print(f"\nRecommendation: {result['recommendation']}")
    if result["recommendation"] == "add":
        print("-> Add 'predicted_quali_position' to features/build.py::FEATURE_COLUMNS.")
    else:
        print("-> No change to the race model's feature set.")


if __name__ == "__main__":
    main()
