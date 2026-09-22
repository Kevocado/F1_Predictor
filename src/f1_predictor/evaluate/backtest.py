"""backtest.py — Phase 1 checkpoint: replay a real, already-completed
Grand Prix using only its POST_QUALIFYING-tier pre-race information (grid
position included, nothing from the race itself), and print the resulting
probability table next to what actually happened. This is the
"directionally sane" sanity check requirement 1's models need before
anything gets served from an API: the actual winner should be clearly
elevated in P(win) relative to grid position, actual podium finishers
should rank near the top of P(podium), no absurd inversions.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..data import jolpica
from ..features import session_build
from ..features import session_state
from ..features.build import build_training_frame
from ..models import dnf as dnf_model
from ..models import manifest as manifest_module
from ..models import race_outcome
from ..models import session_outcome


def backtest_race(season: int, round_: int, seasons: list[int] | None = None) -> pd.DataFrame:
    """No-lookahead by construction twice over: each row's own feature
    values already only reflect that driver's state entering the race
    (features/elo.py etc.), AND the models used here are additionally
    re-trained on only strictly-earlier races — a per-row-correct feature
    isn't enough on its own, since fitting a model on races *after* the
    target race would still leak future patterns into the model itself."""
    seasons = seasons or jolpica.default_seasons()
    df, feature_cols = build_training_frame(seasons=seasons)
    post_quali = df[df["tier"] == session_state.TIER_POST_QUALIFYING]

    before = (post_quali["season"] < season) | ((post_quali["season"] == season) & (post_quali["round"] < round_))
    train_df = post_quali[before]
    race_df = post_quali[(post_quali["season"] == season) & (post_quali["round"] == round_)]
    if race_df.empty:
        raise ValueError(f"No post-qualifying data found for {season} round {round_}")

    manifest = manifest_module.load_manifest()
    candidate = manifest["race_outcome_candidate"]
    # Same hyperparameters the currently-deployed model was actually
    # trained with (models/manifest.py::train_all persists whichever it
    # used, tuned or default) — an honest replay should match production,
    # not silently fall back to defaults if the deployed model is tuned.
    ranker_params = manifest.get("tuned_ranker_params")
    dnf_params = manifest.get("tuned_dnf_params")

    if candidate == "elo":
        theta = race_outcome.theta_from_elo_strength(race_outcome.elo_strengths(race_df))
    else:
        ranker = race_outcome.train_ranker(train_df, feature_cols, hyperparams=ranker_params)
        theta = race_outcome.theta_from_xgb_scores(race_outcome.xgb_scores_for_race(ranker, race_df, feature_cols))

    dnf_clf = dnf_model.train_dnf_model(train_df, feature_cols, hyperparams=dnf_params)
    dnf_prob = dnf_model.predict_dnf_prob(dnf_clf, race_df, feature_cols)

    sim = race_outcome.simulate_race(theta, dnf_prob=dnf_prob, n_trials=10000, seed=0)

    actual = race_df.set_index("driver_id")[["constructor_id", "position", "grid", "dnf", "points"]]
    result = sim.set_index("driver_id").join(actual, how="left").reset_index()
    return result.sort_values("p_win", ascending=False).reset_index(drop=True)


def backtest_session(
    season: int, round_: int, session_type: str = "race", seasons: list[int] | None = None
) -> pd.DataFrame:
    """Generalizes backtest_race's no-lookahead replay to the three new
    session types. session_type="race" is a pure passthrough to
    backtest_race (unchanged, still the race model's own path) — this
    function exists so callers (api/routes.py) can use one name regardless
    of session_type."""
    if session_type == "race":
        return backtest_race(season, round_, seasons=seasons)

    seasons = seasons or jolpica.default_seasons()
    spec = session_outcome.SESSION_SPECS[session_type]
    df, feature_cols = session_build.build_session_training_frame(session_type, seasons=seasons)

    before = (df["season"] < season) | ((df["season"] == season) & (df["round"] < round_))
    train_df = df[before]
    session_df = df[(df["season"] == season) & (df["round"] == round_)]
    if session_df.empty:
        raise ValueError(f"No {session_type} data found for {season} round {round_}")

    ranker = session_outcome.train_session_ranker(train_df, feature_cols, spec)
    dnf_clf = None
    if spec.has_dnf:
        dnf_clf = dnf_model.train_dnf_model(train_df, feature_cols)

    sim = session_outcome.predict_session(ranker, session_df, feature_cols, spec, dnf_clf=dnf_clf, n_trials=10000, seed=0)

    actual_cols = ["constructor_id", spec.target_column]
    if spec.has_dnf:
        actual_cols.append("dnf")
    actual = session_df.set_index("driver_id")[actual_cols].rename(columns={spec.target_column: "position"})
    result = sim.set_index("driver_id").join(actual, how="left").reset_index()
    return result.sort_values("p_win", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a completed GP with only pre-race data.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()

    result = backtest_race(args.season, args.round)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 30)
    cols = ["driver_id", "grid", "position", "p_win", "p_podium", "p_points_finish", "p_dnf", "dnf"]
    print(result[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    winners = result.loc[result["position"] == 1, "driver_id"]
    if not winners.empty:
        winner = winners.iloc[0]
        by_driver = result.set_index("driver_id")
        winner_p_win = by_driver.loc[winner, "p_win"]
        rank = int((result["p_win"] > winner_p_win).sum()) + 1
        print(f"\nActual winner: {winner}")
        print(f"Model's P(win) for actual winner: {winner_p_win:.3f}")
        print(f"Model's rank of actual winner by P(win): {rank}")

    podium = set(result.loc[result["position"].isin([1, 2, 3]), "driver_id"])
    top3_by_model = set(result.nlargest(3, "p_podium")["driver_id"])
    print(f"Actual podium: {podium} | model's top-3 by P(podium): {top3_by_model}")


if __name__ == "__main__":
    main()
