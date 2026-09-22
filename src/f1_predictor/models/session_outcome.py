"""session_outcome.py — the shared engine for the three non-race session
predictors (sprint qualifying, qualifying, sprint), reusing
race_outcome.py's train_ranker/simulate_race (already target-agnostic —
the label comes from whichever DataFrame column is passed in) instead of
duplicating the ranker/Plackett-Luce machinery per session type. See
docs/superpowers/specs/2026-09-22-session-predictors-design.md §1.

Deliberately does NOT cover "race" — the existing race model
(features/build.py, evaluate/backtest.py::backtest_race,
models/manifest.py's race/DNF training block) stays on its own path
unchanged; this module only adds the three new session types.

Each session type has exactly ONE fixed feature-availability point
(feature_cutoff_tier is a label for the API/tracking `tier` field, not a
tier_augment-style masking mechanism) — unlike the race model's
progressively-sharpening 3-tier chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import xgboost as xgb

from ..features import session_state
from . import dnf as dnf_model
from . import race_outcome
from .championship_projection import SPRINT_POINTS_TABLE


@dataclass(frozen=True)
class SessionSpec:
    session_type: str
    target_column: str
    has_dnf: bool
    points_table: dict[int, float] | None
    points_finish_cutoff: int | None
    feature_cutoff_tier: str


SESSION_SPECS: dict[str, SessionSpec] = {
    "sprint_qualifying": SessionSpec(
        session_type="sprint_qualifying",
        target_column="sprint_quali_position",
        has_dnf=False,
        points_table=None,
        points_finish_cutoff=None,
        feature_cutoff_tier=session_state.TIER_POST_PRACTICE,
    ),
    "qualifying": SessionSpec(
        session_type="qualifying",
        target_column="quali_position",
        has_dnf=False,
        points_table=None,
        points_finish_cutoff=None,
        feature_cutoff_tier=session_state.TIER_POST_PRACTICE,
    ),
    "sprint": SessionSpec(
        session_type="sprint",
        target_column="position",
        has_dnf=True,
        points_table=SPRINT_POINTS_TABLE,
        points_finish_cutoff=8,
        feature_cutoff_tier=session_state.TIER_POST_SPRINT_QUALIFYING,
    ),
}


def train_session_ranker(
    train_df: pd.DataFrame, feature_cols: list[str], spec: SessionSpec, hyperparams: dict | None = None
) -> xgb.XGBRanker:
    """Delegates to race_outcome.train_ranker, which groups by
    (season, round, tier) and ranks on a column literally named
    "position" — renaming spec.target_column to "position" and stamping a
    constant "tier" column (spec.feature_cutoff_tier; there's only one
    tier per session type here, so the groupby still groups correctly by
    (season, round)) lets that function run completely unmodified."""
    df = train_df.copy()
    df["position"] = df[spec.target_column]
    df["tier"] = spec.feature_cutoff_tier
    return race_outcome.train_ranker(df, feature_cols, hyperparams=hyperparams)


def predict_session(
    ranker: xgb.XGBRanker,
    session_df: pd.DataFrame,
    feature_cols: list[str],
    spec: SessionSpec,
    dnf_clf: xgb.XGBClassifier | None = None,
    n_trials: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Same Monte Carlo output shape as race_outcome.simulate_race
    (driver_id, p_win, p_podium, p_points_finish, p_dnf,
    expected_position, expected_points). For has_dnf=False session types,
    dnf_clf is ignored even if passed, and p_dnf is exactly 0.0 for every
    driver (empty dnf_prob dict -> draw_classification never marks anyone
    DNF'd) — callers reinterpret p_win/p_podium/p_points_finish as
    pole/top_3/top_10 for those session types; they don't surface p_dnf at
    all in that case."""
    scores = race_outcome.xgb_scores_for_race(ranker, session_df, feature_cols)
    theta = race_outcome.theta_from_xgb_scores(scores)

    dnf_prob: dict[str, float] = {}
    if spec.has_dnf and dnf_clf is not None:
        dnf_prob = dnf_model.predict_dnf_prob(dnf_clf, session_df, feature_cols)

    return race_outcome.simulate_race(
        theta,
        dnf_prob=dnf_prob,
        n_trials=n_trials,
        seed=seed,
        points_table=spec.points_table,
        points_finish_cutoff=spec.points_finish_cutoff or 10,
    )
