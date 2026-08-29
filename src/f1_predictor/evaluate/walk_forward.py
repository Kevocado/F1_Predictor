"""walk_forward.py — season-by-season walk-forward validation for the two
models/race_outcome.py candidates, mirroring PL_Predictor's
evaluate/walk_forward.py: `prepare_folds()` builds the full tier-augmented
feature frame ONCE and slices it by season (no lookahead — every feature is
already computed via shift(1)/expanding past-only aggregation before
slicing), so `evaluate_candidate()` can be called repeatedly against the
same prepared folds without redoing feature engineering.

Scored on the POST_QUALIFYING tier specifically — the fullest pre-race
information state, and the closest analogue to a "final pre-match
prediction" in PL_Predictor's own walk-forward metric. DNF probability is
deliberately excluded from this evaluation (dnf_prob=None when simulating)
so the strength-model race-off isn't conflated with the separately-trained
DNF model's own error — models/dnf.py is evaluated on its own terms
(models/manifest.py logs its held-out accuracy directly, not through here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..data import jolpica
from ..features import session_state
from ..features.build import build_training_frame
from ..models import dnf as dnf_model
from ..models import race_outcome


def prepare_folds(seasons: list[int] | None = None, min_train_seasons: int = 3) -> list[dict]:
    seasons = seasons or jolpica.default_seasons()
    df, feature_cols = build_training_frame(seasons=seasons)
    df = df[df["tier"] == session_state.TIER_POST_QUALIFYING].reset_index(drop=True)

    folds = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]
        train_df = df[df["season"].isin(train_seasons)]
        val_df = df[df["season"] == val_season]
        if train_df.empty or val_df.empty:
            continue
        folds.append(
            {"val_season": val_season, "train_df": train_df, "val_df": val_df, "feature_cols": feature_cols}
        )
    return folds


def _race_groups(df: pd.DataFrame):
    for (season, round_), race in df.groupby(["season", "round"]):
        yield season, round_, race


def evaluate_candidate(
    folds: list[dict],
    candidate: str,
    n_trials: int = 3000,
    feature_cols_override: list[str] | None = None,
    hyperparams: dict | None = None,
) -> pd.DataFrame:
    """`candidate`: "elo" or "xgb_ranker". One row per fold: val_season,
    n_races, log_loss/brier on the win market (P(win) vs. who actually
    won), the metric models/manifest.py::train_all races the two
    candidates on. `feature_cols_override` trains/predicts on a SUBSET of
    the fold's normal feature columns (evaluate/feature_ablation.py's "is
    this feature actually pulling weight" check); `hyperparams` overrides
    the ranker's defaults (evaluate/tune_hyperparams.py's Optuna search).
    Both no-ops when omitted."""
    rows = []
    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        if feature_cols_override is not None:
            feature_cols = feature_cols_override

        ranker = (
            race_outcome.train_ranker(train_df, feature_cols, hyperparams=hyperparams)
            if candidate == "xgb_ranker"
            else None
        )

        y_true: list[int] = []
        y_pred: list[float] = []
        for _season, _round_, race in _race_groups(val_df):
            if candidate == "elo":
                theta = race_outcome.theta_from_elo_strength(race_outcome.elo_strengths(race))
            else:
                theta = race_outcome.theta_from_xgb_scores(
                    race_outcome.xgb_scores_for_race(ranker, race, feature_cols)
                )

            sim = race_outcome.simulate_race(theta, dnf_prob=None, n_trials=n_trials, seed=42).set_index("driver_id")
            winners = race.loc[race["position"] == 1, "driver_id"]
            winner = winners.iloc[0] if not winners.empty else None
            for d in race["driver_id"]:
                if d not in sim.index:
                    continue
                y_true.append(1 if d == winner else 0)
                y_pred.append(sim.loc[d, "p_win"])

        rows.append(
            {
                "val_season": fold["val_season"],
                "n_races": val_df[["season", "round"]].drop_duplicates().shape[0],
                "log_loss": log_loss(y_true, y_pred, labels=[0, 1]) if y_true else np.nan,
                "brier": brier_score_loss(y_true, y_pred) if y_true else np.nan,
            }
        )
    return pd.DataFrame(rows)


def walk_forward_validate(seasons: list[int] | None = None, min_train_seasons: int = 3) -> dict[str, pd.DataFrame]:
    folds = prepare_folds(seasons, min_train_seasons)
    return {
        "elo": evaluate_candidate(folds, "elo"),
        "xgb_ranker": evaluate_candidate(folds, "xgb_ranker"),
    }


def evaluate_dnf_model(folds: list[dict], hyperparams: dict | None = None) -> pd.DataFrame:
    """Held-out log-loss/brier for the DNF reliability model across the
    same walk-forward folds prepare_folds() builds — evaluated
    independently of the race-outcome candidates (a driver's DNF is a
    separate model, not part of the strength-score race-off above). This
    is what evaluate/tune_hyperparams.py tunes the DNF model against."""
    rows = []
    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        clf = dnf_model.train_dnf_model(train_df, feature_cols, hyperparams=hyperparams)
        y_true = val_df["dnf"].astype(int)
        y_pred = clf.predict_proba(val_df[feature_cols])[:, 1]
        rows.append(
            {
                "val_season": fold["val_season"],
                "n": len(val_df),
                "log_loss": log_loss(y_true, y_pred, labels=[0, 1]),
                "brier": brier_score_loss(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)
