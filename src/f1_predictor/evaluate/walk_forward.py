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

Also evaluates the grid-position lookup baseline and the 2-parameter
calibrated candidate (grid_calibration module).
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


def evaluate_grid_lookup(folds: list[dict], n_trials: int = 3000) -> pd.DataFrame:
    """Evaluate the grid-position lookup baseline on walk-forward folds.

    The grid lookup uses historical win/podium rates by grid position,
    computed from the training seasons of each fold (no leakage).
    """
    from .grid_calibration import compute_grid_lookup, grid_lookup_probs, race_df_groups

    rows = []
    for fold in folds:
        train_df, val_df = fold["train_df"], fold["val_df"]

        win_rates, podium_rates = compute_grid_lookup(train_df)

        y_true: list[int] = []
        y_pred_win: list[float] = []

        for _season, _round_, race in race_df_groups(val_df):
            if race["grid"].isna().all():
                continue
            win_probs, _ = grid_lookup_probs(race, win_rates, podium_rates)

            winners = race.loc[race["position"] == 1, "driver_id"]
            winner = winners.iloc[0] if not winners.empty else None

            for d in race["driver_id"]:
                if d not in win_probs:
                    continue
                y_true.append(1 if d == winner else 0)
                y_pred_win.append(win_probs[d])

        if y_true:
            rows.append({
                "val_season": fold["val_season"],
                "n_races": val_df[["season", "round"]].drop_duplicates().shape[0],
                "log_loss": log_loss(y_true, y_pred_win, labels=[0, 1]) if y_true else np.nan,
                "brier": brier_score_loss(y_true, y_pred_win) if y_true else np.nan,
            })
        else:
            rows.append({
                "val_season": fold["val_season"],
                "n_races": val_df[["season", "round"]].drop_duplicates().shape[0],
                "log_loss": np.nan,
                "brier": np.nan,
            })

    return pd.DataFrame(rows)


def evaluate_calibrated_candidate(
    folds: list[dict],
    candidate: str,
    n_trials: int = 3000,
    feature_cols_override: list[str] | None = None,
    hyperparams: dict | None = None,
) -> pd.DataFrame:
    """Evaluate a candidate with 2-parameter grid calibration on win/podium.

    For each fold, the calibration parameters (a, b) are fitted on the
    training seasons' out-of-fold predictions (no leakage), then applied
    to the validation season. Top-10 (points finish) stays un-layered.
    """
    from .grid_calibration import (
        compute_grid_lookup,
        grid_lookup_probs,
        fit_calibration_params,
        apply_calibration,
        race_df_groups,
    )

    rows = []
    for fold in folds:
        train_df, val_df, feature_cols = fold["train_df"], fold["val_df"], fold["feature_cols"]
        if feature_cols_override is not None:
            feature_cols = feature_cols_override

        # Train the candidate model on training data
        ranker = (
            race_outcome.train_ranker(train_df, feature_cols, hyperparams=hyperparams)
            if candidate == "xgb_ranker"
            else None
        )

        # Compute grid lookup rates from training data
        win_rates, podium_rates = compute_grid_lookup(train_df)

        # Collect out-of-fold predictions on training data for calibration fitting
        train_model_win = []
        train_grid_win = []
        train_true_win = []
        train_model_podium = []
        train_grid_podium = []
        train_true_podium = []

        for _season, _round_, race in race_df_groups(train_df):
            if race["grid"].isna().all():
                continue
            if candidate == "elo":
                theta = race_outcome.theta_from_elo_strength(race_outcome.elo_strengths(race))
            else:
                theta = race_outcome.theta_from_xgb_scores(
                    race_outcome.xgb_scores_for_race(ranker, race, feature_cols)
                )
            sim = race_outcome.simulate_race(theta, dnf_prob=None, n_trials=n_trials, seed=42).set_index("driver_id")
            win_probs_grid, podium_probs_grid = grid_lookup_probs(race, win_rates, podium_rates)

            winners = race.loc[race["position"] == 1, "driver_id"]
            winner = winners.iloc[0] if not winners.empty else None
            podium_finishers = set(race.loc[race["position"] <= 3, "driver_id"])

            for d in race["driver_id"]:
                if d not in sim.index or d not in win_probs_grid:
                    continue
                train_model_win.append(sim.loc[d, "p_win"])
                train_grid_win.append(win_probs_grid[d])
                train_true_win.append(1 if d == winner else 0)
                train_model_podium.append(sim.loc[d, "p_podium"])
                train_grid_podium.append(podium_probs_grid[d])
                train_true_podium.append(1 if d in podium_finishers else 0)

        # Fit calibration parameters on training data out-of-fold predictions
        if len(train_model_win) > 10:
            a_win, b_win = fit_calibration_params(
                np.array(train_model_win), np.array(train_grid_win), np.array(train_true_win)
            )
            a_podium, b_podium = fit_calibration_params(
                np.array(train_model_podium), np.array(train_grid_podium), np.array(train_true_podium)
            )
        else:
            a_win, b_win = 1.0, 0.0
            a_podium, b_podium = 1.0, 0.0

        # Evaluate on validation data with calibration applied
        y_true_win: list[int] = []
        y_pred_win: list[float] = []
        y_true_podium: list[int] = []
        y_pred_podium: list[float] = []

        for _season, _round_, race in race_df_groups(val_df):
            if race["grid"].isna().all():
                continue
            if candidate == "elo":
                theta = race_outcome.theta_from_elo_strength(race_outcome.elo_strengths(race))
            else:
                theta = race_outcome.theta_from_xgb_scores(
                    race_outcome.xgb_scores_for_race(ranker, race, feature_cols)
                )
            sim = race_outcome.simulate_race(theta, dnf_prob=None, n_trials=n_trials, seed=42).set_index("driver_id")
            win_probs_grid, podium_probs_grid = grid_lookup_probs(race, win_rates, podium_rates)

            # Model probabilities
            model_win = {d: sim.loc[d, "p_win"] for d in race["driver_id"] if d in sim.index}
            model_podium = {d: sim.loc[d, "p_podium"] for d in race["driver_id"] if d in sim.index}

            # Apply calibration
            cal_win = apply_calibration(model_win, win_probs_grid, a_win, b_win)
            cal_podium = apply_calibration(model_podium, podium_probs_grid, a_podium, b_podium)

            winners = race.loc[race["position"] == 1, "driver_id"]
            winner = winners.iloc[0] if not winners.empty else None
            podium_finishers = set(race.loc[race["position"] <= 3, "driver_id"])

            for d in race["driver_id"]:
                if d not in cal_win:
                    continue
                y_true_win.append(1 if d == winner else 0)
                y_pred_win.append(cal_win[d])
                y_true_podium.append(1 if d in podium_finishers else 0)
                y_pred_podium.append(cal_podium[d])

        if y_true_win:
            rows.append({
                "val_season": fold["val_season"],
                "n_races": val_df[["season", "round"]].drop_duplicates().shape[0],
                "log_loss": log_loss(y_true_win, y_pred_win, labels=[0, 1]),
                "brier": brier_score_loss(y_true_win, y_pred_win),
                "log_loss_podium": log_loss(y_true_podium, y_pred_podium, labels=[0, 1]),
                "brier_podium": brier_score_loss(y_true_podium, y_pred_podium),
                "cal_a_win": a_win,
                "cal_b_win": b_win,
                "cal_a_podium": a_podium,
                "cal_b_podium": b_podium,
            })
        else:
            rows.append({
                "val_season": fold["val_season"],
                "n_races": val_df[["season", "round"]].drop_duplicates().shape[0],
                "log_loss": np.nan,
                "brier": np.nan,
                "log_loss_podium": np.nan,
                "brier_podium": np.nan,
                "cal_a_win": a_win,
                "cal_b_win": b_win,
                "cal_a_podium": a_podium,
                "cal_b_podium": b_podium,
            })

    return pd.DataFrame(rows)


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
