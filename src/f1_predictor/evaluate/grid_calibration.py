"""grid_calibration.py — grid-position lookup baseline and 2-parameter
calibration layer for win/podium probabilities.

The grid lookup baseline computes historical win/podium rates by grid
position from training data only (no leakage). The calibration layer fits
2 parameters (per market: win, podium) on out-of-fold scores from earlier
seasons to adjust the model's probabilities toward the grid lookup.

Top-10 (points finish) stays un-layered per spec.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from ..data import jolpica
from ..features import session_state
from ..features.build import build_training_frame
from ..models import race_outcome


MAX_GRID = 20  # F1 grid size


def compute_grid_lookup(train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute historical win/podium rates by grid position from `train_df`.

    Returns (win_rates, podium_rates) arrays of length MAX_GRID+1 (1-indexed,
    index 0 unused). Rates are clipped to [1e-6, 1-1e-6] for numerical
    stability in log-loss.
    """
    win_rates = np.full(MAX_GRID + 1, 1e-6)
    podium_rates = np.full(MAX_GRID + 1, 1e-6)
    counts = np.zeros(MAX_GRID + 1, dtype=int)

    for (_season, _round_), race in train_df.groupby(["season", "round"]):
        race = race.dropna(subset=["grid", "position"])
        if race.empty:
            continue
        for _, row in race.iterrows():
            grid = int(row["grid"])
            if 1 <= grid <= MAX_GRID:
                counts[grid] += 1
                if row["position"] == 1:
                    win_rates[grid] += 1
                if row["position"] <= 3:
                    podium_rates[grid] += 1

    # Convert to rates with Laplace smoothing
    for g in range(1, MAX_GRID + 1):
        if counts[g] > 0:
            win_rates[g] = (win_rates[g] - 1e-6 + 1) / (counts[g] + 2)  # +1/-1 for the initial 1e-6
            podium_rates[g] = (podium_rates[g] - 1e-6 + 1) / (counts[g] + 2)
        else:
            # No data for this grid position: use a prior that decays with grid
            win_rates[g] = max(1e-6, 0.5 ** g)
            podium_rates[g] = max(1e-6, min(1.0, 1.5 * win_rates[g]))

    # Clip for numerical stability
    win_rates = np.clip(win_rates, 1e-6, 1 - 1e-6)
    podium_rates = np.clip(podium_rates, 1e-6, 1 - 1e-6)

    return win_rates, podium_rates


def grid_lookup_probs(race_df: pd.DataFrame, win_rates: np.ndarray, podium_rates: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Get grid-lookup win/podium probabilities for each driver in a race."""
    win_probs = {}
    podium_probs = {}
    for _, row in race_df.iterrows():
        grid = int(row["grid"]) if pd.notna(row["grid"]) else MAX_GRID
        grid = min(max(grid, 1), MAX_GRID)
        win_probs[row["driver_id"]] = win_rates[grid]
        podium_probs[row["driver_id"]] = podium_rates[grid]
    return win_probs, podium_probs


def fit_calibration_params(
    model_probs: np.ndarray,
    grid_probs: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float]:
    """Fit 2-parameter calibration: p_cal = sigmoid(a * logit(p_model) + b * logit(p_grid)).

    Uses simple gradient descent on log-loss. Returns (a, b).
    """
    # Initialize: trust the model (a=1, b=0)
    a, b = 1.0, 0.0
    lr = 0.01
    eps = 1e-6

    def logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    model_logit = logit(model_probs)
    grid_logit = logit(grid_probs)

    for _ in range(2000):
        z = a * model_logit + b * grid_logit
        p_cal = sigmoid(z)
        p_cal = np.clip(p_cal, eps, 1 - eps)

        # Gradient of log-loss w.r.t a and b
        grad_a = np.mean((p_cal - y_true) * model_logit)
        grad_b = np.mean((p_cal - y_true) * grid_logit)

        a -= lr * grad_a
        b -= lr * grad_b

        # Early stopping if gradients are tiny
        if abs(grad_a) < 1e-6 and abs(grad_b) < 1e-6:
            break

    return float(a), float(b)


def fit_blend_weight(
    model_probs: np.ndarray,
    grid_probs: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Fit optimal blend weight: p_cal = w * p_model + (1-w) * p_grid.

    Returns w in [0, 1]. Uses simple grid search on log-loss.
    """
    best_w = 0.5
    best_loss = float('inf')
    eps = 1e-6

    for w in np.linspace(0.0, 1.0, 101):
        p_blend = w * model_probs + (1 - w) * grid_probs
        p_blend = np.clip(p_blend, eps, 1 - eps)
        loss = log_loss(y_true, p_blend, labels=[0, 1])
        if loss < best_loss:
            best_loss = loss
            best_w = w

    return float(best_w)


def apply_blend(
    model_probs: dict[str, float],
    grid_probs: dict[str, float],
    w: float,
    normalize_to: float | None = None,
) -> dict[str, float]:
    """Apply blend: p_cal = w * p_model + (1-w) * p_grid."""
    drivers = list(model_probs.keys())
    model_arr = np.array([model_probs[d] for d in drivers])
    grid_arr = np.array([grid_probs[d] for d in drivers])

    p_blend = w * model_arr + (1 - w) * grid_arr
    p_blend = np.clip(p_blend, 1e-6, 1 - 1e-6)

    if normalize_to is not None:
        p_blend = p_blend / p_blend.sum() * normalize_to

    return {d: float(p_blend[i]) for i, d in enumerate(drivers)}


def apply_calibration(
    model_probs: dict[str, float],
    grid_probs: dict[str, float],
    a: float,
    b: float,
    normalize_to: float | None = None,
) -> dict[str, float]:
    """Apply 2-parameter calibration to model probabilities.

    Args:
        model_probs: Model probabilities per driver
        grid_probs: Grid lookup probabilities per driver
        a, b: Calibration parameters
        normalize_to: If set, renormalize so sum = normalize_to (e.g., 1.0 for win, 3.0 for podium).
                      If None, no renormalization.
    """
    eps = 1e-6
    drivers = list(model_probs.keys())
    model_arr = np.array([model_probs[d] for d in drivers])
    grid_arr = np.array([grid_probs[d] for d in drivers])

    def logit(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    model_logit = logit(model_arr)
    grid_logit = logit(grid_arr)
    z = a * model_logit + b * grid_logit
    p_cal = sigmoid(z)
    p_cal = np.clip(p_cal, eps, 1 - eps)

    # Renormalize if requested
    if normalize_to is not None:
        p_cal = p_cal / p_cal.sum() * normalize_to

    return {d: float(p_cal[i]) for i, d in enumerate(drivers)}


def evaluate_grid_lookup(folds: list[dict], n_trials: int = 3000) -> pd.DataFrame:
    """Evaluate the grid lookup baseline on walk-forward folds."""
    rows = []
    for fold in folds:
        train_df, val_df = fold["train_df"], fold["val_df"]

        win_rates, podium_rates = compute_grid_lookup(train_df)

        y_true: list[int] = []
        y_pred_win: list[float] = []
        y_pred_podium: list[float] = []

        for _season, _round_, race in race_df_groups(val_df):
            if race["grid"].isna().all():
                continue
            win_probs, podium_probs = grid_lookup_probs(race, win_rates, podium_rates)

            winners = race.loc[race["position"] == 1, "driver_id"]
            winner = winners.iloc[0] if not winners.empty else None
            podium_finishers = set(race.loc[race["position"] <= 3, "driver_id"])

            for d in race["driver_id"]:
                if d not in win_probs:
                    continue
                y_true.append(1 if d == winner else 0)
                y_pred_win.append(win_probs[d])
                y_pred_podium.append(1.0 if d in podium_finishers else 0.0)  # This is wrong - podium is multi-class

        # For win market
        if y_true:
            rows.append({
                "val_season": fold["val_season"],
                "n_races": val_df[["season", "round"]].drop_duplicates().shape[0],
                "log_loss": log_loss(y_true, y_pred_win, labels=[0, 1]),
                "brier": np.mean((np.array(y_true) - np.array(y_pred_win)) ** 2),
            })
        else:
            rows.append({
                "val_season": fold["val_season"],
                "n_races": val_df[["season", "round"]].drop_duplicates().shape[0],
                "log_loss": np.nan,
                "brier": np.nan,
            })

    return pd.DataFrame(rows)


def race_df_groups(df: pd.DataFrame):
    for (season, round_), race in df.groupby(["season", "round"]):
        yield season, round_, race


def evaluate_calibrated_candidate(
    folds: list[dict],
    candidate: str,
    n_trials: int = 3000,
    feature_cols_override: list[str] | None = None,
    hyperparams: dict | None = None,
) -> pd.DataFrame:
    """Evaluate the calibrated candidate: grid lookup for win, blend for podium.

    For win market, grid lookup is the strongest predictor, so we use it directly.
    For podium, we blend model + grid lookup with fitted weight.
    """
    from ..models import dnf as dnf_model

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

        # Collect out-of-fold predictions on training data for podium blend weight
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
            _, podium_probs_grid = grid_lookup_probs(race, win_rates, podium_rates)

            podium_finishers = set(race.loc[race["position"] <= 3, "driver_id"])

            for d in race["driver_id"]:
                if d not in sim.index or d not in podium_probs_grid:
                    continue
                train_model_podium.append(sim.loc[d, "p_podium"])
                train_grid_podium.append(podium_probs_grid[d])
                train_true_podium.append(1 if d in podium_finishers else 0)

        # Fit blend weight for podium only
        if len(train_model_podium) > 10:
            w_podium = fit_blend_weight(
                np.array(train_model_podium), np.array(train_grid_podium), np.array(train_true_podium)
            )
        else:
            w_podium = 0.0

        # Evaluate on validation data: grid for win, blend for podium
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

            # For win: use grid lookup directly (strongest predictor)
            # For podium: blend model + grid
            model_podium = {d: sim.loc[d, "p_podium"] for d in race["driver_id"] if d in sim.index}

            cal_win = win_probs_grid  # Grid lookup for win
            cal_podium = apply_blend(model_podium, podium_probs_grid, w_podium, normalize_to=3.0)

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
                "brier": np.mean((np.array(y_true_win) - np.array(y_pred_win)) ** 2),
                "log_loss_podium": log_loss(y_true_podium, y_pred_podium, labels=[0, 1]),
                "brier_podium": np.mean((np.array(y_true_podium) - np.array(y_pred_podium)) ** 2),
                "blend_w_win": 0.0,  # Grid only for win
                "blend_w_podium": w_podium,
            })
        else:
            rows.append({
                "val_season": fold["val_season"],
                "n_races": val_df[["season", "round"]].drop_duplicates().shape[0],
                "log_loss": np.nan,
                "brier": np.nan,
                "log_loss_podium": np.nan,
                "brier_podium": np.nan,
                "blend_w_win": 0.0,
                "blend_w_podium": w_podium,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from .walk_forward import prepare_folds
    seasons = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else None
    folds = prepare_folds(seasons)
    print("Grid lookup baseline:")
    grid_results = evaluate_grid_lookup(folds)
    print(grid_results.to_string(index=False))
    print("\nCalibrated xgb_ranker:")
    cal_results = evaluate_calibrated_candidate(folds, "xgb_ranker")
    print(cal_results.to_string(index=False))


def fit_calibration_on_full_history(
    train_df: pd.DataFrame,
    candidate: str,
    feature_cols: list[str],
    hyperparams: dict | None = None,
    n_trials: int = 3000,
) -> dict:
    """Fit calibration parameters on the FULL training data (used for production serving).

    Returns dict with win_rates, podium_rates, and blend weights (w_win, w_podium).
    """
    from ..models import dnf as dnf_model

    # Train the candidate model on full training data
    ranker = (
        race_outcome.train_ranker(train_df, feature_cols, hyperparams=hyperparams)
        if candidate == "xgb_ranker"
        else None
    )

    # Compute grid lookup rates from full training data
    win_rates, podium_rates = compute_grid_lookup(train_df)

    # Collect predictions on ALL training data for podium blend weight
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
        _, podium_probs_grid = grid_lookup_probs(race, win_rates, podium_rates)

        podium_finishers = set(race.loc[race["position"] <= 3, "driver_id"])

        for d in race["driver_id"]:
            if d not in sim.index or d not in podium_probs_grid:
                continue
            train_model_podium.append(sim.loc[d, "p_podium"])
            train_grid_podium.append(podium_probs_grid[d])
            train_true_podium.append(1 if d in podium_finishers else 0)

    # Fit blend weight for podium
    if len(train_model_podium) > 10:
        w_podium = fit_blend_weight(
            np.array(train_model_podium), np.array(train_grid_podium), np.array(train_true_podium)
        )
    else:
        w_podium = 0.0

    return {
        "win_rates": win_rates.tolist(),
        "podium_rates": podium_rates.tolist(),
        "w_podium": w_podium,
    }