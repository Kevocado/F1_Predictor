"""live_win_prob.py — requirement 3's live in-race engine: given the exact
state of a race in progress (lap, position, gaps, tyres, pit stops, safety
car), predict P(win)/P(podium) from that point onward. Trained on FastF1
historical lap-by-lap replay (data/fastf1_client.py), served live during an
actual session from OpenF1 (data/openf1.py) — see docs/live_engine_design.md
for the full design this implements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import MODELS_DIR
from ..data import fastf1_client, jolpica
from ..features import elo, team_strength

WIN_MODEL_PATH = MODELS_DIR / "live_win_model.json"
PODIUM_MODEL_PATH = MODELS_DIR / "live_podium_model.json"
OPTUNA_DB_PATH = MODELS_DIR / "optuna_studies.db"


def _load_tuned_params(study_name: str) -> dict | None:
    """Best hyperparameters from an evaluate/tune_live_hyperparams.py
    Optuna study, if one has been run — None (production defaults)
    otherwise. Never a hard failure: tuning is an optional enhancement."""
    if not OPTUNA_DB_PATH.exists():
        return None
    try:
        import optuna

        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{OPTUNA_DB_PATH}")
        return study.best_params if study.trials else None
    except Exception:  # noqa: BLE001
        return None

LIVE_FEATURE_COLUMNS = [
    "lap_number",
    "laps_remaining",
    "current_position",
    "gap_to_leader_s",
    "gap_ahead_s",
    "tyre_age_laps",
    "pace_delta_last_3_laps",
    "pit_stops_so_far",
    "pitted_this_lap",
    "safety_car_active",
    "grid_position",
    "driver_pre_race_strength",
    "constructor_strength",
    "inferred_soc",
    "deploy_time_s",
    "harvest_time_s",
    "clipping_time_s",
]

# 2022+ only: ground-effect-era cars (2022 regs onward) have materially
# different overtaking/tyre-degradation dynamics than 2018-2021 cars, a
# worse match for what the model needs to predict now — see
# docs/live_engine_design.md section 1 for the full reasoning. Narrower
# than jolpica.default_seasons()'s 8-season window on purpose: each FastF1
# session load is slow enough that the full corpus build is a real
# multi-minute job, and era-consistency matters more here than raw sample
# size.
DEFAULT_LIVE_SEASONS_START = 2022


def default_live_seasons() -> list[int]:
    from ..config import CURRENT_SEASON

    return list(range(DEFAULT_LIVE_SEASONS_START, CURRENT_SEASON + 1))


def build_training_corpus(seasons: list[int] | None = None, verbose: bool = True, include_telemetry: bool = False) -> pd.DataFrame:
    """Assembles the full lap-state training corpus across `seasons`:
    computes Elo/team-strength history ONCE (no-lookahead, same discipline
    as features/build.py), then calls
    data/fastf1_client.py::build_lap_snapshots per completed race, slicing
    that race's pre-race strengths out of the precomputed history rather
    than recomputing them per race."""
    seasons = seasons or default_live_seasons()

    results_df = jolpica.load_multi_season_results(seasons)
    elo_hist = elo.compute_elo_history(results_df)
    team_hist = team_strength.compute_team_strength_history(results_df)

    frames = []
    for season in seasons:
        schedule = jolpica.fetch_season_schedule(season)
        now = pd.Timestamp.now(tz="UTC")
        completed = schedule[schedule["race_datetime"].notna() & (schedule["race_datetime"] <= now)]

        for _, race in completed.iterrows():
            round_ = int(race["round"])
            race_results = jolpica.fetch_race_results(season, round_)
            if race_results.empty:
                continue

            code_map = dict(zip(race_results["driver_code"], race_results["driver_id"]))
            driver_to_constructor = dict(zip(race_results["driver_id"], race_results["constructor_id"]))

            elo_row = elo_hist[(elo_hist["season"] == season) & (elo_hist["round"] == round_)]
            elo_dict = dict(zip(elo_row["driver_id"], elo_row["elo_pre_race"]))

            team_row = team_hist[(team_hist["season"] == season) & (team_hist["round"] == round_)]
            team_dict = dict(zip(team_row["constructor_id"], team_row["team_strength_pre_race"]))

            if verbose:
                print(f"Building lap snapshots for {season} R{round_}...")
            
            try:
                snapshots = fastf1_client.build_lap_snapshots(
                    season,
                    round_,
                    code_map,
                    elo_dict,
                    team_dict,
                    driver_to_constructor,
                    include_telemetry=include_telemetry,
                )
                if not snapshots.empty:
                    frames.append(snapshots)
            except Exception as e:
                print(f"Skipping {season} R{round_} due to error: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[LIVE_FEATURE_COLUMNS].copy()
    X["pitted_this_lap"] = X["pitted_this_lap"].astype(int)
    X["safety_car_active"] = X["safety_car_active"].astype(int)
    return X


def race_level_split(corpus: pd.DataFrame, val_frac: float = 0.15, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by WHOLE RACE, not by row — every lap of a given race is
    highly correlated (same driver, same eventual outcome repeated ~60
    times), so a row-level split would leak a race's own outcome across
    train/val and hide overfitting rather than catch it."""
    races = corpus[["season", "round"]].drop_duplicates().sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(races) * val_frac))
    val_keys = races.iloc[:n_val].copy()
    val_keys["is_val"] = True
    merged = corpus.merge(val_keys, on=["season", "round"], how="left")
    is_val = merged["is_val"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return corpus[~is_val], corpus[is_val]


# Confirmed necessary, not precautionary: an unregularized model (200
# trees, depth 5, no held-out early stopping) predicted 99%+ win
# probability for a lap-1 race leader — the corpus's own true base rate
# for that exact state (lap 1, gap_to_leader_s == 0) is 58.6%. Race-level
# early stopping plus subsample/colsample/min_child_weight regularization
# is the fix, not a data problem — gap_to_leader_s is a very separable
# feature that a deep, unregularized tree ensemble will happily memorize
# past the point of genuine calibration.
LIVE_MODEL_PARAMS = dict(
    objective="binary:logistic",
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_lambda=2.0,
    tree_method="hist",
    eval_metric="logloss",
    early_stopping_rounds=30,
)


def train_live_models(
    corpus: pd.DataFrame,
    seed: int = 0,
    win_hyperparams: dict | None = None,
    podium_hyperparams: dict | None = None,
) -> tuple[xgb.XGBClassifier, xgb.XGBClassifier]:
    """`win_hyperparams`/`podium_hyperparams` each override LIVE_MODEL_PARAMS
    independently (e.g. from evaluate/tune_live_hyperparams.py's Optuna
    search, which tunes win and podium as separate studies since they're
    different targets) — omit either for production defaults."""
    train_df, val_df = race_level_split(corpus, seed=seed)
    X_train, X_val = prep_features(train_df), prep_features(val_df)

    win_params = {**LIVE_MODEL_PARAMS, **(win_hyperparams or {})}
    win_model = xgb.XGBClassifier(**win_params)
    win_model.fit(X_train, train_df["won"].astype(int), eval_set=[(X_val, val_df["won"].astype(int))], verbose=False)

    podium_params = {**LIVE_MODEL_PARAMS, **(podium_hyperparams or {})}
    podium_model = xgb.XGBClassifier(**podium_params)
    podium_model.fit(
        X_train, train_df["podium"].astype(int), eval_set=[(X_val, val_df["podium"].astype(int))], verbose=False
    )
    return win_model, podium_model


def predict_live(
    win_model: xgb.XGBClassifier, podium_model: xgb.XGBClassifier, state_df: pd.DataFrame
) -> pd.DataFrame:
    X = prep_features(state_df)
    return pd.DataFrame(
        {
            "driver_id": state_df["driver_id"].values,
            "p_win": win_model.predict_proba(X)[:, 1],
            "p_podium": podium_model.predict_proba(X)[:, 1],
        }
    ).sort_values("p_win", ascending=False).reset_index(drop=True)


def train_all(seasons: list[int] | None = None) -> dict:
    corpus = build_training_corpus(seasons, include_telemetry=True)
    if corpus.empty:
        raise RuntimeError("Live-engine training corpus is empty — no FastF1 sessions loaded successfully.")

    win_params = _load_tuned_params("live_win_model")
    podium_params = _load_tuned_params("live_podium_model")
    if win_params:
        print(f"Using Optuna-tuned live_win_model hyperparameters: {win_params}")
    if podium_params:
        print(f"Using Optuna-tuned live_podium_model hyperparameters: {podium_params}")

    win_model, podium_model = train_live_models(corpus, win_hyperparams=win_params, podium_hyperparams=podium_params)
    win_model.save_model(str(WIN_MODEL_PATH))
    podium_model.save_model(str(PODIUM_MODEL_PATH))

    win_best = win_model.best_iteration
    podium_best = podium_model.best_iteration
    win_val_logloss = win_model.evals_result()["validation_0"]["logloss"][win_best]
    podium_val_logloss = podium_model.evals_result()["validation_0"]["logloss"][podium_best]

    return {
        "n_rows": len(corpus),
        "n_races": int(corpus[["season", "round"]].drop_duplicates().shape[0]),
        "seasons": seasons or default_live_seasons(),
        "win_model_best_iteration": int(win_best),
        "win_model_val_logloss": float(win_val_logloss),
        "podium_model_best_iteration": int(podium_best),
        "podium_model_val_logloss": float(podium_val_logloss),
        "tuned_win_params": win_params,
        "tuned_podium_params": podium_params,
    }


def load_models() -> tuple[xgb.XGBClassifier, xgb.XGBClassifier]:
    win_model = xgb.XGBClassifier()
    win_model.load_model(str(WIN_MODEL_PATH))
    podium_model = xgb.XGBClassifier()
    podium_model.load_model(str(PODIUM_MODEL_PATH))
    return win_model, podium_model


if __name__ == "__main__":
    import json

    summary = train_all()
    print(json.dumps(summary, indent=2, default=str))
