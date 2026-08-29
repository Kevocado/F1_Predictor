"""tune_live_hyperparams.py — Optuna hyperparameter search for the live
in-race win/podium models (models/live_win_prob.py). Builds the lap-state
training corpus ONCE, splits it by race ONCE (same seed across every
trial, for a fair apples-to-apples comparison), then searches
max_depth/learning_rate/subsample/colsample_bytree/min_child_weight/
reg_lambda — `n_estimators` is left high (400) with early stopping still
active per trial, so the search doesn't need to separately tune tree
count.

Resumable, same as evaluate/tune_hyperparams.py: SQLite-backed study,
`load_if_exists=True`.
"""

from __future__ import annotations

import argparse

import optuna
import xgboost as xgb

from ..config import MODELS_DIR
from ..models import live_win_prob

STUDY_DB_PATH = MODELS_DIR / "optuna_studies.db"
WIN_STUDY_NAME = "live_win_model"
PODIUM_STUDY_NAME = "live_podium_model"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _search_space(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _load_study(name: str) -> optuna.Study:
    return optuna.create_study(
        study_name=name, storage=f"sqlite:///{STUDY_DB_PATH}", direction="minimize", load_if_exists=True
    )


def tune(target: str, n_trials: int = 30, seasons: list[int] | None = None, seed: int = 0) -> optuna.Study:
    """`target`: "won" or "podium" — the label column being tuned for."""
    corpus = live_win_prob.build_training_corpus(seasons, verbose=False)
    if corpus.empty:
        raise RuntimeError("Live-engine training corpus is empty — build it first (models/live_win_prob.py).")

    train_df, val_df = live_win_prob.race_level_split(corpus, seed=seed)
    X_train, X_val = live_win_prob.prep_features(train_df), live_win_prob.prep_features(val_df)
    y_train, y_val = train_df[target].astype(int), val_df[target].astype(int)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "n_estimators": 400,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "early_stopping_rounds": 30,
            **_search_space(trial),
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return float(model.evals_result()["validation_0"]["logloss"][model.best_iteration])

    study_name = WIN_STUDY_NAME if target == "won" else PODIUM_STUDY_NAME
    study = _load_study(study_name)
    study.optimize(objective, n_trials=n_trials)
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search for the live in-race win/podium models.")
    parser.add_argument("--target", choices=["win", "podium"], required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    target_col = "won" if args.target == "win" else "podium"
    study = tune(target_col, n_trials=args.n_trials)

    print(f"\nBest value (held-out log-loss): {study.best_value:.5f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\n{len(study.trials)} trials total in this study so far — re-run the same command to add more.")


if __name__ == "__main__":
    main()
