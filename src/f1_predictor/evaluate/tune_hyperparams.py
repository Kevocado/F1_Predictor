"""tune_hyperparams.py — Optuna hyperparameter search for the xgb_ranker
race-outcome candidate and the DNF reliability model, against the SAME
walk-forward folds evaluate/walk_forward.py already builds — a 5-fold
search, not a single held-out split, to avoid overfitting the
hyperparameters the same way a feature can overfit training data (mirrors
PL_Predictor's own tune_hyperparams.py reasoning).

Resumable: each study is backed by a SQLite file
(models/optuna_studies.db), so a search can be stopped (Ctrl-C, a
timeout) and continued later with the same command — Optuna's
`load_if_exists=True` picks up exactly where it left off rather than
starting over.
"""

from __future__ import annotations

import argparse

import optuna

from ..config import MODELS_DIR
from . import walk_forward

STUDY_DB_PATH = MODELS_DIR / "optuna_studies.db"
RANKER_STUDY_NAME = "race_outcome_ranker"
DNF_STUDY_NAME = "dnf_model"

# Quieter than Optuna's default (one line per trial is enough; the default
# also prints internal warnings that just add noise here).
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _ranker_search_space(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def _dnf_search_space(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _load_study(name: str) -> optuna.Study:
    return optuna.create_study(
        study_name=name, storage=f"sqlite:///{STUDY_DB_PATH}", direction="minimize", load_if_exists=True
    )


def tune_ranker(n_trials: int = 40, seasons: list[int] | None = None, eval_n_trials: int = 1500) -> optuna.Study:
    """Objective: mean held-out log-loss on the win market across all
    walk-forward folds (lower is better) — the same metric
    models/manifest.py's candidate race-off is decided on, just searched
    instead of hand-picked. `eval_n_trials` is the Monte Carlo trial count
    used INSIDE each fold's simulate_race call — kept modest during search
    for speed; the final chosen hyperparameters get evaluated at full
    precision by re-running models/manifest.py::train_all."""
    folds = walk_forward.prepare_folds(seasons)

    def objective(trial: optuna.Trial) -> float:
        hyperparams = _ranker_search_space(trial)
        metrics = walk_forward.evaluate_candidate(folds, "xgb_ranker", n_trials=eval_n_trials, hyperparams=hyperparams)
        return float(metrics["log_loss"].mean())

    study = _load_study(RANKER_STUDY_NAME)
    study.optimize(objective, n_trials=n_trials)
    return study


def tune_dnf(n_trials: int = 40, seasons: list[int] | None = None) -> optuna.Study:
    folds = walk_forward.prepare_folds(seasons)

    def objective(trial: optuna.Trial) -> float:
        hyperparams = _dnf_search_space(trial)
        metrics = walk_forward.evaluate_dnf_model(folds, hyperparams=hyperparams)
        return float(metrics["log_loss"].mean())

    study = _load_study(DNF_STUDY_NAME)
    study.optimize(objective, n_trials=n_trials)
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search, resumable across runs.")
    parser.add_argument("--target", choices=["ranker", "dnf"], required=True)
    parser.add_argument("--n-trials", type=int, default=40)
    args = parser.parse_args()

    if args.target == "ranker":
        study = tune_ranker(n_trials=args.n_trials)
    else:
        study = tune_dnf(n_trials=args.n_trials)

    print(f"\nBest value (mean held-out log-loss): {study.best_value:.5f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\n{len(study.trials)} trials total in this study so far — re-run the same command to add more.")
    print(f"Study storage: sqlite:///{STUDY_DB_PATH} (study_name={'race_outcome_ranker' if args.target == 'ranker' else 'dnf_model'})")


if __name__ == "__main__":
    main()
