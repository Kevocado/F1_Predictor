"""feature_ablation.py — does a feature actually help, or does it just
LOOK important? XGBoost's gain-based `feature_importances_` (and even
TreeSHAP attributions in models/explain.py) measure how much the model
leans on a feature, not whether leaning on it improves real predictive
accuracy — a feature can show high importance while adding noise a
simpler model would have ignored. This answers the sharper question
directly: train the xgb_ranker candidate WITH vs. WITHOUT a feature (or
named group of features), on the SAME walk-forward folds
evaluate/walk_forward.py already builds, and compare held-out log-loss on
the win market. A feature "helps" only if removing it makes held-out
log-loss measurably worse.
"""

from __future__ import annotations

import argparse

import pandas as pd

from ..features.build import FEATURE_COLUMNS
from . import walk_forward

# Named groups so a whole feature FAMILY can be ablated at once (e.g. "did
# adding team_form pull its weight" rather than one column at a time).
FEATURE_GROUPS: dict[str, list[str]] = {
    "team_form": ["team_form_avg_position_3", "team_form_points_3"],
    "elo": ["elo_pre_race"],
    "team_strength": ["team_strength_pre_race"],
    "driver_form": [c for c in FEATURE_COLUMNS if c.startswith("form_")],
    "circuit_history": [c for c in FEATURE_COLUMNS if "circuit" in c],
    "weather": ["temp_max_c", "precipitation_mm", "wind_max_kph"],
    "qualifying": ["grid", "quali_position", "quali_gap_to_pole"],
}


def ablate(feature_names: list[str], seasons: list[int] | None = None, n_trials: int = 2000) -> pd.DataFrame:
    """One row per fold: val_season, log-loss WITH all features, log-loss
    WITHOUT `feature_names`, and the delta (positive = removing them hurt,
    i.e. the feature(s) helped)."""
    folds = walk_forward.prepare_folds(seasons)
    ablated_cols = [c for c in FEATURE_COLUMNS if c not in feature_names]

    full = walk_forward.evaluate_candidate(folds, "xgb_ranker", n_trials=n_trials)
    ablated = walk_forward.evaluate_candidate(folds, "xgb_ranker", n_trials=n_trials, feature_cols_override=ablated_cols)

    merged = full[["val_season", "log_loss"]].merge(
        ablated[["val_season", "log_loss"]], on="val_season", suffixes=("_full", "_ablated")
    )
    merged["delta"] = merged["log_loss_ablated"] - merged["log_loss_full"]
    return merged


def ablate_group(group_name: str, seasons: list[int] | None = None, n_trials: int = 2000) -> pd.DataFrame:
    if group_name not in FEATURE_GROUPS:
        raise ValueError(f"Unknown group '{group_name}'. Choices: {list(FEATURE_GROUPS)}")
    return ablate(FEATURE_GROUPS[group_name], seasons=seasons, n_trials=n_trials)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate a feature (or named group) and compare held-out log-loss.")
    parser.add_argument("--group", choices=list(FEATURE_GROUPS), help="A predefined feature group.")
    parser.add_argument("--features", nargs="+", help="Explicit feature column names to ablate together.")
    parser.add_argument("--n-trials", type=int, default=2000)
    args = parser.parse_args()

    if not args.group and not args.features:
        parser.error("Pass --group <name> or --features <col> [<col> ...]. Groups: " + ", ".join(FEATURE_GROUPS))
    feature_names = FEATURE_GROUPS[args.group] if args.group else args.features

    result = ablate(feature_names, n_trials=args.n_trials)
    pd.set_option("display.width", 160)
    print(f"Ablating: {feature_names}\n")
    print(result.to_string(index=False))

    mean_delta = result["delta"].mean()
    verdict = "HELPS" if mean_delta > 0.001 else ("HURTS" if mean_delta < -0.001 else "NO CLEAR EFFECT")
    print(f"\nMean delta (log_loss_ablated - log_loss_full): {mean_delta:+.5f} -> {verdict}")
    print("(positive delta = removing the feature made held-out log-loss worse, i.e. the feature helps)")


if __name__ == "__main__":
    main()
