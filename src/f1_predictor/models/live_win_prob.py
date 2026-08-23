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


def build_training_corpus(seasons: list[int] | None = None, verbose: bool = True) -> pd.DataFrame:
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

            try:
                snaps = fastf1_client.build_lap_snapshots(
                    season, round_, code_map, elo_dict, team_dict, driver_to_constructor
                )
            except Exception as exc:  # noqa: BLE001 - a single missing/corrupt FastF1 session shouldn't kill the whole corpus build
                if verbose:
                    print(f"  skipped {season} round {round_}: {exc}")
                continue

            if not snaps.empty:
                frames.append(snaps)
            if verbose:
                print(f"  {season} round {round_}: {len(snaps)} lap-state rows")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_live_models(corpus: pd.DataFrame) -> tuple[xgb.XGBClassifier, xgb.XGBClassifier]:
    X = corpus[LIVE_FEATURE_COLUMNS].copy()
    X["pitted_this_lap"] = X["pitted_this_lap"].astype(int)
    X["safety_car_active"] = X["safety_car_active"].astype(int)

    win_model = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=200, max_depth=5, learning_rate=0.05, tree_method="hist"
    )
    win_model.fit(X, corpus["won"].astype(int))

    podium_model = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=200, max_depth=5, learning_rate=0.05, tree_method="hist"
    )
    podium_model.fit(X, corpus["podium"].astype(int))
    return win_model, podium_model


def predict_live(
    win_model: xgb.XGBClassifier, podium_model: xgb.XGBClassifier, state_df: pd.DataFrame
) -> pd.DataFrame:
    X = state_df[LIVE_FEATURE_COLUMNS].copy()
    X["pitted_this_lap"] = X["pitted_this_lap"].astype(int)
    X["safety_car_active"] = X["safety_car_active"].astype(int)
    return pd.DataFrame(
        {
            "driver_id": state_df["driver_id"].values,
            "p_win": win_model.predict_proba(X)[:, 1],
            "p_podium": podium_model.predict_proba(X)[:, 1],
        }
    ).sort_values("p_win", ascending=False).reset_index(drop=True)


def train_all(seasons: list[int] | None = None) -> dict:
    corpus = build_training_corpus(seasons)
    if corpus.empty:
        raise RuntimeError("Live-engine training corpus is empty — no FastF1 sessions loaded successfully.")

    win_model, podium_model = train_live_models(corpus)
    win_model.save_model(str(WIN_MODEL_PATH))
    podium_model.save_model(str(PODIUM_MODEL_PATH))

    return {
        "n_rows": len(corpus),
        "n_races": int(corpus[["season", "round"]].drop_duplicates().shape[0]),
        "seasons": seasons or default_live_seasons(),
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
