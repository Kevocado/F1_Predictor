"""live_replay.py — Phase 3 checkpoint: replay a real, held-out historical
race lap-by-lap through models/live_win_prob.py, printing the actual
winner's P(win) at each lap to confirm it converges toward 1.0 as the race
progresses — the same "directionally sane" bar evaluate/backtest.py holds
the pre-race models to.

Also doubles as the live engine's demo/test harness for a day with no real
live session — `--sleep` paces it out like an accelerated real-time replay.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from ..data import fastf1_client, jolpica
from ..features import elo, team_strength
from ..models import live_win_prob


def replay_race(season: int, round_: int, sleep_per_lap: float = 0.0) -> pd.DataFrame:
    """One row per lap: the actual eventual winner's P(win)/P(podium) at
    that lap, and the model's own top pick at that lap — the convergence
    check."""
    seasons = live_win_prob.default_live_seasons()
    results_df = jolpica.load_multi_season_results(seasons)
    elo_hist = elo.compute_elo_history(results_df)
    team_hist = team_strength.compute_team_strength_history(results_df)

    race_results = jolpica.fetch_race_results(season, round_)
    if race_results.empty:
        raise RuntimeError(f"No results for {season} round {round_}")
    code_map = dict(zip(race_results["driver_code"], race_results["driver_id"]))
    driver_to_constructor = dict(zip(race_results["driver_id"], race_results["constructor_id"]))

    elo_row = elo_hist[(elo_hist["season"] == season) & (elo_hist["round"] == round_)]
    elo_dict = dict(zip(elo_row["driver_id"], elo_row["elo_pre_race"]))
    team_row = team_hist[(team_hist["season"] == season) & (team_hist["round"] == round_)]
    team_dict = dict(zip(team_row["constructor_id"], team_row["team_strength_pre_race"]))

    snaps = fastf1_client.build_lap_snapshots(season, round_, code_map, elo_dict, team_dict, driver_to_constructor)
    if snaps.empty:
        raise RuntimeError(f"No lap data for {season} round {round_}")

    actual_winner = snaps.loc[snaps["won"], "driver_id"].iloc[0]
    win_model, podium_model = live_win_prob.load_models()

    rows = []
    for lap_number, lap_df in snaps.groupby("lap_number"):
        preds = live_win_prob.predict_live(win_model, podium_model, lap_df)
        winner_row = preds[preds["driver_id"] == actual_winner]
        top_pick = preds.iloc[0]
        rows.append(
            {
                "lap_number": lap_number,
                "actual_winner": actual_winner,
                "actual_winner_p_win": float(winner_row["p_win"].iloc[0]) if not winner_row.empty else None,
                "model_top_pick": top_pick["driver_id"],
                "model_top_pick_p_win": float(top_pick["p_win"]),
            }
        )
        if sleep_per_lap:
            time.sleep(sleep_per_lap)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a historical race lap-by-lap through the live engine.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between laps.")
    args = parser.parse_args()

    df = replay_race(args.season, args.round, sleep_per_lap=args.sleep)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    print(f"\nActual winner: {df['actual_winner'].iloc[0]}")
    print(
        f"P(win) at lap {int(df['lap_number'].iloc[0])}: {df['actual_winner_p_win'].iloc[0]:.3f} "
        f"-> at lap {int(df['lap_number'].iloc[-1])}: {df['actual_winner_p_win'].iloc[-1]:.3f}"
    )


if __name__ == "__main__":
    main()
