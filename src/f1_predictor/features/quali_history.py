"""quali_history.py — per-driver rolling qualifying features: last-3
qualifying positions and teammate qualifying gap. Every value is computed
via a shift(1) before rolling — a row's features only ever see that
driver's qualifying sessions strictly before it, ensuring no lookahead.
"""

from __future__ import annotations

import pandas as pd


def _parse_time_to_seconds(t) -> float | None:
    if t is None or (isinstance(t, float) and pd.isna(t)) or str(t).strip() == "":
        return None
    parts = str(t).split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def compute_quali_features(quali_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling qualifying features for each driver.

    Adds:
    - quali_avg_position_3: average of last 3 qualifying positions
    - quali_avg_gap_to_pole_3: average of last 3 qualifying gaps to pole (seconds)
    - teammate_quali_gap: gap to teammate in qualifying (seconds, positive = slower than teammate)
    """
    if quali_df.empty:
        return pd.DataFrame(columns=[
            "season", "round", "driver_id",
            "quali_avg_position_3", "quali_avg_gap_to_pole_3", "teammate_quali_gap"
        ])

    df = quali_df.sort_values(["driver_id", "season", "round"]).copy()

    # Parse lap times to seconds
    for col in ["q1", "q2", "q3"]:
        if col in df.columns:
            df[f"{col}_s"] = df[col].map(_parse_time_to_seconds)

    # Best qualifying time
    time_cols = [c for c in ["q1_s", "q2_s", "q3_s"] if c in df.columns]
    if time_cols:
        df["best_quali_time_s"] = df[time_cols].min(axis=1)
    else:
        df["best_quali_time_s"] = float("nan")

    # Rolling average of last 3 qualifying positions (shift(1) for no lookahead)
    out = df[["season", "round", "driver_id", "constructor_id"]].copy()

    # Rolling quali position average
    out["quali_avg_position_3"] = (
        df.groupby("driver_id")["quali_position"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    # Rolling quali gap to pole average
    if "best_quali_time_s" in df.columns:
        # Compute pole time per session
        pole_time = df.groupby(["season", "round"])["best_quali_time_s"].transform("min")
        df["quali_gap_to_pole_s"] = df["best_quali_time_s"] - pole_time

        out["quali_avg_gap_to_pole_3"] = (
            df.groupby("driver_id")["quali_gap_to_pole_s"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )
    else:
        out["quali_avg_gap_to_pole_3"] = float("nan")

    # Teammate quali gap: for each session, compute each driver's gap to their teammate
    # Positive = slower than teammate
    teammate_gaps = []
    for (season, round_), session in df.groupby(["season", "round"]):
        # Only consider drivers with valid quali times and constructor_id
        session = session.dropna(subset=["best_quali_time_s", "constructor_id", "driver_id"])
        if len(session) < 2:
            continue
        # Group by constructor (teammates)
        for constructor_id, team in session.groupby("constructor_id"):
            if len(team) != 2:
                # Not exactly 2 drivers (e.g., 3+ due to mid-season swaps, or 1)
                continue
            team = team.sort_values("best_quali_time_s")
            faster_driver = team.iloc[0]["driver_id"]
            faster_time = team.iloc[0]["best_quali_time_s"]
            slower_driver = team.iloc[1]["driver_id"]
            slower_time = team.iloc[1]["best_quali_time_s"]
            gap = slower_time - faster_time
            teammate_gaps.append({
                "season": season, "round": round_, "driver_id": faster_driver,
                "teammate_quali_gap": -gap  # negative = faster than teammate
            })
            teammate_gaps.append({
                "season": season, "round": round_, "driver_id": slower_driver,
                "teammate_quali_gap": gap  # positive = slower than teammate
            })

    if teammate_gaps:
        teammate_df = pd.DataFrame(teammate_gaps)
        out = out.merge(teammate_df, on=["season", "round", "driver_id"], how="left")
    else:
        out["teammate_quali_gap"] = float("nan")

    return out.reset_index(drop=True)


if __name__ == "__main__":
    from ..data import jolpica
    quali_df = pd.concat([jolpica.load_season_qualifying(s) for s in [2023, 2024]], ignore_index=True)
    result = compute_quali_features(quali_df)
    print(result.head(20))
    print(result.columns.tolist())