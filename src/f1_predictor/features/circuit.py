"""circuit.py — per-driver and per-constructor historical performance at a
given circuit: how has this driver/car tended to do here in the past,
independent of current form. Some circuits suit certain cars/drivers
disproportionately (street circuits, high-downforce tracks) in a way
season-to-date rolling form alone won't capture.
"""

from __future__ import annotations

import pandas as pd

BACK_OF_FIELD_POSITION = 20


def with_circuit(results_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    circuits = schedule_df[["season", "round", "circuit_id"]].drop_duplicates()
    return results_df.merge(circuits, on=["season", "round"], how="left")


def compute_circuit_history(results_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """No-lookahead: each row sees only that driver's/constructor's *prior*
    visits to this same circuit (expanding mean, not a fixed window — a
    circuit is typically visited once a season, so a rolling N-race window
    doesn't map onto "recent form" the way it does elsewhere)."""
    df = with_circuit(results_df, schedule_df).sort_values(["season", "round"]).copy()
    df["finish_position"] = df["position"].fillna(BACK_OF_FIELD_POSITION)

    out = df[["season", "round", "driver_id", "constructor_id", "circuit_id"]].copy()
    driver_grp = df.groupby(["driver_id", "circuit_id"])
    out["driver_circuit_avg_position"] = driver_grp["finish_position"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    out["driver_circuit_avg_points"] = driver_grp["points"].transform(lambda s: s.shift(1).expanding().mean())

    constructor_grp = df.groupby(["constructor_id", "circuit_id"])
    out["constructor_circuit_avg_position"] = constructor_grp["finish_position"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return out.reset_index(drop=True)
