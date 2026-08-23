"""rolling_form.py — per-driver rolling recent-form features: mean finish
position, points-per-race, and DNF rate over each driver's own past N
races. Every value is computed via a shift(1) before rolling — a row's
features only ever see that driver's races strictly before it, the same
no-lookahead discipline PL_Predictor's rolling_form.py uses for teams.
"""

from __future__ import annotations

import pandas as pd

ROLLING_WINDOWS = (3, 5, 10)
BACK_OF_FIELD_POSITION = 20  # fill value for DNF/unclassified rows, so form isn't blind to bad results


def compute_rolling_form(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.sort_values(["driver_id", "season", "round"]).copy()
    df["finish_position"] = df["position"].fillna(BACK_OF_FIELD_POSITION)
    grouped = df.groupby("driver_id")

    out = df[["season", "round", "driver_id"]].copy()
    for window in ROLLING_WINDOWS:
        out[f"form_avg_position_{window}"] = grouped["finish_position"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        out[f"form_avg_points_{window}"] = grouped["points"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        out[f"form_dnf_rate_{window}"] = grouped["dnf"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    return out.reset_index(drop=True)
