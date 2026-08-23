"""safety_car.py — per-circuit incident-rate proxy, derived from jolpica's
`status` field (no ready-made safety-car dataset exists — see
RESEARCH_BRIEF.md). Phase 1 uses historical DNF rate per circuit as the
proxy; OpenF1's race-control messages (2023+) could sharpen this with an
actual safety-car/VSC/red-flag rate once data/openf1.py exists (Phase 3) —
noted here rather than built now, since OpenF1's 2023+ coverage would only
apply to a handful of recent seasons of the historical training set.
"""

from __future__ import annotations

import pandas as pd

from .circuit import with_circuit


def compute_circuit_dnf_rate(results_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """No-lookahead: each race's circuit_dnf_rate reflects only that
    circuit's *prior* races (expanding mean of the per-race DNF rate),
    aggregated once per (season, round, circuit_id) — every driver at a
    race shares the same circuit-level incident rate."""
    df = with_circuit(results_df, schedule_df)
    per_race = df.groupby(["season", "round", "circuit_id"])["dnf"].mean().reset_index()
    per_race = per_race.rename(columns={"dnf": "race_dnf_rate"}).sort_values(["circuit_id", "season", "round"])
    per_race["circuit_dnf_rate"] = per_race.groupby("circuit_id")["race_dnf_rate"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    return per_race[["season", "round", "circuit_id", "circuit_dnf_rate"]]
