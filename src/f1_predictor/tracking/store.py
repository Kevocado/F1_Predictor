"""store.py — SQLite persistence for race and championship prediction
track records. Mirrors PL_Predictor's tracking/store.py: snapshot each
prediction BEFORE the outcome is known, reconcile against actual results
once they land — the only honest way to measure "how good are the
predictions really," and this project's substitute for the market
comparison PL_Predictor's value-bet feature relies on (there is no F1 odds
market — see RESEARCH_BRIEF.md). Comparing predictions across tiers
(pre_weekend -> post_practice -> post_qualifying) tells the "is this
actually working" story instead.

`race_predictions` is keyed on (season, round, driver_id, tier, market) —
the one real schema difference from PL_Predictor: a round legitimately
gets up to 3 snapshots (one per tier) before it's reconciled, not just one.
`championship_snapshots` records one row per driver/constructor each time
models/championship_projection.py::simulate_season reruns, for a "how has
our title-race view evolved" trend. No live in-race snapshot table —
lap-cadence data isn't worth persisting; the live engine (Phase 3) is
validated offline against historical replay instead.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ..config import TRACKING_DB_PATH

MARKET_SPEC = [
    ("win", "p_win"),
    ("podium", "p_podium"),
    ("points_finish", "p_points_finish"),
    ("dnf", "p_dnf"),
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TRACKING_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS race_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            round INTEGER NOT NULL,
            race_name TEXT NOT NULL,
            driver_id TEXT NOT NULL,
            constructor_id TEXT,
            tier TEXT NOT NULL,
            market TEXT NOT NULL,
            predicted_prob REAL NOT NULL,
            session_time TEXT NOT NULL,
            snapshotted_at TEXT NOT NULL,
            model_trained_at TEXT,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_outcome INTEGER,
            actual_finish_position INTEGER,
            actual_dnf INTEGER,
            resolved_at TEXT,
            backfilled INTEGER NOT NULL DEFAULT 0,
            UNIQUE(season, round, driver_id, tier, market)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS championship_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            as_of_round INTEGER NOT NULL,
            championship TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            win_prob REAL NOT NULL,
            top3_prob REAL,
            expected_final_points REAL NOT NULL,
            expected_final_rank REAL,
            snapshotted_at TEXT NOT NULL,
            UNIQUE(season, as_of_round, championship, entity_id)
        )
        """
    )
    return conn


def record_race_predictions(
    sim_table: pd.DataFrame,
    season: int,
    round_: int,
    race_name: str,
    tier: str,
    session_time: str,
    model_trained_at: str | None = None,
    backfilled: bool = False,
) -> int:
    """Snapshot one race's Monte Carlo output (models/race_outcome.py::
    simulate_race's return, with a constructor_id column already joined on
    by the caller) for a given tier. Already-logged (season, round,
    driver_id, tier, market) rows are left untouched — safe to call every
    time a prediction is served."""
    if sim_table.empty:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, r in sim_table.iterrows():
        for market, prob_col in MARKET_SPEC:
            rows.append(
                (
                    season,
                    round_,
                    race_name,
                    r["driver_id"],
                    r.get("constructor_id"),
                    tier,
                    market,
                    float(r[prob_col]),
                    session_time,
                    now,
                    model_trained_at,
                    int(backfilled),
                )
            )
    with _connect() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO race_predictions
                (season, round, race_name, driver_id, constructor_id, tier, market, predicted_prob,
                 session_time, snapshotted_at, model_trained_at, backfilled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount


def _actual_outcome(market: str, driver_row: pd.Series) -> int:
    position = driver_row.get("position")
    has_position = position is not None and not pd.isna(position)
    if market == "win":
        return int(has_position and int(position) == 1)
    if market == "podium":
        return int(has_position and int(position) <= 3)
    if market == "points_finish":
        return int(has_position and int(position) <= 10)
    if market == "dnf":
        return int(bool(driver_row.get("dnf")))
    raise ValueError(f"unknown market: {market}")


def reconcile_predictions(results_df: pd.DataFrame) -> int:
    """Fills in actual outcomes for every still-unresolved race_predictions
    row whose (season, round, driver_id) now has a real result in
    `results_df`. Safe to call repeatedly — only touches resolved=0 rows."""
    if results_df.empty:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    results_lookup = results_df.set_index(["season", "round", "driver_id"])

    with _connect() as conn:
        unresolved = pd.read_sql(
            "SELECT id, season, round, driver_id, market FROM race_predictions WHERE resolved = 0", conn
        )
        if unresolved.empty:
            return 0

        updates = []
        for _, row in unresolved.iterrows():
            key = (row["season"], row["round"], row["driver_id"])
            if key not in results_lookup.index:
                continue
            driver_row = results_lookup.loc[key]
            if isinstance(driver_row, pd.DataFrame):  # duplicate key guard, shouldn't happen
                driver_row = driver_row.iloc[0]
            actual = _actual_outcome(row["market"], driver_row)
            position = driver_row.get("position")
            updates.append(
                (
                    actual,
                    None if position is None or pd.isna(position) else int(position),
                    int(bool(driver_row.get("dnf"))),
                    now,
                    int(row["id"]),
                )
            )

        if updates:
            conn.executemany(
                """
                UPDATE race_predictions
                SET resolved = 1, actual_outcome = ?, actual_finish_position = ?, actual_dnf = ?, resolved_at = ?
                WHERE id = ?
                """,
                updates,
            )
        return len(updates)


def record_championship_snapshot(projection: dict) -> int:
    """One row per driver/constructor from
    models/championship_projection.py::simulate_season's output."""
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for championship in ("drivers", "constructors"):
        for _, r in projection[championship].iterrows():
            rows.append(
                (
                    projection["season"],
                    projection["as_of_round"],
                    championship,
                    r["entity_id"],
                    float(r["win_prob"]),
                    float(r["top3_prob"]),
                    float(r["expected_final_points"]),
                    float(r["expected_final_rank"]),
                    now,
                )
            )
    with _connect() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO championship_snapshots
                (season, as_of_round, championship, entity_id, win_prob, top3_prob,
                 expected_final_points, expected_final_rank, snapshotted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount


def get_track_record(tier: str | None = None) -> dict:
    """Overall hit-rate/calibration summary, optionally filtered to one
    tier — the "predictions get sharper tier by tier" story that
    substitutes for the missing odds-market comparison."""
    with _connect() as conn:
        query = "SELECT tier, market, predicted_prob, actual_outcome FROM race_predictions WHERE resolved = 1"
        params: tuple = ()
        if tier:
            query += " AND tier = ?"
            params = (tier,)
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return {"n_resolved": 0, "by_market": []}

    rows = []
    for (t, market), g in df.groupby(["tier", "market"]):
        brier = float(((g["predicted_prob"] - g["actual_outcome"]) ** 2).mean())
        rows.append(
            {
                "tier": t,
                "market": market,
                "n": int(len(g)),
                "brier": brier,
                "hit_rate": float(g["actual_outcome"].mean()),
                "avg_predicted_prob": float(g["predicted_prob"].mean()),
            }
        )
    return {"n_resolved": int(len(df)), "by_market": rows}


def get_race_prediction(season: int, round_: int, tier: str | None = None) -> list[dict]:
    with _connect() as conn:
        query = "SELECT * FROM race_predictions WHERE season = ? AND round = ?"
        params: list = [season, round_]
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        df = pd.read_sql(query, conn, params=params)
    return df.to_dict(orient="records")


def get_championship_history(season: int, championship: str) -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql(
            "SELECT * FROM championship_snapshots WHERE season = ? AND championship = ? ORDER BY as_of_round",
            conn,
            params=(season, championship),
        )
