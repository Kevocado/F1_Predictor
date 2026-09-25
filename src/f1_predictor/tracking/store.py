"""store.py — SQLite persistence for session and championship prediction
track records. Mirrors PL_Predictor's tracking/store.py: snapshot each
prediction BEFORE the outcome is known, reconcile against actual results
once they land — the only honest way to measure "how good are the
predictions really," and this project's substitute for the market
comparison PL_Predictor's value-bet feature relies on (there is no F1 odds
market — see RESEARCH_BRIEF.md).

`session_predictions` generalizes the original race-only
`race_predictions` table to all four session types (sprint_qualifying,
qualifying, sprint, race) — keyed on (season, round, session_type,
driver_id, tier, market). A migration on first connect carries every
existing race_predictions row over with session_type='race' and
actual_finish_position renamed to actual_position; expected_position is
left NULL for those old rows since it was never actually predicted at the
time (not fabricated retroactively — same honesty rule the rest of this
app follows).

`championship_snapshots` is unchanged from before this migration.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ..config import TRACKING_DB_PATH

# Probability-market column names on the sim_table DataFrame are always
# p_win/p_podium/p_points_finish/p_dnf regardless of session type (see
# models/session_outcome.py::predict_session) — only the STORED market
# name differs, and quali-type sessions have no dnf market at all.
SESSION_MARKET_SPEC: dict[str, list[tuple[str, str]]] = {
    "race": [("win", "p_win"), ("podium", "p_podium"), ("points_finish", "p_points_finish"), ("dnf", "p_dnf")],
    "sprint": [("win", "p_win"), ("podium", "p_podium"), ("points_finish", "p_points_finish"), ("dnf", "p_dnf")],
    "qualifying": [("pole", "p_win"), ("top_3", "p_podium"), ("top_10", "p_points_finish")],
    "sprint_qualifying": [("pole", "p_win"), ("top_3", "p_podium"), ("top_10", "p_points_finish")],
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TRACKING_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            round INTEGER NOT NULL,
            race_name TEXT NOT NULL,
            session_type TEXT NOT NULL,
            driver_id TEXT NOT NULL,
            constructor_id TEXT,
            tier TEXT NOT NULL,
            market TEXT NOT NULL,
            predicted_prob REAL NOT NULL,
            expected_position REAL,
            session_time TEXT NOT NULL,
            snapshotted_at TEXT NOT NULL,
            model_trained_at TEXT,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_outcome INTEGER,
            actual_position INTEGER,
            actual_dnf INTEGER,
            resolved_at TEXT,
            backfilled INTEGER NOT NULL DEFAULT 0,
            UNIQUE(season, round, session_type, driver_id, tier, market)
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
    _migrate_legacy_race_predictions(conn)
    return conn


def _migrate_legacy_race_predictions(conn: sqlite3.Connection) -> None:
    """One-time, idempotent migration: if an old race_predictions table
    exists (pre-dating session_predictions), copy every row over with
    session_type='race' and actual_finish_position renamed to
    actual_position, then drop it. Safe to run on every connect — a no-op
    once the old table is gone."""
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='race_predictions'"
    ).fetchone()
    if not exists:
        return
    conn.execute(
        """
        INSERT OR IGNORE INTO session_predictions
            (season, round, race_name, session_type, driver_id, constructor_id, tier, market,
             predicted_prob, expected_position, session_time, snapshotted_at, model_trained_at,
             resolved, actual_outcome, actual_position, actual_dnf, resolved_at, backfilled)
        SELECT
            season, round, race_name, 'race', driver_id, constructor_id, tier, market,
            predicted_prob, NULL, session_time, snapshotted_at, model_trained_at,
            resolved, actual_outcome, actual_finish_position, actual_dnf, resolved_at, backfilled
        FROM race_predictions
        """
    )
    conn.execute("DROP TABLE race_predictions")
    conn.commit()


def record_session_predictions(
    sim_table: pd.DataFrame,
    season: int,
    round_: int,
    race_name: str,
    session_type: str,
    tier: str,
    session_time: str,
    model_trained_at: str | None = None,
    backfilled: bool = False,
) -> int:
    """Snapshot one session's Monte Carlo output (models/session_outcome.py
    ::predict_session's or race_outcome.py::simulate_race's return, with a
    constructor_id column already joined on by the caller). Already-logged
    (season, round, session_type, driver_id, tier, market) rows are left
    untouched — safe to call every time a prediction is served."""
    if sim_table.empty:
        return 0
    market_spec = SESSION_MARKET_SPEC[session_type]
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, r in sim_table.iterrows():
        expected_position = r.get("expected_position")
        expected_position = None if expected_position is None or pd.isna(expected_position) else float(expected_position)
        for market, prob_col in market_spec:
            rows.append(
                (
                    season,
                    round_,
                    race_name,
                    session_type,
                    r["driver_id"],
                    r.get("constructor_id"),
                    tier,
                    market,
                    float(r[prob_col]),
                    expected_position,
                    session_time,
                    now,
                    model_trained_at,
                    int(backfilled),
                )
            )
    with _connect() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO session_predictions
                (season, round, race_name, session_type, driver_id, constructor_id, tier, market,
                 predicted_prob, expected_position, session_time, snapshotted_at, model_trained_at, backfilled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount


def _actual_outcome(session_type: str, market: str, driver_row: pd.Series) -> int:
    position = driver_row.get("position")
    has_position = position is not None and not pd.isna(position)
    if market in ("win", "pole"):
        return int(has_position and int(position) == 1)
    if market in ("podium", "top_3"):
        return int(has_position and int(position) <= 3)
    if market == "points_finish":
        cutoff = 8 if session_type == "sprint" else 10
        return int(has_position and int(position) <= cutoff)
    if market == "top_10":
        return int(has_position and int(position) <= 10)
    if market == "dnf":
        return int(bool(driver_row.get("dnf")))
    raise ValueError(f"unknown market: {market}")


def reconcile_session_predictions(results_df: pd.DataFrame, session_type: str) -> int:
    """Fills in actual outcomes for every still-unresolved session_predictions
    row of `session_type` whose (season, round, driver_id) now has a real
    result in `results_df` (columns: season, round, driver_id, position,
    and dnf for race/sprint). Safe to call repeatedly — only touches
    resolved=0 rows."""
    if results_df.empty:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    results_lookup = results_df.set_index(["season", "round", "driver_id"])

    with _connect() as conn:
        unresolved = pd.read_sql(
            "SELECT id, season, round, driver_id, market FROM session_predictions "
            "WHERE resolved = 0 AND session_type = ?",
            conn,
            params=(session_type,),
        )
        if unresolved.empty:
            return 0

        updates = []
        for _, row in unresolved.iterrows():
            key = (row["season"], row["round"], row["driver_id"])
            if key not in results_lookup.index:
                continue
            driver_row = results_lookup.loc[key]
            if isinstance(driver_row, pd.DataFrame):
                driver_row = driver_row.iloc[0]
            actual = _actual_outcome(session_type, row["market"], driver_row)
            position = driver_row.get("position")
            updates.append(
                (
                    actual,
                    None if position is None or pd.isna(position) else int(position),
                    int(bool(driver_row.get("dnf"))) if "dnf" in driver_row else None,
                    now,
                    int(row["id"]),
                )
            )

        if updates:
            conn.executemany(
                """
                UPDATE session_predictions
                SET resolved = 1, actual_outcome = ?, actual_position = ?, actual_dnf = ?, resolved_at = ?
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


def made_before_session(snapshotted_at: str, session_time: str) -> bool:
    """True only when both times parse and the snapshot came strictly before
    the session. Anything written at or after the start (a seeded or
    back-filled snapshot) is rebuilt: shown, never counted."""
    try:
        snap = pd.Timestamp(snapshotted_at)
        start = pd.Timestamp(session_time)
    except (TypeError, ValueError):
        return False
    snap = snap.tz_localize("UTC") if snap.tzinfo is None else snap
    start = start.tz_localize("UTC") if start.tzinfo is None else start
    return bool(snap < start)


def _flag_pre_session(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pre_session"] = [made_before_session(a, b) for a, b in zip(df["snapshotted_at"], df["session_time"])]
    return df


def get_session_track_record(session_type: str | None = None, tier: str | None = None) -> dict:
    """Overall hit-rate/calibration summary, optionally filtered to one
    session type and/or tier."""
    with _connect() as conn:
        query = (
            "SELECT season, round, session_type, tier, market, predicted_prob, actual_outcome, snapshotted_at, session_time "
            "FROM session_predictions WHERE resolved = 1"
        )
        params: list = []
        if session_type:
            query += " AND session_type = ?"
            params.append(session_type)
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return {"n_resolved": 0, "by_market": [], "n_rebuilt_sessions": 0}

    # Only snapshots made before their session are judged.
    df = _flag_pre_session(df)
    rebuilt = df.loc[~df["pre_session"], ["season", "round", "session_type", "tier"]].drop_duplicates()
    df = df[df["pre_session"]]
    if df.empty:
        return {"n_resolved": 0, "by_market": [], "n_rebuilt_sessions": int(len(rebuilt))}

    rows = []
    for (st, t, market), g in df.groupby(["session_type", "tier", "market"]):
        brier = float(((g["predicted_prob"] - g["actual_outcome"]) ** 2).mean())
        rows.append(
            {
                "session_type": st,
                "tier": t,
                "market": market,
                "n": int(len(g)),
                "brier": brier,
                "hit_rate": float(g["actual_outcome"].mean()),
                "avg_predicted_prob": float(g["predicted_prob"].mean()),
            }
        )
    return {"n_resolved": int(len(df)), "by_market": rows, "n_rebuilt_sessions": int(len(rebuilt))}


def get_session_accuracy(session_type: str | None = None, tier: str | None = None) -> list[dict]:
    """Per-session accuracy: did the model's own top pick actually land
    there — mirrors the old get_race_accuracy, generalized across markets
    per session type via SESSION_MARKET_SPEC."""
    top_n_by_market = {"win": 1, "pole": 1, "podium": 3, "top_3": 3, "points_finish": 10, "top_10": 10}
    with _connect() as conn:
        query = (
            "SELECT season, round, race_name, session_type, tier, driver_id, market, predicted_prob, actual_outcome, "
            "snapshotted_at, session_time FROM session_predictions WHERE resolved = 1 AND market != 'dnf'"
        )
        params: list = []
        if session_type:
            query += " AND session_type = ?"
            params.append(session_type)
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return []

    rows = []
    for (season, round_, race_name, st, t), race_df in df.groupby(["season", "round", "race_name", "session_type", "tier"]):
        # A session counts only if every snapshot row was made before it.
        pre = all(made_before_session(a, b) for a, b in zip(race_df["snapshotted_at"], race_df["session_time"]))
        row = {
            "season": int(season), "round": int(round_), "race_name": race_name, "session_type": st, "tier": t,
            "rebuilt": not pre,
        }
        for market in race_df["market"].unique():
            n = top_n_by_market.get(market)
            if n is None:
                continue
            market_df = race_df[race_df["market"] == market]
            predicted = set(market_df.nlargest(n, "predicted_prob")["driver_id"])
            actual = set(market_df.loc[market_df["actual_outcome"] == 1, "driver_id"])
            row[f"{market}_predicted"] = sorted(predicted)
            row[f"{market}_actual"] = sorted(actual)
            row[f"{market}_hits"] = len(predicted & actual)
            row[f"{market}_of"] = n
        rows.append(row)

    return sorted(rows, key=lambda r: (r["season"], r["round"]), reverse=True)


def get_session_prediction(season: int, round_: int, session_type: str, tier: str | None = None) -> list[dict]:
    with _connect() as conn:
        query = "SELECT * FROM session_predictions WHERE season = ? AND round = ? AND session_type = ?"
        params: list = [season, round_, session_type]
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
