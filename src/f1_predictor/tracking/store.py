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


def _as_utc_instant(value) -> pd.Timestamp | None:
    """An ISO timestamp as an aware UTC instant, or None when it cannot be read.

    Naive values are UTC, which is how `record_session_predictions` writes
    `snapshotted_at` (`datetime.now(timezone.utc)`) and how a session start
    arrives from the schedule; the localisation is explicit so the answer does
    not depend on the machine this runs on.
    """
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if stamp is pd.NaT or pd.isna(stamp):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def made_before_session(snapshotted_at: str, session_time: str) -> bool:
    """Whether this row's OWN timestamps prove the pick was made before the
    session started.

    DERIVED, on every read, from the two timestamps and nothing else. The
    `backfilled` column exists in `session_predictions` and is deliberately NOT
    consulted: a stored flag can drift out of step with the timestamps it claims
    to describe, and there would be a pre-existing backfill to get wrong.
    `test_the_pre_session_subset_is_never_labelled_with_a_stored_flag` plants a
    row whose flag and timestamps disagree and requires the timestamps to win.

    Both sides are compared as UTC INSTANTS, never as strings and never as
    naive wall clocks. That distinction is the whole risk here, because a
    `session_time` can arrive with any offset, so `2026-09-12T23:30:00+09:00`
    (14:30Z) is half an hour BEFORE a `2026-09-12T15:00:00+00:00` start even
    though the wall clocks read "23:30" against "15:00" and sort the other way.
    `test_made_before_session_is_derived_from_utc_instants_not_wall_clock_
    strings` is that pair plus its mirror, each asserting the string order as a
    precondition; `test_made_before_session_ignores_the_machine_timezone` runs
    it under four host zones.

    Fails CLOSED, and that is the rule's one remaining prohibition: an
    unreadable stamp on either side gives False. Such a row still COUNTS as a
    recorded pick; it is simply never presented as a pre-session one, because
    nothing here can prove that it was.

    A grand prix is a weekend, not a kickoff, and this comparison is about the
    session's start. Nothing in this repo calls it a kickoff.
    """
    snap = _as_utc_instant(snapshotted_at)
    start = _as_utc_instant(session_time)
    if snap is None or start is None:
        return False
    return bool(snap < start)


def _flag_pre_session(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `made_before_session`, derived live from each row's own timestamps."""
    df = df.copy()
    df["made_before_session"] = [
        made_before_session(a, b) for a, b in zip(df["snapshotted_at"], df["session_time"])
    ]
    return df


# The key a counted pick is unique on: `session_predictions`' own UNIQUE key.
# The driver is in it because a `win` call is one call per DRIVER per session --
# twenty drivers, twenty picks -- and the published rows are grouped by
# (session_type, tier, market), so the tier has to be in it too or a qualifying
# snapshot would displace a post-qualifying one.
_COUNTING_KEY = ["season", "round", "session_type", "tier", "driver_id", "market"]


def _earliest_recorded(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, driver, market), the EARLIEST recorded.

    Rule 2 of the 2026-10-01 spec: a later rerun of the model on the same
    session is kept as history, but it neither replaces the counted pick nor
    counts a second time. Without that, re-running until the model was right
    would be free.

    `record_session_predictions` is INSERT OR IGNORE on that UNIQUE key, so no
    writer in this repo can today put two rows on one key: the earliest
    recorded pick IS the row, and a later one is kept nowhere. It is implemented
    in the reader anyway, because a rule enforced only by a primary key stops
    being enforced the moment the key changes, and the failure when it does is
    invisible -- the Brier score simply improves, silently, and a rerun of the
    model gets to grade a second time.

    "Earliest" is by UTC instant, so which row wins does not depend on how its
    timestamp happens to be spelled. A stamp that cannot be parsed cannot be
    proven earliest, so it sorts last and never displaces a row carrying a real
    instant; where a key has only unreadable stamps the first row stands and
    `made_before_session` fails closed on it rather than the pick being lost.

    Reversible in one place: count the LATEST row per key instead (`.head(1)`
    -> `.tail(1)`) if that is ever the better answer.
    """
    if df.empty:
        return df
    ordered = df.assign(
        _instant=[_as_utc_instant(s) for s in df["snapshotted_at"]]
    ).sort_values("_instant", kind="stable", na_position="last")
    return ordered.drop_duplicates(subset=_COUNTING_KEY, keep="first").drop(columns="_instant")


def _by_market(df: pd.DataFrame) -> list[dict]:
    """The per-market ledger, over whatever frame it is handed.

    Split out of `get_session_track_record` so the headline and the pre-session
    subset are computed by the same code over two frames. They cannot then
    disagree about a market family, a tier or a driver -- only about which rows.
    """
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
    return rows


def _per_pick(df: pd.DataFrame, counted: pd.DataFrame) -> list[dict]:
    """Every recorded pick, counted or not, with its timing disclosed.

    A row the rule did not count is still published: rule 1 keeps a rerun as
    history, and hiding it would defeat the point of an append-only record.
    `counted` says which pick the headline scored, `made_before_session` says
    when it was made, and `snapshotted_at` and `session_time` are both on the
    row so a reader can check that comparison for themselves.
    """
    counted_keys = {tuple(row[key] for key in _COUNTING_KEY) for _, row in counted.iterrows()}
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "season": int(row["season"]),
                "round": int(row["round"]),
                "race_name": row.get("race_name"),
                "session_type": row["session_type"],
                "tier": row["tier"],
                "driver_id": row["driver_id"],
                "market": row["market"],
                "predicted_prob": float(row["predicted_prob"]),
                "actual_outcome": int(row["actual_outcome"]),
                "made_before_session": bool(row["made_before_session"]),
                "snapshotted_at": row["snapshotted_at"],
                "session_time": row["session_time"],
                "counted": tuple(row[key] for key in _COUNTING_KEY) in counted_keys,
            }
        )
    return sorted(rows, key=lambda r: (r["snapshotted_at"], r["season"], r["round"], r["driver_id"], r["market"]))


def get_session_track_record(session_type: str | None = None, tier: str | None = None) -> dict:
    """The track record: every recorded pick, and the pre-session subset beside it.

    Kevin, 2026-10-01 (`predictor-hub`
    docs/superpowers/specs/2026-10-01-track-record-counts-every-pick.md, merged
    as predictor-hub #66), verbatim: "i dont really care about picks made after
    kickoff because im always re running the models ... with every model change
    it will stop tracking ... make it that whats recorded remains recorded and
    then just use every prediction we make for the track record stuff."

    Before this, a session was judged only if EVERY one of its rows was
    snapshotted before it started, and a session with any late row was dropped
    whole and reported as `n_rebuilt_sessions`. On the shipped
    `data/tracking.db` that withheld **1,056 of 1,452** graded picks -- 12 of
    17 recorded sessions, and the worst single market published at n=22 when 286
    were recorded. Every one of those was a recorded pick, and the model had
    been re-run on the games; that is the "stops tracking" failure, measured.

    So each return now carries:

      * the headline over COUNTED picks -- one per (session, driver, market),
        the earliest recorded, whenever it was made. `n_resolved` and
        `by_market` keep their names and their meaning as "resolved picks in the
        record"; what changed is which resolved picks that is, so a site reading
        `by_market` reads the fuller record and nothing has to be renamed.
      * `pre_session`, the same keys over the subset whose own timestamps prove
        they were made before their session started, with its own `n_resolved`
        and its own `by_market`. A whole sub-record rather than loose numbers,
        so the secondary figure is a ledger in the same shape as the headline.
      * `n_pre_session` and `n_post_session_picks` for a one-number read, and
        the identity `n_resolved == n_pre_session + n_post_session_picks`.
      * `per_pick`, one row per recorded pick carrying `made_before_session`,
        `snapshotted_at` and `session_time`, so disclosure is per pick and
        auditable rather than only in aggregate.

    `n_rebuilt_sessions` is RETAINED and still counts SESSIONS: how many had at
    least one counted pick recorded at or after their own start. It used to mean
    "left out of the headline"; it no longer does, and keeping the unit it
    already had is what lets the field survive a change of meaning without a
    rename -- a reader who wants the number of withheld sessions subtracts the
    pre-session figure themselves.

    The counting key is the table's own UNIQUE key,
    (season, round, session_type, tier, driver_id, market). The doc's
    "(game, market)" would be DESTRUCTIVE here: a `win` call is one call per
    driver per session, so keying on the session alone would keep one row and
    delete the other nineteen. See
    `tests/test_track_record_counted_picks.py::test_the_counted_key_carries_
    the_driver_so_two_drivers_on_one_market_both_survive`.

    **A session, not a kickoff.** A grand prix is a weekend; the comparison
    above is against the session's own start, and the label says so.
    """
    with _connect() as conn:
        query = (
            "SELECT season, round, race_name, session_type, tier, driver_id, market, predicted_prob, "
            "actual_outcome, snapshotted_at, session_time "
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
        return {
            "n_resolved": 0, "by_market": [], "n_rebuilt_sessions": 0,
            "n_pre_session": 0, "n_post_session_picks": 0,
            "pre_session": {"n_resolved": 0, "by_market": []}, "per_pick": [],
        }

    labelled = _flag_pre_session(df)
    # Rule 2: one counted pick per (session, driver, market), the earliest
    # recorded. Every resolved row is a recorded pick, so the headline covers
    # all of them -- the old rule dropped whole sessions here.
    counted = _earliest_recorded(labelled)
    pre_session = counted[counted["made_before_session"]]

    session_keys = ["season", "round", "session_type", "tier"]
    with_late = counted[~counted["made_before_session"]]
    n_rebuilt_sessions = (
        int(with_late[session_keys].drop_duplicates().shape[0]) if not with_late.empty else 0
    )

    return {
        "n_resolved": int(len(counted)),
        "by_market": _by_market(counted),
        # The secondary figure: the same summariser over the pre-session subset.
        "pre_session": {"n_resolved": int(len(pre_session)), "by_market": _by_market(pre_session)},
        "n_pre_session": int(len(pre_session)),
        "n_post_session_picks": int(len(counted) - len(pre_session)),
        # Retained, same unit, new meaning: sessions with at least one counted
        # pick recorded at or after their own start. Nothing is withheld by it.
        "n_rebuilt_sessions": n_rebuilt_sessions,
        "per_pick": _per_pick(labelled, counted),
    }


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
