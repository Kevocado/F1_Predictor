"""schemas.py — pydantic response models for the FastAPI layer."""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class MissingSeason(BaseModel):
    """A season the model needed and did not get, and why.

    `reason` is required with no default. A skip carrying no reason is the same
    silence the WARNING log line used to be, relocated to a place fewer people
    look at — the whole reason this block exists is that a reader of the
    response has no other way to learn it.
    """
    season: int
    reason: str


class HistoryCoverage(BaseModel):
    """Whether the forecast behind this response was built on the full history
    window, and if not, which seasons are absent.

    An object rather than a bare `history_complete: bool`, for two reasons. A
    false needs to say WHICH seasons and WHY — a lone boolean is the log line
    again, minus the log. And a bare true is unfalsifiable: nothing in the
    payload says which seasons were expected, so a reader cannot check it. The
    requested and loaded lists are there to make the claim checkable.

    `complete` is derived from `missing_seasons` by the validator below, never
    taken on trust: a payload claiming a complete window while naming three
    missing seasons is rejected rather than served.
    """
    complete: bool
    seasons_requested: list[int]
    seasons_loaded: list[int]
    missing_seasons: list[MissingSeason] = []

    @model_validator(mode="after")
    def _complete_matches_the_missing_seasons(self) -> "HistoryCoverage":
        if self.complete is not (not self.missing_seasons):
            raise ValueError(
                f"history coverage claims complete={self.complete} while naming "
                f"{len(self.missing_seasons)} missing season(s); the flag is "
                "derived from the list, not stated beside it"
            )
        return self


class RaceSummary(BaseModel):
    season: int
    round: int
    race_name: str
    circuit_name: str
    race_datetime: str | None
    is_sprint_weekend: bool
    completed: bool


class DriverPrediction(BaseModel):
    driver_id: str
    # Display name. The site used to title-case driver_id, which turns
    # "russell" into "Russell" and drops the first name. The hub needs the
    # same string the site shows, so it lives here rather than being derived
    # twice. Never null: falls back to the id, so a driver with no name in
    # the results feed still renders as something a human can read.
    driver_name: str
    constructor_id: str | None = None
    p_win: float
    p_podium: float
    p_points_finish: float
    p_dnf: float
    # Optional because they are NaN in 287 of the 515 race rows of the
    # committed snapshot (measured: rounds 1-12 and 15 — the tracked-
    # completed path sets them to NaN when the stored prediction has no
    # expected values), pydantic 2.13.5 serialises NaN to null on the way
    # out, and the snapshot-sanitising task in this plan writes literal null
    # where that NaN used to be — a non-optional float rejects None on
    # re-validation, so the endpoint would 500 on its own response.
    expected_position: float | None = None
    expected_points: float | None = None
    actual_position: int | None = None
    actual_dnf: bool | None = None


class RacePredictionResponse(BaseModel):
    season: int
    round: int
    race_name: str
    tier: str
    source: str  # "live" (computed fresh), "tracked" (snapshot made before the session), "rebuilt" (snapshot written after it), "backtest" (historical replay)
    predictions: list[DriverPrediction]
    # When this session starts, from the schedule row. Nullable so
    # the endpoint never 500s on a missing value from an older
    # committed snapshot that predates the field.
    session_datetime: str | None = None
    # Whether this forecast was built on every season the model asked for.
    # None is a real third state, not a default to be filled in: a prediction
    # served from a stored snapshot or a backtest replay was built from a
    # history load this request never made, so there is nothing to report.
    # Defaulting that to `complete: true` would be the one value that is a
    # lie — "nothing was skipped" is not knowable here — so it is None, and
    # the UI reads None as silence rather than as reassurance. Nullable for
    # the same reason as session_datetime: the committed public_snapshot.json
    # predates this field and the public deployment serves it verbatim.
    history: HistoryCoverage | None = None


class SessionDriverPrediction(BaseModel):
    driver_id: str
    constructor_id: str | None = None
    # Qualifying-type markets (sprint_qualifying, qualifying) — omitted for race/sprint.
    p_pole: float | None = None
    p_top_3: float | None = None
    p_top_10: float | None = None
    # Race-type markets (sprint, race) — omitted for sprint_qualifying/qualifying.
    p_win: float | None = None
    p_podium: float | None = None
    p_points_finish: float | None = None
    p_dnf: float | None = None
    # Same null-tolerance as DriverPrediction.expected_position: the routes
    # pass float("nan") for a driver with no stored expected_position (their
    # own fallback), pydantic serialises that NaN to null, and the
    # sanitised snapshot will carry literal null once Task 2 lands — a
    # non-optional float would 500 the three session-prediction endpoints on
    # their own responses.
    expected_position: float | None = None
    actual_position: int | None = None
    actual_dnf: bool | None = None


class SessionPredictionResponse(BaseModel):
    season: int
    round: int
    race_name: str
    session_type: str  # "sprint_qualifying" | "qualifying" | "sprint" | "race"
    tier: str
    source: str  # "live" | "tracked" | "rebuilt" | "backtest"
    predictions: list[SessionDriverPrediction]
    # When this specific session starts, from the schedule row.
    # Nullable for the same reason as RacePredictionResponse.
    session_datetime: str | None = None


# Session ordering from earliest to latest in a race weekend.
# A sprint weekend runs: sprint_qualifying → sprint → qualifying → race.
# A normal weekend runs: qualifying → race.
# The hub uses this to know which session comes next rather than
# picking "the earliest session with a prediction" (which would
# jump straight to a race days early, because the snapshot already
# carries race predictions for future rounds with source: live).
SESSION_ORDER = {
    False: ["qualifying", "race"],
    True: ["sprint_qualifying", "sprint", "qualifying", "race"],
}


class ChampionshipEntry(BaseModel):
    entity_id: str
    win_prob: float
    top3_prob: float
    expected_final_points: float
    expected_final_rank: float


class ChampionshipResponse(BaseModel):
    season: int
    as_of_round: int
    n_trials: int
    standings: list[ChampionshipEntry]


class TrackRecordEntry(BaseModel):
    tier: str
    market: str
    n: int
    brier: float
    hit_rate: float
    avg_predicted_prob: float


class TrackRecordSubset(BaseModel):
    """The secondary figure: the same ledger over the picks made BEFORE the
    session started.

    A whole sub-record rather than three loose numbers, so the pre-session
    figure carries its own n and its own per-market rows exactly as the
    headline does. Its `n_resolved` is the size of that subset by
    construction -- it is the same summariser over the same counted rows,
    filtered by the derivation in `tracking.store.made_before_session`, not a
    figure reconciled by hand.
    """

    n_resolved: int
    by_market: list[TrackRecordEntry]


class TrackRecordPick(BaseModel):
    """One recorded pick, with when it was made.

    Disclosure is per pick, not only in aggregate. `made_before_session` is
    derived on every read from this row's own `snapshotted_at` against its own
    `session_time`, compared as UTC instants and failing closed to False, and
    BOTH timestamps are published beside it so a reader can check the
    comparison rather than take it on trust. The stored `backfilled` column is
    neither consulted nor published: a stored flag is what this rule exists to
    rule out.

    A grand prix is a weekend, not a kickoff -- hence `session_time` and
    `made_before_session` rather than borrowed kickoff language.
    """

    season: int
    round: int
    race_name: str | None = None
    session_type: str
    tier: str
    driver_id: str
    market: str
    predicted_prob: float
    actual_outcome: int
    made_before_session: bool
    snapshotted_at: str
    session_time: str
    # False for a rerun that lost the earliest-pick contest.
    counted: bool = True


class TrackRecordResponse(BaseModel):
    """The track record: every recorded pick, and the pre-session subset.

    `n_resolved` and `by_market` keep their names and their meaning as
    "resolved picks in the record". What changed on 2026-10-01
    (predictor-hub #66) is WHICH resolved picks that is: the headline is every
    counted pick -- one per (session, driver, market), the earliest recorded,
    whenever it was made -- where it used to be the sessions every row of which
    was snapshotted before it started.
    """

    n_resolved: int
    by_market: list[TrackRecordEntry]
    # The pre-session subset beside the headline: what the model would have said
    # on the weekend. The figure to read for live performance.
    pre_session: TrackRecordSubset
    # The two counts behind the reconciliation, for a one-number read:
    # n_resolved == n_pre_session + n_post_session_picks.
    n_pre_session: int = 0
    n_post_session_picks: int = 0
    # RENAMED IN MEANING, name and unit kept: this used to count sessions LEFT
    # OUT of the record. It is now the number of sessions with at least one
    # counted pick recorded at or after their own start -- the disclosure, not
    # the exclusion. Nothing is withheld by it any more.
    n_rebuilt_sessions: int = 0
    # Every recorded pick, counted or not, with its own timestamps and its own
    # pre-session label.
    per_pick: list[TrackRecordPick] = []


class FeatureContribution(BaseModel):
    feature: str
    value: float | None
    contribution: float


class ExplainResponse(BaseModel):
    season: int
    round: int
    driver_id: str
    candidate: str  # "elo" or "xgb_ranker"
    strength_contributors: list[FeatureContribution]  # drives win/podium/points together
    dnf_contributors: list[FeatureContribution]


class RaceAccuracyEntry(BaseModel):
    season: int
    round: int
    race_name: str
    tier: str
    # Snapshot written after the session started: shown, never counted.
    rebuilt: bool = False
    win_predicted: list[str] = []
    win_actual: list[str] = []
    win_hits: int = 0
    win_of: int = 1
    podium_predicted: list[str] = []
    podium_actual: list[str] = []
    podium_hits: int = 0
    podium_of: int = 3
    points_finish_predicted: list[str] = []
    points_finish_actual: list[str] = []
    points_finish_hits: int = 0
    points_finish_of: int = 10
