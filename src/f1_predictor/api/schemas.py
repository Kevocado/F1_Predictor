"""schemas.py — pydantic response models for the FastAPI layer."""

from __future__ import annotations

from pydantic import BaseModel


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


class TrackRecordResponse(BaseModel):
    n_resolved: int
    by_market: list[TrackRecordEntry]
    # Sessions whose only snapshot was written after they started: left out.
    n_rebuilt_sessions: int = 0


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
