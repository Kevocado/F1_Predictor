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
    constructor_id: str | None = None
    p_win: float
    p_podium: float
    p_points_finish: float
    p_dnf: float
    expected_position: float
    expected_points: float
    actual_position: int | None = None
    actual_dnf: bool | None = None


class RacePredictionResponse(BaseModel):
    season: int
    round: int
    race_name: str
    tier: str
    source: str  # "live" (computed fresh), "tracked" (served from tracking store), "backtest" (honest historical replay)
    predictions: list[DriverPrediction]


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
