"""routes.py — thin JSON-serialization layer over the existing
`f1_predictor` package, mirroring PL_Predictor's api/routes.py: no new
modeling logic lives here, just orchestration of the existing data/
features/models/tracking functions and reshaping their output for the
frontend.
"""

from __future__ import annotations

import time

import pandas as pd
import xgboost as xgb
from fastapi import APIRouter, Depends, HTTPException

from ..config import CURRENT_SEASON, PUBLIC_MODE
from ..data import jolpica
from ..evaluate import backtest as backtest_lib
from ..features import build as build_features
from ..features import session_state
from ..models import championship_projection
from ..models import dnf as dnf_model
from ..models import explain as explain_lib
from ..models import manifest as manifest_module
from ..models import race_outcome
from ..tracking import store
from .schemas import (
    ChampionshipEntry,
    ChampionshipResponse,
    DriverPrediction,
    ExplainResponse,
    FeatureContribution,
    RaceAccuracyEntry,
    RacePredictionResponse,
    RaceSummary,
    TrackRecordEntry,
    TrackRecordResponse,
)

router = APIRouter(prefix="/api")


def _admin_only() -> None:
    """Dependency for the one state-changing endpoint (retrain) — 404s it
    unconditionally on the public deployment. Pretending the route doesn't
    exist, rather than 403, avoids advertising an admin surface to a
    public visitor at all. Ported verbatim from PL_Predictor's routes.py."""
    if PUBLIC_MODE:
        raise HTTPException(status_code=404)

_CACHE_TTL_SECONDS = 1800
# Race schedule/results genuinely change within minutes on a race weekend —
# same reasoning as PL_Predictor's _LIVE_CACHE_TTL_SECONDS.
_LIVE_CACHE_TTL_SECONDS = 300
_cache: dict[str, tuple[float, object]] = {}


def _cached(key: str, build_fn, ttl: float = _CACHE_TTL_SECONDS):
    hit = _cache.get(key)
    if hit and (time.time() - hit[0]) < ttl:
        return hit[1]
    value = build_fn()
    _cache[key] = (time.time(), value)
    return value


def _require_manifest() -> dict:
    if not manifest_module.MANIFEST_PATH.exists():
        raise HTTPException(status_code=409, detail="No trained models yet — call POST /api/retrain first.")
    return manifest_module.load_manifest()


def _load_models() -> tuple[str, xgb.XGBRanker | None, xgb.XGBClassifier]:
    manifest = _require_manifest()
    candidate = manifest["race_outcome_candidate"]
    ranker = None
    if candidate == "xgb_ranker":
        ranker = xgb.XGBRanker()
        ranker.load_model(str(manifest_module.RACE_OUTCOME_MODEL_PATH))
    dnf_clf = xgb.XGBClassifier()
    dnf_clf.load_model(str(manifest_module.DNF_MODEL_PATH))
    return candidate, ranker, dnf_clf


def _future_feature_frame(season: int, round_: int, race_row: pd.Series) -> tuple[pd.DataFrame, str]:
    """The feature frame for a race that hasn't happened yet, shared by
    the prediction path and the explain path: current elo/team-strength/
    rolling-form snapshots, plus real qualifying data merged in if this
    weekend has already qualified but not yet raced."""
    tier = session_state.current_tier(race_row)

    schedule = jolpica.fetch_season_schedule(season)
    race_schedule = schedule[schedule["round"] == round_]
    results_df = jolpica.load_season_results(season)
    driver_ids = jolpica.fetch_driver_standings(season)["driver_id"].tolist()
    future_df = championship_projection.build_future_feature_rows(
        season, race_schedule, results_df, schedule, driver_ids
    )

    if tier == session_state.TIER_POST_QUALIFYING:
        quali_df = jolpica.fetch_qualifying(season, round_)
        if not quali_df.empty:
            quali_feat = build_features.quali_features(quali_df)[["driver_id", "quali_position", "quali_gap_to_pole"]]
            future_df = future_df.drop(columns=["quali_position", "quali_gap_to_pole"]).merge(
                quali_feat, on="driver_id", how="left"
            )
            # Official grid can still differ after penalties are applied —
            # qualifying position is the best available pre-race estimate.
            future_df["grid"] = future_df["quali_position"]

    return future_df, tier


def _predict_upcoming_race(season: int, round_: int, race_row: pd.Series) -> tuple[pd.DataFrame, str]:
    """A race that hasn't happened yet: predicted with the production
    manifest model directly (already trained on all history through the
    most recently completed race — no per-request retraining needed)."""
    future_df, tier = _future_feature_frame(season, round_, race_row)
    candidate, ranker, dnf_clf = _load_models()
    if candidate == "xgb_ranker":
        scores = race_outcome.xgb_scores_for_race(ranker, future_df, build_features.FEATURE_COLUMNS)
        theta = race_outcome.theta_from_xgb_scores(scores)
    else:
        theta = race_outcome.theta_from_elo_strength(race_outcome.elo_strengths(future_df))

    dnf_prob = dnf_model.predict_dnf_prob(dnf_clf, future_df, build_features.FEATURE_COLUMNS)
    sim = race_outcome.simulate_race(theta, dnf_prob, n_trials=10000)

    constructor_lookup = future_df.set_index("driver_id")["constructor_id"]
    sim["constructor_id"] = sim["driver_id"].map(constructor_lookup)
    return sim, tier


def _completed_race_prediction(season: int, round_: int) -> tuple[pd.DataFrame, str, str]:
    """A race that's already happened: prefer the honest snapshot recorded
    in tracking/store.py *before* it happened; fall back to an on-demand
    no-lookahead reconstruction (evaluate/backtest.py) if nothing was ever
    snapshotted — same "backfilled" gap PL_Predictor's tracking store
    handles explicitly rather than silently."""
    tracked = store.get_race_prediction(season, round_, tier=session_state.TIER_POST_QUALIFYING)
    if tracked:
        df = pd.DataFrame(tracked)
        # Pivot on driver_id alone (not also constructor_id) — a NULL
        # constructor_id (an older snapshot recorded before backtest_race
        # carried it) would otherwise make pivot_table's default
        # dropna=True silently drop that driver's row entirely.
        pivot = df.pivot_table(
            index="driver_id", columns="market", values="predicted_prob", aggfunc="first"
        ).reset_index()
        pivot = pivot.rename(
            columns={"win": "p_win", "podium": "p_podium", "points_finish": "p_points_finish", "dnf": "p_dnf"}
        )
        extra = df.drop_duplicates("driver_id")[["driver_id", "constructor_id", "actual_finish_position", "actual_dnf"]]
        extra = extra.rename(columns={"actual_finish_position": "actual_position"})
        pivot = pivot.merge(extra, on="driver_id", how="left")
        pivot["expected_position"] = float("nan")
        pivot["expected_points"] = float("nan")
        return pivot, session_state.TIER_POST_QUALIFYING, "tracked"

    result = backtest_lib.backtest_race(season, round_)
    result = result.rename(columns={"position": "actual_position", "dnf": "actual_dnf"})
    return result, session_state.TIER_POST_QUALIFYING, "backtest"


def _race_feature_frame_for_explain(season: int, round_: int) -> tuple[pd.DataFrame, list[str]]:
    """The feature row(s) for explain_prediction — always built from the
    CURRENT production model's training frame, not evaluate/backtest.py's
    no-lookahead retrain (that discipline exists for honest accuracy
    scoring; explain shows what the deployed model actually attributes,
    so using it directly is correct here, and much cheaper — no retrain
    per request)."""
    schedule = jolpica.fetch_season_schedule(season)
    race_rows = schedule[schedule["round"] == round_]
    if race_rows.empty:
        raise HTTPException(status_code=404, detail=f"No such race: {season} round {round_}")
    race_row = race_rows.iloc[0]
    now = pd.Timestamp.now(tz="UTC")
    completed = pd.notna(race_row["race_datetime"]) and race_row["race_datetime"] <= now

    if completed:
        df, feature_cols = _cached(
            f"explain_frame_{season}",
            lambda: build_features.build_training_frame(seasons=jolpica.default_seasons()),
            ttl=_CACHE_TTL_SECONDS,
        )
        race_df = df[
            (df["season"] == season) & (df["round"] == round_) & (df["tier"] == session_state.TIER_POST_QUALIFYING)
        ]
        return race_df, feature_cols

    future_df, _tier = _future_feature_frame(season, round_, race_row)
    return future_df, build_features.FEATURE_COLUMNS


@router.get("/races", response_model=list[RaceSummary])
def list_races(season: int | None = None) -> list[RaceSummary]:
    season = season or CURRENT_SEASON
    schedule = _cached(f"schedule_{season}", lambda: jolpica.fetch_season_schedule(season), ttl=_LIVE_CACHE_TTL_SECONDS)
    now = pd.Timestamp.now(tz="UTC")
    out = []
    for _, r in schedule.iterrows():
        completed = pd.notna(r["race_datetime"]) and r["race_datetime"] <= now
        out.append(
            RaceSummary(
                season=season,
                round=int(r["round"]),
                race_name=r["race_name"],
                circuit_name=r["circuit_name"],
                race_datetime=None if pd.isna(r["race_datetime"]) else r["race_datetime"].isoformat(),
                is_sprint_weekend=bool(r["is_sprint_weekend"]),
                completed=bool(completed),
            )
        )
    return out


def _race_prediction_bundle(season: int, round_: int, race_row: pd.Series, completed: bool) -> tuple[pd.DataFrame, str, str]:
    if completed:
        return _completed_race_prediction(season, round_)
    sim, tier = _predict_upcoming_race(season, round_, race_row)
    return sim, tier, "live"


@router.get("/races/{season}/{round_}/prediction", response_model=RacePredictionResponse)
def get_race_prediction(season: int, round_: int) -> RacePredictionResponse:
    schedule = jolpica.fetch_season_schedule(season)
    race_rows = schedule[schedule["round"] == round_]
    if race_rows.empty:
        raise HTTPException(status_code=404, detail=f"No such race: {season} round {round_}")
    race_row = race_rows.iloc[0]
    now = pd.Timestamp.now(tz="UTC")
    completed = pd.notna(race_row["race_datetime"]) and race_row["race_datetime"] <= now

    # Uncached, this rebuilt the whole future-feature frame (elo/team-
    # strength replay across the season) on every request for an upcoming
    # race, and for a completed race with no tracked snapshot, fell
    # through to backtest_race — which re-trains a real model on a
    # multi-season frame *per request*. Confirmed directly (not assumed)
    # while porting PL_Predictor's PUBLIC_MODE pattern here: PL_Predictor's
    # own OOM was caused by exactly this shape of per-request rebuild
    # bypassing an otherwise-present cache.
    sim, tier, source = _cached(
        f"race_prediction_{season}_{round_}",
        lambda: _race_prediction_bundle(season, round_, race_row, completed),
        ttl=_LIVE_CACHE_TTL_SECONDS,
    )

    predictions = []
    for _, r in sim.iterrows():
        predictions.append(
            DriverPrediction(
                driver_id=r["driver_id"],
                constructor_id=r.get("constructor_id"),
                p_win=float(r["p_win"]),
                p_podium=float(r["p_podium"]),
                p_points_finish=float(r["p_points_finish"]),
                p_dnf=float(r["p_dnf"]),
                expected_position=float(r["expected_position"]) if pd.notna(r.get("expected_position")) else float("nan"),
                expected_points=float(r["expected_points"]) if pd.notna(r.get("expected_points")) else float("nan"),
                actual_position=None if pd.isna(r.get("actual_position")) else int(r.get("actual_position")),
                actual_dnf=None if pd.isna(r.get("actual_dnf")) else bool(r.get("actual_dnf")),
            )
        )
    predictions.sort(key=lambda p: -p.p_win)

    return RacePredictionResponse(
        season=season, round=round_, race_name=race_row["race_name"], tier=tier, source=source, predictions=predictions
    )


@router.get("/races/{season}/{round_}/explain", response_model=ExplainResponse)
def explain_prediction(season: int, round_: int, driver_id: str) -> ExplainResponse:
    """What's driving this driver's prediction — one explanation for
    Win/Podium/Points-finish (they're derived from the same strength
    score, see models/explain.py), a separate one for DNF."""
    race_df, feature_cols = _race_feature_frame_for_explain(season, round_)
    driver_row = race_df[race_df["driver_id"] == driver_id]
    if driver_row.empty:
        raise HTTPException(
            status_code=404, detail=f"No feature row for driver '{driver_id}' in {season} round {round_}"
        )

    candidate, ranker, dnf_clf = _load_models()
    if candidate == "xgb_ranker":
        strength_contribs = explain_lib.explain_strength_xgb(ranker, driver_row, feature_cols)
        strength_cols = feature_cols
    else:
        strength_contribs = explain_lib.explain_strength_elo(driver_row)
        strength_cols = ["elo_pre_race", "team_strength_pre_race"]
    dnf_contribs = explain_lib.explain_dnf(dnf_clf, driver_row, feature_cols)

    strength_top = explain_lib.top_contributors(strength_contribs.iloc[0], driver_row.iloc[0], strength_cols)
    dnf_top = explain_lib.top_contributors(dnf_contribs.iloc[0], driver_row.iloc[0], feature_cols)

    return ExplainResponse(
        season=season,
        round=round_,
        driver_id=driver_id,
        candidate=candidate,
        strength_contributors=[FeatureContribution(**c) for c in strength_top],
        dnf_contributors=[FeatureContribution(**c) for c in dnf_top],
    )


@router.get("/championship/{championship}", response_model=ChampionshipResponse)
def get_championship(championship: str, season: int | None = None, n_trials: int = 2000) -> ChampionshipResponse:
    if championship not in ("drivers", "constructors"):
        raise HTTPException(status_code=400, detail="championship must be 'drivers' or 'constructors'")
    season = season or CURRENT_SEASON
    _require_manifest()
    result = _cached(
        f"championship_{championship}_{season}_{n_trials}",
        lambda: championship_projection.project_championship(season=season, n_trials=n_trials),
        ttl=_LIVE_CACHE_TTL_SECONDS,
    )
    standings = [ChampionshipEntry(**row) for row in result[championship].to_dict(orient="records")]
    return ChampionshipResponse(
        season=result["season"], as_of_round=result["as_of_round"], n_trials=result["n_trials"], standings=standings
    )


@router.get("/track-record", response_model=TrackRecordResponse)
def get_track_record(tier: str | None = None) -> TrackRecordResponse:
    result = store.get_track_record(tier=tier)
    return TrackRecordResponse(
        n_resolved=result["n_resolved"], by_market=[TrackRecordEntry(**row) for row in result["by_market"]]
    )


@router.get("/track-record/by-race", response_model=list[RaceAccuracyEntry])
def get_race_accuracy(tier: str | None = None) -> list[RaceAccuracyEntry]:
    return [RaceAccuracyEntry(**row) for row in store.get_race_accuracy(tier=tier)]


@router.get("/live/current")
def get_live_current() -> dict:
    """Requirement 3: served from models/live_poller.py's in-process
    cache, updated by the background poller started in api/main.py's
    lifespan hook every ~18s while a Race session is actually live."""
    from ..models import live_poller

    return live_poller.get_cached()


@router.post("/retrain", dependencies=[Depends(_admin_only)])
def retrain() -> dict:
    manifest = manifest_module.train_all()
    _cache.clear()
    return {
        "status": "ok",
        "trained_at": manifest["trained_at"],
        "race_outcome_candidate": manifest["race_outcome_candidate"],
        "race_outcome_candidate_metrics": manifest["race_outcome_candidate_metrics"],
    }
