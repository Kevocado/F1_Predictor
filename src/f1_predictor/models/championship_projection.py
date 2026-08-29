"""championship_projection.py — Monte Carlo season simulator for
requirement 2: projects final Drivers'/Constructors' championship standings
by simulating every remaining race of a season many times, reusing
models/race_outcome.py's Plackett-Luce draw per race and awarding real F1
points on top of the current locked-in standings.

Deliberate scope boundaries (stated once here rather than re-litigated
elsewhere):
  - Driver/team strength (Elo, team strength, rolling form) is held at its
    CURRENT snapshot for every remaining race in every trial — only the
    stochastic Plackett-Luce draw varies race to race and trial to trial.
    Dynamically evolving form/Elo from each trial's own simulated results
    would be more "realistic" but multiplies simulation cost and adds a
    second layer of compounding uncertainty on top of the outcome draw
    itself; holding it static is the standard simplification season
    simulators of this kind make.
  - Constructor pairing (which team a driver is on) for remaining races is
    assumed to hold at its current state — no speculative future
    driver-swap modeling.
  - Circuit-specific features (driver/constructor circuit history, circuit
    DNF rate) DO vary per remaining race, since those are known facts about
    the calendar, not simulated outcomes. Weather is left unknown (NaN)
    for future races beyond Open-Meteo's ~16-day forecast horizon.
  - No fastest-lap bonus point — worth at most 1 point and only when
    finishing top 10, a marginal effect not worth the extra "who sets
    fastest lap" modeling layer it would need.
  - A sprint weekend is simulated as two independent Plackett-Luce draws
    (sprint + main race) sharing the same strength/DNF vectors for that
    weekend, each awarding its own points system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..data import entities, jolpica
from ..features import elo, team_strength
from ..features.build import FEATURE_COLUMNS
from ..features.circuit import with_circuit
from . import dnf as dnf_model
from . import manifest as manifest_module
from . import race_outcome

SPRINT_POINTS_TABLE = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


def _current_form_snapshot(results_df: pd.DataFrame, windows: tuple[int, ...] = (3, 5, 10)) -> dict[str, dict]:
    """Each driver's rolling form entering their NEXT (not-yet-happened)
    race — the last `w` races they've actually run, unshifted, since this
    is a genuinely future row rather than a historical training row."""
    df = results_df.sort_values(["driver_id", "season", "round"]).copy()
    df["finish_position"] = df["position"].fillna(20)
    out: dict[str, dict] = {}
    for driver_id, g in df.groupby("driver_id"):
        row = {}
        for w in windows:
            tail = g.tail(w)
            row[f"form_avg_position_{w}"] = tail["finish_position"].mean()
            row[f"form_avg_points_{w}"] = tail["points"].mean()
            row[f"form_dnf_rate_{w}"] = tail["dnf"].mean()
        out[driver_id] = row
    return out


def build_future_feature_rows(
    season: int,
    remaining_schedule: pd.DataFrame,
    results_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    driver_ids: list[str],
) -> pd.DataFrame:
    """One feature row per (remaining round, driver): current elo/team-
    strength/rolling-form snapshots (module docstring), but per-race
    circuit features specific to that remaining round. Session-state-tier
    columns (grid/quali/weather) are NaN — these races haven't happened."""
    elo_ratings = elo.latest_ratings(results_df)
    team_ratings = team_strength.latest_ratings(results_df)
    form_snap = _current_form_snapshot(results_df)
    team_form_snap = team_strength.current_team_form_snapshot(results_df)

    hist = with_circuit(results_df, schedule_df).copy()
    hist["finish_position"] = hist["position"].fillna(20)
    driver_circuit_avg_pos = hist.groupby(["driver_id", "circuit_id"])["finish_position"].mean()
    driver_circuit_avg_pts = hist.groupby(["driver_id", "circuit_id"])["points"].mean()
    constructor_circuit_avg_pos = hist.groupby(["constructor_id", "circuit_id"])["finish_position"].mean()
    circuit_dnf = hist.groupby("circuit_id")["dnf"].mean()

    latest_constructor = (
        entities.driver_constructor_map(results_df).sort_values(["season", "round"]).groupby("driver_id").tail(1)
    ).set_index("driver_id")["constructor_id"]

    rows = []
    for _, race in remaining_schedule.iterrows():
        cid = race["circuit_id"]
        for d in driver_ids:
            if d not in latest_constructor.index:
                continue
            constructor_id = latest_constructor.loc[d]
            row = {
                "season": season,
                "round": int(race["round"]),
                "driver_id": d,
                "constructor_id": constructor_id,
                "elo_pre_race": elo_ratings.get(d, elo.INITIAL_ELO),
                "team_strength_pre_race": team_ratings.get(constructor_id, elo.INITIAL_ELO),
                "driver_circuit_avg_position": driver_circuit_avg_pos.get((d, cid), float("nan")),
                "driver_circuit_avg_points": driver_circuit_avg_pts.get((d, cid), float("nan")),
                "constructor_circuit_avg_position": constructor_circuit_avg_pos.get((constructor_id, cid), float("nan")),
                "circuit_dnf_rate": circuit_dnf.get(cid, float("nan")),
                "grid": float("nan"),
                "quali_position": float("nan"),
                "quali_gap_to_pole": float("nan"),
                "temp_max_c": float("nan"),
                "precipitation_mm": float("nan"),
                "wind_max_kph": float("nan"),
            }
            row.update(form_snap.get(d, {}))
            row.update(team_form_snap.get(constructor_id, {}))
            rows.append(row)

    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df


def _summarize(points_by_entity: dict[str, list[float]]) -> pd.DataFrame:
    entity_ids = list(points_by_entity.keys())
    matrix = np.array([points_by_entity[e] for e in entity_ids])  # (n_entities, n_trials)
    ranks = (-matrix).argsort(axis=0).argsort(axis=0) + 1  # rank 1 = most points, per trial

    rows = [
        {
            "entity_id": e,
            "win_prob": float((ranks[i] == 1).mean()),
            "top3_prob": float((ranks[i] <= 3).mean()),
            "expected_final_points": float(matrix[i].mean()),
            "expected_final_rank": float(ranks[i].mean()),
        }
        for i, e in enumerate(entity_ids)
    ]
    return pd.DataFrame(rows).sort_values("win_prob", ascending=False).reset_index(drop=True)


def simulate_season(
    season: int,
    candidate: str,
    ranker: xgb.XGBRanker | None,
    dnf_clf: xgb.XGBClassifier,
    results_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    n_trials: int = 2000,
    seed: int = 0,
) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    remaining = schedule_df[
        (schedule_df["season"] == season) & schedule_df["race_datetime"].notna() & (schedule_df["race_datetime"] > now)
    ].sort_values("round")
    driver_ids = driver_standings["driver_id"].tolist()

    latest_constructor = (
        entities.driver_constructor_map(results_df).sort_values(["season", "round"]).groupby("driver_id").tail(1)
    ).set_index("driver_id")["constructor_id"]

    # Per-race theta/DNF-prob computed ONCE (held static across trials —
    # see module docstring); each trial only re-draws the stochastic order.
    race_thetas: dict[int, dict[str, float]] = {}
    race_dnf: dict[int, dict[str, float]] = {}

    if candidate == "elo":
        elo_ratings = elo.latest_ratings(results_df)
        team_ratings = team_strength.latest_ratings(results_df)
        combined = {
            d: elo_ratings.get(d, elo.INITIAL_ELO) + team_ratings.get(latest_constructor.get(d), elo.INITIAL_ELO)
            for d in driver_ids
        }
        static_theta = race_outcome.theta_from_elo_strength(combined)
        for round_ in remaining["round"]:
            race_thetas[round_] = static_theta
        future_df = build_future_feature_rows(season, remaining, results_df, schedule_df, driver_ids)
        for round_ in remaining["round"]:
            race_df = future_df[future_df["round"] == round_]
            race_dnf[round_] = dnf_model.predict_dnf_prob(dnf_clf, race_df, FEATURE_COLUMNS)
    else:
        future_df = build_future_feature_rows(season, remaining, results_df, schedule_df, driver_ids)
        for round_ in remaining["round"]:
            race_df = future_df[future_df["round"] == round_]
            scores = race_outcome.xgb_scores_for_race(ranker, race_df, FEATURE_COLUMNS)
            race_thetas[round_] = race_outcome.theta_from_xgb_scores(scores)
            race_dnf[round_] = dnf_model.predict_dnf_prob(dnf_clf, race_df, FEATURE_COLUMNS)

    is_sprint = dict(zip(remaining["round"], remaining["is_sprint_weekend"]))

    rng = np.random.default_rng(seed)
    driver_points_start = dict(zip(driver_standings["driver_id"], driver_standings["points"]))
    constructor_points_start = dict(zip(constructor_standings["constructor_id"], constructor_standings["points"]))
    constructor_ids = sorted(set(constructor_points_start.keys()) | set(latest_constructor.reindex(driver_ids).dropna()))

    driver_final_points: dict[str, list[float]] = {d: [] for d in driver_ids}
    constructor_final_points: dict[str, list[float]] = {c: [] for c in constructor_ids}

    def _award(order: list[str], points_table: dict[int, int], d_points: dict, c_points: dict) -> None:
        for pos, d in enumerate(order, start=1):
            pts = points_table.get(pos, 0)
            if not pts:
                continue
            d_points[d] = d_points.get(d, 0) + pts
            c = latest_constructor.get(d)
            if c is not None:
                c_points[c] = c_points.get(c, 0) + pts

    for _ in range(n_trials):
        d_points = dict(driver_points_start)
        c_points = dict(constructor_points_start)
        for round_ in remaining["round"]:
            theta, dnf_prob = race_thetas[round_], race_dnf[round_]
            if is_sprint.get(round_):
                sprint_order, _ = race_outcome.draw_classification(theta, dnf_prob, rng)
                _award(sprint_order, SPRINT_POINTS_TABLE, d_points, c_points)
            order, _ = race_outcome.draw_classification(theta, dnf_prob, rng)
            _award(order, race_outcome.POINTS_TABLE, d_points, c_points)

        for d in driver_ids:
            driver_final_points[d].append(d_points.get(d, 0))
        for c in constructor_ids:
            constructor_final_points[c].append(c_points.get(c, 0))

    return {
        "season": season,
        "n_trials": n_trials,
        "n_remaining_races": int(len(remaining)),
        "drivers": _summarize(driver_final_points),
        "constructors": _summarize(constructor_final_points),
    }


def project_championship(season: int | None = None, n_trials: int = 2000, seed: int = 0) -> dict:
    """End-to-end: loads the current manifest's chosen race-outcome
    candidate + DNF model, fetches the season's current results/schedule/
    standings, and runs `simulate_season`."""
    from ..config import CURRENT_SEASON

    season = season or CURRENT_SEASON

    results_df = jolpica.load_season_results(season)
    schedule_df = jolpica.fetch_season_schedule(season)
    driver_standings = jolpica.fetch_driver_standings(season)
    constructor_standings = jolpica.fetch_constructor_standings(season)

    manifest = manifest_module.load_manifest()
    candidate = manifest["race_outcome_candidate"]

    ranker = None
    if candidate == "xgb_ranker":
        ranker = xgb.XGBRanker()
        ranker.load_model(str(manifest_module.RACE_OUTCOME_MODEL_PATH))

    dnf_clf = xgb.XGBClassifier()
    dnf_clf.load_model(str(manifest_module.DNF_MODEL_PATH))

    result = simulate_season(
        season, candidate, ranker, dnf_clf, results_df, schedule_df, driver_standings, constructor_standings,
        n_trials=n_trials, seed=seed,
    )
    result["as_of_round"] = int(results_df["round"].max()) if not results_df.empty else 0
    return result
