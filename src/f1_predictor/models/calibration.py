"""calibration.py -- race-level temperature scaling for the three F1 markets.

The Monte Carlo simulator (race_outcome.simulate_race) draws a finishing
order from a Plackett-Luce model over per-driver strengths, then reports the
marginal probabilities of each market. A walk-forward audit of the
2022/2023/2024 hold-outs found those marginals systematically FLATTER than
they should be: the reliability slope of actual-vs-predicted was 2.0-3.1 with
a 95% CI excluding 1.0 on every hold-out and every market (win, podium,
points-finish). Slope > 1 means "actual outcomes vary more across drivers
than the probabilities say they do" -- the classic under-confident shape.

Temperature scaling is the right tool for that: it sharpens the whole
distribution per race without touching the per-driver ordering, so it cannot
invert a single driver's ranking the way a per-row recalibrator can. For a
race with driver probabilities p_i and temperature T:

    p_i' = p_i ** T / sum_j(p_j ** T)

For win the race sums to 1 by construction. For podium and points-finish the
markets are "at most 3" and "at most 10" events, so the analogous
normalisation rescales the race to sum to 3 and 10 respectively, with each
individual probability capped at 1 (a driver cannot be more than certain to
finish in the top 3).

One temperature per market, fitted by minimising log-loss on PRIOR seasons
only -- never on the season being predicted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The three markets temperature scaling is fitted for, and the value each
#: race's probabilities must sum to (1 winner, at most 3 on the podium, at
#: most 10 scoring points).
MARKET_SUMS: dict[str, float] = {"win": 1.0, "podium": 3.0, "points": 10.0}

#: Column names in a per-driver prediction frame.
_MARKET_COLUMNS = {"win": "p_win", "podium": "p_podium", "points": "p_points_finish"}


def _market_column(market: str) -> str:
    try:
        return _MARKET_COLUMNS[market]
    except KeyError:
        raise ValueError(f"unknown market {market!r}; expected one of {sorted(MARKET_COLUMNS)}") from None


def sharpen(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """p_i ** T, normalised. Raises on non-positive probabilities."""
    p = np.asarray(probabilities, dtype=float)
    if np.any(p <= 0):
        # A zero would make p**T zero for any T>0, collapsing the race sum.
        # Floor it rather than propagate a NaN through the renormalisation.
        p = np.clip(p, 1e-12, None)
    return np.power(p, float(temperature))


def capped_proportional_allocate(weights: np.ndarray, target: float) -> np.ndarray:
    """Allocate `target` of probability mass across `weights`, with no entry
    exceeding 1.

    Plain proportional allocation (w_i * target / sum(w)) can push a favourite
    above 1 when the target is larger than 1 — a driver cannot be more than
    certain to make the podium. Capping after the fact would leave the race
    summing below the target, which breaks the "at most 3 on the podium,
    at most 10 scoring" contract the markets rely on.

    So this solves the capped allocation directly: find the scale c such that
    sum(min(1, c * w_i)) == target (water-filling). When there are enough
    drivers, both constraints hold exactly: every entry <= 1 AND the race
    sums to the target. If there are fewer drivers than the target allows
    (a 2-driver race cannot sum to 3 with each entry <= 1), the maximum
    achievable sum is returned instead.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    if n == 0:
        return w
    total_w = w.sum()
    if total_w <= 0:
        return np.full(n, target / n)

    # Start with an unconstrained proportional allocation and cap the
    # offenders, redistributing the freed mass among the rest. Repeat until
    # nothing is capped (or the target is unreachable).
    scale = target / total_w
    active = np.ones(n, dtype=bool)
    allocation = np.zeros(n, dtype=float)
    remaining = target
    for _ in range(n):
        active_weights = w[active]
        if active_weights.sum() <= 0:
            break
        candidate = remaining * active_weights / active_weights.sum()
        over = candidate > 1.0
        if not over.any():
            allocation[active] = candidate
            break
        # fix the capped entries at 1 and redistribute what is left
        capped_idx = np.where(active)[0][over]
        allocation[capped_idx] = 1.0
        remaining -= len(capped_idx)
        active[capped_idx] = False
    return np.minimum(allocation, 1.0)


def apply_temperature_to_race(
    probabilities: np.ndarray, temperature: float, race_sum: float = 1.0
) -> np.ndarray:
    """Sharpen one race's probabilities and rescale so they sum to `race_sum`.

    Sharpening is p_i ** T (a per-race temperature; T > 1 sharpens, T < 1
    flattens). The sharpened weights are then allocated so the race sums to
    `race_sum` with no individual probability above 1 — a driver cannot be
    more than certain to win, make the podium, or score.
    """
    sharpened = sharpen(probabilities, temperature)
    return capped_proportional_allocate(sharpened, race_sum)


def apply_temperatures(
    predictions: pd.DataFrame, temperatures: dict[str, float], race_key: str = "round"
) -> pd.DataFrame:
    """Apply per-market temperature scaling to a per-driver prediction frame.

    Adds `raw_p_win` / `raw_p_podium` / `raw_p_points_finish` columns holding
    the pre-recalibration probabilities, and overwrites `p_win` / `p_podium` /
    `p_points_finish` with the corrected values.

    `race_key` groups rows into races; rows sharing a key value are rescaled
    together. If the column is absent the whole frame is treated as ONE race
    (the serving path simulates one race at a time, so that frame is one race).
    """
    out = predictions.copy()
    if out.empty:
        return out

    grouped = race_key in out.columns

    for market, column in _MARKET_COLUMNS.items():
        if column not in out.columns:
            continue
        raw_col = f"raw_{column}"
        out[raw_col] = out[column]
        temperature = float(temperatures.get(market, 1.0))
        race_sum = MARKET_SUMS[market]
        if grouped:
            out[column] = out.groupby(race_key, sort=False)[column].transform(
                lambda values, _t=temperature, _s=race_sum: apply_temperature_to_race(
                    values.to_numpy(dtype=float), _t, _s
                )
            )
        else:
            out[column] = apply_temperature_to_race(
                out[column].to_numpy(dtype=float), temperature, race_sum
            )
    return out


#: Markets whose temperature scaling is approved to be SERVED.
#:
#: This is a review decision, not a training output: a temperature being
#: fitted says nothing about whether it should change what a reader sees.
#: The ship test in docs/f1_temperature_scaling_ship_test.md decides it —
#: a market is added here only if it passed on ALL THREE expanding-window
#: hold-outs (2022, 2023, 2024) on all three ship tests.
#:
#: It is currently EMPTY: the ship test passed podium on hold-out 2024 and
#: points-finish on hold-out 2023, but no market passed on all three
#: hold-outs, so nothing is served. Fitting still happens (the manifest
#: records the fitted temperatures for auditability); serving is a no-op
#: until a market is approved here.
APPROVED_MARKETS: tuple[str, ...] = ()


def approved_temperatures(manifest: dict) -> dict[str, float]:
    """The temperatures that may be served: fitted AND approved.

    The manifest records what was fitted; `APPROVED_MARKETS` records what
    the ship test cleared. A market is served only if it is in both.
    """
    block = manifest.get("calibration") or {}
    fitted = block.get("temperatures") or {}
    return {
        market: float(fitted[market])
        for market in APPROVED_MARKETS
        if market in fitted
    }


def _log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(
    race_frames: list[pd.DataFrame],
    market: str,
    probability_column: str,
    outcome_column: str,
    race_key: str = "round",
    grid: np.ndarray | None = None,
) -> float:
    """Fit one temperature for one market by minimising log-loss.

    `race_frames` is a list of per-race frames (each a group of drivers in
    one race). The temperature is searched over `grid` (default: a dense
    range around 1) and the one with the lowest pooled log-loss wins. The
    search is deterministic and has no data leakage: the caller supplies
    only PRIOR seasons' races.
    """
    if grid is None:
        # The audit found reliability slopes of 2.0-3.1, i.e. probabilities
        # far too flat, so the useful range extends well above 1. Search up
        # to 10 so the optimum is not pinned to the grid edge.
        grid = np.arange(0.25, 10.01, 0.05)
    if not race_frames:
        return 1.0

    best_t, best_ll = 1.0, float("inf")
    for temperature in grid:
        total_ll, n = 0.0, 0
        for frame in race_frames:
            if probability_column not in frame.columns:
                continue
            probs = apply_temperature_to_race(
                frame[probability_column].to_numpy(dtype=float),
                float(temperature),
                MARKET_SUMS[market],
            )
            y = frame[outcome_column].to_numpy(dtype=float)
            total_ll += _log_loss(y, probs) * len(y)
            n += len(y)
        if n == 0:
            continue
        ll = total_ll / n
        if ll < best_ll:
            best_ll, best_t = ll, float(temperature)
    return best_t


def fit_temperatures(
    race_frames_by_market: dict[str, list[pd.DataFrame]],
    race_key: str = "round",
) -> dict[str, float]:
    """Fit one temperature per market. `race_frames_by_market` maps a market
    name to the list of per-race frames for that market (prior seasons only)."""
    temperatures: dict[str, float] = {}
    for market, frames in race_frames_by_market.items():
        column = _market_column(market)
        outcome = {"win": "actual_win", "podium": "actual_pod", "points": "actual_pts"}[market]
        temperatures[market] = fit_temperature(frames, market, column, outcome, race_key=race_key)
    return temperatures


def fit_temperatures_out_of_sample(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = 5,
    seed: int = 1,
) -> dict[str, float]:
    """Fit the per-market temperatures on OUT-OF-SAMPLE predictions of the
    training seasons, by race-level K-fold.

    Fitting on in-sample predictions is wrong here: a ranker scored on the
    rows it was trained on is over-confident, so the temperature that
    "corrects" it comes out far too large (T=5-8 in the F1 audit, against an
    out-of-sample optimum of T=2-3) and then over-sharpens whatever it is
    applied to. Race-level folds keep a whole race out of training, because
    drivers in one race are not independent of each other.
    """
    from ..features import session_state
    from . import race_outcome

    frame = train_df[train_df["tier"] == session_state.TIER_POST_QUALIFYING].reset_index(drop=True)
    races: list[pd.DataFrame] = []
    for season, round_, race in frame.groupby(["season", "round"], sort=True):
        race = race.copy()
        ranker = race_outcome.train_ranker(frame, feature_cols)
        theta = race_outcome.theta_from_xgb_scores(
            race_outcome.xgb_scores_for_race(ranker, race, feature_cols)
        )
        sim = race_outcome.simulate_race(
            theta, dnf_prob=None, n_trials=3000, seed=42
        ).set_index("driver_id")
        winner_rows = race.loc[race["position"] == 1, "driver_id"]
        winner = winner_rows.iloc[0] if not winner_rows.empty else None
        podium = race.loc[race["position"] <= 3, "driver_id"].values
        points_finish = race.loc[race["position"] <= 10, "driver_id"].values
        race["p_win"] = race["driver_id"].map(sim["p_win"])
        race["p_podium"] = race["driver_id"].map(sim["p_podium"])
        race["p_points_finish"] = race["driver_id"].map(sim["p_points_finish"])
        race["actual_win"] = (race["driver_id"] == winner).astype(float)
        race["actual_pod"] = race["driver_id"].isin(podium).astype(float)
        race["actual_pts"] = race["driver_id"].isin(points_finish).astype(float)
        race["race_id"] = f"{season}-{round_}"
        races.append(race)

    if len(races) < n_folds:
        return {market: 1.0 for market in MARKET_SUMS}

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(races))
    folds = np.array_split(order, n_folds)
    oof: list[pd.DataFrame] = []
    for fold in folds:
        held = {races[i]["race_id"].iloc[0] for i in fold}
        fit_races = [r for r in races if r["race_id"].iloc[0] not in held]
        fit_df = pd.concat(fit_races, ignore_index=True).drop(
            columns=["p_win", "p_podium", "p_points_finish",
                     "actual_win", "actual_pod", "actual_pts", "race_id"],
            errors="ignore",
        )
        ranker = race_outcome.train_ranker(fit_df, feature_cols)
        for i in fold:
            race = races[i]
            theta = race_outcome.theta_from_xgb_scores(
                race_outcome.xgb_scores_for_race(ranker, race, feature_cols)
            )
            sim = race_outcome.simulate_race(
                theta, dnf_prob=None, n_trials=3000, seed=42
            ).set_index("driver_id")
            winner_rows = race.loc[race["position"] == 1, "driver_id"]
            winner = winner_rows.iloc[0] if not winner_rows.empty else None
            podium = race.loc[race["position"] <= 3, "driver_id"].values
            points_finish = race.loc[race["position"] <= 10, "driver_id"].values
            frame_race = race.copy()
            frame_race["p_win"] = frame_race["driver_id"].map(sim["p_win"])
            frame_race["p_podium"] = frame_race["driver_id"].map(sim["p_podium"])
            frame_race["p_points_finish"] = frame_race["driver_id"].map(sim["p_points_finish"])
            frame_race["actual_win"] = (frame_race["driver_id"] == winner).astype(float)
            frame_race["actual_pod"] = frame_race["driver_id"].isin(podium).astype(float)
            frame_race["actual_pts"] = frame_race["driver_id"].isin(points_finish).astype(float)
            oof.append(frame_race)

    return fit_temperatures({market: oof for market in MARKET_SUMS}, race_key="race_id")
