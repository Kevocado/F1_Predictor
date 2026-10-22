"""audit_bootstrap.py -- race-level paired bootstrap for calibration audits.

The audit's whole credibility rests on these intervals, so the resampling is
done in one tested place rather than copy-pasted into each audit script.

Two things this module is careful about, both of which were got wrong in an
earlier draft of the audit scripts:

1. The generator is created ONCE, outside the resampling loop. Creating a
   fresh `np.random.default_rng(seed)` per iteration makes every iteration
   draw the SAME resample, which collapses the interval to a single point
   (lower == upper) and can even exclude the point estimate.
2. The statistic is recomputed on the POOLED resample, and the interval is
   the 2.5 / 97.5 percentiles of the resampled statistics. A percentile
   interval from a correct resample brackets the point estimate with
   overwhelming probability; a degenerate one cannot.

Resampling is at the RACE level: a race is the unit of independent
information (drivers in one race are not independent of each other), so whole
races are drawn with replacement and their drivers pooled.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


def bootstrap_race(
    races: Sequence[pd.DataFrame],
    statistic: Callable[[pd.DataFrame], float],
    n_resamples: int = 2000,
    seed: int = 1,
) -> tuple[float, float]:
    """95% percentile interval of `statistic` over race-level resamples.

    `races` is a sequence of per-race DataFrames. Each resample draws
    `len(races)` races with replacement, concatenates their rows, and
    evaluates `statistic` on the pooled frame.

    Returns (lower, upper) at the 2.5 and 97.5 percentiles.
    """
    if not races:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_races = len(races)
    values = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n_races, n_races)
        pooled = pd.concat([races[j] for j in idx], ignore_index=True)
        values[i] = statistic(pooled)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_paired_delta(
    races: Sequence[pd.DataFrame],
    loss_a: Callable[[pd.DataFrame], np.ndarray],
    loss_b: Callable[[pd.DataFrame], np.ndarray],
    n_resamples: int = 2000,
    seed: int = 1,
) -> tuple[float, float, float]:
    """95% CI of the mean per-row difference (loss_a - loss_b).

    The SAME resampled races are used for both losses, so the comparison is
    paired: a game where the corrected model did badly is compared against
    the same game's raw-model loss, not a different set of games.

    Returns (mean_delta, lower, upper).
    """
    if not races:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_races = len(races)
    deltas = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n_races, n_races)
        pooled = pd.concat([races[j] for j in idx], ignore_index=True)
        a = np.asarray(loss_a(pooled), dtype=float)
        b = np.asarray(loss_b(pooled), dtype=float)
        deltas[i] = float(np.mean(a - b))
    return (
        float(np.mean(deltas)),
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )


def reliability_slope(actual: np.ndarray, predicted: np.ndarray) -> float:
    """OLS slope of actual on predicted. 1.0 = perfectly calibrated."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    xc = predicted - predicted.mean()
    yc = actual - actual.mean()
    denom = float((xc * xc).sum())
    if denom == 0:
        return float("nan")
    return float((xc * yc).sum() / denom)
