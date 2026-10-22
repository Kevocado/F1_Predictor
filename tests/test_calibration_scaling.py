"""Tests for the audit bootstrap and the race-level temperature scaling.

The bootstrap test uses a synthetic case with a KNOWN reliability slope, so
a broken resampler (the earlier per-iteration re-seed, which collapsed every
interval to a point) cannot pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_predictor.evaluate.audit_bootstrap import (
    bootstrap_paired_delta,
    bootstrap_race,
    reliability_slope,
)
from f1_predictor.models import calibration


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------


def _synthetic_races(n_races: int = 40, drivers: int = 8, slope: float = 1.0, seed: int = 0):
    """Races whose actual win rate is exactly `slope` times the predicted rate.

    With slope=1.0 the true calibration slope is 1.0; with slope=2.0 the
    model is under-confident by exactly a factor of two.
    """
    rng = np.random.default_rng(seed)
    races = []
    for r in range(n_races):
        predicted = rng.dirichlet(np.ones(drivers))
        # actual ~ Binomial(n, slope * predicted), so E[actual] = slope * predicted
        actual = rng.binomial(1, np.clip(slope * predicted, 0, 1)).astype(float)
        races.append(pd.DataFrame({"predicted": predicted, "actual": actual, "race": r}))
    return races


def test_reliability_slope_of_synthetic_data_is_known_value():
    """The synthetic generator really does produce the slope it claims."""
    races = _synthetic_races(slope=1.0, seed=1)
    pooled = pd.concat(races, ignore_index=True)
    assert reliability_slope(pooled["actual"], pooled["predicted"]) == pytest.approx(1.0, abs=0.15)


def test_bootstrap_interval_brackets_the_point_estimate():
    """A correct percentile bootstrap brackets the estimate; the earlier
    broken resampler (re-seeded RNG inside the loop) collapsed every interval
    to a point and could exclude the estimate entirely."""
    races = _synthetic_races(slope=1.0, seed=2)
    pooled = pd.concat(races, ignore_index=True)
    point = reliability_slope(pooled["actual"], pooled["predicted"])
    lo, hi = bootstrap_race(
        races, lambda df: reliability_slope(df["actual"], df["predicted"]), n_resamples=500
    )
    assert lo < hi, "degenerate interval: lower == upper means the resampler is broken"
    assert lo <= point <= hi, f"interval [{lo}, {hi}] does not bracket the estimate {point}"


def test_bootstrap_interval_is_not_a_point():
    """The specific symptom of the re-seeded RNG: every resample identical."""
    races = _synthetic_races(slope=2.0, seed=3)
    lo, hi = bootstrap_race(
        races, lambda df: reliability_slope(df["actual"], df["predicted"]), n_resamples=200
    )
    assert hi - lo > 1e-9, "interval collapsed to a point — resamples are identical"


def test_bootstrap_resamples_differ_between_draws():
    """Two calls with different seeds must give different intervals."""
    races = _synthetic_races(slope=1.5, seed=4)
    stat = lambda df: reliability_slope(df["actual"], df["predicted"])  # noqa: E731
    a = bootstrap_race(races, stat, n_resamples=200, seed=1)
    b = bootstrap_race(races, stat, n_resamples=200, seed=2)
    assert a != b


def test_bootstrap_detects_a_real_slope_departure():
    """A synthetic slope of 2.0 must produce an interval that excludes 1.0."""
    races = _synthetic_races(n_races=60, drivers=10, slope=2.0, seed=5)
    lo, hi = bootstrap_race(
        races, lambda df: reliability_slope(df["actual"], df["predicted"]), n_resamples=500
    )
    assert lo > 1.0, f"expected the interval to exclude 1.0, got [{lo}, {hi}]"


def test_bootstrap_paired_delta_detects_an_improvement():
    """loss_a strictly better than loss_b on every race -> delta CI below zero."""
    races = _synthetic_races(n_races=40, drivers=8, slope=1.0, seed=6)
    loss_a = lambda df: (df["predicted"] - df["actual"]) ** 2  # noqa: E731
    loss_b = lambda df: df["actual"] ** 2 + 1.0  # noqa: E731
    mean, lo, hi = bootstrap_paired_delta(races, loss_a, loss_b, n_resamples=300)
    assert lo < 0, f"expected a negative delta CI, got [{lo}, {hi}]"
    assert lo < mean < hi


def test_bootstrap_paired_delta_is_zero_when_losses_match():
    races = _synthetic_races(n_races=20, drivers=6, slope=1.0, seed=7)
    loss = lambda df: (df["predicted"] - df["actual"]) ** 2  # noqa: E731
    mean, lo, hi = bootstrap_paired_delta(races, loss, loss, n_resamples=200)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# temperature scaling
# --------------------------------------------------------------------------


def test_temperature_one_is_identity():
    probs = np.array([0.5, 0.3, 0.2])
    out = calibration.apply_temperature_to_race(probs, 1.0, race_sum=1.0)
    assert out == pytest.approx(probs)


def test_temperature_above_one_sharpens_and_preserves_sum():
    probs = np.array([0.5, 0.3, 0.2])
    out = calibration.apply_temperature_to_race(probs, 2.0, race_sum=1.0)
    assert out.sum() == pytest.approx(1.0)
    # sharper: the leader's share goes up, the tail's goes down
    assert out[0] > probs[0]
    assert out[2] < probs[2]
    # order preserved
    assert out[0] > out[1] > out[2]


def test_temperature_below_one_flattens():
    probs = np.array([0.5, 0.3, 0.2])
    out = calibration.apply_temperature_to_race(probs, 0.5, race_sum=1.0)
    assert out.sum() == pytest.approx(1.0)
    assert out[0] < probs[0]
    assert out[2] > probs[2]


def test_podium_and_points_rescale_to_three_and_ten():
    # A realistic 20-driver grid: the podium (3) and points (10) targets are
    # both reachable with every entry capped at 1.
    probs = np.linspace(0.09, 0.01, 20)
    probs = probs / probs.sum()
    podium = calibration.apply_temperature_to_race(probs, 1.5, race_sum=3.0)
    assert podium.sum() == pytest.approx(3.0)
    assert (podium <= 1.0 + 1e-12).all()
    points = calibration.apply_temperature_to_race(probs, 1.5, race_sum=10.0)
    assert points.sum() == pytest.approx(10.0)
    assert (points <= 1.0 + 1e-12).all()


def test_capped_allocation_respects_both_constraints():
    weights = np.array([0.9, 0.05, 0.05])
    out = calibration.capped_proportional_allocate(weights, 3.0)
    assert out.sum() == pytest.approx(3.0)
    assert (out <= 1.0 + 1e-12).all()


def test_capped_allocation_returns_max_reachable_when_target_unreachable():
    """Two drivers cannot sum to 3 with each entry <= 1."""
    weights = np.array([0.6, 0.4])
    out = calibration.capped_proportional_allocate(weights, 3.0)
    assert out.sum() == pytest.approx(2.0)
    assert (out <= 1.0 + 1e-12).all()


def test_no_probability_exceeds_one_after_rescaling():
    probs = np.array([0.9, 0.05, 0.05])
    out = calibration.apply_temperature_to_race(probs, 3.0, race_sum=3.0)
    assert (out <= 1.0 + 1e-12).all()


def test_apply_temperatures_stores_raw_probabilities():
    n = 20
    frame = pd.DataFrame(
        {
            "round": [1] * n + [2] * n,
            "p_win": list(np.linspace(0.08, 0.02, n)) + list(np.linspace(0.09, 0.01, n)),
            "p_podium": list(np.linspace(0.30, 0.05, n)) + list(np.linspace(0.28, 0.04, n)),
            "p_points_finish": list(np.linspace(0.60, 0.20, n)) + list(np.linspace(0.55, 0.18, n)),
        }
    )
    out = calibration.apply_temperatures(frame, {"win": 2.0, "podium": 1.5, "points": 1.2})
    for raw_col in ("raw_p_win", "raw_p_podium", "raw_p_points_finish"):
        assert raw_col in out.columns
    assert out["raw_p_win"].tolist() == frame["p_win"].tolist()
    # each race's corrected win probabilities still sum to 1
    for _, group in out.groupby("round"):
        assert group["p_win"].sum() == pytest.approx(1.0)
    # podium / points races rescale to 3 / 10
    for _, group in out.groupby("round"):
        assert group["p_podium"].sum() == pytest.approx(3.0)
        assert group["p_points_finish"].sum() == pytest.approx(10.0)


def test_fit_temperature_finds_a_sharper_temperature_for_flat_probabilities():
    """When actuals vary more than the probabilities say, the fitted
    temperature must be > 1 (sharpen), which is the direction this whole
    change is about."""
    rng = np.random.default_rng(11)
    frames = []
    for r in range(40):
        predicted = rng.dirichlet(np.ones(8))
        actual = rng.binomial(1, np.clip(2.0 * predicted, 0, 1)).astype(float)
        frames.append(pd.DataFrame({"predicted": predicted, "actual": actual}))
    temperature = calibration.fit_temperature(
        frames, "win", "predicted", "actual", grid=np.arange(0.5, 4.01, 0.25)
    )
    assert temperature > 1.0


def test_fit_temperature_is_one_for_calibrated_probabilities():
    rng = np.random.default_rng(12)
    frames = []
    for r in range(40):
        predicted = rng.dirichlet(np.ones(8))
        actual = rng.binomial(1, predicted).astype(float)
        frames.append(pd.DataFrame({"predicted": predicted, "actual": actual}))
    temperature = calibration.fit_temperature(
        frames, "win", "predicted", "actual", grid=np.arange(0.5, 4.01, 0.25)
    )
    assert temperature == pytest.approx(1.0, abs=0.5)


# --------------------------------------------------------------------------
# ship-test gate: only approved markets are served
# --------------------------------------------------------------------------


def test_approved_temperatures_is_empty_when_nothing_is_approved():
    manifest = {"calibration": {"temperatures": {"win": 2.5, "podium": 1.5}, "approved_markets": []}}
    assert calibration.approved_temperatures(manifest) == {}


def test_approved_temperatures_returns_only_approved_markets(monkeypatch):
    monkeypatch.setattr(calibration, "APPROVED_MARKETS", ("podium",))
    manifest = {"calibration": {"temperatures": {"win": 2.5, "podium": 1.5, "points": 1.2}}}
    assert calibration.approved_temperatures(manifest) == {"podium": 1.5}


def test_approved_temperatures_ignores_a_market_with_no_fitted_temperature(monkeypatch):
    monkeypatch.setattr(calibration, "APPROVED_MARKETS", ("podium", "points"))
    manifest = {"calibration": {"temperatures": {"podium": 1.5}}}
    assert calibration.approved_temperatures(manifest) == {"podium": 1.5}


def test_approved_temperatures_handles_a_manifest_with_no_calibration_block():
    assert calibration.approved_temperatures({}) == {}


def test_serving_applies_identity_when_nothing_is_approved():
    """The shipped state: no market passed the ship test, so a served
    prediction is unchanged — but the raw probability is still recorded."""
    frame = pd.DataFrame(
        {
            "p_win": [0.5, 0.3, 0.2],
            "p_podium": [0.8, 0.6, 0.4],
            "p_points_finish": [0.9, 0.7, 0.3],
        }
    )
    out = calibration.apply_temperatures(frame, calibration.approved_temperatures({}))
    assert out["p_win"].tolist() == pytest.approx([0.5, 0.3, 0.2])
    assert out["raw_p_win"].tolist() == pytest.approx([0.5, 0.3, 0.2])


def test_serving_applies_the_temperature_for_an_approved_market(monkeypatch):
    monkeypatch.setattr(calibration, "APPROVED_MARKETS", ("win",))
    frame = pd.DataFrame({"p_win": [0.5, 0.3, 0.2]})
    out = calibration.apply_temperatures(frame, {"win": 2.0})
    assert out["p_win"].tolist() != pytest.approx([0.5, 0.3, 0.2])
    assert out["raw_p_win"].tolist() == pytest.approx([0.5, 0.3, 0.2])
    assert out["p_win"].sum() == pytest.approx(1.0)
