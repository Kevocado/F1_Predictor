"""Hand-computed checks for the per-tier metric functions in scripts/tier_metrics.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import tier_metrics  # noqa: E402


def _race(pos, p_win=None, score=None, p_top=None):
    d = {"driver_id": [f"d{i}" for i in range(len(pos))], "position": pos}
    if p_win is not None:
        d["p_win"] = p_win
    if score is not None:
        d["score"] = score
    if p_top is not None:
        d["p_top3"] = p_top
    return pd.DataFrame(d)


def test_win_logloss_multiclass_is_minus_log_prob_of_winner():
    race = _race([2, 1, 3], p_win=[0.2, 0.5, 0.3])
    assert tier_metrics.win_logloss_multiclass(race, "p_win") == pytest.approx(-np.log(0.5))


def test_win_logloss_binary_averages_yes_no_scores_per_driver():
    # two drivers, p=0.5 each, one wins: every row contributes ln 2
    race = _race([1, 2], p_win=[0.5, 0.5])
    assert tier_metrics.win_logloss_binary(race, "p_win") == pytest.approx(np.log(2))


def test_spearman_order_is_one_when_score_matches_finish_and_minus_one_when_reversed():
    aligned = _race([1, 2, 3, 4], score=[4.0, 3.0, 2.0, 1.0])
    reversed_ = _race([1, 2, 3, 4], score=[1.0, 2.0, 3.0, 4.0])
    assert tier_metrics.spearman_order(aligned, "score") == pytest.approx(1.0)
    assert tier_metrics.spearman_order(reversed_, "score") == pytest.approx(-1.0)


def test_brier_topk_matches_hand_computed_value():
    # p=0.8 for a podium finisher, p=0.2 for a non-podium: ((0.8-1)^2 + (0.2-0)^2)/2 = 0.04
    race = _race([1, 5], p_top=[0.8, 0.2])
    assert tier_metrics.brier_topk(race, "p_top3", 3) == pytest.approx(0.04)
