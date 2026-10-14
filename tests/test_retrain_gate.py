"""retrain_gate: accept a retrained candidate only if it is not worse than the
committed model on the same walk-forward folds. Metric dicts are the real
shape (the committed models/manifest.json), perturbed per case."""

import copy
import json

import pytest

from f1_predictor import retrain_gate as rg
from f1_predictor.models.manifest import MANIFEST_PATH


@pytest.fixture
def current():
    return json.loads(MANIFEST_PATH.read_text())


def _folds(m):
    return m["walk_forward_folds"][m["race_outcome_candidate"]]


def _bump(m, delta, seasons=None):
    m = copy.deepcopy(m)
    for f in _folds(m):
        if seasons is None or f["val_season"] in seasons:
            f["log_loss"] += delta
    return m


def test_identical_candidate_is_accepted(current):
    r = rg.compare(current, copy.deepcopy(current))
    assert r["accept"] and not r["reasons"]


def test_better_candidate_is_accepted(current):
    assert rg.compare(current, _bump(current, -0.01))["accept"]


def test_within_tolerance_is_accepted(current):
    assert rg.compare(current, _bump(current, rg.MEAN_TOL / 2))["accept"]


def test_worse_on_average_is_rejected(current):
    r = rg.compare(current, _bump(current, rg.MEAN_TOL * 2))
    assert not r["accept"] and any("mean" in x for x in r["reasons"])


def test_one_badly_worse_fold_is_rejected_even_if_mean_ok(current):
    cand = _bump(current, -0.02)  # all better...
    _folds(cand)[0]["log_loss"] += 0.02 + rg.FOLD_TOL + 0.001  # ...but 2022 collapses
    r = rg.compare(current, cand)
    assert not r["accept"] and any("2022" in x for x in r["reasons"])


def test_missing_fold_is_rejected(current):
    cand = copy.deepcopy(current)
    del _folds(cand)[1]
    r = rg.compare(current, cand)
    assert not r["accept"] and any("missing" in x for x in r["reasons"])


def test_extra_new_season_fold_is_ignored(current):
    cand = copy.deepcopy(current)
    _folds(cand).append({"val_season": 2027, "n_races": 3, "log_loss": 0.5, "brier": 0.1})
    assert rg.compare(current, cand)["accept"]


def test_live_season_fold_with_more_races_uses_loose_bound_and_not_the_mean(current):
    cand = copy.deepcopy(current)
    live = _folds(cand)[-1]
    live["n_races"] += 2
    live["log_loss"] += rg.LIVE_TOL / 2  # noisy bigger val set: tolerated, mean unaffected
    assert rg.compare(current, cand)["accept"]
    live["log_loss"] += rg.LIVE_TOL
    assert not rg.compare(current, cand)["accept"]


def test_fewer_races_in_a_fold_is_rejected(current):
    cand = copy.deepcopy(current)
    _folds(cand)[-1]["n_races"] -= 1
    assert not rg.compare(current, cand)["accept"]


def test_nan_candidate_metric_is_rejected(current):
    cand = copy.deepcopy(current)
    _folds(cand)[0]["log_loss"] = float("nan")
    assert not rg.compare(current, cand)["accept"]


def test_summary_markdown_names_verdict_and_every_fold(current):
    md = rg.summary_markdown(current, _bump(current, 0.05), rg.compare(current, _bump(current, 0.05)))
    assert "REJECTED" in md and all(str(f["val_season"]) in md for f in _folds(current))


def test_needs_retrain_uses_last_race_when_recorded(current):
    m = {**current, "last_race": {"season": 2026, "round": 14}}
    assert rg.needs_retrain(m, 2026, 15)
    assert not rg.needs_retrain(m, 2026, 14)


def test_needs_retrain_falls_back_to_last_fold_race_count(current):
    # committed manifest predates last_race; its 2026 fold saw N races
    # Create a manifest without last_race to test the fallback logic
    m = {k: v for k, v in current.items() if k != "last_race"}
    assert "last_race" not in m
    # Find the 2026 fold's n_races
    folds = [f for f in m["walk_forward_folds"][m["race_outcome_candidate"]] if f["val_season"] == 2026]
    n_races = folds[-1]["n_races"] if folds else 0
    assert rg.needs_retrain(m, 2026, n_races + 1)
    assert not rg.needs_retrain(m, 2026, n_races)


def test_needs_retrain_new_season_with_no_trained_races(current):
    m = {**current, "last_race": {"season": 2025, "round": 24}}
    assert rg.needs_retrain(m, 2026, 1)


def test_tuned_params_fall_back_to_committed_manifest_without_optuna_db(tmp_path, monkeypatch):
    from f1_predictor.models import manifest as mf

    monkeypatch.setattr(mf, "OPTUNA_DB_PATH", tmp_path / "absent.db")
    committed = json.loads(MANIFEST_PATH.read_text())
    assert mf._load_tuned_params("race_outcome_ranker") == committed["tuned_ranker_params"]
    assert mf._load_tuned_params("dnf_model") == committed["tuned_dnf_params"]
