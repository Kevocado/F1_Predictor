import pandas as pd

from f1_predictor.evaluate import quali_feature_ablation


def _fake_race_frame(seasons=None):
    rows = []
    for rnd in range(1, 6):
        for tier in ["pre_weekend", "post_practice", "post_qualifying"]:
            for i, driver in enumerate(["a", "b", "c"], start=1):
                rows.append(
                    {
                        "season": 2024, "round": rnd, "driver_id": driver, "constructor_id": f"t{i % 2}",
                        "tier": tier, "position": i, "dnf": False,
                        "elo_pre_race": 1600 - i * 5, "team_strength_pre_race": 1500.0,
                        "grid": i if tier == "post_qualifying" else float("nan"),
                        "quali_position": i if tier == "post_qualifying" else float("nan"),
                        "quali_gap_to_pole": 0.1 * i if tier == "post_qualifying" else float("nan"),
                    }
                )
    return pd.DataFrame(rows), ["elo_pre_race", "team_strength_pre_race", "grid", "quali_position", "quali_gap_to_pole"]


def _fake_quali_frame(session_type, seasons=None):
    rows = []
    for rnd in range(1, 6):
        for i, driver in enumerate(["a", "b", "c"], start=1):
            rows.append(
                {
                    "season": 2024, "round": rnd, "driver_id": driver, "constructor_id": f"t{i % 2}",
                    "quali_position": i, "elo_pre_race": 1600 - i * 5, "team_strength_pre_race": 1500.0,
                    "sprint_finish_position": float("nan"),
                }
            )
    return pd.DataFrame(rows), ["elo_pre_race", "team_strength_pre_race", "sprint_finish_position"]


def test_run_ablation_returns_brier_scores_for_both_variants(monkeypatch):
    monkeypatch.setattr(quali_feature_ablation.build_features, "build_training_frame", _fake_race_frame)
    monkeypatch.setattr(quali_feature_ablation.session_build, "build_session_training_frame", _fake_quali_frame)

    result = quali_feature_ablation.run_ablation(seasons=[2024])

    assert set(result["with_feature"].keys()) == {"win", "podium", "points_finish", "dnf"}
    assert set(result["without_feature"].keys()) == {"win", "podium", "points_finish", "dnf"}
    assert result["recommendation"] in ("add", "no_change")
    assert all(0.0 <= v <= 1.0 for v in result["with_feature"].values())
