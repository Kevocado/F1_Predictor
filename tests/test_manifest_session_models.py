import pandas as pd

from f1_predictor.models import manifest


def test_train_all_trains_and_saves_session_models(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "RACE_OUTCOME_MODEL_PATH", tmp_path / "race_outcome_ranker.json")
    monkeypatch.setattr(manifest, "DNF_MODEL_PATH", tmp_path / "dnf_model.json")
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "SPRINT_QUALIFYING_MODEL_PATH", tmp_path / "sprint_qualifying_ranker.json")
    monkeypatch.setattr(manifest, "QUALIFYING_MODEL_PATH", tmp_path / "qualifying_ranker.json")
    monkeypatch.setattr(manifest, "SPRINT_MODEL_PATH", tmp_path / "sprint_ranker.json")
    monkeypatch.setattr(manifest, "SPRINT_DNF_MODEL_PATH", tmp_path / "sprint_dnf_model.json")
    monkeypatch.setattr(
        manifest,
        "SESSION_MODEL_PATHS",
        {
            "sprint_qualifying": (tmp_path / "sprint_qualifying_ranker.json", None),
            "qualifying": (tmp_path / "qualifying_ranker.json", None),
            "sprint": (tmp_path / "sprint_ranker.json", tmp_path / "sprint_dnf_model.json"),
        },
    )

    def _fake_race_frame(seasons=None):
        rows = []
        for rnd in range(1, 6):
            for i, driver in enumerate(["a", "b", "c"], start=1):
                rows.append(
                    {
                        "season": 2024, "round": rnd, "driver_id": driver, "constructor_id": f"t{i % 2}",
                        "tier": "post_qualifying", "position": i, "dnf": False,
                        "elo_pre_race": 1600 - i * 5, "team_strength_pre_race": 1500.0,
                    }
                )
        return pd.DataFrame(rows), ["elo_pre_race", "team_strength_pre_race"]

    def _fake_session_frame(session_type, seasons=None):
        rows = []
        target_col = {"sprint_qualifying": "sprint_quali_position", "qualifying": "quali_position", "sprint": "position"}[
            session_type
        ]
        for rnd in range(1, 4):
            for i, driver in enumerate(["a", "b", "c"], start=1):
                row = {
                    "season": 2024, "round": rnd, "driver_id": driver, "constructor_id": f"t{i % 2}",
                    target_col: i, "elo_pre_race": 1600 - i * 5, "team_strength_pre_race": 1500.0,
                }
                if session_type == "sprint":
                    row["dnf"] = False
                    row["sprint_quali_position"] = i
                rows.append(row)
        return pd.DataFrame(rows), ["elo_pre_race", "team_strength_pre_race"] + (
            ["sprint_quali_position"] if session_type == "sprint" else []
        )

    monkeypatch.setattr(manifest, "build_training_frame", _fake_race_frame)
    monkeypatch.setattr(manifest.session_build, "build_session_training_frame", _fake_session_frame)
    monkeypatch.setattr(
        manifest.walk_forward,
        "prepare_folds",
        lambda seasons: [{"train": pd.DataFrame(), "test": pd.DataFrame()}],
    )
    monkeypatch.setattr(
        manifest.walk_forward,
        "evaluate_candidate",
        lambda folds, candidate, hyperparams=None: pd.DataFrame([{"log_loss": 1.0}]),
    )

    result = manifest.train_all(seasons=[2024])

    assert (tmp_path / "sprint_qualifying_ranker.json").exists()
    assert (tmp_path / "qualifying_ranker.json").exists()
    assert (tmp_path / "sprint_ranker.json").exists()
    assert (tmp_path / "sprint_dnf_model.json").exists()
    assert result["session_models"]["qualifying"]["n_training_rows"] == 9
    assert result["session_models"]["sprint"]["n_training_rows"] == 9
