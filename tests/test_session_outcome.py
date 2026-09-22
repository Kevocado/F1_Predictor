import pandas as pd

from f1_predictor.models import session_outcome


def _toy_frame(target_col: str, n_rounds: int = 3) -> pd.DataFrame:
    rows = []
    for rnd in range(1, n_rounds + 1):
        for i, driver in enumerate(["a", "b", "c", "d"], start=1):
            rows.append(
                {
                    "season": 2024,
                    "round": rnd,
                    "driver_id": driver,
                    "constructor_id": f"team_{i % 2}",
                    target_col: i,  # driver "a" always finishes 1st, etc.
                    "elo_pre_race": 1600 - i * 10,
                    "team_strength_pre_race": 1500.0,
                }
            )
    return pd.DataFrame(rows)


def test_session_specs_has_three_non_race_types():
    assert set(session_outcome.SESSION_SPECS.keys()) == {"sprint_qualifying", "qualifying", "sprint"}


def test_session_specs_qualifying_has_no_dnf():
    spec = session_outcome.SESSION_SPECS["qualifying"]
    assert spec.has_dnf is False
    assert spec.target_column == "quali_position"


def test_session_specs_sprint_uses_sprint_points_table():
    from f1_predictor.models.championship_projection import SPRINT_POINTS_TABLE

    spec = session_outcome.SESSION_SPECS["sprint"]
    assert spec.has_dnf is True
    assert spec.points_table == SPRINT_POINTS_TABLE
    assert spec.points_finish_cutoff == 8


def test_train_session_ranker_and_predict_session_roundtrip():
    spec = session_outcome.SESSION_SPECS["qualifying"]
    feature_cols = ["elo_pre_race", "team_strength_pre_race"]
    df = _toy_frame(spec.target_column)

    ranker = session_outcome.train_session_ranker(df, feature_cols, spec)
    session_df = df[df["round"] == 3].reset_index(drop=True)
    sim = session_outcome.predict_session(ranker, session_df, feature_cols, spec, n_trials=500, seed=0)

    assert set(sim["driver_id"]) == {"a", "b", "c", "d"}
    assert (sim["p_dnf"] == 0.0).all(), "qualifying-type sessions have no DNF concept"
    # driver "a" has the strongest elo and always qualifies 1st in the toy
    # data — its win probability should be clearly the highest of the four.
    top = sim.sort_values("p_win", ascending=False).iloc[0]
    assert top["driver_id"] == "a"
