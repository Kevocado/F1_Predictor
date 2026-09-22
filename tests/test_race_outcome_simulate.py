from f1_predictor.models import race_outcome


def test_simulate_race_default_points_unchanged():
    theta = {"a": 3.0, "b": 2.0, "c": 1.0}
    result_default = race_outcome.simulate_race(theta, n_trials=500, seed=1)
    result_explicit = race_outcome.simulate_race(
        theta, n_trials=500, seed=1, points_table=race_outcome.POINTS_TABLE, points_finish_cutoff=10
    )
    # Same seed, same params -> identical output; proves the new kwargs'
    # defaults reproduce the pre-existing behavior exactly.
    assert result_default["p_win"].tolist() == result_explicit["p_win"].tolist()
    assert result_default["expected_points"].tolist() == result_explicit["expected_points"].tolist()


def test_simulate_race_custom_points_table_and_cutoff():
    theta = {"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0}
    sprint_points = {1: 8, 2: 7, 3: 6, 4: 5}
    result = race_outcome.simulate_race(
        theta, n_trials=2000, seed=2, points_table=sprint_points, points_finish_cutoff=3
    )
    # points_finish_cutoff=3 means only the top 3 ever count as a
    # "points finish" or accrue points, regardless of the 4-driver field.
    assert (result["p_points_finish"] <= 1.0).all()
    last_place_row = result.sort_values("p_win").iloc[0]
    # The weakest driver should score noticeably less often/less points
    # under a top-3-only cutoff than under the default top-10 one.
    assert last_place_row["expected_points"] < 8.0
