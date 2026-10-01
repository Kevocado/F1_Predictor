"""A skipped season has to be visible on the RESPONSE, not only in a log line.

F1#21 made `load_history` log a WARNING naming the season and the reason when a
season's fetch fails. That is right, and it is not enough. A log line is read by
whoever is watching the container at the time; the person reading the forecast
gets a response with a pick on it and nothing else. Circuit form is an expanding
mean over whatever history arrived, so a rate-limited season did not fail the
run — it quietly made the model less informed, and the served numbers look
exactly like a normal Sunday's numbers.

The gap is that "the model was told less than usual" is a fact about the
response, and the response had no way to carry it. So it carries it now:

    history: {
      complete: bool,               # True only if EVERY requested season loaded
      seasons_requested: [int],     # what the model asked for
      seasons_loaded: [int],        # what it got
      missing_seasons: [{season, reason}],
    }

Three properties this file pins, each because getting it wrong is a way of
lying:

  * `complete` is derived, never asserted. It is True iff nothing was skipped,
    so it cannot drift from the list beside it. There is no code path that sets
    a bare `True` for a run that dropped a season.

  * A `false` carries WHICH seasons and WHY. `complete: false` on its own is
    the same silence the log line was, moved.

  * Absence is its own state. A prediction served from a stored snapshot or a
    backtest was not built from this request's history load, so it reports
    `history: None` — "not reported", NOT `complete: true`. Defaulting a
    missing flag to the reassuring value is the failure mode this whole change
    exists to prevent, so it is pinned directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1_predictor.api import routes
from f1_predictor.api import schemas
from f1_predictor.data import jolpica

# `schemas` is used qualified rather than imported by name on purpose: running
# this file against code without the flag should produce a list of failing
# tests that each name what is missing, not one collection-time ImportError.
# A single ImportError tells you the module is absent; these assertions are the
# specification, and they are worth reading one failure at a time.

_HISTORY_SEASONS = jolpica.HISTORY_SEASONS


# --- helpers: stub the two loaders load_history loops over --------------------


def _stub_loaders(monkeypatch, results, schedule):
    monkeypatch.setattr(jolpica, "load_season_results", results)
    monkeypatch.setattr(jolpica, "fetch_season_schedule", schedule)


def _results_frame(season: int) -> pd.DataFrame:
    return pd.DataFrame({"driver_id": [f"d{season}"], "position": [1]})


def _schedule_frame(season: int) -> pd.DataFrame:
    return pd.DataFrame({"round": [1], "circuit_id": [f"c{season}"]})


def _all_load(season, force_refresh=False):
    return _results_frame(season)


def _all_schedule(season, force_refresh=False):
    return _schedule_frame(season)


def _history(season: int) -> tuple[int, ...]:
    """The seasons load_history will ask for, mirroring its own arithmetic."""
    return tuple(season - i for i in range(_HISTORY_SEASONS) if season - i > 1950)


# --- 1. the coverage object load_history returns ------------------------------


def test_a_skipped_season_is_reported_on_the_return_value_not_only_in_a_log(monkeypatch, caplog):
    """The fetch-failure skip. 2024 raises the exhausted-retry RuntimeError a
    cold container hitting jolpica's hourly quota produces."""
    exhausted = RuntimeError(
        "Failed to fetch https://api.jolpi.ca/ergast/f1/2024.json after 3 attempts "
        "(last HTTP status 429)"
    )

    def results(season, force_refresh=False):
        if season == 2024:
            raise exhausted
        return _results_frame(season)

    _stub_loaders(monkeypatch, results, _all_schedule)

    with caplog.at_level("WARNING", logger="f1_predictor.data.jolpica"):
        _, _, coverage = jolpica.load_history(2026)

    assert coverage.complete is False, (
        "2024 was skipped; the coverage must not report a complete history"
    )
    assert coverage.seasons_requested == _history(2026)
    assert coverage.seasons_loaded == (2026, 2025)
    assert [m.season for m in coverage.missing_seasons] == [2024]
    assert "429" in coverage.missing_seasons[0].reason, (
        "the reason has to survive onto the payload — 'complete: false' with no "
        "why is the same silence the log line was, moved; got "
        f"{coverage.missing_seasons[0].reason!r}"
    )


def test_the_empty_frame_skip_is_on_the_coverage_too(monkeypatch):
    """The other skip: upstream has no rows for a season it does not carry.
    Same consequence — the window is short — and it used to have no log line at
    all, so it is the one most worth putting on the response."""
    monkeypatch.setattr(jolpica, "load_season_results", _all_load)
    monkeypatch.setattr(
        jolpica,
        "fetch_season_schedule",
        lambda season, force_refresh=False: (
            pd.DataFrame() if season == 2025 else _schedule_frame(season)
        ),
    )

    _, _, coverage = jolpica.load_history(2026)

    assert coverage.complete is False
    assert [m.season for m in coverage.missing_seasons] == [2025]
    assert "schedule" in coverage.missing_seasons[0].reason, (
        "the two skips are different facts — a failed fetch is about this run, "
        "an empty frame is about the calendar — and the reason must say which"
    )


def test_a_window_that_loaded_everything_reports_complete_with_no_misses(monkeypatch):
    """The flag must be true sometimes, or it is decoration. True here means
    every season the model asked for is in `seasons_loaded`, and the test says
    so by checking the two lists agree rather than trusting a bare boolean."""
    _stub_loaders(monkeypatch, _all_load, _all_schedule)

    _, _, coverage = jolpica.load_history(2026)

    assert coverage.complete is True
    assert list(coverage.missing_seasons) == []
    assert coverage.seasons_requested == coverage.seasons_loaded == _history(2026)


def test_complete_is_derived_from_the_skips_so_the_two_cannot_disagree(monkeypatch):
    """`complete` is not a field anybody sets; it is True iff `missing_seasons`
    is empty. Build the object the way the schema does and check the invariant
    holds for every combination the loop can produce."""
    seasons = _history(2026)
    for dropped in ([], [seasons[0]], [seasons[0], seasons[1]], list(seasons)):
        missing = [schemas.MissingSeason(season=s, reason="fetch failed") for s in dropped]
        coverage = schemas.HistoryCoverage(
            complete=not missing,
            seasons_requested=list(seasons),
            seasons_loaded=[s for s in seasons if s not in dropped],
            missing_seasons=missing,
        )
        assert coverage.complete is (not coverage.missing_seasons)
        assert set(coverage.seasons_loaded) | {m.season for m in coverage.missing_seasons} == set(seasons)
        assert not (set(coverage.seasons_loaded) & {m.season for m in coverage.missing_seasons})


def test_a_window_that_loaded_nothing_reports_every_season_missing(monkeypatch):
    """Total loss. `load_history` returns two empty frames; the coverage has to
    say all three seasons are gone rather than reporting an empty-but-complete
    window, which would be the most confident possible lie."""
    def nothing_results(season, force_refresh=False):
        raise RuntimeError("Failed to fetch https://api.jolpi.ca/ergast/f1/x.json after 3 attempts (last HTTP status 429)")

    _stub_loaders(monkeypatch, nothing_results, _all_schedule)

    results_df, _, coverage = jolpica.load_history(2026)

    assert results_df.empty
    assert coverage.complete is False
    assert list(coverage.seasons_loaded) == []
    assert [m.season for m in coverage.missing_seasons] == list(_history(2026))


def test_seasons_requested_never_invents_seasons_the_model_did_not_ask_for(monkeypatch):
    """The 1950 floor. load_history stops walking back at 1950, so a season
    before it is neither requested nor missing — listing it as skipped would
    blame the run for a window it never opened."""
    _stub_loaders(monkeypatch, _all_load, _all_schedule)

    _, _, coverage = jolpica.load_history(1953)

    assert coverage.seasons_requested == (1953, 1952, 1951)
    assert 1950 not in coverage.seasons_requested
    assert list(coverage.missing_seasons) == []


# --- 2. the flag reaches the route's response ---------------------------------


def _future_schedule() -> pd.DataFrame:
    """One far-future round, so the race is never 'completed' and the route
    takes the live path (same fixture shape test_driver_names.py uses)."""
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "round": 20,
                "race_name": "Fake Grand Prix",
                "circuit_id": "fake",
                "circuit_name": "Fake Circuit",
                "race_datetime": pd.Timestamp("2099-01-01", tz="UTC"),
                "qualifying_datetime": pd.Timestamp("2099-01-01", tz="UTC"),
                "is_sprint_weekend": False,
            }
        ]
    )


def _sim() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver_id": "max_verstappen",
                "constructor_id": "red_bull",
                "p_win": 0.31,
                "p_podium": 0.72,
                "p_points_finish": 0.94,
                "p_dnf": 0.06,
                "expected_position": 2.0,
                "expected_points": 18.0,
                "actual_position": None,
                "actual_dnf": None,
            }
        ]
    )


def _coverage(complete: bool, missing: list[schemas.MissingSeason]) -> schemas.HistoryCoverage:
    seasons = _history(2026)
    return schemas.HistoryCoverage(
        complete=complete,
        seasons_requested=list(seasons),
        seasons_loaded=[s for s in seasons if s not in {m.season for m in missing}],
        missing_seasons=missing,
    )


def test_the_race_prediction_response_carries_the_coverage(monkeypatch):
    """The end of the chain. F1#21's warning stops at the log; this is the
    reader's half of the same fact."""
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _future_schedule())
    skipped = _coverage(False, [schemas.MissingSeason(season=2024, reason="fetch failed: RuntimeError (last HTTP status 429)")])
    monkeypatch.setattr(routes, "_predict_upcoming_race", lambda *a: (_sim(), "post_qualifying", skipped))

    result = routes.get_race_prediction(2026, 20)

    assert result.history is not None, (
        "a live prediction built on a short history must say so on the response"
    )
    assert result.history.complete is False
    assert [m.season for m in result.history.missing_seasons] == [2024]


def test_the_feature_frame_reports_the_history_it_actually_loaded(monkeypatch):
    """The seam that decides whether the flag is honest. `_future_feature_frame`
    is the only caller of `load_history` on the prediction path, so if the
    coverage it hands on is not the coverage that run produced, the response is
    reporting something that never happened."""
    def rate_limited(season, force_refresh=False):
        if season == 2024:
            raise RuntimeError("Failed to fetch https://api.jolpi.ca/ergast/f1/2024.json after 3 attempts (last HTTP status 429)")
        return _results_frame(season)

    _stub_loaders(monkeypatch, rate_limited, _all_schedule)
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season, force_refresh=False: _future_schedule())
    monkeypatch.setattr(
        routes.jolpica, "fetch_driver_standings",
        lambda season, round_=None, force_refresh=False: pd.DataFrame({"driver_id": ["max_verstappen"]}),
    )
    monkeypatch.setattr(
        routes.championship_projection, "build_future_feature_rows",
        lambda *a, **k: pd.DataFrame({"driver_id": ["max_verstappen"]}),
    )

    row = _future_schedule().iloc[0]
    _, _, reported = routes._future_feature_frame(2026, 20, row)

    assert [m.season for m in reported.missing_seasons] == [2024], (
        "the frame must report the season the real load_history just skipped, "
        f"not a hand-made one; got {reported}"
    )
    assert reported.complete is False
    assert "429" in reported.missing_seasons[0].reason


def test_a_prediction_that_did_not_load_history_reports_nothing_at_all(monkeypatch):
    """A stored snapshot and a backtest are built from history this request
    never touched, so there is nothing to report. The tempting default is
    `complete: true` — 'no seasons were skipped, after all' — and that is the
    one value that would be a lie: the run that built it may well have skipped
    one, and this payload cannot know. So absence is absence."""
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _future_schedule())
    monkeypatch.setattr(routes, "_predict_upcoming_race", lambda *a: (_sim(), "post_qualifying", None))

    result = routes.get_race_prediction(2026, 20)

    assert result.history is None, (
        "a response with no history load behind it must report history: None, "
        "never complete: true"
    )


def test_the_flag_is_never_a_bare_boolean(monkeypatch):
    """`history_complete: false` on its own is the log line again. The shape
    carries the seasons and the reason, and this pins all three on one object."""
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _future_schedule())
    skipped = _coverage(
        False, [schemas.MissingSeason(season=2024, reason="upstream returned an empty results frame")]
    )
    monkeypatch.setattr(routes, "_predict_upcoming_race", lambda *a: (_sim(), "post_qualifying", skipped))

    payload = routes.get_race_prediction(2026, 20).model_dump()

    assert set(payload["history"]) == {
        "complete", "seasons_requested", "seasons_loaded", "missing_seasons",
    }
    assert payload["history"]["missing_seasons"][0]["season"] == 2024
    assert payload["history"]["missing_seasons"][0]["reason"]
    assert isinstance(payload["history"]["missing_seasons"][0]["reason"], str)


def test_an_older_snapshot_without_the_field_still_validates():
    """The committed public_snapshot.json predates this field, and the public
    deployment serves that file verbatim. A non-nullable field would 500 the
    endpoint on its own response until the next snapshot refresh — the same
    trap `session_datetime` and `driver_name` are nullable for."""
    old = {
        "season": 2024, "round": 5, "race_name": "Fake GP", "tier": "pre_weekend",
        "source": "live", "predictions": [],
    }
    response = schemas.RacePredictionResponse(**old)

    assert response.history is None


def test_the_schema_rejects_a_complete_flag_that_contradicts_its_own_list():
    """The field is derived from `missing_seasons` on the way in, so a hand-built
    payload claiming `complete: true` while naming three missing seasons cannot
    reach a reader."""
    with pytest.raises(ValueError):
        schemas.HistoryCoverage(
            complete=True,
            seasons_requested=[2026, 2025, 2024],
            seasons_loaded=[2026],
            missing_seasons=[schemas.MissingSeason(season=s, reason="fetch failed") for s in (2025, 2024)],
        )