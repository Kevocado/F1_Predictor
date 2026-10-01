"""A rate-limited season must not vanish from the history window.

`load_history` asks for `HISTORY_SEASONS` seasons and then loops over them
wrapping each fetch in `except Exception: ... continue`. The skip was logged at
INFO and the loop moved on, so a season that failed to fetch produced a history
window one season shorter than requested and nothing downstream could tell. The
model reads fewer seasons, circuit form is an expanding mean over what arrived,
and the forecast is simply less informed than the code claims — with no error,
no warning, and no sign in the returned frames that anything was dropped.

The two skip reasons are not the same fact and must not share a log line:

  * the fetch failed (rate limit, connection reset, timeout) — a statement
    about THIS run. jolpica allows 4 req/s and 500 req/hour unauthenticated
    (`config.JOLPICA_MIN_INTERVAL_S` and the comment above it), so a cold
    container refetching three seasons hits that ceiling for real. The retry
    loop in `_get` handles a transient 429; when it is exhausted, the season
    is gone and the run must say so.
  * the upstream returned an empty frame for a season it does not carry — a
    statement about the calendar. This was the case the original comment had
    in mind, and it was the *only* thing logged at all, because the empty-frame
    branch below the `except` had no log line whatsoever.

So both are reported, both name the season, and the failure path names the
reason. And a 429 that the retry loop *does* recover from must not skip the
season and must not warn — bounded backoff succeeding is the system working.
"""

import logging

import pandas as pd
import pytest
import requests

from f1_predictor.data import jolpica


# --- a fake jolpica response, so no test here touches the network ----------


class _Response:
    """Just enough of `requests.Response` for `_get`: a status, headers, and
    the two methods it calls."""

    def __init__(self, status_code: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Client Error", response=self
            )

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """`_get` sleeps between attempts and `_rate_limit` sleeps between calls.
    A test that exercises the backoff must not spend it in wall clock."""
    monkeypatch.setattr(jolpica.time, "sleep", lambda _seconds: None)


def _payload():
    return {"MRData": {"RaceTable": {"Races": [{"round": "1"}]}}}


# --- 1. an exhausted 429 says it was a 429 ----------------------------------


def test_exhausted_retries_name_the_status_that_failed_them(monkeypatch):
    """The reason is what makes the warning actionable. `_get` retries a 429
    with `Retry-After` and then gives up, raising a RuntimeError whose message
    was `Failed to fetch <url> after 3 attempts` — a 429 and a 404 and a
    timeout all produced that same sentence, so the log line downstream could
    not distinguish 'we were rate limited, run this again later' from 'this
    path does not exist'.
    """
    seen = []

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append(url)
        return _Response(429, headers={"Retry-After": "1"})

    monkeypatch.setattr(jolpica.requests, "get", fake_get)

    with pytest.raises(RuntimeError) as excinfo:
        jolpica._get("2024.json")

    assert len(seen) == 3, "a 429 must still be retried, not failed on first sight"
    assert "429" in str(excinfo.value), (
        f"the exhausted-retry error must name the status that caused it; got "
        f"{str(excinfo.value)!r}"
    )


# --- 2. the skipped season is reported, at a level anyone sees --------------


def test_a_season_whose_fetch_failed_is_reported_by_season_and_reason(monkeypatch, caplog):
    """A 429 that outlasts the retry loop skips the season. Before this, the
    skip was `logger.info` — below the default WARNING threshold, so the
    default-configured run printed nothing at all and the history window came
    back one season short with no trace.
    """
    exhausted = RuntimeError("Failed to fetch https://api.jolpi.ca/ergast/f1/2024.json "
                             "after 3 attempts (last HTTP status 429)")

    def results(season, force_refresh=False):
        if season == 2024:
            raise exhausted
        return pd.DataFrame({"driver_id": [f"d{season}"], "position": [1]})

    def schedule(season, force_refresh=False):
        return pd.DataFrame({"round": [1], "circuit_id": [f"c{season}"]})

    monkeypatch.setattr(jolpica, "load_season_results", results)
    monkeypatch.setattr(jolpica, "fetch_season_schedule", schedule)

    with caplog.at_level(logging.WARNING, logger="f1_predictor.data.jolpica"):
        results_df, schedule_df, _coverage = jolpica.load_history(2026)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got {caplog.records}"
    message = warnings[0].getMessage()
    assert "2024" in message, f"the warning must name the season skipped; got {message!r}"
    assert "429" in message, f"the warning must name the reason; got {message!r}"

    # The other seasons are still loaded: one missing season must not cost
    # the run everything it did fetch.
    assert sorted(results_df["season"].unique().tolist()) == [2025, 2026]
    assert sorted(schedule_df["season"].unique().tolist()) == [2025, 2026]


# --- 3. the empty-frame branch was completely silent ------------------------


def test_a_season_the_upstream_returns_empty_is_reported_too(monkeypatch, caplog):
    """`if df.empty or sch.empty: continue` had no log statement of any kind.
    A season dropping out of the window that way was invisible in a way the
    `except` branch was not — same consequence, no line at all.
    """
    monkeypatch.setattr(
        jolpica, "load_season_results",
        lambda season, force_refresh=False: pd.DataFrame({"driver_id": [f"d{season}"]}),
    )
    monkeypatch.setattr(
        jolpica, "fetch_season_schedule",
        lambda season, force_refresh=False: (
            pd.DataFrame() if season == 2025 else pd.DataFrame({"round": [1]})
        ),
    )

    with caplog.at_level(logging.WARNING, logger="f1_predictor.data.jolpica"):
        jolpica.load_history(2026)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got {caplog.records}"
    message = warnings[0].getMessage()
    assert "2025" in message, f"the warning must name the season skipped; got {message!r}"
    assert "schedule" in message, (
        f"the warning must say which frame came back empty; got {message!r}"
    )


# --- 4. a 429 the retry loop recovers from is not a skip --------------------


def test_a_429_that_a_retry_succeeds_on_loads_the_season_and_warns_about_nothing(
    monkeypatch, caplog
):
    """The point of the backoff is that a transient rate limit costs nothing.
    If recovering from a 429 still skipped the season, the honest fix would
    have been to fail the run instead; it must not.
    """
    responses = [
        _Response(429, headers={"Retry-After": "1"}),
        _Response(200, _payload()),
    ]

    def fake_get(url, params=None, headers=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(jolpica.requests, "get", fake_get)

    with caplog.at_level(logging.WARNING, logger="f1_predictor.data.jolpica"):
        data = jolpica._get("2024.json")

    assert data == _payload(), "the retried fetch's data must be returned"
    assert caplog.records == [], (
        f"a recovered 429 is not a skip and must not warn; got {caplog.records}"
    )
