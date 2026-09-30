"""The hermeticity guard itself is a test, because a guard nothing checks is a
comment.

`conftest.py` blocks `socket.connect` and assembles a per-session jolpica cache
from committed data. If either stops working, every test that depends on it
still passes by quietly reaching the network — the exact failure this suite is
supposed to be immune to. So:

  * the trap is shown to actually trap, by trying to open a connection,
  * the 2026 fetch that used to die on a fresh checkout is shown to be served
    from committed data, and
  * every committed 2026 fixture is shown to be a complete response, because a
    truncated one would read as "no data for this round" and quietly shrink a
    season — the same class of quiet loss this PR fixes on the 429 path.
"""

import json
import socket
import subprocess
from pathlib import Path

import pytest

from f1_predictor.data import jolpica

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jolpica"


def test_the_socket_guard_actually_guards():
    """A trap that does not trap proves nothing about the suite."""
    with pytest.raises(RuntimeError, match="hermetic"):
        socket.create_connection(("api.jolpi.ca", 443), timeout=1)


def test_a_jolpica_fetch_reads_from_the_seeded_cache_and_does_not_reach_out(monkeypatch):
    """The fetch the network-blocked run used to die on. Without the seeded
    cache `_get` would try to connect and the guard above would raise, which is
    the failure this converts from a hang into an assertion.
    """
    def explode(*args, **kwargs):
        raise AssertionError("fetched over the network; the cache was not seeded")

    monkeypatch.setattr(jolpica.requests, "get", explode)

    schedule = jolpica.fetch_season_schedule(2026)

    assert len(schedule) == 23, f"expected the 2026 calendar, got {len(schedule)} rows"
    assert {"season", "round", "circuit_id", "race_datetime"} <= set(schedule.columns)


def test_the_2026_fixture_is_committed_rather_than_a_local_artifact():
    """The leak was that 2026 had no cache in a fresh tree. This pins that the
    fixture is in the repository, since `data/cache/` is gitignored and a
    fixture that exists only on the machine that ran the fetch is not a fixture.
    """
    committed = {
        Path(line).name
        for line in subprocess.run(
            ["git", "ls-files", "tests/fixtures"],
            capture_output=True, text=True, check=True, cwd=REPO,
        ).stdout.split()
    }
    on_disk = set(FIXTURES.glob("*.json"))

    assert on_disk, "no 2026 fixture on disk at all"
    missing = {p.name for p in on_disk} - committed
    assert not missing, (
        f"these fixtures are not committed, so a fresh clone silently loses them "
        f"and the test that needs 2026 goes back to the network: {sorted(missing)}"
    )


def _record_count(mrdata: dict) -> int:
    """How many rows the response actually carries. jolpica nests per-round
    results inside a one-entry RaceTable and drivers inside a one-entry
    StandingsTable, so `total` is not always the length of the outer list."""
    race_table = mrdata.get("RaceTable")
    if race_table is not None:
        races = race_table.get("Races") or []
        results = [r for race in races for r in race.get("Results", [])]
        return len(results) or len(races)
    lists = mrdata.get("StandingsTable", {}).get("StandingsLists") or []
    return sum(len(entry.get("DriverStandings", [])) for entry in lists)


def test_no_2026_fixture_is_truncated():
    """Each must be a complete jolpica response with the rows it claims. A
    half-written fixture reads downstream as 'no data for this round' and
    shrinks a season without saying so.
    """
    for path in sorted(FIXTURES.glob("2026_*.json")):
        mrdata = json.loads(path.read_text()).get("MRData")
        assert mrdata, f"{path.name} has no MRData envelope"
        claimed, carried = int(mrdata["total"]), _record_count(mrdata)
        assert carried == claimed, (
            f"{path.name} claims {claimed} rows and carries {carried}; a truncated "
            f"fixture would read downstream as missing data, not as a broken fixture"
        )
