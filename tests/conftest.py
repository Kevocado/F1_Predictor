"""Two properties every test in this suite is entitled to assume.

**1. A test never opens a socket.** `socket.create_connection` and
`socket.socket.connect` raise for the whole session. The guard exists because
the alternative is invisible: a test that forgets a stub does not fail, it
reaches api.jolpi.ca, and the suite is green because the network was up. The
same trap already guards the frontend (`frontend/src/test/setup.ts` rejects
every `fetch`, and `frontend/src/test/no-network.test.ts` pins that it does);
this is the Python half of that arrangement, and it is why the hermetic claim
in this repo's PRs is checkable with `pytest` alone instead of by unplugging
the cable and hoping.

**2. The jolpica cache a test reads is assembled per run, in a temp dir.**
`_cache_or_fetch` returns the cache file when it exists, so a test that wanted
season 2026 fetched it once and every later run read it off this machine's
disk. That is how `tests/test_race_specific_forecasts.py` came to need the
network on a fresh checkout: 2026 is the in-progress season, so unlike 2019-2025
there is no committed cache for it, and a fresh tree had to go and get one.

So the cache directory is a per-session temp directory, seeded with:

  * the immutable history the repo already tracks, `data/cache/jolpica/*.json`
    (2019-2025 — force-added past the `data/cache/` gitignore rule on the
    strength of being finished seasons), and
  * `tests/fixtures/jolpica/*.json`, a frozen snapshot of the 2026 season.

Both are enumerated with `git ls-files`, NOT with `glob("*.json")`. A glob reads
whatever is on disk, and `data/cache/` on a working machine holds two populations
at once: the 312 files the repo tracks, and every cache that machine fetched for
itself while developing — measured 372 on the one this was found on, 60 of them
local-only including `2026_16_results.json`. Those files are invisible to a fresh
clone, so a glob lets a test read history that exists on one machine and nowhere
else: the same test then proves something different there than it does in CI.
Reading git's own index instead makes the seeded cache identical everywhere,
which is the whole point of the arrangement.
`tests/test_cache_seed_is_tracked_only.py` plants an untracked file and asserts
the seed leaves it out, so this paragraph is checked rather than trusted.

The 2026 snapshot is a fixture rather than a cache entry on purpose. Adding it
to `data/cache/jolpica/` would work for the test and be wrong for the app:
`_cache_or_fetch` serves a cache hit without refetching, so a committed
`2026_latest_driver_standings.json` would hand every fresh clone a standings
table frozen at the moment it was committed and never refresh it for the rest
of the season. Under `tests/fixtures/` it is only ever read by a test, where
being a fixed snapshot is the entire point.

Everything else is unchanged: the real `jolpica` code runs, the real 2024-2025
results load, and the cross-season circuit history
`test_race_specific_forecasts.py` asserts on is the genuine article.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from f1_predictor.data import jolpica

REPO = Path(__file__).resolve().parents[1]
TRACKED_CACHE = REPO / "data" / "cache" / "jolpica"
TEST_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jolpica"

_SEED_SOURCES = (TRACKED_CACHE, TEST_FIXTURES)


def _tracked_files(directory: Path) -> list[str]:
    """File names git tracks directly inside `directory`.

    `git ls-files`, not `glob`: `data/cache/` is gitignored and force-added file
    by file, so on a working machine that directory also holds every cache the
    machine fetched for itself — 60 local-only files on the one this was found
    on. A glob seeds those too, and a test that reads them proves something
    here that it cannot prove on any other machine. Ask git what the repository
    contains and the seeded cache is the same everywhere.

    `ls-files` reads the index, so it is also right about files that are tracked
    but currently absent from disk (a sparse or partial checkout): those are
    skipped rather than raising, and the test that needed them fails with the
    real reason instead of an opaque copy error.
    """
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", str(directory.relative_to(REPO))],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    names = (Path(entry).name for entry in listed.split("\0") if entry.endswith(".json"))
    return [name for name in names if (directory / name).is_file()]


def _seed_cache(cache: Path) -> None:
    """Fill `cache` with the tracked jolpica data, and only that.

    Split out of the fixture so a test can call the same code path the suite
    runs on, rather than re-implementing it and passing for the wrong reason.
    """
    for source in _SEED_SOURCES:
        for name in _tracked_files(source):
            shutil.copy(source / name, cache / name)


@pytest.fixture(scope="session", autouse=True)
def no_network():
    """Fail loudly and immediately on any outbound connection.

    A `ProxyError` pointing at a dead port takes 15s to surface and reads like
    a bug in the code under test. This reads like what it is: a test that
    forgot to stub something.
    """
    def blocked(*args, **kwargs):
        raise RuntimeError(
            "a test tried to open a network connection. F1's suite is hermetic: "
            "stub the fetch, or read from data/cache/jolpica/ (finished seasons) "
            "or tests/fixtures/jolpica/ (the 2026 snapshot). See tests/conftest.py."
        )

    saved_connect = socket.socket.connect
    saved_create = socket.create_connection
    socket.socket.connect = blocked
    socket.create_connection = blocked
    try:
        yield
    finally:
        socket.socket.connect = saved_connect
        socket.create_connection = saved_create


@pytest.fixture(scope="session", autouse=True)
def jolpica_cache(tmp_path_factory):
    """Point `jolpica` at a per-session cache, so no test reads or writes
    `data/cache/jolpica/` and none of them depend on what this machine
    happens to have fetched."""
    cache = tmp_path_factory.mktemp("jolpica-cache")
    _seed_cache(cache)

    saved = jolpica.JOLPICA_CACHE_DIR
    jolpica.JOLPICA_CACHE_DIR = cache
    try:
        yield cache
    finally:
        jolpica.JOLPICA_CACHE_DIR = saved
