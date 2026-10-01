"""The per-session jolpica cache must be assembled from files git tracks, and
nothing else.

`tests/conftest.py` seeds a temp cache from two directories with
`source.glob("*.json")`. A glob reads whatever is on DISK. `data/cache/` is
gitignored and force-added file by file, so the directory on any given machine
holds two different populations at once:

  * the 312 files the repo tracks (2019-2025, force-added past the ignore rule),
  * caches this machine fetched for itself while developing.

On the machine this was found on:

    ls data/cache/jolpica/*.json | wc -l          -> 372
    git ls-files data/cache/jolpica | wc -l      -> 312

— 60 files that exist only locally, among them `2026_16_results.json`, the
in-progress season's round 16. None of them is in a fresh clone.

That is not a tidiness complaint. The glob decides what the tests prove. With
it, `test_race_specific_forecasts.py` asserts against a history window that on
this machine includes a season and a round no other machine's copy of that test
will ever load, so a green run proves something different here than it does in
CI. And a local file that has since been deleted upstream, or was truncated
mid-write by an interrupted fetch, is served in preference to nothing at all.

So the seed enumerates `git ls-files` instead of globbing, and this file makes
that non-vacuous: it plants a file git does not track and asserts the seed
leaves it behind. A guard nothing checks is a comment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import conftest as f1_conftest

REPO = Path(__file__).resolve().parents[1]
TRACKED_CACHE = REPO / "data" / "cache" / "jolpica"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jolpica"


def _git_tracked(relative_dir: str) -> set[str]:
    """File names git tracks directly inside `relative_dir`."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", relative_dir],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return {Path(entry).name for entry in out.stdout.split("\0") if entry}


@pytest.fixture
def planted_untracked():
    """A file that exists in `data/cache/jolpica/` and that git does not track —
    the exact shape of the leak, reproduced rather than described.

    `data/cache/` is gitignored, so this file is invisible to `git ls-files`
    while being completely visible to `glob("*.json")`. Removed on the way out
    whatever the test does, so a failure cannot leave the checkout dirty.

    Season 9999 round 99 is deliberately not a real cache name: the fixture
    clears any leftover before writing rather than refusing to run, so a crashed
    previous run cannot turn the next one into a confusing error."""
    path = TRACKED_CACHE / "9999_99_results.json"
    path.unlink(missing_ok=True)
    path.write_text('{"MRData": {"RaceTable": {"Races": []}}, "total": "0"}')
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


# --- 1. the planted file is visible to a glob and invisible to git ------------


def test_a_file_git_does_not_track_is_invisible_to_the_enumeration(planted_untracked):
    """Non-vacuity first: if the planted file were not visible to the old glob,
    every assertion below would pass for the wrong reason."""
    assert planted_untracked.exists()
    assert planted_untracked in list(TRACKED_CACHE.glob("*.json")), (
        "the premise of this file: an untracked file in data/cache/jolpica/ IS "
        "what TRACKED_CACHE.glob('*.json') finds"
    )
    assert planted_untracked.name not in _git_tracked("data/cache/jolpica")


# --- 2. the seed leaves it out, and still carries the tracked history --------


def test_the_seeded_cache_excludes_a_file_that_git_does_not_track(planted_untracked, tmp_path):
    """The fix. A cache file that exists on one machine must not be able to
    change what the suite proves on that machine."""
    cache = tmp_path / "cache"
    cache.mkdir()

    f1_conftest._seed_cache(cache)

    seeded = {p.name for p in cache.glob("*.json")}
    assert planted_untracked.name not in seeded, (
        f"{planted_untracked.name} is not tracked by git, so it exists only on "
        "this machine; seeding it lets a test read history no fresh clone will "
        "have, so the same test proves different things in different places"
    )


def test_the_seeded_cache_is_exactly_the_tracked_set(planted_untracked, tmp_path):
    """The other direction, because a guard that copies nothing also satisfies
    the test above. The tracked 2019-2025 history and the 2026 fixtures must
    all still be there — a fix that emptied the cache would make the suite
    hermetic by refusing to run."""
    cache = tmp_path / "cache"
    cache.mkdir()

    f1_conftest._seed_cache(cache)

    tracked = _git_tracked("data/cache/jolpica") | _git_tracked("tests/fixtures/jolpica")
    assert tracked, "git reports no tracked jolpica data at all; the enumeration is wrong"

    seeded = {p.name for p in cache.glob("*.json")}
    assert seeded == tracked, (
        f"the seeded cache must be exactly the tracked set; missing "
        f"{sorted(tracked - seeded)[:5]}, unexpected {sorted(seeded - tracked)[:5]}"
    )


# --- 3. the real session cache the suite runs on ------------------------------


def test_the_live_session_cache_holds_only_tracked_files(jolpica_cache):
    """The fixture the suite actually uses, asserted on the directory it
    actually produced — not on a re-implementation of it."""
    seeded = {p.name for p in jolpica_cache.glob("*.json")}
    tracked = _git_tracked("data/cache/jolpica") | _git_tracked("tests/fixtures/jolpica")

    assert seeded, "the session cache is empty; the fixture seeded nothing"
    untracked = seeded - tracked
    assert not untracked, (
        f"the live session cache holds {len(untracked)} file(s) git does not "
        f"track, so this run's results depend on this machine's disk: "
        f"{sorted(untracked)[:5]}"
    )


def test_the_committed_fixture_directory_is_entirely_tracked():
    """`tests/fixtures/jolpica/` is the 2026 snapshot. It is not gitignored, so
    this is nearly a tautology — which is the point: it is the assertion that
    catches someone gitignoring it later, the same way a missing 2026 cache went
    unnoticed in the first place."""
    on_disk = {p.name for p in FIXTURES.glob("*.json")}
    tracked = _git_tracked("tests/fixtures/jolpica")

    assert on_disk, "no 2026 fixture on disk at all"
    assert on_disk - tracked == set(), (
        f"fixtures present on disk but not tracked: {sorted(on_disk - tracked)}"
    )