"""The retrain gate compares a CI-trained candidate with the committed
models/manifest.json on the 2022-2025 walk-forward folds. Those folds train on
earlier seasons only, so they reproduce bit-for-bit IF every input is identical.
Open-Meteo archive values are NOT stable between fetches (63 of 175 cached
files differed on a re-fetch, e.g. 2025-10-05 precipitation 3.8 vs 1.8 mm), and
data/cache/open_meteo/ was gitignored, so the CI runner re-fetched weather and
rejected every candidate (run 38419468065: +0.0066/+0.0069/+0.0076).
Experiment: same venv + same jolpica, only open_meteo/ swapped for the
committed-model cache -> deltas exactly +0.0000. So finished races' weather
must be tracked, like the finished-season jolpica cache, and kept that way."""

import subprocess
from pathlib import Path

from f1_predictor.data import jolpica, open_meteo

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "retrain-after-race.yml"


def _tracked_weather() -> set[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "data/cache/open_meteo"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return {Path(p).name for p in out.split("\0") if p}


def test_every_finished_season_race_has_tracked_weather():
    tracked = _tracked_weather()
    missing = []
    for season in range(2019, 2026):
        for r in jolpica.fetch_season_schedule(season).itertuples():
            name = f"{open_meteo._cache_key(r.lat, r.long, r.race_datetime.date().isoformat())}.json"
            if name not in tracked:
                missing.append(name)
    assert not missing, f"{len(missing)} finished races have no committed weather, e.g. {missing[:3]}"


def test_tracked_weather_is_never_a_future_forecast():
    """A committed forecast would be served forever (cache hit) instead of the
    archive value once the race has run."""
    assert max(n[:10] for n in _tracked_weather()) <= "2026-10-12"


def test_workflow_resets_weather_after_cache_restore_and_commits_new_weather():
    """actions/cache restore overwrites checked-out files with an older run's
    re-fetched weather; the accepted run must persist new races' weather so the
    next baseline is frozen too."""
    wf = WORKFLOW.read_text()
    restore = wf.index("Restore data cache")
    assert "git checkout -- data/cache/open_meteo" in wf[restore:wf.index("Is there a new completed race?")]
    commit = wf[wf.index("Commit and push if accepted"):]
    assert "data/cache/open_meteo" in commit
