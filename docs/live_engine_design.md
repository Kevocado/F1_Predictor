# Live in-race engine — implemented

Requirement 3: while a race is actually happening, consume OpenF1's live
feed and continuously update "who's likely to win from here" as laps
unfold — a streaming/continuously-recomputed prediction, not a single
pre-race number that goes stale at lights-out.

**Status: built and wired end-to-end**, validated against real historical
data both via replay and via a synthetic "pretend this real completed
session is live" integration test (no genuinely live session has occurred
during this build — see section 4 for what that leaves untested).

- `data/fastf1_client.py::build_lap_snapshots` — the training-corpus
  extractor. Confirmed against real races across 2022–2026.
- `models/live_win_prob.py` — corpus assembly, training (two regularized
  XGBoost `binary:logistic` classifiers), `predict_live`.
- `data/openf1.py::build_live_state_frame` — the live-inference feature
  builder, same column shape as training.
- `models/live_poller.py` — the background polling loop + in-process
  cache, started from `api/main.py`'s FastAPI lifespan.
- `GET /api/live/current` — serves the cache; `{"live": false}` when no
  session is active.
- `evaluate/live_replay.py` — replay/demo harness (also the Phase 3
  checkpoint script).

## 1. Training corpus

`build_lap_snapshots(season, round_, driver_code_to_id, elo_pre_race,
team_strength_pre_race, driver_to_constructor)` → one row per (lap,
driver) still running that lap:

| Column | Source | Notes |
|---|---|---|
| `lap_number`, `laps_remaining` | FastF1 `Laps.LapNumber` | |
| `current_position` | FastF1 `Laps.Position` | |
| `gap_to_leader_s` | derived: `lap_seconds - min(lap_seconds)` per `LapNumber` | `lap_seconds` = `Laps.Time` (cumulative session time) |
| `gap_ahead_s` | derived: consecutive diff after sorting by `(LapNumber, Position)` | 0 for the leader |
| `tyre_age_laps` | FastF1 `Laps.TyreLife` | |
| `pace_delta_last_3_laps` | derived: driver's rolling-3-lap mean lap time minus the field's median rolling-3-lap mean, same lap | relative pace, not absolute |
| `pit_stops_so_far`, `pitted_this_lap` | FastF1 `Laps.PitInTime`/`PitOutTime` | cumulative count via `groupby(Driver).cumsum()` |
| `safety_car_active` | FastF1 `Laps.TrackStatus` contains `{4,6,7}` | SC / VSC / VSC-ending codes |
| `grid_position` | FastF1 `session.results.GridPosition` | |
| `driver_pre_race_strength`, `constructor_strength` | `features/elo.py` / `features/team_strength.py`, computed ONCE across the whole corpus | ties the live model to the same driver/car strength signal `race_outcome.py` uses |
| `won`, `podium` (labels) | FastF1 `session.results.Position` | that driver's EVENTUAL result |

`driver_code_to_id` bridges FastF1's 3-letter codes (`VER`) to jolpica's
`driver_id` (`max_verstappen`) via jolpica's own `fetch_race_results`
(`driver_code` column).

`models/live_win_prob.py::build_training_corpus(seasons)` fetches jolpica
results for the season range once, computes Elo/team-strength history ONCE
(no-lookahead, same as `features/build.py`), then calls
`build_lap_snapshots` per completed race and concatenates.

**Season range: 2022–present** (`DEFAULT_LIVE_SEASONS_START = 2022` in
`live_win_prob.py`) — current ground-effect-era cars have materially
different dynamics than 2018-2021 cars, and each FastF1 session load is
slow enough that narrowing the window meaningfully cuts corpus-build time
without losing era-relevant signal.

**Actual corpus, as built**: 115,228 rows across 104 races (2022–2026,
essentially the full window — a couple of early attempts were cut short by
jolpica's rate limit, resolved by simply re-running once the limit window
reset). **Confirmed real constraint, not hypothetical**: FastF1 itself
calls jolpica internally for metadata, uncoordinated with this project's
own rate-limited client — the combined call volume can exceed jolpica's
500/hour limit mid-build. `build_training_corpus` already skips (rather
than crashes on) a race whose FastF1 load fails for any reason and logs
which ones, so a partial run is always resumable by re-running later.

## 2. Model

Two XGBoost `binary:logistic` classifiers, **regularized with race-level
held-out early stopping** — confirmed necessary, not precautionary. The
first trained version (200 trees, depth 5, no early stopping) predicted
99%+ win probability for a driver simply leading at lap 1; the corpus's own
true base rate for that exact state (lap 1, `gap_to_leader_s == 0`) is
58.6%. `gap_to_leader_s` is a very separable feature that an unregularized
deep ensemble will happily overfit past the point of genuine calibration.

```python
# models/live_win_prob.py
LIVE_FEATURE_COLUMNS = [
    "lap_number", "laps_remaining", "current_position", "gap_to_leader_s",
    "gap_ahead_s", "tyre_age_laps", "pace_delta_last_3_laps",
    "pit_stops_so_far", "pitted_this_lap", "safety_car_active",
    "grid_position", "driver_pre_race_strength", "constructor_strength",
]

_LIVE_MODEL_PARAMS = dict(
    objective="binary:logistic", n_estimators=400, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=2.0,
    tree_method="hist", eval_metric="logloss", early_stopping_rounds=30,
)
# train/val split is by WHOLE RACE (_race_level_split), not by row — every
# lap of one race shares the same eventual outcome, so a row-level split
# would leak that outcome across train/val and hide overfitting.
```

Result on the real corpus: win model best iteration 119/400 (early
stopping fired well short of the cap), held-out val log-loss 0.059;
podium model best iteration 251/400, val log-loss 0.121. Replayed against
a held-out 2026 race (`evaluate/live_replay.py`), the actual winner's
`p_win` now moves realistically through the race — 0.80 (lap 1) → dips to
0.53 (lap 21) → 0.91 (lap 41) → dips to 0.59 (lap 51) → 0.93 (final lap) —
genuinely uncertain mid-race rather than a flat, overconfident line.

`tyre_compound` was dropped from the feature set rather than resolved as
one-hot/categorical (the original design's open question) — `tyre_age_laps`
already carries most of the same degradation signal, and cutting an
under-populated categorical column simplified the live-serving side (no
compound string parsing needed from OpenF1's `/stints`).

## 3. Live serving (OpenF1)

Verified directly against the real OpenF1 API:

- `GET /v1/sessions?session_key=latest` — confirmed live: returned the
  real 2026 Dutch GP race session on 2026-08-23.
- `GET /v1/position`, `/v1/intervals`, `/v1/race_control` — **confirmed
  gotcha: the `limit` query param breaks all three** (`{"detail": "No
  results found."}` with it set, real data without). `data/openf1.py`
  omits `limit` everywhere and filters/slices client-side.
- `GET /v1/intervals` — `gap_to_leader`/`interval` map onto
  `gap_to_leader_s`/`gap_ahead_s`, **but confirmed can be a literal string
  like `"+1 LAP"` for a lapped driver, not a numeric seconds value** —
  `data/openf1.py::_safe_float` catches this and returns NaN rather than
  raising.
- `GET /v1/stints` — tyre `compound`/`tyre_age_at_start` per stint; also
  the live pit-stop count (`len(stints) - 1`), since `/pit` returned no
  data on manual testing.
- `GET /v1/laps` — per-lap durations, feeds `pace_delta_last_3_laps` the
  same way `Laps.LapTime` does in training.
- Driver identity bridges three separate schemes: OpenF1's `driver_number`
  (int) → `GET /v1/drivers` → `name_acronym` (FastF1's code) →
  `driver_code_to_id` (built from jolpica) → jolpica `driver_id`.
- **No reliable total-lap-count source** — neither OpenF1 nor jolpica's
  schedule expose a race's planned lap count before it finishes. Resolved
  with `CIRCUIT_LAP_COUNTS`, a static table of standard race distance per
  circuit (public, stable F1 knowledge); a circuit missing from it falls
  back to NaN `laps_remaining` rather than a guess.

**Polling** (`models/live_poller.py`, started from `api/main.py`'s
lifespan): every 18s (`config.OPENF1_LIVE_POLL_INTERVAL_S`), checks
`sessions?session_key=latest`; if it's a live Race session, resolves it to
a jolpica season/round via date match, confirms qualifying has happened
(grid needed) and jolpica doesn't yet have a final classification (race
still in progress), builds the live feature frame, and updates the cache.
Comfortably under OpenF1's 30 req/min budget. No websockets — a
`.predict_proba` call is cheap enough that plain polling on both ends
(server → OpenF1, frontend → server) is simple and sufficient.

**Replay/demo harness**: `python -m f1_predictor.evaluate.live_replay
--season 2026 --round 12 [--sleep N]` — replays a real race lap-by-lap
through the trained models, printing the actual winner's `p_win`/model's
top pick at each lap. This is also the Phase 3 checkpoint script.

## 4. Known limitations (confirmed, not resolved)

- **A retired car can show a misleadingly competitive live prediction.**
  `/position` is a sparse event log — it only emits a row when a driver's
  position *changes*, not a periodic snapshot. Confirmed directly: in a
  synthetic "replay a real completed session as if live" test, a driver
  who retired roughly 20 minutes before the checkered flag still showed
  their last (mid-race, P1) position with no error or flag, which the
  model read as "leading with laps still to go." An attempt to filter
  this by update-age was reverted because it's ambiguous with the far more
  common case of a driver who simply held a stable position for a long
  stretch without any change events — that heuristic excluded the actual
  race winner from the results, which is worse than the original problem.
  A real fix needs a cleaner retirement signal (e.g. cross-referencing
  race-control retirement messages), not yet built.
- **OpenF1's ~3s live latency claim is unverified against a genuinely live
  session** — only checked against historical/completed sessions during
  this build, since none was live. The 18s polling interval has slack
  either way.
- **The end-to-end live path is validated synthetically, not against a
  real live race.** `models/live_poller.py::poll_once()` was exercised by
  monkeypatching `is_session_live`/`fetch_latest_session` to point at a
  real, completed 2026 session and temporarily hiding its jolpica
  classification — this confirmed the full context-resolution →
  feature-building → prediction pipeline runs correctly end-to-end against
  real data, but a genuinely in-progress session (data arriving
  incrementally, mid-session edge cases like a red flag or restart) has
  not been observed.
