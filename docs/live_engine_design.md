# Live in-race engine — design (Phase 3, not yet implemented)

Requirement 3: while a race is actually happening, consume OpenF1's live
feed and continuously update "who's likely to win from here" as laps
unfold — a streaming/continuously-recomputed prediction, not a single
pre-race number that goes stale at lights-out.

This document is the concrete plan for it. **Built and verified so far:**
`data/fastf1_client.py::build_lap_snapshots` — confirmed working against a
real 2024 race (Bahrain GP: 1129 lap-state rows, gap-to-leader/position/
tyre/pit-stop columns all correct, `won`/`podium` labels correct). Only
this one data-extraction module exists; the trained model, the OpenF1
live-polling client, and the API/serving layer below are design, not code.

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
| `tyre_compound`, `tyre_age_laps` | FastF1 `Laps.Compound`/`TyreLife` | |
| `pace_delta_last_3_laps` | derived: driver's rolling-3-lap mean lap time minus the field's median rolling-3-lap mean, same lap | relative pace, not absolute |
| `pit_stops_so_far`, `pitted_this_lap` | FastF1 `Laps.PitInTime`/`PitOutTime` | cumulative count via `groupby(Driver).cumsum()` |
| `safety_car_active` | FastF1 `Laps.TrackStatus` contains `{4,6,7}` | SC / VSC / VSC-ending codes |
| `grid_position` | FastF1 `session.results.GridPosition` | |
| `driver_pre_race_strength`, `constructor_strength` | passed in from `features/elo.py` / `features/team_strength.py`, computed ONCE across the whole corpus (not per-race) | ties the live model to the same driver/car strength signal `race_outcome.py` uses |
| `won`, `podium` (labels) | FastF1 `session.results.Position` | that driver's EVENTUAL result — every lap-state row is labeled with the outcome it's predicting toward, the standard win-probability-model pattern |

**Driver identity**: FastF1 uses 3-letter codes (`VER`), not jolpica's
`driver_id` (`max_verstappen`). Confirmed fix: build a
`driver_code_to_id` map per race from jolpica's own `fetch_race_results`
(`driver_code` column already exists there) — no separate mapping table
needed.

**Corpus assembly** (not yet built — belongs in
`models/live_win_prob.py::build_training_corpus(seasons)`):
1. Fetch jolpica results for the season range once, compute
   `elo.compute_elo_history` / `team_strength.compute_team_strength_history`
   ONCE across it (no-lookahead, same as `features/build.py`).
2. For each completed race in range, slice that race's `elo_pre_race`/
   `team_strength_pre_race` out of the precomputed history, build the
   `driver_code_to_id` map from jolpica's results for that race, and call
   `build_lap_snapshots`.
3. Concatenate.

**Season range — recommend 2022–present, not 2018–present.** The
architecture-design pass suggested 2018+ for maximum sample size, but two
things argue for narrowing it: (a) current ground-effect-era cars (2022
regs onward) have materially different overtaking/tyre-degradation
dynamics than 2018-2021 cars, so pre-2022 lap states are a worse match for
what the model needs to predict *now*; (b) each FastF1 session load is
slow (confirmed: even with `telemetry=False`, a single race took real wall
time to fetch and cache) — 2018-present is ~150 races, 2022-present is
~90, both crossing into "several-hour background job" territory the first
time it's built, but 90 is meaningfully cheaper and the era-consistency
argument makes it the better trade, not just the cheaper one. ~90 races ×
~58 laps × ~20 drivers ≈ **100k-105k rows** — smaller than the original
150-200k estimate but still ample for two `binary:logistic` models.

## 2. Model

Two XGBoost `binary:logistic` classifiers (mirrors `models/dnf.py`'s
shape, not a new pattern):

```python
# models/live_win_prob.py (not yet built)
LIVE_FEATURE_COLUMNS = [
    "lap_number", "laps_remaining", "current_position", "gap_to_leader_s",
    "gap_ahead_s", "tyre_age_laps", "pace_delta_last_3_laps",
    "pit_stops_so_far", "pitted_this_lap", "safety_car_active",
    "grid_position", "driver_pre_race_strength", "constructor_strength",
]  # tyre_compound needs one-hot/categorical encoding — see open question below

def train_live_models(corpus: pd.DataFrame) -> tuple[xgb.XGBClassifier, xgb.XGBClassifier]:
    win_model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=200, max_depth=5, tree_method="hist")
    win_model.fit(corpus[LIVE_FEATURE_COLUMNS], corpus["won"].astype(int))
    podium_model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=200, max_depth=5, tree_method="hist")
    podium_model.fit(corpus[LIVE_FEATURE_COLUMNS], corpus["podium"].astype(int))
    return win_model, podium_model

def predict_live(models, state_df: pd.DataFrame) -> pd.DataFrame:
    win_model, podium_model = models
    return pd.DataFrame({
        "driver_id": state_df["driver_id"],
        "p_win": win_model.predict_proba(state_df[LIVE_FEATURE_COLUMNS])[:, 1],
        "p_podium": podium_model.predict_proba(state_df[LIVE_FEATURE_COLUMNS])[:, 1],
    })
```

**Validation plan**: held-out-race walk-forward (same discipline as
`evaluate/walk_forward.py` — train on races strictly before a held-out
race, never on it), but the metric of interest is a *calibration curve
over race progress*: bucket rows by `lap_number / total_laps` (e.g.
deciles) and check that `p_win` at "10% race distance" is appropriately
uncertain while `p_win` at "90% race distance" is appropriately sharp — a
single aggregate log-loss number hides whether the model is well-behaved
across the whole race, not just on average.

**Open question, deliberately left open**: `tyre_compound` is categorical
(`SOFT`/`MEDIUM`/`HARD`/`INTERMEDIATE`/`WET`) — XGBoost handles this via
either one-hot columns or native categorical support
(`enable_categorical=True`, available in the installed xgboost 3.1.2).
Native categorical is less code and the more natural choice for wet/dry
races where the compound signal matters a lot; the tradeoff is XGBoost's
categorical split handling being a comparatively newer feature. Decide at
implementation time, not blocking the design.

## 3. Live serving (OpenF1)

Verified directly against the real OpenF1 API (not assumed from docs):

- `GET /v1/sessions?session_key=latest` — the currently-live (or most
  recently completed) session. Confirmed live: returned the real 2026
  Dutch GP race session on 2026-08-23.
- `GET /v1/position?session_key={key}&driver_number={n}` — position over
  time. **Confirmed gotcha: the `limit` query param breaks this endpoint**
  (`{"detail": "No results found."}` with `limit` set, real data returned
  without it) — same for `/intervals` and `/race_control`. This must NOT
  be assumed to work like `/sessions`' `limit` param; fetch without it and
  slice client-side, or filter server-side with `date>=` instead.
- `GET /v1/intervals?session_key={key}` — `gap_to_leader` and `interval`
  (gap to car ahead) per `driver_number`, timestamped. Maps directly onto
  `gap_to_leader_s`/`gap_ahead_s`.
- `GET /v1/race_control?session_key={key}` — flag/category/message events
  (`category: "Flag"`, `flag: "GREEN"/"YELLOW"/...`) — confirmed real
  messages returned (e.g. "GREEN LIGHT - PIT EXIT OPEN"). Detecting
  safety-car state live: watch for `category` containing `SafetyCar` or
  `flag` transitions, mirroring the FastF1 `TrackStatus` codes the
  training corpus uses — these are two different vocabularies (OpenF1's
  race-control message categories vs. FastF1's numeric track-status
  codes) that both need to collapse onto the same `safety_car_active`
  boolean the model was trained on.
- `GET /v1/laps?session_key={key}&driver_number={n}&lap_number={n}` —
  per-lap sector times, confirmed working, gives the same pace-delta
  inputs as the training corpus's `pace_delta_last_3_laps`.
- Driver identity: OpenF1 uses `driver_number` (int), not FastF1's 3-letter
  code or jolpica's `driver_id` — a THIRD identity scheme. `GET
  /v1/drivers?session_key={key}` gives `driver_number -> name_acronym`
  (matches FastF1's code), so the existing `driver_code_to_id` map (built
  from jolpica) still bridges it: `driver_number -> name_acronym ->
  driver_id`.

**Polling design**: background poller in `api/main.py`'s FastAPI lifespan
hook. Every ~15-20s while `sessions?session_key=latest` indicates a session
is currently live (`date_start <= now <= date_end`), fetch `/intervals` +
`/position` + `/race_control` (3 calls), reshape into the same
`LIVE_FEATURE_COLUMNS` shape as training (deriving `tyre_age_laps`/
`pit_stops_so_far` from `/laps` + `/pit` similarly), call `predict_live`,
cache the result in-process. `GET /api/live/current` serves the cache;
`{"live": false}` when no session is active. 3 calls / ~17s ≈ 10.5 req/min,
comfortably under OpenF1's 30 req/min budget. No websockets — recompute is
cheap (a `.predict_proba` call), so plain polling on both ends (server →
OpenF1, frontend → server) is simple and sufficient, not a shortcut.

**Replay-mode test harness** (needed for demoability on a day with no live
session, and for local dev): feed the poller historical FastF1 laps at
accelerated speed instead of OpenF1 — same `LIVE_FEATURE_COLUMNS` shape,
different source. A `--replay season round speed_multiplier` flag on
whatever script drives the poller loop, not a separate code path through
`predict_live` itself.

## 4. What's genuinely uncertain (flagged, not resolved)

- **Whether OpenF1's 3s latency claim holds during an actual live
  session** — only verifiable by watching a real live race, which hasn't
  happened during this build. The polling interval (15-20s) has enough
  slack under the latency claim either way, but this is untested against
  a truly live session, only against historical replay.
- **Corpus build wall-clock time** — ~90 FastF1 session loads, each
  observed to take non-trivial real time even without telemetry; the full
  corpus build should run as a background job, not inline, and may need
  more than one sitting.
- **Safety-car vocabulary reconciliation** (OpenF1 message categories vs.
  FastF1 track-status codes) is a real integration detail that needs
  hands-on iteration against real race-control messages from a live or
  recent session, not something to get right from the docs alone.
