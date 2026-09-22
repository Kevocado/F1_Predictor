# Session predictors: sprint qualifying, qualifying, sprint, race

## Origin

Started as a spike: "does a predicted-qualifying feature improve the race
model's backtest?" Grew into an architectural change mid-brainstorm once
the actual ask surfaced — qualifying and sprint predictions as permanent,
user-facing predictions in their own right (shown before the race
prediction, tracked and reconciled honestly like race predictions already
are), not just an internal feature-ablation experiment. The original
question is still answered here (see "Feature-ablation experiment"), but
it's now one output of a bigger build, not the whole thing.

## Scope

Four session types get their own predictor, tracked prediction, and UI
section, in chronological weekend order:

1. **Sprint Qualifying** (sprint weekends only) — sets the sprint grid
2. **Sprint** (sprint weekends only)
3. **Qualifying** (every weekend) — sets the main-race grid
4. **Race** (every weekend) — already exists; extended, not rebuilt

Non-sprint weekends show just Qualifying → Race, same as today.

**Explicitly out of scope:** a live, in-session engine for qualifying or
sprint (the existing live in-race engine, Phase 3, stays race-only).
Qualifying/sprint predictions are pre-session only — snapshotted once
before the session starts, reconciled once after, same cadence the race
model already uses pre-race.

## Approach: shared modeling core, distinct endpoints, generalized tracking

Rejected alternatives, for the record:

- **Four separate, duplicated implementations** — simplest to reason
  about individually, but ~4x duplication of `race_outcome.py`'s
  ranker/Plackett-Luce simulation machinery and four near-identical SQL
  tables.
- **One fully generic model + one polymorphic endpoint** (e.g.
  `?session=sprint|race`) — maximum reuse, but forces a rewrite/migration
  of the existing, working race prediction endpoint and schema, and a
  polymorphic endpoint is a worse fit for a frontend that wants four
  distinct, clearly-labeled sections rather than one endpoint with a mode
  switch.

Chosen instead: reuse the modeling engine, keep the endpoints and UI
sections separate, generalize only the tracking table (which really is
the same shape four times over).

## 1. Modeling core & session/tier flow

New module `models/session_outcome.py`. `race_outcome.py`'s
`train_ranker`/`simulate_race` stay as-is — they're already
target-agnostic (the label comes from whichever `train_df` is passed in)
— wrapped by a `SessionSpec` per session type:

```python
SESSION_SPECS = {
    "sprint_qualifying": SessionSpec(
        target="sprint_quali_position", has_dnf=False, points_table=None,
        feature_cutoff_tier="post_practice",
    ),
    "qualifying": SessionSpec(
        target="quali_position", has_dnf=False, points_table=None,
        feature_cutoff_tier="post_sprint",   # non-sprint weekends: post_practice (see below)
    ),
    "sprint": SessionSpec(
        target="sprint_position", has_dnf=True, points_table=SPRINT_POINTS_TABLE,
        feature_cutoff_tier="post_sprint_qualifying",
    ),
    "race": SessionSpec(
        target="position", has_dnf=True, points_table=POINTS_TABLE,
        feature_cutoff_tier="post_qualifying",
    ),
}
```

`features/session_state.py`'s tier system extends from three tiers to a
wider chain covering sprint weekends' extra sessions:

```
pre_weekend -> post_practice -> [post_sprint_qualifying -> post_sprint ->] post_qualifying
```

The two bracketed tiers only apply on sprint weekends. Same
`tier_augment` masking mechanism already in place, just two more tiers
and two more tier-gated columns (`sprint_quali_position`,
`sprint_position`) unlocked progressively — no new mechanism.
`qualifying`'s cutoff tier is conditional on weekend type: `post_sprint`
on a sprint weekend (qualifying happens after the sprint), `post_practice`
on a non-sprint weekend (qualifying is the very next session after
practice) — `SessionSpec.feature_cutoff_tier` resolves per-race from
`is_sprint_weekend` rather than being a single fixed value for
`qualifying`.

`expected_position` (both here and in the tracking schema below) is
computed the same way the existing race model already computes it: the
mean finishing position across `simulate_race`'s Monte Carlo trials, not
a separately-derived rank.

Each session predictor trains the same way `backtest_race` retrains the
race model today: no-lookahead (only strictly-earlier sessions in the
training set), pointed at a different historical target column and
feature-availability cutoff per `SessionSpec`. `has_dnf=False` for both
qualifying session types — DNF isn't a real qualifying concept; a driver
who doesn't set a time just ranks last, which the ranker's label ordering
already handles.

## 2. Tracking schema

One new table, `session_predictions`, replacing `race_predictions`, with
an in-place migration that preserves every existing snapshot:

```sql
CREATE TABLE session_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    round INTEGER NOT NULL,
    race_name TEXT NOT NULL,
    session_type TEXT NOT NULL,   -- 'sprint_qualifying' | 'qualifying' | 'sprint' | 'race'
    driver_id TEXT NOT NULL,
    constructor_id TEXT,
    tier TEXT NOT NULL,
    market TEXT NOT NULL,
    predicted_prob REAL NOT NULL,
    expected_position REAL,        -- new
    session_time TEXT NOT NULL,
    snapshotted_at TEXT NOT NULL,
    model_trained_at TEXT,
    resolved INTEGER NOT NULL DEFAULT 0,
    actual_outcome INTEGER,
    actual_position INTEGER,       -- renamed from actual_finish_position
    actual_dnf INTEGER,
    resolved_at TEXT,
    backfilled INTEGER NOT NULL DEFAULT 0
);
```

**Markets per session type:**
- `sprint_qualifying` / `qualifying`: `pole`, `top_3`, `top_10` (Q3
  reach). No `dnf` market.
- `sprint` / `race`: `win`, `podium`, `points_finish`, `dnf`. Same market
  *names* across both, but `points_finish` means top-8 for sprint vs.
  top-10 for race (`SPRINT_POINTS_TABLE` vs. `POINTS_TABLE`, both already
  exist in `championship_projection.py`) — a threshold difference baked
  into how the market is computed per `SessionSpec.points_table`, not a
  schema difference.

**Migration:** existing `race_predictions` rows get `session_type='race'`
backfilled, `actual_finish_position` renamed to `actual_position`,
`expected_position` left `NULL` for old rows — that value was never
actually predicted at the time, so it isn't fabricated retroactively,
consistent with this app's honesty rule elsewhere (tracked vs. backtest
source labels, tier-honesty). One idempotent migration step in
`store.py`'s connection setup, same place `CREATE TABLE IF NOT EXISTS`
already lives. Test the migration against a copy of the real production
`tracking.db` first — assert row count and content are preserved
before/after — before it runs against the real one.

`store.py`'s functions get session-generic names:
`record_session_prediction`, `get_session_prediction(season, round,
session_type, tier)`, `get_session_accuracy(session_type=None,
tier=None)`, replacing the race-specific equivalents.

## 3. API surface

Four distinct endpoints under the existing `races/{season}/{round}/...`
path:

- `GET /api/races/{season}/{round}/sprint-qualifying-prediction`
- `GET /api/races/{season}/{round}/sprint-prediction`
- `GET /api/races/{season}/{round}/qualifying-prediction`
- `GET /api/races/{season}/{round}/prediction` (existing path, unchanged
  — main race)

The first two 404 cleanly for non-sprint weekends. One generalized
`SessionPredictionResponse` schema (not four separate ones) with
market fields that don't apply to a given session type simply omitted —
a qualifying response carries `p_pole/p_top3/p_top10 + expected_position`,
a race response carries `p_win/p_podium/p_points_finish/p_dnf +
expected_position`.

Backend: `routes.py`'s existing `_race_prediction_bundle`/
`_completed_race_prediction` split generalizes to
`_session_prediction_bundle(season, round_, session_type, ...)`.
`evaluate/backtest.py`'s `backtest_race` generalizes to
`backtest_session(season, round_, session_type)` — same no-lookahead
retrain discipline, pointed at a different target per `SessionSpec`. The
existing `_race_is_completed` honesty check (only "completed" once real
results exist, not merely once the scheduled time has passed) applies
per session type the same way.

Track record: `GET /api/track-record` and `/track-record/by-race` gain an
optional `session_type` query param (omitted = all types combined).

## 4. Frontend

Within a selected race's detail panel, a vertical session timeline in
chronological order: `Sprint Qualifying -> Sprint -> Qualifying -> Race`
on a sprint weekend, `Qualifying -> Race` otherwise. Each section reuses
`DriverPredictionTable`/`ProbabilityHeatCell`/`TierBadge`, generalized to
accept a `columns` config (which markets to render) instead of
hardcoding win/podium/points/dnf.

The next session to actually happen is expanded by default; resolved
sessions show the predicted-vs-actual delta (existing `ResultDelta.tsx`
pattern); sessions further out stay visible but collapsed — the whole
weekend's story top to bottom, not tabs requiring clicks. Track Record
page gains a session-type filter alongside the existing tier filter.

## 5. Testing & validation

**Directional sanity checks**, one per new session type, following
`backtest.py`'s existing pattern (actual winner elevated relative to
grid): actual pole-sitter should rank near the top of predicted
qualifying order; actual sprint winner near the top of predicted sprint
order; same for sprint qualifying.

**Calibration:** a generalized walk-forward replay (`backtest_session`)
computes Brier score per market, per session type, across historical
seasons — same metric the race model is already judged on.

**Test suite:** proportionate to the project's current test coverage
(`test_auth.py`, `test_race_prediction_caching.py` — thin, not a large
suite to match). Additions: `test_session_outcome.py` (each `SessionSpec`
trains and produces sane, non-degenerate probabilities), a migration test
for `session_predictions` against a copy of real `tracking.db`, and
endpoint-shape tests for the three new routes.

**Model artifacts/manifest:** `manifest.py::train_all()` extends to
train and save all four session predictors (new paths alongside
`RACE_OUTCOME_MODEL_PATH`/`DNF_MODEL_PATH`: `SPRINT_QUALIFYING_MODEL_PATH`,
`QUALIFYING_MODEL_PATH`, `SPRINT_MODEL_PATH`), each with its own
candidate/metrics recorded in `manifest.json`.

## 6. Feature-ablation experiment (the original spike question)

Now that the qualifying predictor is permanent, the original question —
"does a predicted-qualifying feature improve the race model?" — becomes
a real, re-runnable evaluation script, `evaluate/quali_feature_ablation.py`,
following the precedent `tune_hyperparams.py`/`tune_live_hyperparams.py`
already set:

1. Train the race model twice via walk-forward: with and without a new
   `predicted_quali_position` feature merged into PRE_WEEKEND/
   POST_PRACTICE tier rows (never overriding the real value once
   POST_QUALIFYING data actually arrives).
2. Compare Brier score across win/podium/points_finish/dnf, specifically
   on PRE_WEEKEND/POST_PRACTICE tier rows (POST_QUALIFYING rows already
   have real qualifying data and gain nothing from a predicted proxy).
3. If it wins: add `predicted_quali_position` to
   `build_features.FEATURE_COLUMNS` permanently, gated the same
   tier-aware NaN-masking way everything else in that tier system already
   works.
4. If it doesn't: no change to the race model's feature set — qualifying
   predictions still ship as their own feature regardless, since that was
   confirmed as wanted independent of this experiment's outcome.

## Open items for the implementation plan

- Exact current-F1 sprint-weekend session ordering/typical datetimes to
  encode in `session_state.py`'s tier-transition logic (FP1 -> Sprint
  Qualifying -> Sprint -> Qualifying -> Race) — verify against
  `jolpica.fetch_season_schedule`'s actual datetime columns for a real
  sprint weekend before wiring tier transitions.
- Whether `championship_projection.py`'s current simplified sprint
  handling (reusing the main race's theta/DNF vectors for sprint, per its
  existing docstring) should be upgraded to use the new dedicated sprint
  predictor's output — not required by this spec, but a natural
  follow-up once the sprint model exists.
