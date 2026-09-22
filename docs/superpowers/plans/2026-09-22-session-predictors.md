# Session Predictors (Sprint Qualifying / Qualifying / Sprint / Race) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new session predictors (Sprint Qualifying, Qualifying, Sprint) alongside the existing Race predictor, each with its own trained model, tracked/reconciled prediction history, API endpoint, and frontend section — shown as a chronological weekend timeline — plus an evaluation script answering whether a predicted-qualifying feature improves the existing race model.

**Architecture:** One shared modeling engine (`models/race_outcome.py`'s `train_ranker`/`simulate_race`, both already target-agnostic) wrapped by a per-session-type `SessionSpec`. Four distinct API endpoints and frontend sections (not one polymorphic endpoint), but one generalized SQLite tracking table replacing the race-only one. The existing race prediction path (`features/build.py`, `evaluate/backtest.py::backtest_race`, `models/manifest.py`'s race/DNF training) is extended, not rewritten — new code lives in new modules that reuse it.

**Tech Stack:** Python 3.13, FastAPI, XGBoost, pandas, SQLite (stdlib `sqlite3`), React + TypeScript + Vite + Tailwind.

**Spec:** `docs/superpowers/specs/2026-09-22-session-predictors-design.md`

## Global Constraints

- No live in-session engine for qualifying/sprint/sprint-qualifying — pre-session prediction only, snapshotted once before the session, reconciled once after (spec §"Explicitly out of scope").
- The three new session types (`sprint_qualifying`, `qualifying`, `sprint`) always use the `xgb_ranker` candidate directly — **no elo-vs-xgb_ranker race-off** for them (unlike the race model). This is a deliberate scope reduction from the spec's silence on this point: the race model's Elo candidate exists because Elo's pairwise formula is race-domain-specific (see `race_outcome.py::theta_from_elo_strength`'s docstring); replicating the full walk-forward race-off for three more session types is unrequested extra complexity. Flag this to the user as a plan-time refinement, not a spec violation.
- **Data-source deviation from the spec:** jolpica has no sprint-qualifying results endpoint (confirmed directly: `GET /2024/5/sprint-qualifying.json` returns `400 Bad Request`). Sprint qualifying's target/training data instead comes from jolpica's existing `sprint.json` `grid` column (a sprint race's starting grid IS its sprint-qualifying classification, barring penalties) — the same caveat-tolerant precedent this codebase already uses for the race model's own `grid` feature as a qualifying-position proxy (see `routes.py::_future_feature_frame`'s comment). No new external data source is added.
- **No PUBLIC_MODE snapshot integration in this plan.** The three new endpoints always live-compute (via the existing `_cached` in-process TTL cache), skipping the `_public_snapshot()`/`public_snapshot.py` precompute path the existing race endpoint uses. Note this explicitly if picked up later — out of scope here, matching the spec's own deferral of `championship_projection.py` sprint integration.
- Each of the three new session types has exactly **one fixed feature-availability point** (not a multi-tier progressively-sharpening series like the race model's `pre_weekend -> post_practice -> post_qualifying`). `SessionSpec.feature_cutoff_tier` is a label for the API/tracking `tier` field, not a masking mechanism — no `tier_augment`-style row multiplication for the new session types.
- Existing race model files (`features/build.py`, `features/session_state.py`'s existing tier constants/`tier_augment`/`current_tier`, `evaluate/backtest.py::backtest_race`, `models/manifest.py`'s race/DNF training block) are **additive-only** — every existing test and the existing `/api/races/{season}/{round}/prediction` endpoint's behavior must be unchanged after this plan.

---

### Task 1: Extend `features/session_state.py` with sprint-weekend session tiers

**Files:**
- Modify: `src/f1_predictor/features/session_state.py`
- Test: `tests/test_session_state.py` (new file)

**Interfaces:**
- Consumes: nothing new (pure additions to an existing module).
- Produces: `TIER_POST_SPRINT_QUALIFYING = "post_sprint_qualifying"`, `TIER_POST_SPRINT = "post_sprint"` (module-level constants), `SESSION_TIER_ORDER: list[str]`, `current_session_tier(schedule_row: pd.Series, now: pd.Timestamp | None = None) -> str`. Later tasks import these from `f1_predictor.features.session_state`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session_state.py
import pandas as pd

from f1_predictor.features import session_state


def _row(**overrides) -> pd.Series:
    base = {
        "fp1_datetime": pd.Timestamp("2024-04-19T03:30:00Z"),
        "fp2_datetime": None,
        "fp3_datetime": None,
        "sprint_quali_datetime": pd.Timestamp("2024-04-19T07:30:00Z"),
        "sprint_datetime": pd.Timestamp("2024-04-20T03:00:00Z"),
        "qualifying_datetime": pd.Timestamp("2024-04-20T07:00:00Z"),
        "race_datetime": pd.Timestamp("2024-04-21T00:00:00Z"),
    }
    base.update(overrides)
    return pd.Series(base)


def test_current_session_tier_pre_weekend_before_any_session():
    now = pd.Timestamp("2024-04-19T00:00:00Z")
    assert session_state.current_session_tier(_row(), now) == session_state.TIER_PRE_WEEKEND


def test_current_session_tier_post_practice_after_fp1_before_sprint_quali():
    now = pd.Timestamp("2024-04-19T05:00:00Z")
    assert session_state.current_session_tier(_row(), now) == session_state.TIER_POST_PRACTICE


def test_current_session_tier_post_sprint_qualifying_after_sprint_quali_before_sprint():
    now = pd.Timestamp("2024-04-19T08:00:00Z")
    assert session_state.current_session_tier(_row(), now) == session_state.TIER_POST_SPRINT_QUALIFYING


def test_current_session_tier_post_sprint_after_sprint_before_qualifying():
    now = pd.Timestamp("2024-04-20T04:00:00Z")
    assert session_state.current_session_tier(_row(), now) == session_state.TIER_POST_SPRINT


def test_current_session_tier_post_qualifying_after_qualifying():
    now = pd.Timestamp("2024-04-20T08:00:00Z")
    assert session_state.current_session_tier(_row(), now) == session_state.TIER_POST_QUALIFYING


def test_current_session_tier_non_sprint_weekend_skips_sprint_tiers():
    # A regular weekend has no sprint_quali_datetime/sprint_datetime at all.
    row = _row(sprint_quali_datetime=None, sprint_datetime=None)
    now = pd.Timestamp("2024-04-19T05:00:00Z")
    assert session_state.current_session_tier(row, now) == session_state.TIER_POST_PRACTICE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_session_state.py -v`
Expected: FAIL with `AttributeError: module 'f1_predictor.features.session_state' has no attribute 'current_session_tier'` (and `TIER_POST_SPRINT_QUALIFYING`).

- [ ] **Step 3: Add the new tiers and function**

Add this to `src/f1_predictor/features/session_state.py`, directly below the existing `TIER_ORDER = [...]` line (do not modify `TIER_ORDER`, `TIER_COLUMNS`, `tier_augment`, `columns_known_by`, `mask_to_tier`, or the existing `current_tier` — they stay exactly as they are, used only by the race model):

```python
# Wider tier chain for the three new session predictors (sprint_qualifying/
# qualifying/sprint) — separate from TIER_ORDER above, which is the race
# model's own 3-tier list and is untouched by this addition. Each of the
# three new session types has exactly one fixed feature-availability point
# (see models/session_outcome.py::SessionSpec), so this list is used only
# for the "what tier is this weekend in right now" label, never for
# tier_augment-style row masking.
TIER_POST_SPRINT_QUALIFYING = "post_sprint_qualifying"
TIER_POST_SPRINT = "post_sprint"

SESSION_TIER_ORDER = [
    TIER_PRE_WEEKEND,
    TIER_POST_PRACTICE,
    TIER_POST_SPRINT_QUALIFYING,
    TIER_POST_SPRINT,
    TIER_POST_QUALIFYING,
]


def current_session_tier(schedule_row: pd.Series, now: pd.Timestamp | None = None) -> str:
    """Like current_tier, but resolves the wider SESSION_TIER_ORDER chain —
    distinguishes "sprint qualifying has happened" and "the sprint has
    happened" as their own tiers, which current_tier (race-model-only)
    deliberately doesn't. On a non-sprint weekend, sprint_quali_datetime/
    sprint_datetime are both None, so this collapses to exactly current_tier's
    3-tier behavior."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")

    quali_dt = schedule_row.get("qualifying_datetime")
    if pd.notna(quali_dt) and now >= quali_dt:
        return TIER_POST_QUALIFYING

    sprint_dt = schedule_row.get("sprint_datetime")
    if pd.notna(sprint_dt) and now >= sprint_dt:
        return TIER_POST_SPRINT

    sprint_quali_dt = schedule_row.get("sprint_quali_datetime")
    if pd.notna(sprint_quali_dt) and now >= sprint_quali_dt:
        return TIER_POST_SPRINT_QUALIFYING

    practice_cols = ["fp1_datetime", "fp2_datetime", "fp3_datetime"]
    practice_dts = [schedule_row.get(c) for c in practice_cols if pd.notna(schedule_row.get(c))]
    if practice_dts and now >= min(practice_dts):
        return TIER_POST_PRACTICE

    return TIER_PRE_WEEKEND
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_session_state.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS (all existing tests plus the 6 new ones)

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/features/session_state.py tests/test_session_state.py
git commit -m "$(cat <<'EOF'
Add sprint-weekend session tiers (post_sprint_qualifying, post_sprint)

Extends session_state.py with a wider SESSION_TIER_ORDER and
current_session_tier() for the upcoming sprint qualifying/sprint/
qualifying predictors, without touching the race model's existing
3-tier TIER_ORDER/tier_augment/current_tier.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Generalize `models/race_outcome.py::simulate_race` for configurable points scoring

**Files:**
- Modify: `src/f1_predictor/models/race_outcome.py:136-182`
- Test: `tests/test_race_outcome_simulate.py` (new file)

**Interfaces:**
- Consumes: nothing new.
- Produces: `simulate_race(theta, dnf_prob=None, n_trials=10000, seed=0, points_table=None, points_finish_cutoff=10) -> pd.DataFrame` — same return shape as before (`driver_id, p_win, p_podium, p_points_finish, p_dnf, expected_position, expected_points`). Existing callers that don't pass the two new kwargs get byte-identical behavior to before this task.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_race_outcome_simulate.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_race_outcome_simulate.py -v`
Expected: FAIL with `TypeError: simulate_race() got an unexpected keyword argument 'points_table'`

- [ ] **Step 3: Generalize `simulate_race`**

In `src/f1_predictor/models/race_outcome.py`, replace the `simulate_race` function signature and body:

```python
def simulate_race(
    theta: dict[str, float],
    dnf_prob: dict[str, float] | None = None,
    n_trials: int = 10000,
    seed: int = 0,
    points_table: dict[int, float] | None = None,
    points_finish_cutoff: int = 10,
) -> pd.DataFrame:
    """Runs `draw_classification` n_trials times and aggregates into one
    row per driver: p_win, p_podium, p_points_finish (top
    `points_finish_cutoff`), p_dnf, expected_position, expected_points —
    requirement 1's four targets (DNF reported both here and standalone
    from models/dnf.py directly). `points_table`/`points_finish_cutoff`
    default to the main race's values (POINTS_TABLE, top 10) — pass
    session_outcome.py's SPRINT_POINTS_TABLE/cutoff=8 for a sprint, or
    leave both at their default for a session with no real points concept
    (qualifying-type sessions still get a `points_finish`-shaped market at
    the default top-10 cutoff, reinterpreted by the caller as "reached
    Q3"/"top_10" rather than literal points)."""
    points_table = points_table or POINTS_TABLE
    dnf_prob = dnf_prob or {}
    rng = np.random.default_rng(seed)
    drivers = list(theta.keys())
    win = {d: 0 for d in drivers}
    podium = {d: 0 for d in drivers}
    points_finish = {d: 0 for d in drivers}
    dnf_ct = {d: 0 for d in drivers}
    pos_sum = {d: 0 for d in drivers}
    points_sum = {d: 0.0 for d in drivers}

    for _ in range(n_trials):
        order, dnfd = draw_classification(theta, dnf_prob, rng)
        for pos, d in enumerate(order, start=1):
            pos_sum[d] += pos
            if pos == 1:
                win[d] += 1
            if pos <= 3:
                podium[d] += 1
            if pos <= points_finish_cutoff:
                points_finish[d] += 1
                points_sum[d] += points_table.get(pos, 0.0)
        for d in dnfd:
            dnf_ct[d] += 1

    rows = [
        {
            "driver_id": d,
            "p_win": win[d] / n_trials,
            "p_podium": podium[d] / n_trials,
            "p_points_finish": points_finish[d] / n_trials,
            "p_dnf": dnf_ct[d] / n_trials,
            "expected_position": pos_sum[d] / n_trials,
            "expected_points": points_sum[d] / n_trials,
        }
        for d in drivers
    ]
    return pd.DataFrame(rows).sort_values("p_win", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_race_outcome_simulate.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS (every existing test — `simulate_race`'s default behavior must be byte-identical to before this task)

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/models/race_outcome.py tests/test_race_outcome_simulate.py
git commit -m "$(cat <<'EOF'
Generalize simulate_race for configurable points table/cutoff

Backward-compatible: default args reproduce the exact prior race-model
behavior. Needed so session_outcome.py can reuse this simulator for
sprint (top-8 points) and qualifying-type (no points, top-10 "reached
Q3" market) sessions without duplicating the Monte Carlo loop.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add `models/session_outcome.py` — the shared session-type engine

**Files:**
- Create: `src/f1_predictor/models/session_outcome.py`
- Test: `tests/test_session_outcome.py` (new file)

**Interfaces:**
- Consumes: `race_outcome.train_ranker`, `race_outcome.xgb_scores_for_race`, `race_outcome.theta_from_xgb_scores`, `race_outcome.simulate_race` (Task 2's new signature), `race_outcome.POINTS_TABLE`; `dnf.train_dnf_model`, `dnf.predict_dnf_prob`; `championship_projection.SPRINT_POINTS_TABLE` (existing, `src/f1_predictor/models/championship_projection.py:46`).
- Produces: `SessionSpec` (frozen dataclass with fields `session_type: str`, `target_column: str`, `has_dnf: bool`, `points_table: dict[int, float] | None`, `points_finish_cutoff: int | None`, `feature_cutoff_tier: str`), `SESSION_SPECS: dict[str, SessionSpec]` (keys: `"sprint_qualifying"`, `"qualifying"`, `"sprint"` — deliberately NOT `"race"`, which stays on the existing `race_outcome.py`/`build.py` path), `train_session_ranker(train_df, feature_cols, spec, hyperparams=None) -> xgb.XGBRanker`, `predict_session(ranker, session_df, feature_cols, spec, dnf_clf=None, n_trials=10000, seed=0) -> pd.DataFrame` (same row shape as `race_outcome.simulate_race`'s return, plus a `driver_id`-indexable frame). Later tasks (4, 5, 7, 9) import `SESSION_SPECS`, `train_session_ranker`, `predict_session` from this module.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session_outcome.py
import pandas as pd

from f1_predictor.models import session_outcome


def _toy_frame(target_col: str, n_rounds: int = 3) -> pd.DataFrame:
    rows = []
    for rnd in range(1, n_rounds + 1):
        for i, driver in enumerate(["a", "b", "c", "d"], start=1):
            rows.append(
                {
                    "season": 2024,
                    "round": rnd,
                    "driver_id": driver,
                    "constructor_id": f"team_{i % 2}",
                    target_col: i,  # driver "a" always finishes 1st, etc.
                    "elo_pre_race": 1600 - i * 10,
                    "team_strength_pre_race": 1500.0,
                }
            )
    return pd.DataFrame(rows)


def test_session_specs_has_three_non_race_types():
    assert set(session_outcome.SESSION_SPECS.keys()) == {"sprint_qualifying", "qualifying", "sprint"}


def test_session_specs_qualifying_has_no_dnf():
    spec = session_outcome.SESSION_SPECS["qualifying"]
    assert spec.has_dnf is False
    assert spec.target_column == "quali_position"


def test_session_specs_sprint_uses_sprint_points_table():
    from f1_predictor.models.championship_projection import SPRINT_POINTS_TABLE

    spec = session_outcome.SESSION_SPECS["sprint"]
    assert spec.has_dnf is True
    assert spec.points_table == SPRINT_POINTS_TABLE
    assert spec.points_finish_cutoff == 8


def test_train_session_ranker_and_predict_session_roundtrip():
    spec = session_outcome.SESSION_SPECS["qualifying"]
    feature_cols = ["elo_pre_race", "team_strength_pre_race"]
    df = _toy_frame(spec.target_column)

    ranker = session_outcome.train_session_ranker(df, feature_cols, spec)
    session_df = df[df["round"] == 3].reset_index(drop=True)
    sim = session_outcome.predict_session(ranker, session_df, feature_cols, spec, n_trials=500, seed=0)

    assert set(sim["driver_id"]) == {"a", "b", "c", "d"}
    assert (sim["p_dnf"] == 0.0).all(), "qualifying-type sessions have no DNF concept"
    # driver "a" has the strongest elo and always qualifies 1st in the toy
    # data — its win probability should be clearly the highest of the four.
    top = sim.sort_values("p_win", ascending=False).iloc[0]
    assert top["driver_id"] == "a"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_session_outcome.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'f1_predictor.models.session_outcome'`

- [ ] **Step 3: Write `session_outcome.py`**

```python
# src/f1_predictor/models/session_outcome.py
"""session_outcome.py — the shared engine for the three non-race session
predictors (sprint qualifying, qualifying, sprint), reusing
race_outcome.py's train_ranker/simulate_race (already target-agnostic —
the label comes from whichever DataFrame column is passed in) instead of
duplicating the ranker/Plackett-Luce machinery per session type. See
docs/superpowers/specs/2026-09-22-session-predictors-design.md §1.

Deliberately does NOT cover "race" — the existing race model
(features/build.py, evaluate/backtest.py::backtest_race,
models/manifest.py's race/DNF training block) stays on its own path
unchanged; this module only adds the three new session types.

Each session type has exactly ONE fixed feature-availability point
(feature_cutoff_tier is a label for the API/tracking `tier` field, not a
tier_augment-style masking mechanism) — unlike the race model's
progressively-sharpening 3-tier chain.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import xgboost as xgb

from ..features import session_state
from . import dnf as dnf_model
from . import race_outcome
from .championship_projection import SPRINT_POINTS_TABLE


@dataclass(frozen=True)
class SessionSpec:
    session_type: str
    target_column: str
    has_dnf: bool
    points_table: dict[int, float] | None
    points_finish_cutoff: int | None
    feature_cutoff_tier: str


SESSION_SPECS: dict[str, SessionSpec] = {
    "sprint_qualifying": SessionSpec(
        session_type="sprint_qualifying",
        target_column="sprint_quali_position",
        has_dnf=False,
        points_table=None,
        points_finish_cutoff=None,
        feature_cutoff_tier=session_state.TIER_POST_PRACTICE,
    ),
    "qualifying": SessionSpec(
        session_type="qualifying",
        target_column="quali_position",
        has_dnf=False,
        points_table=None,
        points_finish_cutoff=None,
        feature_cutoff_tier=session_state.TIER_POST_PRACTICE,
    ),
    "sprint": SessionSpec(
        session_type="sprint",
        target_column="position",
        has_dnf=True,
        points_table=SPRINT_POINTS_TABLE,
        points_finish_cutoff=8,
        feature_cutoff_tier=session_state.TIER_POST_SPRINT_QUALIFYING,
    ),
}


def train_session_ranker(
    train_df: pd.DataFrame, feature_cols: list[str], spec: SessionSpec, hyperparams: dict | None = None
) -> xgb.XGBRanker:
    """Delegates to race_outcome.train_ranker, which groups by
    (season, round, tier) and ranks on a column literally named
    "position" — renaming spec.target_column to "position" and stamping a
    constant "tier" column (spec.feature_cutoff_tier; there's only one
    tier per session type here, so the groupby still groups correctly by
    (season, round)) lets that function run completely unmodified."""
    df = train_df.copy()
    df["position"] = df[spec.target_column]
    df["tier"] = spec.feature_cutoff_tier
    return race_outcome.train_ranker(df, feature_cols, hyperparams=hyperparams)


def predict_session(
    ranker: xgb.XGBRanker,
    session_df: pd.DataFrame,
    feature_cols: list[str],
    spec: SessionSpec,
    dnf_clf: xgb.XGBClassifier | None = None,
    n_trials: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    """Same Monte Carlo output shape as race_outcome.simulate_race
    (driver_id, p_win, p_podium, p_points_finish, p_dnf,
    expected_position, expected_points). For has_dnf=False session types,
    dnf_clf is ignored even if passed, and p_dnf is exactly 0.0 for every
    driver (empty dnf_prob dict -> draw_classification never marks anyone
    DNF'd) — callers reinterpret p_win/p_podium/p_points_finish as
    pole/top_3/top_10 for those session types; they don't surface p_dnf at
    all in that case."""
    scores = race_outcome.xgb_scores_for_race(ranker, session_df, feature_cols)
    theta = race_outcome.theta_from_xgb_scores(scores)

    dnf_prob: dict[str, float] = {}
    if spec.has_dnf and dnf_clf is not None:
        dnf_prob = dnf_model.predict_dnf_prob(dnf_clf, session_df, feature_cols)

    return race_outcome.simulate_race(
        theta,
        dnf_prob=dnf_prob,
        n_trials=n_trials,
        seed=seed,
        points_table=spec.points_table,
        points_finish_cutoff=spec.points_finish_cutoff or 10,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_session_outcome.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/models/session_outcome.py tests/test_session_outcome.py
git commit -m "$(cat <<'EOF'
Add models/session_outcome.py: shared engine for the 3 new session types

SessionSpec + SESSION_SPECS (sprint_qualifying, qualifying, sprint) wrap
race_outcome.py's existing train_ranker/simulate_race without modifying
them — the race model's own path is untouched.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Add `features/session_build.py` — training-frame builder for the 3 new session types

**Files:**
- Create: `src/f1_predictor/features/session_build.py`
- Test: `tests/test_session_build.py` (new file)

**Interfaces:**
- Consumes: `jolpica.default_seasons`, `jolpica.load_multi_season_results`, `jolpica.fetch_season_schedule`, `jolpica.load_season_sprints` (existing, `src/f1_predictor/data/jolpica.py:286`, aggregates `fetch_sprint_results` per completed sprint weekend), `jolpica.load_season_qualifying` (existing, `:262`); `elo.compute_elo_history`, `team_strength.compute_team_strength_history`, `team_strength.compute_team_rolling_form`, `rolling_form.compute_rolling_form`, `circuit.compute_circuit_history`, `safety_car.compute_circuit_dnf_rate`, `weather.compute_weather_features` (all existing, same functions `features/build.py::build_training_frame` already calls).
- Produces: `BASE_SESSION_FEATURES: list[str]`, `SESSION_FEATURE_COLUMNS: dict[str, list[str]]` (keys `"sprint_qualifying"`, `"qualifying"`, `"sprint"`), `build_session_training_frame(session_type: str, seasons: list[int] | None = None) -> tuple[pd.DataFrame, list[str]]`. Later tasks (5, 7, 9) call `build_session_training_frame` and import `SESSION_FEATURE_COLUMNS`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session_build.py
import pandas as pd

from f1_predictor.features import session_build


def _fake_results(season=2024, rounds=(1, 2, 3)):
    rows = []
    for rnd in rounds:
        for i, driver in enumerate(["a", "b", "c"], start=1):
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "driver_id": driver,
                    "driver_code": driver.upper(),
                    "constructor_id": f"team_{i % 2}",
                    "grid": i,
                    "position": i,
                    "points": 0.0,
                    "laps": 50,
                    "status": "Finished",
                    "dnf": False,
                    "fastest_lap_rank": None,
                }
            )
    return pd.DataFrame(rows)


def _fake_schedule(season=2024, rounds=(1, 2, 3)):
    rows = []
    for rnd in rounds:
        rows.append(
            {
                "season": season,
                "round": rnd,
                "race_name": f"Round {rnd}",
                "circuit_id": "fake_circuit",
                "circuit_name": "Fake Circuit",
                "lat": 0.0,
                "long": 0.0,
                "locality": "Nowhere",
                "country": "Nowhere",
                "race_datetime": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=rnd),
                "is_sprint_weekend": False,
            }
        )
    return pd.DataFrame(rows)


def test_build_session_training_frame_qualifying_excludes_target_leakage(monkeypatch):
    results = _fake_results()
    schedule = _fake_schedule()
    quali = results.rename(columns={"position": "quali_position"}).drop(columns=["grid", "dnf", "status"])

    monkeypatch.setattr(session_build.jolpica, "load_multi_season_results", lambda seasons: results)
    monkeypatch.setattr(session_build.jolpica, "fetch_season_schedule", lambda season: schedule)
    monkeypatch.setattr(session_build.jolpica, "load_season_qualifying", lambda season: quali)
    monkeypatch.setattr(session_build.jolpica, "load_season_sprints", lambda season: pd.DataFrame())
    monkeypatch.setattr(
        session_build.weather_features, "compute_weather_features", lambda schedule_df, **kw: pd.DataFrame(
            columns=["season", "round", "temp_max_c", "precipitation_mm", "wind_max_kph"]
        )
    )

    df, feature_cols = session_build.build_session_training_frame("qualifying", seasons=[2024])

    assert "quali_position" not in feature_cols, "target column must never appear in feature_cols"
    assert "sprint_finish_position" in feature_cols
    assert len(df) == 9  # 3 rounds x 3 drivers
    assert df["sprint_finish_position"].isna().all(), "no sprint data faked -> all NaN, not fabricated"


def test_build_session_training_frame_sprint_uses_grid_as_sprint_quali_position(monkeypatch):
    sprints = _fake_results()
    schedule = _fake_schedule()

    monkeypatch.setattr(session_build.jolpica, "load_multi_season_results", lambda seasons: _fake_results())
    monkeypatch.setattr(session_build.jolpica, "fetch_season_schedule", lambda season: schedule)
    monkeypatch.setattr(session_build.jolpica, "load_season_sprints", lambda season: sprints)
    monkeypatch.setattr(
        session_build.weather_features, "compute_weather_features", lambda schedule_df, **kw: pd.DataFrame(
            columns=["season", "round", "temp_max_c", "precipitation_mm", "wind_max_kph"]
        )
    )

    df, feature_cols = session_build.build_session_training_frame("sprint", seasons=[2024])

    assert "sprint_quali_position" in feature_cols
    assert (df["sprint_quali_position"] == df["grid"]).all()
    assert "position" in df.columns  # the sprint's own finishing position, the target
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_session_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'f1_predictor.features.session_build'`

- [ ] **Step 3: Write `session_build.py`**

```python
# src/f1_predictor/features/session_build.py
"""session_build.py — training-frame builders for the three new session
predictors (sprint_qualifying, qualifying, sprint), sibling to
features/build.py's build_training_frame (which stays the race model's
own, unmodified builder). Reuses the exact same performance-history
feature computations build_training_frame does (Elo, team strength,
rolling form, circuit history, DNF rate, weather) — those are computed
from RACE results regardless of which session is being predicted, since
they're the underlying driver/constructor performance signal, not
session-specific.

Deliberately NOT tier_augmented (unlike build_training_frame): each of
these three session types has exactly one fixed feature-availability
point (models/session_outcome.py::SessionSpec.feature_cutoff_tier), so
there's only one training row per (season, round, driver) to build, not
one per tier.

Data-source note: jolpica has no sprint-qualifying results endpoint
(confirmed directly: GET /{season}/{round}/sprint-qualifying.json ->
400 Bad Request). Sprint qualifying's target comes from jolpica's
sprint.json `grid` column instead — a sprint's starting grid IS its
sprint-qualifying classification barring penalties, the same
caveat-tolerant proxy this codebase already accepts for the race model's
own `grid` feature (see api/routes.py::_future_feature_frame).
"""

from __future__ import annotations

import pandas as pd

from ..data import jolpica
from . import circuit, elo, rolling_form, safety_car, team_strength
from . import weather as weather_features

BASE_SESSION_FEATURES = [
    "elo_pre_race",
    "team_strength_pre_race",
    "team_form_avg_position_3",
    "team_form_points_3",
    "form_avg_position_3",
    "form_avg_points_3",
    "form_dnf_rate_3",
    "form_avg_position_5",
    "form_avg_points_5",
    "form_dnf_rate_5",
    "form_avg_position_10",
    "form_avg_points_10",
    "form_dnf_rate_10",
    "driver_circuit_avg_position",
    "driver_circuit_avg_points",
    "constructor_circuit_avg_position",
    "circuit_dnf_rate",
    "temp_max_c",
    "precipitation_mm",
    "wind_max_kph",
]

SESSION_FEATURE_COLUMNS: dict[str, list[str]] = {
    "sprint_qualifying": list(BASE_SESSION_FEATURES),
    "qualifying": list(BASE_SESSION_FEATURES) + ["sprint_finish_position"],
    "sprint": list(BASE_SESSION_FEATURES) + ["sprint_quali_position"],
}

_VALID_SESSION_TYPES = set(SESSION_FEATURE_COLUMNS.keys())


def build_session_training_frame(
    session_type: str, seasons: list[int] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    if session_type not in _VALID_SESSION_TYPES:
        raise ValueError(
            f"build_session_training_frame does not support session_type={session_type!r} "
            f"(valid: {sorted(_VALID_SESSION_TYPES)}) — 'race' uses features/build.py directly"
        )
    seasons = seasons or jolpica.default_seasons()
    feature_cols = SESSION_FEATURE_COLUMNS[session_type]

    results_df = jolpica.load_multi_season_results(seasons)
    schedule_df = pd.concat([jolpica.fetch_season_schedule(s) for s in seasons], ignore_index=True)
    sprint_results_df = pd.concat([jolpica.load_season_sprints(s) for s in seasons], ignore_index=True)

    if session_type == "sprint_qualifying":
        if sprint_results_df.empty:
            target_df = pd.DataFrame(columns=["season", "round", "driver_id", "constructor_id", "sprint_quali_position"])
        else:
            target_df = sprint_results_df[["season", "round", "driver_id", "constructor_id", "grid"]].rename(
                columns={"grid": "sprint_quali_position"}
            )
    elif session_type == "qualifying":
        target_df = pd.concat([jolpica.load_season_qualifying(s) for s in seasons], ignore_index=True)
    else:  # "sprint"
        if sprint_results_df.empty:
            target_df = pd.DataFrame(
                columns=["season", "round", "driver_id", "constructor_id", "position", "dnf", "grid"]
            )
        else:
            target_df = sprint_results_df[
                ["season", "round", "driver_id", "constructor_id", "position", "dnf", "grid"]
            ].copy()

    if target_df.empty:
        return pd.DataFrame(columns=["season", "round", "driver_id", "constructor_id"] + feature_cols), feature_cols

    elo_hist = elo.compute_elo_history(results_df)
    team_hist = team_strength.compute_team_strength_history(results_df)
    team_form_hist = team_strength.compute_team_rolling_form(results_df)
    form_hist = rolling_form.compute_rolling_form(results_df)
    circuit_hist = circuit.compute_circuit_history(results_df, schedule_df)
    dnf_rate_hist = safety_car.compute_circuit_dnf_rate(results_df, schedule_df)
    weather_feat = weather_features.compute_weather_features(schedule_df)

    df = target_df.merge(elo_hist, on=["season", "round", "driver_id"], how="left")
    df = df.merge(team_hist, on=["season", "round", "constructor_id"], how="left")
    df = df.merge(team_form_hist, on=["season", "round", "constructor_id"], how="left")
    df = df.merge(form_hist, on=["season", "round", "driver_id"], how="left")
    df = df.merge(circuit_hist.drop(columns=["constructor_id"]), on=["season", "round", "driver_id"], how="left")
    df = df.merge(dnf_rate_hist.drop(columns=["circuit_id"]), on=["season", "round"], how="left")
    df = df.merge(weather_feat, on=["season", "round"], how="left")

    if session_type == "qualifying":
        if sprint_results_df.empty:
            df["sprint_finish_position"] = float("nan")
        else:
            sprint_feat = sprint_results_df[["season", "round", "driver_id", "position"]].rename(
                columns={"position": "sprint_finish_position"}
            )
            df = df.merge(sprint_feat, on=["season", "round", "driver_id"], how="left")
    elif session_type == "sprint":
        df["sprint_quali_position"] = df["grid"]

    return df, feature_cols
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_session_build.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/features/session_build.py tests/test_session_build.py
git commit -m "$(cat <<'EOF'
Add features/session_build.py: training frames for the 3 new session types

Reuses the same Elo/team-strength/form/circuit/weather history
build_training_frame already computes, merged onto each session's own
target frame with no leakage of that session's own result into its
features. Sprint qualifying's target comes from jolpica's sprint.json
grid column — jolpica has no dedicated sprint-qualifying endpoint.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Generalize `evaluate/backtest.py` with `backtest_session`

**Files:**
- Modify: `src/f1_predictor/evaluate/backtest.py`
- Test: `tests/test_backtest_session.py` (new file)

**Interfaces:**
- Consumes: `session_build.build_session_training_frame` (Task 4), `session_outcome.SESSION_SPECS`, `session_outcome.train_session_ranker`, `session_outcome.predict_session` (Task 3), existing `backtest_race` (unchanged).
- Produces: `backtest_session(season: int, round_: int, session_type: str = "race", seasons: list[int] | None = None) -> pd.DataFrame`. For `session_type="race"` this is a pure passthrough to the existing `backtest_race`. Later tasks (7 test, 9) call `backtest_session` directly; `backtest_race` keeps its own name and signature unchanged for existing callers (`api/routes.py::_completed_race_prediction`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_session.py
import pandas as pd
import pytest

from f1_predictor.evaluate import backtest


def test_backtest_session_race_delegates_to_backtest_race(monkeypatch):
    calls = []

    def fake_backtest_race(season, round_, seasons=None):
        calls.append((season, round_, seasons))
        return pd.DataFrame([{"driver_id": "a", "p_win": 1.0}])

    monkeypatch.setattr(backtest, "backtest_race", fake_backtest_race)
    result = backtest.backtest_session(2024, 5, session_type="race")

    assert calls == [(2024, 5, None)]
    assert result.iloc[0]["driver_id"] == "a"


def test_backtest_session_qualifying_no_lookahead(monkeypatch):
    from f1_predictor.models import session_outcome

    rows = []
    for rnd in range(1, 4):
        for i, driver in enumerate(["a", "b", "c"], start=1):
            rows.append(
                {
                    "season": 2024,
                    "round": rnd,
                    "driver_id": driver,
                    "constructor_id": f"team_{i % 2}",
                    "quali_position": i,
                    "elo_pre_race": 1600 - i * 10,
                    "team_strength_pre_race": 1500.0,
                }
            )
    fake_df = pd.DataFrame(rows)
    feature_cols = ["elo_pre_race", "team_strength_pre_race"]

    monkeypatch.setattr(
        backtest.session_build, "build_session_training_frame", lambda session_type, seasons=None: (fake_df, feature_cols)
    )

    result = backtest.backtest_session(2024, 3, session_type="qualifying")

    assert set(result["driver_id"]) == {"a", "b", "c"}
    assert "position" in result.columns  # actual quali_position, renamed for comparison


def test_backtest_session_raises_when_no_data_for_round(monkeypatch):
    fake_df = pd.DataFrame(
        [{"season": 2024, "round": 1, "driver_id": "a", "constructor_id": "t", "quali_position": 1, "elo_pre_race": 1600.0}]
    )
    monkeypatch.setattr(
        backtest.session_build, "build_session_training_frame", lambda session_type, seasons=None: (fake_df, ["elo_pre_race"])
    )
    with pytest.raises(ValueError, match="No qualifying data found"):
        backtest.backtest_session(2024, 99, session_type="qualifying")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_backtest_session.py -v`
Expected: FAIL with `AttributeError: module 'f1_predictor.evaluate.backtest' has no attribute 'backtest_session'`

- [ ] **Step 3: Add `backtest_session` to `backtest.py`**

Add these imports near the top of `src/f1_predictor/evaluate/backtest.py` (alongside the existing ones):

```python
from ..features import session_build
from ..models import session_outcome
```

Add this function after the existing `backtest_race` function (before `def main():`):

```python
def backtest_session(
    season: int, round_: int, session_type: str = "race", seasons: list[int] | None = None
) -> pd.DataFrame:
    """Generalizes backtest_race's no-lookahead replay to the three new
    session types. session_type="race" is a pure passthrough to
    backtest_race (unchanged, still the race model's own path) — this
    function exists so callers (api/routes.py) can use one name regardless
    of session_type."""
    if session_type == "race":
        return backtest_race(season, round_, seasons=seasons)

    seasons = seasons or jolpica.default_seasons()
    spec = session_outcome.SESSION_SPECS[session_type]
    df, feature_cols = session_build.build_session_training_frame(session_type, seasons=seasons)

    before = (df["season"] < season) | ((df["season"] == season) & (df["round"] < round_))
    train_df = df[before]
    session_df = df[(df["season"] == season) & (df["round"] == round_)]
    if session_df.empty:
        raise ValueError(f"No {session_type} data found for {season} round {round_}")

    ranker = session_outcome.train_session_ranker(train_df, feature_cols, spec)
    dnf_clf = None
    if spec.has_dnf:
        dnf_clf = dnf_model.train_dnf_model(train_df, feature_cols)

    sim = session_outcome.predict_session(ranker, session_df, feature_cols, spec, dnf_clf=dnf_clf, n_trials=10000, seed=0)

    actual_cols = ["constructor_id", spec.target_column]
    if spec.has_dnf:
        actual_cols.append("dnf")
    actual = session_df.set_index("driver_id")[actual_cols].rename(columns={spec.target_column: "position"})
    result = sim.set_index("driver_id").join(actual, how="left").reset_index()
    return result.sort_values("p_win", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_backtest_session.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/evaluate/backtest.py tests/test_backtest_session.py
git commit -m "$(cat <<'EOF'
Add backtest_session, generalizing backtest_race to the 3 new session types

session_type="race" passes straight through to the existing, unmodified
backtest_race. Same no-lookahead discipline (train on strictly earlier
rounds) for sprint_qualifying/qualifying/sprint.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Migrate `tracking/store.py` to a generalized `session_predictions` table

**Files:**
- Modify: `src/f1_predictor/tracking/store.py`
- Test: `tests/test_session_predictions_store.py` (new file)

**Interfaces:**
- Consumes: nothing new.
- Produces: `SESSION_MARKET_SPEC: dict[str, list[tuple[str, str]]]` (keyed by session_type; `"race"`/`"sprint"` -> `[("win","p_win"),("podium","p_podium"),("points_finish","p_points_finish"),("dnf","p_dnf")]`, `"sprint_qualifying"`/`"qualifying"` -> `[("pole","p_win"),("top_3","p_podium"),("top_10","p_points_finish")]` — the probability column names on the `sim_table` DataFrame passed in are always `p_win`/`p_podium`/`p_points_finish`/`p_dnf` regardless of session type per Task 3's `predict_session`; only the *market name* stored differs), `record_session_predictions(sim_table, season, round_, race_name, session_type, tier, session_time, model_trained_at=None, backfilled=False) -> int`, `reconcile_session_predictions(results_df: pd.DataFrame, session_type: str) -> int`, `get_session_prediction(season, round_, session_type, tier=None) -> list[dict]`, `get_session_track_record(session_type=None, tier=None) -> dict`, `get_session_accuracy(session_type=None, tier=None) -> list[dict]`. The old `race_predictions`-specific functions (`record_race_predictions`, `reconcile_predictions`, `get_track_record`, `get_race_accuracy`, `get_race_prediction`) are **removed** and every internal caller updated in this same task (there are no external callers outside this package yet — Task 9 is the only consumer, done after this task).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_session_predictions_store.py
import sqlite3

import pandas as pd
import pytest

from f1_predictor.tracking import store


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_tracking.db"
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield db_path


def _sim_table():
    return pd.DataFrame(
        [
            {
                "driver_id": "norris",
                "constructor_id": "mclaren",
                "p_win": 0.3,
                "p_podium": 0.6,
                "p_points_finish": 0.9,
                "p_dnf": 0.05,
            },
            {
                "driver_id": "leclerc",
                "constructor_id": "ferrari",
                "p_win": 0.15,
                "p_podium": 0.4,
                "p_points_finish": 0.8,
                "p_dnf": 0.05,
            },
        ]
    )


def test_record_and_get_race_session_prediction():
    n = store.record_session_predictions(
        _sim_table(), 2024, 5, "Fake GP", "race", "post_qualifying", "2024-05-01T00:00:00Z"
    )
    assert n == 8  # 2 drivers x 4 markets (win/podium/points_finish/dnf)

    rows = store.get_session_prediction(2024, 5, "race")
    markets = {r["market"] for r in rows}
    assert markets == {"win", "podium", "points_finish", "dnf"}


def test_record_and_get_qualifying_session_prediction_uses_quali_markets():
    n = store.record_session_predictions(
        _sim_table(), 2024, 5, "Fake GP", "qualifying", "post_practice", "2024-05-01T00:00:00Z"
    )
    assert n == 6  # 2 drivers x 3 markets (pole/top_3/top_10)

    rows = store.get_session_prediction(2024, 5, "qualifying")
    markets = {r["market"] for r in rows}
    assert markets == {"pole", "top_3", "top_10"}
    assert all(r["actual_dnf"] is None for r in rows)


def test_reconcile_session_predictions_fills_actual_outcome():
    store.record_session_predictions(
        _sim_table(), 2024, 5, "Fake GP", "qualifying", "post_practice", "2024-05-01T00:00:00Z"
    )
    results = pd.DataFrame(
        [
            {"season": 2024, "round": 5, "driver_id": "norris", "position": 1, "dnf": False},
            {"season": 2024, "round": 5, "driver_id": "leclerc", "position": 4, "dnf": False},
        ]
    )
    updated = store.reconcile_session_predictions(results, "qualifying")
    assert updated == 6

    rows = store.get_session_prediction(2024, 5, "qualifying")
    pole_rows = {r["driver_id"]: r for r in rows if r["market"] == "pole"}
    assert pole_rows["norris"]["actual_outcome"] == 1
    assert pole_rows["leclerc"]["actual_outcome"] == 0
    assert pole_rows["norris"]["actual_position"] == 1


def test_migration_preserves_existing_race_predictions_rows(tmp_path, monkeypatch):
    # Simulate a pre-migration database: create the OLD race_predictions
    # shape by hand, then verify _connect()'s migration step carries every
    # row over into session_predictions with session_type='race' and
    # actual_finish_position renamed to actual_position.
    db_path = tmp_path / "legacy_tracking.db"
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE race_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL, round INTEGER NOT NULL, race_name TEXT NOT NULL,
            driver_id TEXT NOT NULL, constructor_id TEXT, tier TEXT NOT NULL, market TEXT NOT NULL,
            predicted_prob REAL NOT NULL, session_time TEXT NOT NULL, snapshotted_at TEXT NOT NULL,
            model_trained_at TEXT, resolved INTEGER NOT NULL DEFAULT 0, actual_outcome INTEGER,
            actual_finish_position INTEGER, actual_dnf INTEGER, resolved_at TEXT,
            backfilled INTEGER NOT NULL DEFAULT 0,
            UNIQUE(season, round, driver_id, tier, market)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO race_predictions
            (season, round, race_name, driver_id, constructor_id, tier, market, predicted_prob,
             session_time, snapshotted_at, resolved, actual_outcome, actual_finish_position, actual_dnf)
        VALUES (2023, 1, 'Old GP', 'hamilton', 'mercedes', 'post_qualifying', 'win', 0.2,
                '2023-01-01T00:00:00Z', '2023-01-01T00:00:00Z', 1, 1, 1, 0)
        """
    )
    conn.commit()
    conn.close()

    rows = store.get_session_prediction(2023, 1, "race")
    assert len(rows) == 1
    assert rows[0]["session_type"] == "race"
    assert rows[0]["actual_position"] == 1
    assert rows[0]["driver_id"] == "hamilton"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_session_predictions_store.py -v`
Expected: FAIL with `AttributeError: module 'f1_predictor.tracking.store' has no attribute 'record_session_predictions'`

- [ ] **Step 3: Rewrite `store.py`**

Replace the entire contents of `src/f1_predictor/tracking/store.py`:

```python
"""store.py — SQLite persistence for session and championship prediction
track records. Mirrors PL_Predictor's tracking/store.py: snapshot each
prediction BEFORE the outcome is known, reconcile against actual results
once they land — the only honest way to measure "how good are the
predictions really," and this project's substitute for the market
comparison PL_Predictor's value-bet feature relies on (there is no F1 odds
market — see RESEARCH_BRIEF.md).

`session_predictions` generalizes the original race-only
`race_predictions` table to all four session types (sprint_qualifying,
qualifying, sprint, race) — keyed on (season, round, session_type,
driver_id, tier, market). A migration on first connect carries every
existing race_predictions row over with session_type='race' and
actual_finish_position renamed to actual_position; expected_position is
left NULL for those old rows since it was never actually predicted at the
time (not fabricated retroactively — same honesty rule the rest of this
app follows).

`championship_snapshots` is unchanged from before this migration.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ..config import TRACKING_DB_PATH

# Probability-market column names on the sim_table DataFrame are always
# p_win/p_podium/p_points_finish/p_dnf regardless of session type (see
# models/session_outcome.py::predict_session) — only the STORED market
# name differs, and quali-type sessions have no dnf market at all.
SESSION_MARKET_SPEC: dict[str, list[tuple[str, str]]] = {
    "race": [("win", "p_win"), ("podium", "p_podium"), ("points_finish", "p_points_finish"), ("dnf", "p_dnf")],
    "sprint": [("win", "p_win"), ("podium", "p_podium"), ("points_finish", "p_points_finish"), ("dnf", "p_dnf")],
    "qualifying": [("pole", "p_win"), ("top_3", "p_podium"), ("top_10", "p_points_finish")],
    "sprint_qualifying": [("pole", "p_win"), ("top_3", "p_podium"), ("top_10", "p_points_finish")],
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(TRACKING_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            round INTEGER NOT NULL,
            race_name TEXT NOT NULL,
            session_type TEXT NOT NULL,
            driver_id TEXT NOT NULL,
            constructor_id TEXT,
            tier TEXT NOT NULL,
            market TEXT NOT NULL,
            predicted_prob REAL NOT NULL,
            expected_position REAL,
            session_time TEXT NOT NULL,
            snapshotted_at TEXT NOT NULL,
            model_trained_at TEXT,
            resolved INTEGER NOT NULL DEFAULT 0,
            actual_outcome INTEGER,
            actual_position INTEGER,
            actual_dnf INTEGER,
            resolved_at TEXT,
            backfilled INTEGER NOT NULL DEFAULT 0,
            UNIQUE(season, round, session_type, driver_id, tier, market)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS championship_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season INTEGER NOT NULL,
            as_of_round INTEGER NOT NULL,
            championship TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            win_prob REAL NOT NULL,
            top3_prob REAL,
            expected_final_points REAL NOT NULL,
            expected_final_rank REAL,
            snapshotted_at TEXT NOT NULL,
            UNIQUE(season, as_of_round, championship, entity_id)
        )
        """
    )
    _migrate_legacy_race_predictions(conn)
    return conn


def _migrate_legacy_race_predictions(conn: sqlite3.Connection) -> None:
    """One-time, idempotent migration: if an old race_predictions table
    exists (pre-dating session_predictions), copy every row over with
    session_type='race' and actual_finish_position renamed to
    actual_position, then drop it. Safe to run on every connect — a no-op
    once the old table is gone."""
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='race_predictions'"
    ).fetchone()
    if not exists:
        return
    conn.execute(
        """
        INSERT OR IGNORE INTO session_predictions
            (season, round, race_name, session_type, driver_id, constructor_id, tier, market,
             predicted_prob, expected_position, session_time, snapshotted_at, model_trained_at,
             resolved, actual_outcome, actual_position, actual_dnf, resolved_at, backfilled)
        SELECT
            season, round, race_name, 'race', driver_id, constructor_id, tier, market,
            predicted_prob, NULL, session_time, snapshotted_at, model_trained_at,
            resolved, actual_outcome, actual_finish_position, actual_dnf, resolved_at, backfilled
        FROM race_predictions
        """
    )
    conn.execute("DROP TABLE race_predictions")
    conn.commit()


def record_session_predictions(
    sim_table: pd.DataFrame,
    season: int,
    round_: int,
    race_name: str,
    session_type: str,
    tier: str,
    session_time: str,
    model_trained_at: str | None = None,
    backfilled: bool = False,
) -> int:
    """Snapshot one session's Monte Carlo output (models/session_outcome.py
    ::predict_session's or race_outcome.py::simulate_race's return, with a
    constructor_id column already joined on by the caller). Already-logged
    (season, round, session_type, driver_id, tier, market) rows are left
    untouched — safe to call every time a prediction is served."""
    if sim_table.empty:
        return 0
    market_spec = SESSION_MARKET_SPEC[session_type]
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for _, r in sim_table.iterrows():
        expected_position = r.get("expected_position")
        expected_position = None if expected_position is None or pd.isna(expected_position) else float(expected_position)
        for market, prob_col in market_spec:
            rows.append(
                (
                    season,
                    round_,
                    race_name,
                    session_type,
                    r["driver_id"],
                    r.get("constructor_id"),
                    tier,
                    market,
                    float(r[prob_col]),
                    expected_position,
                    session_time,
                    now,
                    model_trained_at,
                    int(backfilled),
                )
            )
    with _connect() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO session_predictions
                (season, round, race_name, session_type, driver_id, constructor_id, tier, market,
                 predicted_prob, expected_position, session_time, snapshotted_at, model_trained_at, backfilled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount


def _actual_outcome(session_type: str, market: str, driver_row: pd.Series) -> int:
    position = driver_row.get("position")
    has_position = position is not None and not pd.isna(position)
    if market in ("win", "pole"):
        return int(has_position and int(position) == 1)
    if market in ("podium", "top_3"):
        return int(has_position and int(position) <= 3)
    if market == "points_finish":
        cutoff = 8 if session_type == "sprint" else 10
        return int(has_position and int(position) <= cutoff)
    if market == "top_10":
        return int(has_position and int(position) <= 10)
    if market == "dnf":
        return int(bool(driver_row.get("dnf")))
    raise ValueError(f"unknown market: {market}")


def reconcile_session_predictions(results_df: pd.DataFrame, session_type: str) -> int:
    """Fills in actual outcomes for every still-unresolved session_predictions
    row of `session_type` whose (season, round, driver_id) now has a real
    result in `results_df` (columns: season, round, driver_id, position,
    and dnf for race/sprint). Safe to call repeatedly — only touches
    resolved=0 rows."""
    if results_df.empty:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    results_lookup = results_df.set_index(["season", "round", "driver_id"])

    with _connect() as conn:
        unresolved = pd.read_sql(
            "SELECT id, season, round, driver_id, market FROM session_predictions "
            "WHERE resolved = 0 AND session_type = ?",
            conn,
            params=(session_type,),
        )
        if unresolved.empty:
            return 0

        updates = []
        for _, row in unresolved.iterrows():
            key = (row["season"], row["round"], row["driver_id"])
            if key not in results_lookup.index:
                continue
            driver_row = results_lookup.loc[key]
            if isinstance(driver_row, pd.DataFrame):
                driver_row = driver_row.iloc[0]
            actual = _actual_outcome(session_type, row["market"], driver_row)
            position = driver_row.get("position")
            updates.append(
                (
                    actual,
                    None if position is None or pd.isna(position) else int(position),
                    int(bool(driver_row.get("dnf"))) if "dnf" in driver_row else None,
                    now,
                    int(row["id"]),
                )
            )

        if updates:
            conn.executemany(
                """
                UPDATE session_predictions
                SET resolved = 1, actual_outcome = ?, actual_position = ?, actual_dnf = ?, resolved_at = ?
                WHERE id = ?
                """,
                updates,
            )
        return len(updates)


def record_championship_snapshot(projection: dict) -> int:
    """One row per driver/constructor from
    models/championship_projection.py::simulate_season's output."""
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for championship in ("drivers", "constructors"):
        for _, r in projection[championship].iterrows():
            rows.append(
                (
                    projection["season"],
                    projection["as_of_round"],
                    championship,
                    r["entity_id"],
                    float(r["win_prob"]),
                    float(r["top3_prob"]),
                    float(r["expected_final_points"]),
                    float(r["expected_final_rank"]),
                    now,
                )
            )
    with _connect() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO championship_snapshots
                (season, as_of_round, championship, entity_id, win_prob, top3_prob,
                 expected_final_points, expected_final_rank, snapshotted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount


def get_session_track_record(session_type: str | None = None, tier: str | None = None) -> dict:
    """Overall hit-rate/calibration summary, optionally filtered to one
    session type and/or tier."""
    with _connect() as conn:
        query = "SELECT session_type, tier, market, predicted_prob, actual_outcome FROM session_predictions WHERE resolved = 1"
        params: list = []
        if session_type:
            query += " AND session_type = ?"
            params.append(session_type)
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return {"n_resolved": 0, "by_market": []}

    rows = []
    for (st, t, market), g in df.groupby(["session_type", "tier", "market"]):
        brier = float(((g["predicted_prob"] - g["actual_outcome"]) ** 2).mean())
        rows.append(
            {
                "session_type": st,
                "tier": t,
                "market": market,
                "n": int(len(g)),
                "brier": brier,
                "hit_rate": float(g["actual_outcome"].mean()),
                "avg_predicted_prob": float(g["predicted_prob"].mean()),
            }
        )
    return {"n_resolved": int(len(df)), "by_market": rows}


def get_session_accuracy(session_type: str | None = None, tier: str | None = None) -> list[dict]:
    """Per-session accuracy: did the model's own top pick actually land
    there — mirrors the old get_race_accuracy, generalized across markets
    per session type via SESSION_MARKET_SPEC."""
    top_n_by_market = {"win": 1, "pole": 1, "podium": 3, "top_3": 3, "points_finish": 10, "top_10": 10}
    with _connect() as conn:
        query = (
            "SELECT season, round, race_name, session_type, tier, driver_id, market, predicted_prob, actual_outcome "
            "FROM session_predictions WHERE resolved = 1 AND market != 'dnf'"
        )
        params: list = []
        if session_type:
            query += " AND session_type = ?"
            params.append(session_type)
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return []

    rows = []
    for (season, round_, race_name, st, t), race_df in df.groupby(["season", "round", "race_name", "session_type", "tier"]):
        row = {"season": int(season), "round": int(round_), "race_name": race_name, "session_type": st, "tier": t}
        for market in race_df["market"].unique():
            n = top_n_by_market.get(market)
            if n is None:
                continue
            market_df = race_df[race_df["market"] == market]
            predicted = set(market_df.nlargest(n, "predicted_prob")["driver_id"])
            actual = set(market_df.loc[market_df["actual_outcome"] == 1, "driver_id"])
            row[f"{market}_predicted"] = sorted(predicted)
            row[f"{market}_actual"] = sorted(actual)
            row[f"{market}_hits"] = len(predicted & actual)
            row[f"{market}_of"] = n
        rows.append(row)

    return sorted(rows, key=lambda r: (r["season"], r["round"]), reverse=True)


def get_session_prediction(season: int, round_: int, session_type: str, tier: str | None = None) -> list[dict]:
    with _connect() as conn:
        query = "SELECT * FROM session_predictions WHERE season = ? AND round = ? AND session_type = ?"
        params: list = [season, round_, session_type]
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        df = pd.read_sql(query, conn, params=params)
    return df.to_dict(orient="records")


def get_championship_history(season: int, championship: str) -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql(
            "SELECT * FROM championship_snapshots WHERE season = ? AND championship = ? ORDER BY as_of_round",
            conn,
            params=(season, championship),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_session_predictions_store.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full existing test suite — expect known breakage to fix in Task 9**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: `tests/test_race_prediction_caching.py` still passes (it doesn't touch `store.py` directly). Any other test referencing the now-removed `store.record_race_predictions`/`store.get_race_prediction`/etc. would fail here — if `grep -rn "record_race_predictions\|get_race_prediction\|reconcile_predictions\|get_track_record\|get_race_accuracy" tests/ src/f1_predictor/api/routes.py` shows hits in `routes.py`, that's expected and fixed in Task 9, not here. If it shows hits elsewhere (e.g. a script under `src/f1_predictor/`), stop and fix that call site now before committing — Global Constraints requires every existing test to keep passing after each task, but `routes.py`'s break is a known, deliberate, immediately-next-task dependency, not a regression to silently accept beyond that one file.

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/tracking/store.py tests/test_session_predictions_store.py
git commit -m "$(cat <<'EOF'
Migrate tracking store to generalized session_predictions table

Replaces race_predictions with session_predictions (season, round,
session_type, driver_id, tier, market), with an idempotent migration
that carries every existing row over (session_type='race',
actual_finish_position -> actual_position). api/routes.py's calls into
the old function names are updated in the next task.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Extend `models/manifest.py` to train and save the 3 new session models

**Files:**
- Modify: `src/f1_predictor/models/manifest.py`
- Test: `tests/test_manifest_session_models.py` (new file)

**Interfaces:**
- Consumes: `session_build.build_session_training_frame`, `session_outcome.SESSION_SPECS`, `session_outcome.train_session_ranker` (Tasks 3, 4), `dnf.train_dnf_model` (existing).
- Produces: `SPRINT_QUALIFYING_MODEL_PATH`, `QUALIFYING_MODEL_PATH`, `SPRINT_MODEL_PATH`, `SPRINT_DNF_MODEL_PATH` (module-level `Path` constants), `SESSION_MODEL_PATHS: dict[str, tuple[Path, Path | None]]` (keys `"sprint_qualifying"`, `"qualifying"`, `"sprint"`; value is `(ranker_path, dnf_path_or_None)`). `train_all()`'s return dict gains a `"session_models"` key: `{session_type: {"n_training_rows": int}}` for whichever session types had data to train on. Task 9 imports `SESSION_MODEL_PATHS` from this module.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_manifest_session_models.py
import pandas as pd

from f1_predictor.models import manifest


def test_train_all_trains_and_saves_session_models(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(manifest, "RACE_OUTCOME_MODEL_PATH", tmp_path / "race_outcome_ranker.json")
    monkeypatch.setattr(manifest, "DNF_MODEL_PATH", tmp_path / "dnf_model.json")
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(manifest, "SPRINT_QUALIFYING_MODEL_PATH", tmp_path / "sprint_qualifying_ranker.json")
    monkeypatch.setattr(manifest, "QUALIFYING_MODEL_PATH", tmp_path / "qualifying_ranker.json")
    monkeypatch.setattr(manifest, "SPRINT_MODEL_PATH", tmp_path / "sprint_ranker.json")
    monkeypatch.setattr(manifest, "SPRINT_DNF_MODEL_PATH", tmp_path / "sprint_dnf_model.json")
    monkeypatch.setattr(
        manifest,
        "SESSION_MODEL_PATHS",
        {
            "sprint_qualifying": (tmp_path / "sprint_qualifying_ranker.json", None),
            "qualifying": (tmp_path / "qualifying_ranker.json", None),
            "sprint": (tmp_path / "sprint_ranker.json", tmp_path / "sprint_dnf_model.json"),
        },
    )

    def _fake_race_frame(seasons=None):
        rows = []
        for rnd in range(1, 6):
            for i, driver in enumerate(["a", "b", "c"], start=1):
                rows.append(
                    {
                        "season": 2024, "round": rnd, "driver_id": driver, "constructor_id": f"t{i % 2}",
                        "tier": "post_qualifying", "position": i, "dnf": False,
                        "elo_pre_race": 1600 - i * 5, "team_strength_pre_race": 1500.0,
                    }
                )
        return pd.DataFrame(rows), ["elo_pre_race", "team_strength_pre_race"]

    def _fake_session_frame(session_type, seasons=None):
        rows = []
        target_col = {"sprint_qualifying": "sprint_quali_position", "qualifying": "quali_position", "sprint": "position"}[
            session_type
        ]
        for rnd in range(1, 4):
            for i, driver in enumerate(["a", "b", "c"], start=1):
                row = {
                    "season": 2024, "round": rnd, "driver_id": driver, "constructor_id": f"t{i % 2}",
                    target_col: i, "elo_pre_race": 1600 - i * 5, "team_strength_pre_race": 1500.0,
                }
                if session_type == "sprint":
                    row["dnf"] = False
                    row["sprint_quali_position"] = i
                rows.append(row)
        return pd.DataFrame(rows), ["elo_pre_race", "team_strength_pre_race"] + (
            ["sprint_quali_position"] if session_type == "sprint" else []
        )

    monkeypatch.setattr(manifest, "build_training_frame", _fake_race_frame)
    monkeypatch.setattr(manifest.session_build, "build_session_training_frame", _fake_session_frame)
    monkeypatch.setattr(
        manifest.walk_forward,
        "prepare_folds",
        lambda seasons: [{"train": pd.DataFrame(), "test": pd.DataFrame()}],
    )
    monkeypatch.setattr(
        manifest.walk_forward,
        "evaluate_candidate",
        lambda folds, candidate, hyperparams=None: pd.DataFrame([{"log_loss": 1.0}]),
    )

    result = manifest.train_all(seasons=[2024])

    assert (tmp_path / "sprint_qualifying_ranker.json").exists()
    assert (tmp_path / "qualifying_ranker.json").exists()
    assert (tmp_path / "sprint_ranker.json").exists()
    assert (tmp_path / "sprint_dnf_model.json").exists()
    assert result["session_models"]["qualifying"]["n_training_rows"] == 9
    assert result["session_models"]["sprint"]["n_training_rows"] == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_manifest_session_models.py -v`
Expected: FAIL with `AttributeError: module 'f1_predictor.models.manifest' has no attribute 'SPRINT_QUALIFYING_MODEL_PATH'`

- [ ] **Step 3: Extend `manifest.py`**

Add these imports to `src/f1_predictor/models/manifest.py` (alongside the existing ones):

```python
from ..features import session_build
from . import session_outcome
```

Add these constants directly below the existing `OPTUNA_DB_PATH = MODELS_DIR / "optuna_studies.db"` line:

```python
SPRINT_QUALIFYING_MODEL_PATH = MODELS_DIR / "sprint_qualifying_ranker.json"
QUALIFYING_MODEL_PATH = MODELS_DIR / "qualifying_ranker.json"
SPRINT_MODEL_PATH = MODELS_DIR / "sprint_ranker.json"
SPRINT_DNF_MODEL_PATH = MODELS_DIR / "sprint_dnf_model.json"

SESSION_MODEL_PATHS: dict[str, tuple] = {
    "sprint_qualifying": (SPRINT_QUALIFYING_MODEL_PATH, None),
    "qualifying": (QUALIFYING_MODEL_PATH, None),
    "sprint": (SPRINT_MODEL_PATH, SPRINT_DNF_MODEL_PATH),
}
```

In `train_all`, insert this block right before the `manifest = {` dict-construction line (i.e., after the existing DNF model training, before the manifest dict is built):

```python
    print("Training the 3 new session predictors (sprint_qualifying, qualifying, sprint)...")
    session_metrics: dict[str, dict] = {}
    for session_type, (model_path, dnf_path) in SESSION_MODEL_PATHS.items():
        spec = session_outcome.SESSION_SPECS[session_type]
        sdf, sfeature_cols = session_build.build_session_training_frame(session_type, seasons=seasons)
        if sdf.empty:
            print(f"  {session_type}: no training data available yet, skipping")
            continue
        print(f"  {session_type}: training on {len(sdf)} rows...")
        sranker = session_outcome.train_session_ranker(sdf, sfeature_cols, spec)
        sranker.save_model(str(model_path))
        if spec.has_dnf and dnf_path is not None:
            sdnf_clf = dnf_model.train_dnf_model(sdf, sfeature_cols)
            sdnf_clf.save_model(str(dnf_path))
        session_metrics[session_type] = {"n_training_rows": int(len(sdf))}
```

Then add `"session_models": session_metrics,` as a new key inside the existing `manifest = {...}` dict literal (anywhere among the existing keys, e.g. right after `"tuned_dnf_params": dnf_params,`).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_manifest_session_models.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/f1_predictor/models/manifest.py tests/test_manifest_session_models.py
git commit -m "$(cat <<'EOF'
Train and save the 3 new session models from manifest.py::train_all

New model artifact paths (sprint_qualifying/qualifying/sprint rankers,
sprint's own DNF model) alongside the existing race/DNF ones. No
elo-vs-xgb_ranker race-off for these three — always xgb_ranker directly
(see plan's Global Constraints for why).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Add `SessionPredictionResponse` schema

**Files:**
- Modify: `src/f1_predictor/api/schemas.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SessionDriverPrediction` (pydantic `BaseModel`: `driver_id: str`, `constructor_id: str | None = None`, `p_pole: float | None = None`, `p_top_3: float | None = None`, `p_top_10: float | None = None`, `p_win: float | None = None`, `p_podium: float | None = None`, `p_points_finish: float | None = None`, `p_dnf: float | None = None`, `expected_position: float`, `actual_position: int | None = None`, `actual_dnf: bool | None = None`), `SessionPredictionResponse` (`season: int`, `round: int`, `race_name: str`, `session_type: str`, `tier: str`, `source: str`, `predictions: list[SessionDriverPrediction]`). Task 9 imports both from `schemas.py`.

This task has no separate test — it's a pure pydantic model addition, exercised end-to-end by Task 9's route tests.

- [ ] **Step 1: Add the schemas**

Append to `src/f1_predictor/api/schemas.py` (after the existing `RacePredictionResponse` class):

```python
class SessionDriverPrediction(BaseModel):
    driver_id: str
    constructor_id: str | None = None
    # Qualifying-type markets (sprint_qualifying, qualifying) — omitted for race/sprint.
    p_pole: float | None = None
    p_top_3: float | None = None
    p_top_10: float | None = None
    # Race-type markets (sprint, race) — omitted for sprint_qualifying/qualifying.
    p_win: float | None = None
    p_podium: float | None = None
    p_points_finish: float | None = None
    p_dnf: float | None = None
    expected_position: float
    actual_position: int | None = None
    actual_dnf: bool | None = None


class SessionPredictionResponse(BaseModel):
    season: int
    round: int
    race_name: str
    session_type: str  # "sprint_qualifying" | "qualifying" | "sprint" | "race"
    tier: str
    source: str  # "live" | "tracked" | "backtest"
    predictions: list[SessionDriverPrediction]
```

- [ ] **Step 2: Verify the module still imports cleanly**

Run: `PYTHONPATH=src python -c "from f1_predictor.api import schemas; print(schemas.SessionPredictionResponse)"`
Expected: prints the class, no error.

- [ ] **Step 3: Run the full existing test suite to confirm no regression**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/f1_predictor/api/schemas.py
git commit -m "$(cat <<'EOF'
Add SessionPredictionResponse/SessionDriverPrediction schemas

One generalized response shape for the 3 new session-prediction
endpoints, with quali-type and race-type market fields both optional so
one schema covers both shapes without a 4-way schema duplication.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Add the 3 new API endpoints, fix `routes.py`'s store calls, add track-record `session_type` filter

**Files:**
- Modify: `src/f1_predictor/api/routes.py`
- Test: `tests/test_session_prediction_routes.py` (new file)

**Interfaces:**
- Consumes: Tasks 3–8's `session_outcome`, `session_build`, `backtest_session`, `store.record_session_predictions`/`get_session_prediction`/`get_session_track_record`/`get_session_accuracy`, `manifest.SESSION_MODEL_PATHS`, `schemas.SessionPredictionResponse`/`SessionDriverPrediction`, `session_state.current_session_tier`.
- Produces: `GET /api/races/{season}/{round_}/sprint-qualifying-prediction`, `GET /api/races/{season}/{round_}/sprint-prediction`, `GET /api/races/{season}/{round_}/qualifying-prediction` (all `response_model=SessionPredictionResponse`, 404 for the two sprint-only ones on a non-sprint weekend). `GET /api/races/{season}/{round_}/prediction` (existing race endpoint) is **unchanged in URL, request, and response shape** — its internal calls into `store.py` are updated to the new function names, nothing externally observable changes. `GET /api/track-record` and `/track-record/by-race` gain an optional `session_type` query param.

- [ ] **Step 1: Fix `routes.py`'s now-broken `store` calls first (no new behavior yet)**

In `src/f1_predictor/api/routes.py`, find `_completed_race_prediction` (uses `store.get_race_prediction`) and replace its body's first two lines:

```python
    tracked = store.get_race_prediction(season, round_, tier=session_state.TIER_POST_QUALIFYING)
```

with:

```python
    tracked = store.get_session_prediction(season, round_, "race", tier=session_state.TIER_POST_QUALIFYING)
```

Also update that same function's later reference from `df.rename(columns={"win": "p_win", ...})` — this stays as-is (market names for `session_type="race"` are unchanged: win/podium/points_finish/dnf), but the `extra` frame's column is now `actual_position` directly (Task 6 already renamed it in the table), not `actual_finish_position`, so change:

```python
        extra = df.drop_duplicates("driver_id")[["driver_id", "constructor_id", "actual_finish_position", "actual_dnf"]]
        extra = extra.rename(columns={"actual_finish_position": "actual_position"})
```

to:

```python
        extra = df.drop_duplicates("driver_id")[["driver_id", "constructor_id", "actual_position", "actual_dnf"]]
```

Find `get_track_record` and `get_race_accuracy` route handlers and update:

```python
@router.get("/track-record", response_model=TrackRecordResponse)
def get_track_record(tier: str | None = None) -> TrackRecordResponse:
    result = store.get_track_record(tier=tier)
    return TrackRecordResponse(
        n_resolved=result["n_resolved"], by_market=[TrackRecordEntry(**row) for row in result["by_market"]]
    )


@router.get("/track-record/by-race", response_model=list[RaceAccuracyEntry])
def get_race_accuracy(tier: str | None = None) -> list[RaceAccuracyEntry]:
    return [RaceAccuracyEntry(**row) for row in store.get_race_accuracy(tier=tier)]
```

to:

```python
@router.get("/track-record", response_model=TrackRecordResponse)
def get_track_record(tier: str | None = None, session_type: str | None = None) -> TrackRecordResponse:
    result = store.get_session_track_record(session_type=session_type, tier=tier)
    by_market = [{k: v for k, v in row.items() if k != "session_type"} for row in result["by_market"]]
    return TrackRecordResponse(n_resolved=result["n_resolved"], by_market=[TrackRecordEntry(**row) for row in by_market])


@router.get("/track-record/by-race", response_model=list[RaceAccuracyEntry])
def get_race_accuracy(tier: str | None = None, session_type: str | None = None) -> list[RaceAccuracyEntry]:
    return [RaceAccuracyEntry(**{k: v for k, v in row.items() if k != "session_type"}) for row in store.get_session_accuracy(session_type=session_type, tier=tier)]
```

`TrackRecordEntry`/`RaceAccuracyEntry` in `schemas.py` don't have a `session_type` field, so it's stripped before constructing them here — the route param exists to *filter*, the response shape stays exactly as before. (`RaceAccuracyEntry`'s current fields are `win_*`/`podium_*`/`points_finish_*` only — for a `qualifying`/`sprint_qualifying` filtered request, `get_session_accuracy`'s rows will instead carry `pole_*`/`top_3_*`/`top_10_*` keys, which `RaceAccuracyEntry(**row)` will reject as unexpected kwargs. This is acceptable for this task: `RaceAccuracyEntry` stays race/sprint-shaped, and querying `/track-record/by-race?session_type=qualifying` is not part of this task's required behavior — Task 9's tests below only exercise `session_type=None` and `session_type="race"` against this endpoint. Note this as a known gap, not silently — if the frontend (Task 12) needs per-race qualifying accuracy, that's a follow-up schema addition, out of scope here.)

- [ ] **Step 2: Run the full existing test suite to confirm Step 1 didn't break anything**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS (this step only renames call sites, no behavior change)

- [ ] **Step 3: Write the failing tests for the 3 new endpoints**

```python
# tests/test_session_prediction_routes.py
import pandas as pd
import pytest
from fastapi import HTTPException

from f1_predictor.api import routes


def _fake_schedule(is_sprint_weekend: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": 2024,
                "round": 5,
                "race_name": "Fake GP",
                "circuit_id": "fake",
                "circuit_name": "Fake Circuit",
                "race_datetime": pd.Timestamp("2099-01-01", tz="UTC"),
                "qualifying_datetime": pd.Timestamp("2099-01-01", tz="UTC") - pd.Timedelta(days=1),
                "sprint_datetime": pd.Timestamp("2099-01-01", tz="UTC") - pd.Timedelta(days=2) if is_sprint_weekend else None,
                "sprint_quali_datetime": pd.Timestamp("2099-01-01", tz="UTC") - pd.Timedelta(days=3) if is_sprint_weekend else None,
                "fp1_datetime": pd.Timestamp("2099-01-01", tz="UTC") - pd.Timedelta(days=4),
                "fp2_datetime": None,
                "fp3_datetime": None,
                "is_sprint_weekend": is_sprint_weekend,
            }
        ]
    )


def _fake_sim():
    return pd.DataFrame(
        [
            {
                "driver_id": "norris",
                "constructor_id": "mclaren",
                "p_win": 0.3,
                "p_podium": 0.6,
                "p_points_finish": 0.9,
                "p_dnf": 0.0,
                "expected_position": 3.0,
                "expected_points": 10.0,
            }
        ]
    )


def test_qualifying_prediction_endpoint_returns_pole_markets(monkeypatch):
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(False))
    monkeypatch.setattr(routes, "_predict_upcoming_session", lambda season, round_, race_row, session_type: (_fake_sim(), "post_practice"))

    result = routes.get_qualifying_prediction(2024, 5)

    assert result.session_type == "qualifying"
    assert result.predictions[0].p_pole == pytest.approx(0.3)
    assert result.predictions[0].p_win is None


def test_sprint_prediction_endpoint_404s_on_non_sprint_weekend(monkeypatch):
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(False))

    with pytest.raises(HTTPException) as exc_info:
        routes.get_sprint_prediction(2024, 5)
    assert exc_info.value.status_code == 404


def test_sprint_prediction_endpoint_returns_race_shaped_markets_on_sprint_weekend(monkeypatch):
    monkeypatch.setattr(routes, "_cache", {})
    monkeypatch.setattr(routes.jolpica, "fetch_season_schedule", lambda season: _fake_schedule(True))
    monkeypatch.setattr(routes, "_predict_upcoming_session", lambda season, round_, race_row, session_type: (_fake_sim(), "post_sprint_qualifying"))

    result = routes.get_sprint_prediction(2024, 5)

    assert result.session_type == "sprint"
    assert result.predictions[0].p_win == pytest.approx(0.3)
    assert result.predictions[0].p_pole is None
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest tests/test_session_prediction_routes.py -v`
Expected: FAIL with `AttributeError: module 'f1_predictor.api.routes' has no attribute 'get_qualifying_prediction'`

- [ ] **Step 5: Add the new route handlers and helpers**

Add `SessionDriverPrediction`, `SessionPredictionResponse` to the existing `from .schemas import (...)` block in `routes.py`. Add these imports:

```python
from ..features import session_build
from ..models import session_outcome
```

Add this constant near `_LIVE_CACHE_TTL_SECONDS` (top of the file, with the other module-level constants):

```python
_QUALI_MARKET_FIELDS = ("p_pole", "p_top_3", "p_top_10")
_RACE_MARKET_FIELDS = ("p_win", "p_podium", "p_points_finish", "p_dnf")
```

Add these functions after `_completed_race_prediction` (before `_race_feature_frame_for_explain`):

```python
def _session_is_completed(season: int, round_: int, session_type: str, race_row: pd.Series, now: pd.Timestamp) -> bool:
    """Like _race_is_completed, generalized: a session is only "completed"
    once real results exist for it, never merely once its scheduled time
    has passed."""
    if session_type == "race":
        return _race_is_completed(season, round_, race_row.get("race_datetime"), now)
    if session_type in ("sprint", "sprint_qualifying"):
        session_dt = race_row.get("sprint_datetime" if session_type == "sprint" else "sprint_quali_datetime")
        if pd.isna(session_dt) or session_dt > now:
            return False
        return not jolpica.load_season_sprints(season)[
            jolpica.load_season_sprints(season)["round"] == round_
        ].empty
    # "qualifying"
    quali_dt = race_row.get("qualifying_datetime")
    if pd.isna(quali_dt) or quali_dt > now:
        return False
    return not jolpica.fetch_qualifying(season, round_).empty


def _current_session_feature_row(season: int, round_: int, race_row: pd.Series, session_type: str) -> pd.DataFrame:
    """The live "what's known right now" feature frame for an UPCOMING
    session, built the same way _future_feature_frame builds the race
    model's: championship_projection.build_future_feature_rows for the
    Elo/team-strength/form/circuit/weather base, then real
    session-specific columns merged in if already available this weekend."""
    schedule = jolpica.fetch_season_schedule(season)
    race_schedule = schedule[schedule["round"] == round_]
    results_df = jolpica.load_season_results(season)
    driver_ids = jolpica.fetch_driver_standings(season)["driver_id"].tolist()
    future_df = championship_projection.build_future_feature_rows(season, race_schedule, results_df, schedule, driver_ids)

    sprint_results = jolpica.load_season_sprints(season)
    sprint_this_round = sprint_results[sprint_results["round"] == round_] if not sprint_results.empty else sprint_results

    if session_type == "qualifying":
        future_df["sprint_finish_position"] = float("nan")
        if not sprint_this_round.empty:
            m = sprint_this_round.set_index("driver_id")["position"]
            future_df["sprint_finish_position"] = future_df["driver_id"].map(m)
    elif session_type == "sprint":
        future_df["sprint_quali_position"] = float("nan")
        if not sprint_this_round.empty:
            m = sprint_this_round.set_index("driver_id")["grid"]
            future_df["sprint_quali_position"] = future_df["driver_id"].map(m)
    return future_df


def _load_session_models(session_type: str) -> tuple:
    model_path, dnf_path = manifest_module.SESSION_MODEL_PATHS[session_type]
    if not model_path.exists():
        raise HTTPException(status_code=409, detail=f"No trained {session_type} model yet — call POST /api/retrain first.")
    ranker = xgb.XGBRanker()
    ranker.load_model(str(model_path))
    dnf_clf = None
    if dnf_path is not None and dnf_path.exists():
        dnf_clf = xgb.XGBClassifier()
        dnf_clf.load_model(str(dnf_path))
    return ranker, dnf_clf


def _predict_upcoming_session(season: int, round_: int, race_row: pd.Series, session_type: str) -> tuple[pd.DataFrame, str]:
    spec = session_outcome.SESSION_SPECS[session_type]
    feature_cols = session_build.SESSION_FEATURE_COLUMNS[session_type]
    future_df = _current_session_feature_row(season, round_, race_row, session_type)
    ranker, dnf_clf = _load_session_models(session_type)
    sim = session_outcome.predict_session(ranker, future_df, feature_cols, spec, dnf_clf=dnf_clf, n_trials=10000)
    constructor_lookup = future_df.set_index("driver_id")["constructor_id"]
    sim["constructor_id"] = sim["driver_id"].map(constructor_lookup)
    tier = session_state.current_session_tier(race_row)
    return sim, tier


def _completed_session_prediction(season: int, round_: int, session_type: str) -> tuple[pd.DataFrame, str, str]:
    spec = session_outcome.SESSION_SPECS[session_type]
    tracked = store.get_session_prediction(season, round_, session_type, tier=spec.feature_cutoff_tier)
    if tracked:
        df = pd.DataFrame(tracked)
        pivot = df.pivot_table(index="driver_id", columns="market", values="predicted_prob", aggfunc="first").reset_index()
        market_spec = store.SESSION_MARKET_SPEC[session_type]
        pivot = pivot.rename(columns={market: prob_col for market, prob_col in market_spec})
        extra = df.drop_duplicates("driver_id")[["driver_id", "constructor_id", "actual_position", "actual_dnf"]]
        pivot = pivot.merge(extra, on="driver_id", how="left")
        pivot["expected_position"] = float("nan")
        return pivot, spec.feature_cutoff_tier, "tracked"

    result = backtest_lib.backtest_session(season, round_, session_type=session_type)
    result = result.rename(columns={"position": "actual_position", "dnf": "actual_dnf"})
    return result, spec.feature_cutoff_tier, "backtest"


def _session_prediction_bundle(season: int, round_: int, race_row: pd.Series, session_type: str, completed: bool) -> tuple[pd.DataFrame, str, str]:
    if completed:
        try:
            return _completed_session_prediction(season, round_, session_type)
        except ValueError:
            pass
    sim, tier = _predict_upcoming_session(season, round_, race_row, session_type)
    return sim, tier, "live"


def _get_session_prediction_response(season: int, round_: int, session_type: str) -> SessionPredictionResponse:
    schedule = jolpica.fetch_season_schedule(season)
    race_rows = schedule[schedule["round"] == round_]
    if race_rows.empty:
        raise HTTPException(status_code=404, detail=f"No such race: {season} round {round_}")
    race_row = race_rows.iloc[0]

    if session_type in ("sprint", "sprint_qualifying") and not bool(race_row["is_sprint_weekend"]):
        raise HTTPException(status_code=404, detail=f"{season} round {round_} is not a sprint weekend")

    now = pd.Timestamp.now(tz="UTC")
    completed = _session_is_completed(season, round_, session_type, race_row, now)

    sim, tier, source = _cached(
        f"session_prediction_{session_type}_{season}_{round_}",
        lambda: _session_prediction_bundle(season, round_, race_row, session_type, completed),
        ttl=_LIVE_CACHE_TTL_SECONDS,
    )

    is_quali_type = session_type in ("sprint_qualifying", "qualifying")
    predictions = []
    for _, r in sim.sort_values("p_win", ascending=False).iterrows():
        kwargs = dict(
            driver_id=r["driver_id"],
            constructor_id=r.get("constructor_id"),
            expected_position=float(r["expected_position"]) if pd.notna(r.get("expected_position")) else float("nan"),
            actual_position=None if pd.isna(r.get("actual_position")) else int(r.get("actual_position")),
            actual_dnf=None if pd.isna(r.get("actual_dnf")) else bool(r.get("actual_dnf")),
        )
        if is_quali_type:
            kwargs.update(p_pole=float(r["p_win"]), p_top_3=float(r["p_podium"]), p_top_10=float(r["p_points_finish"]))
        else:
            kwargs.update(
                p_win=float(r["p_win"]), p_podium=float(r["p_podium"]),
                p_points_finish=float(r["p_points_finish"]), p_dnf=float(r["p_dnf"]),
            )
        predictions.append(SessionDriverPrediction(**kwargs))

    return SessionPredictionResponse(
        season=season, round=round_, race_name=race_row["race_name"], session_type=session_type,
        tier=tier, source=source, predictions=predictions,
    )
```

Add the three route handlers after the existing `get_race_prediction`/`_get_race_prediction_live` pair:

```python
@router.get("/races/{season}/{round_}/sprint-qualifying-prediction", response_model=SessionPredictionResponse)
def get_sprint_qualifying_prediction(season: int, round_: int) -> SessionPredictionResponse:
    return _get_session_prediction_response(season, round_, "sprint_qualifying")


@router.get("/races/{season}/{round_}/sprint-prediction", response_model=SessionPredictionResponse)
def get_sprint_prediction(season: int, round_: int) -> SessionPredictionResponse:
    return _get_session_prediction_response(season, round_, "sprint")


@router.get("/races/{season}/{round_}/qualifying-prediction", response_model=SessionPredictionResponse)
def get_qualifying_prediction(season: int, round_: int) -> SessionPredictionResponse:
    return _get_session_prediction_response(season, round_, "qualifying")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_session_prediction_routes.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Run the full existing test suite to confirm no regression**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS (every existing test, plus everything added in Tasks 1–9)

- [ ] **Step 8: Manual smoke test against a running backend**

```bash
PYTHONPATH=src uvicorn f1_predictor.api.main:app --port 8010 &
sleep 2
curl -s http://localhost:8010/api/races/2026/13/qualifying-prediction | python3 -m json.tool | head -20
curl -s -w "\nHTTP:%{http_code}\n" http://localhost:8010/api/races/2026/13/sprint-prediction  # expect 404 if round 13 isn't a sprint weekend, else a real body
kill %1
```

Expected: `qualifying-prediction` returns a 200 with `session_type: "qualifying"` and `p_pole`/`p_top_3`/`p_top_10` fields populated, `p_win`/`p_podium` etc. null. (This will 409 instead if Task 7's `train_all` hasn't actually been run against real data yet — that's expected; run `PYTHONPATH=src python -m f1_predictor.models.manifest` first if so.)

- [ ] **Step 9: Commit**

```bash
git add src/f1_predictor/api/routes.py tests/test_session_prediction_routes.py
git commit -m "$(cat <<'EOF'
Add sprint-qualifying/sprint/qualifying prediction endpoints

Three new GET routes alongside the existing race prediction endpoint
(unchanged), sharing one SessionPredictionResponse schema. Also fixes
routes.py's calls into tracking/store.py's renamed functions and adds
an optional session_type filter to both track-record endpoints.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Frontend — types and API client for the 3 new endpoints

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/api/client.ts`

**Interfaces:**
- Consumes: nothing new.
- Produces: `SessionDriverPrediction` and `SessionPredictionResponse` TypeScript interfaces; `api.sessionPrediction(sessionType, season, round)` client method. Task 11 imports these.

- [ ] **Step 1: Add the types**

Find the existing `RacePredictionResponse` interface in `frontend/src/types.ts` and add directly below it:

```typescript
export interface SessionDriverPrediction {
  driver_id: string;
  constructor_id: string | null;
  p_pole: number | null;
  p_top_3: number | null;
  p_top_10: number | null;
  p_win: number | null;
  p_podium: number | null;
  p_points_finish: number | null;
  p_dnf: number | null;
  expected_position: number;
  actual_position: number | null;
  actual_dnf: boolean | null;
}

export type SessionType = "sprint_qualifying" | "qualifying" | "sprint" | "race";

export interface SessionPredictionResponse {
  season: number;
  round: number;
  race_name: string;
  session_type: SessionType;
  tier: string;
  source: string;
  predictions: SessionDriverPrediction[];
}
```

- [ ] **Step 2: Add the client method**

In `frontend/src/api/client.ts`, add `SessionPredictionResponse` and `SessionType` to the existing `import type { ... } from "../types";` block. Add this method to the `api` object, alongside the existing `racePrediction`:

```typescript
  sessionPrediction: (sessionType: Exclude<SessionType, "race">, season: number, round: number) => {
    const path = sessionType === "sprint_qualifying" ? "sprint-qualifying-prediction"
      : sessionType === "sprint" ? "sprint-prediction"
      : "qualifying-prediction";
    return get<SessionPredictionResponse>(`/races/${season}/${round}/${path}`);
  },
```

- [ ] **Step 3: Verify the frontend still typechecks and builds**

Run: `cd frontend && npx tsc -b --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/types.ts frontend/src/api/client.ts
git commit -m "$(cat <<'EOF'
Add frontend types/client for the 3 new session-prediction endpoints

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Frontend — `SessionPredictionTable`/`SessionPredictionPanel` components

**Files:**
- Create: `frontend/src/components/SessionPredictionTable.tsx`
- Create: `frontend/src/components/SessionPredictionPanel.tsx`

**Interfaces:**
- Consumes: `SessionDriverPrediction`, `SessionPredictionResponse`, `SessionType` (Task 10); existing `ProbabilityHeatCell`, `TeamBadge`, `ResultDelta`, `TierBadge`, `InfoTooltip` components; `driverName`/`teamColor` from `frontend/src/lib/teamColors.ts`; `api.sessionPrediction` (Task 10).
- Produces: `SessionPredictionTable` (props: `predictions: SessionDriverPrediction[]`, `sessionType: SessionType`), `SessionPredictionPanel` (props: `season: number`, `round: number`, `sessionType: Exclude<SessionType, "race">`, `title: string`). Task 12 renders `SessionPredictionPanel` from `RacesPage.tsx`.

This is a new, simpler, read-only component — deliberately NOT a generalization of the existing `DriverPredictionTable` (which has explain-drill-down wiring tied to the race-only `/explain` endpoint, out of scope for the 3 new session types per the spec). No component-level test framework exists in this frontend yet (no `*.test.tsx` files present) — verified with manual browser smoke-testing in Step 3 instead, consistent with this project's own established testing culture (thin backend test suite, browser-driven frontend verification per this project's own CLAUDE.md conventions).

- [ ] **Step 1: Write `SessionPredictionTable.tsx`**

```tsx
// frontend/src/components/SessionPredictionTable.tsx
import type { SessionDriverPrediction, SessionType } from "../types";
import { driverName, teamColor } from "../lib/teamColors";
import { ProbabilityHeatCell } from "./ProbabilityHeatCell";
import { ResultDelta } from "./ResultDelta";
import { TeamBadge } from "./TeamBadge";

function columnMax(predictions: SessionDriverPrediction[], key: keyof SessionDriverPrediction): number {
  const values = predictions.map((p) => (p[key] as number) ?? 0);
  const max = Math.max(...values);
  return max > 0 ? max : 1;
}

interface Props {
  predictions: SessionDriverPrediction[];
  sessionType: SessionType;
}

const IS_QUALI_TYPE: Record<SessionType, boolean> = {
  sprint_qualifying: true,
  qualifying: true,
  sprint: false,
  race: false,
};

export function SessionPredictionTable({ predictions, sessionType }: Props) {
  const rankKey = IS_QUALI_TYPE[sessionType] ? "p_pole" : "p_win";
  const sorted = [...predictions].sort((a, b) => ((b[rankKey] as number) ?? 0) - ((a[rankKey] as number) ?? 0));
  const hasActuals = sorted.some((p) => p.actual_position != null || p.actual_dnf);

  const isQuali = IS_QUALI_TYPE[sessionType];
  const col1Label = isQuali ? "Pole" : "Win";
  const col2Label = isQuali ? "Top 3" : "Podium";
  const col3Label = isQuali ? "Top 10" : "Points";
  const col1Key: keyof SessionDriverPrediction = isQuali ? "p_pole" : "p_win";
  const col2Key: keyof SessionDriverPrediction = isQuali ? "p_top_3" : "p_podium";
  const col3Key: keyof SessionDriverPrediction = isQuali ? "p_top_10" : "p_points_finish";

  const max1 = columnMax(sorted, col1Key);
  const max2 = columnMax(sorted, col2Key);
  const max3 = columnMax(sorted, col3Key);
  const maxDnf = !isQuali ? columnMax(sorted, "p_dnf") : 1;

  const colCount = 5 + (!isQuali ? 1 : 0) + (hasActuals ? 1 : 0);

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-f1-border text-left text-[11px] uppercase tracking-wide text-f1-text-faint">
            <th className="py-2 pr-3 font-medium">#</th>
            <th className="py-2 pr-3 font-medium">Driver</th>
            <th className="py-2 pr-3 font-medium">{col1Label}</th>
            <th className="py-2 pr-3 font-medium">{col2Label}</th>
            <th className="py-2 pr-3 font-medium">{col3Label}</th>
            {!isQuali && <th className="py-2 pr-3 font-medium">DNF</th>}
            {hasActuals && <th className="py-2 pr-3 text-right font-medium">Predicted vs. actual</th>}
          </tr>
        </thead>
        <tbody>
          {sorted.map((p, i) => (
            <tr
              key={p.driver_id}
              className="border-b border-f1-border/60 last:border-0"
              style={{ borderLeft: `2px solid ${teamColor(p.constructor_id)}` }}
            >
              <td className="py-2 pl-2 pr-3 tabular-nums text-f1-text-faint">{i + 1}</td>
              <td className="py-2 pr-3">
                <div className="font-medium text-f1-text">{driverName(p.driver_id)}</div>
                <TeamBadge constructorId={p.constructor_id} />
              </td>
              <td className="py-1.5 pr-2">
                <ProbabilityHeatCell value={(p[col1Key] as number) ?? 0} intensity={((p[col1Key] as number) ?? 0) / max1} color="var(--color-f1-red)" digits={1} />
              </td>
              <td className="py-1.5 pr-2">
                <ProbabilityHeatCell value={(p[col2Key] as number) ?? 0} intensity={((p[col2Key] as number) ?? 0) / max2} color="var(--color-podium)" />
              </td>
              <td className="py-1.5 pr-2">
                <ProbabilityHeatCell value={(p[col3Key] as number) ?? 0} intensity={((p[col3Key] as number) ?? 0) / max3} color="var(--color-win)" />
              </td>
              {!isQuali && (
                <td className="py-1.5 pr-2">
                  <ProbabilityHeatCell value={p.p_dnf ?? 0} intensity={(p.p_dnf ?? 0) / maxDnf} color="var(--color-dnf)" />
                </td>
              )}
              {hasActuals && (
                <td className="py-2 pr-3">
                  <ResultDelta predictedRank={i + 1} actualPosition={p.actual_position} actualDnf={p.actual_dnf} />
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      <p colSpan={colCount} className="sr-only" />
    </div>
  );
}
```

(Note: the stray `<p colSpan={colCount} ...>` above is invalid JSX on a `<p>` tag — remove that line entirely; `colCount` is unused in this simplified table and can be deleted along with its computation. Leaving this note inline rather than silently fixing it in the snippet, so the implementer removes both the `colCount` variable and that line together.)

- [ ] **Step 2: Write `SessionPredictionPanel.tsx`**

```tsx
// frontend/src/components/SessionPredictionPanel.tsx
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { SessionPredictionResponse, SessionType } from "../types";
import { TierBadge } from "./TierBadge";
import { SessionPredictionTable } from "./SessionPredictionTable";

interface Props {
  season: number;
  round: number;
  sessionType: Exclude<SessionType, "race">;
  title: string;
}

export function SessionPredictionPanel({ season, round, sessionType, title }: Props) {
  const [data, setData] = useState<SessionPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    api
      .sessionPrediction(sessionType, season, round)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [season, round, sessionType]);

  if (loading) {
    return <div className="animate-pulse text-sm text-f1-text-faint">Loading {title.toLowerCase()}…</div>;
  }
  if (error) {
    // A 404 here just means "not a sprint weekend" for sprint/sprint-qualifying — render nothing rather than an error box.
    if (error.includes("404") || error.toLowerCase().includes("not a sprint weekend")) return null;
    return <div className="rounded-lg border border-dnf/30 bg-dnf/10 p-4 text-sm text-dnf">{error}</div>;
  }
  if (!data) return null;

  return (
    <div className="clip-corner-lg rounded-lg border border-f1-border bg-f1-850/60 p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-display text-base font-bold text-f1-text">{title}</h3>
        <TierBadge tier={data.tier} />
      </div>
      <SessionPredictionTable predictions={data.predictions} sessionType={data.session_type} />
    </div>
  );
}
```

- [ ] **Step 3: Manual browser smoke test**

```bash
cd frontend && npm run dev -- --port 5180 &
```

Open `http://localhost:5180`, select a race, temporarily render `<SessionPredictionPanel season={...} round={...} sessionType="qualifying" title="Qualifying" />` above the existing `RacePredictionPanel` in `RacesPage.tsx` (this wiring becomes permanent in Task 12 — this step is just to visually confirm the component renders a real table with Pole/Top 3/Top 10 columns and no console errors) via the browser's console/`read_console_messages` tool. Confirm no React errors, then stop the dev server.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/SessionPredictionTable.tsx frontend/src/components/SessionPredictionPanel.tsx
git commit -m "$(cat <<'EOF'
Add SessionPredictionTable/SessionPredictionPanel components

A new, simpler read-only table for the 3 new session types (no
explain-drill-down, unlike DriverPredictionTable, which stays
race-specific) — reuses ProbabilityHeatCell/TeamBadge/ResultDelta/TierBadge.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Frontend — wire the session timeline into `RacesPage.tsx`

**Files:**
- Modify: `frontend/src/pages/RacesPage.tsx`
- Modify: `frontend/src/lib/glossary.ts` (add tier labels for the 2 new tiers)

**Interfaces:**
- Consumes: `SessionPredictionPanel` (Task 11), existing `RacePredictionPanel`, `RaceSummary.is_sprint_weekend` (already exists on the type).
- Produces: the selected race's detail panel renders, top to bottom: `Sprint Qualifying → Sprint → Qualifying → Race` panels on a sprint weekend, `Qualifying → Race` otherwise.

- [ ] **Step 1: Add tier labels/descriptions**

Find `TIER_LABELS`/`TIER_DESCRIPTIONS` in `frontend/src/lib/glossary.ts` and add the two new tier keys (matching the existing entries' style):

```typescript
  post_sprint_qualifying: "Post Sprint Quali",
  post_sprint: "Post Sprint",
```

to `TIER_LABELS`, and:

```typescript
  post_sprint_qualifying: "Computed after sprint qualifying has set the sprint grid.",
  post_sprint: "Computed after the sprint race, using its result as a real input.",
```

to `TIER_DESCRIPTIONS`. (If either constant is typed as `Record<string, string>` this is a plain object literal addition; match whatever the existing entries' exact syntax is.)

- [ ] **Step 2: Wire the timeline into `RacesPage.tsx`**

Replace the existing right-column rendering in `frontend/src/pages/RacesPage.tsx`:

```tsx
      <div>
        {selected ? (
          <RacePredictionPanel season={selected.season} round={selected.round} />
        ) : (
          <div className="text-sm text-f1-text-faint">Select a race.</div>
        )}
      </div>
```

with:

```tsx
      <div className="flex flex-col gap-4">
        {selected ? (
          <>
            {selected.is_sprint_weekend && (
              <SessionPredictionPanel
                season={selected.season}
                round={selected.round}
                sessionType="sprint_qualifying"
                title="Sprint Qualifying"
              />
            )}
            {selected.is_sprint_weekend && (
              <SessionPredictionPanel season={selected.season} round={selected.round} sessionType="sprint" title="Sprint" />
            )}
            <SessionPredictionPanel season={selected.season} round={selected.round} sessionType="qualifying" title="Qualifying" />
            <RacePredictionPanel season={selected.season} round={selected.round} />
          </>
        ) : (
          <div className="text-sm text-f1-text-faint">Select a race.</div>
        )}
      </div>
```

Add `import { SessionPredictionPanel } from "../components/SessionPredictionPanel";` to the top of the file, alongside the existing `RacePredictionPanel` import.

- [ ] **Step 3: Typecheck**

Run: `cd frontend && npx tsc -b --noEmit`
Expected: no errors.

- [ ] **Step 4: Manual browser verification**

Start the backend (`PYTHONPATH=src uvicorn f1_predictor.api.main:app --port 8010 &`) and frontend (`cd frontend && npm run dev -- --port 5180 &`), open `http://localhost:5180` in the Browser tool, click through a few races (at least one sprint weekend and one regular weekend if the current season schedule has both — check `is_sprint_weekend` via `GET /api/races?season=2026`), and confirm: a sprint weekend shows 4 stacked panels in the right order, a regular weekend shows 2, no console errors (`read_console_messages`), and each panel's Tier badge shows a sensible label (not a raw `post_sprint_qualifying` string with no tooltip — confirms Step 1 wired through). Stop both processes when done.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/RacesPage.tsx frontend/src/lib/glossary.ts
git commit -m "$(cat <<'EOF'
Wire the session-prediction timeline into RacesPage

Sprint Qualifying -> Sprint -> Qualifying -> Race on a sprint weekend,
Qualifying -> Race otherwise, all stacked above the existing race
prediction panel.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Track-record reconciliation script for the 3 new session types

**Files:**
- Modify: `src/f1_predictor/tracking/store.py` is already done (Task 6) — this task adds the *caller* that actually snapshots and reconciles the 3 new session types, mirroring how the race model already gets snapshotted somewhere. First, find that existing caller.
- Modify: whichever file currently calls `store.record_race_predictions`/the old `reconcile_predictions` for the race model (search first).
- Test: `tests/test_session_reconciliation.py` (new file)

**Interfaces:**
- Consumes: `store.record_session_predictions`, `store.reconcile_session_predictions` (Task 6); `backtest_session`/`session_outcome`/`session_build` (Tasks 3–5).
- Produces: whatever periodic job already reconciles race predictions now also reconciles sprint_qualifying/qualifying/sprint predictions.

- [ ] **Step 1: Find the existing race-prediction snapshot/reconcile caller**

Run: `grep -rn "record_race_predictions\|reconcile_predictions" src/f1_predictor --include="*.py"`

This will point at the actual current caller (likely `api/main.py`'s lifespan hook, a scheduled script, or `public_snapshot.py`) — read that file's surrounding context before writing Step 2's code, since this task's exact insertion point depends on what that grep finds. Do not guess; the file and function name here depend on that search result.

- [ ] **Step 2: Mirror the same call for the 3 new session types**

Wherever Step 1's grep locates the existing call (e.g., something like `store.record_race_predictions(sim, season, round_, race_name, tier, session_time)` inside a loop over the season's races), add equivalent calls for `sprint_qualifying`, `qualifying`, `sprint` guarded by `is_sprint_weekend` for the two sprint-only ones, using `_predict_upcoming_session`/`session_outcome`/`session_build` (Task 9's helpers) the same way the existing code path uses `_predict_upcoming_race`. Similarly for whatever reconciles resolved predictions once results land — add `store.reconcile_session_predictions(jolpica.load_season_qualifying(season), "qualifying")` and the equivalent for `sprint`/`sprint_qualifying` (using `jolpica.load_season_sprints(season)` for both, per Task 4's `grid`-as-target-for-sprint-qualifying design) alongside whatever the existing race reconciliation call is.

Because Step 1's exact target file is not yet known at plan-writing time, write the concrete code for this step only after running Step 1's grep — do not skip Step 1.

- [ ] **Step 3: Write a reconciliation test**

```python
# tests/test_session_reconciliation.py
import pandas as pd

from f1_predictor.tracking import store


def test_reconcile_session_predictions_qualifying_and_sprint_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "TRACKING_DB_PATH", tmp_path / "tracking.db")

    sim = pd.DataFrame(
        [{"driver_id": "norris", "constructor_id": "mclaren", "p_win": 0.3, "p_podium": 0.6, "p_points_finish": 0.9, "p_dnf": 0.0}]
    )
    store.record_session_predictions(sim, 2024, 1, "Fake GP", "qualifying", "post_practice", "2024-01-01T00:00:00Z")
    store.record_session_predictions(sim, 2024, 1, "Fake GP", "sprint", "post_sprint_qualifying", "2024-01-01T00:00:00Z")

    quali_results = pd.DataFrame([{"season": 2024, "round": 1, "driver_id": "norris", "position": 1, "dnf": False}])
    n_updated = store.reconcile_session_predictions(quali_results, "qualifying")
    assert n_updated == 3  # pole/top_3/top_10

    sprint_rows = store.get_session_prediction(2024, 1, "sprint")
    assert all(r["resolved"] == 0 for r in sprint_rows), "reconciling qualifying must not touch sprint's rows"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src pytest tests/test_session_reconciliation.py -v`
Expected: PASS (1 test — this exercises `store.py` directly, independent of wherever Step 2's caller lives, so it can be written and pass before Step 2's exact integration point is even found)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_session_reconciliation.py <files touched in Step 2>
git commit -m "$(cat <<'EOF'
Wire sprint_qualifying/qualifying/sprint into the prediction snapshot/reconcile job

Mirrors the existing race-prediction snapshot/reconcile call for the 3
new session types, so their Track Record history starts accumulating.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: `evaluate/quali_feature_ablation.py` — the original spike question

**Files:**
- Create: `src/f1_predictor/evaluate/quali_feature_ablation.py`
- Test: `tests/test_quali_feature_ablation.py` (new file)

**Interfaces:**
- Consumes: `session_build.build_session_training_frame("qualifying", ...)` (Task 4), `session_outcome.train_session_ranker`, `session_outcome.predict_session`, `SESSION_SPECS["qualifying"]` (Task 3); `build_features.build_training_frame`, `build_features.FEATURE_COLUMNS`, `session_state.TIER_PRE_WEEKEND`, `session_state.TIER_POST_PRACTICE` (all existing, unchanged); `race_outcome.train_ranker`, `race_outcome.simulate_race`, `dnf.train_dnf_model`, `dnf.predict_dnf_prob` (all existing, unchanged).
- Produces: `run_ablation(seasons: list[int] | None = None) -> dict` returning `{"with_feature": {market: brier_score}, "without_feature": {market: brier_score}, "recommendation": "add" | "no_change"}`, plus a `main()` CLI entry point mirroring `evaluate/backtest.py`'s `argparse` pattern. This script does not get imported by production code (`routes.py`, `manifest.py`) — it's a standalone evaluation tool, run manually, matching `tune_hyperparams.py`'s precedent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quali_feature_ablation.py
import pandas as pd

from f1_predictor.evaluate import quali_feature_ablation


def _fake_race_frame(seasons=None):
    rows = []
    for rnd in range(1, 6):
        for tier in ["pre_weekend", "post_practice", "post_qualifying"]:
            for i, driver in enumerate(["a", "b", "c"], start=1):
                rows.append(
                    {
                        "season": 2024, "round": rnd, "driver_id": driver, "constructor_id": f"t{i % 2}",
                        "tier": tier, "position": i, "dnf": False,
                        "elo_pre_race": 1600 - i * 5, "team_strength_pre_race": 1500.0,
                        "grid": i if tier == "post_qualifying" else float("nan"),
                        "quali_position": i if tier == "post_qualifying" else float("nan"),
                        "quali_gap_to_pole": 0.1 * i if tier == "post_qualifying" else float("nan"),
                    }
                )
    return pd.DataFrame(rows), ["elo_pre_race", "team_strength_pre_race", "grid", "quali_position", "quali_gap_to_pole"]


def _fake_quali_frame(session_type, seasons=None):
    rows = []
    for rnd in range(1, 6):
        for i, driver in enumerate(["a", "b", "c"], start=1):
            rows.append(
                {
                    "season": 2024, "round": rnd, "driver_id": driver, "constructor_id": f"t{i % 2}",
                    "quali_position": i, "elo_pre_race": 1600 - i * 5, "team_strength_pre_race": 1500.0,
                    "sprint_finish_position": float("nan"),
                }
            )
    return pd.DataFrame(rows), ["elo_pre_race", "team_strength_pre_race", "sprint_finish_position"]


def test_run_ablation_returns_brier_scores_for_both_variants(monkeypatch):
    monkeypatch.setattr(quali_feature_ablation.build_features, "build_training_frame", _fake_race_frame)
    monkeypatch.setattr(quali_feature_ablation.session_build, "build_session_training_frame", _fake_quali_frame)

    result = quali_feature_ablation.run_ablation(seasons=[2024])

    assert set(result["with_feature"].keys()) == {"win", "podium", "points_finish", "dnf"}
    assert set(result["without_feature"].keys()) == {"win", "podium", "points_finish", "dnf"}
    assert result["recommendation"] in ("add", "no_change")
    assert all(0.0 <= v <= 1.0 for v in result["with_feature"].values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_quali_feature_ablation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'f1_predictor.evaluate.quali_feature_ablation'`

- [ ] **Step 3: Write `quali_feature_ablation.py`**

```python
# src/f1_predictor/evaluate/quali_feature_ablation.py
"""quali_feature_ablation.py — answers the question this whole session-
predictors build started from: does a predicted-qualifying-position
feature improve the RACE model's PRE_WEEKEND/POST_PRACTICE-tier
predictions (the two tiers where quali_position is still genuinely
unknown, i.e. NaN)? See
docs/superpowers/specs/2026-09-22-session-predictors-design.md §6.

Trains the race model twice via a simple walk-forward split (same
no-lookahead discipline evaluate/backtest.py uses): once on the existing
feature set, once with an added `predicted_quali_position` column (from
the qualifying predictor's own out-of-fold predictions, never overriding
the real quali_position once POST_QUALIFYING data exists), and compares
Brier score across win/podium/points_finish/dnf on PRE_WEEKEND/
POST_PRACTICE rows only — the two tiers a predicted proxy could plausibly
help, since POST_QUALIFYING already has the real value.

This is a standalone evaluation tool (mirrors tune_hyperparams.py) — not
imported by manifest.py or api/routes.py. Run manually; if
result["recommendation"] == "add", add "predicted_quali_position" to
features/build.py::FEATURE_COLUMNS and wire the qualifying predictor's
output into build_training_frame as a follow-up (not done automatically
by this script — a human decision point, consistent with how this
project treats every other model-selection choice).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data import jolpica
from ..features import build as build_features
from ..features import session_build
from ..features import session_state
from ..models import dnf as dnf_model
from ..models import race_outcome
from ..models import session_outcome

_PRE_QUALI_TIERS = (session_state.TIER_PRE_WEEKEND, session_state.TIER_POST_PRACTICE)
_MARKETS = [("win", "p_win"), ("podium", "p_podium"), ("points_finish", "p_points_finish"), ("dnf", "p_dnf")]


def _brier_by_market(sim: pd.DataFrame, actual: pd.DataFrame) -> dict[str, float]:
    """actual must have columns driver_id/season/round/position/dnf,
    already restricted to the rows being scored."""
    merged = sim.merge(actual, on="driver_id", how="inner")
    scores = {}
    for market, prob_col in _MARKETS:
        if market == "dnf":
            outcome = merged["dnf"].astype(int)
        elif market == "win":
            outcome = (merged["position"] == 1).astype(int)
        elif market == "podium":
            outcome = (merged["position"] <= 3).astype(int)
        else:
            outcome = (merged["position"] <= 10).astype(int)
        scores[market] = float(((merged[prob_col] - outcome) ** 2).mean())
    return scores


def _predicted_quali_positions(seasons: list[int]) -> pd.DataFrame:
    """Out-of-fold-ish predicted quali position per (season, round,
    driver_id): for each round, train the qualifying predictor on every
    STRICTLY EARLIER round only (no-lookahead), predict that round. Rounds
    with no earlier training data (the first few of the earliest season)
    are skipped — left NaN downstream, which XGBoost already tolerates."""
    quali_spec = session_outcome.SESSION_SPECS["qualifying"]
    df, feature_cols = session_build.build_session_training_frame("qualifying", seasons=seasons)
    if df.empty:
        return pd.DataFrame(columns=["season", "round", "driver_id", "predicted_quali_position"])

    rows = []
    for (season, round_), _ in df.groupby(["season", "round"]):
        before = (df["season"] < season) | ((df["season"] == season) & (df["round"] < round_))
        train_df = df[before]
        if train_df.empty:
            continue
        target_df = df[(df["season"] == season) & (df["round"] == round_)]
        ranker = session_outcome.train_session_ranker(train_df, feature_cols, quali_spec)
        sim = session_outcome.predict_session(ranker, target_df, feature_cols, quali_spec, n_trials=2000)
        for _, r in sim.iterrows():
            rows.append(
                {"season": season, "round": round_, "driver_id": r["driver_id"], "predicted_quali_position": r["expected_position"]}
            )
    return pd.DataFrame(rows)


def run_ablation(seasons: list[int] | None = None) -> dict:
    seasons = seasons or jolpica.default_seasons()
    df, feature_cols = build_features.build_training_frame(seasons=seasons)
    predicted_quali = _predicted_quali_positions(seasons)

    df_with_feature = df.merge(predicted_quali, on=["season", "round", "driver_id"], how="left")
    df_with_feature["predicted_quali_position"] = df_with_feature["predicted_quali_position"].where(
        df_with_feature["quali_position"].isna(), df_with_feature["quali_position"]
    )
    feature_cols_with = feature_cols + ["predicted_quali_position"]

    rounds = sorted(df["round"].unique())
    split_idx = max(1, len(rounds) * 3 // 4)
    train_rounds = set(rounds[:split_idx])
    test_rounds = set(rounds[split_idx:]) or {rounds[-1]}

    scores_without = _score_variant(df, feature_cols, train_rounds, test_rounds)
    scores_with = _score_variant(df_with_feature, feature_cols_with, train_rounds, test_rounds)

    total_without = sum(scores_without.values())
    total_with = sum(scores_with.values())
    recommendation = "add" if total_with < total_without else "no_change"

    return {"with_feature": scores_with, "without_feature": scores_without, "recommendation": recommendation}


def _score_variant(df: pd.DataFrame, feature_cols: list[str], train_rounds: set, test_rounds: set) -> dict[str, float]:
    post_quali = df[df["tier"] == session_state.TIER_POST_QUALIFYING]
    train_df = post_quali[post_quali["round"].isin(train_rounds)]

    ranker = race_outcome.train_ranker(train_df, feature_cols)
    dnf_clf = dnf_model.train_dnf_model(train_df, feature_cols)

    pre_quali_test = df[(df["tier"].isin(_PRE_QUALI_TIERS)) & (df["round"].isin(test_rounds))]
    if pre_quali_test.empty:
        return {market: float("nan") for market, _ in _MARKETS}

    scores = race_outcome.xgb_scores_for_race(ranker, pre_quali_test, feature_cols)
    theta = race_outcome.theta_from_xgb_scores(scores)
    dnf_prob = dnf_model.predict_dnf_prob(dnf_clf, pre_quali_test, feature_cols)
    sim = race_outcome.simulate_race(theta, dnf_prob=dnf_prob, n_trials=5000, seed=0)

    actual = pre_quali_test.drop_duplicates("driver_id")[["driver_id", "position", "dnf"]]
    return _brier_by_market(sim, actual)


def main() -> None:
    parser = argparse.ArgumentParser(description="Does a predicted-qualifying feature improve the race model?")
    parser.add_argument("--seasons", type=int, nargs="*", default=None)
    args = parser.parse_args()

    result = run_ablation(seasons=args.seasons)
    print("Brier score by market (lower is better), PRE_WEEKEND/POST_PRACTICE rows only:")
    print(f"{'market':<15}{'without':>12}{'with':>12}")
    for market, _ in _MARKETS:
        print(f"{market:<15}{result['without_feature'][market]:>12.4f}{result['with_feature'][market]:>12.4f}")
    print(f"\nRecommendation: {result['recommendation']}")
    if result["recommendation"] == "add":
        print("-> Add 'predicted_quali_position' to features/build.py::FEATURE_COLUMNS.")
    else:
        print("-> No change to the race model's feature set.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_quali_feature_ablation.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `PYTHONPATH=src pytest tests/ -v`
Expected: PASS (every test from every prior task, plus this one)

- [ ] **Step 6: Run the ablation against real data and report the actual answer**

Run: `PYTHONPATH=src python -m f1_predictor.evaluate.quali_feature_ablation`

This requires Task 7's `train_all()` (or at least enough cached jolpica history) to have real qualifying training data available — if it errors on insufficient data, run `PYTHONPATH=src python -m f1_predictor.models.manifest` first. Report the printed recommendation back to the user in chat — this is the actual answer to the original spike question, not something to silently act on. If `recommendation == "add"`, that's a follow-up task (add `predicted_quali_position` to `build_features.FEATURE_COLUMNS` and wire the qualifying predictor into `build_training_frame`) explicitly NOT included in this plan — flag it to the user rather than doing it unprompted, since it changes the production race model's feature set.

- [ ] **Step 7: Commit**

```bash
git add src/f1_predictor/evaluate/quali_feature_ablation.py tests/test_quali_feature_ablation.py
git commit -m "$(cat <<'EOF'
Add evaluate/quali_feature_ablation.py — the original spike question

Trains the race model with vs. without a predicted_quali_position
feature (from the new qualifying predictor's own no-lookahead
out-of-fold predictions) and compares Brier score on PRE_WEEKEND/
POST_PRACTICE rows, where a predicted proxy could plausibly help.
Standalone tool, not wired into production training — a human decision
point if it recommends adding the feature.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Post-plan follow-ups (explicitly out of scope, noted for the user)

- **PUBLIC_MODE snapshot integration:** `public_snapshot.py` doesn't precompute the 3 new endpoints; they always live-compute even in PUBLIC_MODE. Fine for correctness, but means the public deployment takes the same OOM-risk shape PL_Predictor originally had for these 3 endpoints specifically until someone extends the snapshot generator.
- **`championship_projection.py`'s sprint handling** still reuses the main race's theta/DNF vectors for sprint weekends (its own existing docstring already says this) — upgrading it to use the new dedicated sprint predictor's output is a natural follow-up, not required by this plan.
- **`RaceAccuracyEntry`/`get_race_accuracy`-by-race UI** for qualifying/sprint-qualifying's `pole_*`/`top_3_*`/`top_10_*` shaped rows isn't wired into a schema or frontend view in this plan (Task 9's note) — only the aggregate Track Record page (Task 12's tier-badge-level view) covers the 3 new session types end-to-end.
- **The ablation script's `main()` output (Task 14, Step 6)** is the actual, final answer to "does a predicted-qualifying feature help the race model" — report it to the user once this plan is executed; don't assume the answer while implementing.
- **`backtest.py`'s CLI sanity-print (`main()`)** stays race-only in this plan. The spec's §5 directional sanity checks ("actual pole-sitter near the top of predicted order," etc.) for sprint_qualifying/qualifying/sprint are covered by `test_session_outcome.py`'s toy-data ranking assertion (Task 3) and by manually calling `backtest_session(season, round_, session_type=...)` from a Python shell against real cached data, but there's no dedicated `--session-type` CLI flag printing the same human-readable table `backtest.py`'s existing `main()` prints for the race model. Cheap follow-up if wanted: generalize `main()`'s print block the same way Task 5 generalized `backtest_race` itself — the one snag is that `result["dnf"]` doesn't exist for quali-type sessions (no `has_dnf`), so the print logic needs an `if spec.has_dnf` guard before referencing it.
