# F1 Predictor

A self-hosted Formula 1 prediction dashboard: per-race win/podium/points/DNF
probabilities for every driver on the grid, and a Monte Carlo projection of
the final Drivers'/Constructors' Championship standings — served through a
FastAPI backend, with every prediction tracked against what actually
happens so the model can be judged honestly rather than taken on faith.

Sibling project to `PL_Predictor` (Premier League), reusing its
infrastructure patterns (cache-or-fetch data modules, walk-forward
validation, a dual-model race-off scored on held-out log-loss, snapshot-
then-reconcile tracking) but with a data model and feature/model layer
built from scratch for F1's shape: ~24 multi-session race weekends, a
*driver* (not a team) as the predicted entity, no natural single opponent
per race.

## What it does

- **Race predictions** — win/podium/points-finish/DNF probability for
  every driver, derived from ONE coherent Plackett-Luce Monte Carlo
  simulation over the field (not four separately-fit numbers that could
  contradict each other). Two candidate strength models — driver Elo +
  constructor strength (`features/elo.py`, `features/team_strength.py`) vs.
  an XGBoost learning-to-rank model — are raced against each other on a
  held-out season; whichever wins is served. Constructor strength also
  includes a fast-reacting last-3-race form signal
  (`features/team_strength.py::compute_team_rolling_form`), meant to
  surface a sudden pace shift (a post-summer-break upgrade, say) faster
  than the slow-moving season Elo rating — **checked with
  `evaluate/feature_ablation.py`, and honestly it shows no clear held-out
  accuracy improvement yet** (mean log-loss delta ≈ 0, folds split roughly
  evenly on direction). Kept in because it's not hurting and the intuition
  is sound, but this is flagged rather than oversold — see "Judging model
  quality" below.
- **What's driving a prediction** — click any Win/Podium/Points/DNF cell
  in the frontend for a per-driver feature breakdown (exact TreeSHAP
  values from the deployed model, `models/explain.py`,
  `GET /api/races/{season}/{round}/explain?driver_id=...`). Win, Podium,
  and Points-finish share ONE explanation, honestly — they're derived
  from the same underlying strength score, not three separate models;
  DNF gets its own.
- **Weekly-sharpening predictions** — one XGBoost model, trained to
  legitimately understand "not known yet" via `features/session_state.py`'s
  tier-augmentation, rather than four separately trained models per stage.
  A race weekend's prediction sharpens from PRE_WEEKEND to
  POST_QUALIFYING as real grid/qualifying data comes in.
- **Live in-race engine** — while a race is actually running, two more
  XGBoost models (`models/live_win_prob.py`) predict P(win)/P(podium) from
  the exact in-race state (position, gaps, tyres, pit stops, safety car),
  trained on 115k+ lap-by-lap rows of historical replay
  (`data/fastf1_client.py`) and served live from OpenF1
  (`data/openf1.py`, `models/live_poller.py`) via `GET /api/live/current`.
  See `docs/live_engine_design.md` for what's verified vs. still uncertain
  (a genuinely live session hasn't occurred during this build — validated
  via historical replay and a synthetic live-session integration test
  instead).
- **Championship projection** — a 2000-trial Monte Carlo season simulator
  (`models/championship_projection.py`) projects final Drivers'/
  Constructors' standings from the current locked-in points plus simulated
  remaining races.
- **Honest tracking** — every prediction is snapshotted *before* the race
  into a local SQLite store (`tracking/store.py`) and reconciled against
  results as they land. No odds market exists for F1 (checked directly —
  see `RESEARCH_BRIEF.md`), so this before/after track record — broken
  down by tier, and by individual race (`GET /api/track-record/by-race`:
  did the model call the actual winner/podium/points positions?) — is the
  "is this actually working" story in place of a value-bet comparison.
- **React frontend** — four tabs (Races, Live, Championship, Track
  Record) wired to the live API, verified in-browser against real 2026
  season data. Probability cells are shaded relative to the field for a
  given race/market (not an absolute 0–100% scale, since a raw "9.9%"
  reads as low even when it's the field's best shot), and completed races
  show a predicted-rank-vs-actual-finish arrow (▲/▼) next to each driver.

## How it's built

| Layer | Stack |
|---|---|
| Race outcome / DNF models | XGBoost (`rank:pairwise`, `binary:logistic`), custom Elo/Plackett-Luce (numpy) |
| Live in-race models | Two regularized XGBoost `binary:logistic` classifiers, race-level held-out early stopping |
| Backend | FastAPI, served with `uvicorn` |
| Data sources | [jolpica-f1](https://github.com/jolpica/jolpica-f1) (historical results/standings, 1950–present), [FastF1](https://github.com/theOehrly/Fast-F1) (lap/telemetry, live-model training corpus), [OpenF1](https://openf1.org) (live session data), [Open-Meteo](https://open-meteo.com) (weather) |
| Prediction tracking | SQLite (snapshot → reconcile), local to this install |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

<details>
<summary><strong>Hit a <code>ModuleNotFoundError: No module named 'f1_predictor'</code>?</strong></summary>

<br>Some environments silently skip `.pth`-based editable installs entirely
— confirmed directly during this project's build: even a plain, normally-
named `.pth` file (not just pip's `__editable__*.pth`) got silently
ignored, most likely security tooling that strips `.pth` files on this
machine since they can execute arbitrary code via `import` lines. The
reliable fix is `PYTHONPATH`, not another `.pth` file:

```bash
export PYTHONPATH=$(pwd)/src
```

Set this in every shell you run `python`/`uvicorn`/pytest from (or add it
to your shell profile for this project) — it doesn't persist across new
terminal sessions on its own.

</details>

No API keys needed — every data source is free and keyless. jolpica-f1
asks for a descriptive User-Agent header; set one in `.env` if you want
(see `.env.example`), otherwise a default is used.

## Train the models

```bash
python -m f1_predictor.models.manifest
```

Fetches 8 seasons of historical data from jolpica-f1 (cached to
`data/cache/` after the first run — the first run takes several minutes
due to jolpica's rate limit, subsequent runs are fast), builds the
tier-augmented feature frame, races the Elo and XGBoost-ranker candidates
against each other via walk-forward validation (selecting on the
most-recent-season holdout, not the multi-fold average — an early fold
with little training data unfairly penalizes the more data-hungry
XGBoost candidate relative to what it sees once trained on everything),
and writes `models/manifest.json` + the trained model files.

Sanity-check a specific race:

```bash
python -m f1_predictor.evaluate.backtest --season 2026 --round 12
```

Replays a real, already-completed Grand Prix using only its
post-qualifying pre-race information, and prints the resulting probability
table next to what actually happened.

Train the separate live in-race models (needed for `/api/live/current` —
not part of `manifest.py`, since it's a different data shape entirely):

```bash
python -m f1_predictor.models.live_win_prob
```

Builds a lap-by-lap training corpus from FastF1 historical replay
(2022–present — takes a while the first time, ~10-15 minutes per ~100
races; cached after). Sanity-check it lap-by-lap against a real race:

```bash
python -m f1_predictor.evaluate.live_replay --season 2026 --round 12
```

## Judging model quality

Three separate questions, three separate tools — "does the model look
reasonable," "is a specific feature actually earning its place," and
"are the hyperparameters any good," are genuinely different questions and
conflating them hides real answers.

**Is a feature actually helping, not just scoring high on
`feature_importances_`?** Gain-based importance (and even the per-
prediction TreeSHAP values behind the explain ribbon) measure how much the
model *leans* on a feature, not whether leaning on it improved real
held-out accuracy — a feature can look important while adding noise a
simpler model would have ignored.

```bash
python -m f1_predictor.evaluate.feature_ablation --group team_form
python -m f1_predictor.evaluate.feature_ablation --features grid quali_position
```

Trains the xgb_ranker candidate WITH vs. WITHOUT the named feature(s), on
the same walk-forward folds `manifest.py` uses to pick a candidate, and
reports the held-out log-loss delta. Real result from this build: the
`team_form` group (added to react faster to a mid-season pace shift than
the season-long Elo rating) currently shows **no clear effect** — mean
delta ≈ 0, individual folds split roughly evenly on which direction it
helps. Kept in anyway (the intuition is sound and it isn't hurting), but
this is the honest, checked answer rather than an assumed one.

**Are the hyperparameters any good, or just reasonable defaults?**

```bash
python -m f1_predictor.evaluate.tune_hyperparams --target ranker --n-trials 25
python -m f1_predictor.evaluate.tune_hyperparams --target dnf --n-trials 50
python -m f1_predictor.evaluate.tune_live_hyperparams --target win --n-trials 20
python -m f1_predictor.evaluate.tune_live_hyperparams --target podium --n-trials 20
```

Optuna searches, scored the same way each model is actually evaluated
(walk-forward mean log-loss for `ranker`/`dnf`, race-level held-out
log-loss for the live models) — not a single arbitrary split. Each study
is backed by `models/optuna_studies.db` (SQLite), so a search is
resumable: stop it anytime, re-run the same command later to add more
trials rather than starting over. **The next `manifest.py` /
`live_win_prob.py` training run automatically picks up whatever's been
tuned** — `train_all()` loads each study's best params if any trials
exist, falls back to the hand-picked defaults otherwise, and records
which one it used in `manifest.json` / the live training summary.
`evaluate/backtest.py` reads the same persisted params back out of
`manifest.json` too, so an honest replay of a past race matches what
production actually trained with, not silently-different defaults.

Real results from this build (25 ranker trials, 55 DNF trials, 20 trials
each for the two live models):

| Model | Before | After tuning |
|---|---|---|
| race_outcome ranker (walk-forward mean log-loss) | 0.1552 | **0.1303** |
| race_outcome ranker (last-fold log-loss, the selection metric) | 0.1415 | **0.1138** |
| DNF model (walk-forward mean log-loss) | 0.4283 | 0.4239 |
| Live win model (held-out log-loss) | 0.0594 | 0.0576 |
| Live podium model (held-out log-loss) | 0.1211 | 0.1179 |

The ranker's improvement is the standout — a ~16% relative drop in the
metric `manifest.py` actually races candidates on, from just widening
`max_depth`/`reg_lambda`/`reg_alpha` beyond the hand-picked defaults. All
of these are 20-50 trial runs, not exhaustive searches — re-running with
`--n-trials` set higher (the studies resume, they don't restart) would
likely find more.

## Run the API

```bash
source .venv/bin/activate
uvicorn f1_predictor.api.main:app --reload --host 0.0.0.0 --port 8000
```

Key endpoints (full list at `/docs`):

- `GET /api/races?season=2026` — season schedule
- `GET /api/races/{season}/{round}/prediction` — that race's probability
  table (live-computed if upcoming, served from the tracking store — or
  honestly reconstructed if never snapshotted — if already completed)
- `GET /api/races/{season}/{round}/explain?driver_id=norris` — per-driver
  TreeSHAP feature attribution behind that prediction (one explanation for
  win/podium/points, a separate one for DNF)
- `GET /api/championship/{drivers|constructors}?season=2026` — Monte Carlo
  championship projection
- `GET /api/track-record?tier=post_qualifying` — honest calibration
  summary
- `GET /api/track-record/by-race?tier=post_qualifying` — did each race's
  prediction call the actual winner/podium/points positions?
- `GET /api/live/current` — the live in-race engine's current prediction;
  `{"live": false}` when no session is active
- `POST /api/retrain` — retrain all models on the latest data

## Run the frontend

React 19 + TypeScript + Vite + Tailwind v4, in `frontend/`. Requires the
API running on port 8000 (above) — the frontend calls
`{protocol}//{hostname}:8000/api` derived from wherever the page itself
was loaded from, so it also works from another device on the LAN/Tailscale
without any config.

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:5174`. Four tabs: **Races** (season schedule,
click a race for its probability table — pre-weekend/post-qualifying tier
badge and live/tracked/backtest source badge shown next to it, plus a
predicted-vs-actual arrow once a race is resolved), **Live** (the in-race
engine's current prediction, polls every 10s, shows "no live session"
otherwise), **Championship** (Drivers'/Constructors' projection),
**Track Record** (honest calibration by tier, and by individual race).

## Project layout

```
src/f1_predictor/
  config.py                    paths, env loading, shared constants
  data/                        jolpica.py, entities.py, open_meteo.py, fastf1_client.py, openf1.py — data modules
  features/                    elo.py, team_strength.py (+ rolling team form), rolling_form.py, circuit.py,
                                safety_car.py, weather.py, session_state.py, build.py
  models/                      race_outcome.py, dnf.py, championship_projection.py, manifest.py,
                                live_win_prob.py, live_poller.py, explain.py
  evaluate/                    walk_forward.py, backtest.py, live_replay.py, feature_ablation.py,
                                tune_hyperparams.py, tune_live_hyperparams.py
  tracking/                    store.py
  api/                         main.py, routes.py, schemas.py
frontend/src/
  api/client.ts, types.ts, lib/{teamColors,glossary,format}.ts
  components/                  DriverPredictionTable, ChampionshipTable, RaceAccuracyTable, RaceRow,
                                ProbabilityHeatCell, ResultDelta, ExplainRibbon, TierBadge, TeamBadge, ...
  pages/                       RacesPage, LivePage, ChampionshipPage, TrackRecordPage
docs/
  live_engine_design.md        live in-race engine: what's built, what's verified, what's still uncertain
```

See `RESEARCH_BRIEF.md` for the original requirements/research this was
built from, and `docs/live_engine_design.md` for the live engine's
verified-vs-uncertain details.
