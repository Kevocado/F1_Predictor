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
  held-out season; whichever wins is served.
- **Weekly-sharpening predictions** — one XGBoost model, trained to
  legitimately understand "not known yet" via `features/session_state.py`'s
  tier-augmentation, rather than four separately trained models per stage.
  A race weekend's prediction sharpens from PRE_WEEKEND to
  POST_QUALIFYING as real grid/qualifying data comes in.
- **Championship projection** — a 2000-trial Monte Carlo season simulator
  (`models/championship_projection.py`) projects final Drivers'/
  Constructors' standings from the current locked-in points plus simulated
  remaining races.
- **Honest tracking** — every prediction is snapshotted *before* the race
  into a local SQLite store (`tracking/store.py`) and reconciled against
  results as they land. No odds market exists for F1 (checked directly —
  see `RESEARCH_BRIEF.md`), so this before/after track record — broken
  down by tier — is the "is this actually working" story in place of a
  value-bet comparison.

- **React frontend** — three tabs (Races, Championship, Track Record)
  wired to the live API, verified in-browser against real 2026 season
  data.

**Not yet built**: the live in-race engine (requirement 3 — streams
OpenF1's live feed during an actual race). One building block exists and
is tested (`data/fastf1_client.py::build_lap_snapshots`, confirmed against
a real 2024 race); the trained model, OpenF1 live-polling client, and
API/serving layer are designed in detail but not implemented — see
`docs/live_engine_design.md`.

## How it's built

| Layer | Stack |
|---|---|
| Race outcome / DNF models | XGBoost (`rank:pairwise`, `binary:logistic`), custom Elo/Plackett-Luce (numpy) |
| Backend | FastAPI, served with `uvicorn` |
| Data sources | [jolpica-f1](https://github.com/jolpica/jolpica-f1) (historical results/standings, 1950–present), [FastF1](https://github.com/theOehrly/Fast-F1) (lap/telemetry, reserved for Phase 3), [OpenF1](https://openf1.org) (live session data, reserved for Phase 3), [Open-Meteo](https://open-meteo.com) (weather) |
| Prediction tracking | SQLite (snapshot → reconcile), local to this install |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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
- `GET /api/championship/{drivers|constructors}?season=2026` — Monte Carlo
  championship projection
- `GET /api/track-record?tier=post_qualifying` — honest calibration
  summary
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

Opens at `http://localhost:5174`. Three tabs: **Races** (season schedule,
click a race for its probability table — pre-weekend/post-qualifying tier
badge and live/tracked/backtest source badge shown next to it),
**Championship** (Drivers'/Constructors' projection), **Track Record**
(honest calibration by tier).

## Project layout

```
src/f1_predictor/
  config.py                    paths, env loading, shared constants
  data/                        jolpica.py, entities.py, open_meteo.py, fastf1_client.py — cache-or-fetch data modules
  features/                    elo.py, team_strength.py, rolling_form.py, circuit.py, safety_car.py,
                                weather.py, session_state.py, build.py
  models/                      race_outcome.py, dnf.py, championship_projection.py, manifest.py
  evaluate/                    walk_forward.py, backtest.py
  tracking/                    store.py
  api/                         main.py, routes.py, schemas.py
frontend/src/
  api/client.ts, types.ts, lib/{teamColors,glossary,format}.ts
  components/                  DriverPredictionTable, ChampionshipTable, RaceRow, TierBadge, TeamBadge, ...
  pages/                       RacesPage, ChampionshipPage, TrackRecordPage
docs/
  live_engine_design.md        detailed design for the not-yet-built live in-race engine (requirement 3)
```

See `RESEARCH_BRIEF.md` for the original requirements/research this was
built from, and `docs/live_engine_design.md` for what's left.
