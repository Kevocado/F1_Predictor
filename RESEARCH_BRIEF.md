# F1 Predictor — Project Brief & Research

## Origin

This is a sibling project to `PL_Predictor` (a Premier League match/scoreline/player predictor) at `~/Documents/Projects/Prem_Predictor/PL_Predictor`, built earlier in the same account. That project established a working pattern — a deep free historical bulk data source + a fast free live-current-season supplement, a model layer evaluated with walk-forward validation, live odds for value-bet detection, honest before/after-verified prediction tracking, FastAPI backend + React/TypeScript/Vite/Tailwind frontend — and a prior research pass (summarized below) confirmed F1 has excellent free data to build something similar on, with real structural differences worth respecting rather than forcing football's shape onto this sport.

**This file is a starting brief, not a plan.** The next step is to open a fresh Claude Code session in this folder and have it read this file, then run its own planning process (explore what exists here — nothing yet — then design and confirm an approach) before writing any code.

## What to build (explicit requirements)

1. **Race predictions** — per-race outcome predictions: winner probability, podium probability, points-finish probability, DNF probability, for every driver on the grid.
2. **Live Drivers' and Constructors' Championship predictions** — projected end-of-season standings that update as the season progresses (not just a snapshot of current points, but a genuine projection over the remaining races, analogous to PL_Predictor's "Projected Table" feature).
3. **A current-race prediction engine that streams live data and predicts on the fly** — while a race is actually happening, consume OpenF1's live feed (~3 second latency — see below) and continuously update "who's likely to win from here" as laps unfold, not just a single pre-race prediction that goes stale at lights-out.
4. **Weekly-retrainable race predictions from qualifying + practice data** — a race weekend's prediction should sharpen as real data comes in: a baseline pre-weekend prediction, refined after FP1/FP2/FP3, refined again after qualifying, so there's always a "clean," current prediction available at every stage of a race weekend rather than one static pre-season number.

## Research: data sources (verified against real URLs/endpoints, not assumed)

### Historical results/standings — `jolpica-f1` (Ergast's successor)
The traditional Ergast API (the standard free F1 historical database, 1950–present) **shut down in early 2025**. Its drop-in replacement is **jolpica-f1** — same endpoint shape, just a new host: `ergast.com/api/f1/` → `api.jolpi.ca/ergast/f1/`. Confirmed coverage: Circuits, Constructors, Constructor Standings, Drivers, Driver Standings, Laps, Pitstops, Qualifying, Races, Results, Seasons, Sprint, Status — 1950-present. Free, no API key, requires a custom User-Agent header. Rate limit: 4 req/s / 500 req/hour unauthenticated. Maintained by volunteers, not an institution — smaller continuity guarantee than a project like football-data.co.uk, but FastF1 itself already depends on it, which is a reasonable proxy for durability. Repo: `github.com/jolpica/jolpica-f1`.

### Lap/telemetry/session data — `FastF1` (Python package)
`github.com/theOehrly/Fast-F1`, `docs.fastf1.dev`. Exposes lap times, sector times, full car telemetry (speed/throttle/brake/RPM/gear/GPS), tyre compound + stint data, weather, event schedule, and session results (practice/qualifying/race) as pandas DataFrames. **Telemetry/lap-level data only goes back to 2018** — session results/schedule for older seasons exist but are thinner. No API key required. Session data is 50–100MB; local caching is described as "highly recommended, not optional." Recent versions have built-in jolpica-f1 support for historical results/standings, so FastF1 can likely be the single primary package for most historical/session needs.

### Live in-weekend data — `OpenF1`
`openf1.org`. The standout source for the live in-race engine specifically: free, no key, no signup, rate limit 3 req/s / 30 req/min. Live positions, lap times, pit stops, team radio, weather, and race-control messages (flags/safety car/incidents), with **~3 second latency during a session** — reported as faster than most TV broadcasts. Historical data outside the live window is also free via the same API. **Coverage starts in 2023** — it's a current-era live layer on top of FastF1's deeper historical one, not a replacement for it (same relationship pulselive.com had to football-data.co.uk in PL_Predictor). This is very likely the backbone of requirement #3 above (the live in-race engine).

### Weather
- **Forecast** (pre-race): Open-Meteo (`open-meteo.com`) — free, no key, works for any date/location. Known reference used by other F1 weather tools.
- **In-session/historical**: OpenF1 (2023+) or FastF1's own weather data (2018+).

### Live odds
**The Odds API does not cover Formula 1 / motorsport at all** — checked directly against their full sport catalog. This is a real, structural gap: PL_Predictor's "value bet" feature has no F1 equivalent unless a different odds source is found later. The project should be scoped around pure prediction/tracking rather than assuming a betting-edge feature is available from day one.

### Safety car / incident-rate features
No ready-made dataset. Derivable (not off-the-shelf) from jolpica-f1's `status` field (retirement reasons) plus OpenF1's race-control messages (2023+), aggregated into a per-circuit rate — real feature-engineering work, not a free import.

### Prediction targets & modeling prior art
Credible, directly-derivable targets from the sources above: qualifying position, race finishing position/points, podium probability, DNF probability. XGBoost/gradient boosting dominates public F1 prediction work (same as football), and **F1-specific Elo/Bradley-Terry prior art is real**, not just theoretical — multiple independent open implementations exist (`matthewperron/f1-elo`, `joemarlo/F1-Elo`, `cbowdon/F1Ranking`, plus the public site `f1elo.com`, "Rating Every F1 Driver Since 1950"). The standard adaptation: treat each race as a round-robin of pairwise driver comparisons, since F1 has no natural single "opponent" the way a football match does — conceptually similar to how `penaltyblog` (used in PL_Predictor) already does Elo/Pi ratings for football, even though the F1 implementation would be new, not reused.

### Data source summary

| Need | Source | Free/key | Coverage |
|---|---|---|---|
| Historical results/standings | jolpica-f1 (Ergast-compatible) | free, no key, 500 req/hr | 1950–present |
| Lap/telemetry/tyres | FastF1 | free, no key | 2018–present |
| Live in-weekend/in-race data | OpenF1 | free, no key, 3 req/s | 2023–present |
| Weather forecast | Open-Meteo | free, no key | any date/location |
| Live odds | — | **none found** | The Odds API doesn't cover F1 |

## Why this is a bigger lift than it might look

Two honest, structural differences from PL_Predictor worth planning around from the start rather than discovering midway:

1. **No odds coverage anywhere** kills the value-bet/edge-detection feature entirely — there's no live market to compare a prediction against. Whatever "is this prediction good" story this project tells will need to come from calibration/backtesting against real results, not a market comparison.
2. **The season shape is fundamentally different.** PL_Predictor's whole schema is built around "one match between two teams." F1 is ~24 discrete race weekends, each with multiple sessions (FP1–3, qualifying, sometimes a sprint, then the race), and the entity that's predicted is a *driver* (who can also change teams mid-career) racing against 19 others simultaneously, not two teams head-to-head. The data model, feature pipeline, and UI (grid/podium/standings visualizations, not a fixtures list) need to be designed for this from the ground up — not adapted from PL_Predictor's schema.

A prior research pass estimated this at roughly 70–80% of PL_Predictor's total build effort — most of the *infrastructure patterns* transfer directly (cache-or-fetch data modules, walk-forward validation discipline, an XGBoost-based model layer, FastAPI + React, honest live-prediction tracking that snapshots before results are known), but the core data model and the value-bet layer do not.

## Suggested starting points for the planning session (not decisions — just where to look first)

- Read `PL_Predictor`'s `README.md` and skim `src/pl_predictor/` for the patterns worth intentionally reusing (the `data/*.py` cache-or-fetch idiom, `evaluate/walk_forward.py`'s validation discipline, `tracking/store.py`'s snapshot-then-reconcile pattern) — but treat it as a reference for *shape*, not code to copy-paste; this is a separate repo, separate git history, separate `pyproject.toml`/`package.json`.
- Requirement #4 (retrain from FP/quali data) implies the model needs a notion of "how much of this weekend's real data do we have yet" feeding into feature freshness — worth designing explicitly rather than bolting on later, e.g. distinct feature sets or confidence tiers for pre-weekend / post-practice / post-qualifying / live-race states.
- Requirement #3 (live in-race engine) is the most novel piece with no direct PL_Predictor analogue — it's a streaming/continuously-updating prediction, not a request/response one. Worth scoping early: how often does it actually need to recompute (every OpenF1 poll? every lap? every position change?), and what's shown to the user while it's running.
- Decide early whether jolpica-f1's volunteer-maintained status is an acceptable risk or whether a lightweight local cache/mirror of historical data is worth building up front (mirrors PL_Predictor's own `data/cache/` pattern either way).
