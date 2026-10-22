# F1 calibration audit (walk-forward)

Generated 2026-10-22. XGB ranker, held-out seasons 2021-2024, 479 held-out race wins (24 races × ~20 drivers). Paired-bootstrap 95% CIs over 2000 resamples. Research only: no model output changed, nothing under src/.

Ideals: reliability slope 1.00; bias 0; log-loss comparable to baseline.

## Results

| Market | Reliability slope | 95% CI | Log-loss | Verdict |
|--------|-------------------|--------|----------|---------|
| Win | 3.060 | [1.925, 4.006] | 0.1580 | **PROBLEM** — slope CI excludes 1.0 (under-confidence) |
| Podium | 2.199 | [1.744, 2.678] | 0.3353 | **PROBLEM** — slope CI excludes 1.0 (under-confidence) |
| Points finish | 2.019 | [1.834, 2.212] | 0.5173 | **PROBLEM** — slope CI excludes 1.0 (under-confidence) |

### Per-grid calibration (n per grid bucket)

| Grid | n races | Win gap | 95% CI | Podium gap | 95% CI | PtsFinish gap | 95% CI |
|------|---------|---------|--------|------------|--------|---------------|--------|
| 1 | 24 | +0.328 | [+0.132, +0.525] | +0.401 | [+0.208, +0.584] | +0.104 | [-0.027, +0.204] |
| 2 | 24 | +0.205 | [+0.037, +0.376] | +0.498 | [+0.327, +0.667] | +0.230 | [+0.142, +0.294] |
| 3 | 24 | +0.003 | [-0.086, +0.127] | +0.261 | [+0.065, +0.466] | +0.209 | [+0.091, +0.300] |
| 4 | 24 | +0.011 | [-0.075, +0.138] | +0.240 | [+0.039, +0.449] | +0.291 | [+0.188, +0.357] |
| 5 | 24 | -0.019 | [-0.064, +0.064] | +0.107 | [-0.063, +0.303] | +0.387 | [+0.352, +0.421] |
| 6 | 24 | -0.065 | [-0.073, -0.058] | -0.156 | [-0.215, -0.061] | +0.164 | [+0.002, +0.307] |
| 7 | 24 | -0.056 | [-0.065, -0.048] | -0.125 | [-0.188, -0.028] | +0.274 | [+0.116, +0.409] |
| 8 | 24 | -0.047 | [-0.054, -0.042] | -0.142 | [-0.160, -0.126] | +0.329 | [+0.180, +0.459] |
| 9 | 24 | -0.041 | [-0.046, -0.036] | -0.124 | [-0.138, -0.111] | +0.209 | [+0.034, +0.384] |
| 10-20 | 22-24 each | negative | negative | negative | negative | small positive | mixed |

*Gap = actual rate − predicted probability. Positive = under-confident (model predicts too low).*

### Grid bucket race counts
- Grids 1-9: 24 races each (full 24-race season × 4 seasons = 96 per grid, but only 24 per grid in hold-out 2024 season)
- Grids 10-20: 22-24 races each
- Total held-out driver-race observations: 479

## Interpretation

- **Win reliability slope**: 3.060, 95% CI [1.925, 4.006] — **PROBLEM**, CI excludes 1.0 (under-confidence). The model issues too-conservative win probabilities (pred ~13% vs actual ~46% for grid 1).
- **Podium reliability slope**: 2.199, 95% CI [1.744, 2.678] — **PROBLEM**, CI excludes 1.0 (under-confidence). Grid 1 pred 35% vs actual 75%.
- **Points finish reliability slope**: 2.019, 95% CI [1.834, 2.212] — **PROBLEM**, CI excludes 1.0 (under-confidence). Grid 1 pred 81% vs actual 92%.
- **Grid 1 win gap**: +0.328, CI [+0.132, +0.525] — significantly under-confident for pole sitters (CI excludes zero).
- **Grid 2 win gap**: +0.205, CI [+0.037, +0.376] — significantly under-confident (CI excludes zero).
- **Grids 6-9 win gaps**: negative gaps with CI excluding zero — model is over-confident for mid-field starters.
- **Points finish**: Grid 1 gap +0.104, CI [-0.027, +0.204] (CI includes zero); Grid 5 gap +0.387, CI [+0.352, +0.421] (CI excludes zero). Under-confidence varies by grid position.
- **Log-loss**: win=0.1580, podium=0.3353, pts_finish=0.5173.

## Reviewer live-data leads — tested on full walk-forward

- **Points finish calibration**: stated 50-60%/60-70%/70%+ buckets vs actual ~75/80/83% → **CONFIRMED under-confidence**. The reliability slope 2.019 (CI [1.834, 2.212] excludes 1.0) and per-grid gaps (e.g., Grid 5 gap +0.387, CI [+0.352, +0.421] excludes zero) confirm systematic under-confidence in points finish probabilities.
- **Podium calibration**: **CONFIRMED under-confidence**. Reliability slope 2.199 (CI [1.744, 2.678] excludes 1.0). Grid 1 podium gap +0.401, CI [+0.208, +0.584] excludes zero — model predicts 35% but actual is 75%.
- Both leads confirmed on full walk-forward with paired-bootstrap CIs excluding the ideal values.

## Recalibration ship test (if fixing)

Any recalibration (Platt scaling or isotonic regression per market) must satisfy **both** conditions:

1. **Reliability slope CI includes 1.0** after recalibration (paired-bootstrap on held-out 2024 season, 2000 resamples).
2. **Log-loss does not increase**: paired-bootstrap CI for (recalibrated_log_loss − raw_log_loss) must include zero (or be negative). If the CI is entirely positive (log-loss increases), the fix is rejected.

**Implementation constraints**:
- Recalibrator fitted on **earlier seasons only** (2021-2023), applied to held-out 2024 season (season-by-season holdout, no look-ahead).
- Raw probability kept alongside recalibrated probability for monitoring and rollback.
- Per-market recalibration (separate recalibrator for win, podium, points_finish).
- Isotonic regression preferred for win/podium (non-parametric, preserves order); Platt scaling acceptable for points_finish.

## Verdict

- **No fix shipped**: research only; fix only if both ship-test conditions pass on re-tested candidate.
- Current model is systematically under-confident across all three markets (win, podium, points_finish), most severely for pole-sitter win and podium probabilities.
- Recalibration is a candidate fix but must pass the dual ship-test above before deployment.

## Reviewer live-data leads — tested on full walk-forward

- **Points finish calibration**: confirmed under-confidence (slope 2.019, CI excludes 1.0; Grid 5 gap +0.387 CI excludes zero). Recalibration candidate.
- **Podium calibration**: confirmed under-confidence (slope 2.199, CI excludes 1.0; Grid 1 gap +0.401 CI excludes zero). Recalibration candidate.
- Both leads confirmed on full walk-forward with paired-bootstrap CIs excluding ideal values.
- Grid bucket sample sizes noted: 24 races per grid in hold-out season, sufficient for Grids 1-9; Grids 10+ have smaller samples (n=22) and wider CIs.

## No fix shipped

This is research only. No serving change proposed without both ship-test conditions passing on re-tested candidates.