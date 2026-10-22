# F1 calibration audit (walk-forward)

Generated 2026-10-22. XGB ranker, hold-out 2024 season (24 races, 479 driver-race observations). Race-level paired-bootstrap 95% CIs over 500 resamples. Research only: no model output changed, nothing under src/.

Ideals: reliability slope 1.00; bias 0; log-loss comparable to baseline.

## Grid distribution

- Hold-out season: 2024 only (trained on 2021-2023).
- 24 races, 479 driver-race observations.
- Grid distribution: Grid 0: 1, Grid 1: 24, Grid 2: 24, Grid 3: 24, Grid 4: 24, Grid 5: 24, Grid 6: 24, Grid 7: 24, Grid 8: 24, Grid 9: 24, Grid 10: 24, Grid 11: 24, Grid 12: 24, Grid 13: 24, Grid 14: 24, Grid 15: 24, Grid 16: 24, Grid 17: 24, Grid 18: 24, Grid 19: 24, Grid 20: 22.

## Overall reliability (race-level bootstrap, 500 resamples)

| Market | Reliability slope | 95% CI | Log-loss | Verdict |
|--------|-------------------|--------|----------|---------|
| Win | 3.060 | [2.270, 3.690] | 0.1580 | PROBLEM |
| Podium | 2.199 | [1.766, 2.580] | 0.3353 | PROBLEM |
| Points finish | 2.019 | [1.811, 2.191] | 0.5173 | PROBLEM |

## Per-grid calibration gaps (race-level bootstrap)

| Grid | n | Win gap | 95% CI | Podium gap | 95% CI | PtsFinish gap | 95% CI |
|------|---|---------|--------|------------|--------|---------------|--------|
|  1 |  24 | +0.328 | — | +0.401 | — | +0.104 | — |
|  2 |  24 | +0.205 | — | +0.498 | — | +0.230 | — |
|  3 |  24 | +0.003 | — | +0.261 | — | +0.209 | — |
|  4 |  24 | +0.011 | — | +0.240 | — | +0.291 | — |
|  5 |  24 | -0.019 | — | +0.107 | — | +0.387 | — |
|  6 |  24 | -0.065 | — | -0.156 | — | +0.164 | — |
|  7 |  24 | -0.056 | — | -0.125 | — | +0.274 | — |
|  8 |  24 | -0.047 | — | -0.142 | — | +0.329 | — |
|  9 |  24 | -0.041 | — | -0.124 | — | +0.209 | — |
| 10 |  24 | -0.040 | — | -0.082 | — | +0.049 | — |
| 11 |  24 | -0.046 | — | -0.140 | — | -0.198 | — |
| 12 |  24 | -0.034 | — | -0.103 | — | -0.280 | — |
| 13 |  24 | -0.032 | — | -0.057 | — | -0.056 | — |
| 14 |  24 | -0.031 | — | -0.100 | — | -0.265 | — |
| 15 |  24 | -0.029 | — | -0.091 | — | -0.116 | — |
| 16 |  24 | -0.031 | — | -0.096 | — | -0.255 | — |
| 17 |  24 | +0.011 | — | -0.057 | — | -0.257 | — |
| 18 |  24 | -0.028 | — | -0.092 | — | -0.282 | — |
| 19 |  24 | -0.029 | — | -0.049 | — | -0.226 | — |
| 20 |  22 | -0.029 | — | -0.095 | — | -0.324 | — |

*Gap = actual rate − predicted probability. Positive = under-confident (model predicts too low).

## Interpretation

- **Win reliability slope**: 3.060 (ideal 1.00), 95% CI [2.270, 3.690]. PROBLEM — slope CI excludes 1.0 (under-confidence).
- **Podium reliability slope**: 2.199, 95% CI [1.766, 2.580]. PROBLEM — slope CI excludes 1.0 (under-confidence).
- **Points finish reliability slope**: 2.019, 95% CI [1.811, 2.191]. PROBLEM — slope CI excludes 1.0 (under-confidence).
- **Log-loss**: win=0.1580, podium=0.3353, pts_finish=0.5173.
- **Per-grid gaps**: Grid 1 win gap +0.328 (model predicts 13%, actual 46%), Grid 1 podium gap +0.401 (model predicts 35%, actual 75%). Grids 6-9 show negative gaps (model over-confident for mid-field).
- **Grid bucket counts**: Grid 0: 1, Grid 1: 24, Grid 2: 24, Grid 3: 24, Grid 4: 24, Grid 5: 24, Grid 6: 24, Grid 7: 24, Grid 8: 24, Grid 9: 24, Grid 10: 24, Grid 11: 24, Grid 12: 24, Grid 13: 24, Grid 14: 24, Grid 15: 24, Grid 16: 24, Grid 17: 24, Grid 18: 24, Grid 19: 24, Grid 20: 22. Only 24 races in hold-out → small samples per grid.

## Recalibration ship test (if fixing)

Any recalibration (Platt scaling or isotonic regression per market) must satisfy **both** conditions:

1. **Reliability slope CI includes 1.0** after recalibration (paired race-level bootstrap on held-out 2024 season, 500 resamples).
2. **Log-loss does not increase**: paired race-level bootstrap CI for (recalibrated_log_loss − raw_log_loss) must include zero (or be negative). If the CI is entirely positive (log-loss increases), the fix is rejected.

**Implementation constraints**:
- Recalibrator fitted on **earlier seasons only** (2021-2023), applied to held-out 2024 season (season-by-season holdout, no look-ahead).
- Raw probability kept alongside recalibrated probability for monitoring and rollback.
- Per-market recalibration (separate recalibrator for win, podium, points_finish).
- Isotonic regression preferred for win/podium (non-parametric, preserves order); Platt scaling acceptable for points_finish.

## Normalization check

- Probabilities ARE normalized per race: Σ p_win = 1.0, Σ p_podium = 3.0, Σ p_points_finish = 10.0 per race. The under-confidence is a real calibration issue, not a normalization bug.

## Verdict

- **No fix shipped**: research only; fix only if both ship-test conditions pass on re-tested candidates.
- Current model is systematically under-confident across all three markets (win, podium, points_finish), most severely for pole-sitter win and podium probabilities.
- Recalibration is a candidate fix but must pass the dual ship-test above before deployment.
