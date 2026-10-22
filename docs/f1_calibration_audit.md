# F1 calibration audit (walk-forward)

Generated 2026-10-22. XGB ranker, hold-out 2024 season (24 races, 479 driver-race observations). Race-level paired-bootstrap 95% CIs over 2000 resamples. Research only: no model output changed, nothing under src/.

Ideals: reliability slope 1.00; bias 0; log-loss comparable to baseline.

## Grid distribution

- Hold-out season: 2024 only (trained on 2021-2023).
- 24 races, 479 driver-race observations.
- Grid distribution: Grid 0: 1, Grid 1: 24, Grid 2: 24, Grid 3: 24, Grid 4: 24, Grid 5: 24, Grid 6: 24, Grid 7: 24, Grid 8: 24, Grid 9: 24, Grid 10: 24, Grid 11: 24, Grid 12: 24, Grid 13: 24, Grid 14: 24, Grid 15: 24, Grid 16: 24, Grid 17: 24, Grid 18: 24, Grid 19: 24, Grid 20: 22.

## Overall reliability (race-level bootstrap, 2000 resamples)

| Market | Reliability slope | 95% CI | Log-loss | Verdict |
|--------|-------------------|--------|----------|---------|
| Win | 3.060 | [2.270, 3.690] | 0.1580 | PROBLEM |
| Podium | 2.199 | [1.766, 2.580] | 0.3353 | PROBLEM |
| Points finish | 2.019 | [1.811, 2.191] | 0.5173 | PROBLEM |

## Per-grid calibration gaps (race-level bootstrap, 2000 resamples)

| Grid | n | Win gap | 95% CI | Podium gap | 95% CI | PtsFinish gap | 95% CI |
|------|---|---------|--------|------------|--------|---------------|--------|
|  1 |  24 | +0.328 | [+0.151, +0.506] | +0.401 | [+0.208, +0.581] | +0.104 | [-0.047, +0.215] |
|  2 |  24 | +0.205 | [+0.041, +0.376] | +0.498 | [+0.311, +0.645] | +0.230 | [+0.137, +0.294] |
|  3 |  24 | +0.003 | [-0.084, +0.128] | +0.261 | [+0.089, +0.482] | +0.209 | [+0.097, +0.301] |
|  4 |  24 | +0.011 | [-0.074, +0.133] | +0.240 | [+0.042, +0.462] | +0.291 | [+0.195, +0.359] |
|  5 |  24 | -0.019 | [-0.065, +0.062] | +0.107 | [-0.064, +0.281] | +0.387 | [+0.351, +0.424] |
|  6 |  24 | -0.065 | [-0.073, -0.058] | -0.156 | [-0.216, -0.051] | +0.164 | [-0.010, +0.307] |
|  7 |  24 | -0.056 | [-0.065, -0.049] | -0.125 | [-0.186, -0.036] | +0.274 | [+0.112, +0.405] |
|  8 |  24 | -0.047 | [-0.054, -0.041] | -0.142 | [-0.160, -0.126] | +0.329 | [+0.175, +0.450] |
|  9 |  24 | -0.041 | [-0.045, -0.037] | -0.124 | [-0.137, -0.112] | +0.209 | [+0.016, +0.388] |
| 10 |  24 | -0.040 | [-0.046, -0.035] | -0.082 | [-0.132, +0.007] | +0.049 | [-0.123, +0.232] |
| 11 |  24 | -0.046 | [-0.057, -0.039] | -0.140 | [-0.170, -0.119] | -0.198 | [-0.353, -0.019] |
| 12 |  24 | -0.034 | [-0.036, -0.032] | -0.103 | [-0.110, -0.097] | -0.280 | [-0.402, -0.118] |
| 13 |  24 | -0.032 | [-0.035, -0.028] | -0.057 | [-0.105, +0.029] | -0.056 | [-0.246, +0.140] |
| 14 |  24 | -0.031 | [-0.034, -0.029] | -0.100 | [-0.108, -0.094] | -0.265 | [-0.374, -0.131] |
| 15 |  24 | -0.029 | [-0.031, -0.027] | -0.091 | [-0.095, -0.087] | -0.116 | [-0.280, +0.054] |
| 16 |  24 | -0.031 | [-0.036, -0.026] | -0.096 | [-0.110, -0.084] | -0.255 | [-0.369, -0.113] |
| 17 |  24 | +0.011 | [-0.032, +0.093] | -0.057 | [-0.103, +0.022] | -0.257 | [-0.356, -0.128] |
| 18 |  24 | -0.028 | [-0.031, -0.026] | -0.092 | [-0.099, -0.086] | -0.282 | [-0.374, -0.151] |
| 19 |  24 | -0.029 | [-0.035, -0.024] | -0.049 | [-0.098, +0.037] | -0.226 | [-0.315, -0.098] |
| 20 |  22 | -0.029 | [-0.033, -0.027] | -0.095 | [-0.106, -0.086] | -0.324 | [-0.371, -0.245] |

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

1. **Reliability slope CI includes 1.0** after recalibration (paired race-level bootstrap on held-out 2024 season, 2000 resamples).
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

## Single-season caveat & next steps

- **This audit covers ONE hold-out season (2024, 24 races) trained on 2021-2023.**
- The grid-1/2 under-confidence (pole sitters won ~46% of 24 races, model predicted ~13%) is a SINGLE-SEASON result.
- Next step before any recalibration: run the same test with 2022 and 2023 each as separate hold-outs (expanding-window training: train on 2021→hold-out 2022; train on 2021-22→hold-out 2023), not just 2024. Only if the under-confidence persists across hold-out seasons should a fix be considered.
- Ship rule unchanged: slope CI must contain 1.0 AND log-loss must not be worse (paired-bootstrap CI excluding a positive deviation).

## Normalization check

- Probabilities ARE normalized per race: Σ p_win = 1.0, Σ p_podium = 3.0, Σ p_points_finish = 10.0 per race. The under-confidence is a real calibration issue, not a normalization bug.

## Verdict

- **No fix shipped**: research only; fix only if both ship-test conditions pass on re-tested candidates.
- Current model is systematically under-confident across all three markets (win, podium, points_finish), most severely for pole-sitter win and podium probabilities.
- Recalibration is a candidate fix but must pass the dual ship-test above before deployment.
- Before any recalibration, expand the audit to 2022 and 2023 hold-outs to confirm the pattern is not a single-season artifact.
