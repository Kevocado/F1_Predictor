# F1 temperature-scaling ship test (expanding-window hold-outs)

Generated 2026-10-22. Race-level paired bootstrap, 2000 resamples, resampling whole races. Research + ship test for race-level temperature scaling of the win / podium / points-finish markets.

## Hold-out results

| Hold-out | Train | Test | Market | T | Raw slope (CI) | Corrected slope (CI) | Raw LL | Corrected LL | dLL (CI) | Slope ok | Loss ok | Grid ok | Ship |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 2021 | 2022 | win | 2.95 | 2.315 [1.662,2.819] | 0.816 [0.486,1.055] | 0.1698 | 0.1601 | -0.0102 [-0.0282,+0.0076] | yes | NO | NO (4,5,6) | no |
| 2022 | 2021 | 2022 | podium | 1.70 | 1.853 [1.497,2.152] | 1.067 [0.831,1.254] | 0.3479 | 0.3261 | -0.0221 [-0.0374,-0.0060] | yes | yes | yes | **SHIP** |
| 2022 | 2021 | 2022 | points | 1.05 | 1.356 [1.159,1.539] | 1.296 [1.107,1.472] | 0.5715 | 0.5686 | -0.0029 [-0.0046,-0.0012] | NO | yes | yes | no |
| 2023 | 2021,2022 | 2023 | win | 3.20 | 3.070 [2.538,3.658] | 1.055 [0.783,1.300] | 0.1567 | 0.1202 | -0.0363 [-0.0494,-0.0234] | yes | yes | yes | **SHIP** |
| 2023 | 2021,2022 | 2023 | podium | 1.80 | 1.894 [1.537,2.237] | 0.984 [0.771,1.187] | 0.3488 | 0.3273 | -0.0213 [-0.0357,-0.0064] | yes | yes | NO (3) | no |
| 2023 | 2021,2022 | 2023 | points | 1.25 | 1.483 [1.228,1.714] | 1.166 [0.961,1.351] | 0.5631 | 0.6651 | +0.1033 [-0.0215,+0.2779] | yes | NO | yes | no |
| 2024 | 2021,2022,2023 | 2024 | win | 3.95 | 3.060 [2.270,3.690] | 0.637 [0.363,0.896] | 0.1580 | 0.1416 | -0.0158 [-0.0465,+0.0188] | NO | NO | NO (3) | no |
| 2024 | 2021,2022,2023 | 2024 | podium | 1.50 | 2.199 [1.766,2.580] | 1.269 [0.970,1.550] | 0.3353 | 0.3129 | -0.0219 [-0.0358,-0.0040] | yes | yes | NO (6) | no |
| 2024 | 2021,2022,2023 | 2024 | points | 1.30 | 2.019 [1.811,2.191] | 1.596 [1.442,1.730] | 0.5173 | 0.5299 | +0.0169 [-0.0422,+0.1270] | NO | NO | yes | no |

## Ship rule

A market ships only if it passes all three tests on **all three** hold-outs:

1. Corrected reliability slope CI contains 1.0.
2. Paired race-level log-loss delta (corrected − raw) CI entirely < 0.
3. No grid bucket's paired log-loss delta CI is entirely > 0.

## Why are front-row starters under-predicted?

| Hold-out | Grid 1-2 n | Grid 1-2 pred | Grid 1-2 actual | Grid 3-5 n | Grid 3-5 pred | Grid 3-5 actual |
|---|---|---|---|---|---|---|
| 2022 | 44 | 0.089 | 0.341 | 66 | 0.072 | 0.061 |
| 2023 | 44 | 0.101 | 0.386 | 66 | 0.074 | 0.030 |
| 2024 | 48 | 0.109 | 0.375 | 72 | 0.071 | 0.069 |

**The front row is under-predicted, and the model's grid feature is out-weighted for exactly those rows.** Across the three hold-outs, grid 1-2 drivers are predicted at 0.089/0.101/0.109 while actually winning 0.341/0.386/0.375 — a gap of roughly 25 percentage points, and the same shape in all three hold-outs. Grid 3-5 drivers, by contrast, are close to the mark (predicted 0.071 vs actual 0.069 in the 2024 hold-out), so the miscalibration is specific to the very front of the grid rather than a uniform flatness across the field. The cause is visible in the ranker's feature importances for the 2024 hold-out model: `grid` carries importance 0.0420 and ranks 7 of 23, while `quali_position` carries 0.1273 and ranks 1 — a 3.0x difference. The post-qualifying tier feeds BOTH `grid` and `quali_position` (the model's grid feature is the qualifying position when the official grid is not yet known), so the ranker is effectively seeing the same signal twice and spending most of its weight on the qualifying-position copy, which saturates for front-row starters: pole and P2 sit at quali_position 1 and 2, a tiny part of that feature's range, so the model can only express 'these two are the favourites' coarsely and the 46%-of-24-races pole-sitter win rate is well above the ~10% it issues. Grids 3-5, where quali_position still has room to vary, are calibrated — consistent with the gap being a feature-resolution problem at the top of the grid rather than a probability-scaling problem everywhere.

## Verdict

- **win**: does NOT ship (fails hold-out(s) 2022, 2024).
- **podium**: does NOT ship (fails hold-out(s) 2023, 2024).
- **points**: does NOT ship (fails hold-out(s) 2022, 2023, 2024).

