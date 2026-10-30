# F1 per-tier metrics (measurement only)

Hold-outs: 2022 (train 2021-2021), 2023 (train 2021-2022), 2024 (train 2021-2023). Pooled n = 68 races. Model trained on post_qualifying rows only.

Model value [95% race-level bootstrap CI], with naive baselines. Uniform = equal share; grid = P(event given grid slot) from training seasons (grid baseline is a reference, see notes).

| Metric | Tier | Model | Uniform | Grid |
|---|---|---|---|---|
| Win log-loss (multiclass, lower better) | pre_weekend | 2.6985 [2.6199, 2.7735] | 2.9950 | 1.8114 [1.5895, 2.0514] |
| Win log-loss (multiclass, lower better) | post_practice | 2.6985 [2.6199, 2.7735] | 2.9950 | 1.8114 [1.5895, 2.0514] |
| Win log-loss (multiclass, lower better) | post_qualifying | 2.3544 [2.2335, 2.4657] | 2.9950 | 1.8114 [1.5895, 2.0514] |
| Win log-loss (per-driver binary, lower better) | pre_weekend | 0.1829 [0.1786, 0.1869] | 0.1986 | 0.1365 [0.1223, 0.1520] |
| Win log-loss (per-driver binary, lower better) | post_practice | 0.1829 [0.1786, 0.1869] | 0.1986 | 0.1365 [0.1223, 0.1520] |
| Win log-loss (per-driver binary, lower better) | post_qualifying | 0.1640 [0.1572, 0.1703] | 0.1986 | 0.1365 [0.1223, 0.1520] |
| Spearman rank corr (higher better) | pre_weekend | 0.5500 [0.5025, 0.5965] | 0.0000 | 0.5921 [0.5407, 0.6443] |
| Spearman rank corr (higher better) | post_practice | 0.5500 [0.5025, 0.5965] | 0.0000 | 0.5921 [0.5407, 0.6443] |
| Spearman rank corr (higher better) | post_qualifying | 0.6048 [0.5487, 0.6624] | 0.0000 | 0.5921 [0.5407, 0.6443] |
| Brier podium top-3 (lower better) | pre_weekend | 0.1148 [0.1120, 0.1172] | 0.1276 | 0.0748 [0.0673, 0.0825] |
| Brier podium top-3 (lower better) | post_practice | 0.1148 [0.1120, 0.1172] | 0.1276 | 0.0748 [0.0673, 0.0825] |
| Brier podium top-3 (lower better) | post_qualifying | 0.1049 [0.1008, 0.1087] | 0.1276 | 0.0748 [0.0673, 0.0825] |
| Brier top-6 (lower better) | pre_weekend | 0.1712 [0.1674, 0.1750] | 0.2101 | 0.1205 [0.1098, 0.1317] |
| Brier top-6 (lower better) | post_practice | 0.1712 [0.1674, 0.1750] | 0.2101 | 0.1205 [0.1098, 0.1317] |
| Brier top-6 (lower better) | post_qualifying | 0.1508 [0.1452, 0.1565] | 0.2101 | 0.1205 [0.1098, 0.1317] |
| Brier top-10 (lower better) | pre_weekend | 0.2017 [0.1956, 0.2074] | 0.2500 | 0.1691 [0.1561, 0.1821] |
| Brier top-10 (lower better) | post_practice | 0.2017 [0.1956, 0.2074] | 0.2500 | 0.1691 [0.1561, 0.1821] |
| Brier top-10 (lower better) | post_qualifying | 0.1827 [0.1739, 0.1913] | 0.2500 | 0.1691 [0.1561, 0.1821] |

## Notes

- Races pooled: 68 (2022: 22, 2023: 22, 2024: 24).
- pre_weekend: grid, quali_position, quali_gap_to_pole masked to NaN (no qualifying yet).
- post_practice: NOT evaluable as a separate input. TIER_COLUMNS[post_practice] is empty and the fp_* practice features (best lap rank, long-run pace, tyre deg) are not built (data/fastf1_client.py is Phase 3), so its inputs are byte-identical to pre_weekend. Measured identical to pre_weekend: True.
- post_qualifying: all features present.
- Spearman uniform baseline is the expected value (0) of a random order, not a simulation.
- Multiclass win log-loss: one winner per race. Per-driver binary: each driver's P(win) scored as yes/no.
- Brier top-6 and top-10 come from simulate_race with points_finish_cutoff 6 and 10 (same seed).
