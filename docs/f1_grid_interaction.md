# F1-2 grid-interaction features (research, flag default OFF)

Hold-outs (expanding window, post_qualifying): 2022 (22 races), 2023 (22 races), 2024 (24 races). Pooled n = 68.

Incumbent = PR #84 ranker. Candidate = same ranker plus grid-interaction features. Delta = candidate - incumbent, paired race-level bootstrap 95% CI.

| Metric | Incumbent | Candidate | Delta [95% CI] | Rule |
|---|---|---|---|---|
| win_logloss_multiclass | 2.3544 | 2.1936 | -0.1608 [-0.2120, -0.1087] | CI entirely < 0 |
| win_logloss_binary_per_driver | 0.1640 | 0.1550 | -0.0090 [-0.0119, -0.0061] | report only |
| spearman_rank_corr | 0.6048 | 0.6205 | +0.0157 [-0.0031, +0.0355] | delta >= 0 |
| brier_podium | 0.1049 | 0.0979 | -0.0070 [-0.0086, -0.0054] | delta <= 0 |
| brier_top6 | 0.1508 | 0.1440 | -0.0069 [-0.0089, -0.0048] | delta <= 0 |
| brier_top10 | 0.1827 | 0.1809 | -0.0018 [-0.0039, +0.0003] | delta <= 0 |

## Front-row calibration (grid 1-2, pooled)

| Grid | Incumbent mean p_win | Candidate mean p_win | Actual win rate | n |
|---|---|---|---|---|
| 1 | 0.1174 | 0.1472 | 0.5147 | 68 |
| 2 | 0.0818 | 0.0938 | 0.2206 | 68 |

## Verdict

- win_logloss_multiclass_improves_ci_excludes_zero: PASS
- brier_podium_not_worse: PASS
- brier_top6_not_worse: PASS
- brier_top10_not_worse: PASS
- spearman_not_worse: PASS
- PASS: PASS

Default flag: grid_interaction.GRID_INTERACTION_ENABLED = False (not wired into serving in either case).
