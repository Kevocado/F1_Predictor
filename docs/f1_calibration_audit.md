# F1 calibration audit (walk-forward)

Generated 2026-10-22. XGB ranker, held-out seasons 2021-2024, 479 held-out race wins. Paired-bootstrap 95% CIs over 2000 resamples. Research only: no model output changed, nothing under src/.

Ideals: reliability slope 1.00; bias 0; log-loss comparable to baseline.

## Results

- **Overall reliability slope:** 3.060 (ideal 1.00), 95% CI [2.009, 4.019]. PROBLEM — slope CI excludes 1.0 (under-confidence)
- **Overall log-loss:** 0.1580. Any fix (re-calibration) must not increase log-loss via paired-bootstrap test (interval excluding zero deviation).
- **Per-grid calibration:** Grid 1 drivers were significantly under-confident (pred ~13% actual ~46%, consistent across seasons), indicating the model issues too-conservative win probabilities for pole sitters. Grid 2 drivers similarly under-confident. Grids 3+ had zero wins in this hold-out sample, so calibration at those positions cannot be assessed from this data alone due to small sample size. Any fix (e.g. isotonic recalibration per grid bucket) must not increase log-loss via paired-bootstrap test excluding zero deviation.
- No fix is shipped here: this is research only, and no serving change is proposed without a paired-bootstrap CI including 1.0 on re-tested candidates.

## Reviewer live-data leads (to test against full walk-forward)
*Small-n spot-checks; test at full sample size with paired-bootstrap. Propose fix only if CI excludes zero.*

- **points_finish calibration**: stated 50-60%/60-70%/70%+ buckets vs actual ~75/80/83% → possible UNDER-confidence across all buckets, recalibration candidate.
- **Test**: reuse walk-forward output (already has p_points_finish from simulate_race). Bucket predicted points_finish probs into 50-60/60-70/70%+, compute actual top-10 finish rates per bucket, paired-bootstrap CI on gap. If all buckets' CIs exclude zero → fix warranted (isotonic recalibration of p_points_finish). Note: overall win reliability slope already 3.06 (under-confident), consistent with this lead.
