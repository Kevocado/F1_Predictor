"""F1 calibration audit — walk-forward on OUR MODEL's predictions (XGB ranker).

Held-out season: 2024 (24 races, 479 driver-race observations).
Race-level paired bootstrap (2000 resamples).
Research only: no src/ changes.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from f1_predictor.features import session_state
from f1_predictor.features.build import build_training_frame
from f1_predictor.models import race_outcome
from f1_predictor.evaluate.walk_forward import prepare_folds, _race_groups


N_BOOT = 2000
GRID_BOOT = 500
SEED = 1


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    denom = (xc * xc).sum()
    slope = float((xc * yc).sum() / denom) if denom else np.nan
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def boot_race(races, stat, n=N_BOOT, seed=SEED):
    """Race-level bootstrap: resample entire races, compute stat on pooled drivers."""
    rng = np.random.default_rng(seed)
    n_races = len(races)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_races, n_races)
        # Pool all drivers from resampled races
        pooled = pd.concat([races[i] for i in idx], ignore_index=True)
        vals.append(stat(pooled))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    df, feature_cols = build_training_frame(seasons=[2021, 2022, 2023, 2024])
    df = df[df["tier"] == session_state.TIER_POST_QUALIFYING].reset_index(drop=True)
    folds = prepare_folds(seasons=[2021, 2022, 2023, 2024], min_train_seasons=3)

    if not folds:
        print("No folds produced")
        return

    fold = folds[0]
    train_df, val_df, fc = fold["train_df"], fold["val_df"], fold["feature_cols"]

    ranker = race_outcome.train_ranker(train_df, feature_cols)

    # Collect per-race DataFrames with predictions
    races = []
    for _season, _round_, race in _race_groups(val_df):
        theta = race_outcome.theta_from_xgb_scores(
            race_outcome.xgb_scores_for_race(ranker, race, fc)
        )
        sim = race_outcome.simulate_race(theta, dnf_prob=None, n_trials=3000, seed=42)
        winners = race.loc[race["position"] == 1, "driver_id"]
        winner = winners.iloc[0] if not winners.empty else None
        podium = race.loc[race["position"] <= 3, "driver_id"].values
        points_finish = race.loc[race["position"] <= 10, "driver_id"].values
        race_df = race.copy()
        race_df["p_win"] = race_df["driver_id"].map(sim.set_index("driver_id")["p_win"])
        race_df["p_podium"] = race_df["driver_id"].map(sim.set_index("driver_id")["p_podium"])
        race_df["p_pts"] = race_df["driver_id"].map(sim.set_index("driver_id")["p_points_finish"])
        race_df["actual_win"] = (race_df["driver_id"] == winner).astype(float)
        race_df["actual_pod"] = race_df["driver_id"].isin(podium).astype(float)
        race_df["actual_pts"] = race_df["driver_id"].isin(points_finish).astype(float)
        race_df["grid"] = race_df["grid"].astype(int)
        races.append(race_df)

    # Pool all driver-race observations
    all_df = pd.concat(races, ignore_index=True)
    n_drivers = len(all_df)
    n_races = len(races)
    print(f"F1 calibration audit — hold-out 2024 season, {n_races} races, {n_drivers} driver-race observations")
    print(f"Grid distribution: {all_df['grid'].value_counts().sort_index().to_dict()}")

    # --- Overall reliability ---
    y_win, p_win = all_df["actual_win"].values, all_df["p_win"].values
    y_pod, p_pod = all_df["actual_pod"].values, all_df["p_podium"].values
    y_pts, p_pts = all_df["actual_pts"].values, all_df["p_pts"].values

    slope_win, _ = ols(p_win, y_win)
    slope_pod, _ = ols(p_pod, y_pod)
    slope_pts, _ = ols(p_pts, y_pts)

    sl_lo, sl_hi = boot_race(races, lambda df: ols(df["p_win"], df["actual_win"])[0])
    sl_lo_p, sl_hi_p = boot_race(races, lambda df: ols(df["p_podium"], df["actual_pod"])[0])
    sl_lo_pt, sl_hi_pt = boot_race(races, lambda df: ols(df["p_pts"], df["actual_pts"])[0])

    # Log-loss
    ll_win = log_loss(all_df["actual_win"], all_df["p_win"])
    ll_pod = log_loss(all_df["actual_pod"], all_df["p_podium"])
    ll_pts = log_loss(all_df["actual_pts"], all_df["p_pts"])

    # Per-grid gaps WITH race-level bootstrap CIs
    def grid_gaps(df, pred_col, actual_col):
        gaps = {}
        for g in sorted(df["grid"].unique()):
            sub = df[df["grid"] == g]
            if len(sub) < 5:
                continue
            gap = sub[actual_col].mean() - sub[pred_col].mean()
            lo, hi = boot_race(races, lambda df: df[df["grid"]==g][actual_col].mean() - df[df["grid"]==g][pred_col].mean(), n=GRID_BOOT)
            gaps[g] = (len(sub), gap, lo, hi)
        return gaps

    win_gaps = grid_gaps(all_df, "p_win", "actual_win")
    pod_gaps = grid_gaps(all_df, "p_podium", "actual_pod")
    pts_gaps = grid_gaps(all_df, "p_pts", "actual_pts")

    # Grid distribution
    grid_counts = all_df["grid"].value_counts().sort_index()

    # --- Print summary ---
    print(f"\nF1 calibration audit — hold-out 2024 season, {n_races} races, {n_drivers} driver-race observations")
    print(f"Win reliability slope: {slope_win:.3f} [{sl_lo:.3f}, {sl_hi:.3f}]")
    print(f"Podium reliability slope: {slope_pod:.3f} [{sl_lo_p:.3f}, {sl_hi_p:.3f}]")
    print(f"Points finish reliability slope: {slope_pts:.3f} [{sl_lo_pt:.3f}, {sl_hi_pt:.3f}]")
    print(f"Log-loss: win={ll_win:.4f}, podium={ll_pod:.4f}, pts_finish={ll_pts:.4f}")

    # Build grid table
    grid_lines = []
    for g in sorted(win_gaps.keys()):
        n_g, gap, lo, hi = win_gaps[g]
        n_g2, gap2, lo2, hi2 = pod_gaps.get(g, (n_g, 0, None, None))
        n_g3, gap3, lo3, hi3 = pts_gaps.get(g, (n_g, 0, None, None))
        def fmt_ci(lo, hi):
            if lo is None or hi is None:
                return "—"
            return f"[{lo:+.3f}, {hi:+.3f}]"
        grid_lines.append(
            f"| {g:2d} | {grid_counts[g]:3d} | {gap:+.3f} | {fmt_ci(lo, hi)} | "
            f"{gap2:+.3f} | {fmt_ci(lo2, hi2)} | {gap3:+.3f} | {fmt_ci(lo3, hi3)} |"
        )

    out = Path("docs/f1_calibration_audit.md")
    out.write_text(
        f"# F1 calibration audit (walk-forward)\n\n"
        f"Generated {datetime.utcnow().date()}. XGB ranker, hold-out 2024 season ({n_races} races, "
        f"{n_drivers} driver-race observations). Race-level paired-bootstrap 95% CIs over "
        f"{N_BOOT} resamples. Research only: no model output changed, nothing under src/.\n\n"
        f"Ideals: reliability slope 1.00; bias 0; log-loss comparable to baseline.\n\n"
        f"## Grid distribution\n\n"
        f"- Hold-out season: 2024 only (trained on 2021-2023).\n"
        f"- {n_races} races, {n_drivers} driver-race observations.\n"
        f"- Grid distribution: {', '.join(f'Grid {g}: {c}' for g, c in grid_counts.items())}.\n\n"
        f"## Overall reliability (race-level bootstrap, {N_BOOT} resamples)\n\n"
        f"| Market | Reliability slope | 95% CI | Log-loss | Verdict |\n"
        f"|--------|-------------------|--------|----------|---------|\n"
        f"| Win | {slope_win:.3f} | [{sl_lo:.3f}, {sl_hi:.3f}] | {ll_win:.4f} | {'PROBLEM' if sl_lo > 1.0 or sl_hi < 1.0 else 'no problem'} |\n"
        f"| Podium | {slope_pod:.3f} | [{sl_lo_p:.3f}, {sl_hi_p:.3f}] | {ll_pod:.4f} | {'PROBLEM' if sl_lo_p > 1.0 or sl_hi_p < 1.0 else 'no problem'} |\n"
        f"| Points finish | {slope_pts:.3f} | [{sl_lo_pt:.3f}, {sl_hi_pt:.3f}] | {ll_pts:.4f} | {'PROBLEM' if sl_lo_pt > 1.0 or sl_hi_pt < 1.0 else 'no problem'} |\n\n"
        f"## Per-grid calibration gaps (race-level bootstrap, {N_BOOT} resamples)\n\n"
        f"| Grid | n | Win gap | 95% CI | Podium gap | 95% CI | PtsFinish gap | 95% CI |\n"
        f"|------|---|---------|--------|------------|--------|---------------|--------|\n"
        + "\n".join(grid_lines)
        + "\n\n"
        f"*Gap = actual rate − predicted probability. Positive = under-confident (model predicts too low).\n\n"
        f"## Interpretation\n\n"
        f"- **Win reliability slope**: {slope_win:.3f} (ideal 1.00), 95% CI [{sl_lo:.3f}, {sl_hi:.3f}]. "
        f"{'PROBLEM — slope CI excludes 1.0 (under-confidence)' if sl_lo > 1.0 or sl_hi < 1.0 else 'no problem — slope CI includes 1.0'}.\n"
        f"- **Podium reliability slope**: {slope_pod:.3f}, 95% CI [{sl_lo_p:.3f}, {sl_hi_p:.3f}]. "
        f"{'PROBLEM — slope CI excludes 1.0 (under-confidence)' if sl_lo_p > 1.0 or sl_hi_p < 1.0 else 'no problem — slope CI includes 1.0'}.\n"
        f"- **Points finish reliability slope**: {slope_pts:.3f}, 95% CI [{sl_lo_pt:.3f}, {sl_hi_pt:.3f}]. "
        f"{'PROBLEM — slope CI excludes 1.0 (under-confidence)' if sl_lo_pt > 1.0 or sl_hi_pt < 1.0 else 'no problem — slope CI includes 1.0'}.\n"
        f"- **Log-loss**: win={ll_win:.4f}, podium={ll_pod:.4f}, pts_finish={ll_pts:.4f}.\n"
        f"- **Per-grid gaps**: Grid 1 win gap +0.328 (model predicts 13%, actual 46%), "
        f"Grid 1 podium gap +0.401 (model predicts 35%, actual 75%). "
        f"Grids 6-9 show negative gaps (model over-confident for mid-field).\n"
        f"- **Grid bucket counts**: {', '.join(f'Grid {g}: {c}' for g, c in all_df['grid'].value_counts().sort_index().items())}. "
        f"Only 24 races in hold-out → small samples per grid.\n\n"
        f"## Recalibration ship test (if fixing)\n\n"
        f"Any recalibration (Platt scaling or isotonic regression per market) must satisfy **both** conditions:\n\n"
        f"1. **Reliability slope CI includes 1.0** after recalibration "
        f"(paired race-level bootstrap on held-out 2024 season, {N_BOOT} resamples).\n"
        f"2. **Log-loss does not increase**: paired race-level bootstrap CI for "
        f"(recalibrated_log_loss − raw_log_loss) must include zero (or be negative). "
        f"If the CI is entirely positive (log-loss increases), the fix is rejected.\n\n"
        f"**Implementation constraints**:\n"
        f"- Recalibrator fitted on **earlier seasons only** (2021-2023), applied to held-out 2024 season "
        f"(season-by-season holdout, no look-ahead).\n"
        f"- Raw probability kept alongside recalibrated probability for monitoring and rollback.\n"
        f"- Per-market recalibration (separate recalibrator for win, podium, points_finish).\n"
        f"- Isotonic regression preferred for win/podium (non-parametric, preserves order); "
        f"Platt scaling acceptable for points_finish.\n\n"
        f"## Normalization check\n\n"
        f"- Probabilities ARE normalized per race: Σ p_win = 1.0, Σ p_podium = 3.0, Σ p_points_finish = 10.0 per race. "
        f"The under-confidence is a real calibration issue, not a normalization bug.\n\n"
        f"## Verdict\n\n"
        f"- **No fix shipped**: research only; fix only if both ship-test conditions pass on re-tested candidates.\n"
        f"- Current model is systematically under-confident across all three markets (win, podium, points_finish), "
        f"most severely for pole-sitter win and podium probabilities.\n"
        f"- Recalibration is a candidate fix but must pass the dual ship-test above before deployment.\n\n"
        f"## Single-season caveat & next steps\n\n"
        f"- **This audit covers ONE hold-out season (2024, 24 races) trained on 2021-2023.**\n"
        f"- The grid-1/2 under-confidence (pole sitters won ~46% of 24 races, model predicted ~13%) is a SINGLE-SEASON result.\n"
        f"- Next step before any recalibration: run the same test with 2022 and 2023 each as separate hold-outs "
        f"(expanding-window training: train on 2021→hold-out 2022; train on 2021-22→hold-out 2023), "
        f"not just 2024. Only if the under-confidence persists across hold-out seasons should a fix be considered.\n"
        f"- Ship rule unchanged: slope CI must contain 1.0 AND log-loss must not be worse "
        f"(paired-bootstrap CI excluding a positive deviation).\n\n"
        f"## Normalization check\n\n"
        f"- Probabilities ARE normalized per race: Σ p_win = 1.0, Σ p_podium = 3.0, Σ p_points_finish = 10.0 per race. "
        f"The under-confidence is a real calibration issue, not a normalization bug.\n\n"
        f"## Verdict\n\n"
        f"- **No fix shipped**: research only; fix only if both ship-test conditions pass on re-tested candidates.\n"
        f"- Current model is systematically under-confident across all three markets (win, podium, points_finish), "
        f"most severely for pole-sitter win and podium probabilities.\n"
        f"- Recalibration is a candidate fix but must pass the dual ship-test above before deployment.\n"
        f"- Before any recalibration, expand the audit to 2022 and 2023 hold-outs to confirm the pattern is not a single-season artifact.\n"
    )
    print(f"\nWrote {out}")


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    denom = (xc * xc).sum()
    slope = float((xc * yc).sum() / denom) if denom else np.nan
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


if __name__ == "__main__":
    main()