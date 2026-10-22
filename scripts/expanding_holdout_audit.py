"""F1 expanding-window hold-out calibration audit.

Hold-out A: train 2021 -> test 2022
Hold-out B: train 2021-2022 -> test 2023
Hold-out C: train 2021-2023 -> test 2024

Race-level paired bootstrap (2000 resamples) for slope CI and log-loss.
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
SEED = 1


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    denom = (xc * xc).sum()
    slope = float((xc * yc).sum() / denom) if denom else np.nan
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def boot_race(races, stat, n=2000, seed=1):
    """Race-level bootstrap: resample entire races, compute stat on pooled drivers."""
    rng = np.random.default_rng(seed)
    n_races = len(races)
    vals = []
    for _ in range(n):
        idx = np.random.default_rng(seed).integers(0, len(races), len(races))
        pooled = pd.concat([races[i] for i in idx], ignore_index=True)
        vals.append(stat(pooled))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate_holdout(train_seasons, val_season):
    """Run single hold-out: train on train_seasons, evaluate on val_season."""
    print(f"\n{'='*60}")
    print(f"HOLD-OUT: train={train_seasons} -> val={val_season}")
    print(f"{'='*60}")
    
    # Build training data for all seasons up to val_season
    all_seasons = train_seasons + [val_season]
    df, feature_cols = build_training_frame(seasons=train_seasons + [val_season])
    df = df[df["tier"] == session_state.TIER_POST_QUALIFYING].reset_index(drop=True)
    
    # Split train/val
    train_df = df[df["season"].isin(train_seasons)]
    val_df = df[df["season"] == val_season]
    
    print(f"Train: {len(train_df)} rows from seasons {train_seasons}")
    print(f"Val: {len(val_df)} rows from season {val_season}")
    print(f"Feature cols: {len(feature_cols)}")
    
    # Train ranker
    ranker = race_outcome.train_ranker(train_df, feature_cols)
    
    # Collect predictions per race
    races = []
    for _season, _round_, race in _race_groups(val_df):
        theta = race_outcome.theta_from_xgb_scores(
            race_outcome.xgb_scores_for_race(ranker, race, feature_cols)
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
    
    all_df = pd.concat(races, ignore_index=True)
    n_races = len(races)
    n_drivers = len(all_df)
    
    print(f"Val races: {n_races}, driver-race obs: {len(all_df)}")
    
    # Overall reliability
    y_win, p_win = all_df["actual_win"].values, all_df["p_win"].values
    y_pod, p_pod = all_df["actual_pod"].values, all_df["p_podium"].values
    y_pts, p_pts = all_df["actual_pts"].values, all_df["p_pts"].values
    
    slope_win, _ = ols(all_df["p_win"], all_df["actual_win"])
    slope_pod, _ = ols(all_df["p_podium"], all_df["actual_pod"])
    slope_pts, _ = ols(all_df["p_pts"], all_df["actual_pts"])
    
    sl_lo, sl_hi = boot_race(races, lambda df: ols(df["p_win"], df["actual_win"])[0])
    sl_lo_p, sl_hi_p = boot_race(races, lambda df: ols(df["p_podium"], df["actual_pod"])[0])
    sl_lo_pt, sl_hi_pt = boot_race(races, lambda df: ols(df["p_pts"], df["actual_pts"])[0])
    
    ll_win = log_loss(all_df["actual_win"], all_df["p_win"])
    ll_pod = log_loss(all_df["actual_pod"], all_df["p_podium"])
    ll_pts = log_loss(all_df["actual_pts"], all_df["p_pts"])
    
    print(f"\nResults for {val_season}:")
    print(f"  Win: slope={slope_win:.3f} CI=[{sl_lo:.3f},{sl_hi:.3f}] logloss={log_loss(all_df['actual_win'], all_df['p_win']):.4f}")
    print(f"  Pod: slope={slope_pod:.3f} CI=[{sl_lo_p:.3f},{sl_hi_p:.3f}] logloss={log_loss(all_df['actual_pod'], all_df['p_podium']):.4f}")
    print(f"  Pts: slope={slope_pts:.3f} CI=[{sl_lo_pt:.3f},{sl_hi_pt:.3f}] logloss={log_loss(all_df['actual_pts'], all_df['p_pts']):.4f}")
    print(f"  Under-confidence: win={sl_lo>1.0 or sl_hi<1.0}, pod={sl_lo_p>1.0 or sl_hi_p<1.0}, pts={sl_lo_pt>1.0 or sl_hi_pt<1.0}")
    
    # Grid 1-2 analysis
    grid12 = all_df[all_df["grid"].isin([1, 2])]
    grid35 = all_df[all_df["grid"].isin([3, 4, 5])]
    if len(grid12) > 0:
        print(f"\nGrid 1-2 (front row): n={len(grid12)}, pred_win={grid12['p_win'].mean():.3f}, actual={grid12['actual_win'].mean():.3f}")
    if len(grid35) > 0:
        print(f"Grid 3-5: n={len(grid35)}, pred_win={grid35['p_win'].mean():.3f}, actual={grid35['actual_win'].mean():.3f}")
    
    return {
        "val_season": val_season,
        "n_races": len(races),
        "n_drivers": len(all_df),
        "win_slope": slope_win, "win_ci": (sl_lo, sl_hi), "win_ll": log_loss(all_df["actual_win"], all_df["p_win"]),
        "pod_slope": slope_pod, "pod_ci": (sl_lo_p, sl_hi_p), "pod_ll": log_loss(all_df["actual_pod"], all_df["p_podium"]),
        "pts_slope": slope_pts, "pts_ci": (sl_lo_pt, sl_hi_pt), "pts_ll": log_loss(all_df["actual_pts"], all_df["p_pts"]),
        "underconf_win": sl_lo > 1.0 or sl_hi < 1.0,
        "underconf_pod": sl_lo_p > 1.0 or sl_hi_p < 1.0,
        "underconf_pts": sl_lo_pt > 1.0 or sl_hi_pt < 1.0,
    }


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xc = x - x.mean(); yc = y - y.mean()
    denom = (xc * xc).sum()
    slope = float((xc * yc).sum() / denom) if denom else np.nan
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def boot_race(races, stat, n=2000, seed=1):
    """Race-level bootstrap: resample entire races, compute stat on pooled drivers."""
    rng = np.random.default_rng(seed)
    n_races = len(races)
    vals = []
    for _ in range(n):
        idx = np.random.default_rng(seed).integers(0, len(races), len(races))
        pooled = pd.concat([races[i] for i in idx], ignore_index=True)
        vals.append(stat(pooled))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    # Run all three hold-outs
    results = []
    results.append(evaluate_holdout([2021], 2022))  # Hold-out A
    results.append(evaluate_holdout([2021, 2022], 2023))  # Hold-out B
    results.append(evaluate_holdout([2021, 2022, 2023], 2024))  # Hold-out C
    
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")
    print(f"{'Hold-out':<12} {'Market':<10} {'Slope':<8} {'CI':<20} {'LogLoss':<8} {'Underconf'}")
    print("-" * 80)
    for r in results:
        print(f"{r['val_season']:<12} {'Win':<10} {r['win_slope']:.3f}  [{r['win_ci'][0]:.3f},{r['win_ci'][1]:.3f}]  {r['win_ll']:.4f}  {r['underconf_win']}")
        print(f"{r['val_season']:<12} {'Podium':<10} {r['pod_slope']:.3f}  [{r['pod_ci'][0]:.3f},{r['pod_ci'][1]:.3f}]  {r['pod_ll']:.4f}  {r['underconf_pod']}")
        print(f"{r['val_season']:<12} {'PtsFin':<10} {r['pts_slope']:.3f}  [{r['pts_ci'][0]:.3f},{r['pts_ci'][1]:.3f}]  {r['pts_ll']:.4f}  {r['underconf_pts']}")
        print()
    
    # Check if ALL three hold-outs show under-confidence for ALL markets
    all_underconf = all(r["underconf_win"] and r["underconf_pod"] and r["underconf_pts"] for r in results)
    print(f"ALL THREE HOLD-OUTS SHOW UNDER-CONFIDENCE FOR ALL MARKETS: {all_underconf}")
    
    if not all_underconf:
        print("\nGATE FAILED: Not all hold-outs show under-confidence. STOP - docs only.")
    else:
        print("\nGATE PASSED: Proceed to recalibration implementation.")


if __name__ == "__main__":
    main()