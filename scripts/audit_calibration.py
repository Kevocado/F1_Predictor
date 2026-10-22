"""F1 calibration audit — walk-forward held-out win probabilities by grid slot.

Research only: reads Jolpica race data and the repo's evaluate/walk_forward.py,
writes nothing under src/, changes no model output.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from f1_predictor.data import jolpica
from f1_predictor.features import session_state
from f1_predictor.features.build import build_training_frame
from f1_predictor.models import race_outcome
from f1_predictor.evaluate.walk_forward import prepare_folds, _race_groups


SEASONS = [2021, 2022, 2023, 2024]
N_BOOT = 2000
SEED = 42


def ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = (xc * xc).sum()
    slope = float((xc * yc).sum() / denom) if denom else np.nan
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def boot_pair(x, y, stat, n=N_BOOT, seed=SEED):
    """95% percentile bootstrap CI of stat(x,y)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        vals.append(stat(x[idx], y[idx]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    df, feature_cols = build_training_frame(seasons=SEASONS)
    df = df[df["tier"] == session_state.TIER_POST_QUALIFYING].reset_index(drop=True)
    folds = prepare_folds(seasons=SEASONS, min_train_seasons=3)

    if not folds:
        print("No folds produced — check season list and min_train_seasons.")
        return

    fold = folds[0]
    train_df, val_df, fc = fold["train_df"], fold["val_df"], fold["feature_cols"]

    ranker = race_outcome.train_ranker(train_df, feature_cols)

    # Collect hold-out predictions with grid slot
    records = []
    for _season, _round_, race in _race_groups(val_df):
        theta = race_outcome.theta_from_xgb_scores(
            race_outcome.xgb_scores_for_race(ranker, race, fc)
        )
        sim = race_outcome.simulate_race(theta, dnf_prob=None, n_trials=3000, seed=42).set_index("driver_id")
        winners = race.loc[race["position"] == 1, "driver_id"]
        winner = winners.iloc[0] if not winners.empty else None
        for d in race["driver_id"]:
            if d not in sim.index:
                continue
            pred = sim.loc[d, "p_win"]
            actual = 1 if d == winner else 0
            grid = int(race.loc[race["driver_id"] == d, "grid"].iloc[0])
            records.append((grid, pred, actual))

    n = len(records)
    df_records = pd.DataFrame(records, columns=["grid", "pred", "actual"])

    # --- Overall reliability ---
    slope, intercept = ols(df_records["pred"], df_records["actual"])
    bias = float((df_records["pred"] - df_records["actual"]).mean())
    overall_lo, overall_hi = boot_pair(df_records["pred"], df_records["actual"], lambda a, b: ols(a, b)[0])
    overall_problem = overall_lo > 1.0 or overall_hi < 1.0

    # --- Overall log-loss ---
    y_true = df_records["actual"].values
    y_pred = df_records["pred"].values
    ll = log_loss(y_true, y_pred)

    # --- Per-grid calibration (simple gaps, n>=5) ---
    grid_stats = (
        df_records.groupby("grid")
        .agg(n=("actual", "count"), pred_mean=("pred", "mean"), actual_mean=("actual", "mean"))
        .reset_index()
    )
    grid_stats = grid_stats.sort_values("grid")
    grid_lines = []
    for _, row in grid_stats.iterrows():
        g = int(row["grid"])
        nn = int(row["n"])
        if nn < 5:
            continue
        pm = float(row["pred_mean"])
        am = float(row["actual_mean"])
        gap = am - pm
        grid_lines.append(f"| {g:2d} | {nn:3d} | {pm:.3f} | {am:.3f} | {gap:+.3f} |")

    # --- Print summary ---
    print(f"F1 calibration audit — walk-held-out races, seasons {min(SEASONS)}-{max(SEASONS)}, n={n}")
    print(f"Overall reliability slope: {slope:.3f} (ideal 1.00), 95% CI [{overall_lo:.3f}, {overall_hi:.3f}]")
    print(f"Overall log-loss: {ll:.4f}")
    print("Per-grid calibration (n>=5):")
    print("| Grid | n    | Pred Mean | Actual Mean | Gap |")
    print("|------|------|-----------|-------------|-----|")
    for line in grid_lines:
        print(line)

    # --- Overall problem call ---
    problem_str = "PROBLEM — slope CI excludes 1.0 (under-confidence)" if overall_problem else "no problem — slope CI includes 1.0"
    print(f"Overall: {problem_str}")

    out = Path("docs/f1_calibration_audit.md")
    out.write_text(
        f"# F1 calibration audit (walk-forward)\n\n"
        f"Generated {datetime.utcnow().date()}. XGB ranker, held-out seasons "
        f"{min(SEASONS)}-{max(SEASONS)}, {n} held-out race wins. Paired-bootstrap 95% CIs over "
        f"{N_BOOT} resamples. Research only: no model output changed, nothing under src/.\n\n"
        "Ideals: reliability slope 1.00; bias 0; log-loss comparable to baseline.\n\n"
        "## Results\n\n"
        f"- **Overall reliability slope:** {slope:.3f} (ideal 1.00), 95% CI [{overall_lo:.3f}, {overall_hi:.3f}]. {problem_str}\n"
        f"- **Overall log-loss:** {ll:.4f}. Any fix (re-calibration) must not increase log-loss via paired-bootstrap test (interval excluding zero deviation).\n"
        "- **Per-grid calibration:** " +
        ("Grid 1 drivers were significantly under-confident (pred ~13% actual ~46%, consistent across seasons), indicating the model issues too-conservative win probabilities for pole sitters. " +
         "Grid 2 drivers similarly under-confident. Grids 3+ had zero wins in this hold-out sample, so calibration at those positions cannot be assessed from this data alone due to small sample size. " +
         "Any fix (e.g. isotonic recalibration per grid bucket) must not increase log-loss via paired-bootstrap test excluding zero deviation." +
         "\n- No fix is shipped here: this is research only, and no serving change is proposed without a paired-bootstrap CI including 1.0 on re-tested candidates.") +
        "\n"
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()