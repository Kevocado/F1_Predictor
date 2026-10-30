"""Per-tier metrics for the race outcome model. MEASUREMENT ONLY.

The serving path exposes three tiers (features/session_state.py): pre_weekend,
post_practice, post_qualifying. The production ranker is trained only on
post_qualifying rows (models/manifest.py::train_all), and a tier only changes
which tier-gated columns (grid, quali_position, quali_gap_to_pole) are masked
to NaN at prediction time.

This script follows the walk-forward ship test (scripts/ship_test_temperature.py):
same expanding-window hold-outs, same post_qualifying training regime. For each
hold-out it scores the SAME races under each tier's mask (session_state.tier_augment)
using the one ranker trained on post_qualifying rows. No served model, feature
or probability is changed; this is research code, not imported by serving.

Run from the repo root:
    PYTHONPATH=src:scripts python scripts/tier_metrics.py
Writes docs/f1_tier_metrics.json and docs/f1_tier_metrics.md.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from f1_predictor.features import session_state
from f1_predictor.features.build import build_training_frame
from f1_predictor.models import race_outcome

N_TRIALS = 3000
SIM_SEED = 42
N_BOOT = 2000
BOOT_SEED = 1
EPS = 1e-12

# Same expanding-window hold-outs as scripts/ship_test_temperature.py.
HOLDOUTS = [
    (2022, [2021]),
    (2023, [2021, 2022]),
    (2024, [2021, 2022, 2023]),
]
TIERS = session_state.TIER_ORDER
DOCS = Path(__file__).resolve().parents[1] / "docs"


# ---------------------------------------------------------------------------
# Metric functions. Each takes ONE race as a DataFrame (one row per driver).
# ---------------------------------------------------------------------------


def win_logloss_multiclass(race: pd.DataFrame, prob_col: str) -> float:
    """-log P(actual winner). `prob_col` must sum to 1 across the race."""
    winner = race["position"].to_numpy() == 1
    p = float(race.loc[winner, prob_col].iloc[0])
    return -float(np.log(max(p, EPS)))


def win_logloss_binary(race: pd.DataFrame, prob_col: str) -> float:
    """Mean per-driver binary log-loss of P(this driver wins). Each driver row is
    scored as its own yes/no question, NOT as the multiclass winner distribution."""
    p = np.clip(race[prob_col].to_numpy(dtype=float), EPS, 1 - EPS)
    y = (race["position"].to_numpy() == 1).astype(float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def spearman_order(race: pd.DataFrame, score_col: str) -> float:
    """Spearman rank correlation between a predicted score (higher = expected
    to finish higher) and the actual finishing order. Classified finishers only."""
    done = race["position"].notna()
    rho = spearmanr(race.loc[done, score_col].to_numpy(dtype=float),
                    -race.loc[done, "position"].to_numpy(dtype=float)).correlation
    return float(rho)


def brier_topk(race: pd.DataFrame, prob_col: str, k: int) -> float:
    """Mean per-driver Brier score for 'finishes in the top k'."""
    y = (race["position"] <= k).astype(float).to_numpy()
    return float(np.mean((race[prob_col].to_numpy(dtype=float) - y) ** 2))


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------


def grid_rates(train_df: pd.DataFrame) -> dict[str, pd.Series]:
    """P(event | grid slot) from TRAINING seasons only, add-one smoothed.
    Keys: win, top3, top6, top10. Indexed by grid position."""
    labelled = train_df[train_df["grid"].notna() & train_df["position"].notna()]
    out = {}
    specs = {"win": 1, "top3": 3, "top6": 6, "top10": 10}
    for name, k in specs.items():
        hit = (labelled["position"] <= k).astype(float)
        g = pd.DataFrame({"grid": labelled["grid"].astype(int), "hit": hit})
        agg = g.groupby("grid")["hit"].agg(["sum", "count"])
        out[name] = (agg["sum"] + 1) / (agg["count"] + 2)
    return out


def _grid_lookup(grid: np.ndarray, rates: pd.Series, fallback: float) -> np.ndarray:
    return np.array([rates.get(int(x), fallback) if not np.isnan(x) else fallback for x in grid])


def add_baselines(race: pd.DataFrame, rates: dict[str, pd.Series]) -> pd.DataFrame:
    """`grid_true` is the OFFICIAL grid, read from the unmasked frame. The model's
    own `grid` column is masked per tier and must not feed the baseline."""
    race = race.copy()
    n = len(race)
    race["u_win"] = 1.0 / n
    for name, k in {"top3": 3, "top6": 6, "top10": 10}.items():
        race[f"u_{name}"] = min(k, n) / n
    grid = race["grid_true"].to_numpy(dtype=float)
    for name in ["win", "top3", "top6", "top10"]:
        fallback = float(rates[name].mean())
        race[f"g_{name}"] = _grid_lookup(grid, rates[name], fallback)
    race["g_win_multi"] = race["g_win"] / race["g_win"].sum()
    race["neg_grid"] = -race["grid_true"].astype(float)
    return race


# ---------------------------------------------------------------------------
# Scoring one tier on the hold-out races
# ---------------------------------------------------------------------------


def score_race(race: pd.DataFrame, ranker, feature_cols: list[str]) -> pd.DataFrame:
    scores = race_outcome.xgb_scores_for_race(ranker, race, feature_cols)
    theta = race_outcome.theta_from_xgb_scores(scores)
    sim10 = race_outcome.simulate_race(theta, None, n_trials=N_TRIALS, seed=SIM_SEED,
                                       points_finish_cutoff=10).set_index("driver_id")
    sim6 = race_outcome.simulate_race(theta, None, n_trials=N_TRIALS, seed=SIM_SEED,
                                      points_finish_cutoff=6).set_index("driver_id")
    out = race[["driver_id", "position", "grid_true"]].reset_index(drop=True).copy()
    out["score"] = out["driver_id"].map(scores).astype(float)
    out["p_win"] = out["driver_id"].map(sim10["p_win"]).astype(float)
    out["p_top3"] = out["driver_id"].map(sim10["p_podium"]).astype(float)
    out["p_top6"] = out["driver_id"].map(sim6["p_points_finish"]).astype(float)
    out["p_top10"] = out["driver_id"].map(sim10["p_points_finish"]).astype(float)
    return out


def race_metric_values(f: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-race value of every metric, for the model and each naive baseline."""
    return {
        "win_logloss_multiclass": {
            "model": win_logloss_multiclass(f, "p_win"),
            "uniform": win_logloss_multiclass(f, "u_win"),
            "grid": win_logloss_multiclass(f, "g_win_multi"),
        },
        "win_logloss_binary_per_driver": {
            "model": win_logloss_binary(f, "p_win"),
            "uniform": win_logloss_binary(f, "u_win"),
            "grid": win_logloss_binary(f, "g_win"),
        },
        "spearman_rank_corr": {
            "model": spearman_order(f, "score"),
            # a uniformly random order has expected Spearman 0; no simulation needed
            "uniform": 0.0,
            "grid": spearman_order(f, "neg_grid"),
        },
        "brier_podium": {
            "model": brier_topk(f, "p_top3", 3),
            "uniform": brier_topk(f, "u_top3", 3),
            "grid": brier_topk(f, "g_top3", 3),
        },
        "brier_top6": {
            "model": brier_topk(f, "p_top6", 6),
            "uniform": brier_topk(f, "u_top6", 6),
            "grid": brier_topk(f, "g_top6", 6),
        },
        "brier_top10": {
            "model": brier_topk(f, "p_top10", 10),
            "uniform": brier_topk(f, "u_top10", 10),
            "grid": brier_topk(f, "g_top10", 10),
        },
    }


def mean_ci(values: np.ndarray) -> dict:
    """Mean over races with a race-level percentile bootstrap 95% CI."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    boots = values[idx].mean(axis=1)
    return {
        "mean": round(float(values.mean()), 4),
        "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
    }


def run() -> dict:
    seasons = sorted({s for _, tr in HOLDOUTS for s in tr} | {v for v, _ in HOLDOUTS})
    df, feature_cols = build_training_frame(seasons=seasons)
    post = df[df["tier"] == session_state.TIER_POST_QUALIFYING].reset_index(drop=True)

    # tier -> metric -> variant -> list of per-race values, pooled across hold-outs
    pooled: dict = {t: {} for t in TIERS}
    per_holdout: dict = {}
    holdout_info = []
    tier_row_sets: dict = {t: [] for t in TIERS}

    for val_season, train_seasons in HOLDOUTS:
        train_df = post[post["season"].isin(train_seasons)]
        val_df = post[post["season"] == val_season].reset_index(drop=True)
        # production regime: ranker trained on post_qualifying rows only
        ranker = race_outcome.train_ranker(train_df, feature_cols)
        rates = grid_rates(train_df)
        # the repo's own tier masking, applied to the hold-out season
        augmented = session_state.tier_augment(val_df)
        holdout_info.append({"val_season": val_season, "train_seasons": train_seasons,
                             "n_races": int(val_df[["season", "round"]].drop_duplicates().shape[0]),
                             "n_driver_rows": int(len(val_df))})
        per_holdout[str(val_season)] = {}

        for tier in TIERS:
            tier_df = augmented[augmented["tier"] == tier].merge(
                val_df[["season", "round", "driver_id", "grid"]].rename(columns={"grid": "grid_true"}),
                on=["season", "round", "driver_id"], how="left",
            )
            race_values: list[dict] = []
            for (season, rnd), race in tier_df.groupby(["season", "round"], sort=False):
                race = race.sort_values("driver_id").reset_index(drop=True)
                if not (race["position"] == 1).any():
                    raise ValueError(f"no winner recorded for {season} round {rnd}")
                f = add_baselines(score_race(race, ranker, feature_cols), rates)
                race_values.append(race_metric_values(f))
                tier_row_sets[tier].append((val_season, rnd, f["p_win"].round(12).tolist()))
            per_holdout[str(val_season)][tier] = _summarise(race_values)
            for metric, variants in _collect(race_values).items():
                for variant, vals in variants.items():
                    pooled[tier].setdefault(metric, {}).setdefault(variant, []).extend(vals)

    # the post_practice tier unlocks no columns in this repo, so its inputs equal pre_weekend
    identical = tier_row_sets[session_state.TIER_POST_PRACTICE] == tier_row_sets[session_state.TIER_PRE_WEEKEND]

    pooled_summary = {}
    for tier in TIERS:
        pooled_summary[tier] = {}
        for metric, variants in pooled[tier].items():
            pooled_summary[tier][metric] = {v: mean_ci(np.array(vals)) for v, vals in variants.items()}

    n_races = sum(h["n_races"] for h in holdout_info)
    return {
        "meta": {
            "generated": date.today().isoformat(),
            "model": "xgb_ranker (rank:pairwise), trained on post_qualifying rows only",
            "holdouts": holdout_info,
            "n_races_pooled": n_races,
            "n_trials": N_TRIALS,
            "sim_seed": SIM_SEED,
            "bootstrap": f"race-level percentile, {N_BOOT} resamples, seed {BOOT_SEED}",
            "post_practice_identical_to_pre_weekend": bool(identical),
            "grid_baseline_note": ("grid baseline uses the official grid, which exists only from "
                                   "qualifying onwards; it is a reference row shared by all tiers, "
                                   "not something pre_weekend could use"),
        },
        "pooled": pooled_summary,
        "per_holdout": per_holdout,
    }


def _collect(race_values: list[dict]) -> dict:
    out: dict = {}
    for rv in race_values:
        for metric, variants in rv.items():
            for variant, val in variants.items():
                out.setdefault(metric, {}).setdefault(variant, []).append(val)
    return out


def _summarise(race_values: list[dict]) -> dict:
    return {
        metric: {v: mean_ci(np.array(vals)) for v, vals in variants.items()}
        for metric, variants in _collect(race_values).items()
    }


def write_outputs(result: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "f1_tier_metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    names = {
        "win_logloss_multiclass": "Win log-loss (multiclass, lower better)",
        "win_logloss_binary_per_driver": "Win log-loss (per-driver binary, lower better)",
        "spearman_rank_corr": "Spearman rank corr (higher better)",
        "brier_podium": "Brier podium top-3 (lower better)",
        "brier_top6": "Brier top-6 (lower better)",
        "brier_top10": "Brier top-10 (lower better)",
    }
    lines = [
        "# F1 per-tier metrics (measurement only)",
        "",
        f"Hold-outs: " + ", ".join(f"{h['val_season']} (train {h['train_seasons'][0]}-{h['train_seasons'][-1]})"
                                 for h in result["meta"]["holdouts"])
        + f". Pooled n = {result['meta']['n_races_pooled']} races. Model trained on post_qualifying rows only.",
        "",
        "Model value [95% race-level bootstrap CI], with naive baselines. Uniform = equal share; "
        "grid = P(event given grid slot) from training seasons (grid baseline is a reference, see notes).",
        "",
        "| Metric | Tier | Model | Uniform | Grid |",
        "|---|---|---|---|---|",
    ]
    for metric, label in names.items():
        for tier in TIERS:
            m = result["pooled"][tier][metric]
            def fmt(v):
                return f"{v['mean']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"
            uni = m["uniform"]["mean"]
            lines.append(f"| {label} | {tier} | {fmt(m['model'])} | {uni:.4f} | {fmt(m['grid'])} |")
    lines += [
        "",
        "## Notes",
        "",
        f"- Races pooled: {result['meta']['n_races_pooled']} (2022: {result['meta']['holdouts'][0]['n_races']}, "
        f"2023: {result['meta']['holdouts'][1]['n_races']}, 2024: {result['meta']['holdouts'][2]['n_races']}).",
        "- pre_weekend: grid, quali_position, quali_gap_to_pole masked to NaN (no qualifying yet).",
        "- post_practice: NOT evaluable as a separate input. TIER_COLUMNS[post_practice] is empty and the "
        "fp_* practice features (best lap rank, long-run pace, tyre deg) are not built "
        "(data/fastf1_client.py is Phase 3), so its inputs are byte-identical to pre_weekend. "
        f"Measured identical to pre_weekend: {result['meta']['post_practice_identical_to_pre_weekend']}.",
        "- post_qualifying: all features present.",
        "- Spearman uniform baseline is the expected value (0) of a random order, not a simulation.",
        "- Multiclass win log-loss: one winner per race. Per-driver binary: each driver's P(win) scored as yes/no.",
        "- Brier top-6 and top-10 come from simulate_race with points_finish_cutoff 6 and 10 (same seed).",
        "",
    ]
    (DOCS / "f1_tier_metrics.md").write_text("\n".join(lines))


if __name__ == "__main__":
    write_outputs(run())
