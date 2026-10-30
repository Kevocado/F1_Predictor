"""F1-2: grid-interaction features vs the incumbent ranker. RESEARCH ONLY.

Same expanding-window hold-outs and post_qualifying regime as
scripts/tier_metrics.py (imported for the metric functions, the scoring path
and the grid baseline, so the incumbent reproduces PR #84 exactly). The
candidate adds grid_interaction.GRID_FEATURES to the ranker's feature list.
Every fitted quantity (grid curves, shrinkage k, grid rates, ranker) uses the
training seasons of each hold-out only; training rows use leave-one-race-out.

Pass rule (all must hold on the pooled 2022-2024 hold-outs):
  - win log-loss (multiclass): paired race-level bootstrap 95% CI of
    (candidate - incumbent) lies entirely below 0.
  - podium / top-6 / top-10 Brier and Spearman: mean paired delta not worse
    (Brier delta <= 0, Spearman delta >= 0). CIs are reported.

Run from the repo root:
    PYTHONPATH=src:scripts python scripts/grid_interaction.py
Writes docs/f1_grid_interaction.json and docs/f1_grid_interaction.md.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import tier_metrics as tm
from f1_predictor.data import jolpica
from f1_predictor.features import grid_interaction as gi
from f1_predictor.features import session_state
from f1_predictor.features.build import FEATURE_COLUMNS, build_training_frame
from f1_predictor.models import race_outcome

N_BOOT = 10000
BOOT_SEED = 1
SEASONS = [2021, 2022, 2023, 2024]
DOCS = Path(__file__).resolve().parents[1] / "docs"
POST = session_state.TIER_POST_QUALIFYING
METRICS = [
    "win_logloss_multiclass",
    "win_logloss_binary_per_driver",
    "spearman_rank_corr",
    "brier_podium",
    "brier_top6",
    "brier_top10",
]


def load_frame() -> tuple[pd.DataFrame, list[str]]:
    df, _ = build_training_frame(seasons=SEASONS)
    post = df[df["tier"] == POST].reset_index(drop=True)
    schedule = pd.concat([jolpica.fetch_season_schedule(s) for s in SEASONS], ignore_index=True)
    # build's frame already carries circuit_id_x/_y from two sources; use the schedule's
    post = post.drop(columns=[c for c in post.columns if c.startswith("circuit_id")])
    post = post.merge(schedule[["season", "round", "circuit_id"]], on=["season", "round"], how="left")
    quali = pd.concat([jolpica.load_season_qualifying(s) for s in SEASONS], ignore_index=True)
    post = post.merge(gi.quali_gap_features(quali), on=["season", "round", "driver_id"], how="left")
    return post, FEATURE_COLUMNS + gi.GRID_FEATURES


def score_holdout(post: pd.DataFrame, val_season: int, train_seasons: list[int],
                  feature_cols: list[str], add_grid: bool):
    train_df = post[post["season"].isin(train_seasons)].copy()
    val_df = post[post["season"] == val_season].copy()
    if add_grid:
        raw_train = train_df.copy()
        train_df[gi.GRID_FEATURES[:4]] = gi.grid_features(raw_train, raw_train, loo=True)
        val_df[gi.GRID_FEATURES[:4]] = gi.grid_features(raw_train, val_df, loo=False)
    ranker = race_outcome.train_ranker(train_df, feature_cols)
    rates = tm.grid_rates(train_df)
    records, front = [], []
    for (season, rnd), race in val_df.groupby(["season", "round"], sort=False):
        race = race.sort_values("driver_id").reset_index(drop=True)
        race["grid_true"] = race["grid"]
        f = tm.add_baselines(tm.score_race(race, ranker, feature_cols), rates)
        vals = tm.race_metric_values(f)
        records.append({"season": int(season), "round": int(rnd),
                        **{m: vals[m]["model"] for m in METRICS}})
        for g, p, y in zip(f["grid_true"], f["p_win"], f["position"]):
            if g in (1, 2):
                front.append((int(g), float(p), float(y == 1)))
    return records, front


def paired_ci(inc: np.ndarray, cand: np.ndarray) -> dict:
    d = cand - inc
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "incumbent_mean": round(float(inc.mean()), 4),
        "candidate_mean": round(float(cand.mean()), 4),
        "delta_mean": round(float(d.mean()), 4),
        "delta_ci95": [round(float(np.percentile(boots, 2.5)), 4),
                       round(float(np.percentile(boots, 97.5)), 4)],
    }


def run() -> dict:
    post, cand_cols = load_frame()
    inc_recs, cand_recs, inc_front, cand_front, holdouts = [], [], [], [], []
    for val_season, train_seasons in tm.HOLDOUTS:
        ir, ifr = score_holdout(post, val_season, train_seasons, FEATURE_COLUMNS, add_grid=False)
        cr, cfr = score_holdout(post, val_season, train_seasons, cand_cols, add_grid=True)
        assert [(r["season"], r["round"]) for r in ir] == [(r["season"], r["round"]) for r in cr]
        inc_recs += ir
        cand_recs += cr
        inc_front += ifr
        cand_front += cfr
        holdouts.append({"val_season": val_season, "train_seasons": train_seasons, "n_races": len(ir)})
        print(f"holdout {val_season} done ({len(ir)} races)", flush=True)

    comparisons = {}
    for m in METRICS:
        inc = np.array([r[m] for r in inc_recs], dtype=float)
        cand = np.array([r[m] for r in cand_recs], dtype=float)
        comparisons[m] = paired_ci(inc, cand)
    ll = comparisons["win_logloss_multiclass"]
    verdict = {
        "win_logloss_multiclass_improves_ci_excludes_zero": ll["delta_ci95"][1] < 0,
        "brier_podium_not_worse": comparisons["brier_podium"]["delta_mean"] <= 0,
        "brier_top6_not_worse": comparisons["brier_top6"]["delta_mean"] <= 0,
        "brier_top10_not_worse": comparisons["brier_top10"]["delta_mean"] <= 0,
        "spearman_not_worse": comparisons["spearman_rank_corr"]["delta_mean"] >= 0,
    }
    verdict["PASS"] = all(verdict.values())

    def front_row(rows):
        out = {}
        for g in (1, 2):
            sel = [r for r in rows if r[0] == g]
            out[f"grid_{g}"] = {"mean_p_win": round(float(np.mean([r[1] for r in sel])), 4),
                                "actual_win_rate": round(float(np.mean([r[2] for r in sel])), 4),
                                "n": len(sel)}
        return out

    return {
        "meta": {
            "generated": date.today().isoformat(),
            "baseline_pr": "PR #84 (f1-1-tier-metrics), scripts/tier_metrics.py",
            "incumbent": "xgb_ranker on FEATURE_COLUMNS, post_qualifying rows, trained on training seasons only",
            "candidate": "incumbent + grid_interaction.GRID_FEATURES",
            "holdouts": holdouts,
            "n_races_pooled": len(inc_recs),
            "n_trials": tm.N_TRIALS,
            "sim_seed": tm.SIM_SEED,
            "bootstrap": f"paired race-level percentile, {N_BOOT} resamples, seed {BOOT_SEED}",
            "grid_k_candidates": list(gi.K_GRID),
            "flag_default": gi.GRID_INTERACTION_ENABLED,
        },
        "comparisons": comparisons,
        "verdict": verdict,
        "front_row_diagnostic": {"incumbent": front_row(inc_front), "candidate": front_row(cand_front)},
        "per_race": {"incumbent": inc_recs, "candidate": cand_recs},
    }


def write_outputs(result: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "f1_grid_interaction.json").write_text(json.dumps(result, indent=2) + "\n")
    c = result["comparisons"]
    lines = [
        "# F1-2 grid-interaction features (research, flag default OFF)",
        "",
        f"Hold-outs (expanding window, post_qualifying): "
        + ", ".join(f"{h['val_season']} ({h['n_races']} races)" for h in result["meta"]["holdouts"])
        + f". Pooled n = {result['meta']['n_races_pooled']}.",
        "",
        "Incumbent = PR #84 ranker. Candidate = same ranker plus grid-interaction features. "
        "Delta = candidate - incumbent, paired race-level bootstrap 95% CI.",
        "",
        "| Metric | Incumbent | Candidate | Delta [95% CI] | Rule |",
        "|---|---|---|---|---|",
    ]
    rules = {
        "win_logloss_multiclass": "CI entirely < 0",
        "win_logloss_binary_per_driver": "report only",
        "spearman_rank_corr": "delta >= 0",
        "brier_podium": "delta <= 0",
        "brier_top6": "delta <= 0",
        "brier_top10": "delta <= 0",
    }
    for m in METRICS:
        v = c[m]
        lines.append(f"| {m} | {v['incumbent_mean']:.4f} | {v['candidate_mean']:.4f} | "
                     f"{v['delta_mean']:+.4f} [{v['delta_ci95'][0]:+.4f}, {v['delta_ci95'][1]:+.4f}] | {rules[m]} |")
    fr = result["front_row_diagnostic"]
    lines += ["", "## Front-row calibration (grid 1-2, pooled)", "",
              "| Grid | Incumbent mean p_win | Candidate mean p_win | Actual win rate | n |",
              "|---|---|---|---|---|"]
    for g in ("grid_1", "grid_2"):
        lines.append(f"| {g[-1]} | {fr['incumbent'][g]['mean_p_win']:.4f} | {fr['candidate'][g]['mean_p_win']:.4f} | "
                     f"{fr['candidate'][g]['actual_win_rate']:.4f} | {fr['candidate'][g]['n']} |")
    lines += ["", "## Verdict", ""]
    for k, v in result["verdict"].items():
        lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines += ["", f"Default flag: grid_interaction.GRID_INTERACTION_ENABLED = {result['meta']['flag_default']} "
              "(not wired into serving in either case).", ""]
    (DOCS / "f1_grid_interaction.md").write_text("\n".join(lines))


if __name__ == "__main__":
    write_outputs(run())
