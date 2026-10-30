"""grid_interaction.py — grid-to-finish interaction features (F1-2 research).

OFF by default (GRID_INTERACTION_ENABLED = False). Nothing in the serving
path reads these columns; scripts/grid_interaction.py is the only caller and
it opts in explicitly. Flipping the flag does not retrain or re-wire anything.

Features (all derived from TRAINING seasons only, see grid_features):
  - grid_curve_top3 / grid_curve_win: P(outcome | circuit, grid slot), an
    empirical-Bayes shrinkage of the per-circuit cell rate toward the global
    grid curve. The concentration k is fitted by maximising the beta-binomial
    marginal likelihood over the training cells (type-II maximum likelihood).
  - grid_base_top3 / grid_base_win: the global grid-only probability (the
    simple grid baseline), add-one smoothed.
  - quali_gap_pct: gap to pole as a percent of pole lap time.
  - teammate_gap_pct: own best lap minus teammate's best lap, percent of pole.

Leakage control: rows scored with their own training data (loo=True) have
their own race removed from the counts; rows scored out-of-sample use the
training seasons only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import betabinom

from .build import _parse_time_to_seconds

GRID_INTERACTION_ENABLED = False

K_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
GRID_FEATURES = [
    "grid_curve_top3",
    "grid_curve_win",
    "grid_base_top3",
    "grid_base_win",
    "quali_gap_pct",
    "teammate_gap_pct",
]


def _hit(position: pd.Series, outcome: str) -> pd.Series:
    return (position <= 3) if outcome == "top3" else (position == 1)


def grid_features(train: pd.DataFrame, target: pd.DataFrame, loo: bool) -> pd.DataFrame:
    """Grid-curve features for `target` rows, using counts from `train` only.

    train must carry season, round, circuit_id, grid, position. If loo is True,
    target is train itself and each row's own result is removed from the counts.
    Returns a frame aligned to target.index."""
    t = train[train["grid"].notna() & train["position"].notna() & train["circuit_id"].notna()]
    out = pd.DataFrame(index=target.index)
    for outcome in ("top3", "win"):
        hit_t = _hit(t["position"], outcome).astype(float)
        g_t = t["grid"].astype(int)
        slot_h = hit_t.groupby(g_t).sum()
        slot_n = hit_t.groupby(g_t).size()
        overall = (hit_t.sum() + 1) / (len(hit_t) + 2)
        prior = ((slot_h + 1) / (slot_n + 2)).to_dict()           # global curve by slot
        cells = pd.DataFrame({"c": t["circuit_id"], "g": g_t, "h": hit_t})
        cells = cells.groupby(["c", "g"]).agg(H=("h", "sum"), N=("h", "size"))
        k = _fit_k(cells, prior)

        cell_h = cells["H"].to_dict()
        cell_n = cells["N"].to_dict()
        curve, base = [], []
        for c, g, pos in zip(target["circuit_id"], target["grid"], target["position"]):
            if pd.isna(g) or pd.isna(c):
                curve.append(overall)
                base.append(overall)
                continue
            gi = int(g)
            # the row's own contribution to the counts (only if it was counted)
            counted = loo and pd.notna(pos)
            own = float(pos <= 3 if outcome == "top3" else pos == 1) if counted else 0.0
            own_n = 1 if counted else 0
            m = (slot_h.get(gi, 0.0) - own + 1) / (slot_n.get(gi, 0) - own_n + 2)
            H = cell_h.get((c, gi), 0.0) - own
            N = cell_n.get((c, gi), 0) - own_n
            curve.append((H + k * m) / (N + k))
            base.append(m)
        out[f"grid_curve_{outcome}"] = curve
        out[f"grid_base_{outcome}"] = base
    return out


def _fit_k(cells: pd.DataFrame, prior: dict) -> float:
    """Type-II ML concentration: maximise sum over training cells of the
    beta-binomial log-likelihood with prior mean = global curve at that slot."""
    h = cells["H"].to_numpy(dtype=float)
    n = cells["N"].to_numpy(dtype=float)
    m = np.array([prior[g] for g in cells.index.get_level_values("g")])
    best_k, best_ll = K_GRID[0], -np.inf
    for k in K_GRID:
        ll = float(betabinom.logpmf(h, n, k * m, k * (1 - m)).sum())
        if ll > best_ll:
            best_k, best_ll = k, ll
    return float(best_k)


def quali_gap_features(quali_df: pd.DataFrame) -> pd.DataFrame:
    """quali_gap_pct and teammate_gap_pct per (season, round, driver_id)."""
    df = quali_df.copy()
    df["best_s"] = df[["q1", "q2", "q3"]].map(_parse_time_to_seconds).min(axis=1)
    df["pole_s"] = df.groupby(["season", "round"])["best_s"].transform("min")
    df["quali_gap_pct"] = (df["best_s"] - df["pole_s"]) / df["pole_s"] * 100
    grp = df.groupby(["season", "round", "constructor_id"])["best_s"]
    n_valid = grp.transform("count")
    mate_sum = grp.transform("sum") - df["best_s"]
    both = (n_valid == 2) & df["best_s"].notna()
    df["teammate_gap_pct"] = np.where(both, (df["best_s"] - mate_sum) / df["pole_s"] * 100, np.nan)
    return df[["season", "round", "driver_id", "quali_gap_pct", "teammate_gap_pct"]]
