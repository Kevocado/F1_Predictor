"""explain.py — per-prediction feature attribution via XGBoost's native
TreeSHAP (`pred_contribs=True` — exact Shapley values, no extra
dependency needed beyond xgboost itself).

Requirement: "what's driving this decision," exposed per-driver in the
frontend. The key structural fact this module is built around: Win,
Podium, and Points-finish are NOT three separately-fit models — they're
all derived from ONE per-driver strength score via
`models/race_outcome.py`'s Plackett-Luce Monte Carlo simulation. So there
is exactly ONE explanation behind all three markets (what drove this
driver's predicted strength for this race); DNF is a genuinely separate
model with its own explanation. The API/frontend surface this honestly
rather than pretending there are four independent stories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..features.elo import INITIAL_ELO


def _tree_shap(model: xgb.XGBModel, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    dmatrix = xgb.DMatrix(X[feature_cols], feature_names=feature_cols)
    return model.get_booster().predict(dmatrix, pred_contribs=True)


def explain_strength_xgb(ranker: xgb.XGBRanker, race_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-driver SHAP contributions to the xgb_ranker candidate's raw
    strength score — drives Win/Podium/Points-finish together."""
    contribs = _tree_shap(ranker, race_df, feature_cols)
    out = pd.DataFrame(contribs[:, :-1], columns=feature_cols)
    out.insert(0, "driver_id", race_df["driver_id"].values)
    out["base_value"] = contribs[:, -1]
    return out


def explain_strength_elo(race_df: pd.DataFrame) -> pd.DataFrame:
    """The elo candidate has no tree ensemble to attribute — its strength
    is literally `elo_pre_race + team_strength_pre_race`, so the
    "contribution" of each term is just its own deviation from the
    INITIAL_ELO baseline both ratings start from."""
    out = pd.DataFrame(
        {
            "driver_id": race_df["driver_id"].values,
            "elo_pre_race": race_df["elo_pre_race"].fillna(INITIAL_ELO).to_numpy() - INITIAL_ELO,
            "team_strength_pre_race": race_df["team_strength_pre_race"].fillna(INITIAL_ELO).to_numpy() - INITIAL_ELO,
        }
    )
    out["base_value"] = 2 * INITIAL_ELO
    return out


def explain_dnf(dnf_clf: xgb.XGBClassifier, race_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-driver SHAP contributions to the DNF model's predicted logit —
    a separate model, a separate explanation from the strength score."""
    contribs = _tree_shap(dnf_clf, race_df, feature_cols)
    out = pd.DataFrame(contribs[:, :-1], columns=feature_cols)
    out.insert(0, "driver_id", race_df["driver_id"].values)
    out["base_value"] = contribs[:, -1]
    return out


def top_contributors(contrib_row: pd.Series, raw_row: pd.Series, feature_cols: list[str], n: int = 6) -> list[dict]:
    """Top-N |contribution| features for one driver, with the raw feature
    value alongside the signed contribution — "grid: 3 -> +0.41" is more
    legible than the contribution alone."""
    present = [f for f in feature_cols if f in contrib_row.index]
    vals = contrib_row[present].astype(float)
    ranked = vals.abs().sort_values(ascending=False).head(n)
    return [
        {
            "feature": f,
            "value": None if pd.isna(raw_row.get(f)) else float(raw_row.get(f)),
            "contribution": float(vals[f]),
        }
        for f in ranked.index
    ]
