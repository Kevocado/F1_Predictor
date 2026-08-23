"""dnf.py — DNF reliability classifier: requirement 1's fourth target,
and an input to models/race_outcome.py's Monte Carlo simulation (a DNF'd
driver is removed/tail-ranked per trial). XGBoost binary:logistic on the
same session-state-tiered feature frame features/build.py produces —
driver/constructor reliability history, circuit incident rate
(features/safety_car.py), weather.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

DNF_PARAMS = dict(
    objective="binary:logistic",
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,
    tree_method="hist",
)


def train_dnf_model(train_df: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**DNF_PARAMS)
    model.fit(train_df[feature_cols], train_df["dnf"].astype(int))
    return model


def predict_dnf_prob(model: xgb.XGBClassifier, race_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    probs = model.predict_proba(race_df[feature_cols])[:, 1]
    return dict(zip(race_df["driver_id"], probs))
