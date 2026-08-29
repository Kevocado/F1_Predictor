import argparse
import pandas as pd
import xgboost as xgb
from ..models import live_win_prob
from ..config import CURRENT_SEASON

ENERGY_COLS = [
    "inferred_soc",
    "deploy_time_s",
    "harvest_time_s",
    "clipping_time_s",
]

def evaluate(corpus: pd.DataFrame, use_energy: bool, target="won", seed=0) -> float:
    train_df, val_df = live_win_prob.race_level_split(corpus, seed=seed)
    X_train, X_val = live_win_prob.prep_features(train_df), live_win_prob.prep_features(val_df)
    
    if not use_energy:
        X_train = X_train.drop(columns=ENERGY_COLS, errors="ignore")
        X_val = X_val.drop(columns=ENERGY_COLS, errors="ignore")
        
    y_train = train_df[target].astype(int)
    y_val = val_df[target].astype(int)
    
    params = {
        "objective": "binary:logistic",
        "n_estimators": 400,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "early_stopping_rounds": 30,
        "max_depth": 5,
        "learning_rate": 0.05,
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Return best logloss
    return float(model.evals_result()["validation_0"]["logloss"][model.best_iteration])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=[CURRENT_SEASON])
    parser.add_argument("--target", choices=["won", "podium"], default="won")
    args = parser.parse_args()
    
    print(f"Building training corpus with telemetry for seasons {args.seasons}...")
    corpus = live_win_prob.build_training_corpus(seasons=args.seasons, verbose=True, include_telemetry=True)
    
    if corpus.empty:
        print("Corpus is empty, nothing to evaluate.")
        return
        
    # Check if energy features exist and aren't all NaN
    nan_pct = corpus['inferred_soc'].isna().mean()
    print(f"Energy features built. Missing (NaN) values: {nan_pct:.1%}")
    
    print("\nTraining models (With vs Without Energy Features)...")
    
    logloss_full = evaluate(corpus, use_energy=True, target=args.target)
    logloss_ablated = evaluate(corpus, use_energy=False, target=args.target)
    
    delta = logloss_ablated - logloss_full
    
    print("\n--- RESULTS ---")
    print(f"Log-loss WITH energy features:    {logloss_full:.5f}")
    print(f"Log-loss WITHOUT energy features: {logloss_ablated:.5f}")
    print(f"Delta (Without - With):           {delta:+.5f}")
    
    if delta > 0.001:
        print("Verdict: HELPS (Including energy features improved the accuracy by reducing log-loss)")
    elif delta < -0.001:
        print("Verdict: HURTS (Including energy features made the model worse)")
    else:
        print("Verdict: NO CLEAR EFFECT")

if __name__ == "__main__":
    main()
