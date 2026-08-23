"""manifest.py — trains and persists every Phase 1 model: races the two
models/race_outcome.py candidates against each other via
evaluate/walk_forward.py, trains the winning candidate (and the DNF
reliability model) on the full historical frame, and writes
models/manifest.json + the model files. Mirrors PL_Predictor's
models/manifest.py structure (fit everything, pick the best scoreline
model on held-out metric, persist).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ..config import MODELS_DIR
from ..data import jolpica
from ..evaluate import walk_forward
from ..features import session_state
from ..features.build import build_training_frame
from . import dnf as dnf_model
from . import race_outcome

RACE_OUTCOME_MODEL_PATH = MODELS_DIR / "race_outcome_ranker.json"
DNF_MODEL_PATH = MODELS_DIR / "dnf_model.json"
MANIFEST_PATH = MODELS_DIR / "manifest.json"


def train_all(seasons: list[int] | None = None) -> dict:
    seasons = seasons or jolpica.default_seasons()
    df, feature_cols = build_training_frame(seasons=seasons)
    post_quali = df[df["tier"] == session_state.TIER_POST_QUALIFYING]

    print(f"Training frame: {len(df)} rows ({len(post_quali)} post-qualifying) across seasons {seasons}")
    print("Running walk-forward validation (elo vs. xgb_ranker candidates)...")
    folds = walk_forward.prepare_folds(seasons)
    elo_metrics = walk_forward.evaluate_candidate(folds, "elo")
    xgb_metrics = walk_forward.evaluate_candidate(folds, "xgb_ranker")

    # Selection uses only the LAST fold (trained on every season but the
    # most recent — the most historical data of any fold, closest to what
    # actually gets trained below). The multi-fold average is a separate
    # robustness check (mirrors PL_Predictor's walk_forward.py, explicitly
    # NOT wired into its own candidate selection): an early fold with only
    # 3 training seasons unfairly penalizes the more data-hungry
    # xgb_ranker candidate relative to what it sees once trained on all of
    # `seasons` — confirmed directly: xgb_ranker's held-out log-loss
    # improves monotonically fold-over-fold as training data grows, while
    # the multi-fold mean was still dragged down by its cold-start folds.
    elo_avg = float(elo_metrics["log_loss"].mean())
    xgb_avg = float(xgb_metrics["log_loss"].mean())
    elo_last = float(elo_metrics.iloc[-1]["log_loss"])
    xgb_last = float(xgb_metrics.iloc[-1]["log_loss"])
    chosen = "elo" if elo_last <= xgb_last else "xgb_ranker"
    print(f"Multi-fold mean log-loss  — elo: {elo_avg:.4f}, xgb_ranker: {xgb_avg:.4f}")
    print(f"Last-fold (selection) log-loss — elo: {elo_last:.4f}, xgb_ranker: {xgb_last:.4f} -> chosen: {chosen}")

    print("Training final race_outcome ranker on full history...")
    ranker = race_outcome.train_ranker(post_quali, feature_cols)
    ranker.save_model(str(RACE_OUTCOME_MODEL_PATH))

    print("Training DNF reliability model on full history...")
    dnf_clf = dnf_model.train_dnf_model(post_quali, feature_cols)
    dnf_clf.save_model(str(DNF_MODEL_PATH))

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "feature_cols": feature_cols,
        "race_outcome_candidate": chosen,
        "race_outcome_candidate_metrics": {
            "elo_last_fold_log_loss": elo_last,
            "xgb_ranker_last_fold_log_loss": xgb_last,
            "elo_mean_log_loss": elo_avg,
            "xgb_ranker_mean_log_loss": xgb_avg,
        },
        "walk_forward_folds": {
            "elo": elo_metrics.to_dict(orient="records"),
            "xgb_ranker": xgb_metrics.to_dict(orient="records"),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Wrote {MANIFEST_PATH}")
    return manifest


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


if __name__ == "__main__":
    train_all()
