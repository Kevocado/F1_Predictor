"""retrain_gate.py -- safety gate for the automatic post-race retrain
(.github/workflows/retrain-after-race.yml).

    python -m f1_predictor.retrain_gate needed            # exit 0 = a new race landed, 10 = nothing to do
    python -m f1_predictor.retrain_gate compare OLD NEW   # exit 0 = accept candidate, 11 = reject

Accept rule (compare), on the walk-forward folds recorded in manifest.json
(walk_forward_folds[<chosen candidate>], metric = log_loss, the same number
models/manifest.py selects on). Folds are matched by val_season:

  * a current fold missing from the candidate, a NaN metric, or a fold with
    FEWER races than before  -> reject.
  * "comparable" fold (same n_races): candidate may be worse by at most
    FOLD_TOL on any single fold, and by at most MEAN_TOL on the mean delta
    across comparable folds.
  * "grown" fold (more races -- the in-progress season gained the new race,
    so the validation set changed): not comparable like-for-like, so it is
    left out of the mean and only held to a loose sanity bound LIVE_TOL.
  * folds only in the candidate (a new season) are ignored.

  * GRID LOOKUP FLOOR: the served model (calibrated if calibration_params
    exists, otherwise raw) must not be worse than the grid lookup baseline
    on any comparable fold (delta <= GRID_TOL). This prevents deploying a
    model that loses to a simple grid-position heuristic.

Honest limit: walk-forward trains each fold on EARLIER seasons only, so a
new race in the current season changes only the last (grown) fold's
validation set -- the gate mostly guards against regressions from data or
code changes, it cannot prove the final all-data model is better. The final
model is trained on everything, same as manifest.train_all always did.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

MEAN_TOL = 0.002   # mean log-loss may rise by at most this over comparable folds
FOLD_TOL = 0.01    # ... and no single comparable fold by more than this
LIVE_TOL = 0.03    # sanity bound for the grown (in-progress season) fold
GRID_TOL = 0.000   # served model must not be worse than grid lookup (delta <= 0)


def _folds(manifest: dict, key: str | None = None) -> dict[int, dict]:
    """Extract folds from manifest. If key is None, uses the served model's folds."""
    if key is None:
        # If calibration params exist, the served model is the calibrated one
        if manifest.get("calibration_params") is not None:
            key = "calibrated"
        else:
            key = manifest["race_outcome_candidate"]
    return {int(f["val_season"]): f for f in manifest["walk_forward_folds"].get(key, [])}


def compare(current: dict, candidate: dict) -> dict:
    cur, cand = _folds(current), _folds(candidate)
    grid_cur = _folds(current, "grid_lookup")
    reasons: list[str] = []
    rows: list[dict] = []
    comparable_deltas: list[float] = []
    for season, c in sorted(cur.items()):
        n = cand.get(season)
        if n is None:
            reasons.append(f"fold {season} missing from candidate")
            continue
        delta = n["log_loss"] - c["log_loss"]
        if math.isnan(delta):
            reasons.append(f"fold {season}: candidate log_loss is not a number")
            continue
        if n["n_races"] < c["n_races"]:
            kind = "fewer-races"
            reasons.append(f"fold {season}: candidate saw fewer races ({n['n_races']} < {c['n_races']})")
        elif n["n_races"] > c["n_races"]:
            kind = "grown"
            if delta > LIVE_TOL:
                reasons.append(f"fold {season} (grown val set): log_loss +{delta:.4f} > {LIVE_TOL}")
        else:
            kind = "comparable"
            comparable_deltas.append(delta)
            if delta > FOLD_TOL:
                reasons.append(f"fold {season}: log_loss +{delta:.4f} > {FOLD_TOL}")
            # Grid lookup floor: candidate must not be worse than grid lookup
            g = grid_cur.get(season)
            if g is not None and not math.isnan(g["log_loss"]):
                grid_delta = n["log_loss"] - g["log_loss"]
                if grid_delta > GRID_TOL:
                    reasons.append(f"fold {season}: candidate worse than grid lookup by +{grid_delta:.4f} > {GRID_TOL}")
        rows.append({"val_season": season, "kind": kind, "n_races": [c["n_races"], n["n_races"]],
                     "log_loss": [c["log_loss"], n["log_loss"]], "delta": delta})
    mean_delta = sum(comparable_deltas) / len(comparable_deltas) if comparable_deltas else 0.0
    if mean_delta > MEAN_TOL:
        reasons.append(f"mean log_loss over comparable folds +{mean_delta:.4f} > {MEAN_TOL}")
    return {"accept": not reasons, "reasons": reasons, "mean_delta": mean_delta, "rows": rows}


def summary_markdown(current: dict, candidate: dict, result: dict) -> str:
    verdict = "ACCEPTED" if result["accept"] else "REJECTED"
    lines = [
        f"## Retrain candidate: {verdict}",
        f"current trained_at `{current['trained_at']}` ({current['race_outcome_candidate']}) -> "
        f"candidate `{candidate['trained_at']}` ({candidate['race_outcome_candidate']})",
        "",
        "| val season | kind | races (cur/cand) | log_loss cur | log_loss cand | delta | grid_delta |",
        "|---|---|---|---|---|---|---|",
    ]
    grid_cur = _folds(current, "grid_lookup")
    for r in result["rows"]:
        grid_delta_str = ""
        g = grid_cur.get(r["val_season"])
        if g is not None and not math.isnan(g["log_loss"]):
            grid_delta = r["log_loss"][1] - g["log_loss"]
            grid_delta_str = f"{grid_delta:+.4f}"
        lines.append(f"| {r['val_season']} | {r['kind']} | {r['n_races'][0]}/{r['n_races'][1]} | "
                     f"{r['log_loss'][0]:.4f} | {r['log_loss'][1]:.4f} | {r['delta']:+.4f} | {grid_delta_str} |")
    lines += ["", f"mean delta (comparable folds): {result['mean_delta']:+.4f} (tolerance {MEAN_TOL})"]
    lines += [f"- {x}" for x in result["reasons"]]
    return "\n".join(lines) + "\n"


def needs_retrain(manifest: dict, season: int, latest_completed_round: int) -> bool:
    """True if a completed race exists that the committed manifest has not trained on."""
    last = manifest.get("last_race")
    if last and last["season"] == season:
        trained = last["round"]
    elif last:
        trained = 0
    else:  # manifest predates last_race: the current season's fold counts its races
        folds = [f for f in manifest["walk_forward_folds"][manifest["race_outcome_candidate"]]
                 if f["val_season"] == season]
        trained = folds[-1]["n_races"] if folds else 0
    return latest_completed_round > trained


def _latest_completed_round(season: int) -> int:
    from .data import jolpica

    results = jolpica.load_season_results(season)
    return int(results["round"].max()) if not results.empty else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["needed"]:
        from . import config
        from .models.manifest import load_manifest

        latest = _latest_completed_round(config.CURRENT_SEASON)
        needed = needs_retrain(load_manifest(), config.CURRENT_SEASON, latest)
        print(f"latest completed round {latest}; retrain needed: {needed}")
        sys.exit(0 if needed else 10)
    elif len(args) == 3 and args[0] == "compare":
        old, new = (json.loads(Path(p).read_text()) for p in args[1:])
        result = compare(old, new)
        md = summary_markdown(old, new, result)
        print(md)
        Path("retrain_summary.md").write_text(md)
        sys.exit(0 if result["accept"] else 11)
    else:
        sys.exit("usage: python -m f1_predictor.retrain_gate needed | compare OLD.json NEW.json")
