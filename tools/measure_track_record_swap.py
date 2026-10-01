"""Measure the before -> after of the F1 track-record reversal on the real data.

This one needs no synthetic data. `data/tracking.db` in this checkout holds
1,452 resolved picks, and the reversal is measured on exactly those rows: the
"before" figures are the ones `get_session_track_record` on `origin/main`
publishes, the "after" figures are the ones it publishes on this branch. Both
come from the same file, so the comparison is two rules on one dataset.

Run: python3 tools/measure_track_record_swap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1_predictor.tracking import store  # noqa: E402


def headline_before() -> dict:
    """The rule as it stood on `origin/main`, reimplemented from its definition.

    A session was judged only if EVERY one of its rows was snapshotted before it
    started; a session with any late row was dropped whole. Reimplemented rather
    than checked out, so the comparison is two rules on one database.
    """
    import pandas as pd

    with store._connect() as conn:
        df = pd.read_sql(
            "SELECT season, round, session_type, tier, market, predicted_prob, actual_outcome, "
            "snapshotted_at, session_time FROM session_predictions WHERE resolved = 1",
            conn,
        )
    pre = df.groupby(["season", "round", "session_type", "tier"])["snapshotted_at"].transform(
        lambda stamps: all(store.made_before_session(s, t) for s, t in zip(stamps, df.loc[stamps.index, "session_time"]))
    )
    kept = df[pre]
    session_keys = ["season", "round", "session_type", "tier"]
    return {
        "n_resolved": int(len(kept)),
        "by_market": {
            (st, t, m): int(len(g))
            for (st, t, m), g in kept.groupby(["session_type", "tier", "market"])
        },
        "n_rebuilt_sessions": int(df.loc[~pre, session_keys].drop_duplicates().shape[0]),
    }


def main() -> int:
    before = headline_before()
    after = store.get_session_track_record()

    print(f"data/tracking.db — {after['n_resolved']} resolved picks recorded.\n")
    print(f"{'session/tier/market':46s} {'BEFORE n':>9s} {'AFTER n':>9s} {'PRE-SESSION n':>14s}")
    print("-" * 82)
    for key in sorted({(r["session_type"], r["tier"], r["market"]) for r in after["by_market"]}):
        label = "/".join(key)
        pre = next((r["n"] for r in after["pre_session"]["by_market"]
                    if (r["session_type"], r["tier"], r["market"]) == key), 0)
        print(f"{label:46s} {before['by_market'].get(key, 0):9d} "
              f"{next(r['n'] for r in after['by_market'] if (r['session_type'], r['tier'], r['market']) == key):9d} {pre:14d}")
    print("-" * 82)
    print(f"{'TOTAL':46s} {before['n_resolved']:9d} {after['n_resolved']:9d} {after['n_pre_session']:14d}")
    print()
    print(f"before: {before['n_resolved']} picks published, {before['n_rebuilt_sessions']} sessions withheld whole.")
    print(f"after:  {after['n_resolved']} picks published, {after['n_pre_session']} of them made before the session started,")
    print(f"        {after['n_post_session_picks']} recorded at or after it ({100 * after['n_post_session_picks'] // after['n_resolved']}% of the record).")
    print(f"        identity n_resolved == n_pre_session + n_post_session_picks: "
          f"{after['n_resolved'] == after['n_pre_session'] + after['n_post_session_picks']}")
    assert after["n_resolved"] == after["n_pre_session"] + after["n_post_session_picks"]
    assert after["n_pre_session"] == before["n_resolved"], (
        "the pre-session subset is not the figure the old headline published"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
