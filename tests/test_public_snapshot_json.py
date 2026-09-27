"""A snapshot holding NaN is invalid JSON on disk, whoever reads it.

json.dumps defaults to allow_nan=True, so a single non-finite float used to
be written as a bare NaN token. The public deployment serves that file
verbatim, and starlette renders responses with allow_nan=False -- so the
endpoint 500s rather than returning a body the browser can fail to parse.
"""

import json
import math

import pytest

from f1_predictor.public_snapshot import sanitize_floats


def test_nan_becomes_null():
    assert sanitize_floats({"home_cover_prob": float("nan")}) == {"home_cover_prob": None}


def test_infinities_become_null():
    assert sanitize_floats({"a": float("inf"), "b": float("-inf")}) == {"a": None, "b": None}


def test_walks_nested_structures():
    payload = {"weeks": {"9": {"predictions": {"g1": {"sigma": float("nan")}}}}}
    assert sanitize_floats(payload) == {"weeks": {"9": {"predictions": {"g1": {"sigma": None}}}}}


def test_leaves_real_numbers_and_types_alone():
    payload = {"p": 0.5, "n": 3, "s": "x", "b": True, "none": None, "empty": [], "d": {}}
    assert sanitize_floats(payload) == payload


def test_sanitised_snapshot_survives_a_strict_dump():
    """The regression guard: allow_nan=False is what makes a future NaN loud."""
    snapshot = sanitize_floats({"predictions": {"g1": {"over_prob": float("nan")}}})
    text = json.dumps(snapshot, indent=2, allow_nan=False)
    assert json.loads(text)["predictions"]["g1"]["over_prob"] is None


def test_the_committed_snapshot_is_strict_json():
    """The file on disk must parse with allow_nan=False, not just json.loads."""
    from f1_predictor.config import PUBLIC_SNAPSHOT_PATH

    if not PUBLIC_SNAPSHOT_PATH.exists():
        pytest.skip("no committed snapshot in this checkout")
    text = PUBLIC_SNAPSHOT_PATH.read_text()
    json.loads(text, parse_constant=_reject)  # parse_constant fires on NaN/Infinity


def _reject(name):
    raise AssertionError(f"snapshot contains a bare {name} token")


def test_expected_position_and_points_are_null_not_nan():
    """The two fields the hub's F1 teaser would otherwise render as NaN."""
    row = {"driver_id": "russell", "p_win": 0.15, "expected_position": float("nan"), "expected_points": float("nan")}
    out = sanitize_floats({"predictions": {"1": {"predictions": [row]}}})
    got = out["predictions"]["1"]["predictions"][0]
    assert got["expected_position"] is None
    assert got["expected_points"] is None
    assert got["p_win"] == 0.15
