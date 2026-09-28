"""GET /api/snapshot-meta — when the numbers this site serves were made.

The snapshot carries generated_at at its top level, but every route that
reads it returns a nested block (a season's races, a round's predictions)
and drops it, so no other endpoint can answer "how old are these
numbers?". Uses _public_snapshot() directly rather than
_snapshot_for_season() so the endpoint answers even when no snapshot
matches the current season."""

from fastapi.testclient import TestClient

from f1_predictor.api import routes
from f1_predictor.api.main import app


def test_snapshot_meta_returns_snapshot_generated_at_verbatim(monkeypatch):
    snap = {"generated_at": "2026-09-24T21:41:15.182603+00:00", "season": 2026}
    monkeypatch.setattr(routes, "_public_snapshot", lambda: snap)
    body = TestClient(app).get("/api/snapshot-meta").json()
    assert body == {
        "generated_at": "2026-09-24T21:41:15.182603+00:00",
        "source": "public_snapshot",
    }


def test_snapshot_meta_no_snapshot_reports_live(monkeypatch):
    monkeypatch.setattr(routes, "_public_snapshot", lambda: {})
    resp = TestClient(app).get("/api/snapshot-meta")
    assert resp.status_code == 200
    assert resp.json() == {"generated_at": None, "source": "live"}


def test_snapshot_meta_reachable_as_json(monkeypatch):
    snap = {"generated_at": "2026-09-24T21:41:15.182603+00:00"}
    monkeypatch.setattr(routes, "_public_snapshot", lambda: snap)
    resp = TestClient(app).get("/api/snapshot-meta")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert set(body) == {"generated_at", "source"}
