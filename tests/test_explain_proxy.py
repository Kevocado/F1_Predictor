"""The explainer proxy: it must forward, and it must fail honestly."""
from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from f1_predictor.api import explain as explain_mod


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(explain_mod.router)
    return TestClient(app)


def _resp(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("GET", "http://x"))


def test_forwards_to_the_services_own_route(monkeypatch, client):
    """The site calls /api/explain/f1/<id>; the service serves /explain/f1/<id>."""
    seen = {}

    def fake_get(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return _resp(200, {"headline": "h", "sections": [], "source": "template", "pick_timing": "pre_kickoff"})

    monkeypatch.setattr(explain_mod.httpx, "get", fake_get)
    body = client.get("/api/explain/f1/2026-12-race")
    assert body.status_code == 200
    assert body.json()["headline"] == "h"
    assert seen["url"].endswith("/explain/f1/2026-12-race")


def test_a_session_id_arrives_intact(monkeypatch, client):
    """The id is {season}-{round}-{session}, so it carries hyphens but no
    slashes; the path must not be truncated or re-encoded."""
    seen = {}
    monkeypatch.setattr(
        explain_mod.httpx, "get", lambda url, timeout=None: (seen.update(url=url), _resp(200, {}))[1]
    )
    client.get("/api/explain/f1/2026-12-sprint_qualifying")
    assert seen["url"].endswith("/explain/f1/2026-12-sprint_qualifying")


def test_never_echoes_upstream_settings(monkeypatch, client):
    """The service holds the OpenRouter key. This route must not pass an
    upstream error body through, and its own text is a fixed string."""
    monkeypatch.setattr(
        explain_mod.httpx, "get", lambda url, timeout=None: _resp(500, {"detail": "key sk-abc123 rejected"})
    )
    body = client.get("/api/explain/f1/2026-12-race")
    assert body.status_code == 502
    assert "sk-abc123" not in body.text
    assert body.json()["detail"] == "The summary service is not available."


def test_an_unreachable_explainer_is_a_502_not_a_crash(monkeypatch, client):
    def boom(url, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(explain_mod.httpx, "get", boom)
    body = client.get("/api/explain/f1/2026-12-race")
    assert body.status_code == 502
    assert body.json()["detail"] == "The summary service is not available."


def test_a_timeout_is_a_502_too(monkeypatch, client):
    def slow(url, timeout=None):
        raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(explain_mod.httpx, "get", slow)
    assert client.get("/api/explain/f1/2026-12-race").status_code == 502


def test_no_summary_is_a_404_the_site_can_swallow(monkeypatch, client):
    monkeypatch.setattr(explain_mod.httpx, "get", lambda url, timeout=None: _resp(404, {"detail": "nope"}))
    body = client.get("/api/explain/f1/2026-12-race")
    assert body.status_code == 404
    assert body.json()["detail"] == "No summary for this session."


def test_the_timeout_is_configurable(monkeypatch, client):
    monkeypatch.setattr(explain_mod, "EXPLAINER_TIMEOUT_S", 7.5)
    seen = {}
    monkeypatch.setattr(
        explain_mod.httpx, "get", lambda url, timeout=None: (seen.update(t=timeout), _resp(200, {}))[1]
    )
    client.get("/api/explain/f1/2026-12-race")
    assert seen["t"] == 7.5
