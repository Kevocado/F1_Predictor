"""The /api/explain proxy: what it forwards, what it refuses, and what it never echoes.

Replaces `test_no_explain_proxy.py`, which asserted the absence. An absent route
cannot fail on its own, so those tests could only ever confirm that nothing had
been added; these assert the properties a live proxy has to have.
"""
import pytest
import requests
from fastapi.testclient import TestClient

from f1_predictor.api import explain as explain_module
from f1_predictor.api.main import app


class FakeResponse:
    """A stand-in with a real response attached, the way `requests` delivers one.

    A bare `HTTPError` with no response carries nothing to echo, so a fake built
    that way would leave the no-echo assertions vacuous: any implementation,
    including one returning the upstream body, would pass. The body lives here.
    """

    def __init__(self, status_code=200, payload=None, raw=""):
        self.status_code = status_code
        self._payload = payload
        self.text = raw

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


def _ok(monkeypatch, payload=None):
    body = {"verdict": "v", "factors": []} if payload is None else payload
    monkeypatch.setattr(explain_module.requests, "get", lambda *a, **k: FakeResponse(200, body))
    return body


def test_the_route_is_registered():
    """From OpenAPI, not `app.routes`: FastAPI wraps `include_router` in an
    `_IncludedRouter` with no `.path`, so reading routes directly passes while
    the proxy is still missing."""
    assert "/api/explain/{sport}/{explainer_id}" in set(app.openapi()["paths"])


def test_a_usable_answer_passes_through_verbatim(client, monkeypatch):
    body = _ok(monkeypatch)
    res = client.get("/api/explain/f1/2026-16-race")
    assert res.status_code == 200
    assert res.json() == body


def test_a_non_2xx_becomes_the_fixed_502(client, monkeypatch):
    for status in (404, 500, 502):
        monkeypatch.setattr(
            explain_module.requests, "get", lambda *a, **k: FakeResponse(status, {"detail": "upstream secret"})
        )
        res = client.get("/api/explain/f1/2026-16-race")
        assert res.status_code == 502
        # Equal, not containing: the fixture id coincidence class of mistake
        # (a substring that always matches) cannot satisfy an equality.
        assert res.json() == {"detail": "The summary service is not available."}


def test_a_raised_error_becomes_the_fixed_502(client, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(explain_module.requests, "get", boom)
    res = client.get("/api/explain/f1/2026-16-race")
    assert res.status_code == 502
    assert res.json() == {"detail": "The summary service is not available."}


def test_a_2xx_that_is_not_a_dict_is_a_502(client, monkeypatch):
    monkeypatch.setattr(explain_module.requests, "get", lambda *a, **k: FakeResponse(200, ["not", "a", "dict"]))
    res = client.get("/api/explain/f1/2026-16-race")
    assert res.status_code == 502


def test_a_2xx_that_is_not_json_is_a_502(client, monkeypatch):
    monkeypatch.setattr(
        explain_module.requests, "get", lambda *a, **k: FakeResponse(200, requests.JSONDecodeError("x", "y", 0))
    )
    res = client.get("/api/explain/f1/2026-16-race")
    assert res.status_code == 502


@pytest.mark.parametrize("sport,explainer_id", [
    ("f1", "../x"),
    ("f1", "/etc/passwd"),
    ("f1", "..%2Fetc%2Fpasswd"),
])
def test_a_walk_out_never_reaches_the_explainer(client, monkeypatch, sport, explainer_id):
    calls: list = []
    monkeypatch.setattr(
        explain_module.requests, "get", lambda *a, **k: (calls.append((a, k)), FakeResponse(200, {}))[1]
    )
    res = client.get(f"/api/explain/{sport}/{explainer_id}")
    assert res.status_code in (404, 502)
    assert calls == []


@pytest.mark.parametrize("sport,explainer_id", [
    # Through the HTTP client only the id branch is reachable (the client
    # normalises `secrets/../x` before routing), so `sport` is tested by
    # calling the route function directly.
    ("f1", "secrets/../x"),
    ("f1/x", "2026-16-race"),
    ("f1", "/etc/passwd"),
    ("f1", "../x"),
])
def test_the_guard_refuses_before_any_request_leaves(monkeypatch, sport, explainer_id):
    calls: list = []
    monkeypatch.setattr(
        explain_module.requests, "get", lambda *a, **k: (calls.append((a, k)), FakeResponse(200, {}))[1]
    )
    with pytest.raises(Exception) as excinfo:
        explain_module.explain(sport, explainer_id)
    assert getattr(excinfo.value, "status_code", None) == 502
    assert calls == []


def test_timeout_and_no_redirects_reach_the_call(monkeypatch):
    seen: dict = {}

    def fake_get(url, **kwargs):
        seen.update(kwargs)
        return FakeResponse(200, {"verdict": "v", "factors": []})

    monkeypatch.setattr(explain_module.requests, "get", fake_get)
    TestClient(app, raise_server_exceptions=False).get("/api/explain/f1/2026-16-race")
    assert seen.get("timeout") == explain_module.EXPLAINER_TIMEOUT_S
    assert seen.get("allow_redirects") is False


def test_path_segments_are_quoted(monkeypatch):
    seen: list = []
    monkeypatch.setattr(explain_module.requests, "get", lambda url, **k: (seen.append(url), FakeResponse(200, {}))[1])
    TestClient(app, raise_server_exceptions=False).get("/api/explain/f1/2026%2016%20race")
    assert seen and " " not in seen[0] and "%20" in seen[0]


def test_the_apps_own_routes_still_answer(client):
    """The route-table assertion cannot be satisfied by breaking a neighbour:
    assert a body, not just a status, so the SPA fallback cannot stand in."""
    res = client.get("/openapi.json")
    assert res.status_code == 200
    assert "/api/explain/{sport}/{explainer_id}" in res.json()["paths"]
