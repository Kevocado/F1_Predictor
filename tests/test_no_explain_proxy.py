"""F1 does not serve a plain-English summary, and must not look like it does.

The explainer now refuses `f1` with a 404 (predictor-hub#10, merged and live).
Until this landed, F1's detail views still rendered the v1 panel, which turned
that 404 into "The summary didn't come through. Try again…" — an error box on
every session detail, on a page whose real content was fine.

So the panel and the proxy are gone. The test is here because the *absence* is
the thing that regressed, and an absent route has no way to fail on its own: the
next person to add "a harmless proxy" would be adding a route that can only ever
502, and nothing in the suite would have objected.
"""
from fastapi.testclient import TestClient

from f1_predictor.api.main import app


def _paths() -> set[str]:
    """Every path this app answers, from OpenAPI rather than `app.routes`.

    The first version of this read `getattr(route, "path", "")` off
    `app.routes`, and it PASSED while the proxy was still registered: this
    FastAPI version wraps each `include_router` in an `_IncludedRouter` that has
    no `.path` of its own, so the filter never saw the route. Same mistake as the
    fourth bug this project shipped — reading a structure at a level that does
    not contain the thing you care about — so this uses the framework's own
    public rendering of the route table, which also does not depend on
    `_IncludedRouter` staying the storage layout.
    """
    return set(app.openapi()["paths"])


def test_no_route_serves_the_shared_explainer():
    """Scoped to `/api/explain`, not to the word.

    `/api/races/{season}/{round_}/explain` is F1's own per-race explanation and
    is staying; it has nothing to do with the shared service. A guard matching
    "explain" loosely would demand deleting a working feature, and the obvious
    way to satisfy such a guard would be to rename the survivor.
    """
    offenders = sorted(p for p in _paths() if p.startswith("/api/explain"))
    assert not offenders, (
        f"the app still proxies the shared explainer at {offenders}. It is now "
        f"refused upstream with a 404, so this route can only ever be a 502."
    )


def test_the_races_own_explain_route_is_untouched():
    """The control for the test above, so neither can be satisfied by deleting both."""
    assert "/api/races/{season}/{round_}/explain" in _paths(), (
        "F1's own per-race explain route went missing; the removal was meant to "
        "be the shared-service proxy only"
    )


def test_the_proxy_module_is_gone():
    import importlib

    try:
        importlib.import_module("f1_predictor.api.explain")
    except ModuleNotFoundError:
        return
    raise AssertionError(
        "f1_predictor.api.explain still imports. A dead proxy is worse than "
        "none: it is an endpoint that looks supported."
    )


def test_a_request_for_a_summary_is_a_plain_404_not_a_200():
    """The production risk is a 200, not a 404.

    `app.mount("/", StaticFiles(..., html=True))` sits below where the proxy used
    to be registered. If that catch-all ever started answering `/api/explain/*`
    with `index.html` and a 200, the frontend would be handed an HTML document
    where it expects a summary and fail in a far more confusing place. So the
    assertion is on the *body*, not just the status: no HTML, ever.
    """
    with TestClient(app) as c:
        res = c.get("/api/explain/f1/2026-5-race")
    assert res.status_code == 404, f"got {res.status_code}, expected a plain 404"
    assert "<!doctype html" not in res.text.lower(), (
        "the static catch-all answered with index.html; the SPA fallback must "
        "not swallow a path under /api/"
    )


def test_the_apps_own_routes_still_answer():
    """So the guard above cannot be satisfied by breaking the app.

    Every version of "the explainer is gone" that also breaks `/facts` is not a
    removal, it is an outage, and a test that only checked for absence would pass
    it happily.
    """
    with TestClient(app) as c:
        res = c.get("/facts/2026-5-race")
    assert res.status_code == 200, f"/facts broke while removing the proxy: {res.status_code}"
