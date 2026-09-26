"""Proxy for the plain-English explainer service.

F1's FastAPI serves the site, so Caddy only ever reverse-proxies this app. The
browser therefore cannot reach the explainer directly: it calls
``/api/explain/{sport}/{id}`` here, and this route forwards to the service's
own ``/explain/{sport}/{id}``.

The proxy is deliberately thin and deliberately unforgiving. A summary is a
nice-to-have on top of a race page, so every upstream failure becomes one fixed
502 and the site's own error state takes it from there — the timing tower never
depends on this route answering. No upstream body is ever returned: an
explainer error can carry key material or an internal path.
"""
from __future__ import annotations

import os
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()

# The explainer is reached over the internal compose network by service name.
EXPLAINER_URL = os.getenv("EXPLAINER_URL", "http://predictor-explainer:8090").rstrip("/")
# Deliberately BELOW the browser's 15 s request timeout. This route is a sync
# def, so an in-flight request occupies one of Starlette's worker threads for
# its whole duration: a proxy that waits longer than the client it serves
# spends a thread on an answer nobody is waiting for. A first uncached session
# usually lands here and the panel shows Try again — that is what
# pre-generation is for.
EXPLAINER_TIMEOUT_S = float(os.getenv("EXPLAINER_TIMEOUT_S", "10"))


@router.get("/api/explain/{sport}/{summary_id:path}")
def explain(sport: str, summary_id: str) -> dict:
    """Forward to the explainer, or say plainly that it could not be reached.

    No upstream body is ever returned. Redirects are not followed, and neither
    path segment is allowed to walk out of the explainer's own route.
    """
    if ".." in sport or ".." in summary_id:
        # Rejected before any request goes out: a path segment that walks out
        # of the explainer's route would make this a forwarder to any path on
        # that host. 502 like every other failure, so the site's error state
        # takes it and no upstream (or no request) is involved.
        raise HTTPException(status_code=502, detail="The summary service is not available.")

    url = f"{EXPLAINER_URL}/explain/{quote(sport, safe='')}/{quote(summary_id, safe='')}"
    try:
        response = httpx.get(url, timeout=EXPLAINER_TIMEOUT_S, follow_redirects=False)
        # Every non-2xx, including a 3xx, is a failure. httpx's is_error is
        # 400..599, so a 302 would otherwise be returned to the browser as if
        # it were a summary.
        if not (200 <= response.status_code < 300):
            raise httpx.HTTPError(f"upstream {response.status_code}")
        return response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="The summary service is not available.") from exc
    except ValueError as exc:
        # A 200 that is not JSON: the explainer is misconfigured or something
        # else is answering on its port. A 502, so the site shows its retry
        # state rather than a bare 500 it cannot interpret.
        raise HTTPException(status_code=502, detail="The summary service is not available.") from exc
