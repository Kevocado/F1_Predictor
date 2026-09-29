"""Proxy for the plain-English explainer service.

F1's FastAPI serves the site, so the browser cannot reach the explainer
directly: it calls ``/api/explain/{sport}/{id}`` here, and this route forwards
to the service's own ``/explain/{sport}/{id}``.

The proxy is deliberately thin and deliberately unforgiving. A summary is a
nice-to-have on top of a session page, so a missing, slow or broken explainer
becomes a 502 with a fixed message and the site's own error state takes it from
there -- the session itself never depends on this route answering.

**This module replaces a refusal that was checked and reversed.** The shared
explainer used to 404 `f1` ("a win probability IS the explanation"), which was
true of the win probability and never an argument against the race story around
it. F1 serves now: its `/facts` carries the `win`, `podium`, `points` and `dnf`
the panel draws. See `tests/test_explain_proxy.py`, which asserts the safety
properties an absence never could.
"""
from __future__ import annotations

import os
import sys
from urllib.parse import quote

import requests
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api")

#: Ceiling on the accepted timeout, so a fat-fingered `600` cannot hold a worker
#: thread for ten minutes. Well above the browser's own 15 s read timeout, which
#: is the point at which the client has given up anyway.
_MAX_TIMEOUT_S = 60.0
_DEFAULT_TIMEOUT_S = 15.0


def _positive_float(raw: str, name: str) -> float:
    """Parse a duration that must be a positive number, or fall back to the default.

    `float(os.getenv(...))` at import raises on a typo, and `main.py` imports
    this module -- so `EXPLAINER_TIMEOUT_S=` (set but empty) or `=abc` would
    take the WHOLE API down, not just this route. And `=0` parses fine, then
    makes `requests` raise a bare `ValueError`, which is not a
    `RequestException` and escapes the handler below as a 500: a fourth failure
    shape on a route that promises the site has exactly one.

    Falls back rather than raising, because a bad env var should cost one
    panel, not the whole site. The operator gets a line on stderr at boot.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        print(f"warning: {name}={raw!r} is not a number; using the default", file=sys.stderr)
        return _DEFAULT_TIMEOUT_S
    if not 0 < value <= _MAX_TIMEOUT_S:
        print(f"warning: {name}={raw!r} is outside (0, {_MAX_TIMEOUT_S}]; "
              f"using the default", file=sys.stderr)
        return _DEFAULT_TIMEOUT_S
    return value


# The explainer is reached over the internal compose network by service name.
EXPLAINER_URL = os.getenv("EXPLAINER_URL", "http://predictor-explainer:8090").rstrip("/")
EXPLAINER_TIMEOUT_S = _positive_float(os.getenv("EXPLAINER_TIMEOUT_S", "15"), "EXPLAINER_TIMEOUT_S")

#: Every failure is this message. An upstream error can carry key material or an
#: internal path, and the browser is not the place to find out.
_UNAVAILABLE = "The summary service is not available."


@router.get("/explain/{sport}/{explainer_id:path}")
def explain(sport: str, explainer_id: str):
    """Forward to the explainer, or say plainly that it could not be reached.

    No upstream body is ever returned, on any status: a 2xx is the only thing
    passed through. Redirects are not followed, and neither path segment is
    allowed to walk out of the explainer's own route.
    """
    # Refused BEFORE any request goes out: `..` anywhere in either segment
    # (interior included -- a prefix check lets `secrets/../x` through), a `/`
    # in the sport, or an id that is empty, absolute, or multi-segment. `quote()`
    # would otherwise put `/etc/passwd` on the wire as `%2Fetc%2Fpasswd` and the
    # explainer's ASGI layer would decode it back, landing on a deeper path than
    # the caller named. An id is one relative segment; anything else is refused,
    # as a 502 like every other failure so the site has one shape to handle.
    if ".." in sport or ".." in explainer_id or "/" in sport or not explainer_id or explainer_id.startswith("/"):
        raise HTTPException(status_code=502, detail=_UNAVAILABLE)
    url = f"{EXPLAINER_URL}/explain/{quote(sport, safe='')}/{quote(explainer_id, safe='')}"
    try:
        response = requests.get(url, timeout=EXPLAINER_TIMEOUT_S, allow_redirects=False)
        if not (200 <= response.status_code < 300):
            raise requests.HTTPError(f"upstream {response.status_code}")
        body = response.json()
        if not isinstance(body, dict):
            # Not annotated `-> dict` on purpose: the annotation makes FastAPI
            # infer `response_model=dict`, and a 2xx whose JSON is a list turns
            # into a bare 500 text/plain. Checked here it is the same 502 as
            # every other unusable answer.
            raise requests.HTTPError(f"upstream 2xx body is a {type(body).__name__}")
        return body
    except requests.RequestException as exc:
        # Also covers "a 2xx that is not JSON": `JSONDecodeError` subclasses
        # `RequestException`, so a separate `except ValueError` would be dead.
        raise HTTPException(status_code=502, detail=_UNAVAILABLE) from exc
