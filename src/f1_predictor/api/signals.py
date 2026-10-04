"""signals.py — `GET /api/signals/{session_id}`, the fixture's signal payloads.

Spec `2026-10-01-fixture-signals-design.md` §3 (the contract) and §2 ("No data,
no row"). Phase 1 ships exactly one adapter, `trust`; the others arrive with
their phases, and a session with no honest signal gets an empty list rather than
a placeholder.

The shape of every payload is dictated by the shared component at
`frontend/src/predictor-ui/components/SignalRows.tsx`, which REFUSES several
things a payload could plausibly carry — a rate outside [0, 1], a `reliability_bar`
with no `figures.rate`, a headline that does not state the figure its bar draws
(`HeadlineFigureMismatchError`), a visual it cannot draw. Read that file before
adding a field here; `tests/test_trust_signal.py` asserts against the real
vendored copy so a change to it breaks that suite.

The id shape and the pre-session pick rule are NOT reimplemented here: they come
from `facts.session_pick`, so a signal can never quote a different probability
than the facts block on the same page.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from . import facts
from ..signals import trust

#: `/api`, like `routes.py`'s router, and this was a real defect rather than a
#: style preference. Every other endpoint the frontend calls is served under the
#: `/api` prefix (`routes.py` declares `APIRouter(prefix="/api")`), and the
#: frontend reaches the backend through exactly that prefix in BOTH environments:
#: `client.ts`'s `BASE_URL` is `/api` (baked in at build time by the Dockerfile's
#: `VITE_API_BASE_URL=/api`), and the dev server proxies only `'/api'`
#: (`vite.config.ts`). A router mounted at the root was therefore unreachable
#: from the page in development, and reachable in production only by bypassing
#: `BASE_URL` — so the one signal Phase 1 shipped had no caller. Nothing failed
#: and nothing looked broken: `tests/test_trust_signal.py` drove the app through
#: `TestClient`, which hits paths directly and so never exercised the prefix the
#: browser actually uses. Hence the assertion on `app.openapi()["paths"]` below,
#: which pins the prefixed path and would catch a router mounted at the root.
router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

#: Spec §2: "a fixed rule (not the model) ranks them by `strength` and keeps the
#: top 2-3". Phase 1 has one adapter, so this never truncates anything today; it
#: is here because the rule belongs to the endpoint and not to a component, per
#: `SignalRows`'s own "Rejected: ranking and capping here" section.
MAX_SIGNALS = 3


def signals_for_session(season: int, round_: int, session: str) -> list[dict]:
    """Every signal this session can honestly carry, strongest first.

    Each adapter is asked independently and may return nothing; a session with no
    signal returns `[]`, which the client renders as no rows at all (spec §2: no
    empty states, no filler). An adapter that raises is treated as "no signal"
    rather than being allowed to take down the endpoint — a signal is an
    enhancement on a fixture page, and the page itself must survive its absence.
    """
    ctx = facts.session_pick(season, round_, session)
    game_id = facts.session_id_of(season, round_, session)
    pick = ctx.get("pick") or {}
    found: list[dict] = []

    try:
        signal = trust.trust_signal(game_id, pick.get("prob"))
    except Exception:
        logger.exception("trust signal unavailable for %s", game_id)
        signal = None
    if signal is not None:
        found.append(signal)

    return sorted(found, key=lambda s: s["strength"], reverse=True)[:MAX_SIGNALS]


@router.get("/signals/{session_id}")
def get_signals(session_id: str) -> dict:
    """The signals for one fixture. `{"signals": []}` is a valid, complete answer.

    A malformed id is a 404, matching `/facts/{session_id}`: the id grammar is
    `facts.parse_session_id`'s and this router does not own a second one.
    """
    parsed = facts.parse_session_id(session_id)
    if parsed is None:
        raise HTTPException(status_code=404, detail=f"Unknown session id: {session_id}")
    season, round_, session = parsed

    try:
        signals = signals_for_session(season, round_, session)
    except HTTPException:
        raise
    except Exception:
        # The session itself is unreadable (no prediction and no stored record,
        # or the feature pipeline is down). `facts.session_pick` answers the
        # first of those with a 404, which passes through above; anything else
        # is a server fault this endpoint absorbs so a missing row cannot become
        # a broken fixture page.
        logger.exception("signals unavailable for %s", session_id)
        # The SAME shape as the success path, so a client reading `sport` or `id`
        # does not get a different object only when the backend is failing — the
        # one moment a client is least able to cope with a special case.
        return {"sport": "f1", "id": str(session_id), "signals": []}

    return {"sport": "f1", "id": str(session_id), "signals": signals}