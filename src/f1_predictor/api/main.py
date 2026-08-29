"""main.py — FastAPI app entry point.

Run with:
    uvicorn f1_predictor.api.main:app --reload --host 0.0.0.0 --port 8000

`--host 0.0.0.0` matters for reaching this from another device (phone,
another machine on the LAN) — the default `127.0.0.1` only accepts
connections from the same machine.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import FRONTEND_DIST_DIR
from .routes import router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The live in-race poller (requirement 3) — checks OpenF1 for a live
    # Race session every ~18s and, if found, updates models/live_poller.py's
    # in-process cache /api/live/current serves. A no-op most of the time
    # (no live session most hours of most days); cheap enough to just always
    # run rather than starting/stopping it around race weekends. Runs
    # unconditionally, PUBLIC_MODE or not — it's free, keyless, and is the
    # whole point of the public deployment.
    from ..models import live_poller

    poller_task = asyncio.create_task(live_poller.run_poller())
    yield
    poller_task.cancel()


app = FastAPI(title="F1 Predictor API", lifespan=lifespan)

# Wide open on purpose: this server is only ever meant to be reached over a
# private network — never exposed to the public internet — so there's no
# real origin to restrict to. The public deployment is read-only with no
# admin surface reachable at all (see routes.py::_admin_only) — there's
# nothing left for a login to protect, so it's a plain public site with
# no guest password (same approach as PL_Predictor's api/main.py).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Only present in the public Docker deployment (see repo-root Dockerfile),
# which builds frontend/dist before starting the server — local dev never
# has this directory, and keeps using `npm run dev` + this app's plain JSON
# root below exactly as before.
if FRONTEND_DIST_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
else:

    @app.get("/")
    def root():
        return {"status": "ok", "docs": "/docs"}
