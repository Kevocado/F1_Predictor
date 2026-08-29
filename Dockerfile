# Public, password-gated F1 Predictor deployment — one image serving the
# built React frontend and the FastAPI backend from a single process/origin
# (see api/main.py's StaticFiles mount and api/auth.py's GuestAuthMiddleware).
# Ported directly from PL_Predictor's own Dockerfile/PUBLIC_MODE pattern.
# The private/full app (npm run dev + uvicorn --reload, no PUBLIC_MODE) is
# untouched by this file — it's only used for the public deployment.

FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Baked in at build time, not runtime — Vite inlines import.meta.env values
# into the built bundle (see frontend/src/api/client.ts's BASE_URL).
ENV VITE_API_BASE_URL=/api
RUN npm run build

FROM python:3.13-slim AS backend
WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
# Editable install, matching local dev exactly — NOT a cosmetic choice:
# config.py derives PROJECT_ROOT (and everything under it: MODELS_DIR,
# CACHE_DIR, FRONTEND_DIST_DIR) from `Path(__file__).resolve().parents[2]`.
# A non-editable install copies the package into site-packages, so that
# math would resolve to somewhere under site-packages instead of /app —
# silently breaking every path in the app (confirmed the hard way on
# PL_Predictor's own first deploy attempt). No requirements-lock.txt exists
# in this project yet (unlike PL_Predictor's), so this installs whatever
# pyproject.toml's unpinned deps resolve to at build time.
RUN pip install --no-cache-dir -e .

# Ships with real trained models immediately instead of needing a full
# historical fetch+train before the first prediction can be served — same
# reasoning as PL_Predictor's models/*.json, though here that requires
# force-adding past .gitignore's `models/*.json`/`models/*.db` (see the
# commit that added them for exactly this reason). optuna_studies.db is
# tuning-only tooling state, not a serving artifact — deliberately not
# shipped.
COPY models/manifest.json models/dnf_model.json models/live_win_model.json models/live_podium_model.json models/race_outcome_ranker.json ./models/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 8000
CMD ["sh", "-c", "uvicorn f1_predictor.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
