import type {
  ChampionshipResponse,
  ExplainResponse,
  RaceAccuracyEntry,
  RacePredictionResponse,
  RaceSummary,
  RetrainResponse,
  SessionPredictionResponse,
  SessionType,
  TrackRecordResponse,
} from "../types";
import { createContextLoader, type Explanation, type Signal } from "../predictor-ui";

/** What `GET /api/signals/{session_id}` answers. `signals` is empty rather than
 *  absent when the session has no honest signal — see spec §2. */
export interface SignalsResponse {
  signals: Signal[];
  sport?: string;
  id?: string;
}

// Same-origin /api: the public Docker build serves frontend and backend
// together, and the dev server proxies /api to the local backend (see
// vite.config.ts). VITE_API_BASE_URL still overrides it.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

// A backend that accepts the connection but never answers must still end in
// the page's error state (with Try again), never an endless "Loading…".
export const REQUEST_TIMEOUT_MS = 15_000;

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE_URL}${path}`, { ...init, signal: controller.signal });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new ApiError(body.detail ?? `${res.status} ${res.statusText}`, res.status);
    }
    return res.json();
  } finally {
    clearTimeout(timer);
  }
}

const get = <T,>(path: string) => request<T>(path);
const post = <T,>(path: string) => request<T>(path, { method: "POST" });

export const api = {
  races: (season?: number) => get<RaceSummary[]>(season ? `/races?season=${season}` : "/races"),
  racePrediction: (season: number, round: number) =>
    get<RacePredictionResponse>(`/races/${season}/${round}/prediction`),
  sessionPrediction: (sessionType: Exclude<SessionType, "race">, season: number, round: number) => {
    const path = sessionType === "sprint_qualifying" ? "sprint-qualifying-prediction"
      : sessionType === "sprint" ? "sprint-prediction"
      : "qualifying-prediction";
    return get<SessionPredictionResponse>(`/races/${season}/${round}/${path}`);
  },
  explainPrediction: (season: number, round: number, driverId: string, sessionType: SessionType = "race") =>
    get<ExplainResponse>(
      `/races/${season}/${round}/explain?driver_id=${encodeURIComponent(driverId)}&session_type=${sessionType}`,
    ),
  /** The shared plain-English summary for a session, via this API's explainer
   *  proxy (`/api/explain/f1/{id}` -> the service's `/explain/f1/{id}`). The
   *  path below is relative to `BASE_URL`, which already ends in `/api`, so it
   *  must NOT repeat the prefix -- doing so composed `/api/api/explain/...`,
   *  which 404s in production. The id is built only from the season, the round
   *  and the tab's own session key --
   *  all caller-held values, never reader input -- so there is nothing to
   *  quote or refuse. A session the service has no facts for (sprint
   *  qualifying) answers 404 there, which the panel renders as its retry
   *  state, not as a missing panel. */
  explainSession: (season: number, round: number, session: SessionType) =>
    get<Explanation>(`/explain/f1/${season}-${round}-${session}`),
  /** The Matchup section's data: `<BASE_URL>/explain/f1/<session id>/context`,
   *  the same base `explainSession` uses. */
  loadContext: createContextLoader(`${BASE_URL}/explain/f1`),
  /** Spec §3's per-fixture signal payloads, rendered by the shared
   *  `SignalRows`. The id is built here from the same three values every other
   *  caller uses, so it cannot drift from the one `/explain` and `/facts` read;
   *  the path is relative to `BASE_URL`, which already ends in `/api`, exactly
   *  as `explainSession` above — repeating the prefix would compose
   *  `/api/api/signals/...`.
   *
   *  A session with nothing to say answers `{"signals": []}`, which is a
   *  complete answer and renders as no rows at all (spec §2: "No data, no
   *  row"). A malformed id is a 404, like `/facts/{session_id}`. */
  signals: (season: number, round: number, session: SessionType) =>
    get<SignalsResponse>(`/signals/${season}-${round}-${session}`),
  championship: (championship: "drivers" | "constructors", season?: number) =>
    get<ChampionshipResponse>(
      `/championship/${championship}${season ? `?season=${season}` : ""}`,
    ),
  trackRecord: (tier?: string, sessionType?: string) => {
    const params = new URLSearchParams();
    if (tier) params.set("tier", tier);
    if (sessionType) params.set("session_type", sessionType);
    const qs = params.toString();
    return get<TrackRecordResponse>(qs ? `/track-record?${qs}` : "/track-record");
  },
  raceAccuracy: (tier?: string, sessionType?: string) => {
    const params = new URLSearchParams();
    if (tier) params.set("tier", tier);
    if (sessionType) params.set("session_type", sessionType);
    const qs = params.toString();
    return get<RaceAccuracyEntry[]>(qs ? `/track-record/by-race?${qs}` : "/track-record/by-race");
  },
  retrain: () => post<RetrainResponse>("/retrain"),
};

export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}
