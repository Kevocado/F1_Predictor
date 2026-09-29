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
import type { Explanation } from "../predictor-ui";

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
   *  proxy (`/api/explain/f1/{id}` -> the service's `/explain/f1/{id}`). The id
   *  is built only from the season, the round and the tab's own session key --
   *  all caller-held values, never reader input -- so there is nothing to
   *  quote or refuse. A session the service has no facts for (sprint
   *  qualifying) answers 404 there, which the panel renders as its retry
   *  state, not as a missing panel. */
  explainSession: (season: number, round: number, session: SessionType) =>
    get<Explanation>(`/api/explain/f1/${season}-${round}-${session}`),
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
