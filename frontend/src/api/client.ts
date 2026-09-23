import type {
  ChampionshipResponse,
  ExplainResponse,
  LiveCurrentResponse,
  RaceAccuracyEntry,
  RacePredictionResponse,
  RaceSummary,
  RetrainResponse,
  SessionPredictionResponse,
  SessionType,
  TrackRecordResponse,
} from "../types";

// Derived from wherever this page was loaded from, not hardcoded to
// "localhost" — that would resolve to the *viewing device*, not the
// machine actually running the backend. The public Docker build sets
// VITE_API_BASE_URL=/api (frontend + backend share one origin there);
// local dev leaves it unset and falls back to the separate dev-server port.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? `${window.location.protocol}//${window.location.hostname}:8000/api`;

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function post<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, { method: "POST" });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

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
  explainPrediction: (season: number, round: number, driverId: string) =>
    get<ExplainResponse>(`/races/${season}/${round}/explain?driver_id=${encodeURIComponent(driverId)}`),
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
  liveCurrent: () => get<LiveCurrentResponse>("/live/current"),
  retrain: () => post<RetrainResponse>("/retrain"),
};

export class ApiError extends Error {}
