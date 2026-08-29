import type {
  ChampionshipResponse,
  ExplainResponse,
  LiveCurrentResponse,
  RaceAccuracyEntry,
  RacePredictionResponse,
  RaceSummary,
  RetrainResponse,
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
  explainPrediction: (season: number, round: number, driverId: string) =>
    get<ExplainResponse>(`/races/${season}/${round}/explain?driver_id=${encodeURIComponent(driverId)}`),
  championship: (championship: "drivers" | "constructors", season?: number) =>
    get<ChampionshipResponse>(
      `/championship/${championship}${season ? `?season=${season}` : ""}`,
    ),
  trackRecord: (tier?: string) => get<TrackRecordResponse>(tier ? `/track-record?tier=${tier}` : "/track-record"),
  raceAccuracy: (tier?: string) =>
    get<RaceAccuracyEntry[]>(tier ? `/track-record/by-race?tier=${tier}` : "/track-record/by-race"),
  liveCurrent: () => get<LiveCurrentResponse>("/live/current"),
  retrain: () => post<RetrainResponse>("/retrain"),
};

export class ApiError extends Error {}
