import type {
  ChampionshipResponse,
  LiveCurrentResponse,
  RacePredictionResponse,
  RaceSummary,
  RetrainResponse,
  TrackRecordResponse,
} from "../types";

// Derived from wherever this page was loaded from, not hardcoded to
// "localhost" — that would resolve to the *viewing device*, not the
// machine actually running the backend.
const BASE_URL = `${window.location.protocol}//${window.location.hostname}:8000/api`;

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
  championship: (championship: "drivers" | "constructors", season?: number) =>
    get<ChampionshipResponse>(
      `/championship/${championship}${season ? `?season=${season}` : ""}`,
    ),
  trackRecord: (tier?: string) => get<TrackRecordResponse>(tier ? `/track-record?tier=${tier}` : "/track-record"),
  liveCurrent: () => get<LiveCurrentResponse>("/live/current"),
  retrain: () => post<RetrainResponse>("/retrain"),
};

export class ApiError extends Error {}
