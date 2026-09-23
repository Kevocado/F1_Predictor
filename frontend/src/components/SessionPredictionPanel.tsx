// frontend/src/components/SessionPredictionPanel.tsx
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { SessionPredictionResponse, SessionType } from "../types";
import { TierBadge } from "./TierBadge";
import { SessionPredictionTable } from "./SessionPredictionTable";

interface Props {
  season: number;
  round: number;
  sessionType: Exclude<SessionType, "race">;
  title: string;
}

export function SessionPredictionPanel({ season, round, sessionType, title }: Props) {
  const [data, setData] = useState<SessionPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    api
      .sessionPrediction(sessionType, season, round)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [season, round, sessionType]);

  if (loading) {
    return <div className="animate-pulse text-sm text-f1-text-faint">Loading {title.toLowerCase()}…</div>;
  }
  if (error) {
    // A 404 here just means "not a sprint weekend" for sprint/sprint-qualifying,
    // and a 409 means no trained model exists yet for this session type (the
    // public deployment ships no trained session models — see the Dockerfile's
    // explicit model-file list) — render nothing rather than an error box for
    // either case. This is specific to these 3 session panels; the race panel's
    // error handling is unchanged.
    const lower = error.toLowerCase();
    if (error.includes("404") || lower.includes("not a sprint weekend") || lower.includes("no trained")) return null;
    return <div className="rounded-lg border border-dnf/30 bg-dnf/10 p-4 text-sm text-dnf">{error}</div>;
  }
  if (!data) return null;

  return (
    <div className="clip-corner-lg rounded-lg border border-f1-border bg-f1-850/60 p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-display text-base font-bold text-f1-text">{title}</h3>
        <TierBadge tier={data.tier} />
      </div>
      <SessionPredictionTable predictions={data.predictions} sessionType={data.session_type} />
    </div>
  );
}
