import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RacePredictionResponse } from "../types";
import { SOURCE_DESCRIPTIONS } from "../lib/glossary";
import { TierBadge } from "./TierBadge";
import { InfoTooltip } from "./InfoTooltip";
import { DriverPredictionTable } from "./DriverPredictionTable";

export function RacePredictionPanel({ season, round }: { season: number; round: number }) {
  const [data, setData] = useState<RacePredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    api
      .racePrediction(season, round)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [season, round]);

  if (loading) {
    return <div className="animate-pulse text-sm text-f1-text-faint">Loading prediction…</div>;
  }
  if (error) {
    return <div className="rounded-lg border border-dnf/30 bg-dnf/10 p-4 text-sm text-dnf">{error}</div>;
  }
  if (!data) return null;

  return (
    <div className="clip-corner-lg rounded-lg border border-f1-border bg-f1-850/60 p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-display text-lg font-bold text-f1-text">{data.race_name}</h2>
          <p className="text-xs text-f1-text-faint">
            Round {data.round} · {data.season}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <TierBadge tier={data.tier} />
          <span className="inline-flex items-center gap-1.5 rounded-full border border-f1-border bg-f1-800 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-f1-text-dim">
            {data.source}
            <InfoTooltip text={SOURCE_DESCRIPTIONS[data.source] ?? data.source} align="right" />
          </span>
        </div>
      </div>
      <DriverPredictionTable predictions={data.predictions} />
    </div>
  );
}
