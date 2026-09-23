import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DriverPrediction, SessionDriverPrediction, SessionType } from "../types";
import { SOURCE_DESCRIPTIONS } from "../lib/glossary";
import { TierBadge } from "./TierBadge";
import { InfoTooltip } from "./InfoTooltip";
import { PredictionTable } from "./PredictionTable";

const ALL_SESSIONS: { key: SessionType; label: string }[] = [
  { key: "sprint_qualifying", label: "Sprint Quali" },
  { key: "sprint", label: "Sprint" },
  { key: "qualifying", label: "Qualifying" },
  { key: "race", label: "Race" },
];

function driverPredictionToSession(d: DriverPrediction): SessionDriverPrediction {
  return {
    driver_id: d.driver_id,
    constructor_id: d.constructor_id,
    p_pole: null,
    p_top_3: null,
    p_top_10: null,
    p_win: d.p_win,
    p_podium: d.p_podium,
    p_points_finish: d.p_points_finish,
    p_dnf: d.p_dnf,
    expected_position: d.expected_position,
    actual_position: d.actual_position,
    actual_dnf: d.actual_dnf,
  };
}

interface SessionData {
  race_name: string;
  tier: string;
  source: string;
  predictions: SessionDriverPrediction[];
}

interface Props {
  season: number;
  round: number;
  isSprintWeekend: boolean;
}

export function SessionTimelinePanel({ season, round, isSprintWeekend }: Props) {
  const sessions = isSprintWeekend ? ALL_SESSIONS : ALL_SESSIONS.filter((s) => s.key === "qualifying" || s.key === "race");

  const [selected, setSelected] = useState<SessionType>("race");
  const [data, setData] = useState<SessionData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // A weekend switch (e.g. sprint -> non-sprint) can leave `selected`
  // pointing at a session this race doesn't have -- fall back to Race
  // rather than showing a 404 for a tab that shouldn't even be selectable.
  useEffect(() => {
    if (!sessions.some((s) => s.key === selected)) {
      setSelected("race");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [round, isSprintWeekend]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);

    const request =
      selected === "race"
        ? api.racePrediction(season, round).then((r) => ({
            race_name: r.race_name,
            tier: r.tier,
            source: r.source,
            predictions: r.predictions.map(driverPredictionToSession),
          }))
        : api
            .sessionPrediction(selected, season, round)
            .then((r) => ({ race_name: r.race_name, tier: r.tier, source: r.source, predictions: r.predictions }));

    request
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));

    return () => {
      cancelled = true;
    };
  }, [season, round, selected]);

  return (
    <div className="clip-corner-lg rounded-lg border border-f1-border bg-f1-850/60 p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-display text-lg font-bold text-f1-text">{data?.race_name ?? "…"}</h2>
          <p className="text-xs text-f1-text-faint">
            Round {round} · {season}
          </p>
        </div>
        {data && (
          <div className="flex items-center gap-2">
            <TierBadge tier={data.tier} />
            <span className="inline-flex items-center gap-1.5 rounded-full border border-f1-border bg-f1-800 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-f1-text-dim">
              {data.source}
              <InfoTooltip text={SOURCE_DESCRIPTIONS[data.source] ?? data.source} align="right" />
            </span>
          </div>
        )}
      </div>

      <div className="mb-4 flex gap-1 rounded-lg border border-f1-border bg-f1-900/60 p-1">
        {sessions.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setSelected(key)}
            className={`rounded-md px-3 py-1.5 font-display text-xs font-semibold uppercase tracking-wide transition ${
              selected === key ? "bg-f1-red text-white" : "text-f1-text-dim hover:text-f1-text"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {loading && <div className="animate-pulse text-sm text-f1-text-faint">Loading prediction…</div>}

      {error &&
        (error.includes("404") || error.toLowerCase().includes("not a sprint weekend") ? (
          <p className="py-10 text-center text-sm text-f1-text-faint">Not available for this weekend.</p>
        ) : (
          <div className="rounded-lg border border-dnf/30 bg-dnf/10 p-4 text-sm text-dnf">{error}</div>
        ))}

      {data && !loading && !error && (
        <PredictionTable predictions={data.predictions} season={season} round={round} sessionType={selected} />
      )}
    </div>
  );
}
