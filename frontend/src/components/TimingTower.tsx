import { useState } from "react";
import { api } from "../api/client";
import { pct, pctFine } from "../predictor-ui";
import { driverName, teamColor, teamName } from "../lib/teamColors";
import type { ExplainResponse, SessionDriverPrediction, SessionType } from "../types";
import { ExplainRibbon } from "./ExplainRibbon";

// F1's dialect of the family slate: a broadcast timing tower. One row per
// driver, ranked by the session's headline chance (win, or pole for
// qualifying), with the other markets small beside it.

type Key = keyof SessionDriverPrediction;
type Market = { key: Key; label: string };

const QUALI: Record<SessionType, boolean> = { sprint_qualifying: true, qualifying: true, sprint: false, race: false };
const HEADLINE: Record<"quali" | "race", Market> = {
  quali: { key: "p_pole", label: "Pole" },
  race: { key: "p_win", label: "Win" },
};
const SECONDARY: Record<"quali" | "race", Market[]> = {
  quali: [
    { key: "p_top_3", label: "Top 3" },
    { key: "p_top_10", label: "Top 10" },
  ],
  race: [
    { key: "p_podium", label: "Podium" },
    { key: "p_points_finish", label: "Points" },
    { key: "p_dnf", label: "DNF" },
  ],
};

const value = (p: SessionDriverPrediction, key: Key) => (p[key] as number | null) ?? 0;

function Result({ rank, p }: { rank: number; p: SessionDriverPrediction }) {
  if (p.actual_dnf) return <span className="font-semibold text-pr-loss">DNF</span>;
  if (p.actual_position == null) return null;
  const gain = rank - p.actual_position; // positive: finished ahead of our rank
  return (
    <span className="flex flex-col items-end leading-tight">
      <span className={`font-semibold ${p.actual_position <= 3 ? "text-pr-lean" : "text-pr-text"}`}>Finished P{p.actual_position}</span>
      {gain !== 0 && (
        <span className={`text-xs ${gain > 0 ? "text-pr-win" : "text-pr-loss"}`}>
          {Math.abs(gain)} {Math.abs(gain) === 1 ? "place" : "places"} {gain > 0 ? "better" : "worse"}
        </span>
      )}
    </span>
  );
}

interface Props {
  predictions: SessionDriverPrediction[];
  season: number;
  round: number;
  sessionType: SessionType;
}

export function TimingTower({ predictions, season, round, sessionType }: Props) {
  const kind = QUALI[sessionType] ? "quali" : "race";
  const headline = HEADLINE[kind];
  const secondary = SECONDARY[kind];
  const sorted = [...predictions].sort((a, b) => value(b, headline.key) - value(a, headline.key));
  const best = Math.max(...sorted.map((p) => value(p, headline.key)), 1e-9);
  const hasResults = sorted.some((p) => p.actual_position != null || p.actual_dnf);

  const [open, setOpen] = useState<string | null>(null);
  const [explain, setExplain] = useState<Record<string, ExplainResponse>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  // Each session type explains a different prediction for the same driver.
  const cacheKey = (driverId: string) => `${sessionType}:${driverId}`;

  const toggle = (driverId: string) => {
    if (open === driverId) return setOpen(null);
    setOpen(driverId);
    setError(null);
    if (explain[cacheKey(driverId)]) return;
    setLoading(driverId);
    api
      .explainPrediction(season, round, driverId, sessionType)
      .then((res) => setExplain((prev) => ({ ...prev, [cacheKey(driverId)]: res })))
      .catch(() => setError("We couldn't load what's driving this prediction. Try opening it again."))
      .finally(() => setLoading(null));
  };

  return (
    <div>
      <p className="mb-3 text-xs text-pr-text-dim">
        Ranked by {headline.label.toLowerCase()} chance. Each bar compares a driver with the best in this field. Open a driver to see what's driving it.
      </p>
      <div
        data-testid="tower-head"
        aria-hidden="true"
        className="grid grid-cols-[2.75rem_1fr_5.5rem] items-end gap-x-3 border-b border-pr-rule px-2 pb-2 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim sm:grid-cols-[2.75rem_1fr_6.5rem_14rem]"
      >
        <span>Pos</span>
        <span>Driver</span>
        <span className="text-right">{headline.label}</span>
        <span className="hidden sm:block">{secondary.map((m) => m.label).join(" · ")}</span>
      </div>
      <ol className="divide-y divide-pr-rule">
        {sorted.map((p, i) => {
          const rank = i + 1;
          const expanded = open === p.driver_id;
          const share = value(p, headline.key) / best;
          return (
            <li key={p.driver_id}>
              <button
                type="button"
                aria-expanded={expanded}
                onClick={() => toggle(p.driver_id)}
                className="grid w-full grid-cols-[2.75rem_1fr_5.5rem] items-center gap-x-3 gap-y-1 px-2 py-2.5 text-left transition-colors hover:bg-pr-panel-2 sm:grid-cols-[2.75rem_1fr_6.5rem_14rem]"
              >
                <span className="font-pr-display text-lg font-bold tabular-nums text-pr-text">P{rank}</span>
                <span className="flex min-w-0 items-center gap-2.5">
                  <span aria-hidden="true" className="h-8 w-1 shrink-0 rounded-full" style={{ background: teamColor(p.constructor_id) }} />
                  <span className="min-w-0">
                    <span className="block truncate font-semibold text-pr-text">{driverName(p.driver_id)}</span>
                    <span className="block truncate text-xs text-pr-text-dim">{teamName(p.constructor_id)}</span>
                  </span>
                </span>
                <span className="flex flex-col items-end gap-1">
                  <span data-testid="headline" className="font-pr-display text-2xl font-bold leading-none tabular-nums text-pr-text">
                    <span className="sr-only">{headline.label} </span>
                    {pctFine(value(p, headline.key))}
                  </span>
                  <span aria-hidden="true" className="h-1 w-full overflow-hidden rounded-full bg-pr-panel-2">
                    <span className="block h-full rounded-full bg-pr-accent" style={{ width: `${Math.max(2, share * 100)}%` }} />
                  </span>
                </span>
                <span className="col-span-3 flex flex-wrap items-center justify-between gap-x-3 gap-y-1 pl-[3.5rem] text-xs text-pr-text-dim sm:col-span-1 sm:pl-0">
                  <span>{secondary.map((m) => `${m.label} ${pct(value(p, m.key))}`).join(" · ")}</span>
                  {hasResults && <Result rank={rank} p={p} />}
                </span>
              </button>
              {expanded && (
                <div className="border-t border-pr-rule bg-pr-panel-2/40">
                  <ExplainRibbon
                    title={`What's driving ${driverName(p.driver_id)}'s chances`}
                    note={kind === "race" ? "Win, podium and points share one strength prediction; DNF risk comes from a separate reliability model." : "Pole, top 3 and top 10 share one strength prediction."}
                    contributors={explain[cacheKey(p.driver_id)]?.strength_contributors ?? null}
                    loading={loading === p.driver_id}
                    error={error}
                  />
                  {kind === "race" && (
                    <ExplainRibbon
                      title={`What's driving ${driverName(p.driver_id)}'s DNF risk`}
                      contributors={explain[cacheKey(p.driver_id)]?.dnf_contributors ?? null}
                      loading={loading === p.driver_id}
                      error={null}
                    />
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
