import { useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import type { DriverPrediction, SessionDriverPrediction, SessionType } from "../types";
import { EmptyState, ErrorState, Skeleton, StatusBadge, kickoff } from "../predictor-ui";
import { TierBadge } from "./TierBadge";
import { TimingTower } from "./TimingTower";

const ALL_SESSIONS: { key: SessionType; label: string }[] = [
  { key: "sprint_qualifying", label: "Sprint quali" },
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

// Where the prediction on screen came from, in words. Only a snapshot made
// before the session is ever counted; a rebuilt one (a late snapshot, or a
// no-lookahead backtest of a session never snapshotted) is labelled.
function Source({ source }: { source: string }) {
  if (source === "rebuilt" || source === "backtest") {
    return (
      <div className="flex flex-col items-end gap-1 text-right">
        <StatusBadge status="rebuilt" moment="the session" />
        <span className="max-w-xs text-xs text-pr-text-dim">Built after this session ran, so it isn't counted in the track record.</span>
      </div>
    );
  }
  if (source === "tracked") return <span className="text-xs font-semibold text-pr-win">Snapshot made before the session</span>;
  return <span className="text-xs text-pr-text-dim">Latest model, updated through the weekend</span>;
}

interface Props {
  season: number;
  round: number;
  isSprintWeekend: boolean;
  raceDatetime: string | null;
}

export function SessionTimelinePanel({ season, round, isSprintWeekend, raceDatetime }: Props) {
  const sessions = isSprintWeekend ? ALL_SESSIONS : ALL_SESSIONS.filter((s) => s.key === "qualifying" || s.key === "race");

  const [selected, setSelected] = useState<SessionType>("race");
  const [data, setData] = useState<SessionData | null>(null);
  const [error, setError] = useState<"missing" | "failed" | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

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
      .catch((e) => {
        if (cancelled) return;
        const missing = (e instanceof ApiError && e.status === 404) || String(e?.message).toLowerCase().includes("not a sprint weekend");
        setError(missing ? "missing" : "failed");
      });

    return () => {
      cancelled = true;
    };
  }, [season, round, selected, reloadKey]);

  return (
    <section className="pr-notch rounded-pr border border-pr-rule bg-pr-panel p-4 sm:p-5">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-pr-display text-2xl font-bold uppercase tracking-wide text-pr-text">{data?.race_name ?? "Loading race…"}</h2>
          <p className="text-sm text-pr-text-dim">
            Round {round}
            {raceDatetime ? ` · ${kickoff(raceDatetime)}` : ` · ${season}`}
          </p>
        </div>
        {data && (
          <div className="flex flex-wrap items-start justify-end gap-2">
            <TierBadge tier={data.tier} />
            <Source source={data.source} />
          </div>
        )}
      </div>

      <div role="group" aria-label="Session" className="mb-4 flex flex-wrap gap-1 rounded-pr border border-pr-rule bg-pr-stage p-1">
        {sessions.map(({ key, label }) => (
          <button
            key={key}
            type="button"
            aria-pressed={selected === key}
            onClick={() => setSelected(key)}
            className={`rounded-pr px-3 py-1.5 font-pr-display text-sm font-semibold uppercase tracking-wide transition-colors ${
              selected === key ? "bg-pr-accent text-pr-accent-ink" : "text-pr-text-dim hover:text-pr-text"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {!data && !error && <Skeleton label="Loading prediction…" />}
      {error === "missing" && <EmptyState message="Not available for this weekend." />}
      {error === "failed" && (
        <ErrorState message="We couldn't load this prediction. Check your connection and try again." onRetry={() => setReloadKey((k) => k + 1)} />
      )}
      {data && !error && <TimingTower predictions={data.predictions} season={season} round={round} sessionType={selected} />}
    </section>
  );
}
