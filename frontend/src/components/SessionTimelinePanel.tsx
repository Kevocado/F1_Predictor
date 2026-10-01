import { useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api/client";
import type { DriverPrediction, HistoryCoverage, SessionDriverPrediction, SessionType } from "../types";
import { EmptyState, ErrorState, FixtureExplainer, Skeleton, StatusBadge, kickoff } from "../predictor-ui";
import { driverName } from "../lib/teamColors";
import { HistoryCoverageNote } from "./HistoryCoverageNote";
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
  /** Only the race response carries it: a session forecast is built from the
   *  current season alone, so there is no multi-season window to come up
   *  short. Null on a session tab is a real absence, not a hidden true. */
  history: HistoryCoverage | null;
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

  // The flow's facts: the session's own predictions, no request. The pick is
  // the highest win probability on the board; finished-ness and rightness come
  // from the actuals when the session has them, and a rebuilt snapshot says so
  // in the session's own words. F1 carries no market line, so the flow says
  // the pick and stops -- the reduced panel, by construction rather than by
  // configuration.
  const finite = (x: unknown): number | undefined =>
    typeof x === "number" && Number.isFinite(x) ? x : undefined;
  const flowBundle = useMemo(() => {
    if (!data) return null;
    let top: SessionDriverPrediction | undefined;
    for (const p of data.predictions) {
      const w = finite(p.p_win);
      if (w !== undefined && (!top || w > (finite(top.p_win) ?? -1))) top = p;
    }
    const wasRight =
      top && top.actual_position != null
        ? top.actual_position === 1
        : undefined;
    return {
      driver: top ? driverName(top.driver_id) : undefined,
      pick: top
        ? {
            label: driverName(top.driver_id),
            prob: finite(top.p_win),
            ...(typeof wasRight === "boolean" ? { was_right: wasRight } : {}),
          }
        : undefined,
      pick_timing: data.source === "rebuilt" || data.source === "backtest" ? "rebuilt" : undefined,
    };
  }, [data]);
  const flowState = data?.predictions.some((p) => p.actual_position != null) ? "finished" : "pre-game";

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
            history: r.history ?? null,
          }))
        : api
            .sessionPrediction(selected, season, round)
            .then((r) => ({ race_name: r.race_name, tier: r.tier, source: r.source, predictions: r.predictions, history: null }));

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
      {data && !error && (
        <>
          {/* Above the flow, not below it: when a season was skipped the whole
              forecast is built on less history than usual, and that qualifies
              every number on this panel — so it is said before the pick, not
              after. Stays silent when the window was complete, and when the
              response reports no history load at all. */}
          <HistoryCoverageNote history={data.history} />
          {/* In plain English, above the timing tower: the one-line answer
              before the grid. The flow renders from the session's own
              predictions with no request; the AI summary sits behind the
              button and costs nothing until a reader asks. Reduced by what
              the facts carry: F1 has no market line, so no line is named. */}
          <div className="mb-4">
            <FixtureExplainer
              sport="f1"
              state={flowState}
              bundle={flowBundle}
              request={() => api.explainSession(season, round, selected)}
            />
          </div>
          <TimingTower predictions={data.predictions} season={season} round={round} sessionType={selected} />
        </>
      )}
    </section>
  );
}
