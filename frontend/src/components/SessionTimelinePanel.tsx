import { useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api/client";
import type {
  DriverPrediction,
  HistoryCoverage,
  RaceAccuracyEntry,
  SessionDriverPrediction,
  SessionType,
} from "../types";
import { EmptyState, ErrorState, FixtureExplainer, SignalRows, Skeleton, kickoff } from "../predictor-ui";
import type { Signal } from "../predictor-ui";
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

/** Where the prediction on screen came from, in words — but only for the two
 *  sources the instant block does not already state.
 *
 *  A rebuilt or backtest snapshot used to print its own badge and its own
 *  sentence in this header, and the block prints the same disclosure a few
 *  lines below it: two `Rebuilt after the session` badges and the same "not
 *  counted" claim inside one panel, in two components that cannot see each
 *  other. The block owns the timing now (it reads the same `pick_timing` the
 *  flow bundle carries), so those two sources return nothing here rather than
 *  saying it twice. `tracked` and the fallback stay: the block states WHEN a
 *  pick was made, not WHICH tier's snapshot is on screen. */
function Source({ source }: { source: string }) {
  if (source === "rebuilt" || source === "backtest") return null;
  if (source === "tracked") return <span className="text-xs font-semibold text-pr-win">Snapshot made before the session</span>;
  return <span className="text-xs text-pr-text-dim">Latest model, updated through the weekend</span>;
}

/** The session record as the two numbers the block's strip is built from.
 *
 *  `rebuilt` is the entire reason this is a tally and not a `reduce`: a session
 *  whose only snapshot was written after it ran was never a pick made before
 *  the session, so it contributes to NEITHER sum. Keeping its `win_of` in the
 *  denominator while dropping its `win_hits` from the numerator would book the
 *  session as a miss — the one outcome the field exists to prevent, and the one
 *  the spec's honesty rule ("only a pick made before the start counts") rules
 *  out. The session is still shown on the track-record page; it is simply not
 *  evidence here. */
function tallyRecord(rows: RaceAccuracyEntry[]): { hits: number; settled: number } {
  let hits = 0;
  let settled = 0;
  for (const row of rows) {
    if (row.rebuilt) continue;
    hits += row.win_hits;
    settled += row.win_of;
  }
  return { hits, settled };
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
  // The session record the block's strip reads. `null` means "nothing to say
  // yet" — loading, or unreadable — and the strip is withheld rather than
  // rendered at 0/0, so it never flashes a record that does not exist.
  const [record, setRecord] = useState<{ hits: number; settled: number } | null>(null);

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

  // The record is about the tier and the session on screen, so it waits for the
  // session's own response to say which tier that is, and re-reads when the
  // reader switches. `setRecord(null)` first, deliberately: the strip disappears
  // rather than showing the previous session's tally for a frame.
  const tier = data?.tier ?? null;
  useEffect(() => {
    if (!tier) return;
    let cancelled = false;
    setRecord(null);
    api
      .raceAccuracy(tier, selected)
      .then((rows) => !cancelled && setRecord(tallyRecord(rows)))
      .catch(() => !cancelled && setRecord(null));
    return () => {
      cancelled = true;
    };
  }, [tier, selected]);

  // Spec §3's signal rows for this session. Fetched on mount rather than from
  // behind the explainer button, because §2 makes signals INSTANT — they are
  // computed from stored data and "the signals add none" of the AI's cost, so
  // gating them on a paid click would make a free computed row a paid one.
  //
  // This is the only caller of `GET /api/signals/{session_id}`. The adapter, the
  // endpoint and the shared component were all shipped in Phase 1 and the row
  // had never reached a page: the router was mounted at the root while the
  // frontend addresses the backend through `/api`, so nothing could have called
  // it. `null` means "not known yet, or unreadable"; the render below treats
  // both the same way and shows nothing, per §2's "No data, no row" — a failed
  // signal request must not become this panel's error state, because the panel
  // itself is fine.
  const [signals, setSignals] = useState<Signal[] | null>(null);
  useEffect(() => {
    let cancelled = false;
    api
      .signals(season, round, selected)
      .then((res) => !cancelled && setSignals(res.signals ?? []))
      .catch(() => !cancelled && setSignals(null));
    return () => {
      cancelled = true;
    };
  }, [season, round, selected]);

  // The per-market ledger request (`GET /track-record` ->
  // store.get_session_track_record, which groups by session_type, tier and
  // market and reports n / brier / hit_rate / avg_predicted_prob) is GONE with
  // the "Model's top calls" pop-out, and it is gone because nothing on this
  // panel reads it any more. It was read for one consumer: the ledger line each
  // pop-out row used to carry. Since #27 no row carried a ledger figure at all,
  // so the request had no reader left even while the pop-out was still on the
  // page — it was already dead weight, and removing the section made that
  // visible. F1's real duplicate was never the ledger, it was the probabilities:
  // the per-driver table below already shows every driver's p_win, p_podium and
  // p_points_finish, for all twenty drivers, and the pop-out repeated the same
  // numbers under market headings.
  //
  // NFL, PL and NBA keep their `PicksList`, and this is why the decision is
  // F1-only rather than a change to the shared component: none of them has an
  // equivalent table, so the list is the only place their numbers are shown.

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
          {/* The signal rows, above the explainer for the same reason the
              coverage note is: both are computed from stored data and cost
              nothing, so both belong before the one thing on this panel that
              waits for a reader and a model. `SignalRows` renders nothing at
              all for an empty list — the guard is here so the wrapper does not
              leave an empty margin behind. */}
          {signals && signals.length > 0 && (
            <div className="mb-4">
              <SignalRows signals={signals} />
            </div>
          )}
          {/* In plain English, above the timing tower: the one-line answer
              before the grid. The instant block states the timing, the verdict
              and the session record from the session's own predictions and the
              by-race accuracy endpoint, with no request; the AI summary sits
              behind the button and costs nothing until a reader asks. Reduced
              by what the facts carry: F1 has no market line, so no tile and no
              bar are passed (decision 8 — its insight stays per-race, in the
              tower and the table below). */}
          <div className="mb-4">
            <FixtureExplainer
              sport="f1"
              state={flowState}
              bundle={flowBundle}
              extras={{
                record: record ? { label: "Picks made before the session", ...record } : undefined,
                moment: "the session",
              }}
              request={() => api.explainSession(season, round, selected)}
            />
          </div>
          {/* No "Model's top calls" pop-out, by Kevin's decision, 2026-10-01:
              "the table says everything the f1 model needs." The timing tower
              below already lists every driver's p_win, p_podium, p_points_finish
              and p_dnf — all twenty of them — so the pop-out repeated numbers
              this panel had already printed. He asked for the whole section
              removed rather than trimmed, and this is that removal.
              SessionTimelinePanel.notopcalls.test.tsx holds it: no heading, no
              list, no ledger request, and every driver's figures still present.
              NFL, PL and NBA are untouched and keep theirs — they have no
              equivalent table, so the list is their only showing of the numbers. */}
          <TimingTower predictions={data.predictions} season={season} round={round} sessionType={selected} />
        </>
      )}
    </section>
  );
}
