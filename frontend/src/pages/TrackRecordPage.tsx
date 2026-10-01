import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RaceAccuracyEntry, TrackRecordResponse } from "../types";
import { MARKET_LABELS, TIER_LABELS } from "../lib/glossary";
import { ErrorState, Skeleton, StatTile, pct, pctFine, record } from "../predictor-ui";
import { RaceAccuracyTable } from "../components/RaceAccuracyTable";

const plural = (n: number, one: string, many: string) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

/** The per-market ledger, over whatever rows it is handed.
 *
 * One component for the headline and for the pre-session subset, so the two
 * cannot drift into printing the same column headings with different
 * arithmetic under them. */
function LedgerTable({ rows, testId }: { rows: TrackRecordResponse["by_market"]; testId: string }) {
  return (
    <div className="overflow-x-auto">
      <table data-testid={testId} className="w-full min-w-[30rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-pr-rule text-left font-pr-display text-xs uppercase tracking-wide text-pr-text-dim">
            <th className="py-2 pr-3 font-semibold">Market</th>
            <th className="py-2 pr-3 text-right font-semibold">Predictions</th>
            <th className="py-2 pr-3 text-right font-semibold">Brier</th>
            <th className="py-2 pr-3 text-right font-semibold">Hit rate</th>
            <th className="py-2 pr-3 text-right font-semibold">Avg. predicted</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.market} className="border-b border-pr-rule last:border-0">
              <td className="py-2 pr-3 text-pr-text">{MARKET_LABELS[row.market] ?? row.market}</td>
              <td className="py-2 pr-3 text-right tabular-nums text-pr-text-dim">{row.n.toLocaleString("en-US")}</td>
              <td className="py-2 pr-3 text-right tabular-nums text-pr-text-dim">{row.brier.toFixed(3)}</td>
              <td className="py-2 pr-3 text-right tabular-nums text-pr-text-dim">{pctFine(row.hit_rate)}</td>
              <td className="py-2 pr-3 text-right tabular-nums text-pr-text-dim">{pctFine(row.avg_predicted_prob)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TrackRecordPage() {
  const [data, setData] = useState<TrackRecordResponse | null>(null);
  const [accuracy, setAccuracy] = useState<RaceAccuracyEntry[] | null>(null);
  const [error, setError] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    setError(false);
    setData(null);
    setAccuracy(null);
    // session_type="race" keeps race predictions apart from the qualifying
    // and sprint snapshots stored alongside them.
    Promise.all([api.trackRecord(undefined, "race"), api.raceAccuracy("post_qualifying", "race")])
      .then(([record, byRace]) => {
        setData(record);
        setAccuracy(byRace);
      })
      .catch(() => setError(true));
  }, [reloadKey]);

  if (error) {
    return <ErrorState message="We couldn't load the track record. Check your connection and try again." onRetry={() => setReloadKey((k) => k + 1)} />;
  }
  if (!data || !accuracy) return <Skeleton label="Loading track record…" />;

  const grouped = data.by_market.reduce<Record<string, TrackRecordResponse["by_market"]>>((acc, row) => {
    (acc[row.tier] ??= []).push(row);
    return acc;
  }, {});
  // The headline counts every recorded pick, whenever it was made, so
  // `n_rebuilt_sessions` is no longer what was withheld -- it is how many
  // sessions had a pick recorded at or after their own start, and the
  // reconciliation against the pre-session subset below it.
  const afterStart = data.n_rebuilt_sessions ?? 0;
  const preSession = data.n_pre_session ?? data.pre_session?.n_resolved ?? 0;
  const postSessionPicks = data.n_post_session_picks ?? Math.max(0, data.n_resolved - preSession);
  // The same ledger over the pre-session subset, grouped the same way.
  const preGrouped = (data.pre_session?.by_market ?? []).reduce<Record<string, TrackRecordResponse["by_market"]>>(
    (acc, row) => {
      (acc[row.tier] ??= []).push(row);
      return acc;
    },
    {},
  );

  // The by-race table is a different view and keeps its own strict rule: it
  // judges a session only when its snapshot was written before it started.
  const counted = accuracy.filter((r) => !r.rebuilt);
  const sum = (f: (r: RaceAccuracyEntry) => number) => counted.reduce((n, r) => n + f(r), 0);
  const podiumOf = sum((r) => r.podium_of);
  const pointsOf = sum((r) => r.points_finish_of);

  return (
    <div className="flex flex-col gap-5">
      <section className="rounded-pr border border-pr-rule bg-pr-panel p-4 sm:p-5">
        <h2 className="font-pr-display text-2xl font-bold uppercase tracking-wide text-pr-text">Accuracy by race</h2>
        <p className="mb-4 max-w-2xl text-sm text-pr-text-dim">
          This table judges only sessions snapshotted before they ran.
          {afterStart > 0 &&
            ` ${plural(afterStart, "session has", "sessions have")} at least one pick recorded at or after the start — a re-run of the model, not a pre-session call.`}
          {" "}Each race compares the post-qualifying prediction with what happened: the predicted winner, and how many of the predicted top 3 and top 10 landed there. The full record, including the sessions this table skips, is the calibration ledger below.
        </p>

        {counted.length > 0 ? (
          <div className="mb-4 grid grid-cols-3 gap-3">
            <div data-testid="winners-called">
              <StatTile label="Winners called" value={record(sum((r) => Math.min(r.win_hits, r.win_of)), sum((r) => r.win_of))} />
            </div>
            <StatTile label="Podium places called" value={podiumOf ? pct(sum((r) => r.podium_hits) / podiumOf) : "—"} />
            <StatTile label="Points places called" value={pointsOf ? pct(sum((r) => r.points_finish_hits) / pointsOf) : "—"} />
          </div>
        ) : (
          accuracy.length > 0 && (
            <p className="mb-4 rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-2 text-sm text-pr-text">
              No races have a prediction snapshotted before they ran yet. The record starts with the next race.
            </p>
          )
        )}
        <RaceAccuracyTable entries={accuracy} />
      </section>

      <section className="rounded-pr border border-pr-rule bg-pr-panel p-4 sm:p-5">
        <h2 className="font-pr-display text-2xl font-bold uppercase tracking-wide text-pr-text">Calibration by market</h2>
        <p className="mb-4 max-w-2xl text-sm text-pr-text-dim">
          Every recorded pick, whenever it was made — one counted pick per session, driver and
          market, the first one recorded. A re-run of the model on a session that has already run
          does not replace that pick or count a second time.
          {postSessionPicks > 0 &&
            ` Of these ${data.n_resolved.toLocaleString("en-US")} picks, ${postSessionPicks.toLocaleString("en-US")} were recorded at or after their session's start, and ${preSession.toLocaleString("en-US")} before it.`}
        </p>
        <p data-testid="timing-note" className="mb-4 max-w-3xl rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-2 text-sm text-pr-text-dim">
          {postSessionPicks === 0
            ? "Every pick in this ledger was recorded before its session started, so the pre-session table below is the same number over the same picks."
            : `A pick recorded after the session started still counts, and is never presented as one made before it: every row the API sends carries both timestamps and the derived label. Read the pre-session table below for what the model would have said on the weekend.`}
        </p>
        {data.n_resolved === 0 && <p className="py-4 text-sm text-pr-text-dim">No resolved predictions yet. They appear as sessions are run.</p>}
        {Object.entries(grouped).map(([tier, rows]) => (
          <div key={tier} className="mb-5 last:mb-0">
            <h3 className="mb-2 font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text-dim">{TIER_LABELS[tier] ?? tier}</h3>
            <LedgerTable rows={rows} testId="ledger" />
          </div>
        ))}

        <div className="mt-6 border-t border-pr-rule pt-4">
          <h3 className="mb-2 font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text-dim">
            Made before the session started
          </h3>
          <p className="mb-3 max-w-2xl text-xs leading-relaxed text-pr-text-dim">
            The same ledger over the {preSession.toLocaleString("en-US")} picks whose own
            timestamps prove they were recorded before their session started. There are no betting
            odds to compare F1 predictions against, so each snapshot is checked against the
            result: Brier score, lower is better and 0 is perfect, and the hit rate should track
            the average predicted chance.
          </p>
          {preSession === 0
            ? "No pick has been recorded in time yet, so this figure is empty rather than zero. It fills in as sessions are snapshotted before they start."
            : Object.entries(preGrouped).map(([tier, rows]) => (
                <div key={`pre-${tier}`} className="mb-5 last:mb-0">
                  <h4 className="mb-2 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-faint">{TIER_LABELS[tier] ?? tier}</h4>
                  <LedgerTable rows={rows} testId="pre-session-ledger" />
                </div>
              ))}
        </div>
      </section>
    </div>
  );
}
