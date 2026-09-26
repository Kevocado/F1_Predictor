import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RaceAccuracyEntry, TrackRecordResponse } from "../types";
import { MARKET_LABELS, TIER_LABELS } from "../lib/glossary";
import { ErrorState, Skeleton, StatTile, pct, pctFine, record } from "../predictor-ui";
import { RaceAccuracyTable } from "../components/RaceAccuracyTable";

const plural = (n: number, one: string, many: string) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

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
  const rebuilt = data.n_rebuilt_sessions ?? 0;

  // Summary tiles judge only sessions snapshotted before they ran.
  const counted = accuracy.filter((r) => !r.rebuilt);
  const sum = (f: (r: RaceAccuracyEntry) => number) => counted.reduce((n, r) => n + f(r), 0);
  const podiumOf = sum((r) => r.podium_of);
  const pointsOf = sum((r) => r.points_finish_of);

  return (
    <div className="flex flex-col gap-5">
      <section className="rounded-pr border border-pr-rule bg-pr-panel p-4 sm:p-5">
        <h2 className="font-pr-display text-2xl font-bold uppercase tracking-wide text-pr-text">Accuracy by race</h2>
        <p className="mb-4 max-w-2xl text-sm text-pr-text-dim">
          Only predictions snapshotted before the session count.
          {rebuilt > 0 && ` ${plural(rebuilt, "session was", "sessions were")} rebuilt after they ran and are left out.`}
          {" "}Each race compares the post-qualifying prediction with what happened: the predicted winner, and how many of the predicted top 3 and top 10 landed there.
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
          There are no betting odds to compare F1 predictions against, so each snapshot is checked against the result. Brier score: lower is better, and 0 is perfect. Hit rate should track the average predicted chance.
        </p>
        {data.n_resolved === 0 && <p className="py-4 text-sm text-pr-text-dim">No resolved predictions yet. They appear as races snapshotted in time finish.</p>}
        {Object.entries(grouped).map(([tier, rows]) => (
          <div key={tier} className="mb-5 last:mb-0">
            <h3 className="mb-2 font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text-dim">{TIER_LABELS[tier] ?? tier}</h3>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[30rem] border-collapse text-sm">
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
          </div>
        ))}
      </section>
    </div>
  );
}
