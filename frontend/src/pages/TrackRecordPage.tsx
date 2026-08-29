import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RaceAccuracyEntry, TrackRecordResponse } from "../types";
import { MARKET_LABELS, TIER_LABELS } from "../lib/glossary";
import { pct } from "../lib/format";
import { RaceAccuracyTable } from "../components/RaceAccuracyTable";

export function TrackRecordPage() {
  const [data, setData] = useState<TrackRecordResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [accuracy, setAccuracy] = useState<RaceAccuracyEntry[] | null>(null);
  const [accuracyError, setAccuracyError] = useState<string | null>(null);

  useEffect(() => {
    api.trackRecord().then(setData).catch((e) => setError(e.message));
    api
      .raceAccuracy("post_qualifying")
      .then(setAccuracy)
      .catch((e) => setAccuracyError(e.message));
  }, []);

  const grouped = data
    ? data.by_market.reduce<Record<string, typeof data.by_market>>((acc, row) => {
        (acc[row.tier] ??= []).push(row);
        return acc;
      }, {})
    : {};

  const seasonSummary = accuracy?.length
    ? {
        winHits: accuracy.reduce((sum, r) => sum + Math.min(r.win_hits, r.win_of), 0),
        winOf: accuracy.reduce((sum, r) => sum + r.win_of, 0),
        podiumHits: accuracy.reduce((sum, r) => sum + r.podium_hits, 0),
        podiumOf: accuracy.reduce((sum, r) => sum + r.podium_of, 0),
        pointsHits: accuracy.reduce((sum, r) => sum + r.points_finish_hits, 0),
        pointsOf: accuracy.reduce((sum, r) => sum + r.points_finish_of, 0),
      }
    : null;

  return (
    <div className="flex flex-col gap-5">
      <div className="clip-corner-lg rounded-lg border border-f1-border bg-f1-850/60 p-5">
        <div className="mb-4">
          <h2 className="font-display text-lg font-bold text-f1-text">Track Record</h2>
          <p className="max-w-2xl text-xs text-f1-text-faint">
            There's no odds market for Formula 1 to compare against, so this is the honest alternative: every
            prediction is snapshotted before the race happens and reconciled against what actually happened.
            Brier score (lower is better) and hit rate should improve tier by tier as more of the weekend's real
            data becomes available.
          </p>
        </div>

        {error && <div className="rounded-lg border border-dnf/30 bg-dnf/10 p-4 text-sm text-dnf">{error}</div>}
        {!error && !data && <div className="animate-pulse text-sm text-f1-text-faint">Loading track record…</div>}
        {data && data.n_resolved === 0 && (
          <p className="py-6 text-center text-sm text-f1-text-faint">
            No resolved predictions yet — snapshots accumulate as races complete.
          </p>
        )}

        {Object.entries(grouped).map(([tier, rows]) => (
          <div key={tier} className="mb-5 last:mb-0">
            <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-f1-text-faint">
              {TIER_LABELS[tier] ?? tier}
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[480px] border-collapse text-sm">
                <thead>
                  <tr className="border-b border-f1-border text-left text-[11px] uppercase tracking-wide text-f1-text-faint">
                    <th className="py-2 pr-3 font-medium">Market</th>
                    <th className="py-2 pr-3 font-medium text-right">N</th>
                    <th className="py-2 pr-3 font-medium text-right">Brier</th>
                    <th className="py-2 pr-3 font-medium text-right">Hit rate</th>
                    <th className="py-2 pr-3 font-medium text-right">Avg. predicted</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.market} className="border-b border-f1-border/60 last:border-0">
                      <td className="py-2 pr-3 text-f1-text">{MARKET_LABELS[row.market] ?? row.market}</td>
                      <td className="py-2 pr-3 text-right tabular-nums text-f1-text-dim">{row.n}</td>
                      <td className="py-2 pr-3 text-right tabular-nums text-f1-text-dim">{row.brier.toFixed(4)}</td>
                      <td className="py-2 pr-3 text-right tabular-nums text-f1-text-dim">{pct(row.hit_rate, 1)}</td>
                      <td className="py-2 pr-3 text-right tabular-nums text-f1-text-dim">
                        {pct(row.avg_predicted_prob, 1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>

      <div className="clip-corner-lg rounded-lg border border-f1-border bg-f1-850/60 p-5">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="font-display text-lg font-bold text-f1-text">Accuracy by Race</h2>
            <p className="max-w-2xl text-xs text-f1-text-faint">
              Did the post-qualifying prediction call the actual positions? Predicted winner vs. who actually won,
              and how many of the predicted top-3/top-10 landed there.
            </p>
          </div>
          {seasonSummary && (
            <div className="flex gap-4 text-right text-xs text-f1-text-dim">
              <div>
                <div className="font-display text-lg font-bold text-f1-text">
                  {seasonSummary.winHits}/{seasonSummary.winOf}
                </div>
                Winners called
              </div>
              <div>
                <div className="font-display text-lg font-bold text-f1-text">
                  {pct(seasonSummary.podiumHits / seasonSummary.podiumOf, 0)}
                </div>
                Podium hit rate
              </div>
              <div>
                <div className="font-display text-lg font-bold text-f1-text">
                  {pct(seasonSummary.pointsHits / seasonSummary.pointsOf, 0)}
                </div>
                Points hit rate
              </div>
            </div>
          )}
        </div>

        {accuracyError && (
          <div className="rounded-lg border border-dnf/30 bg-dnf/10 p-4 text-sm text-dnf">{accuracyError}</div>
        )}
        {!accuracyError && !accuracy && (
          <div className="animate-pulse text-sm text-f1-text-faint">Loading race-by-race accuracy…</div>
        )}
        {accuracy && <RaceAccuracyTable entries={accuracy} />}
      </div>
    </div>
  );
}
