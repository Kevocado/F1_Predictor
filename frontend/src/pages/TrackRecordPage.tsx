import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { TrackRecordResponse } from "../types";
import { MARKET_LABELS, TIER_LABELS } from "../lib/glossary";
import { pct } from "../lib/format";

export function TrackRecordPage() {
  const [data, setData] = useState<TrackRecordResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.trackRecord().then(setData).catch((e) => setError(e.message));
  }, []);

  const grouped = data
    ? data.by_market.reduce<Record<string, typeof data.by_market>>((acc, row) => {
        (acc[row.tier] ??= []).push(row);
        return acc;
      }, {})
    : {};

  return (
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
  );
}
