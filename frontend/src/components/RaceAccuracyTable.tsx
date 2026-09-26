import type { RaceAccuracyEntry } from "../types";
import { StatusBadge } from "../predictor-ui";
import { driverName } from "../lib/teamColors";

function hitColor(hits: number, of: number): string {
  const frac = hits / of;
  if (frac >= 0.99) return "text-pr-win";
  if (frac >= 0.5) return "text-pr-lean";
  return "text-pr-loss";
}

const names = (ids: string[]) => ids.map(driverName).join(", ") || "—";

export function RaceAccuracyTable({ entries }: { entries: RaceAccuracyEntry[] }) {
  if (entries.length === 0) return <p className="py-6 text-sm text-pr-text-dim">No resolved races yet.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[40rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-pr-rule text-left font-pr-display text-xs uppercase tracking-wide text-pr-text-dim">
            <th className="py-2 pr-3 font-semibold">Race</th>
            <th className="py-2 pr-3 font-semibold">Predicted winner</th>
            <th className="py-2 pr-3 font-semibold">Actual winner</th>
            <th className="py-2 pr-3 text-right font-semibold">Podium</th>
            <th className="py-2 pr-3 text-right font-semibold">Points</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => {
            const winHit = e.win_hits >= e.win_of;
            // A rebuilt session is shown for reference and never judged.
            const judged = !e.rebuilt;
            return (
              <tr key={`${e.season}-${e.round}`} data-testid="race-row" className="border-b border-pr-rule last:border-0">
                <td className="py-2 pr-3 text-pr-text">
                  <span className="mr-1.5 font-pr-display font-semibold text-pr-text-dim">R{e.round}</span>
                  {e.race_name}
                  {e.rebuilt && (
                    <span className="mt-1 block">
                      <StatusBadge status="rebuilt" moment="the session" />
                    </span>
                  )}
                </td>
                <td className="py-2 pr-3 font-semibold text-pr-text">{names(e.win_predicted)}</td>
                <td className={`py-2 pr-3 font-semibold ${judged ? (winHit ? "text-pr-win" : "text-pr-loss") : "text-pr-text"}`}>
                  {judged && (winHit ? "✓ " : "✗ ")}
                  {names(e.win_actual)}
                </td>
                <td className={`py-2 pr-3 text-right font-semibold tabular-nums ${judged ? hitColor(e.podium_hits, e.podium_of) : "text-pr-text-dim"}`}>
                  {e.podium_hits}/{e.podium_of}
                </td>
                <td className={`py-2 pr-3 text-right font-semibold tabular-nums ${judged ? hitColor(e.points_finish_hits, e.points_finish_of) : "text-pr-text-dim"}`}>
                  {e.points_finish_hits}/{e.points_finish_of}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
