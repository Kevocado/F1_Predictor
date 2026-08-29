import type { RaceAccuracyEntry } from "../types";
import { driverName } from "../lib/teamColors";

function hitColor(hits: number, of: number): string {
  const frac = hits / of;
  if (frac >= 0.99) return "text-win";
  if (frac >= 0.5) return "text-podium";
  return "text-dnf";
}

export function RaceAccuracyTable({ entries }: { entries: RaceAccuracyEntry[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-f1-border text-left text-[11px] uppercase tracking-wide text-f1-text-faint">
            <th className="py-2 pr-3 font-medium">Race</th>
            <th className="py-2 pr-3 font-medium">Predicted winner</th>
            <th className="py-2 pr-3 font-medium">Actual winner</th>
            <th className="py-2 pr-3 font-medium text-right">Podium</th>
            <th className="py-2 pr-3 font-medium text-right">Points</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => {
            const winHit = e.win_hits >= e.win_of;
            return (
              <tr key={`${e.season}-${e.round}`} className="border-b border-f1-border/60 last:border-0 hover:bg-f1-800/40">
                <td className="py-2 pr-3 text-f1-text-dim">
                  <span className="mr-1.5 text-f1-text-faint">R{e.round}</span>
                  {e.race_name}
                </td>
                <td className="py-2 pr-3 font-medium text-f1-text">
                  {e.win_predicted.map(driverName).join(", ") || "—"}
                </td>
                <td className={`py-2 pr-3 font-medium ${winHit ? "text-win" : "text-dnf"}`}>
                  {winHit ? "✓ " : "✗ "}
                  {e.win_actual.map(driverName).join(", ") || "—"}
                </td>
                <td className={`py-2 pr-3 text-right tabular-nums font-semibold ${hitColor(e.podium_hits, e.podium_of)}`}>
                  {e.podium_hits}/{e.podium_of}
                </td>
                <td
                  className={`py-2 pr-3 text-right tabular-nums font-semibold ${hitColor(e.points_finish_hits, e.points_finish_of)}`}
                >
                  {e.points_finish_hits}/{e.points_finish_of}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {entries.length === 0 && <p className="py-6 text-center text-sm text-f1-text-faint">No resolved races yet.</p>}
    </div>
  );
}
