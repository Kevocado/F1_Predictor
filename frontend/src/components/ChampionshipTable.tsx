import type { ChampionshipEntry } from "../types";
import { num, pct } from "../lib/format";
import { ProbabilityBar } from "./ProbabilityBar";

interface Props {
  entries: ChampionshipEntry[];
  renderEntity: (entityId: string) => React.ReactNode;
}

export function ChampionshipTable({ entries, renderEntity }: Props) {
  const sorted = [...entries].sort((a, b) => a.expected_final_rank - b.expected_final_rank);

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-f1-border text-left text-[11px] uppercase tracking-wide text-f1-text-faint">
            <th className="py-2 pr-3 font-medium">Proj.</th>
            <th className="py-2 pr-3 font-medium"></th>
            <th className="py-2 pr-3 font-medium">Title win</th>
            <th className="py-2 pr-3 font-medium">Top 3</th>
            <th className="py-2 pr-3 font-medium text-right">Exp. points</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((e, i) => (
            <tr key={e.entity_id} className="border-b border-f1-border/60 last:border-0 hover:bg-f1-800/40">
              <td className="py-2.5 pr-3 tabular-nums text-f1-text-faint">{i + 1}</td>
              <td className="py-2.5 pr-3 font-medium text-f1-text">{renderEntity(e.entity_id)}</td>
              <td className="py-2.5 pr-3">
                <ProbabilityBar value={e.win_prob} color="var(--color-f1-red)" digits={1} />
              </td>
              <td className="py-2.5 pr-3">
                <ProbabilityBar value={e.top3_prob} color="var(--color-podium)" />
              </td>
              <td className="py-2.5 pr-3 text-right tabular-nums text-f1-text-dim">
                {num(e.expected_final_points, 0)}
                <span className="ml-2 text-[10px] text-f1-text-faint">rank {num(e.expected_final_rank, 1)}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {sorted.length === 0 && <p className="py-6 text-center text-sm text-f1-text-faint">No data yet.</p>}
      <p className="mt-3 text-[11px] text-f1-text-faint">
        Title win probability: {pct(sorted[0]?.win_prob ?? 0, 1)} for the current leader, from a Monte Carlo
        simulation of the remaining season.
      </p>
    </div>
  );
}
