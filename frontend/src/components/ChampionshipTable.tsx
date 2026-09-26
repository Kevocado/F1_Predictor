import type { ReactNode } from "react";
import type { ChampionshipEntry } from "../types";
import { stat } from "../predictor-ui";
import { ProbabilityHeatCell } from "./ProbabilityHeatCell";

interface Props {
  entries: ChampionshipEntry[];
  renderEntity: (entityId: string) => ReactNode;
}

function columnMax(entries: ChampionshipEntry[], key: keyof ChampionshipEntry): number {
  const max = Math.max(...entries.map((e) => e[key] as number));
  return max > 0 ? max : 1;
}

export function ChampionshipTable({ entries, renderEntity }: Props) {
  const sorted = [...entries].sort((a, b) => a.expected_final_rank - b.expected_final_rank);
  const maxWin = columnMax(sorted, "win_prob");
  const maxTop3 = columnMax(sorted, "top3_prob");

  if (sorted.length === 0) return <p className="py-6 text-sm text-pr-text-dim">No projection yet. It appears once the season's first race is complete.</p>;

  return (
    <div>
      <p className="mb-3 text-xs text-pr-text-dim">Shading compares each chance with the best in the field.</p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[34rem] border-collapse text-sm">
          <thead>
            <tr className="border-b border-pr-rule text-left font-pr-display text-xs uppercase tracking-wide text-pr-text-dim">
              <th className="py-2 pr-3 font-semibold">Pos</th>
              <th className="py-2 pr-3 font-semibold">Name</th>
              <th className="py-2 pr-3 font-semibold">Title</th>
              <th className="py-2 pr-3 font-semibold">Top 3</th>
              <th className="py-2 pr-3 text-right font-semibold">Projected points</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((e, i) => (
              <tr key={e.entity_id} className="border-b border-pr-rule last:border-0">
                <td className="py-2.5 pr-3 font-pr-display text-base font-bold tabular-nums text-pr-text">P{i + 1}</td>
                <td className="py-2.5 pr-3 font-semibold text-pr-text">{renderEntity(e.entity_id)}</td>
                <td className="py-1.5 pr-2">
                  <ProbabilityHeatCell value={e.win_prob} intensity={e.win_prob / maxWin} />
                </td>
                <td className="py-1.5 pr-2">
                  <ProbabilityHeatCell value={e.top3_prob} intensity={e.top3_prob / maxTop3} />
                </td>
                <td className="py-2.5 pr-3 text-right tabular-nums">
                  <span className="text-pr-text">{Math.round(e.expected_final_points)}</span>
                  <span className="block text-xs text-pr-text-dim">Projected rank {stat(e.expected_final_rank)}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
