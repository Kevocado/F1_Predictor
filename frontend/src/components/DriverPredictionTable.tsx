import type { DriverPrediction } from "../types";
import { driverName, teamColor } from "../lib/teamColors";
import { ProbabilityHeatCell } from "./ProbabilityHeatCell";
import { ResultDelta } from "./ResultDelta";
import { TeamBadge } from "./TeamBadge";
import { InfoTooltip } from "./InfoTooltip";

function columnMax(predictions: DriverPrediction[], key: keyof DriverPrediction): number {
  const max = Math.max(...predictions.map((p) => p[key] as number));
  return max > 0 ? max : 1;
}

export function DriverPredictionTable({ predictions }: { predictions: DriverPrediction[] }) {
  const sorted = [...predictions].sort((a, b) => b.p_win - a.p_win);
  const hasActuals = sorted.some((p) => p.actual_position != null || p.actual_dnf);

  const maxWin = columnMax(sorted, "p_win");
  const maxPodium = columnMax(sorted, "p_podium");
  const maxPoints = columnMax(sorted, "p_points_finish");
  const maxDnf = columnMax(sorted, "p_dnf");

  return (
    <div>
      <p className="mb-2.5 text-[11px] text-f1-text-faint">
        Shading shows each driver's chance relative to the rest of THIS grid, not an absolute scale — the
        darkest cell in a column is always the field's best shot at that market.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-f1-border text-left text-[11px] uppercase tracking-wide text-f1-text-faint">
              <th className="py-2 pr-3 font-medium">#</th>
              <th className="py-2 pr-3 font-medium">Driver</th>
              <th className="py-2 pr-3 font-medium">Win</th>
              <th className="py-2 pr-3 font-medium">Podium</th>
              <th className="py-2 pr-3 font-medium">Points</th>
              <th className="py-2 pr-3 font-medium">DNF</th>
              {hasActuals && (
                <th className="py-2 pr-3 text-right font-medium">
                  <span className="inline-flex items-center gap-1">
                    Predicted vs. actual
                    <InfoTooltip
                      text="Predicted rank (by win chance) compared to where the driver actually finished. ▲ green = beat the prediction, ▼ red = underperformed it."
                      align="right"
                    />
                  </span>
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => (
              <tr
                key={p.driver_id}
                className="border-b border-f1-border/60 last:border-0 hover:bg-f1-800/40"
                style={{ borderLeft: `2px solid ${teamColor(p.constructor_id)}` }}
              >
                <td className="py-2 pl-2 pr-3 tabular-nums text-f1-text-faint">{i + 1}</td>
                <td className="py-2 pr-3">
                  <div className="font-medium text-f1-text">{driverName(p.driver_id)}</div>
                  <TeamBadge constructorId={p.constructor_id} />
                </td>
                <td className="py-1.5 pr-2">
                  <ProbabilityHeatCell value={p.p_win} intensity={p.p_win / maxWin} color="var(--color-f1-red)" digits={1} />
                </td>
                <td className="py-1.5 pr-2">
                  <ProbabilityHeatCell value={p.p_podium} intensity={p.p_podium / maxPodium} color="var(--color-podium)" />
                </td>
                <td className="py-1.5 pr-2">
                  <ProbabilityHeatCell
                    value={p.p_points_finish}
                    intensity={p.p_points_finish / maxPoints}
                    color="var(--color-win)"
                  />
                </td>
                <td className="py-1.5 pr-2">
                  <ProbabilityHeatCell value={p.p_dnf} intensity={p.p_dnf / maxDnf} color="var(--color-dnf)" />
                </td>
                {hasActuals && (
                  <td className="py-2 pr-3">
                    <ResultDelta predictedRank={i + 1} actualPosition={p.actual_position} actualDnf={p.actual_dnf} />
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
