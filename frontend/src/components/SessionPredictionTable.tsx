// frontend/src/components/SessionPredictionTable.tsx
import type { SessionDriverPrediction, SessionType } from "../types";
import { driverName, teamColor } from "../lib/teamColors";
import { ProbabilityHeatCell } from "./ProbabilityHeatCell";
import { ResultDelta } from "./ResultDelta";
import { TeamBadge } from "./TeamBadge";

function columnMax(predictions: SessionDriverPrediction[], key: keyof SessionDriverPrediction): number {
  const values = predictions.map((p) => (p[key] as number) ?? 0);
  const max = Math.max(...values);
  return max > 0 ? max : 1;
}

interface Props {
  predictions: SessionDriverPrediction[];
  sessionType: SessionType;
}

const IS_QUALI_TYPE: Record<SessionType, boolean> = {
  sprint_qualifying: true,
  qualifying: true,
  sprint: false,
  race: false,
};

export function SessionPredictionTable({ predictions, sessionType }: Props) {
  const rankKey = IS_QUALI_TYPE[sessionType] ? "p_pole" : "p_win";
  const sorted = [...predictions].sort((a, b) => ((b[rankKey] as number) ?? 0) - ((a[rankKey] as number) ?? 0));
  const hasActuals = sorted.some((p) => p.actual_position != null || p.actual_dnf);

  const isQuali = IS_QUALI_TYPE[sessionType];
  const col1Label = isQuali ? "Pole" : "Win";
  const col2Label = isQuali ? "Top 3" : "Podium";
  const col3Label = isQuali ? "Top 10" : "Points";
  const col1Key: keyof SessionDriverPrediction = isQuali ? "p_pole" : "p_win";
  const col2Key: keyof SessionDriverPrediction = isQuali ? "p_top_3" : "p_podium";
  const col3Key: keyof SessionDriverPrediction = isQuali ? "p_top_10" : "p_points_finish";

  const max1 = columnMax(sorted, col1Key);
  const max2 = columnMax(sorted, col2Key);
  const max3 = columnMax(sorted, col3Key);
  const maxDnf = !isQuali ? columnMax(sorted, "p_dnf") : 1;

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-f1-border text-left text-[11px] uppercase tracking-wide text-f1-text-faint">
            <th className="py-2 pr-3 font-medium">#</th>
            <th className="py-2 pr-3 font-medium">Driver</th>
            <th className="py-2 pr-3 font-medium">{col1Label}</th>
            <th className="py-2 pr-3 font-medium">{col2Label}</th>
            <th className="py-2 pr-3 font-medium">{col3Label}</th>
            {!isQuali && <th className="py-2 pr-3 font-medium">DNF</th>}
            {hasActuals && <th className="py-2 pr-3 text-right font-medium">Predicted vs. actual</th>}
          </tr>
        </thead>
        <tbody>
          {sorted.map((p, i) => (
            <tr
              key={p.driver_id}
              className="border-b border-f1-border/60 last:border-0"
              style={{ borderLeft: `2px solid ${teamColor(p.constructor_id)}` }}
            >
              <td className="py-2 pl-2 pr-3 tabular-nums text-f1-text-faint">{i + 1}</td>
              <td className="py-2 pr-3">
                <div className="font-medium text-f1-text">{driverName(p.driver_id)}</div>
                <TeamBadge constructorId={p.constructor_id} />
              </td>
              <td className="py-1.5 pr-2">
                <ProbabilityHeatCell value={(p[col1Key] as number) ?? 0} intensity={((p[col1Key] as number) ?? 0) / max1} color="var(--color-f1-red)" digits={1} />
              </td>
              <td className="py-1.5 pr-2">
                <ProbabilityHeatCell value={(p[col2Key] as number) ?? 0} intensity={((p[col2Key] as number) ?? 0) / max2} color="var(--color-podium)" />
              </td>
              <td className="py-1.5 pr-2">
                <ProbabilityHeatCell value={(p[col3Key] as number) ?? 0} intensity={((p[col3Key] as number) ?? 0) / max3} color="var(--color-win)" />
              </td>
              {!isQuali && (
                <td className="py-1.5 pr-2">
                  <ProbabilityHeatCell value={p.p_dnf ?? 0} intensity={(p.p_dnf ?? 0) / maxDnf} color="var(--color-dnf)" />
                </td>
              )}
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
  );
}
