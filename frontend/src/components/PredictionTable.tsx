import { Fragment, useState } from "react";
import type { SessionDriverPrediction, ExplainResponse, SessionType } from "../types";
import { driverName, teamColor } from "../lib/teamColors";
import { ProbabilityHeatCell } from "./ProbabilityHeatCell";
import { ResultDelta } from "./ResultDelta";
import { TeamBadge } from "./TeamBadge";
import { InfoTooltip } from "./InfoTooltip";
import { ExplainRibbon } from "./ExplainRibbon";
import { api } from "../api/client";

const IS_QUALI_TYPE: Record<SessionType, boolean> = {
  sprint_qualifying: true,
  qualifying: true,
  sprint: false,
  race: false,
};

const STRENGTH_LABEL: Record<SessionType, string> = {
  sprint_qualifying: "Pole / Top 3 / Top 10",
  qualifying: "Pole / Top 3 / Top 10",
  sprint: "Win / Podium / Points",
  race: "Win / Podium / Points",
};

function columnMax(predictions: SessionDriverPrediction[], key: keyof SessionDriverPrediction): number {
  const values = predictions.map((p) => (p[key] as number) ?? 0);
  const max = Math.max(...values);
  return max > 0 ? max : 1;
}

type Section = "strength" | "dnf";

interface Props {
  predictions: SessionDriverPrediction[];
  season: number;
  round: number;
  sessionType: SessionType;
}

export function PredictionTable({ predictions, season, round, sessionType }: Props) {
  const isQuali = IS_QUALI_TYPE[sessionType];
  const rankKey = isQuali ? "p_pole" : "p_win";
  const col1Key: keyof SessionDriverPrediction = isQuali ? "p_pole" : "p_win";
  const col2Key: keyof SessionDriverPrediction = isQuali ? "p_top_3" : "p_podium";
  const col3Key: keyof SessionDriverPrediction = isQuali ? "p_top_10" : "p_points_finish";
  const col1Label = isQuali ? "Pole" : "Win";
  const col2Label = isQuali ? "Top 3" : "Podium";
  const col3Label = isQuali ? "Top 10" : "Points";

  const sorted = [...predictions].sort((a, b) => ((b[rankKey] as number) ?? 0) - ((a[rankKey] as number) ?? 0));
  const hasActuals = sorted.some((p) => p.actual_position != null || p.actual_dnf);

  const max1 = columnMax(sorted, col1Key);
  const max2 = columnMax(sorted, col2Key);
  const max3 = columnMax(sorted, col3Key);
  const maxDnf = !isQuali ? columnMax(sorted, "p_dnf") : 1;

  const [expanded, setExpanded] = useState<{ driverId: string; section: Section } | null>(null);
  const [explainCache, setExplainCache] = useState<Record<string, ExplainResponse>>({});
  const [explainError, setExplainError] = useState<string | null>(null);
  const [loadingDriver, setLoadingDriver] = useState<string | null>(null);

  // Session-type switches reuse driver_id keys across different underlying
  // predictions -- a stale cache entry from the previously-selected session
  // would silently show the wrong explanation, so each session type gets
  // its own cache rather than sharing one across switches.
  const cacheKey = (driverId: string) => `${sessionType}:${driverId}`;

  const handleCellClick = (driverId: string, section: Section) => {
    if (expanded?.driverId === driverId && expanded.section === section) {
      setExpanded(null);
      return;
    }
    setExpanded({ driverId, section });
    setExplainError(null);
    const key = cacheKey(driverId);
    if (!explainCache[key]) {
      setLoadingDriver(driverId);
      api
        .explainPrediction(season, round, driverId, sessionType)
        .then((res) => setExplainCache((prev) => ({ ...prev, [key]: res })))
        .catch((e) => setExplainError(e.message))
        .finally(() => setLoadingDriver(null));
    }
  };

  const colCount = 5 + (!isQuali ? 1 : 0) + (hasActuals ? 1 : 0);

  return (
    <div>
      <p className="mb-2.5 text-[11px] text-f1-text-faint">
        Shading shows each driver's chance relative to the rest of THIS grid, not an absolute scale — the
        darkest cell in a column is always the field's best shot at that market. Click a probability cell to
        see what's driving it.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-f1-border text-left text-[11px] uppercase tracking-wide text-f1-text-faint">
              <th className="py-2 pr-3 font-medium">#</th>
              <th className="py-2 pr-3 font-medium">Driver</th>
              <th className="py-2 pr-3 font-medium">{col1Label}</th>
              <th className="py-2 pr-3 font-medium">{col2Label}</th>
              <th className="py-2 pr-3 font-medium">{col3Label}</th>
              {!isQuali && <th className="py-2 pr-3 font-medium">DNF</th>}
              {hasActuals && (
                <th className="py-2 pr-3 text-right font-medium">
                  <span className="inline-flex items-center gap-1">
                    Predicted vs. actual
                    <InfoTooltip
                      text="Predicted rank (by top market) compared to where the driver actually finished. ▲ green = beat the prediction, ▼ red = underperformed it."
                      align="right"
                    />
                  </span>
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => {
              const isExpanded = expanded?.driverId === p.driver_id;
              const explain = explainCache[cacheKey(p.driver_id)];
              return (
                <Fragment key={p.driver_id}>
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
                      <button
                        className="block w-full cursor-pointer rounded-md ring-f1-red/50 transition hover:ring-2"
                        onClick={() => handleCellClick(p.driver_id, "strength")}
                      >
                        <ProbabilityHeatCell
                          value={(p[col1Key] as number) ?? 0}
                          intensity={((p[col1Key] as number) ?? 0) / max1}
                          color="var(--color-f1-red)"
                          digits={1}
                        />
                      </button>
                    </td>
                    <td className="py-1.5 pr-2">
                      <button
                        className="block w-full cursor-pointer rounded-md ring-podium/50 transition hover:ring-2"
                        onClick={() => handleCellClick(p.driver_id, "strength")}
                      >
                        <ProbabilityHeatCell
                          value={(p[col2Key] as number) ?? 0}
                          intensity={((p[col2Key] as number) ?? 0) / max2}
                          color="var(--color-podium)"
                        />
                      </button>
                    </td>
                    <td className="py-1.5 pr-2">
                      <button
                        className="block w-full cursor-pointer rounded-md ring-win/50 transition hover:ring-2"
                        onClick={() => handleCellClick(p.driver_id, "strength")}
                      >
                        <ProbabilityHeatCell
                          value={(p[col3Key] as number) ?? 0}
                          intensity={((p[col3Key] as number) ?? 0) / max3}
                          color="var(--color-win)"
                        />
                      </button>
                    </td>
                    {!isQuali && (
                      <td className="py-1.5 pr-2">
                        <button
                          className="block w-full cursor-pointer rounded-md ring-dnf/50 transition hover:ring-2"
                          onClick={() => handleCellClick(p.driver_id, "dnf")}
                        >
                          <ProbabilityHeatCell value={p.p_dnf ?? 0} intensity={(p.p_dnf ?? 0) / maxDnf} color="var(--color-dnf)" />
                        </button>
                      </td>
                    )}
                    {hasActuals && (
                      <td className="py-2 pr-3">
                        <ResultDelta predictedRank={i + 1} actualPosition={p.actual_position} actualDnf={p.actual_dnf} />
                      </td>
                    )}
                  </tr>
                  {isExpanded && (
                    <tr className="border-b border-f1-border/60 bg-f1-900/70 last:border-0">
                      <td colSpan={colCount} className="p-0">
                        {expanded.section === "strength" ? (
                          <ExplainRibbon
                            title={`What's driving ${driverName(p.driver_id)}'s ${STRENGTH_LABEL[sessionType]} prediction`}
                            note={`${STRENGTH_LABEL[sessionType]} come from the same underlying strength prediction, so they share one explanation.`}
                            contributors={explain?.strength_contributors ?? null}
                            loading={loadingDriver === p.driver_id}
                            error={explainError}
                          />
                        ) : (
                          <ExplainRibbon
                            title={`What's driving ${driverName(p.driver_id)}'s DNF risk`}
                            note={`A separate model from ${STRENGTH_LABEL[sessionType]}, trained on reliability history.`}
                            contributors={explain?.dnf_contributors ?? null}
                            loading={loadingDriver === p.driver_id}
                            error={explainError}
                          />
                        )}
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
