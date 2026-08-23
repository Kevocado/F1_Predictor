import { positionSuffix } from "../lib/format";

interface Props {
  /** 1-indexed rank in this table, i.e. "we thought this driver was the
   * Nth most likely to win" — always available (unlike the Monte Carlo
   * mean finishing position, which the tracking store doesn't persist),
   * so the arrow shows consistently whether the prediction was computed
   * live, replayed from a backtest, or served from a tracked snapshot. */
  predictedRank: number;
  actualPosition: number | null;
  actualDnf: boolean | null;
}

/** What we predicted vs. what actually happened, with an arrow for the
 * gap — "beat the prediction by 4 places" is a much faster read than
 * comparing two separate numbers. */
export function ResultDelta({ predictedRank, actualPosition, actualDnf }: Props) {
  if (actualDnf) {
    return (
      <div className="flex items-center justify-end gap-1.5 text-xs">
        <span className="text-f1-text-faint">Pred. P{predictedRank}</span>
        <span className="text-dnf">▼</span>
        <span className="font-semibold text-dnf">DNF</span>
      </div>
    );
  }

  if (actualPosition == null) return <span className="text-f1-text-faint">—</span>;

  const delta = predictedRank - actualPosition; // positive = finished better than predicted
  const color = delta > 0 ? "text-win" : delta < 0 ? "text-dnf" : "text-f1-text-faint";
  const arrow = delta > 0 ? "▲" : delta < 0 ? "▼" : "●";

  return (
    <div className="flex items-center justify-end gap-1.5 text-xs">
      <span className="text-f1-text-faint">Pred. P{predictedRank}</span>
      <span className="text-f1-text-faint">→</span>
      <span className={actualPosition <= 3 ? "font-semibold text-podium" : "text-f1-text-dim"}>
        {actualPosition}
        <sup>{positionSuffix(actualPosition)}</sup>
      </span>
      {delta !== 0 && (
        <span className={`font-semibold ${color}`}>
          {arrow}
          {Math.abs(delta)}
        </span>
      )}
    </div>
  );
}
