import type { RaceSummary } from "../types";
import { formatDate } from "../lib/format";

interface Props {
  race: RaceSummary;
  selected: boolean;
  onClick: () => void;
}

export function RaceRow({ race, selected, onClick }: Props) {
  return (
    <button
      onClick={onClick}
      className={`clip-corner flex w-full items-center justify-between gap-3 border-l-2 px-3 py-2.5 text-left transition ${
        selected
          ? "border-f1-red bg-f1-800/80"
          : "border-transparent hover:border-f1-600 hover:bg-f1-850/60"
      }`}
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[11px] tabular-nums text-f1-text-faint">R{race.round}</span>
          <span className="truncate text-sm font-medium text-f1-text">{race.race_name}</span>
          {race.is_sprint_weekend && (
            <span className="rounded bg-f1-700 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide text-f1-gold">
              Sprint
            </span>
          )}
        </div>
        <div className="truncate text-xs text-f1-text-faint">{race.circuit_name}</div>
      </div>
      <div className="shrink-0 text-right">
        <div className="text-xs text-f1-text-dim">{formatDate(race.race_datetime)}</div>
        <div
          className={`text-[10px] font-semibold uppercase tracking-wide ${
            race.completed ? "text-f1-text-faint" : "text-win"
          }`}
        >
          {race.completed ? "Completed" : "Upcoming"}
        </div>
      </div>
    </button>
  );
}
