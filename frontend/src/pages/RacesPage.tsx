import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { RaceSummary } from "../types";
import { ErrorState, Skeleton, StatusBadge, kickoff } from "../predictor-ui";
import { SessionTimelinePanel } from "../components/SessionTimelinePanel";

const day = (race: RaceSummary) => (race.race_datetime ? kickoff(race.race_datetime).split(" · ")[0] : "Date to be confirmed");

export function RacesPage() {
  const [races, setRaces] = useState<RaceSummary[] | null>(null);
  const [error, setError] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const [selectedRound, setSelectedRound] = useState<number | null>(null);

  useEffect(() => {
    setError(false);
    api
      .races()
      .then((data) => {
        setRaces(data);
        // Open on the next race, or the latest completed one once the
        // season's over: what someone opening this page most likely wants.
        const nextUpcoming = data.find((r) => !r.completed);
        const fallback = [...data].reverse().find((r) => r.completed);
        setSelectedRound((nextUpcoming ?? fallback ?? data[0])?.round ?? null);
      })
      .catch(() => setError(true));
  }, [reloadKey]);

  const selected = useMemo(() => races?.find((r) => r.round === selectedRound) ?? null, [races, selectedRound]);
  const nextRound = races?.find((r) => !r.completed)?.round;
  const listRef = useRef<HTMLUListElement>(null);

  // Late in the season the next race sits far down the list: bring the
  // selected one into view (within the list only) when the season loads.
  useEffect(() => {
    const current = listRef.current?.querySelector<HTMLElement>("[aria-current]");
    current?.scrollIntoView?.({ block: "nearest" });
  }, [races]);

  if (error) {
    return <ErrorState message="We couldn't load the season. Check your connection and try again." onRetry={() => setReloadKey((k) => k + 1)} />;
  }
  if (!races) return <Skeleton label="Loading season…" />;

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[18rem_1fr] lg:gap-6">
      <label className="flex flex-col gap-1 text-sm text-pr-text-dim lg:hidden">
        Race
        <select
          value={selectedRound ?? ""}
          onChange={(e) => setSelectedRound(Number(e.target.value))}
          className="rounded-pr border border-pr-rule bg-pr-panel px-3 py-2 text-base text-pr-text"
        >
          {races.map((r) => (
            <option key={r.round} value={r.round}>
              R{r.round} · {r.race_name}
              {r.round === nextRound ? " (next up)" : ""}
            </option>
          ))}
        </select>
      </label>

      <ul ref={listRef} aria-label="Races" className="hidden max-h-[75vh] overflow-y-auto rounded-pr border border-pr-rule bg-pr-panel p-1 lg:block">
        {races.map((race) => {
          const current = race.round === selectedRound;
          return (
            <li key={race.round}>
              <button
                type="button"
                aria-current={current || undefined}
                onClick={() => setSelectedRound(race.round)}
                className={`flex w-full items-center justify-between gap-3 rounded-pr px-3 py-2.5 text-left transition-colors ${
                  current ? "bg-pr-panel-2 ring-1 ring-inset ring-pr-accent" : "hover:bg-pr-panel-2"
                }`}
              >
                <span className="min-w-0 flex-1">
                  <span className="flex items-baseline gap-2">
                    <span className="font-pr-display text-sm font-semibold tabular-nums text-pr-text-dim">R{race.round}</span>
                    <span className="text-sm font-semibold leading-snug text-pr-text">{race.race_name}</span>
                  </span>
                  <span className="flex flex-wrap items-center gap-x-2 text-xs text-pr-text-dim">
                    {day(race)}
                    {race.is_sprint_weekend && <span className="font-semibold uppercase tracking-wide text-pr-lean">Sprint</span>}
                    {race.completed && <span>· Completed</span>}
                  </span>
                </span>
                {race.round === nextRound && (
                  <span className="shrink-0">
                    <StatusBadge status="next" />
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>

      <div className="min-w-0">
        {selected && (
          <SessionTimelinePanel
            season={selected.season}
            round={selected.round}
            isSprintWeekend={selected.is_sprint_weekend}
            raceDatetime={selected.race_datetime}
          />
        )}
      </div>
    </div>
  );
}
