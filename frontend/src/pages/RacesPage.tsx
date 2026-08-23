import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { RaceSummary } from "../types";
import { RaceRow } from "../components/RaceRow";
import { RacePredictionPanel } from "../components/RacePredictionPanel";

export function RacesPage() {
  const [races, setRaces] = useState<RaceSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedRound, setSelectedRound] = useState<number | null>(null);

  useEffect(() => {
    api
      .races()
      .then((data) => {
        setRaces(data);
        // Default to the next upcoming race, or the most recent completed
        // one if the season's over — whichever is most likely what
        // someone opening this page actually wants to see first.
        const nextUpcoming = data.find((r) => !r.completed);
        const fallback = [...data].reverse().find((r) => r.completed);
        setSelectedRound((nextUpcoming ?? fallback ?? data[0])?.round ?? null);
      })
      .catch((e) => setError(e.message));
  }, []);

  const selected = useMemo(() => races?.find((r) => r.round === selectedRound) ?? null, [races, selectedRound]);

  if (error) {
    return <div className="rounded-lg border border-dnf/30 bg-dnf/10 p-4 text-sm text-dnf">{error}</div>;
  }
  if (!races) {
    return <div className="animate-pulse text-sm text-f1-text-faint">Loading season…</div>;
  }

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[340px_1fr]">
      <div className="clip-corner-lg max-h-[75vh] overflow-y-auto rounded-lg border border-f1-border bg-f1-850/40 p-1.5">
        {races.map((race) => (
          <RaceRow
            key={race.round}
            race={race}
            selected={race.round === selectedRound}
            onClick={() => setSelectedRound(race.round)}
          />
        ))}
      </div>
      <div>
        {selected ? (
          <RacePredictionPanel season={selected.season} round={selected.round} />
        ) : (
          <div className="text-sm text-f1-text-faint">Select a race.</div>
        )}
      </div>
    </div>
  );
}
