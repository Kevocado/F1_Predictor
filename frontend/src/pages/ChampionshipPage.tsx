import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ChampionshipResponse } from "../types";
import { ErrorState, Skeleton } from "../predictor-ui";
import { ChampionshipTable } from "../components/ChampionshipTable";
import { driverName } from "../lib/teamColors";
import { TeamBadge } from "../components/TeamBadge";

type Championship = "drivers" | "constructors";
const LABELS: Record<Championship, string> = { drivers: "Drivers", constructors: "Constructors" };

export function ChampionshipPage() {
  const [tab, setTab] = useState<Championship>("drivers");
  const [data, setData] = useState<Record<Championship, ChampionshipResponse | null>>({ drivers: null, constructors: null });
  const [error, setError] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    setError(false);
    api
      .championship(tab)
      .then((d) => setData((prev) => ({ ...prev, [tab]: d })))
      .catch(() => setError(true));
  }, [tab, reloadKey]);

  const current = data[tab];

  return (
    <section className="rounded-pr border border-pr-rule bg-pr-panel p-4 sm:p-5">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-pr-display text-2xl font-bold uppercase tracking-wide text-pr-text">Championship projection</h2>
          {current && (
            <p className="text-sm text-pr-text-dim">
              After round {current.as_of_round} · {current.n_trials.toLocaleString("en-US")} simulated seasons
            </p>
          )}
        </div>
        <div role="group" aria-label="Championship" className="flex gap-1 rounded-pr border border-pr-rule bg-pr-stage p-1">
          {(["drivers", "constructors"] as const).map((key) => (
            <button
              key={key}
              type="button"
              aria-pressed={tab === key}
              onClick={() => setTab(key)}
              className={`rounded-pr px-3 py-1.5 font-pr-display text-sm font-semibold uppercase tracking-wide transition-colors ${
                tab === key ? "bg-pr-accent text-pr-accent-ink" : "text-pr-text-dim hover:text-pr-text"
              }`}
            >
              {LABELS[key]}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <ErrorState message="We couldn't load the championship projection. Check your connection and try again." onRetry={() => setReloadKey((k) => k + 1)} />
      )}
      {!error && !current && <Skeleton label="Loading projection…" />}
      {!error && current && (
        <ChampionshipTable
          entries={current.standings}
          renderEntity={(id) => (tab === "drivers" ? driverName(id) : <TeamBadge constructorId={id} />)}
        />
      )}
    </section>
  );
}
