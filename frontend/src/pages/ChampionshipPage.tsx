import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ChampionshipResponse } from "../types";
import { ChampionshipTable } from "../components/ChampionshipTable";
import { driverName } from "../lib/teamColors";
import { TeamBadge } from "../components/TeamBadge";

type Championship = "drivers" | "constructors";

export function ChampionshipPage() {
  const [tab, setTab] = useState<Championship>("drivers");
  const [data, setData] = useState<Record<Championship, ChampionshipResponse | null>>({
    drivers: null,
    constructors: null,
  });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .championship(tab)
      .then((d) => setData((prev) => ({ ...prev, [tab]: d })))
      .catch((e) => setError(e.message));
  }, [tab]);

  const current = data[tab];

  return (
    <div className="clip-corner-lg rounded-lg border border-f1-border bg-f1-850/60 p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="font-display text-lg font-bold text-f1-text">Championship Projection</h2>
          {current && (
            <p className="text-xs text-f1-text-faint">
              As of round {current.as_of_round} · {current.n_trials.toLocaleString()}-trial Monte Carlo simulation
            </p>
          )}
        </div>
        <div className="flex gap-1 rounded-lg border border-f1-border bg-f1-800 p-1">
          {(["drivers", "constructors"] as const).map((key) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`rounded-md px-3 py-1 text-xs font-semibold capitalize transition ${
                tab === key ? "bg-f1-red text-white" : "text-f1-text-dim hover:text-f1-text"
              }`}
            >
              {key}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="rounded-lg border border-dnf/30 bg-dnf/10 p-4 text-sm text-dnf">{error}</div>}
      {!error && !current && <div className="animate-pulse text-sm text-f1-text-faint">Loading projection…</div>}
      {current && (
        <ChampionshipTable
          entries={current.standings}
          renderEntity={(id) => (tab === "drivers" ? driverName(id) : <TeamBadge constructorId={id} />)}
        />
      )}
    </div>
  );
}
