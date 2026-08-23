import { useState } from "react";
import { RacesPage } from "./pages/RacesPage";
import { ChampionshipPage } from "./pages/ChampionshipPage";
import { TrackRecordPage } from "./pages/TrackRecordPage";

type Tab = "races" | "championship" | "track-record";

function App() {
  const [tab, setTab] = useState<Tab>("races");

  return (
    <div className="mx-auto min-h-screen max-w-6xl px-6 py-8">
      <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="clip-corner flex h-10 w-10 items-center justify-center rounded-lg bg-f1-red font-display text-lg font-black text-white">
            F1
          </div>
          <div>
            <h1 className="font-display text-xl font-extrabold tracking-tight text-f1-text">F1 Predictor</h1>
            <p className="text-xs text-f1-text-faint">Race outcomes &amp; championship projections</p>
          </div>
        </div>
        <nav className="flex flex-wrap gap-1 rounded-lg border border-f1-border bg-f1-850/60 p-1">
          {(
            [
              ["races", "Races"],
              ["championship", "Championship"],
              ["track-record", "Track Record"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`rounded-md px-3.5 py-1.5 text-sm font-medium transition ${
                tab === key ? "bg-f1-red text-white" : "text-f1-text-dim hover:text-f1-text"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {/* All three pages mount immediately and stay mounted, hidden with
          CSS rather than unmounted, so switching tabs never re-fetches
          data that's already loaded. */}
      <main>
        <div style={{ display: tab === "races" ? "block" : "none" }}>
          <RacesPage />
        </div>
        <div style={{ display: tab === "championship" ? "block" : "none" }}>
          <ChampionshipPage />
        </div>
        <div style={{ display: tab === "track-record" ? "block" : "none" }}>
          <TrackRecordPage />
        </div>
      </main>
    </div>
  );
}

export default App;
