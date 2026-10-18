import { useEffect, useState } from "react";
import { AppFrame } from "./predictor-ui";
import { SITES } from "./lib/sites";
import { RacesPage } from "./pages/RacesPage";
import { ChampionshipPage } from "./pages/ChampionshipPage";
import { TrackRecordPage } from "./pages/TrackRecordPage";

const TABS = [
  { id: "races", label: "Races" },
  { id: "championship", label: "Championship" },
  { id: "track-record", label: "Track record" },
];

// Hash tabs: #races, #championship, #track-record. Back/forward keeps the tab
// in sync. The aggregate "Data Hub" tab does not exist for F1.
function tabFromHash(hash: string): string {
  const id = hash.slice(1);
  return TABS.some((t) => t.id === id) ? id : "races";
}

function App() {
  const [tab, setTab] = useState(() => tabFromHash(window.location.hash));

  useEffect(() => {
    const onHashChange = () => setTab(tabFromHash(window.location.hash));
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  // All pages mount immediately and stay mounted, hidden rather than
  // unmounted, so switching tabs never re-fetches data already loaded.
  return (
    <AppFrame
      sport="f1"
      sportName="F1"
      sites={SITES}
      tabs={TABS}
      activeTab={tab}
      paperTradingHref="https://40-160-91-131.sslip.io/paper.html"
      onTab={(id) => {
        setTab(id);
        window.location.hash = `#${id}`;
      }}
    >
      <div data-testid="page-races" hidden={tab !== "races"}>
        <RacesPage />
      </div>
      <div data-testid="page-championship" hidden={tab !== "championship"}>
        <ChampionshipPage />
      </div>
      <div data-testid="page-track-record" hidden={tab !== "track-record"}>
        <TrackRecordPage />
      </div>
    </AppFrame>
  );
}

export default App;
