import { useState } from "react";
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

function App() {
  const [tab, setTab] = useState("races");

  // All pages mount immediately and stay mounted, hidden rather than
  // unmounted, so switching tabs never re-fetches data already loaded.
  return (
    <AppFrame sport="f1" sportName="F1" sites={SITES} tabs={TABS} activeTab={tab} onTab={setTab}>
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
