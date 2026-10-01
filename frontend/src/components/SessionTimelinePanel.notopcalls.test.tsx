/**
 * Kevin, 2026-10-01: "can you remove this whole part from the F1 summary? it
 * doesnt ened it, the table says everything the f1 model needs."
 *
 * He chose REMOVAL over trimming, so the whole "Model's top calls" section is
 * gone from F1's session panel. The reason F1 can carry that and the other
 * sports cannot is structural, and this file is the guard for it:
 *
 *  - F1's per-driver prediction table (the timing tower) ALREADY lists every
 *    driver's `p_win`, `p_podium` and `p_points_finish` — for all twenty
 *    drivers, not a top three. The pop-out repeated those same numbers under
 *    market headings, so on this sport it was pure duplication.
 *  - NFL, PL and NBA have no equivalent table: no column on their panel carries
 *    every player's probability for every market. Their `PicksList` is the only
 *    place the numbers are shown, so theirs stays. Nothing about
 *    `PicksList` changes — this is F1 choosing not to render it.
 *
 * The second half of this file is therefore the load-bearing half: it asserts
 * the table still shows every driver's probabilities after the pop-out goes.
 * A removal that also took the table's figures with it would satisfy a
 * heading-absence assertion and lose the reason the removal was safe.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// userEvent.setup() rather than the direct userEvent.click(): see
// SessionTimelinePanel.test.tsx, and src/test/no-real-time.test.ts.
const user = userEvent.setup();

import { SessionTimelinePanel } from "./SessionTimelinePanel";
import { api } from "../api/client";
import { driverName } from "../lib/teamColors";
import { pct, pctFine } from "../predictor-ui";
import type { RacePredictionResponse, SessionPredictionResponse } from "../types";

const props = { season: 2026, round: 5, isSprintWeekend: true, raceDatetime: "2026-05-03T20:00:00Z" };

/** The twenty starters, so "every driver" is asserted against a real field
 * rather than a three-row fixture that a top-three list could also satisfy.
 * Each driver's probabilities are DISTINCT, so a count of one figure cannot be
 * explained by another driver's identical number. */
const FULL_FIELD = [
  ["max_verstappen", "red_bull", 0.34, 0.71, 0.93, 0.05],
  ["lando_norris", "mclaren", 0.29, 0.66, 0.88, 0.07],
  ["charles_leclerc", "ferrari", 0.18, 0.55, 0.85, 0.08],
  ["oscar_piastri", "mclaren", 0.15, 0.5, 0.83, 0.09],
  ["george_russell", "mercedes", 0.12, 0.42, 0.78, 0.11],
  ["kimi_antonelli", "mercedes", 0.09, 0.38, 0.74, 0.12],
  ["alex_albon", "williams", 0.05, 0.24, 0.61, 0.19],
  ["pierre_gasly", "alpine", 0.01, 0.04, 0.2, 0.41],
  ["fernando_alonso", "aston_martin", 0.08, 0.31, 0.7, 0.14],
  ["lewis_hamilton", "ferrari", 0.11, 0.36, 0.72, 0.13],
  ["isack_hadjar", "racing_bulls", 0.06, 0.27, 0.65, 0.16],
  ["lance_stroll", "aston_martin", 0.04, 0.22, 0.58, 0.17],
  ["carlos_alonso", "aston_martin", 0.02, 0.09, 0.33, 0.38],
  ["esteban_ocon", "alpine", 0.03, 0.11, 0.36, 0.35],
  ["nico_hulkenberg", "sauber", 0.02, 0.08, 0.29, 0.4],
  ["valtteri_bottas", "sauber", 0.01, 0.06, 0.24, 0.43],
  ["yuki_tsunoda", "racing_bulls", 0.02, 0.07, 0.26, 0.42],
  ["liam_lawson", "racing_bulls", 0.01, 0.05, 0.21, 0.44],
  ["gabriel_bortoleto", "sauber", 0.01, 0.03, 0.18, 0.46],
  ["antonio_giovinazzi", "ferrari", 0.01, 0.02, 0.15, 0.48],
].map(([driver_id, constructor_id, p_win, p_podium, p_points_finish, p_dnf]) => ({
  driver_id: driver_id as string,
  constructor_id: constructor_id as string,
  p_win: p_win as number,
  p_podium: p_podium as number,
  p_points_finish: p_points_finish as number,
  p_dnf: p_dnf as number,
  expected_position: 10,
  expected_points: 8,
  actual_position: null,
  actual_dnf: null,
}));

const race = (): RacePredictionResponse =>
  ({
    season: 2026,
    round: 5,
    race_name: "Miami Grand Prix",
    tier: "post_qualifying",
    source: "live",
    predictions: FULL_FIELD,
  }) as RacePredictionResponse;

const quali = (): SessionPredictionResponse => ({
  season: 2026,
  round: 5,
  race_name: "Miami Grand Prix",
  session_type: "qualifying",
  tier: "post_practice",
  source: "live",
  predictions: FULL_FIELD.map((d) => ({
    driver_id: d.driver_id,
    constructor_id: d.constructor_id,
    p_pole: d.p_win,
    p_top_3: d.p_podium,
    p_top_10: d.p_points_finish,
    p_win: null,
    p_podium: null,
    p_points_finish: null,
    p_dnf: null,
    expected_position: d.expected_position,
    actual_position: null,
    actual_dnf: null,
  })),
});

/** Wait until the panel has fully settled its own requests.
 *
 *  The pop-out was gated on the ledger request settling, so asserting its
 *  absence straight after the tower appears is a race: the assertion can pass
 *  simply because the pop-out had not been drawn yet. The record strip is
 *  awaited instead — it appears only once `raceAccuracy` has resolved, which is
 *  the same clock, and `findBy*` drains the microtask queue on its way there, so
 *  by the time this returns the pop-out would have rendered had it survived. */
async function settled(): Promise<void> {
  await screen.findByText("Picks made before the session");
}

beforeEach(() => {
  vi.spyOn(api, "raceAccuracy").mockResolvedValue([]);
  vi.spyOn(api, "racePrediction").mockResolvedValue(race());
  vi.spyOn(api, "sessionPrediction").mockResolvedValue(quali());
  // The panel used to read this ledger for the pop-out alone. Stubbed here so
  // the removal is measured, not assumed: if a request is still made the
  // assertions below cannot tell, so this spy stays in place either way.
  vi.spyOn(api, "trackRecord").mockResolvedValue({ n_resolved: 1320, by_market: [] });
});
afterEach(() => vi.restoreAllMocks());

describe("F1 renders no 'Model's top calls' section", () => {
  it("draws no such heading on a race session", async () => {
    render(<SessionTimelinePanel {...props} />);
    await settled();
    expect(screen.queryByTestId("picks-title")).not.toBeInTheDocument();
    expect(screen.queryByText(/model's top calls/i)).not.toBeInTheDocument();
  });

  it("draws no such heading on a qualifying session either", async () => {
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("tower-head");
    // Switched rather than rendered fresh: the pop-out rebuilt itself per
    // session, so a race-only assertion would leave qualifying unproven.
    await user.click(screen.getByRole("button", { name: "Qualifying" }));
    await settled();
    expect(screen.queryByTestId("picks-title")).not.toBeInTheDocument();
  });

  it("removes the whole list, not just its heading", async () => {
    render(<SessionTimelinePanel {...props} />);
    await settled();
    // A trimmed section would still leave the per-market pop-out cards, which
    // are the duplication itself.
    expect(screen.queryByTestId("picks-list")).not.toBeInTheDocument();
    expect(screen.queryAllByTestId("picks-category")).toHaveLength(0);
    expect(screen.queryAllByTestId("picks-row")).toHaveLength(0);
    // The market words the pop-out used as category headings.
    for (const heading of ["Points finish", "Expected finishing position"]) {
      expect(screen.queryByText(heading)).not.toBeInTheDocument();
    }
  });

  it("does not request the ledger that existed only to feed it", async () => {
    render(<SessionTimelinePanel {...props} />);
    await settled();
    // The per-market ledger was read for the pop-out and nothing else. With the
    // pop-out gone the request is dead weight, and dead weight on every panel
    // view is a claim about the backend's load that nothing needs any more.
    expect(api.trackRecord).not.toHaveBeenCalled();
  });
});

describe("the per-driver prediction table still says everything", () => {
  it("lists all twenty drivers, each with win, podium and points", async () => {
    render(<SessionTimelinePanel {...props} />);
    const tower = await screen.findByTestId("tower-head");
    const rows = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(rows).toHaveLength(FULL_FIELD.length);
    expect(rows).toHaveLength(20);

    // Every driver is named, and each row carries his own three probabilities
    // plus his DNF risk — the figures the pop-out used to repeat.
    for (const d of FULL_FIELD) {
      const row = rows.find((r) => within(r).queryByText(driverName(d.driver_id)));
      expect(row, `no row for ${d.driver_id}`).toBeDefined();
      const text = row!.textContent ?? "";
      // The headline column (win here) carries one decimal below 10%; the
      // secondary figures are whole percents. Both formatters are imported
      // rather than re-derived, so a formatting change here fails the tower's
      // own tests instead of masquerading as one in this file.
      expect(text, `${d.driver_id} win`).toContain(pctFine(d.p_win));
      expect(text, `${d.driver_id} podium`).toContain(`Podium ${pct(d.p_podium)}`);
      expect(text, `${d.driver_id} points`).toContain(`Points ${pct(d.p_points_finish)}`);
      expect(text, `${d.driver_id} dnf`).toContain(`DNF ${pct(d.p_dnf)}`);
    }
    // Non-vacuous: the header says what the headline column ranks on, so the
    // figures above are read as probabilities rather than as noise.
    expect(tower).toHaveTextContent("Win");
  });

  it("keeps the table on a qualifying session, on pole/top 3/top 10", async () => {
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("tower-head");
    await user.click(screen.getByRole("button", { name: "Qualifying" }));
    const tower = await screen.findByTestId("tower-head");
    expect(tower).toHaveTextContent("Pole");
    const rows = within(screen.getByRole("list")).getAllByRole("listitem");
    expect(rows).toHaveLength(20);
    expect(screen.queryByTestId("picks-title")).not.toBeInTheDocument();
  });
});

