import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// userEvent.setup() rather than the direct userEvent.click(): see
// SessionTimelinePanel.test.tsx, and src/test/no-real-time.test.ts.
const user = userEvent.setup();

import { SessionTimelinePanel } from "./SessionTimelinePanel";
import { api } from "../api/client";
import type { RacePredictionResponse, SessionPredictionResponse, TrackRecordResponse } from "../types";

afterEach(() => vi.restoreAllMocks());

const props = { season: 2026, round: 5, isSprintWeekend: true, raceDatetime: "2026-05-03T20:00:00Z" };

/** The race response as the route sends it: four markets, one projection, and
 *  nulls for the qualifying fields a race never has. */
const RACE_PREDICTIONS = [
  { driver_id: "max_verstappen", constructor_id: "red_bull", p_win: 0.34, p_podium: 0.72, p_points_finish: 0.91, p_dnf: 0.05, expected_position: 2.1, expected_points: 19.4, actual_position: null, actual_dnf: null },
  { driver_id: "lando_norris", constructor_id: "mclaren", p_win: 0.28, p_podium: 0.69, p_points_finish: 0.9, p_dnf: 0.06, expected_position: 2.4, expected_points: 18.1, actual_position: null, actual_dnf: null },
  { driver_id: "charles_leclerc", constructor_id: "ferrari", p_win: 0.18, p_podium: 0.55, p_points_finish: 0.85, p_dnf: 0.08, expected_position: 3.6, expected_points: 16.2, actual_position: null, actual_dnf: null },
  { driver_id: "pierre_gasly", constructor_id: "alpine", p_win: 0.01, p_podium: 0.04, p_points_finish: 0.2, p_dnf: 0.41, expected_position: 12.5, expected_points: 4.2, actual_position: null, actual_dnf: null },
];

const race = (over: Partial<RacePredictionResponse> = {}): RacePredictionResponse => ({
  season: 2026, round: 5, race_name: "Miami Grand Prix", tier: "post_qualifying", source: "live",
  predictions: RACE_PREDICTIONS, ...over,
} as RacePredictionResponse);

/** The qualifying response: pole / top 3 / top 10, and p_dnf null because
 *  SESSION_MARKET_SPEC defines no dnf market for a qualifying session. */
const quali = (): SessionPredictionResponse => ({
  season: 2026, round: 5, race_name: "Miami Grand Prix", session_type: "qualifying", tier: "post_practice",
  source: "live",
  predictions: RACE_PREDICTIONS.map((d) => ({
    driver_id: d.driver_id, constructor_id: d.constructor_id,
    p_pole: d.p_win, p_top_3: d.p_podium, p_top_10: d.p_points_finish,
    p_win: null, p_podium: null, p_points_finish: null, p_dnf: null,
    expected_position: d.expected_position, actual_position: null, actual_dnf: null,
  })),
});

/** The sprint response: the race markets again (SESSION_MARKET_SPEC gives
 *  sprint win/podium/points_finish/dnf), at the post-sprint-qualifying tier. */
const sprint = (): SessionPredictionResponse => ({
  season: 2026, round: 5, race_name: "Miami Grand Prix", session_type: "sprint", tier: "post_sprint_qualifying",
  source: "live",
  predictions: RACE_PREDICTIONS.map((d) => ({
    driver_id: d.driver_id, constructor_id: d.constructor_id,
    p_win: d.p_win, p_podium: d.p_podium, p_points_finish: d.p_points_finish, p_dnf: d.p_dnf,
    p_pole: null, p_top_3: null, p_top_10: null,
    expected_position: d.expected_position, actual_position: null, actual_dnf: null,
  })),
});

/** The per-market ledger for the race tier, as `GET /track-record` returns it. */
const RACE_LEDGER: TrackRecordResponse = {
  n_resolved: 1320,
  by_market: [
    { tier: "post_qualifying", market: "win", n: 286, brier: 0.201, hit_rate: 0.28, avg_predicted_prob: 0.27 },
    { tier: "post_qualifying", market: "podium", n: 286, brier: 0.163, hit_rate: 0.51, avg_predicted_prob: 0.5 },
    { tier: "post_qualifying", market: "points_finish", n: 286, brier: 0.089, hit_rate: 0.76, avg_predicted_prob: 0.75 },
    { tier: "post_qualifying", market: "dnf", n: 286, brier: 0.061, hit_rate: 0.09, avg_predicted_prob: 0.1 },
  ],
};

const QUALI_LEDGER: TrackRecordResponse = {
  n_resolved: 132,
  by_market: [
    { tier: "post_practice", market: "pole", n: 22, brier: 0.152, hit_rate: 0.09, avg_predicted_prob: 0.1 },
    { tier: "post_practice", market: "top_3", n: 22, brier: 0.14, hit_rate: 0.32, avg_predicted_prob: 0.3 },
    { tier: "post_practice", market: "top_10", n: 22, brier: 0.062, hit_rate: 0.77, avg_predicted_prob: 0.75 },
  ],
};

let requestedLedgers: (string | undefined)[][] = [];

beforeEach(() => {
  requestedLedgers = [];
  vi.spyOn(api, "raceAccuracy").mockResolvedValue([]);
  // Every call is recorded, so a test can assert the ledger was asked for at
  // this session's tier rather than assuming it was.
  vi.spyOn(api, "trackRecord").mockImplementation(async (tier, sessionType) => {
    requestedLedgers.push([tier, sessionType]);
    if (sessionType === "race") return RACE_LEDGER;
    if (sessionType === "qualifying" || sessionType === "sprint_qualifying") return QUALI_LEDGER;
    return { n_resolved: 0, by_market: [] };
  });
});

const headings = () => screen.queryAllByTestId("picks-category-heading").map((h) => h.textContent);
const rowsIn = (category: string) =>
  within(screen.getAllByTestId("picks-category").find((s) => s.querySelector('[data-testid="picks-category-heading"]')?.textContent === category)!).getAllByTestId("picks-row");

/** Switch session and wait for the picks list to be that session's.
 *
 *  Waiting on the list's own headings rather than on a driver's name: the
 *  timing tower renders the same market words ("Pole") in the same panel, so a
 *  name-based wait resolves against the wrong component and hides which list
 *  was checked. */
async function switchTo(label: string, expected: string[]): Promise<void> {
  await user.click(screen.getByRole("button", { name: label }));
  await waitFor(() => expect(headings()).toEqual(expected));
}

describe("Model's top calls on a race session", () => {
  it("lists win, podium, points finish and DNF, three rows each", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    render(<SessionTimelinePanel {...props} />);
    expect(await screen.findByTestId("picks-title")).toHaveTextContent("Model's top calls");
    expect(headings()).toEqual(["Win", "Podium", "Points finish", "DNF", "Expected finishing position"]);
    for (const category of ["Win", "Podium", "Points finish", "DNF"]) {
      expect(rowsIn(category).length).toBe(3);
    }
  });

  it("gives every market row its own ledger figures, with n", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    render(<SessionTimelinePanel {...props} />);
    const win = await screen.findByTestId("picks-list");
    const row = within(win).getAllByTestId("picks-row").find((r) => r.getAttribute("data-category") === "Win")!;
    expect(row).toHaveTextContent("28% hit rate");
    expect(row).toHaveTextContent("n=286");
    const dnf = within(win).getAllByTestId("picks-row").find((r) => r.getAttribute("data-category") === "DNF")!;
    // DNF's own record, not the win row's: the most likely retirement is a
    // different market and a different number of graded rows' meaning.
    expect(dnf).toHaveTextContent("9% hit rate");
    expect(dnf).not.toHaveTextContent("28% hit rate");
  });

  it("withholds the list until the ledger has answered, rather than claiming no record", async () => {
    let release!: (r: TrackRecordResponse) => void;
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    vi.spyOn(api, "trackRecord").mockImplementation(() => new Promise<TrackRecordResponse>((r) => (release = r)));
    render(<SessionTimelinePanel {...props} />);
    // The session's own numbers are on screen well before the ledger is.
    expect(await screen.findByTestId("tower-head")).toHaveTextContent("Win");
    // But the list is not drawn with "no graded record" on every row: that
    // sentence is a claim about a record that has not been read yet.
    expect(screen.queryByTestId("picks-list")).not.toBeInTheDocument();
    expect(screen.queryByText(/no graded record/i)).not.toBeInTheDocument();
    release(RACE_LEDGER);
    expect(await screen.findByTestId("picks-list")).toHaveTextContent("n=286");
  });

  it("asks the ledger for this session's tier, not a fixed one", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("picks-list");
    expect(requestedLedgers).toContainEqual(["post_qualifying", "race"]);
  });

  it("draws the projection as a key number and says there is no error estimate", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    render(<SessionTimelinePanel {...props} />);
    const projection = await screen.findByText("Expected finishing position");
    const section = projection.closest('[data-testid="picks-category"]')!;
    const rows = within(section as HTMLElement).getAllByTestId("picks-row");
    expect(rows).toHaveLength(3);
    for (const row of rows) {
      expect(row.getAttribute("data-kind")).toBe("projection");
      // A position of 2.1 drawn as a share would read "210%". It must not.
      expect(row.textContent ?? "").not.toMatch(/\d+%/);
    }
    expect(within(section as HTMLElement).getAllByText("no error estimate yet").length).toBe(3);
  });

  it("marks every market row a probability bar", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    render(<SessionTimelinePanel {...props} />);
    const list = await screen.findByTestId("picks-list");
    const markets = within(list).getAllByTestId("picks-row").filter((r) => r.getAttribute("data-category") !== "Expected finishing position");
    expect(markets.length).toBe(12);
    for (const row of markets) expect(row.getAttribute("data-kind")).toBe("probability");
  });
});

describe("Model's top calls on a qualifying session", () => {
  it("lists pole, top 3 and top 10 — and no DNF category at all", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    const session = vi.spyOn(api, "sessionPrediction").mockResolvedValue(quali());
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("picks-list");
    await switchTo("Qualifying", ["Pole", "Top 3", "Top 10", "Expected finishing position"]);
    expect(session).toHaveBeenCalledWith("qualifying", 2026, 5);
    // SESSION_MARKET_SPEC["qualifying"] is pole / top_3 / top_10 with no dnf
    // entry, so a DNF heading here would be a category the model never
    // produced — and an empty one is just as wrong.
    expect(headings()).toEqual(["Pole", "Top 3", "Top 10", "Expected finishing position"]);
    expect(screen.queryByText("DNF")).not.toBeInTheDocument();
    expect(screen.queryAllByText("No pick yet")).toHaveLength(0);
  });

  it("asks the ledger for the qualifying markets and grades nothing as DNF", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    vi.spyOn(api, "sessionPrediction").mockResolvedValue(quali());
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("picks-list");
    await switchTo("Qualifying", ["Pole", "Top 3", "Top 10", "Expected finishing position"]);
    expect(requestedLedgers).toContainEqual(["post_practice", "qualifying"]);
    const rows = within(screen.getByTestId("picks-list")).getAllByTestId("picks-row");
    expect(rows).toHaveLength(12);
    for (const row of rows) {
      // Not a heading, not a detail, not a provenance line: the word DNF does
      // not appear anywhere on a qualifying session's list.
      expect(row.textContent ?? "").not.toMatch(/\bdnf\b/i);
    }
  });

  it("grades pole under the pole market, whose key is not the field name", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    vi.spyOn(api, "sessionPrediction").mockResolvedValue(quali());
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("picks-list");
    await switchTo("Qualifying", ["Pole", "Top 3", "Top 10", "Expected finishing position"]);
    const row = rowsIn("Pole")[0];
    // p_pole grades as market "pole" (n=22, 9% hit rate), not as market
    // "p_pole" and not under the race win record.
    expect(row).toHaveTextContent("9% hit rate");
    expect(row).toHaveTextContent("n=22");
  });
});

describe("the ledger, when it cannot answer", () => {
  it("says no graded record rather than borrowing another market's numbers", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    // A sprint weekend has no resolved ledger rows at all (measured: 0
    // resolved for sprint and sprint_qualifying in data/tracking.db).
    const session = vi.spyOn(api, "sessionPrediction").mockResolvedValue(sprint());
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("picks-list");
    await switchTo("Sprint", ["Win", "Podium", "Points finish", "DNF", "Expected finishing position"]);
    expect(session).toHaveBeenCalledWith("sprint", 2026, 5);
    const list = screen.getByTestId("picks-list");
    const markets = within(list).getAllByTestId("picks-row").filter((r) => r.getAttribute("data-category") !== "Expected finishing position");
    expect(markets.length).toBe(12);
    for (const row of markets) {
      expect(row).toHaveTextContent(/no graded record for/i);
      expect(row.textContent ?? "").not.toMatch(/n=\d/);
    }
  });

  it("still ranks the field when the ledger request fails", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    vi.spyOn(api, "trackRecord").mockRejectedValue(new Error("ledger down"));
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("picks-list");
    expect(headings()).toEqual(["Win", "Podium", "Points finish", "DNF", "Expected finishing position"]);
    expect(rowsIn("Win").length).toBe(3);
    // A ledger that cannot be read is not a page that cannot be read: the
    // ranking is the model's own and does not depend on the record.
    expect(rowsIn("Win")[0]).toHaveTextContent(/no graded record for win yet/i);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});

describe("what the picks list leaves alone", () => {
  it("does not touch the prediction table's own probability bars", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    render(<SessionTimelinePanel {...props} />);
    // Decision 8 leaves F1's per-race surface alone: the timing tower keeps its
    // per-driver bars and this list adds its own. The duplication is
    // deliberate and is not "fixed" here.
    const list = await screen.findByTestId("picks-list");
    expect(screen.getByTestId("tower-head")).toHaveTextContent("Win");
    // The tower still ranks every driver it always did.
    expect(screen.getAllByRole("listitem").length).toBe(RACE_PREDICTIONS.length);
    expect(within(list).getAllByTestId("picks-row").length).toBe(15);
  });

  it("says no price, no edge and no guarantee anywhere in the list", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    render(<SessionTimelinePanel {...props} />);
    const list = await screen.findByTestId("picks-list");
    const text = list.textContent ?? "";
    expect(text.length).toBeGreaterThan(200);
    for (const banned of [/lock/i, /guarantee/i, /best bet/i, /\bedge\b/i, /odds/i, /\$\d/]) {
      expect(text).not.toMatch(banned);
    }
  });

  it("renders no out line, because F1 has no availability feed to attribute", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race());
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("picks-list");
    // F1's real source is session state: there is no injury report, no entry
    // list and no news feed in this repo (measured: zero `news` matches
    // repo-wide, no injuries module). So no driver is claimed out, and the
    // list says nothing rather than inventing a source.
    expect(screen.queryByTestId("picks-out")).not.toBeInTheDocument();
    expect(screen.queryByText(/^Out:/)).not.toBeInTheDocument();
  });
});
