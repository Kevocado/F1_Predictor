/**
 * The instant block on an F1 session: the timing, the verdict and the record,
 * with no tiles and no bar.
 *
 * Every case here states a claim the block has to keep:
 *
 *  - the facts are on screen before anything is asked for. "With the network
 *    blocked" is not a mock this file installs: `src/test/setup.ts` replaces
 *    `globalThis.fetch` with a trap that rejects and names the URL, so a test
 *    that passes here cannot have reached anything.
 *  - F1's reduced rule (spec §A, decision 8): no market tile, no probability
 *    bar. F1's insight stays per-race, so the block contributes timing,
 *    verdict and record and nothing a book would have had to quote.
 *  - the record counts only picks snapshotted before their session. A rebuilt
 *    session is excluded from BOTH sums — that is what `rebuilt` is for, and
 *    it is the one arithmetic in this file worth being pedantic about.
 *  - nothing resolved reads as a dash, never `0/0`.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { SessionTimelinePanel } from "./SessionTimelinePanel";
import { api } from "../api/client";
import type { RaceAccuracyEntry, RacePredictionResponse } from "../types";

const user = userEvent.setup();

const props = { season: 2026, round: 5, isSprintWeekend: false, raceDatetime: "2026-05-03T20:00:00Z" };

function race(source: string): RacePredictionResponse {
  return {
    season: 2026, round: 5, race_name: "Miami Grand Prix", tier: "post_qualifying", source: source as never,
    predictions: [{
      driver_id: "max_verstappen", constructor_id: "red_bull", p_win: 0.34, p_podium: 0.7, p_points_finish: 0.9,
      p_dnf: 0.05, expected_position: 2, expected_points: 18, actual_position: null, actual_dnf: null,
    }],
  };
}

/** One row of `GET /track-record/by-race`, with only the two fields this
 *  feature reads. The predicted/actual name arrays are carried by the real
 *  response and left empty here: nothing under test reads them. */
function accuracy(round: number, rebuilt: boolean, win_hits: number, win_of: number): RaceAccuracyEntry {
  return {
    season: 2026, round, race_name: `Round ${round}`, tier: "post_qualifying", rebuilt,
    win_predicted: [], win_actual: [], win_hits, win_of,
    podium_predicted: [], podium_actual: [], podium_hits: 0, podium_of: 0,
    points_finish_predicted: [], points_finish_actual: [], points_finish_hits: 0, points_finish_of: 0,
  };
}

/** A minimal summary. Only `pick_timing` is read by the assertions below. */
function summary(pick_timing: "pre_kickoff" | "rebuilt" | "none" | "unknown") {
  return {
    verdict: "Norris is the pick, with Verstappen the danger.",
    band: "moderate",
    factors: [{ key: "win", direction: "up", headline: "Pace", text: "The model has Norris." }],
    source: "template" as const,
    model: "",
    generated_at: new Date().toISOString(),
    sport: "f1",
    pick_timing,
  };
}

beforeEach(() => {
  // The session's own numbers. `raceAccuracy` is stubbed here rather than left
  // to the fetch trap: this file is about what the block does with the record,
  // and a rejected request per test is noise on top of a deliberate assertion.
  vi.spyOn(api, "racePrediction").mockResolvedValue(race("live"));
  vi.spyOn(api, "raceAccuracy").mockResolvedValue([]);
});

afterEach(() => vi.restoreAllMocks());

describe("the instant block on an F1 session", () => {
  it("renders the block from the bundle with the network blocked", async () => {
    const explain = vi.spyOn(api, "explainSession");
    render(<SessionTimelinePanel {...props} />);

    const block = await screen.findByTestId("instant-block");
    // The verdict comes from the flow bundle the panel already holds (the
    // highest win probability on the board), not from a request.
    expect(within(block).getByText("Max Verstappen is the pick.")).toBeInTheDocument();
    // The panel loaded its session once and asked for no explanation: this is
    // the whole claim — the facts were already there to be rendered.
    expect(api.racePrediction).toHaveBeenCalledTimes(1);
    expect(explain).not.toHaveBeenCalled();
  });

  it("words the moment as the session, never a kickoff", async () => {
    render(<SessionTimelinePanel {...props} />);
    const block = await screen.findByTestId("instant-block");
    expect(within(block).getByText("Made before the session")).toBeInTheDocument();
    expect(block.innerHTML).not.toContain("kickoff");
  });

  it("draws no tiles and no bar — F1's reduced rule", async () => {
    render(<SessionTimelinePanel {...props} />);
    const block = await screen.findByTestId("instant-block");
    // Non-vacuous: the block is on screen and carries the verdict, so these
    // absences are about a rendered block, not an empty one.
    expect(within(block).getByText("Max Verstappen is the pick.")).toBeInTheDocument();
    // Scoped to the BLOCK, deliberately, and the reason is worth writing down:
    // the hub's block accents the bar segment matching `bundle.pick`, but F1
    // passes no `segments`, so it draws no `pbar-fill` at all. The prediction
    // table and the ExplainRibbon below keep their own bars — decision 8 leaves
    // F1's per-race surface alone, and those bars are not duplication with the
    // block. A document-wide `queryByTestId("pbar-fill")` would be asserting a
    // different product decision; this asks the question actually in play:
    // does the block itself draw one?
    expect(within(block).queryByTestId("pbar-fill")).toBeNull();
    expect(within(block).queryByTestId("pbar-legend")).toBeNull();
    expect(within(block).queryByTestId(/^tile-/)).toBeNull();
    expect(block.innerHTML).not.toContain("market-line");
    expect(block.innerHTML).not.toContain("split-bar");
  });

  it("renders the block above the button, before anything is asked for", async () => {
    const explain = vi.spyOn(api, "explainSession");
    render(<SessionTimelinePanel {...props} />);
    const block = await screen.findByTestId("instant-block");
    const button = screen.getByRole("button", { name: /ai summary/i });
    // Facts first, interpretation after: the block is the finished "what", so
    // the button — and everything it costs — is below it.
    expect(block.compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // On screen before the press, and pressing it was never needed to see it.
    // Awaited: the record arrives on its own request, so a synchronous lookup
    // here is a race against the suite's load rather than a claim about the UI.
    expect(await within(block).findByText("Picks made before the session")).toBeInTheDocument();
    expect(explain).not.toHaveBeenCalled();
  });

  it("prints each figure exactly once, in the block, after the summary lands", async () => {
    // The hub fix this pins: the summary no longer re-renders tiles, bar,
    // legend or record — the block above it already draws each of them, and a
    // second copy is the overlap this phase removes. Asserted DOCUMENT-WIDE,
    // because "the block is the only place" is a claim about the whole panel,
    // not about one subtree: a duplicate would be the summary rendering a
    // figure the block already owns.
    vi.spyOn(api, "raceAccuracy").mockResolvedValue([accuracy(4, false, 3, 4)]);
    vi.spyOn(api, "explainSession").mockResolvedValue(summary("none") as never);
    render(<SessionTimelinePanel {...props} />);
    const block = await screen.findByTestId("instant-block");
    await within(block).findByTestId("record-fill");

    await user.click(screen.getByRole("button", { name: /ai summary/i }));
    const summaryView = await screen.findByTestId("fixture-summary");
    // The block is still mounted under the summary, and still comes first.
    expect(screen.getByTestId("instant-block")).toBe(block);
    expect(block.compareDocumentPosition(summaryView) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    // Every figure the block owns, exactly once, in the whole document.
    expect(screen.getAllByText("Max Verstappen is the pick.")).toHaveLength(1);
    expect(screen.getAllByText("Made before the session")).toHaveLength(1);
    expect(screen.getAllByText("Picks made before the session")).toHaveLength(1);
    expect(screen.getAllByText("3/4")).toHaveLength(1);
    expect(screen.getAllByTestId("record-fill")).toHaveLength(1);
    // And the figures F1 never had: absent from the block AND from the summary,
    // which is what makes the "exactly once" above non-trivial for them.
    expect(screen.queryAllByTestId(/^tile-/)).toHaveLength(0);
    expect(screen.queryAllByTestId("pbar-legend")).toHaveLength(0);
  });

  it("states the rebuilt timing once, in the block", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("rebuilt"));
    render(<SessionTimelinePanel {...props} />);
    // This panel used to print its own `Rebuilt after the session` badge in the
    // header AND let the block print the same disclosure below it — two badges
    // and the same "not counted" claim in one panel. The block owns it, so the
    // badge and its sentence appear exactly once each.
    //
    // Resolved by reading the DOM, not by reading the diff: `StatusBadge` is a
    // bare <span> with no test id, so the count is on the rendered WORDS
    // ("Rebuilt after the session"), which is what a reader actually sees twice.
    const block = await screen.findByTestId("instant-block");
    expect(within(block).getByText("Rebuilt after the session")).toBeInTheDocument();
    // Exactly once in the whole panel: the header's own copy is gone.
    expect(screen.getAllByText("Rebuilt after the session")).toHaveLength(1);
    // And the "not counted" claim is made once, by the block.
    expect(
      screen.getAllByText(/made after the session started, so it is shown for reference and not counted/),
    ).toHaveLength(1);
    expect(screen.queryByText(/isn't counted in the track record/)).toBeNull();
  });

  it("keeps the unverified badge when the schedule gave no start time", async () => {
    // The disclosure the explainer carries as `pick_timing: "unknown"` — a pick
    // the facts cannot place relative to a start time. The block renders the
    // same sentence when F1's bundle carries `pick_timing: "unknown"`; today the
    // bundle carries only `"rebuilt"`, so the wording still arrives with the
    // summary. Either way it must not have been lost: assert the sentence.
    vi.spyOn(api, "explainSession").mockResolvedValue(summary("unknown") as never);
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("fixture-explainer");
    await user.click(screen.getByRole("button", { name: /ai summary/i }));
    expect(await screen.findByText(/did not provide the start time/i)).toBeInTheDocument();
  });
});

describe("the session record", () => {
  it("carries the record and excludes rebuilt sessions from both sums", async () => {
    // One rebuilt session (1 hit of 1) and one honest one (3 of 4). The strip
    // reads 3/4.
    //
    // DEVIATION FROM THE PLAN, deliberately: the plan wrote "3/5" for this seed.
    // That number keeps the rebuilt session's `win_of: 1` in the denominator
    // while dropping its `win_hits: 1` from the numerator, i.e. it books a
    // rebuilt session as a MISS. The plan's own words — "excluding rebuilt
    // sessions is the entire point of that field" — and the Global Constraints
    // ("only a pick made before the start counts") both say otherwise, so the
    // rebuilt row contributes to neither sum and the honest total is 3/4.
    // Counting everything would read 4/5, so the assertion below bites.
    vi.spyOn(api, "raceAccuracy").mockResolvedValue([
      accuracy(3, true, 1, 1),
      accuracy(4, false, 3, 4),
    ]);
    render(<SessionTimelinePanel {...props} />);
    const block = await screen.findByTestId("instant-block");
    expect(await within(block).findByTestId("record-fill")).toBeInTheDocument();
    expect(block).toHaveTextContent("Picks made before the session");
    expect(block).toHaveTextContent("3/4");
    expect(block).not.toHaveTextContent("4/5");
    // One fetch, scoped to the tier and session on screen — not the whole
    // season, and not once per render.
    expect(api.raceAccuracy).toHaveBeenCalledTimes(1);
    expect(api.raceAccuracy).toHaveBeenCalledWith("post_qualifying", "race");
  });

  it("shows a dash, not an invented 0/0, when nothing has resolved", async () => {
    vi.spyOn(api, "raceAccuracy").mockResolvedValue([]);
    render(<SessionTimelinePanel {...props} />);
    const block = await screen.findByTestId("instant-block");
    await within(block).findByText("Picks made before the session");
    expect(within(block).queryByTestId("record-fill")).toBeNull();
    expect(within(block).getByText("—")).toBeInTheDocument();
    expect(block).not.toHaveTextContent("0/0");
  });

  it("shows no record strip while the record is still loading", async () => {
    // Never a flash of 0/0: the strip appears only once there is something to
    // put in it, and stays away when there is not.
    vi.spyOn(api, "raceAccuracy").mockReturnValue(new Promise(() => {}));
    render(<SessionTimelinePanel {...props} />);
    const block = await screen.findByTestId("instant-block");
    expect(within(block).queryByText("Picks made before the session")).toBeNull();
    expect(screen.queryByTestId("record-fill")).toBeNull();
  });
});