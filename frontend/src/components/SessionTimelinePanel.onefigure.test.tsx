/**
 * The reviewer's Phase-1 live follow-ups for F1, measured rather than eyeballed.
 *
 * 1. **A bare heading with nothing under it.** This repo MEASURES CLEAN for that
 *    defect, and this file is what keeps it clean. Measured by rendering
 *    `SessionTimelinePanel` in jsdom and dumping the flow's own children — tag
 *    and text, so a bare heading is visible *as* a heading:
 *
 *      pre-game (`actual_position: null` everywhere) — rows: `[]`. No heading.
 *      in-play  (a NON-PICK driver classified, the pick not) — rows: `[]`.
 *      finished (the pick classified) — TWO real sentences, the result and the
 *      pick's rightness.
 *
 *    The reason is structural, and it is worth stating because it is the whole
 *    reason this repo does not need a code change: F1's bundle never carries
 *    `home_team`/`away_team`. `FixtureFlow`'s pre-game name row is keyed on
 *    exactly those two keys (`FixtureFlow.tsx:128-129`), so for F1 the row
 *    cannot exist — which the vendored package documents as deliberate ("F1
 *    names a driver rather than a home and an away side, so it has no name row
 *    here and the pre-game flow is empty for that sport — deliberately").
 *
 *    So the guard is the point. Nothing asserted that an F1 flow carries no
 *    HEADING: the existing suite asserted the flow does not *restate* things
 *    (`not.toContain("before the session")` and friends), which a bare name
 *    heading would pass. Adding `home_team` to the bundle, or re-vendoring a
 *    `FixtureFlow` that grew a name row, would put an orphan heading above the
 *    AI button with every current test still green. These assertions are what
 *    make that a red test.
 *
 * 2. **One figure, one place.** MEASURED CLEAN: no figure is printed in both the
 *    instant block and the rest of the panel, in any state. F1 passes the block
 *    no tiles and no bar (spec §A decision 8), so the block carries a timing
 *    badge, a verdict sentence and a record strip and **no numeric tile at
 *    all** — every percentage on the panel belongs to the timing tower. That is
 *    asserted per figure below rather than asserted-about.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";

import { SessionTimelinePanel } from "./SessionTimelinePanel";
import { api } from "../api/client";
import type { RaceAccuracyEntry, RacePredictionResponse } from "../types";

const props = { season: 2026, round: 5, isSprintWeekend: false, raceDatetime: "2026-05-03T20:00:00Z" };

/** A session that has not run. Every probability below is a DISTINCT value with
 *  a distinct rendering, so an audit can tell one figure from another: F1's
 *  figures are printed as whole percents by the timing tower, and two drivers
 *  sharing `p_podium` would make a count of "71%" ambiguous. */
function unclassified(): RacePredictionResponse {
  return {
    season: 2026, round: 5, race_name: "Miami Grand Prix", tier: "post_qualifying", source: "live" as never,
    predictions: [
      { driver_id: "max_verstappen", constructor_id: "red_bull", p_win: 0.34, p_podium: 0.71, p_points_finish: 0.93, p_dnf: 0.05, expected_position: 2, expected_points: 18, actual_position: null, actual_dnf: null },
      { driver_id: "lando_norris", constructor_id: "mclaren", p_win: 0.29, p_podium: 0.66, p_points_finish: 0.88, p_dnf: 0.07, expected_position: 3, expected_points: 15, actual_position: null, actual_dnf: null },
    ],
  } as RacePredictionResponse;
}

/** The in-play window: the session is part-way through, but the PICK has not
 *  finished. `SessionTimelinePanel` derives the flow's state from
 *  `predictions.some(p => p.actual_position != null)`, so this state is
 *  "finished" as far as the prop is concerned — and it still renders nothing,
 *  because `FixtureFlow`'s result sentence needs the PICK's own actual. That
 *  is why PL's and NBA's "derive the state once" fix has no F1 counterpart to
 *  make: F1's rows are already conditional on the pick, not on the session. */
function partwayThrough(): RacePredictionResponse {
  const r = unclassified();
  (r.predictions[1] as { actual_position: number | null }).actual_position = 4;
  return r;
}

/** The pick finished, and finished right. */
function finishedRight(): RacePredictionResponse {
  const r = unclassified();
  (r.predictions[0] as { actual_position: number | null }).actual_position = 1;
  (r.predictions[1] as { actual_position: number | null }).actual_position = 4;
  return r;
}

/** The pick finished, and finished wrong — the other branch of the same row. */
function finishedWrong(): RacePredictionResponse {
  const r = unclassified();
  (r.predictions[0] as { actual_position: number | null }).actual_position = 3;
  (r.predictions[1] as { actual_position: number | null }).actual_position = 4;
  return r;
}

function accuracy(round: number, rebuilt: boolean, win_hits: number, win_of: number): RaceAccuracyEntry {
  return {
    season: 2026, round, race_name: `Round ${round}`, tier: "post_qualifying", rebuilt,
    win_predicted: [], win_actual: [], win_hits, win_of,
    podium_predicted: [], podium_actual: [], podium_hits: 0, podium_of: 0,
    points_finish_predicted: [], points_finish_actual: [], points_finish_hits: 0, points_finish_of: 0,
  } as RaceAccuracyEntry;
}

beforeEach(() => {
  vi.spyOn(api, "raceAccuracy").mockResolvedValue([
    accuracy(1, false, 2, 3), accuracy(2, false, 1, 3), accuracy(3, true, 5, 5),
  ]);
});
afterEach(() => vi.restoreAllMocks());

async function openPanel(prediction: RacePredictionResponse) {
  vi.spyOn(api, "racePrediction").mockResolvedValue(prediction);
  const out = render(<SessionTimelinePanel {...props} />);
  await screen.findByTestId("fixture-flow");
  return out;
}

/** The flow's rows, by tag and text — so a bare heading is visible as a heading
 *  rather than as a string that happens to be a driver's surname. */
function flowRows() {
  return Array.from(screen.getByTestId("fixture-flow").children).map((el) => `${el.tagName}: ${el.textContent}`);
}

/** Everything outside the instant block, as a real DOM region rather than a
 *  string subtraction: subtracting the block's textContent can also delete a
 *  coincidentally identical run elsewhere on the panel. */
function outsideBlock() {
  const clone = document.body.cloneNode(true) as HTMLElement;
  for (const el of Array.from(clone.querySelectorAll('[data-testid="instant-block"]'))) el.remove();
  return clone.textContent ?? "";
}
/** Figures are compared as WHOLE TOKENS, never as substrings: `"7%"` occurs
 *  inside `"17%"` and `"4%"` inside `"54%"`, and a substring sweep on this very
 *  panel reports three phantom duplications on a page that has none. */
const tokenise = (s: string) => s.match(/\d+(?:\.\d+)?%?/g) ?? [];
const countToken = (needle: string, hay: string) => tokenise(hay).filter((t) => t === needle).length;

describe("the flow's rows per state, and the absence of a bare heading", () => {
  it("renders no rows and no heading before the session", async () => {
    await openPanel(unclassified());
    const flow = screen.getByTestId("fixture-flow");

    // MEASURED: `[]`. The pre-game name row is keyed on `home_team`/`away_team`
    // and F1's bundle carries neither, so the row cannot exist.
    expect(flowRows()).toEqual([]);

    // The defect this guards against, asserted as the absence of a HEADING so
    // that renaming the race cannot let a bare name back through.
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();
    expect(within(flow).queryByRole("heading")).toBeNull();
    expect(flow.textContent?.trim()).toBe("");

    // The facts are untouched: the block is still there and still names the pick.
    const block = screen.getByTestId("instant-block");
    expect(within(block).getByText("Max Verstappen is the pick.")).toBeInTheDocument();
    expect(within(block).getByText("Made before the session")).toBeInTheDocument();
  });

  it("renders no rows and no heading while the session is part-way through", async () => {
    await openPanel(partwayThrough());
    const flow = screen.getByTestId("fixture-flow");

    // MEASURED: `[]`, even though the panel has already derived the "finished"
    // state from `some(actual_position != null)`. The result sentence needs the
    // PICK's own finish, which is the condition `FixtureFlow` actually applies,
    // so this state has nothing to say and says nothing.
    expect(flowRows()).toEqual([]);
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();
    expect(screen.getByTestId("instant-block")).toBeInTheDocument();
  });

  it("keeps the flow's real sentences once the pick has finished", async () => {
    await openPanel(finishedRight());
    const flow = screen.getByTestId("fixture-flow");

    // MEASURED: two genuine sentences, and the reason this is not a blanket
    // deletion of the flow anywhere in this family.
    expect(flowRows()).toEqual([
      "P: The result is a win for Max Verstappen.",
      "P: The pick rightness: the model's pick was right.",
    ]);
    // Still no heading: the finished rows are sentences, not a name.
    expect(flow.querySelector("h1, h2, h3, h4, h5, h6")).toBeNull();
  });

  it("says the pick was wrong when the pick was wrong", async () => {
    await openPanel(finishedWrong());
    expect(flowRows()).toEqual([
      "P: The result is not a win for Max Verstappen.",
      "P: The pick rightness: the model's pick was wrong.",
    ]);
  });
});

describe("one figure, one place", () => {
  it("draws no probability tile at all, so the block duplicates no percentage", async () => {
    await openPanel(unclassified());
    const block = screen.getByTestId("instant-block");
    // F1's reduced rule (spec §A, decision 8): the block is handed no tiles and
    // no segments, so it contributes timing, verdict and record and nothing a
    // book would have had to quote. This is the reason the percentage audit
    // below is clean, and it is asserted rather than assumed.
    expect(within(block).queryByTestId(/^tile-/)).toBeNull();
    expect(within(block).queryByTestId("pbar-fill")).toBeNull();

    // Stated precisely, because the first draft of this assertion was WRONG:
    // the block is not figure-free. Its record strip carries two numbers, the
    // hits and the settled count. What it carries is no percentage and no
    // decimal — so there is no probability anywhere on this panel that the block
    // could be duplicating.
    const blockFigures = tokenise(block.textContent ?? "");
    expect(blockFigures.filter((t) => t.includes("%") || t.includes("."))).toEqual([]);
    // ...and the two numbers it does carry are the record's own. The record
    // arrives on its own request after the session's, so it is AWAITED: a
    // `getByText` here passed when this file ran alone and failed in the full
    // suite, which is a race in the test, not a flake in the panel.
    expect(await within(block).findByText("3/6")).toBeInTheDocument();
  });

  it("leaves every driver figure to the timing tower alone", async () => {
    await openPanel(unclassified());
    const blockText = screen.getByTestId("instant-block").textContent ?? "";
    const out = outsideBlock();

    // Each of these is a probability the panel prints in exactly one place. If
    // the block ever grew a tile or a bar for F1, one of these counts would go
    // to two and this fails.
    for (const figure of ["34%", "29%", "71%", "66%", "93%", "88%", "5%", "7%"]) {
      expect(countToken(figure, blockText), `${figure} in block`).toBe(0);
      expect(countToken(figure, out), `${figure} in the site`).toBe(1);
    }
  });

  it("finds no figure printed in both the block and the site, in any state", async () => {
    for (const [name, prediction] of [
      ["pre-game", unclassified()],
      ["in-play", partwayThrough()],
      ["finished", finishedRight()],
    ] as const) {
      const { unmount } = await openPanel(prediction);
      const blockText = screen.getByTestId("instant-block").textContent ?? "";
      const out = outsideBlock();
      const whole = document.body.textContent ?? "";
      const figures = [...new Set(tokenise(whole))].filter((t) => t.includes("%") || t.includes("."));
      const both = figures.filter((t) => countToken(t, blockText) > 0 && countToken(t, out) > 0);
      expect(both, `figures in both the block and the site (${name})`).toEqual([]);
      unmount();
    }
  });

  it("states the record once, from the block's own strip", async () => {
    await openPanel(unclassified());
    const block = screen.getByTestId("instant-block");
    // `tallyRecord` sums only the sessions whose snapshot was not rebuilt:
    // 2/3 + 1/3, with the rebuilt 5/5 excluded from BOTH sums. The strip is
    // where the session record lives, and the rest of the panel does not
    // repeat it. Awaited, because the record arrives on its own request.
    expect(await within(block).findByText("Picks made before the session")).toBeInTheDocument();
    expect(within(block).getByText("3/6")).toBeInTheDocument();
    expect(outsideBlock()).not.toContain("3/6");
  });

  it("does not repeat the block's figures after the summary is pressed", async () => {
    const explain = vi.spyOn(api, "explainSession").mockResolvedValue({
      verdict: "Verstappen is the pick, with Norris the danger.",
      band: "moderate",
      factors: [{ key: "win", direction: "up", headline: "Pace", text: "The model has Verstappen." }],
      source: "template" as const,
      model: "",
      generated_at: new Date().toISOString(),
      sport: "f1",
      pick_timing: "pre_kickoff" as const,
    });
    await openPanel(unclassified());
    // Press the real button, so the summary state is the one under test.
    (screen.getByRole("button", { name: /ai summary/i }) as HTMLButtonElement).click();
    const summary = await screen.findByTestId("fixture-summary");

    // The summary draws no tile, bar or record of its own, and the block does
    // not unmount, so the whole-page figure counts are unchanged.
    expect(within(summary).queryByTestId(/^tile-/)).toBeNull();
    for (const figure of ["34%", "29%", "71%", "66%"]) {
      expect(countToken(figure, document.body.textContent ?? ""), figure).toBe(1);
    }
    expect(explain).toHaveBeenCalledTimes(1);
  });
});
