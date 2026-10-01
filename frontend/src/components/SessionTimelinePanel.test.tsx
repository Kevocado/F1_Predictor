import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// userEvent.setup() rather than the direct userEvent.click(): the direct
// API advances real timers between events, which timed this suite out on a
// loaded machine (reproduced 0-for-12 with four concurrent vitest runs).
// src/test/no-real-time.test.ts keeps the whole suite on setup().
const user = userEvent.setup();

import { SessionTimelinePanel } from "./SessionTimelinePanel";
import { ApiError, api } from "../api/client";
import type { HistoryCoverage, RacePredictionResponse } from "../types";

afterEach(() => vi.restoreAllMocks());

beforeEach(() => {
  // The panel reads the by-race accuracy for its strip. Stubbed here rather
  // than left to the fetch trap in src/test/setup.ts, so no test in this file
  // issues a request it does not mean to. The record itself is asserted in
  // SessionTimelinePanel.instant.test.tsx.
  vi.spyOn(api, "raceAccuracy").mockResolvedValue([]);
});

function race(source: string, history?: HistoryCoverage): RacePredictionResponse {
  return {
    season: 2026, round: 5, race_name: "Miami Grand Prix", tier: "post_qualifying", source: source as never,
    predictions: [{
      driver_id: "max_verstappen", constructor_id: "red_bull", p_win: 0.34, p_podium: 0.7, p_points_finish: 0.9,
      p_dnf: 0.05, expected_position: 2, expected_points: 18, actual_position: null, actual_dnf: null,
    }],
    ...(history ? { history } : {}),
  };
}

const props = { season: 2026, round: 5, isSprintWeekend: false, raceDatetime: "2026-05-03T20:00:00Z" };

describe("SessionTimelinePanel", () => {
  it("names the race and its start in the viewer's zone", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("live"));
    render(<SessionTimelinePanel {...props} />);
    expect(await screen.findByRole("heading", { name: "Miami Grand Prix" })).toBeInTheDocument();
    expect(screen.getByText(/Round 5 · Sun 3 May · 3:00 PM CDT/)).toBeInTheDocument();
  });

  it("says plainly when a completed session's pick was snapshotted before it", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("tracked"));
    render(<SessionTimelinePanel {...props} />);
    expect(await screen.findByText("Snapshot made before the session")).toBeInTheDocument();
    expect(screen.queryByText(/Rebuilt/)).not.toBeInTheDocument();
  });

  it.each(["rebuilt", "backtest"])("labels a %s prediction as rebuilt, not counted", async (source) => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race(source));
    render(<SessionTimelinePanel {...props} />);
    // The badge and the sentence now come from the instant block, which reads
    // the same `pick_timing` the flow bundle carries. This panel no longer
    // prints its own copy in the header: one badge, one claim.
    expect(await screen.findByText("Rebuilt after the session")).toBeInTheDocument();
    expect(screen.getAllByText("Rebuilt after the session")).toHaveLength(1);
    expect(screen.getByText(/made after the session started, so it is shown for reference and not counted/)).toBeInTheDocument();
  });

  it("switches session with a labelled group of toggles", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("live"));
    const quali = vi.spyOn(api, "sessionPrediction").mockReturnValue(new Promise(() => {}));
    render(<SessionTimelinePanel {...props} />);
    const group = screen.getByRole("group", { name: "Session" });
    expect(within(group).getByRole("button", { name: "Race" })).toHaveAttribute("aria-pressed", "true");
    await user.click(within(group).getByRole("button", { name: "Qualifying" }));
    expect(quali).toHaveBeenCalledWith("qualifying", 2026, 5);
  });

  it("says a session doesn't exist this weekend, rather than showing an error", async () => {
    vi.spyOn(api, "racePrediction").mockRejectedValue(new ApiError("Not found", 404));
    render(<SessionTimelinePanel {...props} />);
    expect(await screen.findByText("Not available for this weekend.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("recovers from a failed load with Try again", async () => {
    const load = vi.spyOn(api, "racePrediction").mockRejectedValueOnce(new Error("boom")).mockResolvedValue(race("live"));
    render(<SessionTimelinePanel {...props} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("We couldn't load this prediction.");
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("Max Verstappen")).toBeInTheDocument();
    expect(load).toHaveBeenCalledTimes(2);
  });
});

/** The plain-English panel, reduced: F1's facts carry win, podium, points and
 *  dnf and NO line, so the flow says the pick and stops. A market-line tile or
 *  a split-bar market row here would be narrating a disagreement with a book
 *  that quoted nothing — the half of the original refusal that still stands. */
const f1summary = {
  verdict: "Norris is the pick, with Verstappen the danger.",
  band: "moderate",
  factors: [{ key: "win", direction: "up", headline: "Pace", text: "The model has Norris." }],
  source: "template" as const,
  model: "",
  generated_at: new Date().toISOString(),
  sport: "f1",
  pick_timing: "pre_kickoff" as const,
};

describe("the plain-English panel, reduced", () => {
  it("draws no market line and no market row, with the pick still legible", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("live"));
    render(<SessionTimelinePanel {...props} />);
    const flow = await screen.findByTestId("fixture-flow");
    // The pick is legible before anything is asked for. It is the block's
    // verdict now, not a flow sentence: saying it in both would put one fact on
    // the page twice (spec §F).
    const block = await screen.findByTestId("instant-block");
    expect(within(block).getByText("Max Verstappen is the pick.")).toBeInTheDocument();
    expect(flow.innerHTML).not.toContain("market-line");
    expect(flow.innerHTML).not.toContain("split-bar");
    expect(flow.innerHTML).not.toContain("market line");
    // The flow no longer restates the pick, the probabilities or the timing.
    expect(flow.innerHTML).not.toContain("The model picks");
    expect(flow.innerHTML).not.toContain("win probability");
    expect(flow.innerHTML).not.toContain("before the session");
  });

  it("words the moment as the session, never a kickoff", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("live"));
    render(<SessionTimelinePanel {...props} />);
    const block = await screen.findByTestId("instant-block");
    expect(block).toHaveTextContent("before the session");
    expect(block.innerHTML).not.toContain("kickoff");
  });

  it("words a rebuilt snapshot as after the session", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("rebuilt"));
    render(<SessionTimelinePanel {...props} />);
    expect(await screen.findByText(/made after the session started, so it is shown for reference and not counted/)).toBeInTheDocument();
  });

  it("makes no summary request until asked, then asks for this session", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("live"));
    const explain = vi.spyOn(api, "explainSession").mockResolvedValue(f1summary as never);
    render(<SessionTimelinePanel {...props} />);
    await screen.findByTestId("fixture-flow");
    expect(explain).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: /ai summary/i }));
    expect(await screen.findByText("Norris is the pick, with Verstappen the danger.")).toBeInTheDocument();
    expect(explain).toHaveBeenCalledWith(2026, 5, "race");
  });

  it("shows the flow with no request made when the explainer is unreachable", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("live"));
    const explain = vi.spyOn(api, "explainSession").mockRejectedValue(new Error("unreachable"));
    render(<SessionTimelinePanel {...props} />);
    expect(await screen.findByTestId("fixture-flow")).toBeInTheDocument();
    expect(screen.getByText("Max Verstappen")).toBeInTheDocument();
    expect(explain).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: /ai summary/i })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByTestId("fixture-summary")).toBeNull();
  });
});

/** A season the model asked for and did not get. The backend warns about this
 *  in the log; without a word on the page the reader gets numbers that look
 *  exactly like a normal weekend's and no way to know the model knew less. */
const shortHistory: HistoryCoverage = {
  complete: false,
  seasons_requested: [2026, 2025, 2024],
  seasons_loaded: [2026, 2025],
  missing_seasons: [{ season: 2024, reason: "fetch failed: RuntimeError (last HTTP status 429)" }],
};

const completeHistory: HistoryCoverage = {
  complete: true,
  seasons_requested: [2026, 2025, 2024],
  seasons_loaded: [2026, 2025, 2024],
  missing_seasons: [],
};

describe("a season the model never got", () => {
  it("says the forecast rests on a shorter history, and names the season", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("live", shortHistory));
    render(<SessionTimelinePanel {...props} />);

    expect(await screen.findByText("Shorter history")).toBeInTheDocument();
    expect(screen.getByText(/rests on a shorter history/i)).toBeInTheDocument();
    expect(screen.getByText(/2024/)).toBeInTheDocument();
  });

  it("does not call the pick invalid, unreliable, or wrong", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("live", shortHistory));
    render(<SessionTimelinePanel {...props} />);

    const notice = await screen.findByText(/rests on a shorter history/i);
    const wording = notice.textContent ?? "";
    for (const overreach of [/invalid/i, /unreliable/i, /\bwrong\b/i, /ignore this/i, /\bdiscard/i, /\bvoid\b/i]) {
      expect(wording).not.toMatch(overreach);
    }
    // And it says what IS still true, so the notice is a caveat and not a retraction.
    expect(screen.getByText(/still produced a pick/i)).toBeInTheDocument();
  });

  it("shows the pick alongside the caveat, rather than in place of it", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("live", shortHistory));
    render(<SessionTimelinePanel {...props} />);

    expect(await screen.findByText("Shorter history")).toBeInTheDocument();
    // Non-vacuous: the tower renders, so the absence of any suppression above is
    // about a shortened forecast, not an empty panel.
    expect(screen.getByText("Max Verstappen")).toBeInTheDocument();
    expect(screen.getByTestId("instant-block")).toHaveTextContent(/Max Verstappen/);
  });

  it("stays quiet when every season loaded", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("live", completeHistory));
    render(<SessionTimelinePanel {...props} />);

    expect(await screen.findByText("Max Verstappen")).toBeInTheDocument();
    expect(screen.queryByText("Shorter history")).not.toBeInTheDocument();
    expect(screen.queryByText(/rests on a shorter history/i)).not.toBeInTheDocument();
  });

  it("stays quiet on a response that reports no history at all", async () => {
    // A stored snapshot or a backtest was built from a history load this request
    // never made. Absence is absence — it is not licence to claim completeness,
    // and it is not licence to raise a caveat the payload cannot support.
    vi.spyOn(api, "racePrediction").mockResolvedValue(race("tracked"));
    render(<SessionTimelinePanel {...props} />);

    expect(await screen.findByText("Snapshot made before the session")).toBeInTheDocument();
    expect(screen.queryByText("Shorter history")).not.toBeInTheDocument();
  });

  it("lists every missing season, not just the first", async () => {
    vi.spyOn(api, "racePrediction").mockResolvedValue(
      race("live", {
        complete: false,
        seasons_requested: [2026, 2025, 2024],
        seasons_loaded: [2026],
        missing_seasons: [
          { season: 2025, reason: "fetch failed: RuntimeError (last HTTP status 429)" },
          { season: 2024, reason: "upstream returned an empty results frame" },
        ],
      }),
    );
    render(<SessionTimelinePanel {...props} />);

    expect(await screen.findByText(/rests on a shorter history/i)).toBeInTheDocument();
    expect(screen.getByText(/2025/)).toBeInTheDocument();
    expect(screen.getByText(/2024/)).toBeInTheDocument();
  });
});
