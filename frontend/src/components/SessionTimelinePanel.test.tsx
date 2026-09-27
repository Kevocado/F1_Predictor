import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// userEvent.setup() rather than the direct userEvent.click(): the direct
// API advances real timers between events, which timed this suite out on a
// loaded machine (reproduced 0-for-12 with four concurrent vitest runs).
// src/test/no-real-time.test.ts keeps the whole suite on setup().
const user = userEvent.setup();

import { SessionTimelinePanel } from "./SessionTimelinePanel";
import { ApiError, api } from "../api/client";
import type { RacePredictionResponse } from "../types";

afterEach(() => vi.restoreAllMocks());

// The panel renders SessionSummaryPanel, which fetches through api.summary.
// This file never asserted anything about that, and never stubbed it, so
// every test that mounted the panel issued a real
// fetch("/api/explain/<season>-<round>-<session>") against undici — no such
// server here. Those connection attempts and keep-alive sockets are what
// stalled a single test past 20s under four concurrent vitest processes: a
// hang rather than slowness, caused by the test rather than the component.
//
// Stubbed here, once, for the whole file. The panel's own behaviour is covered
// in SessionSummaryPanel.test.tsx; what this file owes the reader is that
// mounting the panel costs no network.
beforeEach(() => {
  vi.spyOn(api, "summary").mockResolvedValue({
    headline: "Max Verstappen is the pick, but this grid is closer than it looks.",
    sections: [{ market: "result", title: "Why Verstappen", text: "He has the highest win chance on the grid." }],
    source: "template", model: "", generated_at: new Date().toISOString(),
    sport: "f1", pick_timing: "pre_kickoff",
  });
});

function race(source: string): RacePredictionResponse {
  return {
    season: 2026, round: 5, race_name: "Miami Grand Prix", tier: "post_qualifying", source: source as never,
    predictions: [{
      driver_id: "max_verstappen", constructor_id: "red_bull", p_win: 0.34, p_podium: 0.7, p_points_finish: 0.9,
      p_dnf: 0.05, expected_position: 2, expected_points: 18, actual_position: null, actual_dnf: null,
    }],
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
    expect(await screen.findByText("Rebuilt after the session")).toBeInTheDocument();
    expect(screen.getByText(/isn't counted in the track record/)).toBeInTheDocument();
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
