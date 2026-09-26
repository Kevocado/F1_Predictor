import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TrackRecordPage } from "./TrackRecordPage";
import { api } from "../api/client";
import type { RaceAccuracyEntry } from "../types";

afterEach(() => vi.restoreAllMocks());

const entry = (round: number, winHit: boolean, rebuilt: boolean): RaceAccuracyEntry => ({
  season: 2026, round, race_name: `GP ${round}`, tier: "post_qualifying", rebuilt,
  win_predicted: ["max_verstappen"], win_actual: [winHit ? "max_verstappen" : "lando_norris"], win_hits: winHit ? 1 : 0, win_of: 1,
  podium_predicted: [], podium_actual: [], podium_hits: 2, podium_of: 3,
  points_finish_predicted: [], points_finish_actual: [], points_finish_hits: 8, points_finish_of: 10,
});

function mock(accuracy: RaceAccuracyEntry[], rebuiltSessions = 0) {
  vi.spyOn(api, "trackRecord").mockResolvedValue({
    n_resolved: 8, n_rebuilt_sessions: rebuiltSessions,
    by_market: [{ tier: "post_qualifying", market: "win", n: 20, brier: 0.04123, hit_rate: 0.05, avg_predicted_prob: 0.05 }],
  });
  vi.spyOn(api, "raceAccuracy").mockResolvedValue(accuracy);
}

describe("TrackRecordPage", () => {
  it("counts only sessions snapshotted before they ran, and says how many were left out", async () => {
    mock([entry(3, true, false), entry(2, false, true), entry(1, false, true)], 2);
    render(<TrackRecordPage />);

    expect(await screen.findByText(/Only predictions snapshotted before the session count/)).toBeInTheDocument();
    expect(screen.getByText(/2 sessions were rebuilt after they ran and are left out/)).toBeInTheDocument();
    // The summary tiles judge the one pre-race session only.
    expect(screen.getByTestId("winners-called")).toHaveTextContent("1/1");
    const rows = screen.getAllByTestId("race-row");
    expect(within(rows[1]).getByText("Rebuilt after the session")).toBeInTheDocument();
    expect(within(rows[1]).queryByText(/✗/)).not.toBeInTheDocument();
  });

  it("says so when no session has been snapshotted in time yet, instead of a 0/0", async () => {
    mock([entry(1, true, true)], 1);
    render(<TrackRecordPage />);
    expect(await screen.findByText(/No races have a prediction snapshotted before they ran yet/)).toBeInTheDocument();
    expect(screen.queryByTestId("winners-called")).not.toBeInTheDocument();
  });

  it("writes scores in plain numbers", async () => {
    mock([entry(3, true, false)]);
    render(<TrackRecordPage />);
    expect(await screen.findByText("0.041")).toBeInTheDocument();
  });

  it("recovers from a failed load with Try again", async () => {
    vi.spyOn(api, "trackRecord").mockRejectedValueOnce(new Error("x")).mockResolvedValue({ n_resolved: 0, by_market: [] });
    vi.spyOn(api, "raceAccuracy").mockResolvedValue([]);
    render(<TrackRecordPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent("We couldn't load the track record.");
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText(/No resolved predictions yet/)).toBeInTheDocument();
  });
});
