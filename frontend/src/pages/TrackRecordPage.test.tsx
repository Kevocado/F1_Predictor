import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// userEvent.setup() rather than the direct userEvent.click(): the direct
// API advances real timers between events, which timed this suite out on a
// loaded machine (reproduced 0-for-12 with four concurrent vitest runs).
// src/test/no-real-time.test.ts keeps the whole suite on setup().
const user = userEvent.setup();

import { TrackRecordPage } from "./TrackRecordPage";
import { api } from "../api/client";
import type { RaceAccuracyEntry, TrackRecordResponse } from "../types";

afterEach(() => vi.restoreAllMocks());

const entry = (round: number, winHit: boolean, rebuilt: boolean): RaceAccuracyEntry => ({
  season: 2026, round, race_name: `GP ${round}`, tier: "post_qualifying", rebuilt,
  win_predicted: ["max_verstappen"], win_actual: [winHit ? "max_verstappen" : "lando_norris"], win_hits: winHit ? 1 : 0, win_of: 1,
  podium_predicted: [], podium_actual: [], podium_hits: 2, podium_of: 3,
  points_finish_predicted: [], points_finish_actual: [], points_finish_hits: 8, points_finish_of: 10,
});

function mock(
  accuracy: RaceAccuracyEntry[],
  rebuiltSessions = 0,
  ledger: Partial<TrackRecordResponse> = {},
) {
  // The shape `store.get_session_track_record` returns since 2026-10-01: the
  // headline over every recorded pick, and the pre-session subset beside it.
  // The two differ on purpose, and the page has to say which is which.
  vi.spyOn(api, "trackRecord").mockResolvedValue({
    n_resolved: 8, n_rebuilt_sessions: rebuiltSessions,
    by_market: [{ tier: "post_qualifying", market: "win", n: 20, brier: 0.04123, hit_rate: 0.05, avg_predicted_prob: 0.05 }],
    pre_session: { n_resolved: 4, by_market: [{ tier: "post_qualifying", market: "win", n: 10, brier: 0.021, hit_rate: 0.1, avg_predicted_prob: 0.1 }] },
    n_pre_session: 4, n_post_session_picks: 4,
    ...ledger,
  } as TrackRecordResponse);
  vi.spyOn(api, "raceAccuracy").mockResolvedValue(accuracy);
}

describe("TrackRecordPage", () => {
  it("counts every recorded pick in the ledger, and keeps the by-race table on its own rule", async () => {
    // The old test was "counts only sessions snapshotted before they ran, and says
    // how many were left out", and asserted the strings "Only predictions
    // snapshotted before the session count" and "2 sessions were rebuilt after
    // they ran and are left out." Both are gone: the ledger counts every
    // recorded pick, and nothing is described as left out.
    mock([entry(3, true, false), entry(2, false, true), entry(1, false, true)], 2);
    render(<TrackRecordPage />);

    // The by-race table keeps its strict rule, and says so in its own words.
    expect(await screen.findByText(/This table judges only sessions snapshotted before they ran/)).toBeInTheDocument();
    expect(screen.getByText(/2 sessions have at least one pick recorded at or after the start/)).toBeInTheDocument();
    // The summary tiles judge the one pre-session session only.
    expect(screen.getByTestId("winners-called")).toHaveTextContent("1/1");
    const rows = screen.getAllByTestId("race-row");
    expect(within(rows[1]).getByText("Made after the session")).toBeInTheDocument();
    expect(within(rows[1]).queryByText(/✗/)).not.toBeInTheDocument();
  });

  it("discloses how much of the headline was recorded after the session started", async () => {
    mock([entry(3, true, false)], 1);
    render(<TrackRecordPage />);

    // The headline's own numbers, and the split between the two figures.
    expect(await screen.findByText(/Every recorded pick, whenever it was made/)).toBeInTheDocument();
    expect(screen.getByText(/Of these 8 picks, 4 were recorded at or after their session's start, and 4 before it/)).toBeInTheDocument();
    expect(screen.getByTestId("timing-note")).toHaveTextContent(
      "A pick recorded after the session started still counts, and is never presented as one made before it",
    );
    // The old copy is gone.
    expect(screen.queryByText(/Only predictions snapshotted before the session count/)).toBeNull();
    expect(screen.queryByText(/and are left out/)).toBeNull();
  });

  it("prints the pre-session ledger with its own n, beside the headline", async () => {
    mock([entry(3, true, false)], 1);
    render(<TrackRecordPage />);

    expect(await screen.findByText("Made before the session started")).toBeInTheDocument();
    // The headline's win row is n=20 at Brier 0.041; the secondary is n=10 at
    // 0.021. Two ledgers, so the n and the rate both have to differ visibly.
    expect(within(screen.getByTestId("ledger")).getByText("20")).toBeInTheDocument();
    expect(within(screen.getByTestId("pre-session-ledger")).getByText("10")).toBeInTheDocument();
    expect(within(screen.getByTestId("pre-session-ledger")).getByText("0.021")).toBeInTheDocument();
  });

  it("says the two figures are the same picks rather than printing the ledger twice", async () => {
    // "One figure, one place": when nothing was recorded late, the pre-session
    // subset IS the headline, so it is named and not reprinted.
    mock([entry(3, true, false)], 0, {
      pre_session: { n_resolved: 8, by_market: [] },
      n_pre_session: 8, n_post_session_picks: 0,
    });
    render(<TrackRecordPage />);

    expect(await screen.findByTestId("timing-note")).toHaveTextContent(
      "Every pick in this ledger was recorded before its session started",
    );
    expect(screen.queryByTestId("pre-session-ledger")).toBeNull();
  });

  it("says the pre-session figure is empty rather than showing a zero", async () => {
    mock([entry(3, true, true)], 3, {
      pre_session: { n_resolved: 0, by_market: [] },
      n_pre_session: 0, n_post_session_picks: 8,
    });
    render(<TrackRecordPage />);

    expect(
      await screen.findByText(/No pick has been recorded in time yet, so this figure is empty rather than zero/),
    ).toBeInTheDocument();
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
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText(/No resolved predictions yet/)).toBeInTheDocument();
  });
});
