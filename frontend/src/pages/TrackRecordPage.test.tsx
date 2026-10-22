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
    // 3 snapshotted (counted) races plus 1 rebuilt session. The summary
    // strip judges only the counted races; the rebuilt row carries its own
    // "Made after the session" disclosure.
    mock([entry(3, true, false), entry(2, true, false), entry(1, true, false), entry(4, false, true)], 2);
    render(<TrackRecordPage />);

    expect(await screen.findByText(/This table judges only sessions snapshotted before they ran/)).toBeInTheDocument();
    expect(screen.getByText(/2 sessions have at least one pick recorded at or after the start/)).toBeInTheDocument();
    // 3 counted races, all winners called.
    expect(screen.getByTestId("winners-called")).toHaveTextContent("3/3");
    const rows = screen.getAllByTestId("race-row");
    expect(within(rows[3]).getByText("Made after the session")).toBeInTheDocument();
    // Wrong calls are words, never colour alone: no ✗ anywhere.
    expect(screen.queryByText(/✗/)).not.toBeInTheDocument();
  });

  it("discloses how much of the headline was recorded after the session started", async () => {
    mock([entry(3, true, false)], 1);
    render(<TrackRecordPage />);

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
    expect(within(screen.getByTestId("ledger")).getByText("20")).toBeInTheDocument();
    expect(within(screen.getByTestId("pre-session-ledger")).getByText("10")).toBeInTheDocument();
    expect(within(screen.getByTestId("pre-session-ledger")).getByText("0.021")).toBeInTheDocument();
  });

  it("says the two figures are the same picks rather than printing the ledger twice", async () => {
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

  it("says too early to judge when fewer than 3 races are scored", async () => {
    // 2 counted races: the summary strip is replaced by the honest early
    // state rather than showing percentages as if they meant something.
    mock([entry(3, true, false), entry(2, false, false)], 0);
    render(<TrackRecordPage />);

    expect(await screen.findByText(/Too early to judge: 2 races scored/)).toBeInTheDocument();
    expect(screen.queryByTestId("winners-called")).not.toBeInTheDocument();
  });

  it("keeps jargon out of visible text: no Brier, hit rate, tier, ECE or CRPS", async () => {
    mock([entry(3, true, false)], 0);
    render(<TrackRecordPage />);

    expect(screen.queryByText(/Brier/i)).toBeNull();
    expect(screen.queryByText(/hit rate/i)).toBeNull();
    expect(screen.queryByText(/ECE/i)).toBeNull();
    expect(screen.queryByText(/CRPS/i)).toBeNull();
    // Plain words stand in their place, each with an explainer.
    expect((await screen.findAllByText(/Probability error/)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText(/Right calls/)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText(/Confidence band/)).length).toBeGreaterThan(0);
  });

  it("writes scores in plain numbers", async () => {
    mock([entry(3, true, false)]);
    render(<TrackRecordPage />);
    expect(await screen.findByText("0.041")).toBeInTheDocument();
  });

  it("expands a race to show the predicted top three with probabilities and Right/Wrong calls", async () => {
    const race: RaceAccuracyEntry = {
      season: 2026, round: 3, race_name: "GP 3", tier: "post_qualifying", rebuilt: false,
      win_predicted: ["max_verstappen"], win_actual: ["max_verstappen"], win_hits: 1, win_of: 1,
      podium_predicted: ["max_verstappen", "lando_norris", "charles_leclerc"],
      podium_actual: ["max_verstappen", "lando_norris", "charles_leclerc"],
      podium_hits: 3, podium_of: 3,
      points_finish_predicted: [], points_finish_actual: [], points_finish_hits: 0, points_finish_of: 10,
    };
    mock([race], 0);
    vi.spyOn(api, "racePrediction").mockResolvedValue({
      season: 2026, round: 3, race_name: "GP 3", tier: "post_qualifying", source: "tracked",
      predictions: [
        { driver_id: "max_verstappen", constructor_id: "red_bull", p_win: 0.68, p_podium: 0.85, p_points_finish: 0.95, p_dnf: 0.03, expected_position: 1, expected_points: 25, actual_position: 1, actual_dnf: false },
        { driver_id: "lando_norris", constructor_id: "mclaren", p_win: 0.18, p_podium: 0.7, p_points_finish: 0.9, p_dnf: 0.05, expected_position: 2, expected_points: 18, actual_position: 2, actual_dnf: false },
        { driver_id: "oscar_piastri", constructor_id: "mclaren", p_win: 0.07, p_podium: 0.55, p_points_finish: 0.8, p_dnf: 0.06, expected_position: 3, expected_points: 15, actual_position: 4, actual_dnf: false },
        { driver_id: "charles_leclerc", constructor_id: "ferrari", p_win: 0.04, p_podium: 0.4, p_points_finish: 0.7, p_dnf: 0.08, expected_position: 4, expected_points: 12, actual_position: 3, actual_dnf: false },
      ],
    } as unknown as Awaited<ReturnType<typeof api.racePrediction>>);
    render(<TrackRecordPage />);

    await user.click(await screen.findByRole("button", { name: "Show detail" }));

    // Predicted top three in the detail, each with its probability and a worded call.
    // Oscar Piastri is detail-only: he is not the predicted winner, so his name only
    // appears once the row is expanded.
    expect(await screen.findByText("Oscar Piastri")).toBeInTheDocument();
    expect(screen.getByText("7.0%")).toBeInTheDocument();
    expect(screen.getAllByText("Right")).toHaveLength(2);
    expect(screen.getAllByText("Wrong")).toHaveLength(1);
    // Verstappen (68%) and Norris (18%) are right; their chances are detail-only.
    expect(screen.getByText("68%")).toBeInTheDocument();
    expect(screen.getByText("18%")).toBeInTheDocument();
    // The actual top three sit beside the predicted ones.
    expect(await screen.findByText("Max Verstappen, Lando Norris, Charles Leclerc")).toBeInTheDocument();

    await user.click(await screen.findByRole("button", { name: "Hide detail" }));
    expect(screen.queryByText("Oscar Piastri")).not.toBeInTheDocument();
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
