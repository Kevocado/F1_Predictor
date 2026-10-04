/**
 * The trust badge on an F1 session panel.
 *
 * The adapter and the shared component were both already built and tested
 * (`src/f1_predictor/signals/trust.py`, `tests/test_trust_signal.py`,
 * `predictor-ui.trust-contract.test.tsx`), and `GET /signals/{session_id}` is
 * registered on the API — but nothing on the panel ever asked for it. The badge
 * existed in code and had never been on screen.
 *
 * What this file holds, because each is a way the row could go wrong:
 *
 *  - it is INSTANT. The signal is fetched on mount and rendered without the
 *    reader pressing the AI button, which is spec §2's "Signals are instant".
 *    Asking first would make a computed row a paid one.
 *  - it asks with the panel's OWN session id, `{season}-{round}-{session}` —
 *    the same shape `facts.session_id_of` builds, and the same one
 *    `api.explainSession` already uses. A second id grammar here would 404.
 *  - `{"signals": []}` renders NOTHING. Spec §2: "No data, no row. No empty
 *    states, no 'unknown' rows, no filler." An empty list that renders a
 *    placeholder is the failure this test exists to prevent.
 *  - a failed or slow signals request leaves the panel exactly as it was. A
 *    signal is an enhancement; it must not become the page's error state.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import { SessionTimelinePanel } from "./SessionTimelinePanel";
import { api } from "../api/client";
import type { RacePredictionResponse } from "../types";
import type { Signal } from "../predictor-ui";

const props = { season: 2026, round: 5, isSprintWeekend: false, raceDatetime: "2026-05-03T20:00:00Z" };

/** The payload `f1_predictor.signals.trust.trust_signal` builds. */
const trust = (over: Partial<Signal> = {}): Signal => ({
  kind: "trust",
  sport: "f1",
  game_id: "2026-5-race",
  headline: {
    text: "When the model says ~65%, its picks landed 76% of the time",
    figures: { rate: 0.7556, stated_prob: 0.6549 },
  },
  n: 45,
  source: "1,584 resolved picks in this project's tracking.db, 0.6-0.7 probability band, all markets pooled",
  as_of: "2026-10-03T16:44:10Z",
  strength: 0.474,
  pre_kickoff_only: true,
  visual: "reliability_bar",
  ...over,
});

function race(): RacePredictionResponse {
  return {
    season: 2026, round: 5, race_name: "Miami Grand Prix", tier: "post_qualifying", source: "live" as never,
    predictions: [{
      driver_id: "max_verstappen", constructor_id: "red_bull", p_win: 0.34, p_podium: 0.7, p_points_finish: 0.9,
      p_dnf: 0.05, expected_position: 2, expected_points: 18, actual_position: null, actual_dnf: null,
    }],
  };
}

let signals: ReturnType<typeof vi.spyOn>;
let explain: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  vi.spyOn(api, "racePrediction").mockResolvedValue(race());
  vi.spyOn(api, "raceAccuracy").mockResolvedValue([]);
  signals = vi.spyOn(api, "signals").mockResolvedValue({ signals: [trust()] });
  // Spied rather than merely absent so the "instant" claim below is a real
  // assertion on the call count. `src/test/setup.ts` traps `globalThis.fetch`
  // and rejects, so an unmocked call would fail loudly instead of passing
  // quietly — this spy is what lets the panel's AI request be inert here.
  explain = vi.spyOn(api, "explainSession").mockResolvedValue(undefined as never);
});

afterEach(() => vi.restoreAllMocks());

describe("the trust badge on the session panel", () => {
  it("renders the row without the reader asking for anything", async () => {
    render(<SessionTimelinePanel {...props} />);
    expect(await screen.findByText(/its picks landed 76% of the time/)).toBeInTheDocument();
    // Instant: no AI call was needed to put it on screen.
    expect(explain).not.toHaveBeenCalled();
  });

  it("asks with the panel's own session id", async () => {
    render(<SessionTimelinePanel {...props} />);
    await screen.findByText(/its picks landed/);
    expect(signals).toHaveBeenCalledWith(2026, 5, "race");
  });

  it("renders nothing at all when the sport has no signal", async () => {
    signals.mockResolvedValue({ signals: [] } as never);
    render(<SessionTimelinePanel {...props} />);
    // The fixture still renders; the signal area contributes no node.
    expect(await screen.findByText("Miami Grand Prix")).toBeInTheDocument();
    expect(screen.queryByText(/its picks landed/)).not.toBeInTheDocument();
  });

  it("leaves the panel intact when the signals request fails", async () => {
    signals.mockRejectedValue(new Error("signals down"));
    render(<SessionTimelinePanel {...props} />);
    expect(await screen.findByText("Miami Grand Prix")).toBeInTheDocument();
    expect(screen.queryByText(/its picks landed/)).not.toBeInTheDocument();
    // And it is not the panel's error state either.
    expect(screen.queryByRole("button", { name: /try again/i })).not.toBeInTheDocument();
  });
});