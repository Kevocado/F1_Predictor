/**
 * The `trust` payload F1's adapter emits, run through the REAL vendored
 * `SignalRows` — not a copy of it, and not a hand-written mirror of its rules.
 *
 * ## Why this file exists at all, given `SignalRows.test.tsx` upstream
 *
 * The component's own suite (in predictor-hub, deliberately NOT vendored —
 * `SYNC.json` carries runtime files only) already proves its own five
 * behaviours. What nothing proved is that **F1's adapter produces a payload the
 * component accepts**. Those are different claims: the component can be entirely
 * correct about its rules and still refuse every payload this sport sends.
 *
 * So this renders `SignalRows` with a payload of the adapter's exact shape, and
 * pins the four ways that shape can be wrong. The figures here are FIXED rather
 * than read from `data/tracking.db`, and that is deliberate: the DB is refreshed
 * by an automated commit, so a payload copied from it would need re-copying on
 * every refresh and would fail this suite for a reason that has nothing to do
 * with the contract. `tests/test_trust_signal.py` asserts the adapter's LIVE
 * payload against the same rules, reading them out of the vendored source so a
 * change there breaks that suite too. Between the two, a component change breaks
 * a test in this repo without anything here being re-written by hand.
 *
 * The payload is the one `f1_predictor.signals.trust.trust_signal` builds:
 * `kind: trust`, `visual: reliability_bar`, `figures.rate` a share plus the
 * model-side `figures.stated_prob`, `source` naming the band and the pooled
 * market count, `as_of` a full ISO instant.
 */
import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  SignalRows,
  SPEC_MIN_N,
  MAX_HEADLINE_WORDS,
  signalIsDrawn,
  HeadlineFigureMismatchError,
  type Signal,
} from "./predictor-ui";

/** What `trust.trust_signal` emits, with figures pinned. */
const f1Trust = (over: Partial<Signal> = {}): Signal => ({
  kind: "trust",
  sport: "f1",
  game_id: "2026-15-race",
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

describe("the F1 trust payload against the real SignalRows", () => {
  it("renders, and draws the rate the headline states", () => {
    render(<SignalRows signals={[f1Trust()]} />);

    const row = screen.getByTestId("signal-row");
    expect(row).toHaveAttribute("data-kind", "trust");
    expect(row).toHaveAttribute("data-visual", "reliability_bar");

    // The headline the adapter's wording rule produces, unabridged: 12 words is
    // exactly `MAX_HEADLINE_WORDS`, so `data-clipped` must be "false". If the
    // cap ever drops below the adapter's sentence this goes red, which is the
    // point — a clipped row would lose prose but keep the figure, and nobody
    // would notice from the DOM that the sentence had changed shape.
    const headline = screen.getByTestId("signal-headline");
    expect(headline).toHaveAttribute("data-clipped", "false");
    expect(headline).toHaveTextContent(
      "When the model says ~65%, its picks landed 76% of the time",
    );

    // The bar draws `figures.rate` and nothing else, so its width IS the figure.
    expect(screen.getByTestId("signal-bar-fill")).toHaveStyle({ width: "75.56%" });

    // `n` and the date are outside the disclosure, always visible: the sample
    // size is the one number a sceptic needs without opening anything.
    expect(screen.getByTestId("signal-evidence")).toHaveTextContent("n=45");
  });

  it("keeps the sample size and the band reachable in the source line", async () => {
    const user = userEvent.setup();
    render(<SignalRows signals={[f1Trust()]} />);

    const toggle = screen.getByTestId("signal-evidence-toggle");
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await user.click(toggle);

    // `source` is where the pooling disclosure lives, and the pooled markets are
    // the single most important caveat on this row. If it ever stopped rendering,
    // a reader would see "76% of the time" with nothing saying which markets.
    expect(screen.getByTestId("signal-evidence")).toHaveTextContent(
      "1,584 resolved picks in this project's tracking.db, 0.6-0.7 probability band, all markets pooled",
    );
  });

  it("DROPS a row whose n is under the floor rather than rendering the rate", () => {
    // The adapter never emits this — `TRUST_MIN_N` omits the signal entirely.
    // This pins the BACKSTOP: even if an adapter did, the rate does not reach the
    // page, and the whole list renders nothing rather than a bar with no words.
    const thin = f1Trust({ n: SPEC_MIN_N - 1 });
    expect(signalIsDrawn(thin)).toBe(false);
    render(<SignalRows signals={[thin]} />);
    expect(screen.queryByTestId("signal-row")).toBeNull();
    expect(screen.queryByTestId("signal-rows-title")).toBeNull();
  });

  it("DROPS a blank headline rather than rendering a bare bar", () => {
    const blank = f1Trust({ headline: { text: "   ", figures: { rate: 0.7556 } } });
    expect(signalIsDrawn(blank)).toBe(false);
    render(<SignalRows signals={[blank]} />);
    expect(screen.queryByTestId("signal-row")).toBeNull();
  });

  it("accepts n exactly at the floor", () => {
    // The boundary the floor is stated at, from both sides. `>=` not `>`.
    expect(signalIsDrawn(f1Trust({ n: SPEC_MIN_N }))).toBe(true);
  });

  it("REFUSES a headline that states a figure the bar does not draw", () => {
    // THROWS, it does not return false — the component's "WHY THIS THROWS
    // RATHER THAN DROPPING THE HEADLINE" section is explicit that this is the
    // one refusal in `signalIsDrawn` which raises, because a bare bar with no
    // words beside it degrades to the very state the rule forbids.
    //
    // This test was first written as `expect(signalIsDrawn(x)).toBe(false)`,
    // which is the natural guess and which passed against a hand-written
    // stand-in for this function. It blew up here. Running the real component is
    // the only reason it is written correctly.
    //
    // The two cases below are the two ways the adapter's rounding rule could be
    // wrong, and both are 500s on the page rather than warnings.
    const overPrecise = f1Trust({
      headline: { text: "When the model says ~65%, its picks landed 75.6% of the time", figures: { rate: 0.7556 } },
    });
    expect(() => signalIsDrawn(overPrecise)).toThrow(HeadlineFigureMismatchError);

    const wrongWay = f1Trust({
      headline: { text: "When the model says ~65%, its picks landed 61% of the time", figures: { rate: 0.7556 } },
    });
    expect(() => signalIsDrawn(wrongWay)).toThrow(HeadlineFigureMismatchError);

    // The truncation the component allows on purpose: 75.56 draws as 76%, and a
    // writer truncating that same rate may say 75%. The adapter rounds rather
    // than truncating, but the band is allowed and this pins that it is.
    const truncated = f1Trust({
      headline: { text: "When the model says ~65%, its picks landed 75% of the time", figures: { rate: 0.7556 } },
    });
    expect(signalIsDrawn(truncated)).toBe(true);
  });

  it("renders rows in the order the endpoint ranked them, uncapped", () => {
    // Spec §2 puts the rank in the signals layer and `SignalRows` explicitly
    // refuses to re-cap ("Rejected: ranking and capping here"). If a cap were
    // ever added here, a third signal the endpoint ranked in would vanish for a
    // reason the endpoint cannot see.
    const strong = f1Trust({ strength: 0.81, game_id: "2026-1-race" });
    const weak = f1Trust({ strength: 0.07, game_id: "2026-2-race" });
    render(<SignalRows signals={[strong, weak]} />);

    const rows = screen.getAllByTestId("signal-row");
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByTestId("signal-headline")).toBeInTheDocument();
    expect(rows[0].getAttribute("data-row")).toContain("2026-1-race");
    expect(rows[1].getAttribute("data-row")).toContain("2026-2-race");
  });

  it("says the adapter's sentence is inside the cap", () => {
    const words = f1Trust().headline.text.split(/\s+/).filter(Boolean);
    expect(words.length).toBeLessThanOrEqual(MAX_HEADLINE_WORDS);
  });
});