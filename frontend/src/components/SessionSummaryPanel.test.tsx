import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SessionSummaryPanel } from "./SessionSummaryPanel";

const explanation = {
  headline: "Norris starts from the front, but the race is closer than the grid suggests.",
  sections: [
    { market: "result", title: "Why Norris", text: "He has the highest win probability of anyone on the grid." },
  ],
  source: "template" as const,
  model: "",
  generated_at: new Date().toISOString(),
  sport: "f1",
  pick_timing: "pre_kickoff" as const,
};

describe("SessionSummaryPanel", () => {
  it("asks for this session's summary and shows its headline", async () => {
    const fetcher = vi.fn().mockResolvedValue(explanation);
    render(<SessionSummaryPanel sessionId="2026-12-race" fetcher={fetcher} />);
    expect(await screen.findByText(explanation.headline)).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledWith("f1", "2026-12-race");
  });

  it("is collapsed to the headline, with a real toggle", async () => {
    const fetcher = vi.fn().mockResolvedValue(explanation);
    render(<SessionSummaryPanel sessionId="2026-12-race" fetcher={fetcher} />);
    await screen.findByText(explanation.headline);
    // The body is withheld, not scrolled: a race page is dense already.
    expect(screen.queryByText(explanation.sections[0].text)).not.toBeInTheDocument();
    const toggle = screen.getByRole("button", { name: "Read the race story" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(toggle);
    expect(screen.getByText(explanation.sections[0].text)).toBeInTheDocument();
  });

  it("says the session's moment, not a kickoff", async () => {
    const fetcher = vi.fn().mockResolvedValue({ ...explanation, pick_timing: "rebuilt" });
    render(<SessionSummaryPanel sessionId="2026-12-race" fetcher={fetcher} />);
    expect(await screen.findByText("Rebuilt after the session")).toBeInTheDocument();
    expect(screen.getByText(/after the session started/)).toBeInTheDocument();
    expect(screen.queryByText(/kickoff/i)).not.toBeInTheDocument();
  });

  it("offers a retry that asks again, and recovers", async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error("explainer down"));
    render(<SessionSummaryPanel sessionId="2026-12-race" fetcher={fetcher} />);
    const retry = await screen.findByRole("button", { name: "Try again" });
    fetcher.mockResolvedValue(explanation);
    await userEvent.click(retry);
    expect(await screen.findByText(explanation.headline)).toBeInTheDocument();
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("renders nothing at all when there is no explainer", () => {
    const { container } = render(<SessionSummaryPanel sessionId="2026-12-race" />);
    expect(container).toBeEmptyDOMElement();
  });
});
