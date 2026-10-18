import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

// userEvent.setup() rather than the direct userEvent.click(): the direct
// API advances real timers between events, which timed this suite out on a
// loaded machine (reproduced 0-for-12 with four concurrent vitest runs).
// src/test/no-real-time.test.ts keeps the whole suite on setup().
const user = userEvent.setup();

afterEach(() => vi.restoreAllMocks());

import App from "./App";

function renderApp() {
  vi.spyOn(globalThis, "fetch").mockImplementation(() => new Promise(() => {}));
  return render(<App />);
}

describe("App family frame", () => {
  it("carries the family wordmark, the F1 accent and a switcher to every sport", () => {
    const { container } = renderApp();
    expect(screen.getByRole("heading", { name: "F1 Predictor" })).toBeInTheDocument();
    expect(container.querySelector("[data-sport='f1']")).not.toBeNull();
    const sports = screen.getByRole("navigation", { name: "Sports" });
    expect(within(sports).getByRole("link", { name: "F1" })).toHaveAttribute("aria-current", "page");
    expect(within(sports).getByRole("link", { name: "NBA" }).getAttribute("href")).toMatch(/^https:\/\/nba\./);
    expect(within(sports).getByRole("link", { name: "Paper Trading" })).toBeInTheDocument();
  });

  it("switches pages with the page tabs and keeps every page mounted", async () => {
    renderApp();
    const pages = screen.getByRole("tablist", { name: "Pages" });
    expect(within(pages).getAllByRole("tab").map((b) => b.textContent)).toEqual(["Races", "Championship", "Track record"]);
    expect(within(pages).getByRole("tab", { name: "Races" })).toHaveAttribute("aria-selected", "true");

    await user.click(within(pages).getByRole("tab", { name: "Championship" }));
    expect(within(pages).getByRole("tab", { name: "Championship" })).toHaveAttribute("aria-selected", "true");
    // Hidden, not unmounted: switching back never refetches.
    expect(screen.getByTestId("page-races")).not.toBeVisible();
    expect(screen.getByTestId("page-championship")).toBeVisible();
  });
});
