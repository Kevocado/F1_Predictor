import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";

afterEach(() => vi.restoreAllMocks());

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
  });

  it("switches pages with the page tabs and keeps every page mounted", async () => {
    renderApp();
    const pages = screen.getByRole("navigation", { name: "Pages" });
    expect(within(pages).getAllByRole("button").map((b) => b.textContent)).toEqual(["Races", "Championship", "Track record"]);
    expect(within(pages).getByRole("button", { name: "Races" })).toHaveAttribute("aria-current", "page");

    await userEvent.click(within(pages).getByRole("button", { name: "Championship" }));
    expect(within(pages).getByRole("button", { name: "Championship" })).toHaveAttribute("aria-current", "page");
    // Hidden, not unmounted: switching back never refetches.
    expect(screen.getByTestId("page-races")).not.toBeVisible();
    expect(screen.getByTestId("page-championship")).toBeVisible();
  });
});
