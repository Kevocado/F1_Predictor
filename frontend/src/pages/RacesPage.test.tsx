import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RacesPage } from "./RacesPage";
import { api } from "../api/client";
import type { RaceSummary } from "../types";

afterEach(() => vi.restoreAllMocks());

const race = (round: number, name: string, completed: boolean, sprint = false): RaceSummary => ({
  season: 2026, round, race_name: name, circuit_name: `${name} circuit`,
  race_datetime: `2026-0${round}-0${round}T13:00:00Z`, is_sprint_weekend: sprint, completed,
});

const season = [race(1, "Australian Grand Prix", true), race(2, "Chinese Grand Prix", false, true), race(3, "Japanese Grand Prix", false)];

describe("RacesPage", () => {
  it("opens on the next race and marks it next up", async () => {
    vi.spyOn(api, "races").mockResolvedValue(season);
    const load = vi.spyOn(api, "racePrediction").mockReturnValue(new Promise(() => {}));
    render(<RacesPage />);
    const list = await screen.findByRole("list", { name: "Races" });
    const next = within(list).getByRole("button", { name: /Chinese Grand Prix/ });
    expect(next).toHaveAttribute("aria-current", "true");
    expect(within(next).getByText("Next up")).toBeInTheDocument();
    expect(within(next).getByText("Sprint")).toBeInTheDocument();
    expect(within(list).getByRole("button", { name: /Australian Grand Prix/ })).toHaveTextContent("Completed");
    expect(load).toHaveBeenCalledWith(2026, 2);
  });

  it("offers the season as one select on phones, and switching race loads it", async () => {
    vi.spyOn(api, "races").mockResolvedValue(season);
    const load = vi.spyOn(api, "racePrediction").mockReturnValue(new Promise(() => {}));
    render(<RacesPage />);
    const select = await screen.findByRole("combobox", { name: "Race" });
    await userEvent.selectOptions(select, "3");
    expect(load).toHaveBeenLastCalledWith(2026, 3);
  });

  it("recovers from a failed season load with Try again", async () => {
    vi.spyOn(api, "races").mockRejectedValueOnce(new Error("boom")).mockResolvedValue(season);
    vi.spyOn(api, "racePrediction").mockReturnValue(new Promise(() => {}));
    render(<RacesPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent("We couldn't load the season.");
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("list", { name: "Races" })).toBeInTheDocument();
  });
});
