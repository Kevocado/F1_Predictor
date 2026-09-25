import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ChampionshipPage } from "./ChampionshipPage";
import { api } from "../api/client";

afterEach(() => vi.restoreAllMocks());

const projection = {
  season: 2026, as_of_round: 12, n_trials: 10000,
  standings: [
    { entity_id: "lando_norris", win_prob: 0.62, top3_prob: 0.97, expected_final_points: 402.4, expected_final_rank: 1.4 },
    { entity_id: "max_verstappen", win_prob: 0.003, top3_prob: 0.4, expected_final_points: 301, expected_final_rank: 3.4 },
  ],
};

describe("ChampionshipPage", () => {
  it("shows each driver's title chance and projected rank in words", async () => {
    vi.spyOn(api, "championship").mockResolvedValue(projection);
    render(<ChampionshipPage />);
    expect(await screen.findByText("Lando Norris")).toBeInTheDocument();
    expect(screen.getByText("After round 12 · 10,000 simulated seasons")).toBeInTheDocument();
    expect(screen.getByText("Projected rank 3.4")).toBeInTheDocument();
    expect(screen.getByText("0.3%")).toBeInTheDocument();
  });

  it("switches between drivers and constructors with a labelled toggle", async () => {
    const load = vi.spyOn(api, "championship").mockResolvedValue(projection);
    render(<ChampionshipPage />);
    const group = screen.getByRole("group", { name: "Championship" });
    expect(within(group).getByRole("button", { name: "Drivers" })).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(within(group).getByRole("button", { name: "Constructors" }));
    expect(load).toHaveBeenLastCalledWith("constructors");
  });

  it("recovers from a failed load with Try again", async () => {
    vi.spyOn(api, "championship").mockRejectedValueOnce(new Error("x")).mockResolvedValue(projection);
    render(<ChampionshipPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent("We couldn't load the championship projection.");
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("Lando Norris")).toBeInTheDocument();
  });
});
