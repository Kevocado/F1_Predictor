import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TimingTower } from "./TimingTower";
import { api } from "../api/client";
import type { SessionDriverPrediction } from "../types";

afterEach(() => vi.restoreAllMocks());

function driver(id: string, team: string, p_win: number, over: Partial<SessionDriverPrediction> = {}): SessionDriverPrediction {
  return {
    driver_id: id, constructor_id: team, p_pole: null, p_top_3: null, p_top_10: null,
    p_win, p_podium: Math.min(1, p_win * 2.5), p_points_finish: 0.8, p_dnf: 0.06,
    expected_position: 5, actual_position: null, actual_dnf: null, ...over,
  };
}

const grid = [
  driver("lando_norris", "mclaren", 0.21),
  driver("max_verstappen", "red_bull", 0.34),
  driver("oliver_bearman", "haas", 0.003),
];

describe("TimingTower", () => {
  it("ranks drivers by the headline chance, P1 first, with their team", () => {
    render(<TimingTower predictions={grid} sessionType="race" season={2026} round={5} />);
    const rows = screen.getAllByRole("listitem");
    expect(within(rows[0]).getByText("P1")).toBeInTheDocument();
    expect(within(rows[0]).getByText("Max Verstappen")).toBeInTheDocument();
    expect(within(rows[0]).getByText("Red Bull")).toBeInTheDocument();
    expect(within(rows[2]).getByText("Oliver Bearman")).toBeInTheDocument();
  });

  it("makes the win chance the one big number, with one decimal under 10%", () => {
    render(<TimingTower predictions={grid} sessionType="race" season={2026} round={5} />);
    expect(screen.getByTestId("tower-head")).toHaveTextContent("Win");
    const rows = screen.getAllByRole("listitem");
    expect(within(rows[0]).getByTestId("headline")).toHaveTextContent("34%");
    // One decimal under 10%: a 0.3% win chance is information, never "0%".
    expect(within(rows[2]).getByTestId("headline")).toHaveTextContent("0.3%");
    expect(within(rows[0]).getByText(/Podium 85%/)).toBeInTheDocument();
    expect(within(rows[0]).getByText(/DNF 6%/)).toBeInTheDocument();
  });

  it("leads qualifying with pole, and shows top 3 / top 10", () => {
    const quali = grid.map((d) => ({ ...d, p_win: null, p_podium: null, p_points_finish: null, p_dnf: null, p_pole: d.p_win, p_top_3: 0.5, p_top_10: 0.9 }));
    render(<TimingTower predictions={quali} sessionType="qualifying" season={2026} round={5} />);
    expect(screen.getByTestId("tower-head")).toHaveTextContent("Pole");
    expect(screen.getAllByText(/Top 3 50%/).length).toBe(3);
    expect(screen.queryByText(/DNF/)).not.toBeInTheDocument();
  });

  it("shows where each driver actually finished once the session is done", () => {
    const done = [
      driver("max_verstappen", "red_bull", 0.34, { actual_position: 3 }),
      driver("lando_norris", "mclaren", 0.21, { actual_position: 1 }),
      driver("oliver_bearman", "haas", 0.003, { actual_dnf: true }),
    ];
    render(<TimingTower predictions={done} sessionType="race" season={2026} round={5} />);
    const rows = screen.getAllByRole("listitem");
    expect(within(rows[0]).getByText("Finished P3")).toBeInTheDocument();
    expect(within(rows[1]).getByText("Finished P1")).toBeInTheDocument();
    expect(within(rows[1]).getByText("1 place better")).toBeInTheDocument();
    expect(within(rows[2]).getByText("DNF")).toBeInTheDocument();
  });

  it("opens what's driving a driver's chances, fetching it once", async () => {
    const explain = vi.spyOn(api, "explainPrediction").mockResolvedValue({
      season: 2026, round: 5, driver_id: "max_verstappen", candidate: "x",
      strength_contributors: [{ feature: "grid", value: 1, contribution: 0.4 }],
      dnf_contributors: [],
    });
    render(<TimingTower predictions={grid} sessionType="race" season={2026} round={5} />);
    const toggle = screen.getByRole("button", { name: /Max Verstappen/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(toggle);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(await screen.findByText("Grid position")).toBeInTheDocument();
    await userEvent.click(toggle);
    await userEvent.click(toggle);
    expect(explain).toHaveBeenCalledTimes(1);
  });
});
