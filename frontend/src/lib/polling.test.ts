import { describe, expect, it } from "vitest";
import { IDLE_POLL_MS, SESSION_WINDOW_POLL_MS, isSessionWindow, pollIntervalMs } from "./polling";

const race = "2026-11-01T20:00:00Z";
const at = (iso: string) => new Date(iso);

describe("polling", () => {
  it("polls every minute inside the window and every ten minutes outside", () => {
    expect(pollIntervalMs(true)).toBe(SESSION_WINDOW_POLL_MS);
    expect(pollIntervalMs(false)).toBe(IDLE_POLL_MS);
  });
  it("the window runs from 48h before the race to 6h after it", () => {
    expect(isSessionWindow(race, at("2026-10-30T19:59:00Z"))).toBe(false);
    expect(isSessionWindow(race, at("2026-10-30T20:01:00Z"))).toBe(true); // qualifying day
    expect(isSessionWindow(race, at("2026-11-02T01:59:00Z"))).toBe(true);
    expect(isSessionWindow(race, at("2026-11-02T02:01:00Z"))).toBe(false);
  });
  it("is never live without a usable race date", () => {
    expect(isSessionWindow(null)).toBe(false);
    expect(isSessionWindow("not a date")).toBe(false);
  });
});
