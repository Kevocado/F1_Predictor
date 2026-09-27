/**
 * No frontend test may reach the network.
 *
 * A test that leaves one client method unstubbed does not fail — it issues a
 * real request to a server that does not exist and waits on undici. That
 * happened here: `SessionTimelinePanel` renders a panel that fetches through
 * `api.summary`, `SessionTimelinePanel.test.tsx` never stubbed it, and a single
 * test stalled past 20s under four concurrent vitest processes. It read as
 * flakiness and was really six tests making six pointless requests.
 *
 * This is a global trap rather than a per-file reminder, so the next test that
 * forgets a spy fails immediately and says what it did, instead of costing a
 * minute of wall clock and a bisect.
 */
import { afterEach, beforeEach, expect, it } from "vitest";

const realFetch = globalThis.fetch;
let calls: string[] = [];

beforeEach(() => {
  calls = [];
  globalThis.fetch = ((input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    calls.push(url);
    return Promise.reject(new Error(`a test made a real network call: ${url}`));
  }) as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = realFetch;
});

it("leaves the network alone in an ordinary test run", () => {
  // The trap is installed for every test file; this one asserts the trap
  // itself works, so a later change cannot quietly disable it.
  expect(typeof globalThis.fetch).toBe("function");
  expect(calls).toEqual([]);
  return globalThis.fetch("/api/explain/f1/2026-12-race").then(
    () => { throw new Error("the trap did not reject"); },
    () => { /* expected */ },
  ).then(() => {
    expect(calls).toEqual(["/api/explain/f1/2026-12-race"]);
  });
});
