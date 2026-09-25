import { afterEach, describe, expect, it, vi } from "vitest";
import { REQUEST_TIMEOUT_MS, api } from "./client";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("api client", () => {
  it("calls the same-origin /api, never another port on this host", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("[]"));
    await api.races();
    expect(String(fetch.mock.calls[0][0])).toBe("/api/races");
  });

  it("gives up after the timeout instead of loading forever", async () => {
    vi.useFakeTimers();
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    );
    const pending = api.races();
    const assertion = expect(pending).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS);
    await assertion;
  });
});
