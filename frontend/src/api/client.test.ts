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

  /* The URL a reader's browser actually asks for, asserted as the final string
     that reaches `fetch`.
     `api.explainSession` was the one call site that spelled its own path
     `/api/explain/f1/...` while `BASE_URL` already ends in `/api`, so the
     browser requested `/api/api/explain/f1/...` and the site answered 404 --
     the "Get the AI summary" button could not work in production.

     No test caught it because every other test mocks `api.explainSession`
     itself, which is where the URL stops existing: mocking the method deletes
     the string this test is about. So this one stubs the NETWORK layer and
     reads the composed URL back off the `fetch` call, which is the only place
     `BASE_URL + path` is ever actually joined. A mock of the api client cannot
     make this pass while the path is wrong, and neither can a change to
     `VITE_API_BASE_URL` alone. */
  it("asks for the session summary under one /api, not two", async () => {
    const fetch = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(() => Promise.resolve(new Response("{}")));
    await api.explainSession(2026, 16, "race");
    await api.explainSession(2026, 16, "qualifying");
    expect(fetch.mock.calls.map((c) => String(c[0]))).toEqual([
      "/api/explain/f1/2026-16-race",
      "/api/explain/f1/2026-16-qualifying",
    ]);
  });

  it("builds the context loader on the same base as explainSession", async () => {
    const fetch = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(() => Promise.resolve(new Response("{}")));
    await api.loadContext("2026-16-race");
    expect(fetch.mock.calls.map((c) => String(c[0]))).toEqual(["/api/explain/f1/2026-16-race/context"]);
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
