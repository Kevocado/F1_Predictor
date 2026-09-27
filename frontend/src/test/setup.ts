import "@testing-library/jest-dom/vitest";
import { configure } from "@testing-library/react";

// Testing Library's own 1s ceiling on findBy* is not a budget derived from
// anything: it is shorter than a jsdom render of these pages takes on a
// loaded machine, so findByText failed while the page was still arriving.
// The symptom was a TestingLibraryElementError showing the loading skeleton —
// "Unable to find ... " against a DOM that had not finished rendering yet,
// which reads exactly like a real failure and is not one.
//
// Raising it costs a passing test nothing (findBy returns the moment the
// element appears) and a genuinely missing element still fails, just later.
// src/test/no-real-time.test.ts is the other half: it keeps the tests off real
// timers, so the only thing this ceiling has to cover is slow CPU.
configure({ asyncUtilTimeout: 10_000 });

// No test may reach the network. A test that leaves one client method
// unstubbed does not fail — it issues a real request to a server that does not
// exist and waits on undici, which is how a single test here stalled past 20s
// under four concurrent vitest processes. Reject instead, naming the URL, so
// the cost is a clear assertion rather than a minute of wall clock.
const realFetch = globalThis.fetch;
globalThis.fetch = ((input: RequestInfo | URL) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  return Promise.reject(new Error(
    `a test made a real network call to ${url}. Stub the client method it should have used.`,
  ));
}) as typeof fetch;

export { realFetch };
