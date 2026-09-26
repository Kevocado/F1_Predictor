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
