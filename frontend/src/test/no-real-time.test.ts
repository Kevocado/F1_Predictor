/**
 * F1's frontend suite timed out on a loaded machine.
 *
 * Reproduced 0-for-12 under four concurrent vitest processes: 2-8 tests per
 * run failed with "Test timed out in 5000ms", never a wrong assertion. Every
 * failing file used the DIRECT `userEvent.click(...)` API, which advances real
 * timers between events; the five that never failed used no userEvent at all,
 * or a single click.
 *
 * The fix is `userEvent.setup()`, whose instance methods do not schedule real
 * time. Asserting the COUNT of setup()-based clicks is what stops this from
 * creeping back: a file that reintroduces the direct API fails here.
 */
import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = resolve(HERE, "..");

function testFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) return testFiles(full);
    return /\.test\.tsx?$/.test(name) ? [full] : [];
  });
}

const files = testFiles(SRC).filter((f) => f.endsWith(".tsx"));

describe("F1 frontend tests do not wait on real time", () => {
  it("finds the component test files (guards the sweep below)", () => {
    expect(files.length).toBeGreaterThan(3);
  });

  it.each(files)("%s uses userEvent.setup(), not the direct click API", (file) => {
    const rel = relative(SRC, file);
    // Strip comments first: the explanation above each fix names the API it
    // replaces, and a guard that matched its own prose would be useless.
    const source = readFileSync(file, "utf8")
      .replace(/\/\*[\s\S]*?\*\//g, "")
      .replace(/(^|[^:])\/\/.*$/gm, "$1");

    // The direct API: userEvent.click(...) / .selectOptions(...) / .type(...).
    // These schedule real time and are what timed out.
    const direct = source.match(/userEvent\.(?!setup\b)[a-zA-Z]+\(/g) ?? [];
    expect(
      direct,
      `${rel} calls userEvent's direct API (${[...new Set(direct)].join(", ")}), ` +
        `which advances real timers and can time out on a loaded machine. ` +
        `Use a setup instance instead: const user = userEvent.setup().`,
    ).toHaveLength(0);

    // And anything that waits on a raw timer, which has the same problem.
    const rawWaits = source.match(/new Promise\(\(?[^)]*\)?\s*=>\s*setTimeout/g) ?? [];
    expect(
      rawWaits,
      `${rel} waits on a raw setTimeout. Assert the state change directly, or ` +
        `await the user-visible result with findBy*.`,
    ).toHaveLength(0);
  });
});
