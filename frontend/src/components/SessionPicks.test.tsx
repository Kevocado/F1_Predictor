import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";

import { PicksList } from "../predictor-ui";
import { buildPicks, pickCategories, PROJECTION_CATEGORY, type DriverRow } from "./SessionPicks";

/** A race/sprint field: the four markets SESSION_MARKET_SPEC defines for
 *  those session types, plus a projection, all as the API sends them. */
const RACE_DRIVERS: DriverRow[] = [
  {
    driver_id: "max_verstappen", constructor_id: "red_bull",
    p_win: 0.34, p_podium: 0.72, p_points_finish: 0.91, p_dnf: 0.05,
    p_pole: null, p_top_3: null, p_top_10: null,
    expected_position: 2.1, expected_points: 19.4, actual_position: null, actual_dnf: null,
  },
  {
    driver_id: "lando_norris", constructor_id: "mclaren",
    p_win: 0.28, p_podium: 0.69, p_points_finish: 0.9, p_dnf: 0.06,
    p_pole: null, p_top_3: null, p_top_10: null,
    expected_position: 2.4, expected_points: 18.1, actual_position: null, actual_dnf: null,
  },
  {
    driver_id: "charles_leclerc", constructor_id: "ferrari",
    p_win: 0.18, p_podium: 0.55, p_points_finish: 0.85, p_dnf: 0.08,
    p_pole: null, p_top_3: null, p_top_10: null,
    expected_position: 3.6, expected_points: 16.2, actual_position: null, actual_dnf: null,
  },
  {
    driver_id: "oscar_piastri", constructor_id: "mclaren",
    p_win: 0.15, p_podium: 0.5, p_points_finish: 0.83, p_dnf: 0.09,
    p_pole: null, p_top_3: null, p_top_10: null,
    expected_position: 4.0, expected_points: 15.4, actual_position: null, actual_dnf: null,
  },
  {
    // The most likely retirement of the field: DNF ranks on the same number the
    // model produced, highest first.
    driver_id: "pierre_gasly", constructor_id: "alpine",
    p_win: 0.01, p_podium: 0.04, p_points_finish: 0.2, p_dnf: 0.41,
    p_pole: null, p_top_3: null, p_top_10: null,
    expected_position: 12.5, expected_points: 4.2, actual_position: null, actual_dnf: null,
  },
];

/** A qualifying field: pole / top 3 / top 10 and NO dnf at all, which is what
 *  SESSION_MARKET_SPEC and models/session_outcome.py::SESSION_SPECS say
 *  (`has_dnf=False` for both qualifying types). */
const QUALI_DRIVERS: DriverRow[] = RACE_DRIVERS.map((d) => ({
  ...d,
  p_win: null, p_podium: null, p_points_finish: null, p_dnf: null,
  p_pole: d.p_win, p_top_3: d.p_podium, p_top_10: d.p_points_finish,
}));

/** The per-market ledger as `GET /track-record` returns it, filtered to one
 *  session type and tier (store.get_session_track_record groups by
 *  session_type, tier, market and reports n, brier, hit_rate,
 *  avg_predicted_prob). */
const RACE_LEDGER = {
  n_resolved: 1320,
  by_market: [
    { tier: "post_qualifying", market: "win", n: 286, brier: 0.201, hit_rate: 0.28, avg_predicted_prob: 0.27 },
    { tier: "post_qualifying", market: "podium", n: 286, brier: 0.163, hit_rate: 0.51, avg_predicted_prob: 0.5 },
    { tier: "post_qualifying", market: "points_finish", n: 286, brier: 0.089, hit_rate: 0.76, avg_predicted_prob: 0.75 },
    { tier: "post_qualifying", market: "dnf", n: 286, brier: 0.061, hit_rate: 0.09, avg_predicted_prob: 0.1 },
  ],
};

const QUALI_LEDGER = {
  n_resolved: 132,
  by_market: [
    { tier: "post_practice", market: "pole", n: 22, brier: 0.152, hit_rate: 0.09, avg_predicted_prob: 0.1 },
    { tier: "post_practice", market: "top_3", n: 22, brier: 0.14, hit_rate: 0.32, avg_predicted_prob: 0.3 },
    { tier: "post_practice", market: "top_10", n: 22, brier: 0.062, hit_rate: 0.77, avg_predicted_prob: 0.75 },
  ],
};

const categoriesOf = (data: Parameters<typeof buildPicks>[0]) => buildPicks(data).categories;

/** A test that loops over the rows it is given passes on an EMPTY list, so
 *  every such test states how many rows it meant to look at first. Without
 *  this, half of this file was green against a build that rendered nothing. */
function expectRows(categories: { rows: unknown[] }[], atLeast = 3): void {
  const rows = categories.flatMap((c) => c.rows);
  expect(rows.length).toBeGreaterThanOrEqual(atLeast);
}

/* ── 0 · a row is the player, the team and the prediction ─────────────────── */

/**
 * Kevin, 2026-10-01: a top call is the player, the team and the prediction.
 * Nothing else. So no row carries a provenance sentence, a ± margin, a "no
 * graded record" line, a Brier score, a hit rate or an `n`, and the row shows
 * its figure rather than disclaiming what backs it.
 *
 * These assert on the BUILT row, not only the rendered page: the shared
 * component stopped drawing `provenance`, and this stops the site building it.
 */
describe("a row is the driver, the team and the prediction", () => {
  it("carries no provenance, no ± and no ledger text on any built row", () => {
    const built = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    expectRows(built.categories);
    for (const { rows } of built.categories) {
      for (const row of rows) {
        const text = Object.entries(row)
          .filter(([k]) => !["key", "name", "team", "value", "kind"].includes(k))
          .map(([, v]) => String(v))
          .join(" ")
          .toLowerCase();
        for (const banned of ["graded record", "hit rate", "brier", "n=", "±", "error estimate", "ledger"]) {
          expect(text, `stripped text on a built row: ${banned}`).not.toContain(banned);
        }
        // No provenance or margin field at all, rather than an empty one.
        expect(row).not.toHaveProperty("provenance");
        expect(row).not.toHaveProperty("margin");
      }
    }
  });

  it("says no ledger figure on the page, with or without a ledger behind it", () => {
    for (const ledger of [RACE_LEDGER, null]) {
      const view = render(
        <PicksList
          categories={buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger }).categories}
        />,
      );
      const text = screen.getByTestId("picks-list").textContent!.toLowerCase();
      for (const banned of ["graded record", "hit rate", "brier", "n=", "±", "nothing to check it against"]) {
        expect(text, `stripped text on the page: ${banned}`).not.toContain(banned);
      }
      view.unmount();
    }
  });

  it("still names the driver, the team and the figure", () => {
    const built = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    const win = built.categories.find((c) => c.category === "Win")!;
    expect(win.rows[0]).toMatchObject({ name: "Max Verstappen", kind: "probability" });
    expect(win.rows[0].value).toBe(0.34);
    expect(win.rows[0].team).toBeTruthy();

    const view = render(<PicksList categories={built.categories} />);
    const rendered = screen.getAllByTestId("picks-row").filter((r) => r.getAttribute("data-category") === "Win")[0];
    expect(rendered.querySelector('[data-testid="picks-value"]')!.textContent).toBe("34%");
    view.unmount();
  });

  it("a projection row shows the position alone, with no error-estimate note", () => {
    const built = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    const view = render(<PicksList categories={built.categories} />);
    const proj = screen
      .getAllByTestId("picks-row")
      .filter((r) => r.getAttribute("data-category") === PROJECTION_CATEGORY)[0];
    expect(proj).toHaveAttribute("data-kind", "projection");
    expect(proj.querySelector('[data-testid="picks-value"]')!.textContent).toBe("2.1");
    expect(proj.textContent!.toLowerCase()).not.toContain("error estimate");
    expect(proj.textContent).not.toContain("±");
    view.unmount();
  });
});

/* ── 1 · the session's own markets, and only those ────────────────────────── */

describe("categories per session type", () => {
  it("a race session lists win, podium, points finish and DNF", () => {
    expect(categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER }).map((c) => c.category)).toEqual([
      "Win", "Podium", "Points finish", "DNF", "Expected finishing position",
    ]);
  });

  it("a sprint session lists the same four — SESSION_MARKET_SPEC gives sprint the race markets", () => {
    expect(categoriesOf({ sessionType: "sprint", drivers: RACE_DRIVERS, ledger: RACE_LEDGER }).map((c) => c.category)).toEqual([
      "Win", "Podium", "Points finish", "DNF", "Expected finishing position",
    ]);
  });

  it("a qualifying session lists pole, top 3 and top 10", () => {
    expect(categoriesOf({ sessionType: "qualifying", drivers: QUALI_DRIVERS, ledger: QUALI_LEDGER }).map((c) => c.category)).toEqual([
      "Pole", "Top 3", "Top 10", "Expected finishing position",
    ]);
  });

  it("a sprint qualifying session lists the qualifying markets", () => {
    expect(categoriesOf({ sessionType: "sprint_qualifying", drivers: QUALI_DRIVERS, ledger: QUALI_LEDGER }).map((c) => c.category)).toEqual([
      "Pole", "Top 3", "Top 10", "Expected finishing position",
    ]);
  });

  it("has no DNF category at all for a qualifying session — absent, not empty", () => {
    // SESSION_MARKET_SPEC["qualifying"] is pole / top_3 / top_10 with no dnf
    // entry, so a DNF list would be a category the model never produced. An
    // empty heading is the defect this pins: the category must not be built.
    for (const sessionType of ["qualifying", "sprint_qualifying"] as const) {
      const built = buildPicks({ sessionType, drivers: QUALI_DRIVERS, ledger: QUALI_LEDGER });
      // Non-vacuous: the three qualifying categories ARE there, DNF is not.
      expect(built.categories.map((c) => c.category)).toEqual(["Pole", "Top 3", "Top 10", "Expected finishing position"]);
      expect(built.categories.some((c) => c.category.toLowerCase().includes("dnf"))).toBe(false);
      for (const row of built.categories.flatMap((c) => c.rows)) {
        expect(row.detail).not.toBe("DNF");
        // The row that once said which market it quoted now says nothing at
        // all, so no qualifying row can carry a DNF ledger line.
        expect(row).not.toHaveProperty("provenance");
      }
    }
  });

  it("does not read a DNF probability into a race list when the field sends one it cannot grade", () => {
    // The categories come from the spec, not from which fields happen to be
    // non-null: a stray p_dnf on a qualifying field must not create a fourth
    // market category there either.
    const stray = { ...QUALI_DRIVERS[0], p_dnf: 0.3 };
    const built = buildPicks({ sessionType: "qualifying", drivers: [stray, ...QUALI_DRIVERS.slice(1)], ledger: QUALI_LEDGER });
    expect(built.categories.map((c) => c.category)).toEqual(["Pole", "Top 3", "Top 10", "Expected finishing position"]);
  });
});

/* ── 2 · three rows, a ceiling and not a quota ────────────────────────────── */

describe("three rows per category", () => {
  it("shows the model's top three by that market's own number, highest first", () => {
    const win = categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER })[0];
    expect(win.rows.map((r) => r.name)).toEqual(["Max Verstappen", "Lando Norris", "Charles Leclerc"]);
  });

  it("ranks DNF on the DNF probability, not on the finishing markets", () => {
    const dnf = categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER })[3];
    expect(dnf.rows.map((r) => r.name)).toEqual(["Pierre Gasly", "Oscar Piastri", "Charles Leclerc"]);
  });

  it("never shows a fourth row, however large the field", () => {
    const field = Array.from({ length: 20 }, (_, i) => ({
      ...RACE_DRIVERS[0],
      driver_id: `driver_${i}`,
      p_win: 0.4 - i * 0.01,
    }));
    const built = categoriesOf({ sessionType: "race", drivers: field, ledger: RACE_LEDGER });
    // Five lists (four markets plus the projection) at three each.
    expectRows(built, 15);
    expect(built).toHaveLength(5);
    for (const category of built) {
      expect(category.rows.length).toBeLessThanOrEqual(3);
      expect(category.rows.length).toBeGreaterThan(0);
    }
  });

  it("shows fewer when fewer qualify, and pads nothing", () => {
    const two = RACE_DRIVERS.slice(0, 2);
    expect(categoriesOf({ sessionType: "race", drivers: two, ledger: RACE_LEDGER })[0].rows).toHaveLength(2);
  });

  it("keeps one market per list — a points probability never appears under DNF", () => {
    const built = categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    expectRows(built, 15);
    for (const category of built) {
      // Every row's `detail` is its own list's label and nothing else, so no
      // market's number can be filed under a different market's heading.
      const expected = pickCategories[category.category];
      const detail = expected ? expected.detail : "Expected finish";
      for (const row of category.rows) expect(row.detail).toBe(detail);
    }
    // And the four markets each rank on their own field, which the values
    // prove: the DNF list's top row is the highest p_dnf, not the highest
    // p_win, and the two differ in this field.
    const byName = (c: string) => categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER })
      .find((x) => x.category === c)!.rows;
    const gasly = RACE_DRIVERS.find((d) => d.driver_id === "pierre_gasly")!;
    expect(byName("DNF")[0].name).toBe("Pierre Gasly");
    expect(byName("DNF")[0].value).toBe(gasly.p_dnf);
    expect(byName("Win")[0].value).toBe(RACE_DRIVERS[0].p_win);
  });
});

/* ── 3 · probability and projection never share a row ─────────────────────── */

describe("probability vs projection", () => {
  it("marks every market row a probability, so the bar reads a share", () => {
    const built = categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    expectRows(built, 15);
    const markets = built.filter((c) => c.category !== PROJECTION_CATEGORY);
    expect(markets.flatMap((c) => c.rows).length).toBe(12);
    for (const category of markets) {
      for (const row of category.rows) expect(row.kind).toBe("probability");
    }
  });

  it("carries the model's own probability as the value, unrounded and unscaled", () => {
    const win = categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER })[0];
    expect(win.rows[0].value).toBe(0.34);
  });

  it("offers expected_position as a projection in its own list, never inside a probability list", () => {
    const built = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    const projection = built.categories.find((c) => c.category === "Expected finishing position");
    expect(projection).toBeDefined();
    expect(projection!.rows.every((r) => r.kind === "projection")).toBe(true);
    // A projection list is about position, so no probability field may ride in it.
    for (const row of projection!.rows) {
      expect(row.value).toBeLessThan(30);
      expect(row.detail).toBe("Expected finish");
    }
  });

  it("never renders an expected_* figure as a percentage", () => {
    const built = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    const projections = built.categories.flatMap((c) => c.rows).filter((r) => r.kind === "projection");
    expect(projections.length).toBeGreaterThanOrEqual(3);
    for (const row of projections) {
      const { container } = render(
        <dl>
          <dt>{row.name}</dt>
          <dd data-testid="v" data-kind={row.kind}>{String(row.value)}</dd>
        </dl>,
      );
      expect(container.textContent).not.toMatch(/%/);
    }
  });

  it("carries no margin rather than a zero one, when no MAE exists", () => {
    // No per-driver position MAE exists anywhere in F1_Predictor (measured:
    // zero matches for mae/mean_absolute outside the vendored package), so a
    // ± would be a fabricated number. The margin is absent -- and since
    // 2026-10-01 the row says nothing about its absence either.
    const built = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    const projection = built.categories.find((c) => c.category === "Expected finishing position");
    expect(projection).toBeDefined();
    expect(projection!.rows.length).toBeGreaterThanOrEqual(3);
    for (const row of projection!.rows) expect(row.margin).toBeUndefined();
  });

  it("omits the projection list entirely when no expected_position came back", () => {
    // Measured: 287 of the 515 race rows in the committed snapshot carry
    // expected_position as null (schemas.py says so), so this is a state the
    // page really is in, not a hypothetical.
    const blind = RACE_DRIVERS.map((d) => ({ ...d, expected_position: null }));
    const built = buildPicks({ sessionType: "race", drivers: blind, ledger: RACE_LEDGER });
    // The four probability lists are still there; only the projection is gone.
    expect(built.categories.map((c) => c.category)).toEqual(["Win", "Podium", "Points finish", "DNF"]);
    expect(built.categories.flatMap((c) => c.rows).length).toBe(12);
  });
});

/* ── 4 · the ledger reaches no row ────────────────────────────────────────── */

/**
 * Kevin, 2026-10-01: the per-market ledger line is gone from every row. What it
 * used to guarantee is still guaranteed, and more strongly: no row carries any
 * figure from any market's ledger, so none can borrow another's record or invent
 * an `n`. These tests hold that rather than the removed wording.
 */
describe("the ledger reaches no row", () => {
  /** Every string a built row carries beyond name, team and figure. */
  const rowExtras = (row: Record<string, unknown>): string =>
    Object.entries(row)
      .filter(([k]) => !["key", "name", "team", "value", "kind"].includes(k))
      .map(([, v]) => String(v))
      .join(" ");

  const marketRows = (built: ReturnType<typeof buildPicks>) =>
    built.categories.filter((c) => c.category !== PROJECTION_CATEGORY).flatMap((c) => c.rows);

  it("no market row carries any of its ledger's figures or its n", () => {
    const markets = marketRows(buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER }));
    expect(markets).toHaveLength(12);
    for (const row of markets) {
      const text = rowExtras(row as unknown as Record<string, unknown>);
      expect(text).not.toMatch(/hit rate|brier|n=|graded/i);
      expect(row).not.toHaveProperty("provenance");
    }
  });

  it("a market row is the same whichever ledger is behind it, and whichever tier", () => {
    // The rule this replaces: no market borrows another market's record, and no
    // tier borrows another's. Now no row reads the ledger at all, so the rows
    // are byte-identical across all three.
    const withRace = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    const withEmpty = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: { n_resolved: 0, by_market: [] } });
    const otherTier = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER, tier: "pre_weekend" });
    const noLedger = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: null });
    const shape = (b: typeof withRace) => b.categories.map((c) => [c.category, c.rows.map((r) => [r.name, r.value, r.kind])]);
    expect(shape(withEmpty)).toEqual(shape(withRace));
    expect(shape(otherTier)).toEqual(shape(withRace));
    expect(shape(noLedger)).toEqual(shape(withRace));
  });

  it("the projection row borrows no market's n, and says no error estimate", () => {
    const projection = categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER })
      .find((c) => c.category === PROJECTION_CATEGORY)!;
    expect(projection.rows.length).toBe(3);
    for (const row of projection.rows) {
      expect(rowExtras(row as unknown as Record<string, unknown>)).not.toMatch(/n=\d|error estimate|graded/i);
      expect(row).not.toHaveProperty("provenance");
      expect(row).not.toHaveProperty("margin");
    }
  });

  it("a market with no graded record renders the same as one with 286 of them", () => {
    // A sprint weekend has no resolved ledger rows at all (measured: sprint 0
    // resolved, sprint_qualifying 0 resolved in data/tracking.db). It used to
    // read "no graded record for win yet"; it now reads as any other market.
    const sprint = buildPicks({ sessionType: "sprint", drivers: RACE_DRIVERS, ledger: { n_resolved: 0, by_market: [] } });
    const markets = marketRows(sprint);
    expect(markets).toHaveLength(12);
    for (const row of markets) {
      const text = rowExtras(row as unknown as Record<string, unknown>);
      expect(text).not.toMatch(/graded|n=|hit rate/i);
      // The figure the model produced is still there, unrounded.
      expect(typeof row.value).toBe("number");
    }
  });
});

/* ── 5 · the out driver ───────────────────────────────────────────────────── */

describe("a driver who is not in this session", () => {
  const OUT = {
    name: "Logan Sargeant",
    source: "Not in this session's field — the model has no prediction for him",
    dated: "Miami GP weekend",
  };

  /** The seeded case that matters: a driver who IS in the field the model
   *  ranked, and who must not be ranked at all. `drivers` is his row. */
  const withOut = (out: typeof OUT[] = [OUT]) =>
    buildPicks({
      sessionType: "race",
      drivers: [
        { ...RACE_DRIVERS[0], driver_id: "logan_sargeant", p_win: 0.31, p_podium: 0.71, p_points_finish: 0.9, p_dnf: 0.05, expected_position: 2.2 },
        ...RACE_DRIVERS,
      ],
      ledger: RACE_LEDGER,
      out,
    });

  it("leaves the ranking entirely: no row, no bar, no rank", () => {
    // Seeded with a driver whose numbers would place him SECOND in Win and
    // Podium. If the out filter did not run he would be the second row of two
    // lists, which is the "flagged in place" defect the spec forbids.
    const built = withOut();
    const win = built.categories.find((c) => c.category === "Win")!;
    expect(win.rows.length).toBe(3);
    expect(win.rows.map((r) => r.name)).toEqual(["Max Verstappen", "Lando Norris", "Charles Leclerc"]);
    const names = built.categories.flatMap((c) => c.rows.map((r) => r.name));
    expect(names.length).toBeGreaterThanOrEqual(15);
    expect(names).not.toContain(OUT.name);
  });

  it("is dropped from the projection list too, not only the markets", () => {
    const built = withOut();
    const projection = built.categories.find((c) => c.category === PROJECTION_CATEGORY)!;
    expect(projection.rows.map((r) => r.name)).not.toContain(OUT.name);
  });

  it("appears once, below the lists, attributed and dated", () => {
    expect(withOut().out).toEqual([OUT]);
  });

  it("is never passed as a ranked row, so PicksList cannot refuse it in render", () => {
    const built = withOut();
    expect(built.categories.flatMap((c) => c.rows).length).toBeGreaterThanOrEqual(15);
    for (const row of built.categories.flatMap((c) => c.rows)) {
      expect(row.out).not.toBe(true);
      expect(row.name).not.toBe(OUT.name);
    }
  });

  it("says nothing at all when no driver is out — no 'no news' row", () => {
    const built = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    expectRows(built.categories, 15);
    expect(built.out).toEqual([]);
  });
});

/* ── 6 · the shipped component, driven by what we build ───────────────────── */

describe("rendered through the shipped PicksList", () => {
  const renderList = (data: Parameters<typeof buildPicks>[0]) =>
    render(<PicksList categories={buildPicks(data).categories} out={buildPicks(data).out} />);

  it("titles the list 'Model's top calls'", () => {
    renderList({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    expect(screen.getByTestId("picks-title")).toHaveTextContent("Model's top calls");
  });

  it("draws no DNF heading for a qualifying session — and no empty one either", () => {
    renderList({ sessionType: "qualifying", drivers: QUALI_DRIVERS, ledger: QUALI_LEDGER });
    const headings = screen.getAllByTestId("picks-category-heading").map((h) => h.textContent);
    expect(headings).toEqual(["Pole", "Top 3", "Top 10", "Expected finishing position"]);
    expect(screen.queryByText("DNF")).not.toBeInTheDocument();
    // Not rendered-and-empty either: the category does not exist, so there is
    // no heading and no "No pick yet" badge standing in for one.
    expect(screen.queryAllByText("No pick yet")).toHaveLength(0);
  });

  it("draws all four race categories, DNF among them", () => {
    renderList({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    const headings = screen.getAllByTestId("picks-category-heading").map((h) => h.textContent);
    expect(headings).toEqual(["Win", "Podium", "Points finish", "DNF", "Expected finishing position"]);
  });

  it("renders every row as a probability bar, and the projection as a bare number", () => {
    renderList({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    for (const row of screen.getAllByTestId("picks-row")) {
      const kind = row.getAttribute("data-kind");
      expect(kind === "probability" || kind === "projection").toBe(true);
    }
    const projection = screen.getAllByTestId("picks-row").find((r) => r.getAttribute("data-kind") === "projection");
    expect(projection).toBeDefined();
    // The projection is its own figure and nothing else: no ±, and no sentence
    // about there being no error estimate.
    expect(within(projection!).getByTestId("picks-value")).toHaveTextContent("2.1");
    expect(projection!.textContent).not.toContain("±");
    expect(projection!.textContent!.toLowerCase()).not.toContain("error estimate");
    // A probability row draws a share bar; a projection must not.
    expect(within(projection!).queryByTestId("picks-bar")).toBeNull();
  });

  it("prints no n and no ledger figure anywhere on the page", () => {
    renderList({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    const rows = screen.getAllByTestId("picks-row");
    const winRow = rows.find((r) => r.getAttribute("data-category") === "Win")!;
    expect(winRow).not.toHaveTextContent("n=286");
    expect(document.body.textContent ?? "").not.toMatch(/hit rate|brier|graded record/i);
  });

  it("shows the out driver once, below the lists, and out of every ranking", () => {
    renderList({
      sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER,
      out: [{ name: "Logan Sargeant", source: "Not in this session's field", dated: "Miami GP weekend" }],
    });
    const out = screen.getByTestId("picks-out");
    expect(within(out).getAllByText(/Logan Sargeant/)).toHaveLength(1);
    for (const row of screen.getAllByTestId("picks-row")) expect(row).not.toHaveTextContent("Logan Sargeant");
  });

  it("never names a price, an edge or a guarantee", () => {
    const { container } = renderList({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    // Non-vacuous: there is text to be searched, and it is the model's own.
    expect((container.textContent ?? "").length).toBeGreaterThan(200);
    for (const banned of [/lock/i, /guarantee/i, /best bet/i, /\bedge\b/i, /\bvalue bet/i, /odds/i, /\$\d/]) {
      expect(container.textContent ?? "").not.toMatch(banned);
    }
  });
});
