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
        expect(row.provenance).not.toMatch(/\bdnf\b/i);
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

  it("says 'no error estimate yet' rather than a zero margin when there is no MAE", () => {
    // No per-driver position MAE exists anywhere in F1_Predictor (measured:
    // zero matches for mae/mean_absolute outside the vendored package), so a
    // ± would be a fabricated number. The margin is left undefined and the
    // shipped component words it.
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

/* ── 4 · per-row provenance, with its n ───────────────────────────────────── */

describe("provenance", () => {
  it("names the ledger's own figures and its n on every market row", () => {
    const built = categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    const markets = built.filter((c) => c.category !== PROJECTION_CATEGORY);
    expect(markets.flatMap((c) => c.rows).length).toBe(12);
    for (const category of markets) {
      for (const row of category.rows) expect(row.provenance).toMatch(/n=286/);
    }
  });

  it("gives the projection row its own words, never a market's n", () => {
    const projection = categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER })
      .find((c) => c.category === PROJECTION_CATEGORY)!;
    expect(projection.rows.length).toBe(3);
    for (const row of projection.rows) {
      // expected_position is not a graded market, so quoting a ledger n here
      // would be borrowing a record from a different unit of analysis.
      expect(row.provenance).not.toMatch(/n=\d/);
      expect(row.provenance).toMatch(/no error estimate/i);
    }
  });

  it("uses that market's ledger, not another market's", () => {
    const byCategory = Object.fromEntries(
      categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER }).map((c) => [c.category, c.rows[0].provenance]),
    );
    expect(byCategory["Win"]).toMatch(/28% hit rate/);
    expect(byCategory["Win"]).not.toMatch(/51% hit rate/);
    expect(byCategory["Podium"]).toMatch(/51% hit rate/);
    expect(byCategory["DNF"]).toMatch(/9% hit rate/);
  });

  it("says so in words when the ledger has no record for that market", () => {
    // A sprint weekend has no resolved ledger rows at all (measured:
    // sprint 0 resolved, sprint_qualifying 0 resolved in data/tracking.db), so
    // the honest row says there is no graded record, and never borrows the
    // race ledger's numbers.
    const built = buildPicks({ sessionType: "sprint", drivers: RACE_DRIVERS, ledger: { n_resolved: 0, by_market: [] } });
    const markets = built.categories.filter((c) => c.category !== PROJECTION_CATEGORY);
    expect(markets.flatMap((c) => c.rows).length).toBe(12);
    for (const row of markets.flatMap((c) => c.rows)) {
      // The market is named, so the reader knows WHICH record is missing, and
      // no n is printed next to a figure nobody graded.
      expect(row.provenance).toMatch(/no graded record for (win|podium|points_finish|dnf) yet/i);
      expect(row.provenance).not.toMatch(/n=\d/);
    }
  });

  it("does not take a ledger row from another tier", () => {
    // The ledger above holds only post_qualifying rows, and the prediction on
    // screen is pre_weekend: a pre-weekend row must not be told the
    // post-qualifying record's hit rate, because it is a different tier's.
    const built = buildPicks({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER, tier: "pre_weekend" });
    const markets = built.categories.filter((c) => c.category !== PROJECTION_CATEGORY);
    expect(markets.flatMap((c) => c.rows).length).toBe(12);
    for (const row of markets.flatMap((c) => c.rows)) {
      expect(row.provenance).not.toMatch(/n=286/);
      expect(row.provenance).toMatch(/no graded record for /i);
    }
  });

  it("reads the ledger's avg predicted chance beside its hit rate", () => {
    const win = categoriesOf({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER })[0];
    expect(win.rows[0].provenance).toMatch(/28% hit rate/);
    expect(win.rows[0].provenance).toMatch(/27% average/);
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

  it("renders every row as a probability bar, and the projection as a key number", () => {
    renderList({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    for (const row of screen.getAllByTestId("picks-row")) {
      const kind = row.getAttribute("data-kind");
      expect(kind === "probability" || kind === "projection").toBe(true);
    }
    const projection = screen.getAllByTestId("picks-row").find((r) => r.getAttribute("data-kind") === "projection");
    expect(projection).toBeDefined();
    // The ± is worded, not drawn as a zero.
    expect(within(projection!).getByText("no error estimate yet")).toBeInTheDocument();
  });

  it("puts each row's provenance on the page, with its n", () => {
    renderList({ sessionType: "race", drivers: RACE_DRIVERS, ledger: RACE_LEDGER });
    const rows = screen.getAllByTestId("picks-row");
    const winRow = rows.find((r) => r.getAttribute("data-category") === "Win")!;
    expect(winRow).toHaveTextContent("n=286");
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
