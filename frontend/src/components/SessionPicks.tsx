/**
 * The model's own top calls for one session, ranked per market.
 *
 * The list itself is the shipped `PicksList` from the vendored hub package
 * (predictor-hub #63). This file is only the F1 dialect: it decides which
 * markets a session type has, takes each row's number straight from the API,
 * and writes the provenance line from the per-market ledger. It composes no
 * confidence language of its own, so there is no sentence here that could
 * claim a price, an edge or a guarantee — no odds feed exists in this repo.
 *
 * Four measured facts shape every decision below.
 *
 * 1. **DNF exists for race and sprint only.** `SESSION_MARKET_SPEC` in
 *    `src/f1_predictor/tracking/store.py` gives race/sprint
 *    win/podium/points_finish/dnf and qualifying/sprint_qualifying
 *    pole/top_3/top_10 with **no dnf entry at all**;
 *    `models/session_outcome.py::SESSION_SPECS` sets `has_dnf=False` for both
 *    qualifying types. So the qualifying lists are built from the SPEC, not
 *    from whichever `p_*` fields happen to be non-null: a DNF heading on a
 *    qualifying session would be a category the model never produced, which is
 *    why the categories are absent rather than empty.
 *
 * 2. **A probability and a projection are different numbers.**
 *    `p_points_finish` is a share and `expected_points` is a count; they are
 *    never in one row. Every `p_*` market here is `kind: "probability"` and
 *    draws a share bar. `expected_position` gets its own list as
 *    `kind: "projection"` — a position, not a share — and PicksList renders it
 *    as a key number.
 *
 * 3. **There is no error estimate for `expected_position`.** No per-driver
 *    position MAE exists anywhere in F1_Predictor (measured: zero matches for
 *    mae/mean_absolute outside the vendored package, and no such field on
 *    `DriverPrediction` or `SessionDriverPrediction`). So the row carries no
 *    `margin` and the shipped component words it "no error estimate yet",
 *    which is a true statement; `± 0` would be a claim.
 *
 * 4. **Provenance is per row and per market, with its n.**
 *    `GET /track-record` -> `store.get_session_track_record` groups by
 *    (session_type, tier, market) and reports `n`, `brier`, `hit_rate` and
 *    `avg_predicted_prob` for that market only. Every row quotes its own
 *    market's figures. Where the ledger has no row for that market or tier,
 *    the row says so in words — it never borrows another market's record, and
 *    never invents an n. (Measured in `data/tracking.db`: 1320 resolved race
 *    rows and 132 qualifying rows; 0 resolved for sprint and sprint
 *    qualifying, so a sprint weekend's rows say "no graded record yet".)
 */
import { MAX_ROWS_PER_CATEGORY, type OutPlayer, type PickRow } from "../predictor-ui";
import { pct } from "../predictor-ui";
import { driverName, teamName } from "../lib/teamColors";
import type { SessionType, TrackRecordResponse } from "../types";

/** One driver on a session tab. The race response and the three session
 *  responses disagree on which `p_*` fields they carry, so every one of them
 *  is nullable here: the markets a session type does not have are absent, not
 *  zero. */
export type DriverRow = {
  driver_id: string;
  constructor_id: string | null;
  p_win: number | null;
  p_podium: number | null;
  p_points_finish: number | null;
  p_dnf: number | null;
  p_pole: number | null;
  p_top_3: number | null;
  p_top_10: number | null;
  expected_position: number | null;
  /** Carried so a caller cannot accidentally file it as a probability. The
   *  list never renders it: `expected_points` is a points projection and
   *  `p_points_finish` is a share, and they are different numbers. */
  expected_points?: number | null;
  actual_position: number | null;
  actual_dnf: boolean | null;
};

/** The markets, in the order the list shows them.
 *
 *  `field` is the probability this row is ranked by. `market` is the key the
 *  ledger groups by — the first element of SESSION_MARKET_SPEC's pair, which
 *  is NOT the field name for qualifying (`p_pole` grades as market `pole`).
 *  Getting that pairing wrong would put a qualifying row's provenance on a
 *  market the ledger never graded, so the two are named separately on
 *  purpose. */
type PickCategory = {
  /** The heading, and the key `pickCategories` is looked up by. */
  label: string;
  /** The word on the row itself. */
  detail: string;
  /** The `p_*` field this list ranks. Always a probability. */
  field: "p_win" | "p_podium" | "p_points_finish" | "p_dnf" | "p_pole" | "p_top_3" | "p_top_10";
  /** The ledger's market key for that field. */
  market: string;
  /** Which session types have this market at all. */
  sessions: SessionType[];
};

const RACE_SESSIONS: SessionType[] = ["race", "sprint"];
const QUALI_SESSIONS: SessionType[] = ["qualifying", "sprint_qualifying"];

/** Every list this file can build, by heading. Exported so a test can assert
 *  one category per list from the same table the render reads, instead of a
 *  second copy of the mapping in the test. */
export const pickCategories: Record<string, PickCategory> = {
  Win: { label: "Win", detail: "Win", field: "p_win", market: "win", sessions: RACE_SESSIONS },
  Podium: { label: "Podium", detail: "Podium", field: "p_podium", market: "podium", sessions: RACE_SESSIONS },
  "Points finish": {
    label: "Points finish", detail: "Points", field: "p_points_finish", market: "points_finish", sessions: RACE_SESSIONS,
  },
  // No DNF key with quali sessions in `sessions`: the category is built from
  // this table, so a qualifying session cannot reach it even if a stray
  // `p_dnf` arrives in the payload.
  DNF: { label: "DNF", detail: "DNF", field: "p_dnf", market: "dnf", sessions: RACE_SESSIONS },
  Pole: { label: "Pole", detail: "Pole", field: "p_pole", market: "pole", sessions: QUALI_SESSIONS },
  "Top 3": { label: "Top 3", detail: "Top 3", field: "p_top_3", market: "top_3", sessions: QUALI_SESSIONS },
  "Top 10": { label: "Top 10", detail: "Top 10", field: "p_top_10", market: "top_10", sessions: QUALI_SESSIONS },
};

/** The heading of the projection list, kept out of `pickCategories` because it
 *  is not a market and the ledger does not grade it. */
export const PROJECTION_CATEGORY = "Expected finishing position";

/** Ordered so the list reads the way the tower does: the headline market
 *  first, then the rest of the session's markets, then the projection. */
const ORDER: string[] = [
  "Win", "Pole", "Podium", "Top 3", "Points finish", "Top 10", "DNF", PROJECTION_CATEGORY,
];

const finite = (x: number | null | undefined): number | undefined =>
  typeof x === "number" && Number.isFinite(x) ? x : undefined;

/** This market's own ledger row, and only that one.
 *
 *  The API is called with the session type and tier already applied, but the
 *  tier is re-checked here: a ledger that carries several tiers must not let a
 *  pre-weekend row's record speak for a post-qualifying prediction. `session_type`
 *  is not in the response (the route strips it), so the market key plus the
 *  tier is the whole match. */
function ledgerFor(
  ledger: TrackRecordResponse | null,
  market: string,
  tier: string | null,
): TrackRecordResponse["by_market"][number] | null {
  if (!ledger) return null;
  for (const row of ledger.by_market) {
    if (row.market !== market) continue;
    if (tier && row.tier !== tier) continue;
    return row;
  }
  return null;
}

/** The provenance line, in the caller's own figures.
 *
 *  With a ledger row: the market's hit rate, the average chance the model gave
 *  for it, the Brier score and the number of graded records behind all three —
 *  `n` travels with them, because a hit rate over 22 rows is not the same
 *  claim as one over 286 and a reader cannot see that without it.
 *
 *  Without one: the honest sentence. A sprint weekend has no resolved ledger
 *  rows at all, and a market with no graded record cannot borrow another
 *  market's numbers, so the row says so and says nothing else. */
function provenanceFor(
  market: string,
  row: { n: number; brier: number; hit_rate: number; avg_predicted_prob: number } | null,
): string {
  if (!row) return `No graded record for ${market} yet — nothing to check it against.`;
  return `${pct(row.hit_rate)} hit rate · ${pct(row.avg_predicted_prob)} average chance · Brier ${row.brier.toFixed(3)} · n=${row.n}`;
}

/** A driver the reader cannot rank: they have no number for this market.
 *
 *  `field` is a `PickCategory["field"]`, so it is one of the seven `p_*`
 *  columns and the lookup cannot wander onto `driver_id` or
 *  `expected_points`. A null there is a market this session does not have,
 *  not a zero, and the driver is left out of that list rather than ranked
 *  last on a number nobody produced. */
const rankable = (row: DriverRow, field: PickCategory["field"]): number | undefined => finite(row[field]);

export interface BuildPicksInput {
  sessionType: SessionType;
  drivers: DriverRow[];
  ledger: TrackRecordResponse | null;
  /** The tier the prediction on screen came from. Its ledger, or none. */
  tier?: string | null;
  /** Drivers who are not in this session. Shown once, below the lists. */
  out?: OutPlayer[];
}

export interface BuiltPicks {
  categories: { category: string; rows: PickRow[] }[];
  out: OutPlayer[];
}

export function buildPicks({ sessionType, drivers, ledger, tier = null, out = [] }: BuildPicksInput): BuiltPicks {
  // A driver named in `out` is out of this ranking, by name, before anything
  // is ranked. The out line below the lists is the only place they appear.
  const outNames = new Set(out.map((p) => p.name));
  const field = drivers.filter((d) => !outNames.has(driverName(d.driver_id)));

  const categories: { category: string; rows: PickRow[] }[] = [];

  for (const label of ORDER) {
    // The projection is not a market; it is built from expected_position and
    // is left out entirely when no driver has one.
    if (label === PROJECTION_CATEGORY) {
      const rows = field
        .map((d) => ({ d, position: finite(d.expected_position) }))
        .filter((x): x is { d: DriverRow; position: number } => x.position !== undefined)
        .sort((a, b) => a.position - b.position)
        .slice(0, MAX_ROWS_PER_CATEGORY)
        .map(({ d, position }) => ({
          key: `${sessionType}-expected_position-${d.driver_id}`,
          name: driverName(d.driver_id),
          team: teamName(d.constructor_id),
          detail: "Expected finish",
          value: position,
          // A position is a place, not a share: kind "projection" is what
          // keeps PicksList from drawing it as a percentage bar. No `margin`:
          // see fact 3 above.
          kind: "projection" as const,
          provenance: "Position the model expects — no error estimate for this yet",
        }));
      if (rows.length > 0) categories.push({ category: label, rows });
      continue;
    }

    const spec = pickCategories[label];
    if (!spec.sessions.includes(sessionType)) continue;

    const rows = field
      .map((d) => ({ d, value: rankable(d, spec.field) }))
      .filter((x): x is { d: DriverRow; value: number } => x.value !== undefined)
      // The model's own number for THIS market, highest first. DNF ranks on
      // p_dnf like any other, so the list reads "most likely to retire".
      .sort((a, b) => b.value - a.value)
      .slice(0, MAX_ROWS_PER_CATEGORY)
      .map(({ d, value }) => ({
        key: `${sessionType}-${spec.market}-${d.driver_id}`,
        name: driverName(d.driver_id),
        team: teamName(d.constructor_id),
        detail: spec.detail,
        value,
        kind: "probability" as const,
        provenance: provenanceFor(spec.market, ledgerFor(ledger, spec.market, tier)),
      }));

    // A market no driver has a number for is not a category. An empty heading
    // would be a claim that the model produced a ranking it did not.
    if (rows.length > 0) categories.push({ category: label, rows });
  }

  return { categories, out };
}
