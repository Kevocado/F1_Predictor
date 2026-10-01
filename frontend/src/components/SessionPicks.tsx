/**
 * The model's own top calls for one session, ranked per market.
 *
 * The list itself is the shipped `PicksList` from the vendored hub package
 * (predictor-hub #63, re-synced to #69). This file is only the F1 dialect: it
 * decides which markets a session type has and takes each row's number
 * straight from the API. It composes no confidence language of its own, so
 * there is no sentence here that could claim a price, an edge or a guarantee —
 * no odds feed exists in this repo.
 *
 * Kevin, 2026-10-01: a top call is simple — the driver, the team and the
 * prediction. So a row here is `{key,name,team,detail,value,kind}` and nothing
 * else. The per-market ledger line (`n`, `brier`, `hit_rate`,
 * `avg_predicted_prob`, or "no graded record yet") and the position row's
 * "no error estimate" note were both true and both removed: a market with no
 * graded record now renders exactly like one that has 286 of them.
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
 *    `margin`, and since 2026-10-01 it carries no sentence about the absence of
 *    one either: it shows its position, which is what a top call is.
 *
 * 4. **Provenance is gone, and with it the ledger's per-market figures.**
 *    `GET /track-record` -> `store.get_session_track_record` still groups by
 *    (session_type, tier, market) and still reports `n`, `brier`, `hit_rate` and
 *    `avg_predicted_prob`, and no row reads any of it now. The rule that
 *    enforced — a market with no graded record borrows nothing, and invents no
 *    `n` — is now true a fortiori, since no row quotes a number from the ledger
 *    at all. (Measured in `data/tracking.db`: 1320 resolved race rows and 132
 *    qualifying rows; 0 resolved for sprint and sprint qualifying, which used
 *    to read "no graded record yet" and now simply reads as any other market.)
 */
import { MAX_ROWS_PER_CATEGORY, type OutPlayer, type PickRow } from "../predictor-ui";
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

/**
 * The per-market ledger lookup went with the row text (Kevin, 2026-10-01).
 *
 *  It existed only to write `provenanceFor`, which put each market's `n`,
 *  `brier`, `hit_rate` and `avg_predicted_prob` on the row — or, for a market
 *  with no ledger row, said "no graded record yet". A row is now the driver, the
 *  team and the prediction, so `ledger` and `tier` are still accepted on
 *  `BuildPicksInput` (callers pass them; the shipped ledger is unchanged) but
 *  nothing reads them.
 *
 *  The rule that lookup enforced is still true and is asserted elsewhere: a
 *  market with no graded record never borrowed another market's numbers,
 *  because no row carries a number from the ledger at all. A sprint weekend,
 *  which has no resolved rows, therefore renders exactly like a race weekend.
 *
 *  Removed here: `ledgerFor()`, the `tier` re-check, and the `pct` import.
 */

/**
 * A driver the reader cannot rank: they have no number for this market.
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
  /** No longer read: kept so callers need not change. Since 2026-10-01 no row
   *  quotes a ledger figure, so a `null` ledger changes nothing on the page. */
  ledger: TrackRecordResponse | null;
  /** The tier the prediction on screen came from. No longer read: kept so
   *  callers need not change, since no row quotes a ledger figure. */
  tier?: string | null;
  /** Drivers who are not in this session. Shown once, below the lists. */
  out?: OutPlayer[];
}

export interface BuiltPicks {
  categories: { category: string; rows: PickRow[] }[];
  out: OutPlayer[];
}

export function buildPicks({ sessionType, drivers, out = [] }: BuildPicksInput): BuiltPicks {
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
          // no per-driver position MAE exists (see fact 3), and the row shows
          // its figure rather than a sentence about the absence of one.
          kind: "projection" as const,
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
      }));

    // A market no driver has a number for is not a category. An empty heading
    // would be a claim that the model produced a ranking it did not.
    if (rows.length > 0) categories.push({ category: label, rows });
  }

  return { categories, out };
}
