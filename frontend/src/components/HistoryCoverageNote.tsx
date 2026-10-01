import type { HistoryCoverage } from "../types";

/**
 * A season the model asked for and never got.
 *
 * The backend skips a season rather than failing the whole run when a fetch is
 * rate-limited, so the forecast a reader is looking at was produced from a
 * SHORTER history than the model normally uses — circuit form is an expanding
 * mean over whatever arrived, and a circuit is raced about once a season, so a
 * missing season can be the difference between knowing a circuit and not.
 *
 * Two things this deliberately does not say. It does not say the pick is
 * invalid, wrong, or should be ignored: the model genuinely produced a pick
 * from the data it had, and that pick is the pick. And it does not colour the
 * badge red — nothing failed. A red badge would invent a fault the payload does
 * not contain, which is the mistake predictor-ui's `unverified` status already
 * documents for itself ("the schedule never supplied the start time, so
 * nothing went wrong. A red badge would invent a failure that isn't there").
 *
 * Wording follows the same shape as StatusBadge's neutral statuses: words that
 * carry the meaning, colour that only reinforces them.
 */
export function HistoryCoverageNote({ history }: { history: HistoryCoverage | null | undefined }) {
  // Only a known-short window gets a notice. `complete: true` says every
  // season loaded; `null` says this response has no history load behind it at
  // all, and neither is a reason to interrupt a reader.
  if (!history || history.complete) return null;

  const missing = history.missing_seasons.map((m) => m.season);
  const seasons = missing.map((s) => String(s)).join(", ");
  const loaded = history.seasons_loaded.length;

  return (
    <div className="mb-4 rounded-pr border border-pr-rule bg-pr-panel-2 p-3">
      <span className="mb-1 inline-flex items-center whitespace-nowrap rounded-pr border border-pr-rule px-1.5 py-0.5 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim">
        Shorter history
      </span>
      <p className="text-xs leading-relaxed text-pr-text-dim">
        The {seasons} season{missing.length > 1 ? "s" : ""} couldn&apos;t be loaded, so this forecast
        rests on a shorter history than usual — {loaded} of the {history.seasons_requested.length}{" "}
        seasons the model normally reads. The model still produced a pick from the{" "}
        {loaded === 1 ? "season" : "seasons"} it did get, so the numbers below are its real output,
        just informed by less history than a normal weekend&apos;s would be.
      </p>
    </div>
  );
}