import type { FeatureContribution } from "../types";
import { FEATURE_LABELS } from "../lib/glossary";
import { stat } from "../predictor-ui";

interface Props {
  title: string;
  note?: string;
  contributors: FeatureContribution[] | null;
  loading: boolean;
  error: string | null;
}

// What pushes a prediction up or down, in words: the raw model contribution
// (a log-odds fraction) means nothing to a fan, so the bar carries size and
// the word carries direction.
export function ExplainRibbon({ title, note, contributors, loading, error }: Props) {
  const maxAbs = contributors?.length ? Math.max(...contributors.map((c) => Math.abs(c.contribution)), 1e-6) : 1;

  return (
    <div className="px-4 py-3">
      <div className="mb-1 font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text">{title}</div>
      {note && <p className="mb-2.5 text-xs text-pr-text-dim">{note}</p>}

      {loading && <div role="status" className="text-xs text-pr-text-dim">Working out what's driving it…</div>}
      {error && <div role="alert" className="text-xs text-pr-loss">{error}</div>}

      {contributors && !loading && !error && (
        <ul className="flex flex-col gap-2">
          {contributors.map((c) => {
            const halfWidthPct = (Math.abs(c.contribution) / maxAbs) * 50;
            const positive = c.contribution >= 0;
            return (
              <li key={c.feature} className="grid grid-cols-[1fr_auto] items-center gap-3 text-xs">
                <div>
                  <div className="mb-1 text-pr-text">
                    {FEATURE_LABELS[c.feature] ?? c.feature}
                    {c.value != null && <span className="ml-1.5 text-pr-text-dim">({stat(c.value)})</span>}
                  </div>
                  <div aria-hidden="true" className="relative h-1.5 w-full overflow-hidden rounded-full bg-pr-panel-2">
                    <div className="absolute left-1/2 h-full w-px bg-pr-rule" />
                    <div
                      className={`absolute h-full rounded-full ${positive ? "bg-pr-win" : "bg-pr-loss"}`}
                      style={positive ? { left: "50%", width: `${halfWidthPct}%` } : { right: "50%", width: `${halfWidthPct}%` }}
                    />
                  </div>
                </div>
                <span className={`w-14 shrink-0 text-right font-semibold ${positive ? "text-pr-win" : "text-pr-loss"}`}>{positive ? "Raises" : "Lowers"}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
