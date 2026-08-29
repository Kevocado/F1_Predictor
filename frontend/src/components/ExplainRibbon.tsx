import type { FeatureContribution } from "../types";
import { FEATURE_LABELS } from "../lib/glossary";

interface Props {
  title: string;
  note?: string;
  contributors: FeatureContribution[] | null;
  loading: boolean;
  error: string | null;
}

export function ExplainRibbon({ title, note, contributors, loading, error }: Props) {
  const maxAbs = contributors?.length ? Math.max(...contributors.map((c) => Math.abs(c.contribution)), 1e-6) : 1;

  return (
    <div className="px-4 py-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-f1-text-dim">{title}</div>
      {note && <p className="mb-2.5 text-[11px] text-f1-text-faint">{note}</p>}

      {loading && <div className="animate-pulse text-xs text-f1-text-faint">Computing explanation…</div>}
      {error && <div className="text-xs text-dnf">{error}</div>}

      {contributors && !loading && !error && (
        <div className="flex flex-col gap-2">
          {contributors.map((c) => {
            const halfWidthPct = (Math.abs(c.contribution) / maxAbs) * 50;
            const positive = c.contribution >= 0;
            return (
              <div key={c.feature} className="grid grid-cols-[1fr_auto] items-center gap-3 text-xs">
                <div>
                  <div className="mb-0.5 text-f1-text-dim">{FEATURE_LABELS[c.feature] ?? c.feature}</div>
                  <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-f1-700">
                    <div className="absolute left-1/2 h-full w-px bg-f1-600" />
                    <div
                      className="absolute h-full rounded-full"
                      style={
                        positive
                          ? { left: "50%", width: `${halfWidthPct}%`, background: "var(--color-win)" }
                          : { right: "50%", width: `${halfWidthPct}%`, background: "var(--color-dnf)" }
                      }
                    />
                  </div>
                </div>
                <div className="w-28 shrink-0 text-right tabular-nums">
                  <span className="text-f1-text-faint">{c.value != null ? c.value.toFixed(2) : "—"}</span>
                  <span className={`ml-1.5 font-semibold ${positive ? "text-win" : "text-dnf"}`}>
                    {positive ? "+" : ""}
                    {c.contribution.toFixed(3)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
