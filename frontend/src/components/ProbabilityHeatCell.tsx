import { pctFine } from "../predictor-ui";

interface Props {
  value: number;
  /** 0-1, this value relative to the strongest entrant in the same column,
   * not an absolute scale: "9.9%" can be the best shot in a 20-car field,
   * and shading against the field says so at a glance. One hue only. */
  intensity: number;
}

export function ProbabilityHeatCell({ value, intensity }: Props) {
  const clamped = Math.max(0, Math.min(1, intensity));
  const alphaPct = Math.round(8 + 55 * clamped);
  return (
    <div
      className={`rounded-pr px-2 py-1.5 text-center text-sm font-semibold tabular-nums ${clamped > 0.6 ? "text-pr-text" : "text-pr-text-dim"}`}
      style={{ background: `color-mix(in srgb, var(--color-pr-accent) ${alphaPct}%, transparent)` }}
    >
      {pctFine(value)}
    </div>
  );
}
