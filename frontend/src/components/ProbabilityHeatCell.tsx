import { pct } from "../lib/format";

interface Props {
  value: number;
  /** 0-1, this value relative to the strongest driver in the same market
   * for the SAME race — not an absolute scale. A raw "9.9%" reads as low
   * out of 100, even when it's actually the best in a 22-driver field;
   * shading relative to the field gives the number context at a glance. */
  intensity: number;
  color: string;
  digits?: number;
}

export function ProbabilityHeatCell({ value, intensity, color, digits = 0 }: Props) {
  const clamped = Math.max(0, Math.min(1, intensity));
  const alphaPct = Math.round(10 + 65 * clamped);
  const textStrong = clamped > 0.6;

  return (
    <div
      className="rounded-md px-2 py-1.5 text-center text-sm font-semibold tabular-nums"
      style={{
        background: `color-mix(in srgb, ${color} ${alphaPct}%, transparent)`,
        color: textStrong ? "#fff" : "var(--color-f1-text-dim)",
      }}
    >
      {pct(value, digits)}
    </div>
  );
}
