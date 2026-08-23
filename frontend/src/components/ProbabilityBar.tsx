import { pct } from "../lib/format";

interface Props {
  value: number;
  color?: string;
  digits?: number;
}

export function ProbabilityBar({ value, color = "var(--color-f1-red)", digits = 0 }: Props) {
  const widthPct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-f1-700">
        <div className="h-full rounded-full" style={{ width: `${widthPct}%`, background: color }} />
      </div>
      <span className="w-10 shrink-0 text-right text-xs tabular-nums text-f1-text-dim">{pct(value, digits)}</span>
    </div>
  );
}
