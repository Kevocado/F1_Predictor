import { TIER_DESCRIPTIONS, TIER_LABELS } from "../lib/glossary";
import { InfoTooltip } from "./InfoTooltip";

export function TierBadge({ tier }: { tier: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-f1-border bg-f1-800 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-f1-text-dim">
      {TIER_LABELS[tier] ?? tier}
      <InfoTooltip text={TIER_DESCRIPTIONS[tier] ?? tier} />
    </span>
  );
}
