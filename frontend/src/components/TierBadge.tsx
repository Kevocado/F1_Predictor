import { TIER_DESCRIPTIONS, TIER_LABELS } from "../lib/glossary";
import { InfoTooltip } from "./InfoTooltip";

export function TierBadge({ tier }: { tier: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-pr border border-pr-rule bg-pr-panel-2 px-2 py-1 font-pr-display text-xs font-semibold uppercase tracking-wide text-pr-text-dim">
      {TIER_LABELS[tier] ?? tier}
      <InfoTooltip text={TIER_DESCRIPTIONS[tier] ?? tier} />
    </span>
  );
}
