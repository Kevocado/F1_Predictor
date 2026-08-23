import { teamColor, teamName } from "../lib/teamColors";

export function TeamBadge({ constructorId }: { constructorId: string | null | undefined }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-f1-text-dim">
      <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: teamColor(constructorId) }} />
      {teamName(constructorId)}
    </span>
  );
}
