export function pct(value: number, digits = 0): string {
  if (Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number, digits = 1): string {
  if (Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function formatDate(iso: string | null): string {
  if (!iso) return "TBD";
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export function positionSuffix(pos: number): string {
  if (pos % 10 === 1 && pos % 100 !== 11) return "st";
  if (pos % 10 === 2 && pos % 100 !== 12) return "nd";
  if (pos % 10 === 3 && pos % 100 !== 13) return "rd";
  return "th";
}
