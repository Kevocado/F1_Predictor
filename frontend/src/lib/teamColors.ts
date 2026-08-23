// Real-world team livery colors, keyed on jolpica's constructor_id (the
// same id the backend uses everywhere) — confirmed against the live 2026
// grid. A constructor not in this map (future entrant not seen yet) falls
// back to a neutral grey rather than crashing.

export const TEAM_COLORS: Record<string, string> = {
  mercedes: "#27F4D2",
  ferrari: "#E8002D",
  mclaren: "#FF8000",
  red_bull: "#3671C6",
  rb: "#6692FF",
  alpine: "#0093CC",
  haas: "#B6BABD",
  audi: "#B00020",
  williams: "#64C4FF",
  aston_martin: "#229971",
  cadillac: "#0A2540",
};

export const TEAM_NAMES: Record<string, string> = {
  mercedes: "Mercedes",
  ferrari: "Ferrari",
  mclaren: "McLaren",
  red_bull: "Red Bull",
  rb: "RB",
  alpine: "Alpine",
  haas: "Haas",
  audi: "Audi",
  williams: "Williams",
  aston_martin: "Aston Martin",
  cadillac: "Cadillac",
};

const FALLBACK_COLOR = "#7a7f8a";

export function teamColor(constructorId: string | null | undefined): string {
  if (!constructorId) return FALLBACK_COLOR;
  return TEAM_COLORS[constructorId] ?? FALLBACK_COLOR;
}

export function teamName(constructorId: string | null | undefined): string {
  if (!constructorId) return "—";
  return TEAM_NAMES[constructorId] ?? constructorId;
}

export function driverName(driverId: string): string {
  return driverId
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
