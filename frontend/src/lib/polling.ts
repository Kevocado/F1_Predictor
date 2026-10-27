/** Polling helpers for the F1 weekend pages. */

export const SESSION_WINDOW_POLL_MS = 60_000;
export const IDLE_POLL_MS = 10 * 60 * 1000;

/** 60 s while a session is live, 10 min otherwise. */
export function pollIntervalMs(live: boolean): number {
  return live ? SESSION_WINDOW_POLL_MS : IDLE_POLL_MS;
}

export function selectPollInterval(live: boolean): number {
  return pollIntervalMs(live);
}

export function getPollInterval(live: boolean): number {
  return pollIntervalMs(live);
}

/** A weekend is "live" when now sits inside the window around its race. */
export function isSessionWindow(raceDatetime: string | null, now: Date = new Date()): boolean {
  if (!raceDatetime) return false;
  const race = Date.parse(raceDatetime);
  if (Number.isNaN(race)) return false;
  const diff = race - now.getTime();
  // Qualifying is roughly a day before the race; keep polling fast from 48h
  // before the race until 6h after it.
  return diff >= -6 * 60 * 60 * 1000 && diff <= 48 * 60 * 60 * 1000;
}

export function isLiveWeekend(raceDatetime: string | null, now: Date = new Date()): boolean {
  return isSessionWindow(raceDatetime, now);
}
