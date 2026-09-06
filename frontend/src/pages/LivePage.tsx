import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { LiveCurrentResponse } from "../types";
import { driverName, teamColor } from "../lib/teamColors";
import { TeamBadge } from "../components/TeamBadge";
import { ProbabilityHeatCell } from "../components/ProbabilityHeatCell";

const POLL_INTERVAL_MS = 10_000;

interface LivePrediction {
  driver_id: string;
  p_win: number;
  p_podium: number;
  constructor_id: string | null;
}

export function LivePage() {
  const [data, setData] = useState<LiveCurrentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      api
        .liveCurrent()
        .then((d) => !cancelled && setData(d))
        .catch((e) => !cancelled && setError(e.message));
    };
    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const live = data as (LiveCurrentResponse & {
    season?: number;
    round?: number;
    race_name?: string;
    lap_number?: number;
    updated_at?: string;
    predictions?: LivePrediction[];
  }) | null;

  return (
    <div className="clip-corner-lg rounded-lg border border-f1-border bg-f1-850/60 p-5">
      <div className="mb-4">
        <h2 className="font-display text-lg font-bold text-f1-text">Live Race</h2>
        <p className="max-w-2xl text-xs text-f1-text-faint">
          Polls every 10s. Trained on lap-by-lap historical replay (2022–present); updates from OpenF1's live
          feed once a race actually goes green.
        </p>
      </div>

      {error && <div className="rounded-lg border border-dnf/30 bg-dnf/10 p-4 text-sm text-dnf">{error}</div>}

      {!error && !live && <div className="animate-pulse text-sm text-f1-text-faint">Checking for a live session…</div>}

      {live && !live.live && live.blocked && (
        <div className="rounded-lg border border-dnf/30 bg-dnf/10 p-4 text-sm text-dnf">
          <p className="font-medium">Live predictions unavailable — OpenF1 is blocking free access right now.</p>
          <p className="mt-1 text-xs text-f1-text-faint">
            OpenF1 now requires a paid API key for any access (including historical queries) for the entire
            duration of a live session. A race may genuinely be live — we just can't reach the feed for it
            without a key.
          </p>
        </div>
      )}

      {live && !live.live && !live.blocked && (
        <p className="py-10 text-center text-sm text-f1-text-faint">
          No race is live right now. Check back during a Grand Prix — this updates automatically once one starts.
        </p>
      )}

      {live?.live && live.predictions && (
        <>
          <div className="mb-3 flex items-center justify-between text-xs text-f1-text-dim">
            <span>
              {live.race_name} · Round {live.round} · {live.season}
            </span>
            <span>Lap {live.lap_number}</span>
          </div>
          <table className="w-full min-w-[420px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-f1-border text-left text-[11px] uppercase tracking-wide text-f1-text-faint">
                <th className="py-2 pr-3 font-medium">#</th>
                <th className="py-2 pr-3 font-medium">Driver</th>
                <th className="py-2 pr-3 font-medium">Win</th>
                <th className="py-2 pr-3 font-medium">Podium</th>
              </tr>
            </thead>
            <tbody>
              {live.predictions.map((p, i) => (
                <tr
                  key={p.driver_id}
                  className="border-b border-f1-border/60 last:border-0"
                  style={{ borderLeft: `2px solid ${teamColor(p.constructor_id)}` }}
                >
                  <td className="py-2 pl-2 pr-3 tabular-nums text-f1-text-faint">{i + 1}</td>
                  <td className="py-2 pr-3">
                    <div className="font-medium text-f1-text">{driverName(p.driver_id)}</div>
                    <TeamBadge constructorId={p.constructor_id} />
                  </td>
                  <td className="py-1.5 pr-2">
                    <ProbabilityHeatCell value={p.p_win} intensity={p.p_win} color="var(--color-f1-red)" digits={1} />
                  </td>
                  <td className="py-1.5 pr-2">
                    <ProbabilityHeatCell value={p.p_podium} intensity={p.p_podium} color="var(--color-podium)" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
