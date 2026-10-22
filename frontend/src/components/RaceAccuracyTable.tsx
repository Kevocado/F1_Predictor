import { useState, useCallback, Fragment } from "react";
import type { RaceAccuracyEntry } from "../types";
import { driverName } from "../lib/teamColors";
import { StatusBadge, pctFine } from "../predictor-ui";
import { api } from "../api/client";

const names = (ids: string[]) => ids.map(driverName).join(", ") || "—";

function hitColor(hits: number, of: number): string {
  const frac = hits / of;
  if (frac >= 0.99) return "text-pr-win";
  if (frac >= 0.5) return "text-pr-lean";
  return "text-pr-loss";
}

interface RaceDetail {
  predicted: { driver_id: string; p_win: number }[];
  actual: string[];
}

function detailKey(e: RaceAccuracyEntry) {
  return `${e.season}-${e.round}`;
}

/** Right / Wrong chip: the words carry the meaning, so colour is never the
 *  only signal. */
function VerdictChip({ right }: { right: boolean }) {
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 text-xs font-semibold leading-none ${
        right
          ? "border-pr-win text-pr-win bg-pr-win/10"
          : "border-pr-loss text-pr-loss bg-pr-loss/10"
      }`}
    >
      {right ? "Right" : "Wrong"}
    </span>
  );
}

export function RaceAccuracyTable({ entries }: { entries: RaceAccuracyEntry[] }) {
  if (entries.length === 0) return <p className="py-6 text-sm text-pr-text-dim">No resolved races yet.</p>;
  const [detail, setDetail] = useState<Record<string, RaceDetail | "loading">>({});

  const toggle = useCallback((e: RaceAccuracyEntry) => {
    const key = detailKey(e);
    const current = detail[key];
    if (current === "loading") return;
    if (current) {
      setDetail((d) => {
        const nd = { ...d };
        delete nd[key];
        return nd;
      });
      return;
    }
    setDetail((d) => ({ ...d, [key]: "loading" }));
    api
      .racePrediction(e.season, e.round)
      .then((pred) => {
        const top3 = [...pred.predictions]
          .sort((a, b) => (b.p_win ?? 0) - (a.p_win ?? 0))
          .slice(0, 3)
          .map((d) => ({ driver_id: d.driver_id, p_win: d.p_win ?? 0 }));
        setDetail((d) => ({ ...d, [key]: { predicted: top3, actual: e.podium_actual } }));
      })
      .catch(() => {
        setDetail((d) => {
          const nd = { ...d };
          delete nd[key];
          return nd;
        });
      });
  }, [detail]);

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[40rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-pr-rule text-left font-pr-display text-xs uppercase tracking-wide text-pr-text-dim">
            <th className="py-2 pr-3 font-semibold">Race</th>
            <th className="py-2 pr-3 font-semibold">Predicted winner</th>
            <th className="py-2 pr-3 font-semibold">Actual winner</th>
            <th className="py-2 pr-3 text-right font-semibold">Podium</th>
            <th className="py-2 pr-3 text-right font-semibold">Points</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => {
            const key = detailKey(e);
            const winHit = e.win_hits >= e.win_of;
            const judged = !e.rebuilt;
            const d = detail[key];
            const isOpen = d !== undefined && d !== "loading";
            return (
              <Fragment key={key}>
                <tr key={key} data-testid="race-row" className="border-b border-pr-rule last:border-0">
                  <td className="py-2 pr-3 text-pr-text">
                    <span className="mr-1.5 font-pr-display font-semibold text-pr-text-dim">R{e.round}</span>
                    {e.race_name}
                    {e.rebuilt && (
                      <span className="mt-1 block">
                        <StatusBadge status="rebuilt" moment="the session" />
                      </span>
                    )}
                    <button
                      type="button"
                      aria-expanded={isOpen}
                      onClick={() => toggle(e)}
                      className="ml-2 inline-flex items-center rounded border border-pr-rule bg-pr-panel px-2 py-0.5 text-xs font-semibold text-pr-text hover:bg-pr-panel-2"
                    >
                      {isOpen ? "Hide detail" : "Show detail"}
                    </button>
                  </td>
                  <td className="py-2 pr-3 font-semibold text-pr-text">{names(e.win_predicted)}</td>
                  <td className={`py-2 pr-3 font-semibold ${judged ? (winHit ? "text-pr-win" : "text-pr-loss") : "text-pr-text"}`}>
                    {judged && (winHit ? "✓ " : "✗ ")}
                    {names(e.win_actual)}
                  </td>
                  <td className={`py-2 pr-3 text-right font-semibold tabular-nums ${judged ? hitColor(e.podium_hits, e.podium_of) : "text-pr-text-dim"}`}>
                    {e.podium_hits}/{e.podium_of}
                  </td>
                  <td className={`py-2 pr-3 text-right font-semibold tabular-nums ${judged ? hitColor(e.points_finish_hits, e.points_finish_of) : "text-pr-text-dim"}`}>
                    {e.points_finish_hits}/{e.points_finish_of}
                  </td>
                </tr>
                {isOpen && d && (
                  <tr key={`${key}-detail`} className="border-b border-pr-rule/60 last:border-0">
                    <td colSpan={5} className="py-3 pr-3">
                      <div className="flex flex-wrap gap-6 text-sm">
                        <div>
                          <p className="text-xs font-semibold uppercase tracking-wide text-pr-text-dim mb-1">
                            Predicted top 3
                          </p>
                          {d.predicted.map((p, i) => {
                            const right = d.actual.includes(p.driver_id);
                            return (
                              <div key={p.driver_id} className="flex items-center gap-2 mb-1">
                                <span className="w-4 text-xs text-pr-text-faint">{i + 1}.</span>
                                <span className="text-pr-text">{driverName(p.driver_id)}</span>
                                <span className="text-pr-text-dim tabular-nums">{pctFine(p.p_win)}</span>
                                <VerdictChip right={right} />
                              </div>
                            );
                          })}
                        </div>
                        <div>
                          <p className="text-xs font-semibold uppercase tracking-wide text-pr-text-dim mb-1">
                            Actual top 3
                          </p>
                          <p className="text-pr-text">{names(d.actual)}</p>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
