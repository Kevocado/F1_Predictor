import { useCallback, useEffect, useState } from "react";
import { ExplainerPanel, type Explanation } from "../predictor-ui";

/**
 * The "In plain English" panel for one F1 session, collapsed to its headline.
 *
 * A race page is already dense — grid, tower, form, standings — so the race
 * story is one line until the reader asks for it. Collapsing it is the point,
 * not a compromise: the story is worth reading but not worth the space unless
 * it was chosen.
 *
 * `fetcher` is optional. With no explainer deployed, or a session the service
 * has nothing for, this renders nothing rather than an empty box.
 */
export function SessionSummaryPanel({
  sessionId,
  fetcher,
  sport = "f1",
  className = "",
}: {
  sessionId: string;
  fetcher?: (sport: string, id: string) => Promise<Explanation>;
  sport?: string;
  className?: string;
}) {
  const [data, setData] = useState<Explanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  const load = useCallback(() => {
    if (!fetcher) return;
    let cancelled = false;
    setLoading(true);
    setError(false);
    fetcher(sport, sessionId)
      .then((r) => { if (!cancelled) setData(r); })
      .catch(() => { if (!cancelled) { setData(null); setError(true); } })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [fetcher, sport, sessionId]);

  useEffect(() => load(), [load]);

  if (!fetcher) return null;
  return (
    <div className={className}>
      <ExplainerPanel
        data={data}
        loading={loading}
        error={error}
        onRetry={() => load()}
        collapsed
        // The moment a session pick has to beat is the session, never a kickoff.
        moment="the session"
      />
    </div>
  );
}
