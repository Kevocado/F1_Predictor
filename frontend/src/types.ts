// Mirrors src/f1_predictor/api/schemas.py exactly — keep these in sync by hand.

export interface RaceSummary {
  season: number;
  round: number;
  race_name: string;
  circuit_name: string;
  race_datetime: string | null;
  is_sprint_weekend: boolean;
  completed: boolean;
}

export interface DriverPrediction {
  driver_id: string;
  constructor_id: string | null;
  p_win: number;
  p_podium: number;
  p_points_finish: number;
  p_dnf: number;
  expected_position: number;
  expected_points: number;
  actual_position: number | null;
  actual_dnf: boolean | null;
}

export interface FeatureContribution {
  feature: string;
  value: number | null;
  contribution: number;
}

export interface ExplainResponse {
  season: number;
  round: number;
  driver_id: string;
  candidate: string;
  strength_contributors: FeatureContribution[];
  dnf_contributors: FeatureContribution[];
}

export type Tier = "pre_weekend" | "post_practice" | "post_qualifying";
// "tracked": snapshotted before the session. "rebuilt": a stored snapshot
// written after it. "backtest": a no-lookahead replay of a session that was
// never snapshotted. Only "tracked" ever counts toward the track record.
export type PredictionSource = "live" | "tracked" | "rebuilt" | "backtest";

/** One season the model asked for and did not get. `reason` is never empty:
 *  a skip with no explanation is the silence this replaced. */
export interface MissingSeason {
  season: number;
  reason: string;
}

/** Whether this forecast was built on every season the model asked for.
 *  `complete` is true only when `missing_seasons` is empty — the lists are here
 *  so a reader can check that rather than take the boolean on faith. */
export interface HistoryCoverage {
  complete: boolean;
  seasons_requested: number[];
  seasons_loaded: number[];
  missing_seasons: MissingSeason[];
}

export interface RacePredictionResponse {
  season: number;
  round: number;
  race_name: string;
  tier: Tier;
  source: PredictionSource;
  predictions: DriverPrediction[];
  /** Absent means "not reported", not "complete": a stored snapshot or a
   *  backtest was built from a history load this request never made, so
   *  nothing here can speak for it. Never defaulted to a reassuring value. */
  history?: HistoryCoverage | null;
}

export interface SessionDriverPrediction {
  driver_id: string;
  constructor_id: string | null;
  p_pole: number | null;
  p_top_3: number | null;
  p_top_10: number | null;
  p_win: number | null;
  p_podium: number | null;
  p_points_finish: number | null;
  p_dnf: number | null;
  expected_position: number;
  actual_position: number | null;
  actual_dnf: boolean | null;
}

export type SessionType = "sprint_qualifying" | "qualifying" | "sprint" | "race";

export interface SessionPredictionResponse {
  season: number;
  round: number;
  race_name: string;
  session_type: SessionType;
  tier: string;
  source: string;
  predictions: SessionDriverPrediction[];
}

export interface ChampionshipEntry {
  entity_id: string;
  win_prob: number;
  top3_prob: number;
  expected_final_points: number;
  expected_final_rank: number;
}

export interface ChampionshipResponse {
  season: number;
  as_of_round: number;
  n_trials: number;
  standings: ChampionshipEntry[];
}

export interface TrackRecordEntry {
  tier: string;
  market: string;
  n: number;
  brier: number;
  hit_rate: number;
  avg_predicted_prob: number;
}

/** The secondary figure: the same ledger over the picks made BEFORE the
 *  session started. A grand prix is a weekend, not a kickoff — hence "session". */
export interface TrackRecordSubset {
  n_resolved: number;
  by_market: TrackRecordEntry[];
}

/** One recorded pick, with when it was made. Both timestamps are sent so the
 *  `made_before_session` comparison can be checked rather than trusted. */
export interface TrackRecordPick {
  season: number;
  round: number;
  race_name: string | null;
  session_type: string;
  tier: string;
  driver_id: string;
  market: string;
  predicted_prob: number;
  actual_outcome: number;
  made_before_session: boolean;
  snapshotted_at: string;
  session_time: string;
  /** False for a rerun that lost the earliest-pick contest. */
  counted: boolean;
}

export interface TrackRecordResponse {
  /**
   * The HEADLINE: every counted pick, one per (session, driver, market), the
   * earliest recorded, whenever it was made. Before 2026-10-01 it was the
   * sessions every row of which was snapshotted before it started, which on
   * the shipped database withheld 1,056 of 1,452 recorded picks. Same name,
   * fuller record.
   */
  n_resolved: number;
  by_market: TrackRecordEntry[];
  /**
   * The pre-session subset beside the headline: what the model would have
   * said on the weekend. The figure to read for live performance.
   *
   * Optional on the wire even though the backend always sends it: during a
   * rolling deploy the old API shape is still in front of this bundle for a
   * few minutes, and a missing subset is read as "not measured" rather than
   * crash a page. `n_pre_session` falls back to 0 and the table is omitted.
   */
  pre_session?: TrackRecordSubset;
  /** `n_resolved === n_pre_session + n_post_session_picks`. */
  n_pre_session?: number;
  n_post_session_picks?: number;
  /**
   * RENAMED IN MEANING, name and unit kept: it used to count sessions LEFT OUT
   * of the record. It is now how many sessions had at least one counted pick
   * recorded at or after their own start — the disclosure, not the exclusion.
   */
  n_rebuilt_sessions?: number;
  /** Every recorded pick, counted or not, with its own timestamps and label. */
  per_pick?: TrackRecordPick[];
}

export interface RaceAccuracyEntry {
  season: number;
  round: number;
  race_name: string;
  tier: string;
  /** Snapshot written after the session ran: shown, never counted. */
  rebuilt: boolean;
  win_predicted: string[];
  win_actual: string[];
  win_hits: number;
  win_of: number;
  podium_predicted: string[];
  podium_actual: string[];
  podium_hits: number;
  podium_of: number;
  points_finish_predicted: string[];
  points_finish_actual: string[];
  points_finish_hits: number;
  points_finish_of: number;
}

export interface RetrainResponse {
  status: string;
  trained_at: string;
  race_outcome_candidate: string;
  race_outcome_candidate_metrics: Record<string, number>;
}
