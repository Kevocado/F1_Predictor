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

export type Tier = "pre_weekend" | "post_practice" | "post_qualifying";
export type PredictionSource = "live" | "tracked" | "backtest";

export interface RacePredictionResponse {
  season: number;
  round: number;
  race_name: string;
  tier: Tier;
  source: PredictionSource;
  predictions: DriverPrediction[];
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

export interface TrackRecordResponse {
  n_resolved: number;
  by_market: TrackRecordEntry[];
}

export interface LiveCurrentResponse {
  live: boolean;
}

export interface RetrainResponse {
  status: string;
  trained_at: string;
  race_outcome_candidate: string;
  race_outcome_candidate_metrics: Record<string, number>;
}
