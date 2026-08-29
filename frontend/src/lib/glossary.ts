export const TIER_LABELS: Record<string, string> = {
  pre_weekend: "Pre-weekend",
  post_practice: "Post-practice",
  post_qualifying: "Post-qualifying",
};

export const TIER_DESCRIPTIONS: Record<string, string> = {
  pre_weekend:
    "Before any on-track session — based on season form, career driver/car strength, and circuit history only.",
  post_practice:
    "After free practice — sharpened with practice pace once that data is available.",
  post_qualifying:
    "After qualifying — the fullest pre-race information state, including grid position, the single strongest predictor of a race result.",
};

export const SOURCE_DESCRIPTIONS: Record<string, string> = {
  live: "Computed fresh, right now, from the latest available data.",
  tracked: "Served from the snapshot recorded before this race happened — the honest, un-hindsight-biased prediction.",
  backtest:
    "This race happened before a snapshot was ever recorded, so this is an honest reconstruction: retrained using only data available before it, exactly as if predicted live.",
};

export const MARKET_LABELS: Record<string, string> = {
  win: "Win",
  podium: "Podium",
  points_finish: "Points",
  dnf: "DNF",
};

// Mirrors src/f1_predictor/features/build.py::FEATURE_COLUMNS — human
// labels for the explain ribbon. A feature missing from this map falls
// back to its raw name rather than crashing.
export const FEATURE_LABELS: Record<string, string> = {
  elo_pre_race: "Driver Elo rating",
  team_strength_pre_race: "Constructor strength (season)",
  team_form_avg_position_3: "Team's avg. finish (last 3 races)",
  team_form_points_3: "Team's avg. points (last 3 races)",
  form_avg_position_3: "Driver's avg. finish (last 3 races)",
  form_avg_points_3: "Driver's avg. points (last 3 races)",
  form_dnf_rate_3: "Driver's DNF rate (last 3 races)",
  form_avg_position_5: "Driver's avg. finish (last 5 races)",
  form_avg_points_5: "Driver's avg. points (last 5 races)",
  form_dnf_rate_5: "Driver's DNF rate (last 5 races)",
  form_avg_position_10: "Driver's avg. finish (last 10 races)",
  form_avg_points_10: "Driver's avg. points (last 10 races)",
  form_dnf_rate_10: "Driver's DNF rate (last 10 races)",
  driver_circuit_avg_position: "Driver's history at this circuit",
  driver_circuit_avg_points: "Driver's points history at this circuit",
  constructor_circuit_avg_position: "Team's history at this circuit",
  circuit_dnf_rate: "This circuit's DNF rate",
  grid: "Grid position",
  quali_position: "Qualifying position",
  quali_gap_to_pole: "Qualifying gap to pole",
  temp_max_c: "Forecast max temperature",
  precipitation_mm: "Forecast precipitation",
  wind_max_kph: "Forecast wind speed",
};
