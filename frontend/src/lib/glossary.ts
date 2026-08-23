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
