"""Collect Kalshi's F1 markets as a free, public record of what the market thought.

Kalshi lists one binary contract per DRIVER per race -- "Main Race: Max
Verstappen wins" -- plus a podium ladder, a top-10 ladder, a pole-position
contract, a season-long championship contract, and a set of F1 contracts the
model publishes no probability for (fastest lap, constructors, awards,
next-year questions). Every contract is priced in dollars in [0, 1] (0.62 =
62%). Reading them needs no account, key or credits.

Snapshots taken before the race are the "lines" we had no historical source
for; the same contracts settle afterwards with a yes/no `result`, so each
snapshot is its own labelled outcome and needs no results join to be scored.

The paper ledger reads these rows BY NAME -- `snapshot_ts`, `series`, `stat`,
`ticker`, `event_ticker`, `player`, `title`, `line`, `yes_bid`, `yes_ask`,
`no_bid`, `no_ask`, `last`, `volume`, `open_interest`, `status`, `result`,
`settlement_value`, `occurrence`, `close_time` -- so the keys are exactly the
ones the NFL and PL collectors write. A renamed field is a row that is
silently skipped, not a loud failure. `title` is the load-bearing one: the
ledger reads the DRIVER out of the contract's own title, so it is Kalshi's
title, verbatim.

`floor_strike` is null on every F1 contract -- there is no threshold to
strike -- so `line` is null. A missing or zero bid/ask means no quote, not a
0% market.

`occurrence_datetime` is the session start, and Kalshi's own copy of it runs
about SIX HOURS LATE: for the Singapore, Dutch and Azerbaijan races Kalshi's
own finalisation timestamp (`close_time`) is published BEFORE its stated
occurrence, which cannot be true of a race start. The value is recorded EXACTLY
as Kalshi publishes it, uncorrected. The paper ledger compares the earlier of
Kalshi's occurrence and the session time the model snapshot knows, so it needs
the raw value to reason about, and a "corrected" one would be a guess dressed
up as a fact.
"""
from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

#: The stat a Kalshi F1 series the model cannot price is recorded under.
#: The paper ledger skips such a series by this exact name
#: (`f1_unpriceable_series` in predictor-hub's services/paper/paper/log.py), so
#: it can COUNT the series -- "this exists and we cannot price it" -- instead
#: of the series silently vanishing.
UNPRICEABLE = "f1_unpriceable_series"

#: Kalshi series ticker -> the market the paper ledger prices it as.
#:
#: The five priceable ones are the ledger's own F1 segments, copied from
#: predictor-hub's `F1_SERIES_MARKETS` (services/paper/paper/sources.py) --
#: the names the ledger's `F1_MARKETS` table and its segment allow-list use.
#: Everything else Kalshi lists under KXF1* is collected too, marked
#: UNPRICEABLE, so the ledger can count it. A series added to Kalshi later
#: needs a line here; without one it is not collected at all.
SERIES = {
    # -- the five the model publishes a probability for --
    "KXF1": "championship",              # "Will <driver> win the F1 Drivers Championship?"
    "KXF1RACE": "race_win",              # "Main Race: <driver> wins"
    "KXF1RACEPODIUM": "race_podium",    # "... to finish top 3"
    "KXF1TOP10": "top10",                # "... to finish top 10"
    "KXF1POLE": "pole",                  # "<driver> is awarded Pole Position"
    # -- the rest of Kalshi's F1 series: collected, counted, not priced --
    "KXF1ACTION": UNPRICEABLE,           # FIA Action of the Year award
    "KXF1BIGGESTMOVER": UNPRICEABLE,
    "KXF1CHINA": UNPRICEABLE,
    "KXF1CONSTRUCTORS": UNPRICEABLE,
    "KXF1DELAY": UNPRICEABLE,
    "KXF1FASTESTLAP": UNPRICEABLE,
    "KXF1FASTLAP": UNPRICEABLE,
    "KXF1FASTPIT": UNPRICEABLE,
    "KXF1H2H": UNPRICEABLE,
    "KXF1NEXTTEAM": UNPRICEABLE,
    "KXF1OCCUR": UNPRICEABLE,            # "does the race happen"
    "KXF1POLEPOSITION": UNPRICEABLE,
    "KXF1QUALIFY": UNPRICEABLE,
    "KXF1RACESPRINT": UNPRICEABLE,
    "KXF1RACETOP10": UNPRICEABLE,
    "KXF1RACETOP5": UNPRICEABLE,
    "KXF1RACETOPX": UNPRICEABLE,
    "KXF1RETIRE": UNPRICEABLE,
    "KXF1SPRINTFASTLAP": UNPRICEABLE,
    "KXF1SPRINTPOLE": UNPRICEABLE,
    "KXF1SPRINTTOP10": UNPRICEABLE,
    "KXF1SPRINTTOP5": UNPRICEABLE,
    "KXF1SPRINTTOPCONSTRUCTOR": UNPRICEABLE,
    "KXF1TOP5": UNPRICEABLE,
    "KXF1TOPCONSTRUCTOR": UNPRICEABLE,
}
LINES_DIR = Path("data/lines")
PAGE_LIMIT = 200
SETTLED_LOOKBACK_DAYS = 4


def _dollars(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v


def normalize(market: dict, series: str, snapshot_ts: str) -> dict:
    """One flat row per market. Pure; the unit of everything downstream."""
    title = market.get("title") or ""
    return {
        "snapshot_ts": snapshot_ts,
        "series": series,
        "stat": SERIES.get(series),
        "ticker": market.get("ticker"),
        "event_ticker": market.get("event_ticker"),
        # An F1 contract is about a DRIVER, not a player prop, and the driver
        # is read out of the title by the ledger -- so this stays empty rather
        # than being a second, differently-spelled copy of the same fact.
        "player": None,
        # Kalshi's own title, verbatim. The ledger reads the driver out of it
        # ("Main Race: Max Verstappen wins"), so it is not optional and it is
        # not rewritten here.
        "title": title,
        "line": _dollars(market.get("floor_strike")),
        "yes_bid": _dollars(market.get("yes_bid_dollars")),
        "yes_ask": _dollars(market.get("yes_ask_dollars")),
        "no_bid": _dollars(market.get("no_bid_dollars")),
        "no_ask": _dollars(market.get("no_ask_dollars")),
        "last": _dollars(market.get("last_price_dollars")),
        "volume": _dollars(market.get("volume_fp")),
        "open_interest": _dollars(market.get("open_interest_fp")),
        "status": market.get("status"),
        "result": market.get("result") or None,
        "settlement_value": market.get("expiration_value") or None,
        # Kalshi's own occurrence, as published -- late by about six hours
        # against the real session start, and deliberately not corrected.
        "occurrence": market.get("occurrence_datetime") or market.get("expected_expiration_time"),
        "close_time": market.get("close_time"),
    }


def _get(session, path: str, params: dict) -> dict:
    for attempt in range(4):
        r = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if r.status_code == 429:  # public API rate limit: back off, never hammer
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Kalshi rate-limited {path} after retries")


def fetch_series(session, series: str, status: str, min_close_ts: int | None = None) -> list[dict]:
    markets: list[dict] = []
    cursor = None
    while True:
        params = {"series_ticker": series, "status": status, "limit": PAGE_LIMIT}
        if min_close_ts is not None:
            params["min_close_ts"] = min_close_ts
        if cursor:
            params["cursor"] = cursor
        body = _get(session, "/markets", params)
        markets.extend(body.get("markets") or [])
        cursor = body.get("cursor")
        if not cursor:
            return markets


def collect(session=None, now: datetime | None = None) -> list[dict]:
    """Open markets (the pre-race lines) plus the last few days of settled ones (their outcomes)."""
    session = session or requests.Session()
    now = now or datetime.now(timezone.utc)
    ts = now.isoformat(timespec="seconds")
    since = int((now - timedelta(days=SETTLED_LOOKBACK_DAYS)).timestamp())
    rows: list[dict] = []
    for series in SERIES:
        for status, min_close in (("open", None), ("settled", since)):
            for m in fetch_series(session, series, status, min_close):
                rows.append(normalize(m, series, ts))
    return rows


def write_snapshot(rows: list[dict], out_dir: Path = LINES_DIR, now: datetime | None = None) -> Path | None:
    """Append to today's gzip JSONL file (one file per UTC day, so the repo grows by one small file a day)."""
    if not rows:
        return None
    now = now or datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"kalshi_f1_lines_{now:%Y-%m-%d}.jsonl.gz"
    with gzip.open(path, "at", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def main() -> int:
    rows = collect()
    path = write_snapshot(rows)
    open_rows = sum(1 for r in rows if r["status"] == "active")
    print(f"kalshi lines: {len(rows)} rows ({open_rows} open) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
