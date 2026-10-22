"""The Kalshi F1 collector: the rows the paper ledger reads by name.

The row shape is the same one the NFL and PL collectors write, and the paper
ledger reads these keys BY NAME: `snapshot_ts`, `series`, `stat`, `ticker`,
`event_ticker`, `player`, `title`, `line`, `yes_bid`, `yes_ask`, `no_bid`,
`no_ask`, `last`, `volume`, `open_interest`, `status`, `result`,
`settlement_value`, `occurrence`, `close_time`. A renamed field is a silently
skipped row, not a loud failure.

`title` is the load-bearing one: an F1 contract is ONE DRIVER's contract
("Main Race: Max Verstappen wins") and the ledger reads the driver out of the
title. It is Kalshi's own title, verbatim.

The fixtures below are real Kalshi payloads, read off the live public API on
2026-10-21 for the United States Grand Prix.
"""
import gzip
import json
from datetime import datetime, timezone

from f1_predictor.data import kalshi_lines as kl

#: KXF1RACE-UNIS26-VER -- Max Verstappen to win the United States Grand Prix.
VER = {
    "ticker": "KXF1RACE-UNIS26-VER", "event_ticker": "KXF1RACE-UNIS26",
    "title": "Main Race: Max Verstappen wins", "floor_strike": None,
    "status": "active", "result": "",
    "yes_bid_dollars": "0.4700", "yes_ask_dollars": "0.4800",
    "no_bid_dollars": "0.5200", "no_ask_dollars": "0.5300",
    "last_price_dollars": "0.4700", "volume_fp": "21796.55",
    "open_interest_fp": "21398.00",
    "occurrence_datetime": "2026-10-26T02:00:00Z", "close_time": "2026-11-02T02:00:00Z",
    "expiration_value": "",
}

#: The priceable series the paper ledger prices, by the stat it prices them as.
#: Copied from predictor-hub's `F1_SERIES_MARKETS` (services/paper/paper/
#: sources.py): the ledger's own name for each one.
PRICEABLE = {
    "KXF1RACE": "race_win",
    "KXF1RACEPODIUM": "race_podium",
    "KXF1TOP10": "top10",
    "KXF1POLE": "pole",
    "KXF1": "championship",
}


def test_normalize_keeps_the_quotes_as_probabilities_and_the_occurrence_as_kalshi_publishes_it():
    row = kl.normalize(VER, "KXF1RACE", "2026-10-21T12:00:00+00:00")
    assert row["stat"] == "race_win"
    assert (row["yes_bid"], row["yes_ask"]) == (0.47, 0.48)
    assert (row["no_bid"], row["no_ask"]) == (0.52, 0.53)
    assert row["last"] == 0.47
    assert row["volume"] == 21796.55 and row["open_interest"] == 21398.00
    # Kalshi's own occurrence, verbatim. It runs about six hours LATE against
    # the real session start, and it is NOT corrected here: the ledger does the
    # earlier-of-two-times comparison itself and reasons about the raw value.
    assert row["occurrence"] == "2026-10-26T02:00:00Z"
    assert row["close_time"] == "2026-11-02T02:00:00Z"
    # No strike on an F1 contract, so there is no threshold to record.
    assert row["line"] is None
    assert row["result"] is None and row["status"] == "active"


def test_the_title_is_kalshis_own_title_verbatim_and_unrewritten():
    # The ledger reads the DRIVER out of the title, so a rewritten title is a
    # contract it cannot price at all.
    race = kl.normalize(VER, "KXF1RACE", "t")
    pole = kl.normalize({**VER, "ticker": "KXF1POLE-UNIS26-VER",
                          "event_ticker": "KXF1POLE-UNIS26",
                          "title": "Max Verstappen is awarded Pole Position"},
                         "KXF1POLE", "t")
    podium = kl.normalize({**VER, "title": "Main Race: Max Verstappen to finish top 3"},
                           "KXF1RACEPODIUM", "t")
    championship = kl.normalize({**VER, "title": "Will Max Verstappen win the F1 Drivers Championship?"},
                                 "KXF1", "t")
    assert race["title"] == "Main Race: Max Verstappen wins"
    assert pole["title"] == "Max Verstappen is awarded Pole Position"
    assert podium["title"] == "Main Race: Max Verstappen to finish top 3"
    assert championship["title"] == "Will Max Verstappen win the F1 Drivers Championship?"


def test_a_settled_market_carries_its_result_and_settlement_value():
    row = kl.normalize({**VER, "status": "finalized", "result": "yes",
                        "expiration_value": "Max Verstappen"}, "KXF1RACE", "t")
    assert row["result"] == "yes" and row["settlement_value"] == "Max Verstappen"
    assert row["status"] == "finalized"


def test_missing_prices_are_none_not_zero():
    row = kl.normalize({"title": "Main Race: Max Verstappen wins"}, "KXF1RACE", "t")
    assert row["yes_bid"] is None and row["yes_ask"] is None
    assert row["no_bid"] is None and row["no_ask"] is None
    assert row["last"] is None and row["volume"] is None


def test_player_is_none_for_every_f1_series_because_these_are_driver_markets():
    for series in kl.SERIES:
        assert kl.normalize(VER, series, "t")["player"] is None, series


def test_the_series_the_model_cannot_price_are_marked_unpriceable_not_priceable():
    # The ledger skips a series it has no probability for by name, so the
    # collector records the same marker rather than dropping the series.
    assert kl.UNPRICEABLE == "f1_unpriceable_series"
    assert kl.SERIES["KXF1TOP5"] == kl.UNPRICEABLE
    assert kl.normalize(VER, "KXF1TOP5", "t")["stat"] == "f1_unpriceable_series"
    # The priceable five keep the ledger's own market names, so a row is never
    # confused for a market the ledger does not price.
    assert {s: kl.SERIES[s] for s in PRICEABLE} == PRICEABLE
    priceable_stats = {v for v in kl.SERIES.values() if v != kl.UNPRICEABLE}
    assert priceable_stats == set(PRICEABLE.values())


def test_every_kalshi_f1_series_is_collected_so_none_of_them_vanishes():
    # Every series Kalshi lists under KXF1* is here, so a series the model
    # cannot price still reaches the ledger as a countable row.
    assert all(s.startswith("KXF1") for s in kl.SERIES)
    assert set(PRICEABLE) <= set(kl.SERIES)
    assert kl.SERIES["KXF1RACE"] == "race_win" and kl.SERIES["KXF1POLE"] == "pole"


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b

    def raise_for_status(self):
        assert self.status_code < 400


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        if params.get("cursor") == "p2":
            return _Resp({"markets": [VER], "cursor": ""})
        return _Resp({"markets": [VER],
                      "cursor": "p2" if params["series_ticker"] == "KXF1RACE"
                      and params["status"] == "open" else ""})


def test_collect_pages_through_cursors_and_asks_for_open_and_recently_settled():
    s = _Session()
    rows = kl.collect(s, now=datetime(2026, 10, 21, 12, tzinfo=timezone.utc))
    statuses = {(c["series_ticker"], c["status"]) for c in s.calls}
    assert ("KXF1RACE", "open") in statuses and ("KXF1POLE", "settled") in statuses
    assert {s for s, _ in statuses} == set(kl.SERIES)
    assert any(c.get("cursor") == "p2" for c in s.calls)
    # The settled half asks for the last few days by close time, so the rows
    # carry Kalshi's own result rather than a result we worked out.
    settled = [c for c in s.calls if c["status"] == "settled"]
    assert settled and all(c.get("min_close_ts") for c in settled)
    # Derived from len(SERIES), never a literal: adding a series must not
    # silently break this count.
    assert len(rows) == len(kl.SERIES) * 2 + 1
    assert all(r["snapshot_ts"].startswith("2026-10-21") for r in rows)
    # Every series lands a row, and the ones the model cannot price are
    # marked as such rather than dropped.
    assert {r["stat"] for r in rows} == set(PRICEABLE.values()) | {kl.UNPRICEABLE}


def test_write_snapshot_appends_one_gzip_file_per_utc_day(tmp_path):
    rows = [kl.normalize(VER, "KXF1RACE", "t")]
    now = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)
    p1 = kl.write_snapshot(rows, tmp_path, now)
    p2 = kl.write_snapshot(rows, tmp_path, now)
    assert p1 == p2 and p1.name == "kalshi_f1_lines_2026-10-21.jsonl.gz"
    with gzip.open(p1, "rt") as fh:
        lines = fh.read().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["title"] == "Main Race: Max Verstappen wins"
    assert kl.write_snapshot([], tmp_path, now) is None
