"""Kalshi settled markets + trades for geopolitical / macro series (price & volume only; no identities).

API: https://api.elections.kalshi.com/trade-api/v2 (public, unauthenticated; strict rate limit -> we
pace requests and back off on 429).
  GET /series                                        all series (category, tags)
  GET /markets?series_ticker=X&status=settled        settled after the live/historical cutoff
  GET /historical/markets?series_ticker=X            settled before the cutoff (GET /historical/cutoff)
  GET /markets/trades?ticker=T  and  /historical/trades?ticker=T

Stages (checkpointed in ingest_state):  series -> markets (source kalshi_markets, key=series) ->
trades (source kalshi_trades, key=ticker, in-scope first then macro, by volume desc).

Usage:
  python -m pmflow.data.ingest_kalshi --stage markets
  python -m pmflow.data.ingest_kalshi --stage trades [--max-markets 2000]
"""
from __future__ import annotations

import argparse
import json
import logging

from .common import Http, HttpError, apply_schema, connect, fnum, setup_logging, state_get, state_set, ts_from_any
from .taxonomy import classify

K = "https://api.elections.kalshi.com/trade-api/v2"
log = logging.getLogger("ingest_kalshi")
http = Http(min_interval=0.7, max_retries=12)
CATEGORIES = ("World", "Politics", "Economics")
CAT_TAG = {"World": "World", "Politics": "Politics", "Economics": "Economy", "Elections": "Elections"}


def load_series(conn) -> None:
    d = http.get_json(f"{K}/series", timeout=120)
    rows = [(s["ticker"], s.get("title"), s.get("category"), s.get("tags") or [], s.get("frequency"),
             json.dumps({k: s.get(k) for k in ("categories", "settlement_sources", "contract_url")}))
            for s in d.get("series", [])]
    with conn.cursor() as cur:
        cur.executemany("""INSERT INTO kalshi_series VALUES (%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (series_ticker) DO UPDATE SET title=EXCLUDED.title, category=EXCLUDED.category,
                           tags=EXCLUDED.tags, frequency=EXCLUDED.frequency, raw=EXCLUDED.raw""", rows)
    log.info("kalshi series: %d", len(rows))


MCOLS = ["ticker", "event_ticker", "series_ticker", "title", "subtitle", "yes_sub_title", "rules_primary", "category",
         "market_type", "status", "result", "open_time", "close_time", "settlement_ts", "volume", "open_interest",
         "last_price", "pm_category", "geo_subtype", "in_scope", "macro_comparison", "raw"]
MUPSERT = (f"INSERT INTO kalshi_markets ({', '.join(MCOLS)}) VALUES ({', '.join(['%s'] * len(MCOLS))}) "
           "ON CONFLICT (ticker) DO UPDATE SET " + ", ".join(f"{c}=EXCLUDED.{c}" for c in MCOLS[1:]))


def market_row(m: dict, series: tuple) -> tuple:
    st, stitle, scat, stags = series
    tags = list(stags or []) + ([CAT_TAG[scat]] if scat in CAT_TAG else [])
    tax = classify(m.get("title"), stitle, tags, None)
    vol = fnum(m.get("volume_fp", m.get("volume")))
    return (m["ticker"], m.get("event_ticker"), st, m.get("title"), m.get("subtitle"), m.get("yes_sub_title"),
            m.get("rules_primary"), scat, m.get("market_type"), m.get("status"), m.get("result"),
            ts_from_any(m.get("open_time")), ts_from_any(m.get("close_time")), ts_from_any(m.get("settlement_ts")),
            vol, fnum(m.get("open_interest_fp", m.get("open_interest"))),
            fnum(m.get("last_price_dollars", m.get("last_price"))), tax["category"], tax["geo_subtype"],
            tax["in_scope"], tax["macro_comparison"],
            json.dumps({k: m.get(k) for k in ("expiration_value", "settlement_value_dollars", "floor_strike",
                                                "cap_strike", "strike_type", "no_sub_title", "can_close_early")}))


def markets_stage(conn) -> None:
    series = conn.execute("SELECT series_ticker, title, category, tags FROM kalshi_series WHERE category = ANY(%s) "
                          "ORDER BY series_ticker", (list(CATEGORIES),)).fetchall()
    for i, s in enumerate(series):
        key = s[0]
        st = state_get(conn, "kalshi_markets", key)
        if st and st["status"] == "done":
            continue
        n = 0
        for path, extra in (("/markets", {"status": "settled"}), ("/historical/markets", {})):
            cursor = None
            while True:
                p = {"series_ticker": key, "limit": 1000, **extra}
                if cursor:
                    p["cursor"] = cursor
                try:
                    d = http.get_json(K + path, p)
                except HttpError as e:
                    log.warning("series %s %s: %s", key, path, e)
                    break
                ms = d.get("markets") or []
                if ms:
                    with conn.cursor() as cur:
                        cur.executemany(MUPSERT, [market_row(m, s) for m in ms])
                    n += len(ms)
                cursor = d.get("cursor")
                if not cursor or not ms:
                    break
        state_set(conn, "kalshi_markets", key, "done", n_rows=n)
        if i % 100 == 0:
            log.info("series %d/%d (%s): %d markets", i + 1, len(series), key, n)


def trades_stage(conn, max_markets: int) -> None:
    conn.execute("""INSERT INTO ingest_state (source, key, status, priority)
                    SELECT 'kalshi_trades', ticker, 'pending', (CASE WHEN in_scope THEN 2 ELSE 1 END)*1e13 + coalesce(volume,0)
                    FROM kalshi_markets WHERE (in_scope OR macro_comparison) AND coalesce(volume,0) > 0
                    ON CONFLICT (source, key) DO NOTHING""")
    todo = conn.execute("""SELECT key FROM ingest_state WHERE source='kalshi_trades' AND status <> 'done'
                           ORDER BY priority DESC LIMIT %s""", (max_markets,)).fetchall()
    for i, (ticker,) in enumerate(todo):
        n = 0
        for path in ("/markets/trades", "/historical/trades"):
            cursor = None
            while True:
                p = {"ticker": ticker, "limit": 1000}
                if cursor:
                    p["cursor"] = cursor
                try:
                    d = http.get_json(K + path, p)
                except HttpError as e:
                    log.warning("trades %s %s: %s", ticker, path, e)
                    break
                ts = d.get("trades") or []
                if ts:
                    with conn.cursor() as cur:
                        cur.executemany(
                            "INSERT INTO kalshi_trades (trade_id, ticker, ts, yes_price, no_price, count, taker_side, "
                            "is_block_trade) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                            [(t["trade_id"], t["ticker"], ts_from_any(t.get("created_time")),
                              fnum(t.get("yes_price_dollars", t.get("yes_price"))),
                              fnum(t.get("no_price_dollars", t.get("no_price"))),
                              fnum(t.get("count_fp", t.get("count"))), t.get("taker_side"), t.get("is_block_trade"))
                             for t in ts])
                    n += len(ts)
                cursor = d.get("cursor")
                if not cursor or not ts:
                    break
        state_set(conn, "kalshi_trades", ticker, "done", n_rows=n)
        if i % 50 == 0:
            log.info("kalshi trades %d/%d %s: %d", i + 1, len(todo), ticker, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["markets", "trades", "all"], default="all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-markets", type=int, default=100000)
    a = ap.parse_args()
    setup_logging("ingest_kalshi")
    apply_schema()
    with connect() as conn:
        if a.stage in ("markets", "all"):
            if not conn.execute("SELECT 1 FROM kalshi_series LIMIT 1").fetchone():
                load_series(conn)
            markets_stage(conn)
        if a.stage in ("trades", "all"):
            trades_stage(conn, a.max_markets)
    log.info("kalshi stage %s finished", a.stage)


if __name__ == "__main__":
    main()
