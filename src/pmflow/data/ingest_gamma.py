"""Ingest metadata for ALL closed Polymarket markets from the Gamma API.

Endpoint: GET https://gamma-api.polymarket.com/markets/keyset?closed=true&include_tag=true
          &end_date_min=..&end_date_max=..&limit=100[&after_cursor=..]
(offset pagination on /markets is capped; /markets/keyset is the supported deep pagination,
max 100 rows/page; include_tag=true attaches market tags.)

Work is partitioned into one-day end_date windows; each window is a checkpoint row in
ingest_state(source='gamma_markets', key='YYYY-MM-DD') storing the last cursor, so a killed run
resumes mid-window. Rows are upserted on market_id, so reruns are idempotent.

Usage:
  python -m pmflow.data.ingest_gamma --resume [--workers 8] [--start 2020-01-01] [--end 2026-12-31]
  python -m pmflow.data.ingest_gamma --refresh-recent 3   # re-pull the last 3 days of windows
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

from .common import Http, apply_schema, connect, fnum, jload, setup_logging, state_get, state_set, ts_from_any
from .resolution import derive_resolution
from .taxonomy import classify

GAMMA = "https://gamma-api.polymarket.com"
SOURCE = "gamma_markets"
log = logging.getLogger("ingest_gamma")
http = Http(min_interval=0.05)

RAW_KEEP = ["umaResolutionStatuses", "umaEndDate", "groupItemTitle", "groupItemThreshold", "fee",
            "orderPriceMinTickSize", "orderMinSize", "negRiskRequestID", "negRiskOther", "enableOrderBook",
            "fpmmLive", "marketMakerAddress", "creator", "cyom", "rfqEnabled", "umaBond", "umaReward",
            "customLiveness", "fallbackResolutionType", "resolutionMechanism", "spread", "bestBid",
            "bestAsk", "restricted", "volumeAmm", "liquidityClob", "liquidityAmm", "sportsMarketType",
            "gameStartTime", "seriesColor", "umaResolutionStatus", "updatedAt"]

COLS = ["market_id", "condition_id", "question_id", "slug", "question", "description", "outcomes",
        "clob_token_ids", "outcome_prices", "event_id", "event_slug", "event_title", "tag_labels",
        "tag_slugs", "gamma_category", "market_type", "neg_risk", "neg_risk_market_id", "created_at",
        "start_date", "end_date", "closed_time", "active", "closed", "archived", "volume_usd",
        "volume_clob_usd", "liquidity_usd", "last_trade_price", "uma_resolution_status", "resolved_by",
        "resolution_source", "automatically_resolved", "resolution_status", "winning_outcome_index",
        "winning_outcome", "resolved_yes", "raw", "category", "geo_subtype", "is_scheduled_election", "in_scope",
        "macro_comparison", "taxonomy_rule", "taxonomy_version"]
# to save disk, long text / raw payload are dropped for high-volume out-of-scope verticals
SLIM_CATEGORIES = {"sports", "crypto"}

UPSERT = (f"INSERT INTO pm_markets ({', '.join(COLS)}) VALUES ({', '.join(['%s'] * len(COLS))}) "
          "ON CONFLICT (market_id) DO UPDATE SET "
          + ", ".join(f"{c}=EXCLUDED.{c}" for c in COLS if c != "market_id")
          + ", ingested_at=now()")


def market_row(m: dict) -> tuple:
    ev = (m.get("events") or [{}])[0] or {}
    tags = m.get("tags") or []
    res = derive_resolution(m)
    raw = {k: m[k] for k in RAW_KEEP if k in m}
    raw["events"] = [{"id": e.get("id"), "slug": e.get("slug"), "title": e.get("title"),
                      "seriesSlug": e.get("seriesSlug")} for e in (m.get("events") or [])]
    tax = classify(m.get("question"), ev.get("title"), [t.get("label") for t in tags if t.get("label")],
                   m.get("category"))
    slim = tax["category"] in SLIM_CATEGORIES
    return (
        int(m["id"]), (m.get("conditionId") or None), m.get("questionID"), m.get("slug"), m.get("question"),
        None if slim else m.get("description"), json.dumps(jload(m.get("outcomes"))), json.dumps(jload(m.get("clobTokenIds"))),
        json.dumps(jload(m.get("outcomePrices"))),
        int(ev["id"]) if ev.get("id") and str(ev.get("id")).isdigit() else None, ev.get("slug"), ev.get("title"),
        [t.get("label") for t in tags if t.get("label")], [t.get("slug") for t in tags if t.get("slug")],
        m.get("category"), m.get("marketType"), m.get("negRisk"), m.get("negRiskMarketID"),
        ts_from_any(m.get("createdAt")), ts_from_any(m.get("startDate")), ts_from_any(m.get("endDate")),
        ts_from_any(m.get("closedTime")), m.get("active"), m.get("closed"), m.get("archived"),
        fnum(m.get("volumeNum", m.get("volume"))), fnum(m.get("volumeClob")),
        fnum(m.get("liquidityNum", m.get("liquidity"))), fnum(m.get("lastTradePrice")),
        m.get("umaResolutionStatus"), m.get("resolvedBy"), m.get("resolutionSource"),
        m.get("automaticallyResolved"), res["resolution_status"], res["winning_outcome_index"],
        res["winning_outcome"], res["resolved_yes"], None if slim else json.dumps(raw),
        tax["category"], tax["geo_subtype"], tax["is_scheduled_election"], tax["in_scope"],
        tax["macro_comparison"], tax["taxonomy_rule"], tax["taxonomy_version"],
    )


def run_window(day: date) -> int:
    key = day.isoformat()
    a = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    b = a + timedelta(days=1) - timedelta(seconds=1)
    with connect() as conn:
        st = state_get(conn, SOURCE, key)
        if st and st["status"] == "done":
            return 0
        cursor = st["cursor"] if st else None
        n = st["n_rows"] if st else 0
        state_set(conn, SOURCE, key, "running", cursor=cursor)
        while True:
            p = {"closed": "true", "include_tag": "true", "limit": 100,
                 "end_date_min": a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "end_date_max": b.strftime("%Y-%m-%dT%H:%M:%SZ")}
            if cursor:
                p["after_cursor"] = cursor
            d = http.get_json(f"{GAMMA}/markets/keyset", p)
            ms = d.get("markets") or []
            if ms:
                rows = [market_row(m) for m in ms]
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.executemany(UPSERT, rows)
                n += len(rows)
            cursor = d.get("next_cursor")
            if not cursor or not ms:
                state_set(conn, SOURCE, key, "done", cursor=None, n_rows=n)
                return n
            state_set(conn, SOURCE, key, "running", cursor=cursor, n_rows=n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="skip windows already done (default behaviour)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None, help="last end_date day (default: today + 400 days)")
    ap.add_argument("--refresh-recent", type=int, default=0, help="reset the last N days' windows first")
    args = ap.parse_args()
    setup_logging("ingest_gamma")
    apply_schema()
    start = date.fromisoformat(args.start)
    today = datetime.now(timezone.utc).date()
    end = date.fromisoformat(args.end) if args.end else today + timedelta(days=400)
    if args.refresh_recent:
        with connect() as c:
            c.execute("UPDATE ingest_state SET status='pending', cursor=NULL WHERE source=%s AND key >= %s",
                      (SOURCE, (today - timedelta(days=args.refresh_recent)).isoformat()))
    # newest first: recent years are much denser and most relevant
    days = [end - timedelta(days=i) for i in range((end - start).days + 1)]
    total = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(args.workers, thread_name_prefix="gw") as ex:
        futs = {ex.submit(run_window, d): d for d in days}
        for i, f in enumerate(as_completed(futs)):
            d = futs[f]
            try:
                n = f.result()
            except Exception as e:  # keep going; window stays 'running' w/ cursor -> resumable
                log.error("window %s failed: %s", d, e)
                with connect() as c:
                    state_set(c, SOURCE, d.isoformat(), "error", error=str(e)[:500])
                continue
            with lock:
                total += n
            if n or i % 100 == 0:
                log.info("window %s: %d markets (run total %d, %d/%d windows)", d, n, total, i + 1, len(days))
    log.info("gamma ingest finished: %d market rows upserted this run", total)


if __name__ == "__main__":
    main()
