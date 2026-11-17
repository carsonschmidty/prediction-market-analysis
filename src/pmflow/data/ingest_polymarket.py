"""Polymarket trades / fills / price history / wallet activity ingestion (resumable, idempotent).

Stages (run in this priority order; each is checkpointed in ingest_state):
  metadata  -> pmflow.data.ingest_gamma (all closed markets) + taxonomy apply
  trades    -> data-api /v2/trades?condition=<cid> (taker-only, the default)      source=pm_trades_taker
  wallets   -> data-api /v2/activity?user=<wallet> for wallets in in-scope trades   source=pm_wallet_activity
  fills     -> data-api /v2/trades?condition=<cid>&takerOnly=false (maker+taker)   source=pm_trades_fill
  prices    -> clob /prices-history?market=<token>&interval=all&fidelity=1440 and
               hourly bars in 14-day windows (fidelity=60)                         source=pm_prices

Markets are queued in-scope first, then macro comparison set, by Gamma volume descending, so a
partial run holds the highest-volume subset. Each work unit stores its pagination cursor after
every page, so a killed process resumes where it stopped (`--resume` is the default behaviour).

Usage:
  python -m pmflow.data.ingest_polymarket --stage trades --workers 6
  python -m pmflow.data.ingest_polymarket --stage wallets --workers 6 --min-wallet-usd 1000
  python -m pmflow.data.ingest_polymarket --stage fills --workers 4
  python -m pmflow.data.ingest_polymarket --stage prices --workers 4
  python -m pmflow.data.ingest_polymarket --status
"""
from __future__ import annotations

import argparse
import os
import shutil
import hashlib
import io
import uuid
import logging
import threading
import time
from datetime import datetime, timezone

from .common import Http, HttpError, apply_schema, connect, setup_logging, state_set

DATA_API = "https://data-api.polymarket.com/v2"
CLOB = "https://clob.polymarket.com"
log = logging.getLogger("ingest_polymarket")
http = Http(min_interval=0.0)
STOP = threading.Event()

# which markets are queued for trade-level ingestion
SCOPE_SQL = """
SELECT condition_id, volume_usd,
       CASE WHEN in_scope THEN 2 WHEN macro_comparison THEN 1 ELSE 0 END AS tier
FROM pm_markets
WHERE closed AND condition_id IS NOT NULL AND (in_scope OR macro_comparison)
  AND coalesce(volume_usd,0) >= %s
"""


def enqueue_markets(conn, source: str, min_volume: float) -> int:
    rows = conn.execute(SCOPE_SQL, (min_volume,)).fetchall()
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO ingest_state (source, key, status, priority) VALUES (%s,%s,'pending',%s)
               ON CONFLICT (source, key) DO UPDATE SET priority=EXCLUDED.priority""",
            [(source, cid, tier * 1e13 + (vol or 0)) for cid, vol, tier in rows])
    return len(rows)


def claim(conn, source: str) -> tuple[str, str | None, int] | None:
    with conn.transaction():
        row = conn.execute(
            """SELECT key, cursor, n_rows FROM ingest_state
               WHERE source=%s AND status IN ('pending','running_stale')
               ORDER BY priority DESC NULLS LAST LIMIT 1 FOR UPDATE SKIP LOCKED""",
            (source,)).fetchone()
        if not row:
            return None
        conn.execute("UPDATE ingest_state SET status='running', updated_at=now() WHERE source=%s AND key=%s",
                     (source, row[0]))
    return row[0], row[1], row[2]


# ---------------------------------------------------------------- trades

TRADE_COPY_COLS = ("key_hash", "market_id", "outcome_index", "is_buy", "price", "size", "wallet", "tx_hash",
                   "ts", "role")
ROLE_CODE = {"taker": 0, "fill": 1}


def _hex(v: str | None) -> bytes:
    v = (v or "").lower()
    if v.startswith("0x"):
        v = v[2:]
    try:
        return bytes.fromhex(v)
    except ValueError:
        return v.encode()


def key_uuid(*parts) -> uuid.UUID:
    return uuid.UUID(bytes=hashlib.md5("|".join(str(p) for p in parts).encode()).digest())


class MarketInfo:
    def __init__(self, conn, cid: str):
        row = conn.execute("SELECT market_id, clob_token_ids FROM pm_markets WHERE condition_id=%s "
                           "ORDER BY market_id LIMIT 1", (cid,)).fetchone()
        if not row:
            raise RuntimeError(f"condition {cid} not in pm_markets")
        self.market_id = row[0]
        toks = row[1] if isinstance(row[1], list) else []
        self.tok_idx = {str(t): i for i, t in enumerate(toks)}

    def outcome_index(self, x: dict) -> int:
        i = self.tok_idx.get(str(x.get("token_id")))
        if i is not None:
            return i
        oi = x.get("outcome_index")
        return int(oi) if oi is not None and int(oi) < 100 else -1


def insert_trades(conn, rows: list[dict], role: str, mi: MarketInfo) -> int:
    rc = ROLE_CODE[role]
    recs = []
    for x in rows:
        price = float(x.get("price") or 0)
        size = float(x.get("size") or 0)
        tx = x.get("transaction_hash") or ""
        w = x.get("proxy_wallet") or ""
        recs.append((key_uuid(tx.lower(), x.get("token_id"), w.lower(), x.get("side"), price, size, rc),
                     mi.market_id, mi.outcome_index(x), x.get("side") == "BUY", price, size, _hex(w), _hex(tx),
                     datetime.fromtimestamp(int(x["timestamp"]), tz=timezone.utc), rc))
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE IF NOT EXISTS _tr (LIKE pm_trades) ON COMMIT DELETE ROWS")
            with cur.copy(f"COPY _tr ({', '.join(TRADE_COPY_COLS)}) FROM STDIN") as cp:
                for r in recs:
                    cp.write_row(r)
            cur.execute(f"""INSERT INTO pm_trades ({', '.join(TRADE_COPY_COLS)})
                            SELECT {', '.join(TRADE_COPY_COLS)} FROM _tr ON CONFLICT (key_hash) DO NOTHING""")
            return cur.rowcount


def trades_unit(conn, source: str, cid: str, cursor: str | None, n: int, role: str, max_pages: int) -> None:
    pages = 0
    mi = MarketInfo(conn, cid)
    while not STOP.is_set():
        p = {"condition": cid, "limit": 1000}
        if role == "fill":
            p["takerOnly"] = "false"
        if cursor:
            p["cursor"] = cursor
        try:
            d = http.get_json(f"{DATA_API}/trades", p)
        except HttpError as e:
            if e.status in (400, 404) and cursor:
                # cursor rejected (expired?) -> restart from newest; upserts make this safe
                log.warning("%s cursor rejected for %s (%s); restarting from head", source, cid, e)
                cursor = None
                continue
            raise
        rows = d.get("data") or []
        if rows:
            insert_trades(conn, rows, role, mi)
            n += len(rows)
        pages += 1
        pg = d.get("pagination") or {}
        cursor = pg.get("next_cursor")
        oldest = rows[-1]["timestamp"] if rows else None
        if not pg.get("has_more") or not cursor or not rows:
            state_set(conn, source, cid, "done", cursor=None, n_rows=n,
                      extra={"oldest_ts": oldest, "finished": datetime.now(timezone.utc).isoformat()})
            log.info("%s %s done: %d rows", source, cid[:12], n)
            return
        state_set(conn, source, cid, "running", cursor=cursor, n_rows=n, extra={"oldest_ts": oldest})
        if max_pages and pages >= max_pages:
            state_set(conn, source, cid, "partial", cursor=cursor, n_rows=n, extra={"oldest_ts": oldest})
            log.info("%s %s page cap reached: %d rows (resume later)", source, cid[:12], n)
            return
    state_set(conn, source, cid, "pending", cursor=cursor, n_rows=n)


# ---------------------------------------------------------------- wallets

def enqueue_wallets(conn, min_usd: float) -> int:
    """Queue wallets that placed a taker trade >= min_usd in an in-scope market (priority = max trade)."""
    n = conn.execute(
        """INSERT INTO ingest_state (source, key, status, priority)
           SELECT 'pm_wallet_activity', '0x' || encode(t.wallet, 'hex'), 'pending', max(t.price * t.size)
           FROM pm_trades t JOIN pm_markets m USING (market_id)
           WHERE m.in_scope AND t.role = 0 AND t.price * t.size >= %s
           GROUP BY t.wallet
           ON CONFLICT (source, key) DO UPDATE SET priority=greatest(ingest_state.priority, EXCLUDED.priority)""",
        (min_usd,)).rowcount
    return n


ACT_COLS = ("key_hash", "wallet", "ts", "type", "condition_id", "outcome_index", "is_buy", "price", "size",
            "usdc_size", "tx_hash")


def wallet_unit(conn, wallet: str, cursor: str | None, n: int, max_pages: int) -> None:
    pages = 0
    complete = False
    wb = _hex(wallet)
    while not STOP.is_set():
        p = {"user": wallet, "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        d = http.get_json(f"{DATA_API}/activity", p)
        rows = d.get("data") or []
        if rows:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute("CREATE TEMP TABLE IF NOT EXISTS _wa (LIKE pm_wallet_activity) ON COMMIT DELETE ROWS")
                    with cur.copy(f"COPY _wa ({', '.join(ACT_COLS)}) FROM STDIN") as cp:
                        for x in rows:
                            tx = (x.get("transaction_hash") or "").lower()
                            side = x.get("side") or None
                            oi = x.get("outcome_index")
                            cp.write_row((
                                key_uuid(wallet, tx, x.get("type"), x.get("token_id"), side, x.get("size"),
                                         x.get("price"), x.get("timestamp")),
                                wb, datetime.fromtimestamp(int(x["timestamp"]), tz=timezone.utc), x.get("type"),
                                _hex(x.get("condition_id")) if x.get("condition_id") else None,
                                int(oi) if oi is not None and int(oi) < 100 else None,
                                (side == "BUY") if side else None, x.get("price"), x.get("size"),
                                x.get("usdc_size"), _hex(tx) if tx else None))
                    cur.execute(f"INSERT INTO pm_wallet_activity ({', '.join(ACT_COLS)}) SELECT {', '.join(ACT_COLS)} "
                                "FROM _wa ON CONFLICT DO NOTHING")
            n += len(rows)
        pages += 1
        pg = d.get("pagination") or {}
        cursor = pg.get("next_cursor")
        if not pg.get("has_more") or not cursor or not rows:
            complete = True
            break
        if max_pages and pages >= max_pages:
            break
        state_set(conn, "pm_wallet_activity", wallet, "running", cursor=cursor, n_rows=n)
    if STOP.is_set() and not complete:
        state_set(conn, "pm_wallet_activity", wallet, "pending", cursor=cursor, n_rows=n)
        return
    conn.execute(
        """INSERT INTO pm_wallets (wallet, activity_fetched_at, activity_complete, n_activity_rows,
                                   first_activity_ts, first_trade_ts)
           SELECT %s, now(), %s, %s, min(ts), min(ts) FILTER (WHERE type='TRADE')
           FROM pm_wallet_activity WHERE wallet=%s
           ON CONFLICT (wallet) DO UPDATE SET activity_fetched_at=now(), activity_complete=EXCLUDED.activity_complete,
             n_activity_rows=EXCLUDED.n_activity_rows, first_activity_ts=EXCLUDED.first_activity_ts,
             first_trade_ts=EXCLUDED.first_trade_ts""",
        (wb, complete, n, wb))
    # newest-first paging: an incomplete wallet's true first-seen time is earlier than what we hold
    state_set(conn, "pm_wallet_activity", wallet, "done" if complete else "partial", cursor=cursor, n_rows=n)


# ---------------------------------------------------------------- prices

def enqueue_prices(conn) -> int:
    rows = conn.execute(
        """SELECT market_id::text || ':' || (o.idx - 1)::text, volume_usd, CASE WHEN in_scope THEN 2 ELSE 1 END,
                  o.tok, extract(epoch from start_date), extract(epoch from coalesce(closed_time, end_date))
           FROM pm_markets, jsonb_array_elements_text(clob_token_ids) WITH ORDINALITY o(tok, idx)
           WHERE closed AND (in_scope OR macro_comparison) AND jsonb_typeof(clob_token_ids)='array'""").fetchall()
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO ingest_state (source, key, status, priority, extra) VALUES ('pm_prices',%s,'pending',%s,
               jsonb_build_object('token',%s::text,'start',%s::float8,'end',%s::float8))
               ON CONFLICT (source, key) DO UPDATE SET priority=EXCLUDED.priority""",
            [(k, tier * 1e13 + (vol or 0), tok, s, e) for k, vol, tier, tok, s, e in rows])
    return len(rows)


def prices_unit(conn, key: str) -> None:
    ext = conn.execute("SELECT extra FROM ingest_state WHERE source='pm_prices' AND key=%s", (key,)).fetchone()[0] or {}
    mid, oi = (int(x) for x in key.split(":"))
    token = ext["token"]
    out: list[tuple] = []
    d = http.get_json(f"{CLOB}/prices-history", {"market": token, "interval": "all", "fidelity": 1440})
    out += [(mid, oi, 1440, datetime.fromtimestamp(h["t"], tz=timezone.utc), h["p"]) for h in d.get("history", [])]
    s, e = ext.get("start"), ext.get("end")
    if s and e:
        t = int(s)
        while t < e and not STOP.is_set():
            t2 = min(int(e) + 86400, t + 14 * 86400)
            try:
                d = http.get_json(f"{CLOB}/prices-history", {"market": token, "startTs": t, "endTs": t2, "fidelity": 60})
                out += [(mid, oi, 60, datetime.fromtimestamp(h["t"], tz=timezone.utc), h["p"])
                        for h in d.get("history", [])]
            except HttpError as err:
                log.warning("prices %s window %s: %s", key, t, err)
            t = t2
    with conn.cursor() as cur:
        cur.executemany("INSERT INTO pm_price_history VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING", out)
    state_set(conn, "pm_prices", key, "done", n_rows=len(out))


# ---------------------------------------------------------------- driver

MIN_FREE_GB = float(os.environ.get("PMFLOW_MIN_FREE_GB", "3"))


def disk_ok() -> bool:
    free = shutil.disk_usage("/home/user").free / 1e9
    if free < MIN_FREE_GB:
        log.error("free disk %.1f GB < %.1f GB: stopping workers (resume later with the same command)", free, MIN_FREE_GB)
        STOP.set()
        return False
    return True


def worker(stage: str, source: str, args) -> None:
    conn = connect()
    while not STOP.is_set() and disk_ok():
        job = claim(conn, source)
        if not job:
            return
        key, cursor, n = job
        try:
            if stage == "trades":
                trades_unit(conn, source, key, cursor, n, "taker", args.max_pages)
            elif stage == "fills":
                trades_unit(conn, source, key, cursor, n, "fill", args.max_pages)
            elif stage == "wallets":
                wallet_unit(conn, key, cursor, n, args.max_wallet_pages)
            elif stage == "prices":
                prices_unit(conn, key)
        except Exception as e:
            log.error("%s %s failed: %s", source, key[:16], e)
            try:
                conn.rollback()
            except Exception:
                conn = connect()
            st = conn.execute("SELECT cursor FROM ingest_state WHERE source=%s AND key=%s", (source, key)).fetchone()
            state_set(conn, source, key, "error", cursor=st[0] if st else None, error=str(e)[:500])
            time.sleep(5)


SOURCES = {"trades": "pm_trades_taker", "fills": "pm_trades_fill", "wallets": "pm_wallet_activity",
           "prices": "pm_prices"}


def status() -> None:
    with connect() as c:
        for r in c.execute("""SELECT source, status, count(*), sum(n_rows) FROM ingest_state
                              GROUP BY 1,2 ORDER BY 1,2""").fetchall():
            print(f"{r[0]:22s} {r[1]:10s} units={r[2]:>8d} rows={int(r[3] or 0):>12d}")
        for q in ["SELECT 'pm_markets', count(*) FROM pm_markets",
                  "SELECT 'pm_markets in_scope', count(*) FROM pm_markets WHERE in_scope",
                  "SELECT 'pm_trades', count(*) FROM pm_trades",
                  "SELECT 'pm_wallet_activity', count(*) FROM pm_wallet_activity",
                  "SELECT 'pm_price_history', count(*) FROM pm_price_history",
                  "SELECT 'kalshi_markets', count(*) FROM kalshi_markets",
                  "SELECT 'kalshi_trades', count(*) FROM kalshi_trades"]:
            print("%-22s %d" % tuple(c.execute(q).fetchone()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["trades", "fills", "wallets", "prices"], default="trades")
    ap.add_argument("--resume", action="store_true", help="resume from checkpoints (always on)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-volume", type=float, default=0.0, help="skip markets below this Gamma volume")
    ap.add_argument("--max-pages", type=int, default=0, help="per-market page cap (0 = unlimited)")
    ap.add_argument("--min-wallet-usd", type=float, default=1000.0)
    ap.add_argument("--max-wallet-pages", type=int, default=50)
    ap.add_argument("--no-enqueue", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        status()
        return
    setup_logging(f"ingest_polymarket_{args.stage}")
    apply_schema()
    source = SOURCES[args.stage]
    with connect() as c:
        # anything left 'running' by a killed process is resumable from its cursor
        c.execute("UPDATE ingest_state SET status='running_stale' WHERE source=%s AND status IN ('running','error')", (source,))
        if not args.no_enqueue:
            if args.stage in ("trades", "fills"):
                n = enqueue_markets(c, source, args.min_volume)
            elif args.stage == "wallets":
                n = enqueue_wallets(c, args.min_wallet_usd)
            else:
                n = enqueue_prices(c)
            log.info("enqueued/refreshed %d units for %s", n, source)
    threads = [threading.Thread(target=worker, args=(args.stage, source, args), name=f"{args.stage}{i}")
               for i in range(args.workers)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        STOP.set()
        for t in threads:
            t.join()
    log.info("stage %s finished (no more pending units)", args.stage)


if __name__ == "__main__":
    main()
