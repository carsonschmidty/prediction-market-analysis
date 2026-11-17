"""Materialise the evaluation snapshot grid ``eval_snapshots`` and the event split table ``eval_event_splits``.

Design (see docs/evaluation_baseline.md):

* A snapshot is (market, cutoff_ts). cutoff_ts lies on a fixed UTC calendar grid every 6 hours
  (00/06/12/18 UTC). Every predictor may only use information with timestamp strictly < cutoff_ts.
* Candidate snapshots: every grid point after the market's first taker trade and before its
  ``closed_time`` (a market that is still open is observable live; nothing else about the future is used).
* ``is_live`` filter, computable at cutoff: last-trade YES price in [0.03, 0.97], >= 1 trade in the last
  72h, >= 3 taker trades and >= $200 taker notional in the last 7 days.
* Prices: taker trades only (role 0), YES-equivalent (outcome 0 price, or 1 - price for outcome 1).
  For the ~29 non Yes/No binary markets, "YES" means outcome index 0.
* Split: by event (``event_id``). event_close = max closed_time of its resolved in-scope/macro markets.
  train: event_close < TRAIN_END; val: TRAIN_END <= event_close < TEST_START and cutoff >= TRAIN_END;
  test: event_close >= TEST_START and cutoff >= TEST_START. Snapshots of val/test events whose cutoff
  falls before their period start get split ``purged_val`` / ``purged_test``: never scored; ``purged_val``
  rows may be used to refit a final model on train+val (all their labels are known before TEST_START),
  ``purged_test`` rows must never be used for anything.
* Stored rows: only candidates with last-trade price in [0.005, 0.995] and a trade in the last 7 days
  (superset of every live-filter variant in the sensitivity analysis).

Run: ``python -m pmflow.evaluation.snapshots``  (idempotent; rebuilds the tables, ~minutes).
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import psycopg

DSN = os.environ.get("PMFLOW_DSN", "postgresql://pmflow:pmflow@localhost:5432/pmflow")
log = logging.getLogger("pmflow.evaluation.snapshots")

GRID_HOURS = 6
TRAIN_END = "2026-04-01 00:00:00+00"   # val period starts here
TEST_START = "2026-07-01 00:00:00+00"  # test period starts here
DATA_END = "2026-11-17 00:00:00+00"    # last full day of trade data at build time

# primary live filter (parameters are stored per row so alternatives can be recomputed)
LIVE_P_LO, LIVE_P_HI = 0.03, 0.97
LIVE_MAX_AGE_H = 72
LIVE_MIN_N7D = 3
LIVE_MIN_USD7D = 200.0


def connect() -> psycopg.Connection:
    return psycopg.connect(DSN, autocommit=True)


SQL_MARKETS = """
DROP TABLE IF EXISTS eval_markets;
CREATE TABLE eval_markets AS
SELECT m.market_id, m.event_id, m.event_slug, m.event_title, m.question,
       m.in_scope, m.macro_comparison,
       CASE WHEN m.in_scope THEN 'in_scope' ELSE 'macro' END AS universe,
       m.geo_subtype, m.category, m.neg_risk, m.neg_risk_market_id,
       m.outcomes->>0 AS yes_outcome,
       (m.resolved_yes IS NOT NULL) AS is_yes_no,
       COALESCE(m.resolved_yes, m.winning_outcome_index = 0)::int AS y,
       m.created_at, m.end_date, m.closed_time,
       s.first_trade_ts, s.last_trade_ts, s.n_taker, s.taker_usd
FROM pm_markets m
LEFT JOIN pm_market_trade_stats s USING (market_id)
WHERE (m.in_scope OR m.macro_comparison) AND m.resolution_status = 'resolved'
  AND m.winning_outcome_index IS NOT NULL AND jsonb_array_length(m.outcomes) = 2;
ALTER TABLE eval_markets ADD PRIMARY KEY (market_id);
CREATE INDEX ON eval_markets (event_id);
ANALYZE eval_markets;
"""

# 6h bins of taker trades, with running cumulative sums so window stats are two lookups.
SQL_BINS = f"""
SET work_mem = '256MB';
DROP TABLE IF EXISTS eval_trade_bins6h;
CREATE UNLOGGED TABLE eval_trade_bins6h AS
WITH t AS (
  SELECT t.market_id,
         to_timestamp(floor(extract(epoch FROM t.ts) / {GRID_HOURS * 3600}) * {GRID_HOURS * 3600}) AS bin,
         t.ts,
         CASE WHEN t.outcome_index = 0 THEN t.price ELSE 1 - t.price END AS p_yes,
         t.size, t.price * t.size AS usd
  FROM pm_trades t JOIN eval_markets m USING (market_id)
  WHERE t.role = 0 AND t.ts < m.closed_time
), b AS (
  SELECT market_id, bin, count(*) AS n, sum(usd) AS usd, sum(size) AS sh, sum(p_yes * size) AS vw,
         (array_agg(p_yes ORDER BY ts DESC))[1] AS last_p, max(ts) AS last_ts
  FROM t GROUP BY 1, 2
)
SELECT market_id, bin, n, usd, sh, vw, last_p, last_ts,
       sum(n)   OVER w AS cum_n,
       sum(usd) OVER w AS cum_usd,
       sum(sh)  OVER w AS cum_sh,
       sum(vw)  OVER w AS cum_vw
FROM b
WINDOW w AS (PARTITION BY market_id ORDER BY bin ROWS UNBOUNDED PRECEDING);
CREATE INDEX ON eval_trade_bins6h (market_id, bin);
ANALYZE eval_trade_bins6h;
"""

SQL_EVENTS = f"""
DROP TABLE IF EXISTS eval_event_splits;
CREATE TABLE eval_event_splits AS
SELECT event_id, max(closed_time) AS event_close, min(created_at) AS event_open, count(*) AS n_markets,
       bool_or(in_scope) AS has_in_scope, bool_or(macro_comparison) AS has_macro,
       CASE WHEN max(closed_time) < '{TRAIN_END}' THEN 'train'
            WHEN max(closed_time) < '{TEST_START}' THEN 'val'
            ELSE 'test' END AS split,
       CASE WHEN max(closed_time) < '{TRAIN_END}' THEN '-infinity'::timestamptz
            WHEN max(closed_time) < '{TEST_START}' THEN '{TRAIN_END}'::timestamptz
            ELSE '{TEST_START}'::timestamptz END AS split_min_cutoff
FROM eval_markets GROUP BY event_id;
ALTER TABLE eval_event_splits ADD PRIMARY KEY (event_id);
"""

SQL_SNAPSHOTS = f"""
SET work_mem = '256MB';
DROP TABLE IF EXISTS eval_snapshots;
CREATE TABLE eval_snapshots AS
WITH g AS (
  SELECT m.market_id, m.event_id, c AS cutoff_ts
  FROM eval_markets m,
       generate_series(
         to_timestamp(floor(extract(epoch FROM m.first_trade_ts) / {GRID_HOURS * 3600}) * {GRID_HOURS * 3600})
           + interval '{GRID_HOURS} hours',
         m.closed_time - interval '1 microsecond',
         interval '{GRID_HOURS} hours') AS c
  WHERE m.first_trade_ts IS NOT NULL
), x AS (
  SELECT g.*,
         l.last_p, l.last_ts, l.vw / NULLIF(l.sh, 0) AS vwap_last_bin_any, l.bin AS last_bin,
         l.cum_n, l.cum_usd, l.cum_sh, l.cum_vw,
         d1.cum_n AS n_a24, d1.cum_usd AS usd_a24, d1.cum_sh AS sh_a24, d1.cum_vw AS vw_a24,
         d7.cum_n AS n_a7d, d7.cum_usd AS usd_a7d
  FROM g
  JOIN LATERAL (SELECT * FROM eval_trade_bins6h b WHERE b.market_id = g.market_id AND b.bin < g.cutoff_ts
                ORDER BY b.bin DESC LIMIT 1) l ON true
  LEFT JOIN LATERAL (SELECT cum_n, cum_usd, cum_sh, cum_vw FROM eval_trade_bins6h b
                     WHERE b.market_id = g.market_id AND b.bin < g.cutoff_ts - interval '24 hours'
                     ORDER BY b.bin DESC LIMIT 1) d1 ON true
  LEFT JOIN LATERAL (SELECT cum_n, cum_usd FROM eval_trade_bins6h b
                     WHERE b.market_id = g.market_id AND b.bin < g.cutoff_ts - interval '7 days'
                     ORDER BY b.bin DESC LIMIT 1) d7 ON true
)
SELECT
  (x.market_id::bigint * 100000 + (extract(epoch FROM x.cutoff_ts)::bigint / {GRID_HOURS * 3600}) % 100000)
      AS snapshot_id,
  x.market_id, x.event_id, m.universe, m.geo_subtype, m.neg_risk, m.y,
  x.cutoff_ts,
  (extract(hour FROM x.cutoff_ts AT TIME ZONE 'UTC') = 0) AS grid_24h,
  x.last_p::real AS price_yes,
  ((x.cum_vw - COALESCE(x.vw_a24, 0)) / NULLIF(x.cum_sh - COALESCE(x.sh_a24, 0), 0))::real AS vwap_24h,
  CASE WHEN x.last_bin = x.cutoff_ts - interval '{GRID_HOURS} hours'
       THEN x.vwap_last_bin_any END::real AS vwap_6h,
  (extract(epoch FROM x.cutoff_ts - x.last_ts) / 3600)::real AS last_trade_age_h,
  (x.cum_n - COALESCE(x.n_a24, 0))::int AS n_trades_24h,
  (x.cum_usd - COALESCE(x.usd_a24, 0))::real AS usd_24h,
  (x.cum_n - COALESCE(x.n_a7d, 0))::int AS n_trades_7d,
  (x.cum_usd - COALESCE(x.usd_a7d, 0))::real AS usd_7d,
  x.cum_n::int AS n_trades_cum,
  x.cum_usd::real AS usd_cum,
  (extract(epoch FROM x.cutoff_ts - m.created_at) / 3600)::real AS hours_since_open,
  (extract(epoch FROM m.end_date - x.cutoff_ts) / 3600)::real AS hours_to_end_date_diag,
  (extract(epoch FROM m.closed_time - x.cutoff_ts) / 3600)::real AS hours_to_close_diag,
  (m.created_at < '2023-11-18'::timestamptz) AS pre_trade_window_market,
  CASE WHEN x.cutoff_ts >= e.split_min_cutoff THEN e.split ELSE 'purged_' || e.split END AS split
FROM x
JOIN eval_markets m USING (market_id)
JOIN eval_event_splits e ON e.event_id = x.event_id
-- storage: drop rows that no live-filter variant can ever select (stale > 7d or price outside [0.005, 0.995])
WHERE x.last_p BETWEEN 0.005 AND 0.995 AND x.cutoff_ts - x.last_ts <= interval '7 days';
"""

SQL_SNAP_POST = f"""
ALTER TABLE eval_snapshots ADD COLUMN is_live boolean GENERATED ALWAYS AS (
     price_yes BETWEEN {LIVE_P_LO} AND {LIVE_P_HI}
 AND last_trade_age_h <= {LIVE_MAX_AGE_H}
 AND n_trades_7d >= {LIVE_MIN_N7D}
 AND usd_7d >= {LIVE_MIN_USD7D}) STORED;
ALTER TABLE eval_snapshots ADD PRIMARY KEY (snapshot_id);
CREATE INDEX ON eval_snapshots (market_id, cutoff_ts);
CREATE INDEX ON eval_snapshots (split) WHERE is_live;
ANALYZE eval_snapshots;
COMMENT ON TABLE eval_snapshots IS
 'Evaluation snapshot grid (6h UTC calendar grid). Predictors may only use data with ts < cutoff_ts. '
 'Primary evaluation set: is_live AND split IN (train,val,test); purged_val usable only for train+val refit, purged_test never. *_diag columns use future info (diagnostic only). '
 'Owner: evaluation agent; see docs/evaluation_baseline.md';
"""

SQL_DIAG = """
DROP TABLE IF EXISTS eval_diag_snapshots;
CREATE TABLE eval_diag_snapshots AS
WITH anchors AS (
  SELECT m.market_id, m.event_id, a.anchor, a.h, m.closed_time - make_interval(hours => a.h) AS cutoff_ts
  FROM eval_markets m CROSS JOIN (VALUES ('close', 1), ('close', 6), ('close', 24), ('close', 72)) a(anchor, h)
  UNION ALL
  SELECT m.market_id, m.event_id, 'last_trade', a.h, m.last_trade_ts - make_interval(hours => a.h)
  FROM eval_markets m CROSS JOIN (VALUES (1), (6), (24), (72)) a(h)
  WHERE m.last_trade_ts IS NOT NULL
)
SELECT a.market_id, a.event_id, a.anchor, a.h AS hours_before, a.cutoff_ts, m.universe, m.geo_subtype, m.y,
       l.p_yes::real AS price_yes, (extract(epoch FROM a.cutoff_ts - l.ts) / 3600)::real AS last_trade_age_h,
       e.split AS event_split
FROM anchors a
JOIN eval_markets m USING (market_id)
JOIN eval_event_splits e ON e.event_id = a.event_id
JOIN LATERAL (SELECT CASE WHEN t.outcome_index = 0 THEN t.price ELSE 1 - t.price END AS p_yes, t.ts
              FROM pm_trades t WHERE t.market_id = a.market_id AND t.ts < a.cutoff_ts AND t.role = 0
              ORDER BY t.ts DESC LIMIT 1) l ON true;
COMMENT ON TABLE eval_diag_snapshots IS
 'DIAGNOSTIC ONLY: snapshots anchored on closed_time / last trade (not knowable live, leaky anchor). Never use for model selection.';
"""


def run(sql: str, what: str) -> None:
    t0 = time.time()
    with connect() as c:
        c.execute(sql)
    log.info("%s done in %.0fs", what, time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-bins", action="store_true", help="keep eval_trade_bins6h (~hundreds of MB)")
    ap.add_argument("--skip-diag", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(SQL_MARKETS, "eval_markets")
    run(SQL_BINS, "eval_trade_bins6h")
    run(SQL_EVENTS, "eval_event_splits")
    run(SQL_SNAPSHOTS, "eval_snapshots")
    run(SQL_SNAP_POST, "eval_snapshots post")
    if not a.skip_diag:
        run(SQL_DIAG, "eval_diag_snapshots")
    if not a.keep_bins:
        run("DROP TABLE IF EXISTS eval_trade_bins6h;", "drop bins")


if __name__ == "__main__":
    main()
