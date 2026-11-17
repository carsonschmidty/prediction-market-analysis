"""Derived tables built from the raw ingested data (re-runnable).

  pm_market_trade_stats       per market: trade counts by role, first/last trade, distinct wallets, notional
  pm_wallet_market_summary    fills market_id / category / market close time / resolution columns

Usage: python -m pmflow.data.derive
"""
from __future__ import annotations

from .common import apply_schema, connect, setup_logging

SQL = [
    """CREATE TABLE IF NOT EXISTS pm_market_trade_stats (
           market_id bigint PRIMARY KEY,
           n_taker bigint, n_maker bigint, taker_usd double precision,
           first_trade_ts timestamptz, last_trade_ts timestamptz, n_taker_wallets bigint,
           taker_complete boolean, maker_complete boolean)""",
    "TRUNCATE pm_market_trade_stats",
    """INSERT INTO pm_market_trade_stats
       SELECT t.market_id,
              count(*) FILTER (WHERE role = 0), count(*) FILTER (WHERE role = 1),
              sum(price * size) FILTER (WHERE role = 0), min(ts), max(ts),
              count(DISTINCT wallet) FILTER (WHERE role = 0), false, false
       FROM pm_trades t GROUP BY t.market_id""",
    """UPDATE pm_market_trade_stats s SET taker_complete = true
       FROM pm_markets m JOIN ingest_state i ON i.source = 'pm_trades_taker' AND i.key = m.condition_id AND i.status = 'done'
       WHERE m.market_id = s.market_id""",
    """UPDATE pm_market_trade_stats s SET maker_complete = true
       FROM pm_markets m JOIN ingest_state i ON i.source = 'pm_trades_fill' AND i.key = m.condition_id AND i.status = 'done'
       WHERE m.market_id = s.market_id""",
    """UPDATE pm_wallet_market_summary w SET market_id = m.market_id, category = m.category,
              market_closed_time = m.closed_time, market_resolution_status = m.resolution_status
       FROM pm_markets m WHERE m.condition_id = w.condition_id
         AND (w.market_id IS NULL OR w.market_resolution_status IS DISTINCT FROM m.resolution_status)""",
    """UPDATE pm_wallets w SET pre_window_activity = EXISTS (
           SELECT 1 FROM pm_wallet_market_summary s WHERE s.wallet = w.wallet AND s.buy_usd = 0 AND s.split_usd = 0
             AND s.sell_usd + s.redeem_usd + s.merge_usd > 0)
              OR w.first_activity_ts < timestamptz '2023-12-01'""",
    "CREATE INDEX IF NOT EXISTS pm_wms_wallet_first_idx ON pm_wallet_market_summary (wallet, first_ts)",
    "ANALYZE pm_trades", "ANALYZE pm_markets", "ANALYZE pm_wallet_market_summary",
]


def main() -> None:
    log = setup_logging("derive")
    apply_schema()
    with connect() as c:
        for q in SQL:
            c.execute(q)
            log.info("ok: %s", " ".join(q.split())[:80])


if __name__ == "__main__":
    main()
