"""Small, committed exports under artifacts/data/ (the Postgres instance is ephemeral).

  market_counts_by_year.csv        closed Polymarket markets by close year x category x in_scope, with
                                   resolved / ambiguous / unresolved / void counts
  market_counts_by_year_pivot.csv  resolved markets: year x category (+ in_scope, scheduled-election cols)
  pm_inscope_markets.parquet/.csv  in-scope closed markets: metadata, outcome, taxonomy, trade coverage
  pm_macro_markets.parquet         macro / central-bank comparison set (same columns)
  pm_election_markets.parquet      scheduled-election markets tagged geopolitical/domestic (excluded from core)
  kalshi_markets_scoped.parquet    Kalshi settled markets classified in-scope or macro
  coverage_summary.json            row counts and date ranges for every table

Usage: python -m pmflow.data.export
"""
from __future__ import annotations

import json

import pandas as pd

from .common import ARTIFACTS, connect, setup_logging

YEAR = "extract(year from coalesce(closed_time, end_date))::int"

MARKET_COLS = f"""m.market_id, m.condition_id, m.question_id, m.slug, m.question, m.description, m.outcomes::text,
    m.clob_token_ids::text, m.outcome_prices::text, m.event_id, m.event_slug, m.event_title,
    array_to_string(m.tag_labels, '|') AS tags, m.created_at, m.start_date, m.end_date, m.closed_time,
    m.volume_usd, m.liquidity_usd, m.neg_risk, m.neg_risk_market_id, m.uma_resolution_status,
    m.resolution_status, m.winning_outcome_index, m.winning_outcome, m.resolved_yes, m.category, m.geo_subtype,
    m.is_scheduled_election, m.in_scope, m.macro_comparison, m.taxonomy_rule, m.taxonomy_version,
    s.n_taker, s.n_maker, s.taker_usd, s.first_trade_ts, s.last_trade_ts, s.n_taker_wallets,
    coalesce(s.taker_complete, false) AS taker_trades_complete, coalesce(s.maker_complete, false) AS maker_fills_complete"""


def q(conn, sql: str, *params) -> pd.DataFrame:
    cur = conn.execute(sql, params or None)
    return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


def main() -> None:
    log = setup_logging("export")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        c.execute("CREATE TABLE IF NOT EXISTS pm_market_trade_stats (market_id bigint PRIMARY KEY, n_taker bigint, "
                  "n_maker bigint, taker_usd double precision, first_trade_ts timestamptz, last_trade_ts timestamptz, "
                  "n_taker_wallets bigint, taker_complete boolean, maker_complete boolean)")
        counts = q(c, f"""SELECT {YEAR} AS year, category, in_scope, coalesce(is_scheduled_election,false) AS scheduled_election,
                    count(*) FILTER (WHERE resolution_status='resolved') AS n_resolved,
                    count(*) FILTER (WHERE resolution_status='ambiguous_5050') AS n_ambiguous_5050,
                    count(*) FILTER (WHERE resolution_status='cancelled_or_void') AS n_cancelled_or_void,
                    count(*) FILTER (WHERE resolution_status IN ('unresolved','multi_outcome_unclear')) AS n_unresolved,
                    count(*) AS n_closed,
                    round(sum(volume_usd) FILTER (WHERE resolution_status='resolved')) AS resolved_volume_usd
                FROM pm_markets WHERE closed GROUP BY 1,2,3,4 ORDER BY 1,2,3,4""")
        counts.to_csv(ARTIFACTS / "market_counts_by_year.csv", index=False)
        res = counts.groupby(["year", "category"])["n_resolved"].sum().unstack(fill_value=0)
        res["in_scope_true"] = counts[counts.in_scope == True].groupby("year")["n_resolved"].sum()  # noqa: E712
        res["in_scope_false"] = counts[counts.in_scope != True].groupby("year")["n_resolved"].sum()  # noqa: E712
        res["scheduled_election"] = counts[counts.scheduled_election].groupby("year")["n_resolved"].sum()
        res = res.fillna(0).astype(int)
        res.loc["total"] = res.sum()
        res.to_csv(ARTIFACTS / "market_counts_by_year_pivot.csv")
        log.info("counts by year:\n%s", res.to_string())

        base = f"SELECT {MARKET_COLS} FROM pm_markets m LEFT JOIN pm_market_trade_stats s USING (market_id) WHERE m.closed AND "
        ins = q(c, base + "m.in_scope ORDER BY m.volume_usd DESC NULLS LAST")
        ins.to_parquet(ARTIFACTS / "pm_inscope_markets.parquet", index=False)
        ins.drop(columns=["description"]).to_csv(ARTIFACTS / "pm_inscope_markets.csv", index=False)
        q(c, base + "m.macro_comparison ORDER BY m.volume_usd DESC NULLS LAST").to_parquet(
            ARTIFACTS / "pm_macro_markets.parquet", index=False)
        q(c, base + "m.is_scheduled_election ORDER BY m.volume_usd DESC NULLS LAST").drop(columns=["description"]).to_parquet(
            ARTIFACTS / "pm_election_markets.parquet", index=False)
        k = q(c, """SELECT ticker, event_ticker, series_ticker, title, subtitle, yes_sub_title, rules_primary, category,
                           status, result, open_time, close_time, settlement_ts, volume, open_interest, last_price,
                           pm_category, geo_subtype, in_scope, macro_comparison,
                           (SELECT count(*) FROM kalshi_trades t WHERE t.ticker = k.ticker) AS n_trades
                    FROM kalshi_markets k WHERE in_scope OR macro_comparison ORDER BY volume DESC NULLS LAST""")
        k.to_parquet(ARTIFACTS / "kalshi_markets_scoped.parquet", index=False)

        cov = {}
        for name, sql in {
            "pm_markets_closed": "SELECT count(*), min(closed_time)::text, max(closed_time)::text FROM pm_markets WHERE closed",
            "pm_markets_in_scope": "SELECT count(*), min(closed_time)::text, max(closed_time)::text FROM pm_markets WHERE closed AND in_scope",
            "pm_markets_in_scope_resolved": "SELECT count(*), min(closed_time)::text, max(closed_time)::text FROM pm_markets WHERE closed AND in_scope AND resolution_status='resolved'",
            "pm_markets_macro": "SELECT count(*), min(closed_time)::text, max(closed_time)::text FROM pm_markets WHERE closed AND macro_comparison",
            "pm_trades_taker": "SELECT count(*), min(ts)::text, max(ts)::text FROM pm_trades WHERE role=0",
            "pm_trades_maker": "SELECT count(*), min(ts)::text, max(ts)::text FROM pm_trades WHERE role=1",
            "pm_price_history": "SELECT count(*), min(ts)::text, max(ts)::text FROM pm_price_history",
            "pm_wallets": "SELECT count(*), min(first_activity_ts)::text, max(first_activity_ts)::text FROM pm_wallets",
            "pm_wallet_market_summary": "SELECT count(*), min(first_ts)::text, max(first_ts)::text FROM pm_wallet_market_summary",
            "kalshi_markets": "SELECT count(*), min(close_time)::text, max(close_time)::text FROM kalshi_markets",
            "kalshi_trades": "SELECT count(*), min(ts)::text, max(ts)::text FROM kalshi_trades",
        }.items():
            n, a, b = c.execute(sql).fetchone()
            cov[name] = {"rows": n, "min_ts": a, "max_ts": b}
        for src, st, n, rows in c.execute("SELECT source, status, count(*), sum(n_rows) FROM ingest_state GROUP BY 1,2 ORDER BY 1,2"):
            cov.setdefault("ingest_state", {}).setdefault(src, {})[st] = {"units": n, "rows": int(rows or 0)}
        scope_cov = c.execute("""SELECT count(*), count(*) FILTER (WHERE i.status='done'),
                                        sum(m.volume_usd), sum(m.volume_usd) FILTER (WHERE i.status='done')
                                 FROM pm_markets m LEFT JOIN ingest_state i ON i.source='pm_trades_taker' AND i.key=m.condition_id
                                 WHERE m.closed AND m.in_scope""").fetchone()
        cov["in_scope_taker_trade_coverage"] = {"markets": scope_cov[0], "markets_done": scope_cov[1],
                                                "volume_usd": scope_cov[2], "volume_usd_done": scope_cov[3]}
        (ARTIFACTS / "coverage_summary.json").write_text(json.dumps(cov, indent=2, default=str))
        log.info("coverage: %s", json.dumps(cov, default=str)[:2000])


if __name__ == "__main__":
    main()
