"""Base tables for point-in-time wallet attributes (features role).

``feat_market_meta``  one row per corpus market (in_scope OR macro_comparison): the static
                      metadata the features use, plus resolution columns that are used ONLY to
                      compute a wallet's realised PnL on markets that closed before a cutoff.
``feat_wallet_hist``  one row per wallet that ever took a taker trade in a corpus market, with
                      sorted arrays that let the feature code answer "as of time c" questions
                      with a binary search, without per-pair rows (keeps the table ~250 MB):

    first_t        first in-corpus taker trade (epoch s). Left-censored at 2023-11-18.
    first_big_t    first in-scope taker trade with notional >= $1k (eligibility rule that
                   decides whether the wallet's full-history row in pm_wallets may be used).
    mkt_ids/mkt_first_t   corpus markets the wallet traded, sorted by first trade time.
    res_t/res_pnl    for RESOLVED corpus markets: closed_time (epoch s, int, sorted) and the
                   wallet's taker PnL in that market using only its trades with ts < closed_time
                   (cash flow + payout of net shares of the winning outcome).
                   A feature at cutoff c may use only entries with res_t < c.
    day_end_t/cum_usd     cumulative taker notional through the end of each UTC day the wallet
                   traded (epoch of the following midnight). Use entries with day_end_t <= c.

Everything is computed from taker rows (role 0) of corpus markets only, so "first seen" and
"prior markets" mean first seen in geopolitical + central-bank markets since 2023-11-18, not on
Polymarket overall (pm_wallets has the latter for a subset of wallets).

The builder is parameterised by source relation names so the leakage test can rebuild the same
table from a copy of the DB truncated at a cutoff.

CLI: python -m pmflow.features.base [--rebuild]
"""
from __future__ import annotations

import argparse
import time

from .db import connect

META_SQL = """
create table {target} as
select market_id::int as market_id, condition_id, question, created_at, start_date, end_date,
       closed_time, resolution_status, winning_outcome_index, resolved_yes,
       category, geo_subtype, in_scope, macro_comparison, outcomes
from {markets}
where in_scope or macro_comparison
"""

HIST_SQL = """
create {temp} table {target} as
with corpus as (
    select market_id, closed_time, in_scope,
           case when resolution_status = 'resolved' and winning_outcome_index in (0, 1)
                then winning_outcome_index end as win
    from {meta}
),
t as (
    select tr.wallet, tr.market_id, tr.outcome_index, tr.is_buy, tr.price, tr.size, tr.ts,
           c.closed_time, c.win, c.in_scope
    from {trades} tr join corpus c on c.market_id = tr.market_id
    where tr.role = 0
),
pair as (
    select wallet, market_id, min(ts) as first_ts, max(win) as win, max(closed_time) as closed_time,
           sum(case when ts < closed_time then case when is_buy then -price*size else price*size end end) as cf,
           sum(case when ts < closed_time and outcome_index = 0 then case when is_buy then size else -size end end) as sh0,
           sum(case when ts < closed_time and outcome_index = 1 then case when is_buy then size else -size end end) as sh1
               from t group by 1, 2
),
pa as (
    select wallet,
           min(first_ts) as first_ts,
           array_agg(market_id::int order by first_ts, market_id) as mkt_ids,
           array_agg(extract(epoch from first_ts)::int order by first_ts, market_id) as mkt_first_t,
           array_agg(ceil(extract(epoch from closed_time))::int order by closed_time, market_id)
               filter (where win is not null and cf is not null) as res_t,
           array_agg((coalesce(cf, 0) + case win when 0 then coalesce(sh0, 0) else coalesce(sh1, 0) end)::real
                     order by closed_time, market_id)
               filter (where win is not null and cf is not null) as res_pnl
    from pair group by 1
),
dd as (
    select wallet, ((ts at time zone 'UTC')::date + 1) as day_end, sum(price*size) as usd
    from t group by 1, 2
),
dc as (
    select wallet, day_end, sum(usd) over (partition by wallet order by day_end) as cum
    from dd
),
da as (
    select wallet,
           array_agg(extract(epoch from (day_end::timestamp at time zone 'UTC'))::int order by day_end) as day_end_t,
           array_agg(cum::real order by day_end) as cum_usd
    from dc group by 1
),
big as (
    select wallet, min(ts) as first_big_ts from t
    where in_scope and price*size >= 1000 group by 1
)
select pa.wallet, extract(epoch from pa.first_ts)::float8 as first_t,
       extract(epoch from big.first_big_ts)::float8 as first_big_t,
       pa.mkt_ids, pa.mkt_first_t, pa.res_t, pa.res_pnl, da.day_end_t, da.cum_usd
from pa join da using (wallet) left join big using (wallet)
"""


def build_meta(conn, target="feat_market_meta", markets="pm_markets"):
    with conn.cursor() as cur:
        cur.execute(f"drop table if exists {target}")
        cur.execute(META_SQL.format(target=target, markets=markets))
        cur.execute(f"alter table {target} add primary key (market_id)")
    conn.commit()


def build_hist(conn, target="feat_wallet_hist", trades="pm_trades", meta="feat_market_meta", temp=False):
    with conn.cursor() as cur:
        cur.execute("set work_mem = '256MB'")
        cur.execute(f"drop table if exists {target}")
        cur.execute(HIST_SQL.format(target=target, trades=trades, meta=meta, temp="temp" if temp else ""))
        cur.execute(f"alter table {target} add primary key (wallet)")
        cur.execute(f"analyze {target}")
    conn.commit()


def main(argv=None):
    ap = argparse.ArgumentParser(description="build feat_market_meta + feat_wallet_hist")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("select to_regclass('feat_wallet_hist') is not null")
        exists = cur.fetchone()[0]
    if exists and not a.rebuild:
        print("feat_wallet_hist exists; use --rebuild")
        return
    t0 = time.time()
    build_meta(conn)
    print(f"feat_market_meta built {time.time()-t0:.0f}s")
    build_hist(conn)
    with conn.cursor() as cur:
        cur.execute("select count(*), pg_size_pretty(pg_total_relation_size('feat_wallet_hist')) from feat_wallet_hist")
        print("feat_wallet_hist", cur.fetchone(), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
