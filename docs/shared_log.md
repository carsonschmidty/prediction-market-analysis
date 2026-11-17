# Shared findings log

Append-only. One entry per finding or blocker: `## [YYYY-MM-DD HH:MM UTC] <agent> — <title>`, then a few lines.
Prefix blockers with `BLOCKER:`. Never edit another agent's entries.

## [2026-11-17 21:30 UTC] labeling — seed set of 25 reported cases written
`artifacts/labeling/reported_cases.csv` holds 25 rows (23 Polymarket, 2 Kalshi), with loader `pmflow.labeling.cases`.
- 19 rows in scope. Grades: A=11, B=10, C=4.
- 5 negative cases: PM-002, 012, 013, 014 are wrong-side or unvindicated flow (PM-012 is fresh wallets on a March-31 Iran ceasefire that resolved NO). PM-021 (Israel election) is pending.
- 3 contrast cases: PM-022 French whale, PM-023 Warsh, PM-017 CZ.
- 10 rows have a minute/hour news timestamp, covering 6 distinct events: Maduro 2026-01-03T06:00Z, Nobel 2025-10-10T09:00Z, Israel strike 2025-06-12T23:30Z, US strike 2026-02-28T06:00Z, ceasefire 2026-04-07T22:32Z, and Israeli exit polls 2026-10-27T20:00Z (assumed).
- 21 rows were matched to Gamma market_id/condition_id through gamma-api public-search.
- Note for data/features: Gamma `closedTime` is NOT the news time. It can lag hours or weeks (e.g. Warsh: news Jan 29, close Mar 4), so use it only as an upper bound.

## [2026-11-17 21:30 UTC] labeling — BLOCKER: news hosts unreachable, so cases were verified only from search extracts
WebFetch DNS-fails for news sites (www.npr.org, www.theblock.co, wtop.com). The proxy status also lists 403s for dune.com, arxiv.org, api.gdeltproject.org and polygon-rpc.com.
- Every case row was built from web-search result extracts of the cited URLs, not from full articles (column `source_access`). A human should spot-check primary articles before thresholds are tuned on reported sizes and prices.
- GDELT (the obvious news-timestamp API) is blocked.
- gamma-api.polymarket.com works now, but the proxy log shows a transient 403 earlier today.

## [2026-11-17 21:30 UTC] labeling — pm_markets match pending ingestion
`pm_markets` had ~965k rows during this session, but only 5 of the 21 case markets are present so far (by market_id).
- `pg_market_id` is left blank in the CSV.
- Re-run once ingestion completes: `python -m pmflow.labeling.cases --write-matches artifacts/labeling/case_market_matches.csv`
- Data agent: please make sure these Gamma ids are ingested along with their trades: 916440, 956450, 560868, 532742, 551651, 532741, 1198423, 916732, 1706788, 1466015, 635246, 582457, 516906, 682705, 253591, 537916.

## [2026-11-17 21:30 UTC] labeling — harness refused .md report writes from this agent
Writing `docs/reported_cases.md` and `docs/labeling_plan.md` was rejected by the tool layer ("subagents should return findings as text"). Both documents' full content was returned to the orchestrator in the hand-back message, ready to save.

## [2026-11-17 20:45 UTC] data — Polymarket trade source that works: data-api v2 (v1 retired)
- `data-api.polymarket.com/trades` (v1) returns HTTP 410 "Data API v1 has been retired". The replacement is
  `https://data-api.polymarket.com/v2/trades?condition=<conditionId>&limit=1000&cursor=<next_cursor>`
  (param is `condition`, not `market`; limit max 1000; signed cursor pagination; full history reachable —
  e.g. 1,157,456 taker trades for "Will Donald Trump be inaugurated?" back to market start, ~4-5k rows/s/stream).
- Default is taker-only (`takerOnly=true`): one row per taker order with `proxy_wallet`, side, price, size,
  token, tx hash, timestamp. `takerOnly=false` adds the maker rows (same tx hash). We store taker rows (role 0)
  and, in a later pass, maker rows (role 1) = takerOnly=false rows minus rows identical to a taker row.
- Rows also carry display name / pseudonym / bio / profile image: we deliberately DROP these; wallets are kept
  only as opaque addresses.
- Unknown query params are silently ignored (e.g. start/end on /v2/trades), so do not assume a filter applied.
- Wallet history: `/v2/activity?user=<wallet>&sortDirection=ASC` (TRADE/REDEEM/SPLIT/MERGE/...; supports
  `start`/`end`) gives exact first-seen and per-market history across all of Polymarket. Used for pm_wallets /
  pm_wallet_market_summary.
- Gamma: `/markets` offset pagination is capped (422 "offset too large"); use `/markets/keyset` (max 100/page,
  `after_cursor`), `include_tag=true` for tags; `/events*` with limit>50 requires `Accept-Encoding: gzip`.
- CLOB `prices-history`: `interval=all` works only with coarse `fidelity` (1440 ok, 60 returns empty);
  finer bars need `startTs/endTs` windows (≈ <= 2 weeks per call).

## [2026-11-17 20:45 UTC] data — BLOCKER: Goldsky Polymarket subgraphs are deprecated; on-chain hosts unreachable
- All `api.goldsky.com/.../subgraphs/{orderbook,activity,pnl,positions,oi}-subgraph/*/gn` return
  `ENDPOINT_DEPRECATED`: "paused and deprecated following Polymarket's migration to V2 — the data is stale and
  incorrect". Its suggested replacement `edge.goldsky.com` is blocked by the proxy (403). Not used.
- Wallet funding source (first USDC inflow / funding address) needs on-chain indexing. Blocked hosts (proxy 403):
  `polygon-rpc.com`, `rpc.ankr.com`, `polygon-mainnet.g.alchemy.com`, `api.polygonscan.com`, `api.etherscan.io`,
  `api.dune.com`, `edge.goldsky.com`, `gateway.thegraph.com`, `api.thegraph.com`. To unblock: allow any one Polygon
  JSON-RPC host (for USDC/USDC.e Transfer logs into proxy wallets) or `api.dune.com` (+ API key). Funding-source
  features are therefore NOT available; nothing is imputed.
- Historical order-book snapshots: no public endpoint (CLOB `/book` is live-only and returns "Invalid token id"
  for closed markets). Not available. Price history (CLOB prices-history) and trades are the only microstructure.
- `docs.polymarket.com` is also blocked (403), so API behaviour above was established empirically.
- Possible alternative (not used): HF dataset `SII-WANGZJ/Polymarket_data` (third-party, MIT) has on-chain
  OrderFilled events with maker+taker through 2026-10-04, but trades.parquet alone is 45.8 GB > our free disk.

## [2026-11-17 20:45 UTC] data — Disk is the binding constraint (~16 GB free)
- Trades are stored compactly (bytea wallets/hashes, market_id + outcome_index instead of condition/token
  text): ~295 B/row incl. indexes. Use view `pm_trades_v` for readable columns.
- Ingest workers stop automatically below a free-disk floor (trades/wallets 4 GB, prices 5 GB, maker fills 8 GB)
  so other roles keep room for their tables. Raw wallet activity rows are aggregated, not stored.
- Sports/crypto market rows keep question/tags but drop description + raw payload.

## [2026-11-17 21:25 UTC] data — BLOCKER/GAP: Data API trade + wallet history only reaches back to ~2023-11-18
- `/v2/trades` and `/v2/activity` return nothing older than ~2023-11-18 (looks like a rolling ~3-year retention:
  e.g. "Will Erdoğan win the 2023 Turkish presidential election?" ($4.8M volume) returns 0 trades; the earliest
  trade we hold is 2023-11-18 21:12 UTC). Markets that closed before then have metadata + outcome but NO trades;
  markets straddling the boundary have truncated early history.
- Consequence for wallet features: first-seen / prior-market counts are LEFT-CENSORED at 2023-11-18. Wallets
  that redeem/sell positions they never bought inside the window are flagged `pm_wallets.pre_window_activity`.
  A wallet "first seen" in Nov–Dec 2023 may be much older. Treat freshness features before ~2024-01 as unreliable.
- Only route to earlier trades found: HF dataset `SII-WANGZJ/Polymarket_data` (on-chain OrderFilled since
  2022-11-21, 45.8 GB for trades.parquet) — does not fit our disk; would need ~60 GB free or range-read tooling.
- Pre-2022 (AMM era) Polymarket trades are not available from any reachable source.

## [2026-11-17 21:30 UTC] data — Data layer ready: markets, taxonomy, in-scope taker trades complete
- `pm_markets`: 5,538,585 closed Polymarket markets with conservative resolution labels and taxonomy
  (`category`, `geo_subtype`, `is_scheduled_election`, `in_scope`, `macro_comparison`, `taxonomy_rule`).
  In scope: 9,706 (9,700 resolved). Macro comparison: 1,321. Counts: `artifacts/data/market_counts_by_year*.csv`.
- `pm_trades` (view `pm_trades_v`): 22.46M taker trades, complete history (within the Data API window) for all
  in-scope + macro markets; 100% wallet-attributed (proxy wallet); 978,553 distinct in-scope taker wallets.
- Taxonomy hand-check (final rules): precision 98.2% on 110 in-scope, est. recall ≈95% from 120 hard negatives.
- Still running in background (resumable): wallet histories (50.9k wallets) → maker fills → price history;
  Kalshi markets → trades. Progress: `.venv/bin/python -m pmflow.data.ingest_polymarket --status`.
- Gamma `volume_usd` ≠ taker notional (2–8× larger); use trade-level notional.
