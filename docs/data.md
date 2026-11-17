# Data layer (owner: data agent, `src/pmflow/data/`)

Public data only. Wallets are opaque 20-byte addresses. The display names, pseudonyms, bios and
profile images that the Polymarket Data API returns next to trades are dropped at ingestion.

## Quick start / resume

```bash
scripts/start_db.sh                                        # idempotent: init/start Postgres 16, role+db pmflow, apply schema
nohup scripts/run_polymarket_pipeline.sh > data/logs/pipeline.nohup 2>&1 &   # full resumable pipeline
.venv/bin/python -m pmflow.data.ingest_polymarket --status # progress (ingest_state + table counts)
tail -f data/logs/pipeline.log data/logs/ingest_polymarket_*.log
nohup .venv/bin/python -m pmflow.data.ingest_kalshi --stage all > data/logs/kalshi.nohup 2>&1 &
```

Individual stages (each is checkpointed in `ingest_state`; re-running continues where it stopped, upserts make
reruns idempotent; a killed unit resumes from its stored cursor):

| command | what |
|---|---|
| `python -m pmflow.data.ingest_gamma --resume [--workers 10] [--start/--end]` | all closed Polymarket markets (one-day `end_date` windows) |
| `python -m pmflow.data.taxonomy --apply` / `--sample N` | (re)classify every market / write hand-check sample |
| `python -m pmflow.data.ingest_polymarket --stage trades` | taker trades, in-scope markets then macro set, by volume desc |
| `... --stage wallets --min-wallet-usd 1000` | wallet histories for wallets with a >= $1k taker trade in an in-scope market |
| `... --stage fills` | maker fills (passive side) for the same markets |
| `... --stage prices` | CLOB price history (daily + hourly) |
| `python -m pmflow.data.derive` | `pm_market_trade_stats`, fills market columns of `pm_wallet_market_summary` |
| `python -m pmflow.data.export` | small exports to `artifacts/data/` |
| `python -m pmflow.data.ingest_kalshi --stage markets|trades|all` | Kalshi settled markets + trades |

Workers stop by themselves when free disk falls below a floor (4 GB trades/wallets, 5 GB prices, 8 GB maker
fills; override with `PMFLOW_MIN_FREE_GB`). Disk (~16 GB free at start) is the binding constraint.

## Sources actually used (verified 2026-11-17)

| need | endpoint | notes |
|---|---|---|
| market metadata | `gamma-api.polymarket.com/markets/keyset?closed=true&include_tag=true&end_date_min&end_date_max&limit=100&after_cursor=` | `/markets` offset pagination is capped (422); keyset max 100/page; `include_tag` gives market tags; `/events*` with limit>50 requires gzip |
| trades + wallet | `data-api.polymarket.com/v2/trades?condition=<conditionId>&limit=1000&cursor=` | v1 `/trades` is retired (HTTP 410). Default = taker rows only; `takerOnly=false` adds maker rows. Full history reachable (1.16M rows for the largest market tested). Unknown params are silently ignored |
| wallet history | `data-api.polymarket.com/v2/activity?user=<wallet>&sortDirection=ASC&limit=1000&cursor=` | TRADE/REDEEM/SPLIT/MERGE/REWARD/... across all markets; ASC gives exact first-seen |
| price history | `clob.polymarket.com/prices-history?market=<tokenId>&interval=all&fidelity=1440`, and `startTs/endTs` 14-day windows with `fidelity=60` | `interval=all` with fidelity 60 returns empty for closed markets |
| Kalshi | `api.elections.kalshi.com/trade-api/v2`: `/series`, `/markets?status=settled`, `/historical/markets`, `/markets/trades`, `/historical/trades` | settled-before-cutoff data moved to `/historical/*` (`/historical/cutoff`); heavy 429 rate limiting from our egress |

Not usable: Goldsky Polymarket subgraphs (`ENDPOINT_DEPRECATED`, "stale and incorrect" after the V2 migration);
`edge.goldsky.com`, Polygon RPCs, Polygonscan, Dune, The Graph, docs.polymarket.com (all proxy 403). Historical
order books: no public source (CLOB `/book` is live only). Wallet funding source (first USDC inflow): blocked,
needs a Polygon RPC or Dune. See `docs/shared_log.md`.

## Schema (`src/pmflow/data/schema.sql`) – what downstream roles should use

* **`pm_markets`** (PK `market_id` = Gamma id; `condition_id` joins Data API). Text: `question`, `description`
  (NULL for sports/crypto to save disk), `slug`, `event_id/slug/title`, `tag_labels[]`. Time: `created_at`,
  `start_date`, `end_date` (scheduled), `closed_time` (actual close ≈ resolution). Size: `volume_usd`,
  `liquidity_usd`. Outcomes: `outcomes`, `clob_token_ids` (aligned jsonb arrays), `outcome_prices`.
  Resolution: `resolution_status` ∈ {resolved, ambiguous_5050, cancelled_or_void, unresolved,
  multi_outcome_unclear}, `winning_outcome_index`, `winning_outcome`, `resolved_yes` (Yes/No markets only).
  Taxonomy: `category`, `geo_subtype`, `is_scheduled_election`, `in_scope`, `macro_comparison`,
  `taxonomy_rule` (which rule fired), `taxonomy_version`.
* **`pm_trades`** (compact) / **`pm_trades_v`** (readable view). One row per fill:
  `market_id, outcome_index, is_buy, price, size` (shares; USD = price*size), `wallet` (bytea), `tx_hash`
  (bytea), `ts`, `role` (0 = taker/aggressor, 1 = maker/passive). Unique `key_hash` = md5 of
  (tx, token, wallet, side, price, size, role). `pm_trades_v` adds `condition_id, token_id, outcome,
  side, usd_notional, wallet/tx_hash as 0x-hex, role text`. Index `(market_id, ts)`, `(wallet, ts)`.
  Join wallets as `'0x'||encode(wallet,'hex')` or compare bytea directly.
* **`pm_market_trade_stats`**: per market taker/maker counts, first/last trade, distinct taker wallets,
  `taker_complete` / `maker_complete` (full history fetched).
* **`pm_wallets`**: per fetched wallet `first_activity_ts`, `first_trade_ts` (exact, oldest-first fetch),
  `n_markets`, `n_activity_rows`, `activity_complete` (false = capped at 10k activities; summary then covers
  only the wallet's earliest activity – such wallets are by construction not "fresh").
* **`pm_wallet_market_summary`** (PK wallet, condition_id): `first_ts, last_ts, n_trades, buy_usd, sell_usd,
  redeem_usd, merge_usd, split_usd, net_cashflow_usd`, plus `market_id, category, market_closed_time,
  market_resolution_status`. **Point-in-time rule:** for a feature at time t, use rows with `first_ts < t`;
  for prior PnL / win rate, only markets with `market_closed_time < t`. Note the aggregates include activity
  after t within those markets (e.g. a redemption after t of a market closed before t) – the realized outcome
  is fixed by then, but trades after t in still-open markets must not be used (filter on `market_closed_time`).
  Prior-history coverage exists only for wallets in `pm_wallets` (those with a >= $1k in-scope taker trade).
* **`pm_price_history`** (market_id, outcome_index, fidelity_min ∈ {1440, 60}, ts, price).
* **`kalshi_markets`** (ticker, series, title, rules, `result` yes/no, open/close/settlement times, volume,
  our taxonomy columns `pm_category, geo_subtype, in_scope, macro_comparison`), **`kalshi_trades`**
  (trade_id, ticker, ts, yes/no price, count, taker_side). No identities exist on Kalshi.
* **`ingest_state`** (source, key, status, cursor, n_rows): checkpoints/coverage of every unit of work.

## Resolution rules (`resolution.py`)

`resolved` only if the market is closed, UMA status (when present) is `resolved`, exactly one outcome pays
>= 0.99 and all others <= 0.01. All-≈0.5 → `ambiguous_5050`; all-zero → `cancelled_or_void`; anything else
(UMA proposed/disputed, non-corner prices) → `unresolved`. Nothing is guessed.

## Taxonomy (`taxonomy.py`)

Ordered transparent rules (first match wins) over Polymarket tags, the legacy Gamma `category` field and
keyword regexes on question + event title. Categories: geopolitical, macro_central_bank,
elections_domestic_politics, sports, crypto, entertainment, other. Geo sub-types: military, treaty_ceasefire
(incl. diplomacy/summits/visits), sanctions (incl. tariffs/trade measures), leadership_change, emergency_policy
(executive orders, emergency powers, pardons, closures), other. `in_scope` = geopolitical and not a scheduled
election. US executive actions (tariffs, executive orders, pardons, strikes, cabinet departures) count as
government decisions → in scope; US domestic legislative/legal/polling markets do not. Markets that carry geo
tags but resolve on prices, AI rankings or prizes (oil "hit $X", Nobel) are `other`. Scheduled elections are
flagged `is_scheduled_election` (geopolitical category for national/foreign elections, else
elections_domestic_politics) so they can be added back. `macro_comparison` = central-bank decision markets
(Fed/ECB/BoE/...); statistical releases (CPI, jobs, PMI) are macro category but not in the comparison set.
Hand-check results: see "Coverage" below and `artifacts/data/taxonomy_handcheck.csv`.

## Coverage (snapshot 2026-11-17 21:30 UTC; live numbers in `artifacts/data/coverage_summary.json`)

* **Markets**: 5,538,585 closed Polymarket markets (all Gamma `closed=true` markets with an end date;
  close times 2020-10-25 → 2026-11-17). Resolution: 5,464,400 resolved, 73,015 ambiguous_5050 (mostly sports
  props), 1,015 unresolved, 155 cancelled_or_void. Gamma itself holds few 2020–21 markets (sparse ids). A spot
  check by id found ~3% of markets in one id block absent from keyset results (NCAA placeholder markets) – small,
  not chased. Markets with a NULL end date cannot be reached by the date-window partitioning.
* **In scope**: 9,706 closed markets (9,700 resolved, 3 ambiguous), $7.76B Gamma volume. Sub-types:
  military 3,540, other 2,496, treaty_ceasefire 1,538, leadership_change 1,299, emergency_policy 520,
  sanctions 313. Macro comparison set: 1,321 markets. Scheduled elections (excluded): 16,687 resolved.
* **Taker trades**: all 11,388 queued markets (in-scope + macro + some since-reclassified) fetched to completion:
  22.46M taker rows, 2023-11-18 → 2026-11-17; in-scope: 16.9M rows, 9,583 of 9,706 markets have >=1 trade,
  978,553 distinct taker wallets. Wallet attribution: 100% of rows carry a proxy wallet.
  **Hard limit: the Data API has no trades before ~2023-11-18** (rolling retention), so in-scope markets that
  closed earlier (~76) have no trades and markets straddling the boundary are truncated at the start.
* Gamma `volume_usd` is typically 2–8× the sum of taker notional (price × size) for the same market (Gamma
  appears to count both sides / share volume). Use trade-level notional for flow features, not Gamma volume.
* **Maker fills**: validated on one market (taker size per tx matches the sum of maker sizes in 99.7% of txs;
  makers ≈ 2× taker rows). The `fills` stage is queued after `wallets` and stops at 8 GB free disk – expect only
  the highest-volume in-scope markets to be covered. Check `pm_market_trade_stats.maker_complete`.
* **Wallet histories**: queued for 50,906 wallets with a >= $1k taker trade in an in-scope market; running
  (~3–4 wallets/s, several hours). Check `--status` (`pm_wallet_activity` units) and `pm_wallets`.
  First-seen is left-censored at ~2023-11-18 (`pm_wallets.pre_window_activity` flags wallets active earlier).
* **Price history**: queued after fills (daily + hourly bars per outcome token).
* **Kalshi**: 481 of 3,570 World/Politics/Economics series expanded (title/tag pre-filter because of severe 429
  rate limiting); markets stage in progress (4,404 markets so far), trades stage follows automatically.
* **Not available**: wallet funding sources (on-chain hosts blocked), historical order books, pre-2023-11 trades,
  AMM-era (2020–22) trades.

## Taxonomy hand-check (`artifacts/data/taxonomy_handcheck.csv`)

| sample | n | errors | estimate |
|---|---|---|---|
| random in-scope, rules v1-pre | 120 | 8 FP (pageants/Eurovision, celebrity meetings with Trump, Musk-TikTok, Trump arrest) | precision 93% → rules fixed |
| random out-of-scope (all categories) | 100 | 0 FN (99% sports/crypto) | — |
| hard negatives (non-sports/crypto out-of-scope), pre | 150 | 3 FN (US cabinet departures, "next UK PM" tagged as election) | fixed |
| **random in-scope, final rules** | 110 | 2 FP (Miss France – since fixed; a snap presidential election) | **precision 98.2%** (95% CI ≈ 93.6–99.8%) |
| **hard negatives, final rules** (pool 56k political/macro/entertainment out-of-scope) | 120 | 1 FN (tariff decision routed to macro – since fixed) | FN rate 0.8% → ≈470 missed markets → **recall ≈ 95%** (wide CI, ≈ 80–99%) |

Sports/crypto pools (5.2M markets) contained no in-scope markets in the random sample. Borderline calls we
count as in scope: ship-transit / UKMTO-notice count markets around Hormuz/Bab el-Mandeb, "Trump signs an
executive order on <date>", court rulings on executive orders. Downstream roles can filter on `geo_subtype` or
`taxonomy_rule` to tighten.
