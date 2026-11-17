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

## Coverage

See `artifacts/data/coverage_summary.json` (regenerated by `python -m pmflow.data.export`) and the final
section below, updated at hand-off.
