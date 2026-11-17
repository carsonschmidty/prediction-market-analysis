SET client_min_messages = warning;
-- pmflow data schema (owned by the data agent: src/pmflow/data/).
-- Idempotent: every statement is CREATE ... IF NOT EXISTS / ADD COLUMN IF NOT EXISTS.
-- Conventions:
--   * all timestamps are timestamptz (UTC); prices are in [0,1] USDC per share; sizes in shares.
--   * wallets are opaque lower-case 0x addresses. We deliberately do NOT store display names,
--     pseudonyms, bios or profile images that the Polymarket Data API returns alongside trades.
--   * other roles create their own tables with lbl_/feat_/mdl_/eval_ prefixes.

-- ---------------------------------------------------------------- ingestion state
CREATE TABLE IF NOT EXISTS ingest_state (
    source      text        NOT NULL,          -- e.g. 'gamma_markets', 'pm_trades_taker', 'kalshi_trades'
    key         text        NOT NULL,          -- unit of work: a date window, a condition id, a wallet ...
    status      text        NOT NULL DEFAULT 'pending',  -- pending | running | done | error | partial
    cursor      text,                          -- last pagination cursor (resume point)
    n_rows      bigint      NOT NULL DEFAULT 0,
    priority    double precision,              -- higher = earlier (e.g. market volume)
    error       text,
    extra       jsonb,
    updated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, key)
);
CREATE INDEX IF NOT EXISTS ingest_state_status_idx ON ingest_state (source, status, priority DESC);

-- ---------------------------------------------------------------- Polymarket markets
CREATE TABLE IF NOT EXISTS pm_markets (
    market_id            bigint PRIMARY KEY,   -- Gamma market id
    condition_id         text,                 -- CTF condition id (0x...), key for trades
    question_id          text,
    slug                 text,
    question             text,
    description          text,
    outcomes             jsonb,                -- e.g. ["Yes","No"]
    clob_token_ids       jsonb,                -- token ids aligned with outcomes
    outcome_prices       jsonb,                -- final/last outcome prices as reported by Gamma
    event_id             bigint,
    event_slug           text,
    event_title          text,
    tag_labels           text[],
    tag_slugs            text[],
    gamma_category       text,                 -- legacy 'category' field (old markets only)
    market_type          text,
    neg_risk             boolean,
    neg_risk_market_id   text,
    created_at           timestamptz,
    start_date           timestamptz,
    end_date             timestamptz,          -- scheduled end
    closed_time          timestamptz,          -- actual close (usually ~resolution time)
    active               boolean,
    closed               boolean,
    archived             boolean,
    volume_usd           double precision,
    volume_clob_usd      double precision,
    liquidity_usd        double precision,
    last_trade_price     double precision,
    uma_resolution_status text,
    resolved_by          text,
    resolution_source    text,
    automatically_resolved boolean,
    -- derived resolution (see pmflow.data.resolution)
    resolution_status    text,     -- resolved | ambiguous_5050 | unresolved | cancelled_or_void | multi_outcome_unclear
    winning_outcome_index integer, -- index into outcomes; NULL unless resolution_status='resolved'
    winning_outcome      text,
    resolved_yes         boolean,  -- for Yes/No markets: TRUE if 'Yes' won; NULL otherwise
    -- derived taxonomy (see pmflow.data.taxonomy)
    category             text,     -- geopolitical | macro_central_bank | elections_domestic_politics | sports | crypto | entertainment | other
    geo_subtype          text,     -- military | leadership_change | sanctions | treaty_ceasefire | emergency_policy | other (geopolitical only)
    is_scheduled_election boolean,
    in_scope             boolean,  -- core set: geopolitical/government-decision, not scheduled elections
    macro_comparison     boolean,  -- secondary comparison set (central bank / macro decisions)
    taxonomy_rule        text,     -- which rule fired (transparent audit trail)
    taxonomy_version     text,
    raw                  jsonb,    -- trimmed Gamma payload (no images)
    ingested_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS pm_markets_condition_idx ON pm_markets (condition_id);
CREATE INDEX IF NOT EXISTS pm_markets_scope_idx ON pm_markets (in_scope, volume_usd DESC);
CREATE INDEX IF NOT EXISTS pm_markets_category_idx ON pm_markets (category);
CREATE INDEX IF NOT EXISTS pm_markets_end_idx ON pm_markets (end_date);
CREATE INDEX IF NOT EXISTS pm_markets_event_idx ON pm_markets (event_id);

-- ---------------------------------------------------------------- Polymarket trades
-- Source: data-api.polymarket.com/v2/trades?condition=<conditionId>.
-- Stored compactly (disk is the binding constraint: ~250 B/row instead of ~1 KB): hashes/addresses as
-- bytea, market as Gamma market_id, outcome as index into pm_markets.outcomes / clob_token_ids.
-- Use the view pm_trades_v for readable text columns.
-- role 0 = taker: default call (takerOnly=true); the aggressor side of each match. One row per
--                (tx, taker wallet, token), sizes aggregated over the maker orders it hit.
-- role 1 = maker: from the takerOnly=false call (taker AND maker rows), minus rows identical to a stored
--                taker row -> the passive counterparties. Only present for markets whose fills pass ran
--                (ingest_state source='pm_trades_fill', status='done').
CREATE TABLE IF NOT EXISTS pm_trades (
    key_hash       uuid        NOT NULL,   -- md5(tx|token|wallet|side|price|size|role): natural unique key
    market_id      bigint      NOT NULL,   -- -> pm_markets.market_id
    outcome_index  smallint    NOT NULL,   -- position of the traded token in pm_markets.clob_token_ids
    is_buy         boolean     NOT NULL,   -- side from this wallet's perspective
    price          double precision NOT NULL,
    size           double precision NOT NULL,   -- shares; USD notional = price*size
    wallet         bytea       NOT NULL,   -- 20-byte proxy wallet (opaque id)
    tx_hash        bytea       NOT NULL,
    ts             timestamptz NOT NULL,
    role           smallint    NOT NULL    -- 0 taker, 1 maker
);
CREATE UNIQUE INDEX IF NOT EXISTS pm_trades_key_uq ON pm_trades (key_hash);
CREATE INDEX IF NOT EXISTS pm_trades_market_ts_idx ON pm_trades (market_id, ts);
CREATE INDEX IF NOT EXISTS pm_trades_wallet_ts_idx ON pm_trades (wallet, ts);

CREATE OR REPLACE VIEW pm_trades_v AS
SELECT t.key_hash, t.market_id, m.condition_id, t.outcome_index,
       m.clob_token_ids ->> t.outcome_index AS token_id,
       m.outcomes ->> t.outcome_index AS outcome,
       CASE WHEN t.is_buy THEN 'BUY' ELSE 'SELL' END AS side,
       t.price, t.size, t.price * t.size AS usd_notional,
       '0x' || encode(t.wallet, 'hex') AS wallet, '0x' || encode(t.tx_hash, 'hex') AS tx_hash, t.ts,
       CASE t.role WHEN 0 THEN 'taker' ELSE 'maker' END AS role
FROM pm_trades t JOIN pm_markets m USING (market_id);

DROP VIEW IF EXISTS pm_trades_maker;

-- ---------------------------------------------------------------- Polymarket price history
CREATE TABLE IF NOT EXISTS pm_price_history (
    market_id     bigint   NOT NULL,
    outcome_index smallint NOT NULL,  -- token = pm_markets.clob_token_ids[outcome_index]
    fidelity_min  smallint NOT NULL,  -- bar size requested from CLOB (1440 = daily, 60 = hourly)
    ts            timestamptz NOT NULL,
    price         real NOT NULL,
    PRIMARY KEY (market_id, outcome_index, fidelity_min, ts)
);

-- ---------------------------------------------------------------- Polymarket wallet histories
-- Source: data-api.polymarket.com/v2/activity?user=<wallet>&sortDirection=ASC (TRADE, REDEEM, SPLIT, MERGE,
-- REWARD, CONVERSION ...) across ALL Polymarket markets, not only in-scope ones. Fetched oldest-first with a
-- page cap (default 10 x 1000 rows): first-seen is exact; for capped wallets (activity_complete=false) the
-- per-market summary covers only their earliest activity. Raw activity rows are aggregated, not stored.
-- one row per wallet whose activity we fetched (coverage bookkeeping)
CREATE TABLE IF NOT EXISTS pm_wallets (
    wallet              bytea PRIMARY KEY,
    activity_fetched_at timestamptz,
    activity_complete   boolean,     -- FALSE if we hit the per-wallet page cap (first-seen then unknown)
    n_activity_rows     integer,
    n_markets           integer,     -- distinct markets touched (within fetched activity)
    first_activity_ts   timestamptz, -- earliest activity of any type returned by /v2/activity
    first_trade_ts      timestamptz
);

-- Per (wallet, market) aggregate of the wallet's activity; market_* columns filled by pmflow.data.derive.
-- Point-in-time use: a feature at time t may only use rows with first_ts < t and, for PnL / win
-- flags, only markets whose market_closed_time < t.
CREATE TABLE IF NOT EXISTS pm_wallet_market_summary (
    wallet          bytea NOT NULL,
    condition_id    text NOT NULL,   -- lower-case 0x hex, joins pm_markets.condition_id
    market_id       bigint,
    category        text,
    first_ts        timestamptz,
    last_ts         timestamptz,
    n_trades        integer,
    buy_usd         double precision,
    sell_usd        double precision,
    redeem_usd      double precision,
    merge_usd       double precision,
    split_usd       double precision,
    market_closed_time timestamptz,
    market_resolution_status text,
    net_cashflow_usd double precision,   -- sell+redeem+merge-buy-split: realized PnL once resolved and redeemed
    PRIMARY KEY (wallet, condition_id)
);

-- ---------------------------------------------------------------- Kalshi (price/volume only, no identity)
CREATE TABLE IF NOT EXISTS kalshi_markets (
    ticker            text PRIMARY KEY,
    event_ticker      text,
    series_ticker     text,
    title             text,
    subtitle          text,
    yes_sub_title     text,
    rules_primary     text,
    category          text,     -- Kalshi series category
    market_type       text,
    status            text,
    result            text,     -- yes | no | '' (void/unknown)
    open_time         timestamptz,
    close_time        timestamptz,
    settlement_ts     timestamptz,
    volume            double precision,
    open_interest     double precision,
    last_price        double precision,
    pm_category       text,     -- our taxonomy applied to Kalshi text
    geo_subtype       text,
    in_scope          boolean,
    macro_comparison  boolean,
    raw               jsonb,
    ingested_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS kalshi_markets_series_idx ON kalshi_markets (series_ticker);

CREATE TABLE IF NOT EXISTS kalshi_trades (
    trade_id      text PRIMARY KEY,
    ticker        text NOT NULL,
    ts            timestamptz NOT NULL,
    yes_price     double precision,
    no_price      double precision,
    count         double precision,
    taker_side    text,
    is_block_trade boolean,
    ingested_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS kalshi_trades_ticker_ts_idx ON kalshi_trades (ticker, ts);

CREATE TABLE IF NOT EXISTS kalshi_series (
    series_ticker text PRIMARY KEY,
    title         text,
    category      text,
    tags          text[],
    frequency     text,
    raw           jsonb
);
