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

## [2026-11-17 21:35 UTC] evaluation — eval_snapshots grid + FIXED splits are ready (use these, do not redefine)
Build: `.venv/bin/python -m pmflow.evaluation.snapshots` (idempotent, ~1 min). Tables:
- **`eval_snapshots`** (1.51M rows, PK `snapshot_id`): `snapshot_id, market_id, event_id, universe` ('in_scope'|'macro'),
  `geo_subtype, neg_risk, y` (1 = YES/outcome-0 won), `cutoff_ts, grid_24h, price_yes` (last TAKER trade before cutoff,
  YES-equivalent: outcome-1 trades mapped to 1-p), `vwap_6h, vwap_24h, last_trade_age_h, n_trades_24h, usd_24h,
  n_trades_7d, usd_7d, n_trades_cum, usd_cum, hours_since_open, pre_trade_window_market, split, is_live`, and two
  LEAKY diagnostic columns `hours_to_close_diag, hours_to_end_date_diag` (never use as features/filters).
- Grid: fixed UTC calendar grid every 6h (00/06/12/18), from first taker trade until the market's `closed_time`
  (open-ness is knowable live). **Rule: a predictor for a snapshot may only use data with ts < cutoff_ts.**
- **Primary evaluation set = `is_live AND split IN ('train','val','test')`**. is_live (all computable at cutoff):
  price_yes in [0.03,0.97], last trade <= 72h ago, >= 3 taker trades and >= $200 taker notional in last 7 days.
- **Splits (by event_id; event_close = max closed_time of its resolved in-scope/macro markets):**
  train = events closing < 2026-04-01; val = events closing in [2026-04-01, 2026-07-01) AND cutoff >= 2026-04-01;
  test = events closing >= 2026-07-01 AND cutoff >= 2026-07-01. Snapshots of val/test events before their period
  start are `purged_val` (may be used only to refit a final model on train+val) / `purged_test` (never use).
  Table `eval_event_splits(event_id, event_close, split, ...)`. No event straddles splits; no training label resolves
  after any val/test cutoff.
- Live counts (in-scope): train 378,644 snaps / 4,099 mkts / 2,081 events; val 43,713 / 1,019 / 281; test 105,302 /
  2,003 / 541. Macro: train 81,948/475/179; val 9,750/128/48; test 31,389/363/95. Use `grid_24h` (00:00 UTC only)
  for cheaper feature development (~1/4 of rows).
- Predictions for the harness: a table/parquet `(snapshot_id, p_model)`; see `docs/evaluation_baseline.md`.
- `eval_diag_snapshots`: DIAGNOSTIC 1/6/24/72h-before-closed_time / last-trade anchors. Leaky anchors, never for model selection.

## [2026-11-17 21:50 UTC] evaluation — Baseline A done; harness + leakage audit ready (docs/evaluation_baseline.md)
- Baseline A = raw last-trade price, primary live snapshots, event-clustered 95% CIs. In-scope test: Brier 0.1249
  [0.1121, 0.1394], log loss 0.398 [0.365, 0.436]. Macro test: Brier 0.2193 [0.1806, 0.2607]. Climatology test: 0.229 / 0.286.
- **Recalibrated price (Platt/isotonic fitted on train) does NOT beat raw price out of sample.** In-scope test ΔBrier
  −0.0001 [−0.0013, +0.0010]; worse on val (−0.0050) and on macro test (−0.0146). Raw price is the bar. Also compare
  models with `eval_baseline_preds.p_iso` so that "re-learned calibration" does not count as a flow signal.
- MDE (α=.05, power .8): in-scope test ≈ 0.0012–0.0016 Brier (~1% relative) for models moving prices ~2pp; macro test ≈ 0.007–0.011.
- Trading hurdle: reconstructed fills cost ≈ 1c spread + ≈1.6c impact at $100. The calibration-edge reference
  strategy loses 9% per $ on in-scope test (−6.7% even without impact).
- Harness: `python -m pmflow.evaluation.harness --pred <parquet|table:...> --name <run> [--trade]`. Leakage audit:
  `pmflow.evaluation.leakage.audit_sql_feature / audit_python_feature / check_source_timestamps`. Features/modeling:
  please run the audit on your features and log it.
- Caveat: val/test are dominated by the 2026 Iran/Middle-East conflict cluster. Cross-event correlation means the CIs
  are optimistic. Val (Apr–Jun 2026) is a YES-heavy regime (base rate 0.43 vs mean price 0.32).

## [2026-11-17 21:50 UTC] features — base tables + feature library ready; snapshot build running
- `feat_market_meta` (corpus = in_scope OR macro_comparison, 11,027 markets) and `feat_wallet_hist` (1.22M wallets,
  438 MB: per-wallet sorted arrays for point-in-time first-seen / prior markets / realised PnL on markets closed before
  the cutoff / cumulative notional by UTC day). Build: `python -m pmflow.features.base --rebuild` (~1 min).
- Library: `pmflow.features.core.compute_window_features(market_id, cutoff_ts)` / `compute_features_batch(df)`.
  Leakage test (`tests/features/test_leakage.py`) passes: features on full DB == features on a DB copy truncated at
  the cutoff, 8 (market, cutoff) pairs, all 78 columns; negative control (truncate 1h earlier) changes 60/78 columns.
- `feat_snapshot_features` (keyed by eval_snapshots.snapshot_id) building now for `grid_24h AND is_live AND
  split <> 'purged_test'` (178k rows, ~170 MB). The 6-hourly live grid would be ~0.7 GB; not built (disk).
- GAP (data agent): case markets with NO trades because out of corpus: PM-003 Nobel/Machado 560868 (taxonomy
  `other`), PM-018/019/020/022 (out of scope, expected). PM-012 market (Gamma 1466015, "US x Iran ceasefire by
  March 31?") and PM-021 (682705) are not in pm_markets at all. If trades for 560868 and 1466015 can be ingested,
  the case study gains one seed positive and the key negative.
- pm_wallets coverage at 21:45 UTC: 769 wallets (550 activity_complete) of ~51k queued. Full-history features
  (`full_cov_share`, `fws72_full`, `dom_log_prior_mkts_full`) reflect coverage at build time only.

## [2026-11-17 22:15 UTC] features — snapshot features, weak labels and case study done (docs/features.md)
- `feat_snapshot_features` (150 MB, PK snapshot_id): 178,378 rows = eval_snapshots `grid_24h AND is_live AND
  split <> 'purged_test'`; 15 cutoff-level + 56 window features x {1,6,24,72}h (`<name>_<L>h`). Built from
  trades with ts < cutoff_ts only; leakage tests pass (incl. stored rows vs truncated-DB recompute). The 6-hourly live
  grid is NOT materialised (~0.6 GB; disk). Rebuild/extend: `python -m pmflow.features.snapshots [--set live]`.
  Sample parquet: artifacts/features/feat_snapshot_features_sample2pct.parquet.
- `feat_weak_labels` (48 MB): labeling-plan conditions a-d + pattern flags (no outcome) and L1/N1/L2/N2 label
  columns (outcome; label only). In-scope, summed over lookbacks: strong pattern L1/N1 = train 6/24, val 1/0,
  test 2/0; it fires on <=0.024% of snapshots. Hit rates of possible/loose patterns ~ equal the dominant side's entry
  price (excess -0.02..+0.06 train): no evidence the pattern beats the price.
- Case study (taker data): the labeling-plan footprint fires in 0 of 64 pre-news case windows; pre-news windows
  show elevated volume / new-wallet counts but NOT one-sided fresh flow; the dominant taker side was the eventual
  winner in only 11/44 positive-case windows. PM-001's reported ~$33k fresh-account YES buying is visible (one new
  wallet, $32.5k at 0.074); PM-007/PM-011 "fresh wallet" claims are not visible in taker trades.
- Modeling: wallet features are corpus-relative (fresh = new to geopolitical/macro markets); full-history columns
  (`full_*`) cover only ~2k wallets. Prefer `fresh_nsn_*`, `dom_*`, `whale_*` with care; all thresholds untuned.

## [2026-11-17 23:05 UTC] modeling — stages 1-6 done on TRAIN/VAL; result: informed-flow features add nothing over price (docs/modeling.md)
- Rows: 24h-grid primary snapshots (train 115k / val 13.5k / test 34k). Baseline A reproduced exactly on the 6h grid;
  on our rows in-scope val Brier 0.1777, macro val 0.1055.
- Market-anchored models (price logit as fixed offset; LightGBM init_score, shallow + strong L2; rounds chosen on
  event-grouped walk-forward folds inside train). On VAL no flow / wallet / weak-label-flag / context group beats the
  same LightGBM given price only (lgbm_all - lgbm_price_only: in-scope -0.0007 [-0.0014,-0.0001], macro -0.0011
  [-0.0022,+0.0001] Brier); a shuffled-feature placebo reproduces the val "gain" over raw price (+0.0030 vs +0.0033);
  flow-only ridge (no intercept) +0.0002 [0.0000,0.0003]. With per-snapshot weights walk-forward picks 0 rounds.
- FINDING for evaluation (please read docs/modeling.md §7): window-selection bias. Val and each walk-forward fold
  contain only events that closed inside the window; YES "by date" markets close early, so these sets are YES-heavy
  (market-weighted ybar - price: all train +0.014, walk-forward OOF rows +0.062, val +0.068; train snapshots <=7d
  before event_close +0.098 vs >180d -0.080, diagnostic only). Any intercept / YES-shift fitted on such rows "wins"
  val (stack with free intercept: +0.0097) without flow information. Test (resolved by 2026-11-17) is built the same
  way. Candidates were therefore restricted to no-free-intercept models, and a price-only control is shipped.
- Exploratory system one (gate NOT passed; ~75 CPU-min): ModernBERT encoder of gliclass-instruct-edge-v1.0 with a
  fresh residual head, run in JAX (validated vs onnxruntime, max |logit diff| 2e-4). Question-only text does as well
  as text+flow descriptors on val (topic/regime tilt, no train-OOF gain); descriptors add nothing.
- BLOCKER (worked around, logged): PyTorch not installable — PyPI wheel is CUDA-13 only and needs ~4 GB nvidia libs
  (disk), download.pytorch.org is proxy-blocked (403). GLiClass library and tasksource-decider-nano zero-shot not run.
- Pre-registered (3): PRIMARY preds_lgbm_anchored, preds_stack_s1 (no intercept), preds_ridge_flow; control
  preds_ctrl_lgbm_price_only. All in artifacts/modeling/, all splits incl. test. No test metric computed by modeling.

## [2026-11-17 23:25 UTC] orchestrator — pm_markets trimmed to scoped categories to free disk (user-approved)
- Dropped 5.50M out-of-scope `pm_markets` rows (sports, crypto, entertainment, other). Kept 36,350: every
  geopolitical / elections / macro row, all in-scope (9,706) and macro-comparison (1,321) markets, every market
  with trades or evaluation snapshots, and the Nobel case market 560868. Table 5.4 GB -> 112 MB; free disk 13 GB.
- Full-universe counts by year can no longer be recomputed from Postgres; the committed
  `artifacts/data/market_counts_by_year*.csv` are the record. Rebuilding needs a full Gamma metadata re-ingest.
- A future `ingest_gamma --resume` only adds newly closed markets (all categories); re-trim if disk gets tight.
- Maker-fills stage started (3 workers, alongside the wallets stage); it still stops below 8 GB free.
