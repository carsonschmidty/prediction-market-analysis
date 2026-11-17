# Features (owner: features agent, `src/pmflow/features/`)

Point-in-time "informed-flow" window features from public Polymarket taker trades. Wallets are
opaque IDs; nothing here links them to people, and no report lists wallet addresses.

## Quick start / rebuild

```bash
.venv/bin/pip install -e '.[features]'
.venv/bin/python -m pmflow.features.base --rebuild          # feat_market_meta + feat_wallet_hist (~1 min)
.venv/bin/python -m pmflow.features.snapshots --workers 2   # feat_snapshot_features (resumable, ~7 min)
.venv/bin/python -m pmflow.features.weak_labels             # feat_weak_labels + artifacts/features/weak_label_summary.csv
.venv/bin/python -m pmflow.features.case_study              # case study CSVs + plots
.venv/bin/python -m pytest -q tests/features                # unit + leakage tests (needs Postgres)
```

Library: `from pmflow.features import compute_window_features, compute_features_batch, to_wide`.

* `compute_window_features(market_id, cutoff_ts, lookbacks=(1, 6, 24, 72))` gives one row per lookback.
* `compute_features_batch(df_or_pairs)` takes a frame with `market_id` and `cutoff_ts`, plus any extra
  key columns such as `snapshot_id`. It runs market by market: one trade load and one wallet-history
  fetch per market, then vectorised as-of lookups.
* `to_wide(long)` returns one row per key with `<feature>_<L>h` columns.

## Tables and files

| object | contents | size |
|---|---|---|
| `feat_market_meta` | corpus markets (in_scope OR macro_comparison, 11,027): static metadata. The resolution columns are used only for wallet PnL on markets closed before a cutoff | <10 MB |
| `feat_wallet_hist` | 1.22M wallets: per-wallet sorted arrays (first in-corpus trade, first in-scope trade of $1k or more, corpus markets with first-trade times, closed_time and taker PnL of resolved markets, cumulative notional by UTC day) | 438 MB |
| `feat_snapshot_features` | wide, PK `snapshot_id` (= eval_snapshots), 178,378 rows: `grid_24h AND is_live AND split <> 'purged_test'`. 15 cutoff-level columns + 56 window features × 4 lookbacks (242 columns, float4) | 150 MB |
| `feat_snapshot_build_state` | resumability bookkeeping (market_id, snapset) | tiny |
| `feat_weak_labels` | PK `snapshot_id`: window quantities, conditions a–d, pattern flags (no outcome), and label columns (outcome) per lookback | 48 MB |
| `artifacts/features/feat_snapshot_features_sample2pct.parquet` | deterministic sample `snapshot_id % 50 = 0` (7,033 rows). The full table is 150 MB in Postgres, too big for artifacts; rebuild with the command above | 5.9 MB |
| `artifacts/features/feat_weak_labels.parquet` | all 178k rows; ids, split, conditions, flags, labels | 1.2 MB |
| `artifacts/features/case_*.csv`, `*.png`, `pattern_base_rates.csv`, `weak_label_*.csv` | case study, base rates, weak-label counts | small |

The full 6-hourly live grid (`--set live`, about 713k snapshots, about 0.6 GB) was **not** built
because of disk (about 6 GB free, shared). The command is resumable if disk is freed.

Notes on the snapshot table:

* Postgres lower-cases column names, so `burst_B` is stored as `burst_b_<L>h`.
* `px_last` matches `eval_snapshots.price_yes` except where several trades share the same second.
  There the max difference is 0.02, from tie order; mean |diff| is 1e-5.

## As-of rules (how leakage is prevented)

For a cutoff c, every feature uses rows with ts < c only:

* **Market trades.** Window [c − L, c) on taker rows (role 0) of the market. Baselines are windows that
  end at or before c − L.
* **Wallet first-seen.** The wallet's first taker trade in any corpus market (`feat_wallet_hist.first_t`).
  For a trade at t this is ≤ t by construction.
* **Prior markets.** The number of corpus markets whose first trade by the wallet was before c,
  excluding the current market.
* **Prior win rate / PnL.** Only markets with closed_time < c that resolved cleanly. PnL is computed from
  the wallet's taker trades with ts < closed_time: cash flow plus the payout of net shares of the winner.
  Win means PnL > 0.
* **Concentration (CONC).** The wallet's notional in this market before c, divided by its cumulative
  corpus notional through the last UTC day ending ≤ c plus its same-day notional in this market. The
  denominator misses the wallet's same-day trades in *other* markets, so CONC is slightly overstated.
  That is a bias, not a leak.
* **Full Polymarket history** (`pm_wallets`, `pm_wallet_market_summary`). Used only if the wallet's
  first in-scope trade of $1k or more (the ingestion selection rule) happened before c and
  `activity_complete`. Only summary rows with first_ts < c are used. Realised PnL from the summary is
  **not** used, because it contains post-cutoff redemptions.
* **Market metadata.** created_at, the scheduled end_date, category and geo_subtype only. closed_time
  and resolution of the market itself are never used.
  * Caveat: Gamma's end_date is the final value. If a market's end date was edited, the value seen live
    may have differed. We cannot reconstruct that.

**Leakage test** (`tests/features/test_leakage.py`): **passes**.

* **Setup.** For 8 (market, cutoff) pairs, we built session-temp copies of the DB truncated at the cutoff:
  * trades of the market and of every wallet that traded it, ts < c;
  * markets that closed at or after c have their resolution nulled;
  * pm_wallets and the summary filtered to first_ts < c;
  * `feat_wallet_hist` rebuilt from the truncated copy with the production builder.
* **Samples.** 4 case markets at their news edge, including a non-round cutoff second, and 4 random
  markets at 60% of their life.
* **Result.**
  * All 78 output columns are identical between the full DB and the truncated copy.
  * A batch with a later cutoff added gives identical rows for the earlier cutoff.
  * 5 random stored `feat_snapshot_features` rows equal a truncated-DB recomputation, within float4.
* **Negative control.** Truncating 1h before the cutoff changes 60 of 78 columns, so the comparison is
  sensitive.
* **Bug found by this test.** In the first build, the burstiness column was stored NULL (a column-case
  bug). It was fixed and the table rebuilt.

## Conventions

* **YES-equivalent.** Outcome index 0 = YES. For the about 0.3% of corpus markets that are not Yes/No,
  it is the first listed outcome. The trade's YES direction is d = +1 for BUY outcome 0 or SELL
  outcome 1, else −1. Signed notional = d × price × size, the USD actually exchanged. YES-eq price =
  p for outcome 0 and 1 − p for outcome 1.
* **dom** = sign(NSN) of the window. "Dominant side" means trades with d = dom.
* **Robust z.** z = (x − median(base)) / max(1.4826·MAD, 1.2533·mean|dev|, $1).
  * The base needs at least 3 windows, otherwise z is NaN.
  * The floor stops quiet markets (MAD = 0) from producing infinite z.
* **Fresh (in-sample)** = first in-corpus taker trade less than 72h before the trade.
  * This means "new to geopolitical and central-bank markets since 2023-11-18", **not** new to
    Polymarket.

## Feature dictionary

Cutoff-level columns, one per snapshot:

| name | definition | coverage |
|---|---|---|
| hours_since_open | (c − created_at)/h | all |
| hours_since_first_trade | c − first taker trade (truncated at 2023-11-18) | if any trade < c |
| hours_to_end | scheduled end_date − c (can be negative; end_date is final Gamma value) | all |
| hour_utc, dow_utc | of c | all |
| wallet_age_censored | c < 2024-01-01: wallet-age/fresh features unreliable (left-censored first-seen) | flag |
| market_history_truncated | market created before ~2023-11-19: early trades missing | flag |
| px_last | last taker trade YES-eq price before c | if any trade |
| log_gv_total, log_gv_7d | log1p taker notional since first trade / last 7 days | all |
| n_trades_total, n_wallets_total | taker trades / distinct taker wallets before c | all |
| category, geo_subtype, in_scope | taxonomy (ex ante) | all |

Window columns, suffixed `_1h`, `_6h`, `_24h` or `_72h`. Each uses window W = [c − L, c).

| family | name | definition |
|---|---|---|
| flow | n_trades, n_wallets, gv | count, distinct wallets, gross taker USD |
| flow | nsn | net signed YES-eq notional |
| flow | os | one-sidedness \|nsn\|/gv |
| flow | gv_z, nsn_z, n_base | robust z of gv / nsn vs up to 20 preceding non-overlapping same-length windows |
| flow | gv_ratio_base | gv / median baseline gv |
| flow | wl_z_nsn, wl_n_base, wl_base_med_abs_nsn | labeling-plan z: nsn vs ≤20 windows ending ≥7 days before c; median \|nsn\| of those |
| flow | dom, dom_gv, dom_n_wallets | dominant direction (+1 YES / −1 NO / 0), its notional, its wallets |
| flow | p_entry | share-weighted VWAP of the dominant side, in that side's own price terms (≤0.3 means long odds) |
| flow | vwap_yes | VWAP (YES-eq) |
| size | max_trade, q50_trade, q90_trade | trade-notional distribution |
| size | top5_trade_share | share of gv in the 5 largest trades |
| size | n_ge1k, n_ge10k, share_ge1k | trades ≥$1k/$10k; share of gv from ≥$1k trades |
| timing | burst_B (stored burst_b) | Goh–Barabási burstiness of inter-trade gaps (≥3 trades) |
| timing | peak10m_share | max share of window trades in one 10-minute bin |
| price | dp | YES-eq price change: last trade before c − last trade before window start |
| price | dp_dom | dp × dom (did price move toward the flow) |
| price | impact_per_10k | dp per $10k signed notional (\|nsn\| ≥ $100) — **proxy** for price impact |
| urgency | urg_prem | notional-weighted d × (price − median of previous 20 trades) — **proxy** for spread crossing |
| urgency | urg_share_1c | share of notional paying >1c over that reference |
| wallet | fws72, fws24 | fresh-wallet share of dominant-side notional (first in-corpus trade <72h / <24h before the trade) |
| wallet | fresh_nsn, fresh_nsn_share | net signed notional from fresh trades (both sides); / gv |
| wallet | dom_age_med_d | notional-weighted median in-corpus age (days) on the dominant side |
| wallet | dom_newmkt_share | dominant notional from wallets with 0 prior corpus markets |
| wallet | dom_log_prior_mkts | notional-weighted mean log1p(prior markets) |
| wallet | dom_winrate, dom_winrate_cov | notional-weighted prior win rate among dominant wallets with ≥3 resolved prior markets; share of notional covered |
| wallet | dom_prior_pnl | sum of prior realised taker PnL of dominant wallets |
| wallet | top1_share, top3_share | share of dominant notional from top-1/top-3 wallets |
| wallet | conc_top3 | notional-weighted CONC of the top-3 dominant wallets — **proxy**: share of the wallet's lifetime-to-date notional in this market |
| wallet | conc80_share | dominant notional from wallets with CONC ≥ 0.8 |
| co-timing | n_new_wallets_dom | wallets whose first-ever corpus trade is in W, in this market, on the dominant side |
| co-timing | n_clusters3, max_cluster | clusters of those wallets with first trades ≤30 min apart (single linkage); number of clusters of size ≥3; largest cluster — **proxy** for coordinated fresh wallets (no funding data) |
| co-timing | cluster_share | dominant notional from wallets in clusters of size ≥3 |
| whale filter | whale_hist_share | dominant notional from high-history wallets (≥20 prior corpus markets) |
| whale filter | whale_hist_mixed_share | dominant notional from high-history wallets with ≥5 resolved markets and win rate in [0.35, 0.65] ("whales, mixed record") |
| whale filter | hist_good_share | dominant notional from wallets with ≥5 resolved markets and win rate >0.65 |
| whale filter | fresh_conc_share | dominant notional from fresh (<72h at first window trade, or ≤1 prior market) AND CONC ≥0.8 wallets ("fresh, concentrated") |
| full history | full_cov_share | dominant notional covered by an eligible complete pm_wallets row |
| full history | fws72_full, dom_log_prior_mkts_full | freshness / prior markets from full Polymarket history, over covered notional only |

Thresholds are fixed constants in `core.py`; none were tuned on outcomes.

## Weak labels (`feat_weak_labels`, labeling_plan §1–2)

### Quantities and conditions

The quantities per lookback are NSN, GV, OS, p_entry, FWS = fws72, TOP3 = top3_share,
CONC = conc_top3 and z = wl_z_nsn.

| condition | definition |
|---|---|
| (a) | dom·z ≥ 4, \|NSN\| ≥ 5 × the baseline median \|NSN\|, and \|NSN\| ≥ $10k |
| (b) | OS ≥ 0.7 |
| (c) | p_entry ≤ 0.3 |
| (d) | FWS ≥ 0.5, or (TOP3 ≥ 0.5 and CONC ≥ 0.8) |

(e) holds by construction, because snapshot windows end at the cutoff. Generic snapshots have no news
timestamp.

### Patterns and labels

* **Pattern flags.** These do not use outcomes and can serve as features:
  * `pat_strong` = a & b & c & d
  * `pat_possible` = a & b & (c | d)
  * `pat_loose` = (dom·z ≥ 2, \|NSN\| ≥ $5k) & b & (c | d). This is a sensitivity variant and is not
    in the plan.
* **Labels.** These use y and are **label only**:
  * `lbl_vindicated_L` = the dominant side won
  * `lbl_L1` = strong & vindicated; `lbl_N1` = strong & not vindicated
  * `lbl_L2` / `lbl_N2` are the same for possible. L2 includes L1.
* **Thresholds.** The plan's fixed values. Nothing was tuned on any split.

### Results

Primary snapshot set: grid_24h, live, non-purged.

Firing rate on all 144k in-scope grid snapshots (`pattern_base_rates.csv`):

| | 1h | 6h | 24h | 72h |
|---|---|---|---|---|
| strong | 0.001% | 0.010% | 0.021% | 0.024% |
| possible | 0.04% | 0.13% | 0.26% | 0.35% |
| loose | 0.12% | 0.32% | 0.57% | 0.66% |

Counts, in-scope, summed over the 4 lookbacks. Windows overlap, so one episode counts several times.

| split | L1 | N1 | L2 | N2 |
|---|---|---|---|---|
| train | 6 | 24 | 507 | 363 |
| val | 1 | 0 | 12 | 24 |
| test | 2 | 0 | 25 | 53 |

* **Strong pattern, N1:L1.** Train 4.0, i.e. the strong pattern is right 20% of the time. Val and test
  have only 1 and 2 firings, all right; that is far too few to read anything into.
* **Precision vs price** (`weak_label_precision.csv`). Precision is about what the entry price already
  implies.
  * Strong pattern, train: hit rate 0–40% per lookback against a mean p_entry of 0.08–0.14. That is
    1–11 firings per lookback in 1–10 markets, so it is noise-level.
  * possible/loose, in-scope train: hit rate − mean p_entry = −0.017 to +0.055 per lookback.
  * possible/loose, test: −0.07 to +0.08 per lookback.
  * The val split is a YES-heavy regime, so excesses there are not specific to the pattern.
* **Conclusion.** The labeling-plan pattern is very rare in taker data. When it fires, it is not right
  more often than the market price already implies.

## Case study (task 2): is the reported footprint visible in our taker data?

**Files.**

| file | contents |
|---|---|
| `case_feature_comparison.csv` | long: case × lookback × feature |
| `case_claim_checks.csv` | report claims vs trades |
| `case_skipped.csv` | cases not studied and why |
| `pattern_base_rates.csv` | pattern firing rates in the pool |
| `case_percentiles_24h.png` | percentiles vs matched controls |
| `case_trajectories_72h.png` | 72h pre-news price and signed flow |
| `pattern_rates.png` | pattern rates: cases vs baselines vs pool |

**Matches.** `case_market_matches.csv` was re-run after ingestion: 19 of 25 cases are matched to
pm_markets.

**Studied.** 14 cases with trades in 16 case-market rows (PM-015 spans 3 markets; PM-007/008/013 share
one market). That is 9 positive case IDs in 11 markets, plus the contrasts PM-002, 013, 023 and the
ambiguous PM-017.

**Not studied.**

* PM-003 (Nobel, market 560868): the taxonomy marks it `other`, so it has **no trades**.
* PM-012: the market (Gamma 1466015) is not in pm_markets.
* PM-014, PM-010: no market or timestamp.
* PM-018/019/020/022: out of scope, no trades.
* PM-021: the market is not in pm_markets.
* KX-*: Kalshi.

**Design.**

* **Pre-news cutoff.** The lower edge of `news_window()`. For day-precision cases that is 00:00Z of the
  date, which is conservative.
* **Baselines.** 30 daily cutoffs from the same market, 7–36 days earlier.
* **Matched controls.** The 50 non-case in-scope snapshots nearest on (px_last, log_gv_7d).
* **Pool.** All 144k in-scope grid snapshots.

### Findings (negative, mostly)

1. **The labeling-plan pattern never fires in any pre-news case window.** This holds for strong,
   possible and loose: 0 of 16 × 4 windows. Pool base rates are 0.001–0.66%, so the pattern is
   not just rare in cases. The reaction windows fire once (PM-015 market 514824, loose/possible, 1h).

2. **Volume and attention rise, but that is not one-sided "informed" flow.**
   * Against the same-market baselines, the 11 positive case markets have median percentiles of:
     * gross volume 0.98–1.00;
     * new dominant-side wallets 0.91–1.0;
     * max co-timed cluster 0.91–1.0.
   * Against matched controls these drop to 0.78–0.90.
   * Much of this is markets heating up before their deadline. It is not specific to the cases.
   * One-sidedness is **below** typical (median percentile 0.25–0.44).
   * Fresh-wallet share of dominant notional (fws72) is ordinary (median percentile 0.55–0.76; the
     maximum pre-news fws72 at L ≥ 6h is 0.48, in PM-015 market 514824 at 72h).
   * Top-3 wallet share is unusually low (median percentile vs baselines 0.02–0.08).
   * p_entry is not long-odds: it is above 0.3 in most cases, because the dominant side is usually NO.

3. **The dominant pre-news taker flow was usually on the losing side.**
   * Across the 11 positive markets × 4 lookbacks, the window's dominant side was the eventual winner
     in only **11 of 44** windows (25%).
   * Net flow toward the eventual winner was negative in 8–9 of 11 markets at every lookback. PM-007
     alone saw −$2.9M net in the 72h before the strike reports, and PM-011 −$1.2M.
   * Plausible reasons:
     * steady NO-buying ("deadline decay" carry) at 0.85–0.95 dominates taker flow in "by date X"
       markets;
     * informed buyers who used resting limit orders are invisible in taker-only data.
   * PM-013 (the "whale wrong" NO book) is the dominant pattern in our data for that market.

4. **Fresh wallets did lean toward the winner in some cases, but not reliably.**
   * Fresh-wallet net flow toward the winner over 72h was positive in 6 of 11 positive markets:
     PM-001 +$25k, PM-004 +$43k, PM-005 +$91k, and the three PM-015 markets.
   * It was negative in 5: PM-006, PM-007 (−$320k), PM-009, PM-011 (−$134k), PM-016.
   * In the snapshot pool, the direction of fresh-wallet flow (|fresh_nsn_24h| > $1k) carries no stable
     edge over price. P(y) − mean price:
     * train: +0.03 when fresh flow is YES, +0.00 when NO;
     * test: −0.09 and −0.03.
   * This is a descriptive check, not a model evaluation.

5. **Base rate / false-positive pressure.** Even the loose pattern fires in about 0.3–0.7% of in-scope
   snapshots. On the 6-hourly grid that is roughly 1 firing per 150–300 market-snapshots, and none of
   those firings was a reported case.

### Claim checks

Aggregates only, in `case_claim_checks.csv`. `window_end_rule=reported` uses the report's own end time
and is used only to test the claim; it is not a feature window.

* **PM-001: consistent.**
  * Report: about $33k in 13 bets at 5–12%, from a week-old account active only in Venezuela markets.
  * Data: in the reported window, one YES-side wallet has $32.5k notional in 7 taker trades at VWAP
    0.074. It was new to the corpus at its first window trade (age 0 d, 0 prior corpus markets) and
    started about 78h before the news.
  * It is the only wallet within ±50% of the reported size.
  * The trade count differs (7 vs 13). Maker fills are not in our data.
* **PM-008: partially consistent.**
  * Report: an experienced account, $87k at 0.17, from 71 min before the news.
  * Data: the largest YES wallet in 04:49–06:00Z has $148k at VWAP 0.199. It was first in the corpus
    510 days earlier, has 48 prior corpus markets, and started 44 min before 06:00Z.
  * This shows that an experienced, non-fresh large buyer existed. We cannot and do not assert it is
    the same account.
  * Under our conservative cutoff (05:00Z, the hour-precision lower edge) most of this flow is excluded.
* **PM-011: not visible as described.**
  * Report: fresh wallets created the same day, one of $72k at 2.9–10.3%, and 50+ new wallets.
  * Data: 9 YES-side wallets within ±50% of $72k. **None** is new to the corpus; the youngest was first
    seen 84 days earlier.
  * There are 606 YES-side wallets new within 72h, but they account for only 12% of YES notional, and
    fresh-wallet net flow was −$134k (toward NO).
  * Either the reported wallets traded as makers or below our window, or the reports' fresh-wallet
    claims refer to on-chain age that our corpus-relative measure cannot see.
* **PM-007: not visible as described.**
  * Report: six fresh wallets with $1.0–1.2M.
  * Data: YES-side notional in the 24h before 06:00Z was $2.28M. Wallets new within 7 days contributed
    20% of it, about $447k. Fresh net flow was negative overall.
  * The largest YES wallet ($352k) has a 509-day history.
* **PM-004/005 (Israel strike, June 2025): partially consistent.**
  * In the 72h before the news, new-to-corpus wallets were net YES (+$43k, +$91k) while other wallets
    were net NO.
  * The largest YES wallets were small: $14k in PM-004 and $34k in PM-005, both from very young
    wallets.
  * No report sizes are available to compare.
* **PM-015/016 (pardons):** small markets. The fresh-wallet YES lean is visible in 514824 (fws72 = 0.48
  over 72h), but no wallet is near the reported $64k.

### Overall

* Some individual reported trades can be found in the taker data, notably PM-001 and an
  experienced-account footprint for PM-008.
* The **window-level aggregate footprint** that the features and labeling plan look for (large,
  one-sided, long-odds, fresh-dominated flow) is **not** visible before the news in any studied case.
* The reported informed flow is a small fraction of total taker volume, and it is usually outweighed
  by opposite-side flow.

## Known weaknesses

* **Taker-only.**
  * Maker (resting-order) flow exists for only about 4k rows, so informed traders using limit orders
    are invisible.
  * Signed flow is aggressor flow, not position change.
  * Wallet PnL and win rate are taker-only proxies.
* **Wallet age is relative to our corpus.**
  * "Fresh" = new to geopolitical and central-bank markets since 2023-11-18. A sports veteran looks
    fresh.
  * Among 1,630 covered wallets with complete full history (pm_wallets), the in-corpus first trade was
    within 72h of the true first Polymarket trade for only 63%; the median gap was 0.13 d.
  * Full-history coverage is 2,068 wallets (1,630 complete) of the about 51k queued at build time.
    `full_*` columns are mostly empty and depend on when the table was built (re-run to refresh).
  * Wallet age before 2024-01 is left-censored (`wallet_age_censored`).
* **No funding source, balances or order books.** Common-funder clustering is replaced by the co-timing
  proxy. Spread and impact are trade-price proxies.
* **Daily-grid only.** The table covers grid_24h snapshots (00:00 UTC). Hour-of-day variation within a
  day is not represented.
* **Correlation.** Overlapping lookbacks and consecutive daily snapshots of the same market are highly
  correlated, so weak-label counts overstate the number of independent episodes (see the `*_mkts`
  columns).
* **Matched controls** match on price and 7-day volume only, at 00:00Z. Case cutoffs are at other hours.
