# Evaluation design and Baseline A (owner: evaluation agent)

Status 2026-11-17. Code: `src/pmflow/evaluation/`. Outputs: `artifacts/evaluation/`. Tables: `eval_*`.
Rebuild everything:

```bash
.venv/bin/python -m pmflow.evaluation.snapshots      # eval_markets, eval_event_splits, eval_snapshots, eval_diag_snapshots (~1 min)
.venv/bin/python -m pmflow.evaluation.baseline       # Baseline A tables/figures + eval_baseline_preds (~1 min)
.venv/bin/python -m pmflow.evaluation.leakage        # snapshot invariants + truncation audit (must exit 0)
.venv/bin/python -m pytest src/pmflow/evaluation/tests -q
```

The question is whether flow features beat the market price out of sample. Everything below is set up so
that the price is a hard baseline to beat, and so that the test split stays untouched until a model is final.

## 1. Snapshot design

**Unit.** A snapshot is (market, `cutoff_ts`). Predictors may only use data with timestamp **strictly < cutoff_ts**.

**Universe.** Resolved binary markets (`resolution_status='resolved'`, two outcomes), either `in_scope` (9,700)
or `macro_comparison` (1,320). Always report the two separately (`universe` column). `y = 1` if YES won. For the
29 binary markets that are not Yes/No ("Up/Down", "Nothing/Something", ...), "YES" means outcome index 0.

**Grid.** A fixed UTC calendar grid every 6 hours (00/06/12/18 UTC). It runs from the first 6h boundary after the
market's first taker trade until the last boundary before `closed_time`. Why 6h:
- Informed-flow signals decay within hours, and news lands at any time of day. A 24h grid would put up to
  a day between a burst of flow and the snapshot.
- A coarser grid buys no extra statistical independence. Snapshots within a market (and within an event) are
  highly autocorrelated whatever the spacing, so inference is event-clustered.
- `grid_24h` (00:00 UTC only) is a 1/4 subset for cheap feature development. Baseline A is the same on it
  (test in-scope Brier 0.1244 vs 0.1249).

**Why the grid is not anchored on resolution.** Grid points come from the calendar, not from `closed_time`,
the outcome or "N hours before resolution". The only use of `closed_time` is to stop the grid once the market
has closed, and whether a market is still open is known live. Snapshots after the news but before the
(lagging) `closed_time` stay in, but they trade near 0/1 and the price band removes most of them.

**Price at cutoff (`price_yes`).** The last taker trade (role 0) before the cutoff, converted to YES-equivalent:
an outcome-1 trade at p counts as 1-p. We use taker trades only so that the table does not change while maker
fills are being ingested. Alternatives are stored: `vwap_6h` (the last 6h bin, NULL if it had no trade) and
`vwap_24h`. All three give the same Baseline A to within 0.0004 Brier.

**Live filter (`is_live`), computable at the cutoff:**
- `price_yes` in [0.03, 0.97];
- the last trade is ≤ 72h old;
- ≥ 3 taker trades and ≥ $200 taker notional in the last 7 days.

Stored rows are a superset that every filter variant can use: price in [0.005, 0.995] and a trade within 7 days.

**Other columns:**
- `hours_since_open` (from `created_at`);
- activity: `n_trades_24h, usd_24h, n_trades_7d, usd_7d, n_trades_cum, usd_cum` (all strictly before the cutoff);
- `pre_trade_window_market` (created before 2023-11-18, so its trade history is left-truncated);
- `hours_to_close_diag, hours_to_end_date_diag`: **these use the future. Use them only for diagnostic
  breakdowns, never as features or filters.**

**Diagnostic anchored views (`eval_diag_snapshots`).** Snapshots at 1/6/24/72h before `closed_time` and before
the last trade, as the brief asks. They are **not tradable and are leaky anchors**. Near the close they
are trivial: at 1h before close, 98% of in-scope prices are already outside [0.03, 0.97] and Brier is about 0.001.
They show nothing about predictive skill. See `baseline_diag_anchored_DIAG.csv`.

**Multi-outcome / negRisk events.** Each binary market is scored on its own YES price. Markets in one event
are strongly dependent: mutually exclusive in negRisk events, nested in "by date X" ladders. Three things handle this:
- all CIs and tests resample whole events;
- splits are by event;
- event-equal weighting is reported.

We do not renormalise negRisk prices to sum to 1. A model may do that, but only with prices from before the cutoff.

## 2. Splits (fixed before any model results; do not change)

`event_close` = the max `closed_time` over the event's resolved in-scope/macro markets (table `eval_event_splits`).

| split | events | snapshots scored |
|---|---|---|
| train | event_close < 2026-04-01 | all of the event's snapshots (all cutoffs are earlier) |
| val | 2026-04-01 ≤ event_close < 2026-07-01 | cutoff ≥ 2026-04-01 |
| test | event_close ≥ 2026-07-01 | cutoff ≥ 2026-07-01 (data end 2026-11-17) |
| purged_val | val events, cutoff < 2026-04-01 | never scored; may be used only to refit a final model on train+val |
| purged_test | test events, cutoff < 2026-07-01 | **never use** |

Why this split:
- Every training label is known before any val cutoff, and every train+val label before any test cutoff.
  `closed_time` lags the news, so it is a conservative upper bound on when a label became known.
- No event has scored snapshots in two splits.
- Purging the early snapshots of long-lived val/test events gives the embargo. Without it, a test-event
  snapshot from, say, February 2026 would be scored by a model trained on outcomes that resolved later.
- A held-out period of 4.5 months (Jul–Nov 2026) holds about 1/3 of all 2026 markets.

Workflow:
1. Select models on val.
2. Optionally refit on train + val + purged_val.
3. Score test once.

**Primary evaluation set:** `is_live AND split IN ('train','val','test')`. Counts (`split_counts.csv`):

| universe | split | snapshots | markets | events | base rate (snap) | base rate (market) | mean price |
|---|---|---|---|---|---|---|---|
| in_scope | train | 378,644 | 4,099 | 2,081 | 0.234 | 0.297 | 0.252 |
| in_scope | val | 43,713 | 1,019 | 281 | 0.434 | 0.391 | 0.315 |
| in_scope | test | 105,302 | 2,003 | 541 | 0.327 | 0.380 | 0.320 |
| macro | train | 81,948 | 475 | 179 | 0.296 | 0.244 | 0.323 |
| macro | val | 9,750 | 128 | 48 | 0.523 | 0.391 | 0.491 |
| macro | test | 31,389 | 363 | 95 | 0.486 | 0.452 | 0.426 |

Purged (not scored): in-scope 51,424 purged_val and 134,894 purged_test live snapshots; macro 10,543 and 11,655.

Val and test are dominated by the 2026 Iran / Middle East / Russia-Ukraine conflict cluster. The largest test
events are "Saudi military action against Yemen on…", "Israel military action against Lebanon on…" and
"US announces end of Iranian blockade by…". Different events on the same conflict are correlated, and event
clustering does not capture that, so the CIs below are, if anything, too narrow.

## 3. Baseline A: market price as P(YES)

Snapshot-weighted, event-clustered bootstrap 95% CIs (2,000 resamples of events). Source: `baseline_headline.csv`.

| universe | split | Brier price [95% CI] | log loss price [95% CI] | ECE | calib. slope | Brier climatology |
|---|---|---|---|---|---|---|
| in_scope | train | 0.1119 [0.1023, 0.1219] | 0.361 [0.337, 0.387] | 0.023 | 1.14 | 0.179 |
| in_scope | val | 0.1795 [0.1569, 0.2034] | 0.541 [0.480, 0.606] | 0.119 | 0.98 | 0.286 |
| in_scope | **test** | **0.1249 [0.1121, 0.1394]** | **0.398 [0.365, 0.436]** | 0.024 | 1.14 | 0.229 |
| macro | train | 0.0991 [0.0827, 0.1149] | 0.332 [0.287, 0.373] | 0.053 | 1.34 | 0.209 |
| macro | val | 0.1057 [0.0674, 0.1540] | 0.348 [0.249, 0.470] | 0.051 | 1.25 | 0.301 |
| macro | **test** | **0.2193 [0.1806, 0.2607]** | **0.636 [0.538, 0.741]** | 0.120 | 0.60 | 0.286 |

Climatology is the train base rate per universe.

**Weighting sensitivity.** The test in-scope Brier is 0.1249 per snapshot, 0.1128 [0.1023, 0.1245] per market
(equal weight) and 0.1032 [0.0913, 0.1156] per event (equal weight). Long-lived, heavily traded markets are
harder. Report all three. The harness does this automatically.

**Breakdowns (CSV):**
- `baseline_by_year.csv`: in-scope Brier is about 0.10 in 2024–25 and 0.15 in 2026.
- `baseline_by_subtype.csv`
- `baseline_by_price_bucket.csv`
- `baseline_by_negrisk.csv`
- `baseline_by_time_to_close_DIAG.csv` (diagnostic): prices are worst calibrated in the last 1–3 days before
  close. Markets that resolve YES then trade well below their eventual outcome, because news arrives
  before close. This is exactly where informed flow would show up, but bucketing by time-to-close uses the future.

**Reliability.** `baseline_reliability.csv` and `reliability_price_vs_platt.png`:
- In-scope train and test are close to the diagonal, with a mild longshot pattern: prices in [0.03, 0.1)
  resolve YES 3–5% of the time against a mean price of 5.9%.
- In-scope val (Apr–Jun 2026) is off: YES resolved far more often than priced (base rate 0.43 vs mean price 0.32).
  This is a regime shift, not a stable bias.
- Macro test is poorly calibrated (slope 0.60).

### Does recalibrated price beat raw price? No, not out of sample.

Platt scaling (`y ~ logit(price)`) and isotonic regression were fitted on train only, per universe. Delta =
Brier(price) − Brier(alternative), so a positive value means the alternative is better.
Source: `baseline_paired_vs_price.csv`.

| universe | split | Platt ΔBrier [95% CI], p | isotonic ΔBrier [95% CI], p |
|---|---|---|---|
| in_scope | train (in-sample) | +0.0002 [−0.0006, +0.0009], 0.61 | +0.0008 [+0.0002, +0.0014], 0.005 |
| in_scope | val | **−0.0050** [−0.0070, −0.0033], <0.001 | −0.0045 [−0.0063, −0.0029] |
| in_scope | test | −0.0001 [−0.0013, +0.0010], 0.88 | −0.0005 [−0.0015, +0.0004], 0.29 |
| macro | test | **−0.0146** [−0.0222, −0.0073], <0.001 | −0.0154 [−0.0236, −0.0074] |

The fitted in-scope Platt slope is 1.14 (prices slightly too compressed toward 0.5), but the correction does not
carry over to later periods. The favourite-longshot-type bias is small and unstable. **Raw price is the bar.**
A model that mostly re-learns a calibration map will look good in-sample and fail on val/test, so the harness
compares models with raw price, and they should also be checked against `p_iso`/`p_platt`
(`eval_baseline_preds`).

### Filter sensitivity (`baseline_filter_sensitivity.csv`)

In-scope test Brier by filter variant:

| variant | in-scope test Brier |
|---|---|
| primary | 0.1249 |
| band [0.01, 0.99] | 0.1015 |
| band [0.05, 0.95] | 0.1382 |
| band [0.10, 0.90] | 0.1666 |
| no activity filter | 0.1285 |
| strict activity (≤24h, ≥20 trades, ≥$5k / 7d) | 0.1067 |
| 24h grid | 0.1244 |
| excluding the last 7d before close (diagnostic) | 0.1195 |

The absolute Brier moves a lot with the price band, because it depends on how many near-certain snapshots
are included. **Only compare models on identical rows.** The harness enforces this. The conclusion that
recalibration adds nothing holds under every variant: the Platt gain on in-scope test is within ±0.001 under
all of them, and is negative on macro test under all of them.

## 4. What a model must beat, and how large an effect is detectable

- Primary target: **paired ΔBrier and Δlog-loss against raw `price_yes` on in-scope test**, positive with an
  event-clustered CI excluding 0. Report per-snapshot, per-market and per-event weighting, macro separately,
  and also against `p_iso`. Absolute levels to beat: Brier 0.1249 and log loss 0.398 (in-scope test, snapshot weights).
- **Minimum detectable effect** (`mde.csv`): α = 0.05 two-sided, 80% power, MDE = 2.80 × the event-clustered
  SE of the paired delta. The SE depends on how far a model moves away from the price. We estimate it from
  real alternative forecasts that move prices by about 2–2.5 pp on average:
  - **in-scope test MDE ≈ 0.0012–0.0016 Brier (≈1.0–1.3% of the baseline Brier)**, event-weighted about 0.0012–0.0013.
    A model that moves prices more (for example, large moves on a few flagged snapshots) will have a larger SE.
    Its MDE has to be computed from its own deltas, which the harness reports.
  - in-scope val ≈ 0.0025;
  - **macro test ≈ 0.007–0.011 (≈5%)**: with only 95 events, macro can detect only large effects.

  Plausible gains from informed flow are concentrated in a small fraction of snapshots, so the average ΔBrier
  will be small. Also check the high-confidence flag precision (below) and conditional analyses on the flagged
  subset, with the subset defined before looking.
- **Trading hurdle.** A reference run trades whenever isotonic-recalibrated price differs from price by ≥ 0.03,
  taking the first signal per market and direction (`runs/ref_isotonic_recal*`). Reconstructed fills cost on
  average about 1.0–1.4c from the spread (next same-side taker trades vs last trade) plus about 1.5–1.7c from the
  default impact model at $100 stake. Results:

  | costs | in-scope test return per $ | in-scope train return per $ |
  |---|---|---|
  | default | **−9.2% [−12.4%, −5.8%]** | −2.0% [−4.0%, +0.1%] |
  | spread only, no impact | −6.7% [−10.0%, −3.2%] | +0.6% |

  A signal therefore needs more than about 3c of real edge per trade to survive costs.

## 5. Harness (`pmflow.evaluation.harness`)

Input is a parquet/csv file or a table with `(snapshot_id, p_model[, flag])`. Rows outside the primary set
are ignored, and coverage is reported. Baseline A is re-scored on exactly the rows that have predictions.

```bash
.venv/bin/python -m pmflow.evaluation.harness --pred artifacts/modeling/preds.parquet --name m1
.venv/bin/python -m pmflow.evaluation.harness --pred table:mdl_preds --name m1 --flag-col flag \
    --trade --edge 0.05 --stake 100 --fee-bps 0 --impact-coef 0.05 --latency-s 60
```

Output goes to `artifacts/evaluation/runs/<name>/`:
- `summary.md/csv`: per universe × split × weighting. Columns: Brier, log loss, ECE, calibration
  intercept/slope (model and base), ΔBrier/Δlog-loss with event-clustered bootstrap CI, bootstrap p, and a
  cluster-robust t-test p over events;
- `by_year`, `by_subtype`, `reliability`, `coverage`;
- `flags`: precision of high-confidence flags (`flag` = +1 YES underpriced / −1 NO underpriced, or edges
  |p_model − price| ≥ 0.05/0.10/0.20) against the price-implied hit rate, with a clustered CI of the excess;
- with `--trade`: `trades.csv` and `trades_summary.csv`.

Python: `compare(pred_df, load_snapshots())`. Metrics live in `metrics.py` (brier, log_loss, ece,
reliability_table, fit_platt, paired_test, bootstrap_ratio, snapshot_weights, min_detectable_effect).

**Tradability simulator (`tradability.py`, `TradeParams`).** There are no historical order books, so fills
are rebuilt from public taker trades:
- Entry is allowed from cutoff + `latency_s` (60s).
- The fill is the share-weighted price of the next `n_ref` = 3 taker trades, within `max_wait_h` = 6h, whose
  aggressor bought our side. Those trades executed at the ask, so the spread is included.
- Fallback: the next trade of either side, plus a spread proxy (median opposite-aggressor price jump over the
  last 50 trades, minimum 0.005). Otherwise the signal is unfilled. Fill rates are 73–89% in-scope.
- Size impact is `impact_coef · sqrt(stake / max(usd_24h, $500))`, plus `extra_slippage`.
- Fees: `fee_bps` on notional, or `fee_model='prob'` with fee_rate·min(p, 1−p) per share. Default 0, because
  geopolitical markets have generally been fee-free; run sensitivity with fees > 0.
- Positions are held to resolution, and returns are event-clustered.
- Limitations:
  - the fills assume our order would have executed alongside the observed aggressors;
  - there is no queue or depth model;
  - impact is not calibrated (no book data);
  - for markets with no subsequent trade the signal is unfilled, which biases toward liquid markets.

**Leakage audit (`leakage.py`):**
- `check_source_timestamps(df)`: every `max_source_ts` must be < cutoff_ts.
- `audit_sql_feature(sql, sample)`: recomputes a SQL feature with all trades at or after the cutoff physically
  removed, using a session-temp `pm_trades`/`pm_trades_v` that shadows the real one. Results must be identical.
  Use `all_markets=True, lookback='30 days'` for cross-market wallet features.
- `audit_python_feature(f, sample)`: the same check for a Python function.
- `check_snapshot_selection()`: grid and split invariants.

Tests in `src/pmflow/evaluation/tests/` (14 pass) cover the metrics and include negative controls: a feature
that reads 1 day past the cutoff is flagged. **Feature owners: please run `audit_sql_feature` / `audit_python_feature`
on your feature code and log the result.** Note: `pm_wallet_market_summary` is aggregated over all time and is
not truncatable this way (see its point-in-time rule in `docs/data.md`).

## 6. Caveats

- Trades exist only from 2023-11-18. Markets created earlier have truncated histories
  (`pre_trade_window_market`), and wallet "freshness" before about 2024-01 is left-censored.
- `closed_time` lags the news. "Time to close" breakdowns are diagnostic and use future information.
- Last-trade price is noisy for thin markets. The activity filter and VWAP alternatives bound this, and
  Baseline A does not depend on the choice.
- The 2026 Middle-East conflict cluster dominates val and test, and cross-event correlation within a conflict
  is not captured by event clustering. Treat CIs as lower bounds on uncertainty, and check results per conflict
  and per `geo_subtype` before believing a gain.
- Val (Apr–Jun 2026) is an unusual regime (YES-heavy). A model tuned hard on val may not generalise.
- Macro has 48 val and 95 test events, so only large effects are detectable there.
- `pm_price_history` is not used. Everything is derived from taker trades, and maker fills (role 1) are ignored by design.
