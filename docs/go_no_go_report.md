# Go / no-go report: do informed-flow features beat the market price out of sample?

Owner: evaluation agent (final phase). Date: 2026-11-17. Code: `src/pmflow/evaluation/final_{audit,score,trade,extra}.py`,
`topics.py`. Tables and figures: `artifacts/evaluation/final/`. No new DB tables were created.

Timeline (UTC). This is the order in which the work was done:

| when | step |
|---|---|
| 23:38–23:45 | audit (section 2) |
| 23:40:29 | analysis plan frozen (`analysis_plan.json`, section 3) |
| 23:46:53 | the test split was scored, once (`score_summary.json`) |
| 23:47 | tradability and extras |

Nothing was tuned or added after scoring.

## 1. Verdict: NO-GO

The brief says to proceed to paper trading only if all three criteria hold on the held-out period. Only one holds.

**(1) Beats market-price-only Brier out of sample: FAIL.**
- The pre-registered primary candidate `lgbm_anchored` does beat raw price on in-scope test, market-weighted.
  - ΔBrier +0.0014 [95% CI +0.0006, +0.0022].
  - That gain is a price recalibration. The price-only LightGBM control, which sees no flow data, does at least
    as well: ΔBrier +0.0018 [+0.0005, +0.0033] vs raw price.
- The pre-registered primary test, primary candidate minus control, is **−0.0004 [−0.0011, +0.0003]**
  (p = 0.28; Δ log loss −0.0014 [−0.0033, +0.0006]).
- This matches the modeling agent's pre-registered expectation (≤ 0): flow adds nothing.
- Against raw price, the gain is not robust either: under snapshot weighting it is +0.0005 [−0.0006, +0.0015].
- Neither secondary candidate passes after Holm correction (section 4.2).

**(2) Stays well calibrated: PASS, but uninformative.**
- In-scope test ECE is 0.028 for the candidate vs 0.036 for price; the calibration slope is 1.12.
- The control is better calibrated still (ECE 0.015), so calibration is not evidence of flow information.

**(3) Positive simulated returns after costs: FAIL.**
- Trading the frozen flags with default costs gives an in-scope return of **−6.4% per $ [−21.5%, +8.9%]**
  (374 trades, 166 events). The price-only control gets −6.3%.
- The return is negative under every cost variant except some macro rows, and those are regime effects
  (section 5).

Overall: **no evidence that informed-flow features add predictive value over the market price.** The data are
consistent with a zero effect. The test can exclude an in-scope flow gain larger than about +0.0003 Brier over the
price-only control (the upper CI bound, roughly 0.3% of baseline Brier). It cannot exclude smaller effects, and it
cannot speak to flow that is invisible in taker-only data (section 7).

## 2. Audit findings (completed before scoring)

Severity scale: none, low, medium, high. **No leakage of outcome or post-cutoff information into any candidate
was found.**

| # | check | result | severity |
|---|---|---|---|
| A1 | `feat_snapshot_features` reproduction on a fresh random sample: 40 test, 30 val and 10 train snapshots, unrestricted market size, using the feature agent's truncation procedure (trades of the market and of all its wallets cut at the cutoff; resolutions of markets closing at or after the cutoff nulled; `feat_wallet_hist`, `pm_wallets` and the summary rebuilt or filtered) (`audit_features.csv`) | 32/80 rows identical in all 238 columns. In 48/80 rows only the `full_*` columns (`full_cov_share`, `fws72_full`, `dom_log_prior_mkts_full`) differ, and they differ identically with and without truncation. That is build-time drift (pm_wallets coverage grew after the build), not leakage. 2/80 rows differ in `dom_winrate_72h` only under truncation. Traced to wallets whose realised PnL is exactly $0: summation order gives ±1e-14, which flips the `PnL > 0` win test. A float tie, not future information. | none (`full_*` drift low; not model inputs) |
| A2 | Specific traps: (i) prior win rate/PnL using markets that closed after the cutoff; (ii) all-time wallet tables | (i) Covered by A1, since closures at or after the cutoff are nulled: no difference. (ii) `pm_wallet_market_summary` is used only for `full_*`, gated on `first_ts < cutoff`. `activity_complete` is a lifetime attribute (a mild selection), but no `full_*` column is in any candidate. `feat_wallet_hist` is point-in-time by array lookup (verified by A1). Residual: a market that closed before the cutoff but had its resolution finalised after it (UMA dispute) would leak its outcome into win rate. This is not testable without resolution timestamps; win-rate features carry ~0 importance. | low |
| B | Model inputs | Candidate feature lists (75 + price_logit; ridge 6) contain no `lbl_*`, `*_diag`, `hours_to_end`, `full_*`, `closed_time` or resolution field. The engineered features read none of them. The modeling frame equals the DB feature table on a 3,000-row val/test sample (0 mismatches in 241 columns; price, y and split identical). `s1_informed_score` (trained on train `lbl_L2`) is shipped but is not a stack input. The text input is the question only (no description, no end date). A regex scan of val/test questions for outcome-revealing wording found 29 hits, all benign ("confirmed as Fed Chair"). | none |
| C | No val/test row in any fit | Refitting LightGBM (22 rounds), the price-only control (38) and ridge (λ = 0.1) on train rows only (115,252 rows, cutoffs ≤ 2026-03-31, event_close ≤ 2026-03-31) reproduces the frozen train/val/test predictions exactly (max \|diff\| = 0). The system-one PCA plus head refit on train embeddings reproduces `z_s1` exactly, and `z_lgbm` matches. Guards: every fit calls `assert_fit_rows`; stage-4 meta-models are fitted on train out-of-fold rows; flag thresholds were chosen on val. | none |
| C2 | "No intercept" in `stack_s1` / `ridge_flow` | The modeling `OffsetLogit` standardises features with the train mean. A model "without intercept" therefore carries an implicit constant, −Σβμ/σ: **+0.049 logit (toward YES) for `stack_s1`**, +0.009 for ridge (fit: resid = 0.049 + 1.096·z_lgbm + 0.333·z_s1, exact to 1e-7). The constant is learned on train out-of-fold rows, which are YES-heavy by window selection, so the pre-registered stack still has a small level term. It is not leakage, but it is a deviation from the documented design (bug in `modeling/models.py`, not fixed; worked around by the level-only controls below). | medium (for interpreting stack_s1) |
| D | Splits | All 8 snapshot invariants hold. 0 events have snapshots in two scored splits. 0 prediction rows are purged or non-live; 0 `purged_test` rows are in the feature table. No train market closes on or after 2026-04-01. | none |
| D2 | Topic linkage (`audit_topic_overlap.csv`) | 100% of test snapshots are in topic clusters that also occur in train (10 topics; 8 coarse theatres). Middle East = 56% of in-scope test; Iran/Gulf alone 37%. Recurring title templates (e.g. "Israel military action against Lebanon on …?") link 7.8% of in-scope and 79% of macro test snapshots to a train template. This is not leakage (every train label resolved before 2026-04-01), but it does mean the effective number of independent units is far below 534 events. | dependence (affects CIs) |
| E | Pre-registration (`audit_prereg.csv`) | Row counts match docs/modeling.md: 162,996 = train 115,252 / val 13,516 / test 34,228. Thresholds match: lgbm 0.03, stack 0.05, ridge 0.05 (the rule was not met; it never flags), control 0.05. Stored flags equal the flags recomputed from p_model and the threshold (0 mismatches). Files were written at 23:31 UTC, before this phase. SHA-256 below. | none |
| F | Maker rows (4.63M role-1 rows now in `pm_trades`; 11 test markets affected) | Snapshots, features, `feat_wallet_hist` and the tradability simulator all filter `role = 0`. Recounted taker trades before the cutoff on 300 random val/test snapshots match the stored `n_trades_cum` exactly. Without the role filter, 1/300 would change. **No stored input could have been changed by the maker ingestion.** | none |
| G | `end_date` edits | `end_date` is in no candidate input, but the bias-robust subset uses it. Test markets whose question deadline is more than 7 days after the `end_date`: 2.1% for NO and 3.6% for YES, so there is no sign of `end_date` being systematically reset to the close date for early-YES markets. Remaining risk: unverifiable individual edits. | low |
| H | News/resolution timestamps | `closed_time` is used only to stop the grid (openness is known live) and to form splits. Case news timestamps are unverified (search extracts). | low |

SHA-256 of the frozen files:

| file | SHA-256 |
|---|---|
| `preds_lgbm_anchored.parquet` | `abf59c1d5e1144856cb2837391fbb99bee02075798210730cd9b16cd8d5658f5` |
| `preds_stack_s1.parquet` | `e825228588534a44db83bb68c269e3a7700a136a3c9f32abd9e290e7e0ecaefa` |
| `preds_ridge_flow.parquet` | `56001cada2df6a705c02aa6f7037700f7394f2ff7a65845a1c56df99d15193b2` |
| `preds_ctrl_lgbm_price_only.parquet` | `86f8b71be0c599997890f521e10b8f1a5331c63f0f89824fd7c1e49df286f74d` |

### Window-selection bias, verified independently (y vs price only, before any model was scored)

YES-excess is ȳ − mean price, market-weighted (`audit_window_bias.csv`). Test contains only markets that resolved
by 2026-11-17. A market whose deadline is later is in test only if it resolved YES early.

| in-scope rows | markets | YES-excess |
|---|---|---|
| all test | 1,920 | +0.026 |
| test, `end_date` ≤ 2026-11-10 (**bias-robust**) | 1,724 | **+0.006** |
| test, `end_date` after the window (+30 d) | 118 | **+0.216** |
| test, ≤ 7 d to event_close (DIAG) | 824 | +0.140 |
| test, 90–180 d to event_close (DIAG) | 163 | −0.107 |
| train all / val all | – | +0.019 / +0.075 |

Macro test shows the same pattern: +0.086 overall, +0.045 in the robust subset and +0.281 in the complement (for
example Treasury-yield "before 2027" ladders, where only the YES rungs have resolved). The modeling agent's §7 is
confirmed. On test the artefact is concentrated in 14% of in-scope snapshots (10% of markets) and 17% of macro
snapshots and markets.

## 3. Analysis plan (frozen 2026-11-17T23:40:29Z, before scoring; binding copy `analysis_plan.json`)

**Rows.** Frozen predictions, test split, 24h grid, live rows: in-scope 26,417 snapshots / 1,920 markets /
534 events; macro 7,811 / 344 / 94. The two universes are scored separately.

**Primary (single confirmatory test).** In-scope, `lgbm_anchored` vs `ctrl_lgbm_price_only`, paired ΔBrier,
market weighting, event-clustered bootstrap (B = 2000) plus cluster-robust t. Everything else is
secondary/exploratory: `stack_s1`, `ridge_flow`, all macro results, breakdowns, flags, `pat_*`, the s1 score, the
trading grid and the cases.

**GO criteria.**
1. `lgbm_anchored` beats raw price and the control: CI > 0, same sign under snapshot and event weighting, beats
   its own level-only version, positive vs the control on the bias-robust subset, and the sign survives theatre
   clustering. A secondary candidate counts only with Holm-adjusted p < 0.05 over the 3 candidate-vs-control tests.
2. ECE ≤ ECE(price) + 0.01 and \|slope − 1\| ≤ 0.25 (in-scope, market-weighted).
3. Default-cost return per $ > 0 with event-clustered CI lower bound > 0, above both the control and the
   rate-matched naive-YES control, and still > 0 at 1% fees.

**Window-selection controls.** Every candidate is compared with:
- raw price;
- the price-only control;
- `ctrl_shift_train` (+0.111 logit, train-fitted);
- its own **level-only** version (logit price + the candidate's mean test residual, label-free);
- `p_iso` / `p_platt` (train-fitted).

Also: the bias-robust subset (`end_date` ≤ data end − 7 d; sensitivity 21 d), the descriptive complement, and DIAG
time-to-close breakdowns.

**Coarse inference.** Topic (10) and theatre (8–9) cluster bootstrap with cluster-t (G − 1 df), plus
leave-one-theatre-out. With G ≤ 10 these are indicative only.

**MDE.** 2.80 × the clustered SE of each candidate's own delta.

## 4. Test results

Δ = loss(comparator) − loss(model), so positive means the model is better. 95% event-clustered CIs.
Source: `score_paired.csv`.

### 4.1 In-scope (geopolitical)

Baseline A on these rows: Brier 0.1147 market-weighted, 0.1244 per snapshot, 0.1039 per event; log loss 0.370 /
0.397 / 0.338.

| model vs comparator | market-weighted ΔBrier | snapshot | event | Δ log loss (market) | MDE (market) |
|---|---|---|---|---|---|
| **lgbm_anchored − control (PRIMARY)** | **−0.0004 [−0.0011, +0.0003], p = 0.28** | +0.0002 [−0.0009, +0.0013] | −0.0007 [−0.0015, −0.0000] | −0.0014 [−0.0033, +0.0006] | 0.0010 |
| lgbm_anchored − raw price | +0.0014 [+0.0006, +0.0022] | +0.0005 [−0.0006, +0.0015] | +0.0022 [+0.0016, +0.0029] | +0.0041 [+0.0022, +0.0061] | 0.0011 |
| lgbm_anchored − its level-only version | +0.0015 [+0.0007, +0.0023] | | | | |
| control − raw price | +0.0018 [+0.0005, +0.0033] | | | +0.0055 [+0.0020, +0.0091] | |
| stack_s1 − control | +0.0005 [−0.0000, +0.0010], p = 0.068 | +0.0001 [−0.0005, +0.0007] | +0.0009 [+0.0004, +0.0014] | +0.0007 [−0.0011, +0.0026] | 0.0007 |
| stack_s1 − raw price | +0.0023 [+0.0007, +0.0039] | +0.0004 [−0.0017, +0.0026] | +0.0038 [+0.0025, +0.0052] | +0.0062 [+0.0023, +0.0102] | 0.0022 |
| ridge_flow − raw price | +0.0001 [−0.0001, +0.0002] | | | +0.0002 [−0.0002, +0.0006] | 0.0002 |
| ridge_flow − control | −0.0018 [−0.0031, −0.0005] | | | | |
| p_iso − raw price (train-fitted isotonic) | −0.0012 [−0.0020, −0.0005] | | | | |

**Multiplicity.** Holm-adjusted p over the three candidate-vs-control tests (two-sided):

| candidate | Holm p | note |
|---|---|---|
| lgbm_anchored | 0.28 | |
| stack_s1 | 0.14 | |
| ridge_flow | 0.019 | in the *wrong* direction: the control beats it |

No candidate beats the control.

**Robustness.**
- **Bias-robust subset** (`end_date` ≤ 2026-11-10; 22,793 snapshots / 482 events):
  - lgbm − control: +0.0001 [−0.0006, +0.0009] market-weighted, +0.0012 [+0.0002, +0.0022] per snapshot,
    −0.0002 [−0.0009, +0.0004] per event;
  - stack − control: +0.0003 [−0.0002, +0.0009].
  - Results are the same at a 21-day margin.
  - Exploratory: one of many secondary cells (lgbm − control per snapshot on the robust subset) is nominally
    positive. Its sign flips across weightings, so it is not a finding.
- **Bias-prone complement** (114 events, descriptive): lgbm − control −0.0049 [−0.0073, −0.0026]; stack − raw
  price +0.0119, mostly level. This is where "beats price" comes from.
- **Theatre-clustered** (8 clusters):
  - lgbm − raw price: p = 0.09;
  - lgbm − control: [−0.0027, +0.0011], p = 0.65;
  - stack − control: [+0.0002, +0.0027], cluster-t p = 0.13.
- **Leave one theatre out** (lgbm − control): ranges from −0.0016 (without the Middle East) to +0.0004 (without
  Russia–Ukraine). It is never significantly positive.
- **Breakdowns** (`score_breakdowns.csv`, exploratory; lgbm − control):
  - no geo_subtype or month is significantly positive;
  - Russia–Ukraine is significantly negative, −0.0036 [−0.0049, −0.0021];
  - the 0–10% price bucket gives +0.0006 [+0.0002, +0.0009], which is tiny.

**Reading.**
- `lgbm_anchored` gains over raw price because of the calibration learned from price on train: the control alone
  gives +0.0018, more than the candidate.
- `stack_s1`'s extra +0.0005 over the control is not significant after correction, flips sign with weighting, and
  shrinks on the bias-robust subset. Its components are a +0.067-logit mean YES residual (including the implicit
  +0.049 constant, C2) and the text/topic tilt documented on val. It is not flow.
- `ridge_flow`, the only pure-flow model, moves prices by 0.3 pp and is indistinguishable from price:
  +0.0001 [−0.0001, +0.0002]. Its MDE is 0.0002, so **a pure-flow effect larger than about 0.0002 Brier is
  excluded for this linear specification.**

### 4.2 Macro (central banks; secondary, 94 events)

| model vs comparator | market ΔBrier | snapshot | event |
|---|---|---|---|
| lgbm_anchored − control | −0.0023 [−0.0050, +0.0003] | −0.0006 [−0.0031, +0.0018] | −0.0020 [−0.0045, +0.0004] |
| lgbm_anchored − raw price | +0.0030 [+0.0013, +0.0050] | +0.0015 [+0.0001, +0.0032] | +0.0026 [+0.0015, +0.0039] |
| stack_s1 − control | −0.0007 [−0.0025, +0.0011] | +0.0008 [−0.0013, +0.0029] | −0.0004 [−0.0025, +0.0016] |
| ridge_flow − raw price | +0.0001 [−0.0001, +0.0003] | | |
| control − raw price | +0.0053 [+0.0010, +0.0098] | | |

- Baseline A Brier: 0.167 market-weighted, 0.220 per snapshot.
- On the macro bias-robust subset (74 events), stack − control is nominally positive per snapshot and per event
  (+0.0021, +0.0019) but not market-weighted (+0.0010 [−0.0008, +0.0028]). Exploratory, not adjusted; the full
  macro set gives −0.0007.
- The MDE is 0.003–0.004, so only large effects are detectable.
- Macro test is YES-heavy (ȳ 0.44 vs price 0.36), and the price-only recalibration captures it.

### 4.3 Calibration (`score_calibration.csv`, `score_reliability_test.png`; market-weighted, event-bootstrap CIs)

| | in-scope ECE | slope [CI] | macro ECE | macro slope |
|---|---|---|---|---|
| raw price | 0.036 [0.022, 0.055] | 1.17 [1.08, 1.30] | 0.093 | 0.98 |
| lgbm_anchored | 0.028 [0.015, 0.046] | 1.12 [1.03, 1.25] | 0.090 | 0.95 |
| stack_s1 | 0.019 | 1.11 | 0.087 | 0.95 |
| control | 0.015 | 1.06 | 0.083 | 0.89 |
| p_iso (train) | 0.045 | 1.01 | 0.124 | 0.71 |

### 4.4 High-confidence flags (`score_flags.csv`)

- On test, 100% of flags are YES flags, as on val.
- In-scope `lgbm_anchored` (0.03): 2,012 flags in 516 markets / 214 events.
  - Hit rate 0.475 vs price-implied 0.458, excess **+0.016 [−0.077, +0.117]**.
  - Measured against the control's probability on the same rows: −0.028 [−0.121, +0.072].
  - Bias-robust subset: −0.035 [−0.133, +0.072].
- `stack_s1`: +0.042 [−0.039, +0.123]; robust −0.039.
- Control flags: +0.009 [−0.078, +0.105].
- Macro flags look strong (lgbm +0.34 [0.06, 0.45], 24 events). The control's flags show the same (+0.22) and the
  robust-subset excess is +0.006: this is the YES-heavy macro regime, not skill.
- CI half-widths of about ±0.1 mean only large precision gains could have been detected.

## 5. Tradability (`trade_summary.csv`; secondary)

**Assumptions.**
- Fills are rebuilt from public taker trades: the share-weighted price of the next 3 same-side taker trades after
  cutoff + latency, within 6 h; otherwise the next trade plus a spread proxy.
- Impact = coef · sqrt(stake / max(usd_24h, $500)).
- Fees are on notional; positions are held to resolution.
- Defaults: coef 0.05, $100, 60 s latency, 0 fees, first signal per market and direction.
- There is no order book: queue and depth are not modelled, and impact is uncalibrated.
- Fill rates are 64–73% in-scope, which biases trades toward liquid markets.

| strategy (in-scope, default costs) | trades / events | hit rate | mean fill | ret per $ [95% CI] | total PnL |
|---|---|---|---|---|---|
| lgbm_anchored @0.03 | 374 / 166 | 0.497 | 0.516 | **−6.4% [−21.5%, +8.9%]** | −$2,385 |
| stack_s1 @0.05 | 459 / 194 | 0.475 | 0.497 | −8.2% [−21.7%, +5.6%] | −$3,753 |
| ridge_flow @0.05 | 0 (never flags) | – | – | – | – |
| price-only control @0.05 | 167 / 109 | 0.353 | 0.387 | −6.3% [−26.1%, +14.0%] | −$1,055 |
| naive YES at lgbm's flag rate (10 draws) | ~520 each | | | mean −23% (range −29% to −18%) | |
| buy YES at every market's first test snapshot | 1,189 / 425 | 0.379 | 0.416 | −18.7% [−29.1%, −7.6%] | −$22,228 |

**Edge after the price moves (lgbm).** The model's edge at the cutoff is +3.4c. Cost vs the cutoff price is +3.9c
(spread about 2.4c plus impact about 1.5c), leaving **−0.6c at the fill**. The realised edge at the fill is −1.9c
(stack: +2.3c of model edge left at the fill, realised −2.2c).

**Sensitivity (lgbm, in-scope)**, varying one assumption at a time from the default:

| assumption | return per $ |
|---|---|
| impact 0 | −3.1% |
| impact 0.1 | −9.4% |
| impact 0.2 | −14.8% |
| fees 0.5% | −6.9% |
| fees 1% | −7.4% |
| fees 2% | −8.4% |
| latency 1 h | −7.5% |
| latency 6 h | −5.4% |
| stake $10 | −4.1% |
| stake $1k | −12.7% |
| stake $10k | −27.2% [−38.9%, −15.6%] |
| first signal per event | −6.6% |
| all signals (overlapping) | −11.1% |

No variant has a positive in-scope point estimate.

**Macro.** Point estimates are positive (lgbm +16% [−27%, +58%], 41 trades / 20 events; with all overlapping
signals +74% [+12%, +96%] in 21 events). Equally, the price-only control earns +30% and naive YES +18%. Macro test is
a YES-heavy window-selection regime (section 2), so these are regime effects, not skill.

## 6. Informed-flow score and weak patterns (`extra_patterns.csv`, `extra_s1_informed_score.csv`; exploratory)

Test, in-scope, any lookback; hit rate of the dominant side vs its price at the cutoff:

| signal | snapshots / markets / events | hit rate | price-implied | excess [95% CI] |
|---|---|---|---|---|
| `pat_strong` | 1 / 1 / 1 | 1.00 | 0.31 | n = 1, uninterpretable |
| `pat_possible` | 64 / 41 / 35 | 0.344 | 0.299 | +0.045 [−0.067, +0.176] |
| `pat_loose` | 194 / 91 / 62 | 0.438 | 0.395 | +0.043 [−0.029, +0.135] |
| `s1_informed_score` ≥ train p99 (dominant 24h side) | 140 / 73 / 58 | 0.493 | 0.456 | +0.037 [−0.027, +0.106] |
| `s1_informed_score` ≥ train p95 | 1,424 / 354 / 181 | 0.617 | 0.611 | +0.005 [−0.044, +0.042] |

- Macro test: `pat_possible` −0.138 [−0.183, −0.058] (6 events), `pat_loose` +0.012.
- Against p_entry or the control, the results are the same.

**What cannot be concluded.** With 35–62 events, effects smaller than about ±0.10 in hit rate are undetectable.
The strong pattern fired once. On train, the same patterns showed −0.02 to +0.06 excess. The s1 score mostly
detects the loose-pattern text descriptor (49% of p99 rows carry `pat_loose_24h`). Nothing here shows the
patterns carry information beyond the price, and nothing rules out a modest effect.

## 7. Reported-case check (`extra_cases.csv`; anecdotal)

- No seed case falls in test. Only PM-011 ("US x Iran ceasefire by April 7?", val, news 2026-04-07T22:32Z) has
  scored snapshots before the news.
- On 2026-04-01 and 04-02 (00:00Z) the price was 0.10 and 0.09. All candidates agreed with it, within about 1pp:

  | | 04-01 | 04-02 |
  |---|---|---|
  | lgbm | 0.092 | 0.084 |
  | stack | 0.100 | 0.087 |
  | ridge | 0.100 | 0.083 |
  | control | 0.091 | 0.080 |

  None flagged. The market resolved YES.
- From 04-03 to 04-07 the price was 0.012–0.023, below the 0.03 live band, so **no model scored the window in
  which the reported long-odds bets (2.9–10.3%) were placed.**
- The primary snapshot design structurally excludes the deepest long-shot regime, where the reported informed
  trades live.

## 8. What was and was not tested

**Tested:**
- the three pre-registered candidates and the control, on held-out Jul–Nov 2026;
- in-scope and macro separately;
- three weightings;
- event, topic and theatre clustering;
- bias-robust subsets;
- calibration, flags, trading with sensitivity, weak patterns, the s1 score, one case.

**Not tested:**
- the 6-hourly grid (features exist only at 00:00 UTC);
- prices below 0.03 or above 0.97;
- maker-side flow, true wallet age or funding, order books;
- event-study windows around news times;
- a full system-one model (PyTorch unavailable; JAX substitute; exploratory heads only);
- Kalshi;
- any model refit on train + val.

## 9. Threats to validity

1. **Taker-only data.** Informed traders who post limit orders are invisible, and signed flow is aggressor flow.
   About 4.6M maker rows are now ingested but were not used: by design, and not complete.
2. **Truncated, corpus-relative wallet histories.** "Fresh" means new to geopolitical/macro markets since
   2023-11. Full-history coverage was about 2k wallets at build time, at best around 40% of notional, and the
   `full_*` columns were excluded.
3. **No funding-source data** (on-chain hosts blocked) and **no order books** (fills and impact are proxies).
4. **Unverified case news timestamps** (search extracts only).
5. **One correlated 2026 regime.** The Middle East is 56% of in-scope test, and every test topic also occurs in
   train. Event-clustered CIs are too narrow; theatre-level inference has only 8 clusters.
6. **Window-selection bias.** It exists in test (section 2) and inflates any YES-shift. It is controlled here only
   partially: the robust subset relies on final Gamma `end_date`s.
7. **Small macro sample** (94 events; MDE about 0.003).
8. **PyTorch-less system one.** A JAX encoder substitute, one exploratory fine-tune, question-only text; possible
   pre-training contamination.
9. **Live-band selection.** Snapshots below 3% are excluded, which removes the long-shot regime of the reported
   cases (section 7).
10. **Implicit constant in the "no-intercept" models** (C2). A modeling-code subtlety; it does not change the
    verdict.

## 10. "No evidence of an effect" vs "evidence of no effect"

- In-scope, the primary candidate's increment over the control is −0.0004 with an upper 95% bound of +0.0003. Its
  MDE is 0.0010 (about 0.9% of baseline Brier).
- **Evidence of no effect, at this resolution:** an increment of +0.0010 Brier or more from these features in this
  pipeline is ruled out with roughly 80% power, and even +0.0003 lies outside the CI. For the pure-flow ridge,
  anything above about 0.0002 is ruled out.
- **Only absence of evidence** for:
  - effects concentrated in rare episodes: flag and pattern counts are too small, with excess CIs of ±0.1;
  - macro (MDE about 0.003);
  - anything carried by maker flow, funding links or sub-3% prices, which this data cannot see.

## 11. What would have to change for the hypothesis to deserve another test (ranked)

1. **Maker-side fills for all in-scope markets**, so that passive informed flow and true position changes become
   visible. This is the single largest blind spot.
2. **True wallet first-seen and funding data** (a Polygon RPC or Dune for USDC inflows, plus full-history
   activity for all wallets above a size threshold), so that "fresh" and "linked" mean what the reports mean.
3. **News timestamps at scale** (GDELT or a timestamped news corpus), validated on the case set.
4. **An event-study design around detected news times** instead of calendar snapshots: windows ending before the
   news, including sub-3% prices, with returns measured from the pre-news entry.
5. **Separation of the regime.** Score only markets whose deadline falls inside the window, or use a rolling
   design whose test outcomes are all resolvable. Get more independent conflicts and years rather than one 2026
   cluster.
6. **Order-book snapshots going forward** (live CLOB `/book` polling), to calibrate fills and impact.

## 12. Reproduce

```bash
.venv/bin/python -m pmflow.evaluation.final_audit     # audit (no test metric), ~6 min
.venv/bin/python -m pmflow.evaluation.final_score     # test scoring, ~1 min
.venv/bin/python -m pmflow.evaluation.final_trade     # tradability, <1 min
.venv/bin/python -m pmflow.evaluation.final_extra     # patterns, s1 score, cases
.venv/bin/python -m pytest src/pmflow/evaluation/tests -q   # 17 tests
```
