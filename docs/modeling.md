# Modeling (owner: modeling agent, `src/pmflow/modeling/`)

Status 2026-11-17. Outputs: `artifacts/modeling/`. Bulky intermediates: `data/modeling/` (gitignored).

**Question.** Given a snapshot of market activity, do informed-flow features improve the prediction of
P(resolves YES) over the market price alone?

**Answer from TRAIN walk-forward and VAL: no.**
- No feature group adds anything once the price is recalibrated. On val, every LightGBM flow, wallet,
  weak-label-flag or context variant does no better than, and often significantly worse than, the same
  model given price only.
- Re-fitting the whole pipeline on randomly shuffled feature rows gives the same val "gains" as the real
  features.
- The only flow-only model (no intercept) moves prices by 0.3pp on average. Its val gain is
  +0.0002 Brier, about a tenth of the MDE, and it does not replicate in train out-of-fold.
- Every apparent val gain over raw price is a level or calibration shift toward YES. Most of that shift is a
  window-selection artefact (Section 7), not a flow signal.
- The exploratory "system one" encoder does not change this.
- **No test-split metric was computed at any point.**

## 0. Reproduce

```bash
.venv/bin/pip install -e '.[modeling]'
.venv/bin/python -m pmflow.modeling.data               # modeling frame -> data/modeling/snap.parquet (15 s)
.venv/bin/python -m pmflow.modeling.stage1_baseline    # Baseline A on our rows (train/val only)
.venv/bin/python -m pmflow.modeling.stage2_tabular     # anchored ridge + LightGBM, ablations, placebo (~4 min)
.venv/bin/python -m pmflow.modeling.stage3_system_one embed     # serialise + embed 163k snapshots (~31 min CPU)
.venv/bin/python -m pmflow.modeling.stage3_system_one heads     # heads on frozen embeddings (1 min)
.venv/bin/python -m pmflow.modeling.stage3_system_one finetune  # time-boxed fine-tune (<= 40 min + val pass)
.venv/bin/python -m pmflow.modeling.stage4_stack       # stack, calibration, flags, frozen predictions (1 min)
.venv/bin/python -m pytest src/pmflow/modeling/tests -q
```

Seeds are fixed (`data.SEED = 20261117`). LightGBM runs with `deterministic=True`, and stage 4 asserts that its
refit reproduces stage 2 exactly (max |diff| = 0). HF models are pinned:
- `knowledgator/gliclass-instruct-edge-v1.0` @ `727be8a4…`;
- `onnx-community/ettin-encoder-32m-ONNX` @ `37b0e13e…`, used only to validate the encoder.

## 1. Data, rows, guards

**Rows.** `eval_snapshots` with `is_live AND split IN ('train','val','test')`, inner-joined to
`feat_snapshot_features` (the 24h grid only), plus the `pat_*`/`cond_*` columns of `feat_weak_labels`.

| | rows |
|---|---|
| train | 115,252 |
| val | 13,516 |
| test | 34,228 (prediction only) |

**Never loaded as features:**
- `hours_to_close_diag`, `hours_to_end_date_diag`, any `purged_*` row;
- every `lbl_*` column. These are loaded only by `data.load_aux_labels()`, as the target of the auxiliary
  head, on train only;
- `hours_to_end` (feature table). It uses Gamma's final `end_date`, which can be edited after the fact. YES
  markets close early, so it is a leak risk. It appears in a single `DIAG_` ablation and in no candidate.
- Market `description` text. Only the `question` is used.

**Guards** (`data.assert_fit_rows`) are called inside every fit (`OffsetLogit.fit`, `AnchoredLGBM.fit`, the PCA
fit, the fine-tune). They raise on any non-train row or any cutoff ≥ 2026-07-01. `evaluate.score` refuses
test and purged rows. Tests in `src/pmflow/modeling/tests/test_anchoring.py` (13 tests):
- offset and anchoring logic. LightGBM `raw_score` excludes `init_score`, so the price logit is added back.
  With 0 rounds, p = price. A strong ridge penalty gives p = price.
- the guards, including a spy test that runs the real fitting helpers on a frame containing val and test
  rows and checks that only train rows reach a fit;
- the walk-forward folds are time-ordered and event-disjoint;
- the frozen prediction files cover all splits.

**Walk-forward folds inside train.** Fold k fits on train events with `event_close < lo_k` and scores events
with `lo_k ≤ event_close < hi_k` and `cutoff ≥ lo_k`, which mirrors how val is built. Boundaries are
2025-01-01 / 07-01 / 10-01 / 2026-01-01 / 02-15 / 04-01, giving 5 folds and 32,259 out-of-fold rows. The
folds are event-grouped by construction.

**Weights.** The primary scheme is per market: each market sums to 1, so no single long-lived market
dominates. Sensitivity runs use per-event and per-snapshot weights.

**Tables.** Val tables report the market-weighted delta unless stated otherwise. Delta = loss(price) −
loss(model), so positive means the model is better. CIs are 95% event-clustered bootstraps. In-scope and macro
are always reported separately.

## 2. Stage 1: Baseline A on our rows (sanity)

Given p_model = price, the harness reproduces the published numbers exactly on the full 6-hourly grid:
- in-scope: train 0.1119, val 0.1795;
- macro: train 0.0991, val 0.1057.

On our 24h-grid rows (`stage1_baseline_modeling_rows.csv`, snapshot-weighted):

| universe | split | snapshots | events | base rate | mean price | Brier [95% CI] | log loss [95% CI] |
|---|---|---|---|---|---|---|---|
| in-scope | train | 94,768 | 2,003 | 0.233 | 0.252 | 0.1115 [0.1019, 0.1208] | 0.360 [0.335, 0.383] |
| in-scope | val | 11,069 | 273 | 0.432 | 0.315 | 0.1777 [0.1558, 0.2036] | 0.536 [0.475, 0.604] |
| macro | train | 20,484 | 178 | 0.296 | 0.323 | 0.0989 [0.0845, 0.1136] | 0.331 [0.292, 0.372] |
| macro | val | 2,447 | 47 | 0.525 | 0.493 | 0.1055 [0.0672, 0.1530] | 0.346 [0.249, 0.465] |

## 3. Stage 2: Baseline B (market-anchored tabular models)

**Design.** Every model is logit p = logit(price) + f(x), with the price logit as a fixed offset.
- LightGBM uses `init_score`, shallow trees (depth 2, 4 leaves, ≥400 rows per leaf), λ_L2 = 50, learning
  rate 0.03 and feature/bagging fractions 0.6/0.7. The number of rounds is chosen on the walk-forward
  out-of-fold log loss, averaged over folds with equal weight, and **may be 0**.
- Ridge is an L2 logistic regression with offset. λ is chosen on walk-forward.
- Feature groups were fixed before any val result was seen (`data.GROUPS`):
  - flow (25): signed net notional toward YES, signed one-sidedness and volume at 6/24/72h, volume and NSN
    z-scores, price changes, urgency and impact proxies, size shares;
  - wallet (22): fresh-wallet signed flow and share, wallet age and prior markets, win rate, concentration,
    clusters, whale and history shares, all signed by the dominant side;
  - flags (15): signed `pat_strong/possible/loose` at each lookback, plus "any";
  - context (15): age, activity totals, burstiness, neg-risk, universe, sub-type, category.
- GBM variants also get `price_logit` as a feature, for price × flow interactions.
- The simplest model is ridge on 6 pre-specified features: `s_nsn_24h`, `s_fresh_nsn_24h`, `s_os_24h`,
  `gv_z_24h`, `s_pat_loose_any`, `s_pat_possible_any`.

**Controls.**
- intercept-only shift;
- Platt (free slope and intercept);
- LightGBM on price only, i.e. a re-learned calibration map;
- the shuffled-feature placebo. Feature rows are permuted jointly within train and within val, keeping
  price and y, and the whole pipeline is refit (5 seeds).

### 3.1 Val vs raw price (`stage2_val_results.csv`; market-weighted)

| model | in-scope ΔBrier | in-scope ΔLL | macro ΔBrier | macro ΔLL |
|---|---|---|---|---|
| ctrl_intercept (+0.11 logit) | +0.0029 [+0.0018, +0.0041] | +0.0076 [+0.0041, +0.0110] | +0.0009 [−0.0003, +0.0022] | +0.0008 [−0.0022, +0.0046] |
| ctrl_platt (slope 1.18) | +0.0039 [+0.0017, +0.0059] | +0.0088 [+0.0014, +0.0153] | +0.0031 [−0.0003, +0.0062] | +0.0130 [−0.0004, +0.0248] |
| lgbm_price_only (control) | +0.0040 [+0.0023, +0.0056] | +0.0085 [+0.0037, +0.0129] | +0.0024 [+0.0005, +0.0044] | +0.0110 [+0.0048, +0.0167] |
| **placebo lgbm_all, 5 seeds (mean)** | **+0.0030** (range 0.0028–0.0031) | | **+0.0018** | |
| ridge (6 features + intercept) | +0.0013 [+0.0007, +0.0018] | +0.0034 [+0.0018, +0.0049] | +0.0005 [−0.0001, +0.0011] | +0.0007 [−0.0007, +0.0023] |
| placebo ridge (mean) | +0.0012 | | +0.0004 | |
| **ridge_noint (flow only, no intercept)** | **+0.0002 [+0.0000, +0.0003]** | +0.0004 [−0.0001, +0.0009] | +0.0001 [−0.0000, +0.0002] | +0.0003 [−0.0001, +0.0007] |
| lgbm_flow | +0.0040 [+0.0021, +0.0059] | +0.0094 [+0.0039, +0.0142] | +0.0021 [+0.0008, +0.0036] | +0.0100 [+0.0061, +0.0141] |
| lgbm_wallet | +0.0029 [+0.0015, +0.0043] | +0.0069 [+0.0028, +0.0103] | +0.0016 [+0.0005, +0.0027] | +0.0068 [+0.0034, +0.0100] |
| lgbm_flags | +0.0031 [+0.0018, +0.0043] | +0.0068 [+0.0032, +0.0101] | +0.0019 [+0.0004, +0.0034] | +0.0084 [+0.0038, +0.0127] |
| lgbm_context | +0.0041 [+0.0024, +0.0057] | +0.0096 [+0.0050, +0.0136] | +0.0013 [+0.0002, +0.0024] | +0.0076 [+0.0041, +0.0110] |
| **lgbm_all** | +0.0033 [+0.0020, +0.0045] | +0.0080 [+0.0047, +0.0110] | +0.0013 [+0.0004, +0.0024] | +0.0063 [+0.0036, +0.0091] |
| lgbm_all_monotone | same as lgbm_all to 4 dp | | | |
| DIAG lgbm_all + hours_to_end | +0.0037 [+0.0023, +0.0051] | +0.0092 [+0.0056, +0.0125] | +0.0014 [+0.0003, +0.0025] | +0.0067 [+0.0037, +0.0098] |

### 3.2 The comparison that answers the question: each model vs its control (`stage2_increments_vs_control.csv`)

| model − control | where | in-scope ΔBrier | in-scope ΔLL | macro ΔBrier | macro ΔLL |
|---|---|---|---|---|---|
| lgbm_flow − lgbm_price_only | val | +0.0000 [−0.0005, +0.0006] | +0.0009 [−0.0010, +0.0027] | −0.0003 [−0.0011, +0.0005] | −0.0010 [−0.0032, +0.0017] |
| lgbm_wallet − lgbm_price_only | val | −0.0010 [−0.0016, −0.0005] | −0.0016 [−0.0034, +0.0001] | −0.0009 [−0.0020, +0.0000] | −0.0042 [−0.0070, −0.0013] |
| lgbm_flags − lgbm_price_only | val | −0.0009 [−0.0013, −0.0005] | −0.0017 [−0.0028, −0.0005] | −0.0005 [−0.0010, −0.0001] | −0.0026 [−0.0039, −0.0011] |
| lgbm_context − lgbm_price_only | val | +0.0001 [−0.0005, +0.0007] | +0.0011 [−0.0007, +0.0028] | −0.0012 [−0.0026, +0.0001] | −0.0034 [−0.0067, −0.0000] |
| **lgbm_all − lgbm_price_only** | val | **−0.0007 [−0.0014, −0.0001]** | −0.0005 [−0.0025, +0.0017] | **−0.0011 [−0.0022, +0.0001]** | −0.0047 [−0.0077, −0.0010] |
| lgbm_all − lgbm_price_only | train OOF | −0.0001 [−0.0007, +0.0006] | −0.0002 [−0.0020, +0.0017] | +0.0002 [−0.0010, +0.0012] | −0.0012 [−0.0046, +0.0027] |
| lgbm_flow − lgbm_price_only | train OOF | +0.0006 [−0.0001, +0.0014] | +0.0009 [−0.0012, +0.0031] | +0.0002 [−0.0009, +0.0013] | −0.0001 [−0.0030, +0.0031] |
| ridge_noint − price | train OOF | −0.0001 [−0.0002, +0.0001] | −0.0002 [−0.0006, +0.0002] | +0.0000 [−0.0001, +0.0002] | +0.0001 [−0.0004, +0.0007] |
| ridge − ctrl_intercept | val | −0.0016 [−0.0023, −0.0009] | −0.0042 [−0.0063, −0.0022] | −0.0004 [−0.0012, +0.0002] | −0.0001 [−0.0024, +0.0016] |

### 3.3 Other stage-2 evidence

- **Train walk-forward vs raw price.** Pooled out-of-fold, market-weighted:
  - lgbm_all: in-scope +0.0004 [−0.0007, +0.0015], macro −0.0001 [−0.0010, +0.0008];
  - ridge_noint: −0.0001 [−0.0002, +0.0001];
  - the intercept control *loses*: −0.0012 [−0.0026, +0.0000].

  By fold (`stage2_train_oof_by_fold.csv`), fold 0 (2025H1, small training set) is negative for every model
  at −0.008 to −0.024 log loss. The later folds are positive for the calibration-type models too, so nothing
  flow-specific shows up.
- **Weighting sensitivity.**
  - With per-snapshot weights, walk-forward selects **0 rounds** for lgbm_all, i.e. the model *is* the price.
    Ridge with snapshot weights is significantly worse than price on val (−0.0019 [−0.0026, −0.0012]).
  - With per-event weights the gains halve: lgbm_all is +0.0018.
  - Any apparent gain therefore depends on the weighting. It is never robust.
- **Placebo.** With feature rows shuffled, walk-forward still picks 24–31 rounds, because `price_logit` is
  not shuffled. Val gains are then +0.0028 to +0.0031 in-scope, the same as the real lgbm_all (+0.0033),
  and train out-of-fold gains are −0.0002 to +0.0001. **This is what "no signal" looks like in this
  pipeline. The real features are indistinguishable from it.**
- **Importance** on val (lgbm_all; `stage2_importance_shap.csv`/`.png`, `stage2_importance_perm_groups.csv`):
  - By mean |SHAP|: price_logit 0.096, neg_risk 0.028, dp_72h 0.024, burstiness 0.019, dp_24h 0.011.
    Every wallet feature is below 0.0025.
  - Group permutation (increase in val log loss): price 0.0044, context 0.0021, flow 0.0009–0.0011 (almost
    all price-change momentum, dp), wallet 0.0001, flags 0.0000.
  - Weak-label flags fire on ≤0.7% of snapshots and are never used by the trees.
- **Level-shift diagnostic** (`stage2_val_level_shift_diag.csv`). Replacing each model's residual by its own
  val-mean constant explains the intercept, ridge and ridge_wevent gains completely. The remaining
  "non-level" part of the GBM gains is the price-calibration map (lgbm_price_only alone gives +0.0055 over its
  constant), not flow.
- **Simplest-model coefficients** (ridge_noint, per SD of feature, logit units):
  - s_nsn_24h +0.023;
  - s_fresh_nsn_24h **−0.017**: fresh-wallet flow toward YES predicts slightly *less* YES. This matches the
    feature agent's case study.
  - s_os_24h +0.012;
  - gv_z −0.002, flags ≈ 0.

  All are tiny. The model moves prices by 0.34pp on average.

**Gate decision (stage 2 → 3): NOT passed.** No flow, wallet or flag group gives a credible val gain over the
price-only control in either universe. Several are significantly worse. Nothing exceeds the val MDE of about
0.0025, and the placebo reproduces the "gains".

## 4. Stage 3: exploratory "system one" (gate not passed; one small, time-boxed run)

This section is clearly labelled **exploratory**. The full version (multi-seed, walk-forward fine-tunes) was
not built because the gate failed. Compute used: 31 min embedding + about 1 min heads + 31 min
fine-tune, roughly 65 CPU-min in total.

**What was run.**
- **Backbone.** The ModernBERT (Ettin-32m) encoder inside `knowledgator/gliclass-instruct-edge-v1.0`
  (Apache-2.0), used **with a fresh binary residual head**. The GLiClass label-scoring head and library were
  not used, because PyTorch could not be installed (see blockers).
- **Implementation.** We implemented the encoder in JAX (`encoder_jax.py`) and validated it against
  onnxruntime on the published ONNX export of the same architecture. MLM logits agree to max |diff| 2e-4
  (mean 5e-6, logit scale ≈ 4) at sequence lengths 57 and 303, so both global and sliding-window attention are
  exercised.
- **Input.** One English text per snapshot, at most 128 tokens (the p50 is about 113, so some long questions
  are truncated). Example in `stage3_text_examples.csv`. The text contains:
  - the market question and the category/sub-type;
  - time since open, binned;
  - a price bucket;
  - six binned, verbalised 24h flow descriptors: direction of net flow, volume vs usual, fresh-wallet net
    side and size, price move, largest trade, and the loose-pattern flag;
  - everything strictly from pre-cutoff features. There is no end date, no description and no trade-sequence
    serialisation (not done, time box).
- **(a) Frozen-encoder heads.** Mean-pooled embeddings, then PCA (32 components, fitted on train), then a
  ridge offset-logit head. λ is chosen on walk-forward, giving out-of-fold predictions on train and a
  train-fit head on val/test. Variants:
  - `s1_text_desc`: full text;
  - `s1_text_only`: question only. This doubles as the contamination / topic check from the research report.
- **(b) Fine-tune.** The whole encoder plus the residual head, with logit(price) as the offset. Settings:
  AdamW-style Adam with lr 2e-5 for the encoder and 1e-3 for the head, batch 32, 1 epoch over a 24k
  market-balanced train subsample, a 40-min budget, single seed, train → val only (no walk-forward).
- **(c) Auxiliary "informed-flow" score.** A logistic head on the embeddings for "`lbl_L2` at any lookback",
  fitted on train only. There are 467 positives in train and only 12 in val, so it is weak by construction.

**Results on val** (`stage3_heads_results.csv`, `stage3_finetune_val.csv`; market-weighted, vs raw price):

| model | in-scope ΔBrier | in-scope ΔLL | macro ΔBrier | macro ΔLL | train OOF in-scope ΔBrier |
|---|---|---|---|---|---|
| s1_text_desc (frozen) | +0.0022 [+0.0006, +0.0038] | +0.0057 [+0.0008, +0.0101] | +0.0001 [−0.0012, +0.0012] | +0.0021 [−0.0011, +0.0054] | +0.0004 [−0.0010, +0.0018] |
| s1_text_only (question only) | +0.0028 [+0.0007, +0.0048] | +0.0079 [+0.0016, +0.0137] | −0.0002 [−0.0026, +0.0020] | −0.0003 [−0.0059, +0.0052] | −0.0001 [−0.0015, +0.0011] |
| s1_text_desc_noint | +0.0017 [+0.0003, +0.0031] | +0.0042 [−0.0002, +0.0081] | −0.0002 [−0.0015, +0.0009] | +0.0016 [−0.0019, +0.0046] | +0.0008 [−0.0004, +0.0021] |
| s1_finetune (24k, 1 epoch, 750 steps) | **−0.0170 [−0.0342, −0.0015]** | −0.0603 [−0.1112, −0.0164] | −0.0122 [−0.0304, +0.0034] | −0.0183 [−0.0620, +0.0223] | not run (no walk-forward) |

**Reading.**
- **The question text alone does at least as well as question plus flow descriptors.** Its val gain is
  in-scope only, with a mean residual of +0.08 logit toward YES. It vanishes in train out-of-fold.
  It is therefore a topic/regime effect: which kinds of question resolved YES in Apr–Jun 2026, possibly
  helped by encoder pre-training contamination. It is not flow.
- The verbalised flow descriptors add nothing on top of the text.
- **The fine-tune overfits and is significantly worse than price on in-scope val.** Its train loss fell from
  0.370 (price only) to 0.284 within one epoch. Each training market contributes many near-identical
  snapshots, so the encoder learns "this question resolves YES/NO" (market identity), which does not transfer
  to new events. It moves val prices by 11pp on average (`stage3_finetune_val.csv`, `stage3_finetune_loss.csv`).
  A credible fine-tune would need market-level deduplication, walk-forward early stopping and several seeds,
  which is the full programme that the gate did not justify.
- The auxiliary informed-flow head has val AUC 0.997, but this is trivial. The text contains the
  loose-pattern descriptor, and L2 requires the pattern, so the head detects "pattern present". It says
  nothing about whether the flow was informed. Among pattern firings, vindication tracks the entry price
  (`docs/features.md`).
- The score is shipped as `s1_informed_score` in `preds_stack_s1.parquet`, for completeness only.
- `tasksource/tasksource-decider-nano` zero-shot was **not run**. It would need its custom PyTorch code
  (`decider.py`), and PyTorch is not installable here.
- The hosted Jev API was skipped, per the brief.

**Gate (stage 3 → full system one): not passed, and the exploratory run gives no reason to revisit.**

## 5. Stage 4: market-anchored stack (`stage4_val_results.csv`)

- **Model.** logit p = logit(price) + β0 + β1·z_lgbm + β2·z_s1. The z's are each component's residual logit.
  Walk-forward out-of-fold values are used on train, and train-fit values on val and test. The β's are
  L2-shrunk and fitted on the 32,259 train out-of-fold rows, with λ chosen on walk-forward over those rows.
- **With a free intercept:** β0 = +0.40, val in-scope +0.0097 [+0.0043, +0.0149] and macro +0.0023
  [−0.0026, +0.0080]. Per-component Platt-on-residual calibration likewise learns a = +0.46 and gives val
  +0.010. **These numbers are not a flow signal.** The out-of-fold rows are YES-heavy by construction,
  with market-weighted ȳ = 0.377 vs mean price 0.315 (Section 7). Val is built the same way
  (ȳ = 0.397 vs 0.329), so a learned YES-shift "validates".
  - Excluded from the candidates.
- **No intercept (β0 = 0, the pre-registered stack):** β1 = 1.10 per logit unit of z_lgbm, β2 = 0.33 per unit
  of z_s1. Val:
  - in-scope +0.0053 [+0.0029, +0.0076], log loss +0.0134 [+0.0067, +0.0195];
  - macro +0.0021 [+0.0003, +0.0039].

  Its mean val residual is +0.070 logit (in-scope). A constant shift of that size alone gives +0.0019 of the
  +0.0053. The rest is lgbm's price recalibration plus the text/topic tilt of z_s1, neither of which is flow.

## 6. Stage 5 calibration and stage 6 flags

- **Calibration** (`stage5_val_ece.csv`, `stage5_val_reliability.csv/png`; ECE market-weighted with event-bootstrap CI):

  | forecast | in-scope ECE | macro ECE |
  |---|---|---|
  | price | 0.075 [0.051, 0.109] | 0.058 [0.043, 0.089] |
  | z_lgbm raw | 0.068 [0.043, 0.101] | 0.054 [0.038, 0.089] |
  | z_lgbm + Platt | 0.025 | 0.053 |

  The Platt "improvement" is the YES shift (a = +0.46) that matches val's regime and the window selection,
  so it is not adopted. Isotonic was not used, per the report.
- **Flags** (`stage6_val_flags.csv`). flag = sign(p − price) when |p − price| ≥ threshold.
  - **Rule (fixed before looking):** choose the threshold with the highest pooled val excess precision,
    with at least 10 flagged events and a clustered 95% CI lower bound > 0; otherwise freeze 0.05.
  - **Frozen thresholds:** lgbm_anchored **0.03**, stack_s1 **0.05**, ridge_flow **0.05** (the rule was not
    met; at ≥ 0.03 it never flags on val), control ctrl_lgbm_price_only **0.05**.
  - **Caveat.** On val, 99–100% of flags are YES flags. Their excess precision (+0.25 for lgbm at 0.03) is
    the YES-heavy val regime and window selection. The price-only control, which has no flow features,
    behaves the same way (same table): 100% YES flags, val excess +0.17 at 0.03 and +0.21 at 0.05. Do not read
    flag precision without that control.
  - lgbm_anchored never moves price by ≥ 0.05, so harness edges ≥ 0.05 will be empty for it.

## 7. Window-selection bias (important for the evaluator)

- **Mechanism.** Val (and each walk-forward fold) contains only events that *closed inside the window*,
  scored at cutoffs inside the window. "By date X" markets that resolve YES close early, so they are
  over-represented. NO-resolving markets run to their deadline, often past the window end.
- **Size of the effect.** Market-weighted YES-excess (ȳ − mean price):

  | rows | YES-excess |
  |---|---|
  | all train | +0.014 |
  | train walk-forward out-of-fold rows | +0.062 |
  | val | +0.068 |

  By time from cutoff to `event_close` (diagnostic only, it uses the future), train YES-excess is:

  | time to event_close | YES-excess |
  |---|---|
  | ≤ 7 days | +0.098 |
  | 7–30 days | +0.002 |
  | 30–90 days | −0.041 |
  | 90–180 days | −0.066 |
  | > 180 days | −0.080 |
- **What follows.** An intercept or "shift toward YES" fitted on fold rows, or chosen on val, exploits
  this artefact. The test split (events resolved by data end, 2026-11-17) is built the same way and may
  carry the same artefact. A YES-shift could "win" test without any information about flow.
  - This is why no pre-registered candidate has a free intercept fitted on out-of-fold rows.
  - It is also why `ctrl_lgbm_price_only` is shipped, so that the evaluator can separate calibration and
    level effects from flow.

## 8. Pre-registration (frozen before any test metric; no test metric was computed by this agent)

Exactly three candidates. Frozen predictions for all splits (train 115,252 / val 13,516 / test 34,228 rows,
24h grid) are in `artifacts/modeling/preds_<model>.parquet`. Columns are `snapshot_id, p_model, flag, edge,
split, universe, fit_kind`. Train rows are in-sample fits of the train-only models. Val and test rows are
out of sample. Nothing was refit on val: the test predictions come from exactly the models scored on val.

1. **PRIMARY: `lgbm_anchored`** (`preds_lgbm_anchored.parquet`).
   - Model: market-anchored LightGBM with `init_score = logit(price)`, all 4 feature groups plus
     price_logit (75 features), per-market weights, 22 rounds (chosen on train walk-forward), fitted on train.
   - Frozen flag threshold: 0.03.
   - Extra column: `p_model_train_oof` (walk-forward out-of-fold values for train rows).
   - **Pre-registered hypothesis:** on in-scope test it does *not* beat raw price by more than the price-only
     control `ctrl_lgbm_price_only` does. Flow adds nothing, so ΔBrier(lgbm_anchored − ctrl) ≤ 0 within CI.
     Report macro separately.
2. **`stack_s1`** (`preds_stack_s1.parquet`).
   - Model: logit(price) + 1.10·z_lgbm + 0.33·z_s1, no intercept, with weights fitted on train
     out-of-fold rows.
   - Frozen flag threshold: 0.05.
   - Extra columns: `z_lgbm`, `z_s1`, `s1_informed_score`.
3. **`ridge_flow`, the simplest alternative** (`preds_ridge_flow.parquet`).
   - Model: offset logit, no intercept, on the 6 pre-specified flow/flag features, λ = 0.1.
   - Frozen flag threshold: 0.05.
   - This is the pure flow effect with no level or calibration term.

**Control (not a candidate):** `preds_ctrl_lgbm_price_only.parquet`, the same LightGBM given price only
(38 rounds). Compare every candidate with it as well as with raw price and `p_iso`.

## 9. Not done, and why

- **Full system-one programme** (walk-forward fine-tunes, 3 seeds, sequence-of-trades serialisation,
  proxy-target pretraining). Not done: the stage-2 gate failed and the exploratory run showed nothing.
- **PyTorch / GLiClass library.** Not usable. The only installable PyTorch wheel (2.15, PyPI) is the
  CUDA 13 build. It imports only with about 4 GB of `nvidia-*` libraries, which do not fit on the shared disk
  (6.8 GB free). The CPU wheel index (download.pytorch.org) is blocked by the proxy (403). We used the JAX
  re-implementation instead, validated against ONNX.
- **tasksource-decider-nano zero-shot.** Not run: it needs PyTorch.
- **Refit on train + val + purged_val.** Deliberately not done. Val is an unusual, YES-heavy regime with the
  window-selection artefact, and refitting would make the test predictions differ from the validated models.
- **6-hourly grid.** Features exist only on the 24h grid, so all modeling rows are the 24h subset. Baseline A
  is recomputed on those rows.
- **TabPFN / TabICL side baselines.** Not run (time; no signal to chase).

## 10. Caveats

- **Taker-only data.** Maker (limit-order) flow, where informed traders may sit, is invisible. Signed flow
  is aggressor flow, not position change.
- **Thin wallet histories.** "Fresh" means new to the corpus (geo and macro since 2023-11), not new to
  Polymarket. Full-history columns are almost empty and were not used. Win-rate features cover a minority
  of notional.
- **Correlated events.** Val and test are dominated by the 2026 Middle-East / Russia-Ukraine cluster.
  Event-clustered CIs do not capture cross-event correlation within a conflict, so the CIs are too narrow.
- **Val regime.** The YES base rate is 0.43 against a mean price of 0.32 (in-scope). Part of this is
  window selection (Section 7). Every val "gain" over raw price in this document is consistent with a shift
  toward YES plus price recalibration.
- **Macro.** Only 47 val events, so only large effects are detectable.
- **Encoder contamination.** The question-only head gains on val in-scope but not on train out-of-fold. The
  pre-trained encoder may encode knowledge of 2025–26 events, which we cannot rule out.
- **Weighting.** Results depend on it. Snapshot weighting selects "price only" (0 rounds).
