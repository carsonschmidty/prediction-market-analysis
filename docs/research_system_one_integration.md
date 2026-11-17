# System One Integration for Informed-Flow Resolution Forecasting

Research report, 2026-11-17. Scope: the open-source "system one" classifier, how it is fed, how it is combined with a LightGBM flow model and the market price, how the result is calibrated and evaluated, and the compliance context. Written for a team deciding what to build in a two-week, CPU-only prototype.

---

## 1. Recommendation

Build a **market-anchored stacking model**. The final log-odds equal the market's log-odds plus small, ridge-shrunk corrections from two components. One is a heavily constrained LightGBM on point-in-time wallet, flow, price-impact and timing features; it owns every raw number. The other is a small open "system one" encoder that reads only text context (market question, event type, recent headlines) plus a few coarse, verbalised flow descriptors, never raw trade tables. Fit the corrections with logistic regression on out-of-fold, walk-forward predictions, so the default answer is "the market is right" and the components must earn any weight they get. This combines options (a) and (d) in Section 6. Use gating (c) only as an interaction term and to save compute, and reject (b). For the CPU prototype the system one model is **`tasksource/tasksource-decider-nano`**. It is Apache-2.0, has 149M parameters and an 8k-token ModernBERT/Ettin backbone, uses the Jev-style typed `noul`/`choice` interface, and is the best-calibrated open-licence decision model under 300M parameters on the public Decision Index board. Use it zero-shot first and fine-tune it only if out-of-fold log loss improves. The fallback is `knowledgator/gliclass-instruct-edge-v1.0` (32.7M, Apache-2.0). With a GPU, use `knowledgator/gliclass-v4-edge` (440M, Apache-2.0). Expect most of the measurable edge, if there is any, to come from LightGBM and timing features rather than from the transformer. The closest published evidence finds that order-flow models struggle to beat the Polymarket price itself (OpenMarket, arXiv 2607.26245), and that a learned pooling gate largely defers to market prices (arXiv 2609.12101). Plan a go/no-go test against the market-only baseline, not an absolute accuracy target.

---

## 2. How to read this report (evidence quality and access limits)

- **Network limits during this review.** Only `huggingface.co` could be opened. Fetches of other hosts (typesafe.ai, openrouter.ai, docs.aimlapi.com, inovex.de, deeplearning.ai) failed at DNS/proxy, and the GitHub tool refused repositories outside this project. Every source in Section 13 was opened. Papers were read as their **abstracts only**, through the Hugging Face papers index (`huggingface.co/api/papers/<arXiv id>`). Claims that rest on full-text details are marked as such or avoided.
- **Web-search-only material** (platform terms, CFTC actions, the VPIN/PIN and options-insider literature, press coverage of Polymarket insider cases) was seen only as search-engine summaries and could not be opened. It is listed separately in Section 14 as *unverified leads*, not as sources. Every claim that depends on it is tagged **[unverified]**.
- Tags used below: **[measured]** = numbers from an opened dataset or model card; **[abstract]** = stated in an opened paper abstract; **[standard practice]** = well-established method knowledge not re-sourced here; **[speculative]** = our judgement with thin or no direct evidence.

---

## 3. What "Typesafe's Jev" is: verified versus not verified

**Verified from opened sources (all third-party, none from TypeSafe itself):**

- Jev is a **hosted, proprietary model from typesafe.ai**. No weights exist on Hugging Face (the `typesafe-ai` organisation does not resolve there). The Decision Index methodology records Jev as "TypeSafe hosted API (jev-1.14.0) over HTTPS", with earlier payloads built against `jev-1.13.0` [measured: Decision Index `methodology.json`].
- Its interface takes a **`state` (text or JSON) plus named typed questions** (`choice`, `noul` = yes/no returning P(true), `score` = ordered levels) at `POST /v1/systemone`, and returns a probability for every option with no generated text. Two open model cards reproduce this request/response shape so that clients written for Jev can call local models (`knowledgator/gliclass-v4-edge`, `fastino/GLiDE-0.4B`). The GLiClass card also links an official-looking `typesafe-ai/WorkflowEvals` repository, which could not be opened.
- **Independent calibration measurement** on the community Decision Index (n = 34,411 scored answers over 31 benchmarks): accuracy 0.743, mean confidence 0.817, **ECE 0.080**. It is overconfident in the top bin (mean confidence 0.98 against accuracy 0.92 for the 56% of answers above 0.9) and much worse on retrieval tasks (ECE 0.245) [measured: `di_index.json`, Jev entry].
- Hosted latency measured by the same lab: median 196 ms and p95 297 ms per request, network included [measured].
- The lab's run cost USD 12.26 for 291.8M input and 44.5M output tokens. That works out to about $0.042 per million input tokens if output is free, which is consistent with, but does not confirm, the pricing claims in search results [measured; pricing itself unverified].
- On the Decision Index 0.3 public suite, Jev 1.13 scores a public skill of 57.96 (GLiClass v4 card) and Jev 1.14 scores 60.7 on the current board. Several open models now exceed it (GLiDE 27B, karpo-decision-31B and others) [measured].

**Not verified (do not rely on these):** Jev's architecture, parameter count, context length, training method, training data, knowledge cutoff, release date and official pricing. Search summaries describe Jev as "transformer-based, not autoregressive, trained with Reinforcement Learning for Calibrated Decisions", released 2026-09-15, but no TypeSafe primary document could be opened. The Decision Index notes that Jev's cutoff is unpublished, so contamination on forecasting tasks "is not certified". **We make no claims about Jev's internals in this report.**

**Implication.** "Jev-like" is now a well-defined *interface* with a public benchmark (Decision Index) and dozens of open reproductions. That interface (state plus typed question, returning calibrated option probabilities) is what we should copy. It also lets us swap models behind one API.

---

## 4. Q1: Candidate open system one models

### 4.1 Comparison table

Licence, size and context come from the Hugging Face API and `config.json`. "DI skill", "DI ECE" and "GPU latency" come from the community Decision Index board, which runs **zero-shot, generic decision tasks** on one RTX PRO 6000 GPU. They show general competence and out-of-the-box calibration only, not performance on trade windows. Whatever is chosen will be recalibrated on our data (Section 7). CPU figures are in Section 4.3.

| Model ID | Licence | Params | Max context (tokens) | Type | DI public skill | DI ECE | GPU median latency | CPU fine-tune on 4 cores / 15 GB | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **`tasksource/tasksource-decider-nano`** | Apache-2.0 | 149M | 7,999 pos. (card: pairs up to 8,192) | Joint cross-encoder on `cross-encoder/ettin-reranker-150m-v1`; Jev-style `choice`/`noul` | 33.5 | **0.070** | 13 ms | Feasible with short inputs (see 4.3) | Card claims #1 under 300M on DI 0.4 Full. Custom `decider.py` (PyTorch + transformers only). Overconfident in the top bin (0.976 confidence vs 0.910 accuracy). |
| `knowledgator/gliclass-instruct-edge-v1.0` | Apache-2.0 | 32.7M | 7,999 (Ettin-32m) | GLiClass uni-encoder, prompt + labels | not on board | n/a | n/a | Easy | Smallest credible instruction-following option. Supports few-shot examples and label descriptions. |
| `knowledgator/gliclass-edge-v3.0` | Apache-2.0 | 32.7M | 7,999 | GLiClass | not on board | n/a | n/a | Easy | Older; LoRA-trained on logic tasks (card). |
| `knowledgator/gliclass-v4-edge` | Apache-2.0 | 440M | DeBERTa-v3-large (512 abs. config; card runs `max_length` 2,048 to 16,384) | GLiClass v4, ships a Jev-compatible server | 47.5 (card) | not listed (v4.1: 0.118, *under*-confident) | 11 ms (card), 21.6 ms (v4.1 board) | Slow but possible with LoRA; better on GPU | Card: "Scores are calibrated on held-out data". `gliclass-v4.1-edge` has **no licence in its metadata** and a 15-byte README, so avoid it until clarified. |
| `fastino/GLiDE-0.4B` | **Fastino Community License** (custom) | 395M | 7,999 | Ettin-400m encoder, 255-way decision readout | 51.1 | **0.049** | 18 ms | Possible, slow | Best-calibrated small model on the board. The licence requires a commercial agreement for production commercial use by entities with revenue above $10M, "Built with Fastino GLiDE" attribution for any commercial use, and **share-alike for fine-tunes** (derivatives inherit these terms). It counts "internal operations of a for-profit entity" as commercial use. Research use is allowed. |
| `fastino/GLiNER2.5-Decide` | Apache-2.0 | 486M (card says 340M) | 512 | GLiNER2 schema classifier (DeBERTa-v3-large) | 11.5 | 0.088 | 23 ms | Built-in trainer (`gliner2[train]`) | Weak under the Jev mapping on the board; its card benchmark is exact-match on Fastino's own suite. |
| `fastino/gliner2.5-small-v1` | Apache-2.0 | 74M | 512 | GLiNER2 (DeBERTa-v3-xsmall) | not on board | n/a | n/a | Easy | Good if entity extraction from news (actors, places) is also wanted. |
| `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` | MIT | 70.8M | 512 | NLI cross-encoder (entailment vs not) | not on board | n/a | n/a | Easy | Classic NLI zero-shot baseline ([abstract] 2312.17543). `shreyanbr/system-one-zeroshot` wraps it in the Jev schema without further training. |
| `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` | Apache-2.0 | 150M | 8,192 | NLI cross-encoder | not on board | n/a | n/a | Feasible | Long context; NLI framing. |
| `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | MIT | 184M | 512 | NLI cross-encoder | not on board | n/a | n/a | Feasible | Widely used baseline. |
| `jhu-clsp/ettin-encoder-32m` / `-68m` / `-150m` | MIT | 32M / 68M / 150M | 7,999 | Plain encoders (no decision head) | n/a | n/a | n/a | Easy / easy / feasible | Best raw backbones per [abstract] 2507.11412. Use with a fresh binary head if typed-decision pretraining does not help. |
| `answerdotai/ModernBERT-base` | Apache-2.0 | 150M | 8,192 | Plain encoder | n/a | n/a | n/a | Feasible | [abstract] 2412.13663. Designed for GPU efficiency. |
| `tardellirs/crivo-150m`, `MLukauskas/amber-*` | **CC-BY-NC-4.0** | 17M to 1B | 7,999 | Decision encoders | 34 to 53 | 0.06 to 0.15 | 6 to 63 ms | n/a | **Excluded**: non-commercial licences. |
| `Prior-Labs/tabpfn_3_5` (tabular, not text) | **Non-commercial** custom licence | n/a | n/a | Tabular in-context learner | n/a | n/a | n/a | n/a | The licence forbids using its outputs "for internal commercial decision-making", which would cover trading. **Excluded.** |
| `Prior-Labs/TabPFN-v2-clf` (tabular) | Prior Labs License (Apache-2.0 + attribution, per card) | n/a | n/a | Tabular in-context learner | n/a | n/a | n/a | CPU inference fine for small n | Usable **as a quant-side baseline** next to LightGBM. |
| `jingang/TabICL-clf` (tabular) | BSD-3-Clause | n/a | n/a | Tabular in-context learner | n/a | n/a | n/a | CPU feasible | TabICLv2 claims state of the art on TabArena/TALENT ([abstract] 2602.11139). Second quant-side baseline. |
| TypeSafe Jev (reference) | Proprietary API | unknown | unknown | unknown | 60.7 | 0.080 | 196 ms (hosted) | Not fine-tunable by us | See Section 3. |

Other notes:

- **Many "Jev" names on Hugging Face are unrelated or empty.** A search for "jev" returns dozens of repositories created this month, many with 0 downloads and no card. Only models with a board entry or a substantive card are listed above.
- The strongest sub-1B entries on the board (`chiranko/PKU-KCL-Meteor-0.8B`, `Valen-Team/Valen-Decision-1.5-0.8B`, `fastino/GLiDE-0.8B`) are **autoregressive Qwen3.5-0.8B decoders used as scorers**. They fall outside the "non-autoregressive" brief and are not considered further.
- **Calibration ranking among small open models (zero-shot, generic tasks):** GLiDE-0.4B (0.049) < tasksource-decider-nano (0.070) < Jev (0.080) < GLiNER2.5-Decide (0.088) < GLiClass v4.1 edge (0.118) [measured]. Treat this as a tiebreaker only. After fine-tuning on our data, calibration must be re-estimated from scratch.

### 4.2 Why `tasksource/tasksource-decider-nano` for the CPU prototype

1. **Licence.** Apache-2.0, with no share-alike or revenue-threshold clauses. GLiDE-0.4B is better calibrated, but its licence would bind any fine-tune we make.
2. **Fit to the brief.** It is non-autoregressive, follows the Jev `state`/`questions` interface, and returns softmax probabilities over the supplied options. A `noul` question ("Will this market resolve YES?" or "Is this window informed flow?") maps directly to our binary targets.
3. **Long context.** About 8k tokens is enough for a market description, resolution rules and several headlines, which the 512-token DeBERTa models cannot fit.
4. **Size.** At 149M parameters it is the largest model we expect to fine-tune comfortably on 4 CPU cores (Section 4.3). Its card describes encoding the state once per request through a tree attention mask, which keeps multi-question inference cheap.
5. **Risk.** It uses custom inference code (`decider.py`). Pin the revision and read the code before running it, because it executes Python from the repository.

**Fallbacks.** Use `knowledgator/gliclass-instruct-edge-v1.0` (32.7M) if 149M is too slow. If typed-decision pretraining turns out not to matter after fine-tuning, use `jhu-clsp/ettin-encoder-68m` with a plain classification head.

**With a GPU.** Use `knowledgator/gliclass-v4-edge` (440M, Apache-2.0, a Jev-compatible server is in the card) and fine-tune it fully or with LoRA. Use `fastino/GLiDE-0.4B` only after legal review of the Fastino Community License. A GPU also makes it practical to run several seeds and nested cross-validation for the transformer, which matters more than model size when only about 100 positives exist.

### 4.3 CPU feasibility on this box (4 cores, 15 GB, no GPU)

CPU_BENCH_PLACEHOLDER

---

## 5. Q2: Input representation (text tokens, numeric embeddings, or keep numbers out)

**Evidence.**

- Gradient-boosted trees remain state of the art on medium-sized tabular data of about 10k samples. One stated reason is that neural networks are not robust to uninformative features and struggle with irregular target functions ([abstract] 2207.08815). XGBoost beat proposed deep tabular models even on those models' own datasets and needed far less tuning ([abstract] 2106.03253).
- The largest comparison (19 algorithms, 176 datasets) finds the GBDT-versus-NN gap often negligible. However, **GBDTs are much better on skewed or heavy-tailed features** ([abstract] 2305.02997), which describes trade sizes, wallet ages and price impact exactly.
- Learned numeric embeddings (piecewise-linear, periodic) materially help deep tabular models ([abstract] 2203.05556). Benefiting from them means training those layers, which needs data we do not have.
- Number-as-text tokenisation is fragile. Even frontier LLMs show systematic, tokenisation-dependent arithmetic errors ([abstract] 2402.14903). Continuous number encodings such as xVal require training a model from scratch ([abstract] 2310.02989), so they do not apply to a pretrained encoder.
- Serialising time series as digit strings works for *large* LLMs on forecasting, but calibration degrades with alignment-tuned models ([abstract] 2310.07820). Simple linear models beat many transformer time-series forecasters ([abstract] 2205.13504).
- Language-model tabular learners (TabuLa-8B) beat XGBoost in 1 to 32-shot settings ([abstract] 2406.12031). That is an 8B-parameter model, and LLMs also memorise popular tables, which inflates few-shot results ([abstract] 2404.06209).
- Tabular foundation models (TabPFN-2.5, TabICLv2, TabPFN-3) now claim to beat tuned GBDTs on small tables ([abstract] 2511.08667, 2602.11139, 2605.13986). They are worth running as baselines, subject to licence (Section 4.1).

**Recommendation.** Keep raw numeric trade data **out of the transformer**. LightGBM (and optionally TabPFN-v2 or TabICL as a check) owns all numbers. The system one model gets:

1. The text it is uniquely good at: market title, resolution criteria, event type, and the headlines published before the window cutoff.
2. **At most five verbalised, binned flow descriptors**, computed point-in-time, for example: "Largest bet in window: top 1% of this market's history. Wallet first seen: less than 24 hours before. Bet direction: against the current favourite. Price moved 4 points within 10 minutes."

Descriptor (2) is a hedge [speculative]: it lets the encoder condition text relevance on whether anything unusual happened. Ablate it in week 2 (text only versus text plus descriptors). No published evidence was found on binned verbalisation versus raw digits for small encoders, so **the evidence here is thin**.

---

## 6. Q3: Integration architecture

### 6.1 Options compared

| Option | What it is | Strengths | Failure modes |
|---|---|---|---|
| **(a) Stacking**: logistic-regression meta-model on log-odds, fitted on out-of-fold predictions | `logit p = w0 + w1·logit(p_mkt) + w2·logit(p_lgbm) + w3·logit(p_s1)` | Few parameters, so it is stable with about 100 positives. Weights are interpretable and test directly whether each component adds information beyond the market. | Out-of-fold predictions must come from *time-forward* folds, or the meta-model learns leaked confidence. Collinearity between `p_lgbm` and `p_s1`. Weights drift across regimes (election seasons versus wars). |
| **(b) System one output as a LightGBM feature** | LightGBM takes `p_s1` with the flow features | One model to serve. | Needs *nested* out-of-fold transformer predictions inside every LightGBM fold, and our data is too small for that nesting. A tree will split hard on a single strong feature and overfit it. The joint model becomes hard to diagnose. **Rejected.** |
| **(c) Gating**: system one runs only on windows LightGBM flags | Two-stage cascade | Saves compute, and the transformer only ever sees "interesting" windows. | The training distribution for system one becomes conditional on the gate, so changing the gate threshold silently invalidates it. Gate errors are unrecoverable (false negatives never reach stage 2). With about 100 positives, gating discards most labelled negatives that system one needs. |
| **(d) Bayesian update with the market price as prior** | `logit p = logit(p_mkt) + Σ likelihood-ratio terms` | Principled. The default is the market, and the market is a strong forecaster: a 43-feature walk-forward logistic model on Polymarket BTC 15-minute markets did *not* beat the order-book-implied probability (OpenMarket, [abstract] 2607.26245). | Treats component outputs as conditionally independent evidence, which double-counts when LightGBM and system one react to the same flow. The market price is not calibrated everywhere (thin books, long-shots) and does not equal the true prior. If the likelihood ratios are not regularised, it overreacts to rare flags. |

### 6.2 Recommended: market-anchored, shrunk stacking (options a and d)

```
logit p_final = logit(p_mkt) + β0 + β1·z_lgbm + β2·z_s1 + β3·z_lgbm·1[gate] + β4·g(liquidity)·z_lgbm
```

- `z_*` is the component's calibrated log-odds minus `logit(p_mkt)`, which is the component's *disagreement* with the market. Where a component is trained on an "informed-window" target rather than on resolution, use its calibrated logit directly.
- `logit(p_mkt)` enters as a fixed **offset** (coefficient 1). This is the Bayesian-prior part of (d). As a robustness check, also fit a variant with a free coefficient, since prediction-market long-shots may be miscalibrated [standard practice; the favourite-longshot bias was not re-sourced here].
- The `β` terms are fitted by **L2-regularised logistic regression** on time-forward out-of-fold predictions, so they are shrunk toward 0 and the default prediction is the market. This is the same idea as the "competence gate" in [abstract] 2609.12101: estimate source weights from resolved outcomes, shrink uncertain ones toward a global weight, and recalibrate. That paper found real gains over an external forecast on structured questions but **no significant improvement on the ForecastBench market subset, where the gate largely deferred to the market**. That is the honest prior for our project.
- Gating enters only through the `β3` interaction and as a compute shortcut at serving time (system one need not run on unflagged windows if `β2` is near 0 there). It never changes which windows system one is *trained* on.
- `g(liquidity)` allows the flow signal to matter more in thin markets, where a single informed wallet can move price but the price may also be noisier [speculative].

### 6.3 Component diagram, described in words

1. **Ingest** (owned by `data/`). Polymarket on-chain fills with ground-truth aggressor direction, plus market metadata, book snapshots and Kalshi trades and markets. Point-in-time news headlines with first-seen timestamps.
2. **Windowing.** For each market and each decision timestamp *t*, build a trade window ending at *t*, and freeze the market mid price, spread and depth at *t*.
3. **Feature store** (owned by `features/`). Point-in-time wallet features (age, prior trade count, prior markets, funding-source cluster ID treated as opaque), flow features (signed volume, imbalance, bet size relative to depth, burstiness), price-impact features and timing features (time to scheduled events, time since last headline).
4. **Component A, LightGBM.** Reads the features and outputs `p_lgbm` (resolution, or informed-window probability), then calibrates it with Platt scaling on its own out-of-fold predictions.
5. **Component B, system one.** Reads the rendered `state` (text context plus five verbalised descriptors) and one `noul` question, outputs `p_s1`, then calibrates it with temperature or Platt scaling on its own out-of-fold predictions.
6. **Market prior.** `p_mkt` at *t*, using the mid price; executable bid/ask is used for the trading simulation.
7. **Meta-model.** The shrunk logistic stacker from 6.2 outputs `p_final`. A Venn-Abers wrapper adds an interval `[p0, p1]` as an "insufficient evidence" flag.
8. **Evaluation and simulation** (owned by `evaluation/`). Walk-forward scoring against the market-only baseline, plus a fee- and slippage-aware paper-trading simulation. No live orders.

---

## 7. Q4: Calibration

| Stage | Method | Why |
|---|---|---|
| System one, after fine-tuning | **Temperature scaling** (1 parameter) or Platt scaling (2 parameters) on out-of-fold logits | Temperature scaling is "surprisingly effective" for modern neural nets ([abstract] 1706.04599). Pretrained transformers are fairly calibrated in-domain and temperature scaling improves this further, while label smoothing helps out-of-domain ([abstract] 2003.07892). Low parameter count suits the small sample. |
| LightGBM | **Platt scaling** on out-of-fold scores | Two parameters. Isotonic regression and histogram binning are sample-hungry: binning needs O(B/ε²) samples against O(1/ε²) for scaling methods, but scaling methods can be *less calibrated than reported* ([abstract] 1909.10155). A large post-hoc calibration benchmark finds **smooth calibration functions outperform binning-based ones** ([abstract] 2605.30188). |
| Final output | The meta-model is itself a logistic (Platt-type) recalibration. Add an **inductive Venn-Abers** wrapper once more than about 300 resolved outcomes exist | Venn-Abers predictors give calibration guarantees under minimal assumptions for transformer outputs ([abstract] 2205.10586). The width of the `[p0, p1]` interval is a useful "do not trade" signal when data is thin [standard practice]. |
| Isotonic | Avoid until there are at least ~1,000 calibration points per stratum | It is step-shaped, overfits small samples and outputs probabilities of exactly 0 or 1 [standard practice; consistent with 1909.10155 and 2605.30188]. |

**Class imbalance.** Do not oversample positives before calibration. If class weights or negative subsampling are used to train LightGBM or system one, correct the logit back to the true base rate (subtract the log of the sampling ratio) *before* stacking [standard practice]. Report calibration on the natural class distribution of each time fold. With dozens to hundreds of positives, binned ECE is very noisy. Prefer **log loss, Brier score and its reliability/resolution decomposition**, and reliability curves with bootstrap bands clustered by event. Report ECE only alongside its confidence interval.

---

## 8. Q5: Small-data strategy

1. **Label at the trade-window level, and also define an abundant proxy target.** Two targets:
   - (i) *Resolution*: does the market resolve YES. One label per market, and correlated within an event.
   - (ii) *Pre-news drift proxy*: did the price move at least *k* points in the bet's direction within *h* hours after the window, before any public headline. This yields thousands of labels from public data.
   
   Pretrain LightGBM and fine-tune system one on (ii), then let the meta-model learn how (ii) predicts (i). This is the main lever against scarcity [speculative but standard in spirit: proxy-label pretraining].
2. **Weak supervision for "informed window" labels.** Write labelling functions: fresh wallet plus large size plus one-sided plus less than X hours before the first headline plus later correct. Combine them with a label model. Programmatic weak supervision is a mature field with standard benchmarks showing large variance across methods and sources ([abstract] 2202.05433, 2109.11377), so evaluate the label model on the hand-labelled set, not on itself. **Circularity trap:** if a labelling function uses "later correct", do not use the resulting labels to predict *resolution*. Use them only for the informed-window target.
3. **Positive-unlabelled framing.** Publicly documented insider cases are a biased tiny subset of true informed flow. In equities, the SEC is reported to litigate only about 8% of takeover deals with suspicious options activity (Augustin, Brenner and Subrahmanyam, [unverified]). Treat unlabelled windows as unlabelled, not negative.
4. **Transfer.** Directly reusable: on-chain prediction-market microstructure work, especially Polymarket-v1's finding that tick-rule and bulk-volume classification are near random on Polymarket and that inferred VPIN diverges from ground truth ([abstract] 2606.04217). Use ground-truth aggressor direction, never inferred direction. Equities and options insider literature (abnormal short-dated out-of-the-money call volume before takeovers [unverified]) transfers as **feature ideas** (abnormal size relative to the instrument's own history, short horizon, out-of-the-money equivalent = long-shot side), not as training data.
5. **Synthetic augmentation.** The project rules (`CLAUDE.md`) forbid substituting synthetic data for real data. **Do not fabricate trades or windows.** Acceptable augmentations, applied to training text only and never to evaluation data: paraphrasing the market title, reordering headlines, varying the descriptor order. Their value for small encoders is unproven here [speculative].
6. **Few-shot alternative for the text side.** SetFit-style contrastive fine-tuning of a small sentence encoder reaches competitive few-shot accuracy and is an order of magnitude faster to train than prompt-based methods ([abstract] 2209.11055). It is a cheap CPU baseline for system one.
7. **Avoiding overfitting.**
   - LightGBM: `num_leaves` ≤ 8, `min_data_in_leaf` ≥ 20, strong `lambda_l2`, monotone constraints where economics dictates, and at most ~25 features chosen *before* looking at test folds.
   - System one: at most 2 to 3 epochs, early stopping on time-forward validation log loss, and averaging over at least 3 seeds.
   - Freeze the feature list and hyperparameter grid in writing before week 2, log every experiment, and report the number of configurations tried [standard practice].

---

## 9. Q6: Evaluation design

**Validation scheme.**

- **Walk-forward by decision timestamp, grouped by event.** All markets of one event (for example every "Will X happen by date D" ladder) and any cross-listed Kalshi market go in the same fold. Cross-platform duplicates are common: about 6% of events are listed concurrently across venues ([abstract] 2601.01706), and markets within topical clusters have strongly dependent outcomes ([abstract] 2512.02436).
- **Purge and embargo.** Drop training windows whose market resolves after the test fold starts, and leave a gap of at least the longest feature lookback [standard practice].
- Use at least 4 to 6 expanding folds. Because positives are few, report per-fold metrics, not only pooled ones.

**Leakage traps specific to event markets.**

1. Any field written after *t*: final price, resolution flag, dispute status, market description edits or clarifications, volume totals.
2. Wallet features using the wallet's *future*: later profitability, later "known insider" lists, cluster assignments computed on the full history. Every wallet feature must be computed only from blocks before *t*.
3. News timestamps from sources that backdate or edit stories. Use first-seen crawl time, not the publication field.
4. Pretrained-model contamination. An encoder or news model trained on text after *t* may "know" the outcome. This is a known problem in forecasting evaluation ([abstract] 2506.00723), and the Decision Index itself flags it for Jev on ForecastBench. Keep text features low-information about outcomes (event type, headlines before *t*), and run a sanity check that system one cannot predict outcomes from the market title alone.
5. **Outcome-selected labels.** Insider cases are discovered *because* a long-shot bet won. A model trained on them learns "long-shot plus won", so any test on the same case list is circular.
6. Execution artefacts. Polymarket off-chain matches can revert on-chain ("ghost fills"), with 980k reverted filled orders documented and over 24% of fills reverting at peak ([abstract] 2606.16852). Use settled fills only.
7. Inferred trade direction. Tick rule and bulk-volume classification are near random on Polymarket ([abstract] 2606.04217), so use on-chain aggressor direction.
8. Train/test overlap through correlated series. This failure mode is documented for time-series foundation-model evaluation ([abstract] 2510.13654).

**Testing whether the combined model beats the market.**

- **Primary metric.** Paired difference in **log loss** between `p_final` and `p_mkt`, at identical timestamps and on identical rows. Secondary: Brier score and **Brier skill score relative to the market**. Confidence intervals come from a **block bootstrap clustered by event**, plus a Diebold-Mariano-style test on per-event loss differentials [standard practice].
- **Calibration.** Reliability curves for `p_final` and `p_mkt` on the same rows, plus calibration slope and intercept.
- **Simulated returns, paper only.**
  - Trade only when `p_final` differs from the *executable* price by more than fees plus half-spread plus modelled slippage taken from book depth at *t*.
  - Cap position size by depth.
  - Report hit rate, profit and loss per unit risked, and drawdown, and deflate Sharpe-type statistics for the number of configurations tried.
  - The OpenMarket authors' simulated strategy lost money (-0.116 normalised payoff units per attempted trade) under stated fee and slippage assumptions ([abstract] 2607.26245). Use that as a reality check.
  - Fee schedules for Polymarket and Kalshi were **not verified** in this review; take them from the platforms' current documentation.
- **Go/no-go rule for the prototype.** Continue only if the log-loss improvement over the market is positive in most folds *and* its event-clustered 90% interval excludes zero on the pooled set.

---

## 10. Q7: Prior art

**Prediction-market data and microstructure (opened, abstract-level):**

- **Polymarket-v1 Database** (2606.04217). 1.20 billion trades with 100% ground-truth aggressor direction, Nov 2022 to Apr 2026. Classical trade classifiers are near random, and inferred VPIN and order-flow imbalance are biased. This is the most important data reference for this project.
- **OpenMarket** (2607.26245). A walk-forward logistic model on 43 microstructure features did not beat the Polymarket book-implied probability out of sample.
- **Ghosts of Polymarket** (2606.16852). Reverted settlements and attack patterns.
- **SoK on decentralised prediction-market microstructure** (2510.15612). Covers design variants including manipulation resistance.
- **Arbitrage on Polymarket** (2508.03474).
- **Cross-venue Law-of-One-Price violations** (2601.01706). Persistent 2 to 4% deviations.
- **Semantic Trading** (2512.02436). LLM clustering of related markets.
- **PolyBench** (2604.14199). Only 2 of 7 LLMs earned positive returns on live Polymarket states.
- **KalshiBench** (2512.16030). Frontier LLMs are systematically overconfident on Kalshi questions; only one model achieved a positive Brier skill score.
- **Kalshi macro contracts forecasting crypto volatility** (2604.01431). Kalshi prices carry information beyond conventional instruments.

**Forecasting with markets as a baseline:**

- ForecastBench (2409.19839): expert forecasters beat the best LLM.
- Retrieval-augmented LM forecasting near the crowd aggregate (2402.18563).
- Competence-gated pooling (2609.12101): defers to market prices on the market subset.
- Verifiable rewards for calibrated forecasting, which matched betting-market calibration on NFL win probability (2607.00164).

**Informed-trading measurement [all unverified; search summaries only]:**

- PIN models (Easley, O'Hara and co-authors).
- **VPIN** (Easley, López de Prado and O'Hara, *Review of Financial Studies* 2012), and the **Andersen and Bondarenko critique** (*Journal of Financial Markets* 2014; *Review of Finance* 2015), which found that VPIN based on bulk-volume classification has no incremental predictive power once volume and volatility are controlled for. Polymarket-v1's abstract independently shows that inferred VPIN is unreliable on Polymarket.
- **Augustin, Brenner and Subrahmanyam** (*Management Science* 2019): abnormal options volume before about 25% of 1,859 US takeovers, concentrated in short-dated out-of-the-money calls.

**Polymarket-specific insider studies [unverified]:**

- Mitts and Ofir (2026), a wallet screen over about 93k markets and 50k wallets, summarised on the Harvard Law School Forum post "From Iran to Taylor Swift".
- The Anti-Corruption Data Collective's report flagging 152 wallets.
- Bloomberg coverage reporting that 57% of top flagged wallets were created less than 24 hours before trading.
- arXiv 2605.02286 on documented insider cases. It is not in the Hugging Face index and could not be opened.

**On-chain wallet clustering.** No primary source could be opened in this review; **evidence gap**. Treat funding-source clusters as opaque IDs per project rules.

---

## 11. Q8: Compliance

**Everything in this section is unverified** (search-engine summaries only). Have counsel review it before any live use. The project already commits to no live trading, public data only, and no identification of individuals (`CLAUDE.md`).

- **Polymarket.** Search summaries describe a **March 2026 update** adding explicit insider-trading rules to Polymarket's Terms of Use and the Polymarket US rulebook. The three prohibited categories are trading on stolen or confidential information, trading on illegal tips, and trading on events the user can influence. Manipulation (spoofing, wash trading, front-running) is also prohibited, with penalties including suspension and trade cancellation. Read the current Terms of Use and rulebook directly. Also check geographic eligibility (international Polymarket versus the CFTC-regulated Polymarket US) and the API and data terms for the endpoints we use.
- **Kalshi.** It is a CFTC-designated contract market, and its rulebook reportedly prohibits insider trading and trading on events one can influence. Kalshi has reportedly frozen accounts and disclosed internal enforcement cases.
- **CFTC.** It reportedly issued a **prediction-markets enforcement advisory on 2026-02-25**, relying on CEA §6(c)(1) and Regulation 180.1 for fraud-based misuse of nonpublic information, and an **advance notice of proposed rulemaking on event contracts on 2026-03-12**. One law-firm summary notes that Rule 180.1 is narrower than a general ban on trading while aware of material nonpublic information.
- **Our position.** A model built only from public on-chain data and public news is, in principle, analysis of public information, not trading on material nonpublic information. Open questions for counsel [speculative]:
  1. Whether systematically copying flow that the model believes reflects misappropriated information creates exposure under misappropriation or "tippee"-style theories, or under platform rules on front-running.
  2. Whether publishing flagged wallets could be defamatory or breach platform terms.
  3. Jurisdictional eligibility to trade on each venue.
  4. Data-use limits in each API's terms.

  The safe operating mode for now is research and paper trading only, with aggregate reporting that never names wallets.

---

## 12. Two-week prototype plan, top risks

### 12.1 Plan (CPU box; owners follow the `src/pmflow/` subpackages)

| Day | Work | Exit criterion |
|---|---|---|
| 1 | Freeze the label definitions: (i) resolution and (ii) pre-news drift proxy, plus weak-supervision rules for the informed-window label. Write the leakage checklist (Section 9). | Signed-off label spec. List of geopolitical markets with decision timestamps. |
| 2 | Point-in-time window builder using settled fills and ground-truth aggressor direction. Capture market mid, spread and depth at *t*. | Unit test showing no field is later than *t*. |
| 3 | Walk-forward, event-grouped fold generator with purge and embargo. Market-only baseline metrics per fold. | Baseline log loss and Brier score per fold. |
| 4 to 5 | LightGBM with ≤ 25 features, constrained, out-of-fold Platt calibration. TabPFN-v2 / TabICL as side baselines on the same folds. | Out-of-fold `p_lgbm`. Paired log-loss difference against the market. |
| 6 | Render the system one `state` text (context plus ≤ 5 descriptors). Run **zero-shot** `tasksource/tasksource-decider-nano` with one `noul` question on all windows. Time it. | Out-of-fold zero-shot `p_s1` and its log-loss difference against the market. |
| 7 to 8 | Fine-tune system one on proxy target (ii) using time-forward folds, 3 seeds, ≤ 3 epochs, input ≤ 512 tokens. Fall back to `gliclass-instruct-edge-v1.0` if throughput is too low. | Fine-tuned out-of-fold `p_s1`. Keep it only if it beats zero-shot on validation log loss. |
| 9 | Ablations: text only versus text plus descriptors; system one on all windows versus gated. | Ablation table. |
| 10 | Market-anchored shrunk stacker (Section 6.2). Coefficients with bootstrap confidence intervals. | `β` estimates and intervals; check whether `β2` is distinguishable from 0. |
| 11 | Calibration report: reliability curves, Brier decomposition, Venn-Abers intervals if n allows. | Calibration section. |
| 12 | Paper-trading simulation with executable prices, fees, depth-capped size and deflated statistics. | Simulated profit and loss with sensitivity to fees and slippage. |
| 13 | Robustness: Kalshi-only fold, thin versus thick markets, excluding documented insider cases. | Robustness table. |
| 14 | Go/no-go write-up against the rule in Section 9. | Decision memo. |

### 12.2 Top five risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | **No edge over the market.** Published evidence says this is the likely outcome (2607.26245, 2609.12101). | The market-anchored design makes "no edge" a clean, cheap finding. Pre-register the go/no-go rule. Concentrate effort on thin, slow-to-update markets where the market prior is weakest [speculative]. |
| 2 | **Label scarcity and outcome-selected labels** (models learn "long-shot won"). | Proxy-target pretraining, positive-unlabelled framing, and weak-supervision rules kept separate from the resolution target. Evaluate with documented cases excluded. |
| 3 | **Temporal leakage** through wallet history, edited metadata, backdated news or contaminated pretrained models. | Point-in-time feature tests, event-grouped purged walk-forward folds, first-seen news timestamps, and a title-only contamination check for system one. |
| 4 | **System one overfits or adds nothing** at about 100 positives on CPU. | Zero-shot first. Fine-tune only on the abundant proxy target. Use 3 seeds and temperature scaling, and drop the component if `β2` is about 0. Most of the budget goes to LightGBM and evaluation. |
| 5 | **Compliance and licence exposure** (platform insider rules, CFTC posture, model licences such as the Fastino Community License, TabPFN-3.5's non-commercial terms, CC-BY-NC decision models). | Research and paper trading only. No wallet naming. Counsel review of Section 11 before any live use. Restrict the model shortlist to Apache-2.0, MIT and BSD components. |

---

## 13. Sources (all opened during this review)

Hugging Face model cards and API records (licence, size, context, card claims):

- https://huggingface.co/tasksource/tasksource-decider-nano (README) and https://huggingface.co/api/models/tasksource/tasksource-decider-nano
- https://huggingface.co/knowledgator/gliclass-v4-edge (README) and https://huggingface.co/api/models/knowledgator/gliclass-v4-edge
- https://huggingface.co/api/models/knowledgator/gliclass-v4.1-edge and https://huggingface.co/knowledgator/gliclass-v4.1-edge/raw/main/README.md
- https://huggingface.co/knowledgator/gliclass-instruct-edge-v1.0 (README) and API record
- https://huggingface.co/knowledgator/gliclass-edge-v3.0 (README) and API record
- https://huggingface.co/api/models/knowledgator/gliclass-modern-base-v3.0, https://huggingface.co/api/models/knowledgator/gliclass-instruct-base-v1.0, https://huggingface.co/api/models/knowledgator/gliclass-multilang-edge, https://huggingface.co/api/models/knowledgator/gliclass-v4-2b
- https://huggingface.co/fastino/GLiDE-0.4B (README) and https://huggingface.co/fastino/GLiDE-0.4B/raw/main/LICENSE
- https://huggingface.co/fastino/GLiNER2.5-Decide (README) and API record
- https://huggingface.co/fastino/gliner2.5-small-v1 (README) and API record; https://huggingface.co/api/models/fastino/GLiNER2.5-Decide-1B
- https://huggingface.co/MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33, https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0, https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0 (READMEs and API records)
- https://huggingface.co/jhu-clsp/ettin-encoder-32m (README); API records for `jhu-clsp/ettin-encoder-17m`, `-68m`, `-150m`, `jhu-clsp/mmBERT-base`
- https://huggingface.co/answerdotai/ModernBERT-base (README) and API record; API record for `answerdotai/ModernBERT-large`
- API records: `microsoft/deberta-v3-xsmall`, `microsoft/deberta-v3-small`, `microsoft/deberta-v3-base`, `cross-encoder/nli-deberta-v3-small`, `cross-encoder/ettin-reranker-150m-v1`, `facebook/bart-large-mnli`, `BAAI/bge-reranker-v2-m3`, `chandar-lab/NeoBERT`, `EuroBERT/EuroBERT-210m`, `ProsusAI/finbert`, `urchade/gliner_multi-v2.1`, `tardellirs/crivo-150m`, `MLukauskas/amber-0.4b-v2`, `Horizon-Labs/multilingual-zeroshot-small`, `Elfsong/jev-ettin-150m-v16`, `shreyanbr/system-one-zeroshot`, `leobitz/jev-berta-base-zeroshot-classifier`, `mohameddhameem/systemone-jev-qp`, `jingang/TabICL-clf`, `jingang/TabICL`
- https://huggingface.co/Prior-Labs/tabpfn_3_5 (README with licence gating text), https://huggingface.co/Prior-Labs/TabPFN-v2-clf (README)

Decision Index (third-party benchmark of Jev and its open reproductions):

- https://huggingface.co/spaces/multimodalart/jev-decision-index (README), https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/methodology.json, https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/data/index.json
- https://huggingface.co/datasets/multimodalart/decision-index-leaderboard (README)
- https://huggingface.co/spaces/pngwn/open-jev (README)

Papers (abstracts read via the Hugging Face papers API, `https://huggingface.co/api/papers/<id>`; human-readable page `https://huggingface.co/papers/<id>`):

- 2412.13663 ModernBERT; 2507.11412 Ettin (Seq vs Seq); 2111.09543 DeBERTaV3; 2311.08526 GLiNER; 2507.18546 GLiNER2; 2508.07662 GLiClass; 2312.17543 NLI universal classifiers; 1909.00161 zero-shot text classification benchmark
- 2207.08815 Why tree-based models still outperform deep learning; 2106.03253 Tabular data: DL is not all you need; 2305.02997 When do neural nets outperform boosted trees; 2203.05556 Embeddings for numerical features; 2106.11959 Revisiting DL for tabular data
- 2310.07820 LLMs are zero-shot time-series forecasters; 2205.13504 Are transformers effective for time-series forecasting; 2403.07815 Chronos; 2310.02989 xVal; 2402.14903 Tokenization counts
- 2207.01848 TabPFN; 2511.08667 TabPFN-2.5; 2605.13986 TabPFN-3; 2502.05564 TabICL; 2602.11139 TabICLv2; 2406.12031 TabuLa-8B; 2404.06209 Elephants never forget
- 1706.04599 On calibration of modern neural networks; 2003.07892 Calibration of pre-trained transformers; 1909.10155 Verified uncertainty calibration; 2205.10586 Venn-Abers for NLU; 2605.30188 CalArena
- 2209.11055 SetFit; 2109.11377 WRENCH; 2202.05433 Survey on programmatic weak supervision
- 2606.04217 Polymarket-v1 Database; 2606.16852 Ghosts of Polymarket; 2607.26245 OpenMarket; 2508.03474 Arbitrage in prediction markets; 2510.15612 SoK DePM microstructure; 2601.01706 Semantic non-fungibility; 2512.02436 Semantic Trading; 2604.14199 PolyBench; 2512.16030 KalshiBench; 2604.01431 Kalshi macro contracts and crypto volatility
- 2409.19839 ForecastBench; 2402.18563 Approaching human-level forecasting; 2506.00723 Pitfalls in evaluating LM forecasters; 2510.13654 Information leakage in TSFM evaluation; 2609.12101 Competence-gated pooling; 2607.00164 Verifiable rewards for calibrated forecasting

## 14. Unverified leads (seen only in search-result summaries; NOT opened, NOT sources)

Every claim tagged **[unverified]** above rests on these. Open them before relying on them.

- Jev descriptions: datacamp.com/blog/system-one-models-jev; inovex.de review of TypeSafe Jev; digitalocean.com "What is Jev"; deeplearning.ai The Batch "models built to do one thing well"; docs.aimlapi.com Jev API reference; openrouter.ai Jev guide.
- Polymarket rules: businesswire.com release of 2026-03-20; defirate.com "Polymarket publishes enhanced market integrity rules"; decrypt.co/285857 (pre-update terms).
- Kalshi and CFTC: covers.com (Kalshi insider-trading probe, 2026-02-24); taftlaw.com enforcement-trends note; willkie.com CFTC advisory summary; kslaw.com "New trick, same crime"; Dechert "The CFTC, prediction markets and event contracts".
- Insider studies: corpgov.law.harvard.edu/2026/03/25/from-iran-to-taylor-swift-informed-trading-in-prediction-markets; acdatacollective.org "Insider risks in Polymarket political markets"; arxiv.org/pdf/2605.02286; zenodo.org/records/23694013.
- Microstructure and options: papers.ssrn.com/abstract=2062450 (VPIN and the Flash Crash: A Comment); ideas.repec.org Andersen and Bondarenko (*Review of Finance* 2015); econpapers.repec.org Augustin, Brenner and Subrahmanyam (*Management Science* 65(12), 2019).
