# Weak-labeling plan (trade-window level)

## Unit
A label is assigned to a pair (market m, window W). A window is W = [t_cut - L, t_cut), where
L is one of {1h, 6h, 24h, 72h}. The cutoff t_cut is the lower edge of the news interval
(`ReportedCase.news_window()[0]`, or the estimate of section 3).
- "Pre" windows end before t_cut.
- "Reaction" windows start at or after the upper edge of the news interval.
- Baseline windows use the same L, are drawn from the same market at least 7 days before t_cut, and
  exclude any other known news interval.

## 1. Window quantities (all computable from trades before t_cut)
NO trades are mapped to YES-equivalent notional.

- **NSN:** net signed notional. YES bought minus YES sold, in USD.
- **GV:** gross volume.
- **OS:** one-sidedness, |NSN| / GV.
- **p_entry:** VWAP of the dominant side.
- **FWS:** fresh-wallet share of the dominant side's notional. A wallet is fresh if its first-ever
  Polymarket trade was less than 72h before that trade.
- **TOP3:** share of the dominant side held by the top-3 wallets.
- **CONC:** for the top wallets, the share of their lifetime notional that is in m.
- **z:** robust z-score of NSN against the baseline windows, (NSN - median) / (1.4826 * MAD).

## 2. Labels
**L1, likely informed (strong).** All of the following:
- (a) z >= 4 and NSN >= 5x the baseline median |NSN| and NSN >= $10k.
- (b) OS >= 0.70.
- (c) p_entry <= 0.30, i.e. the market priced the side at 70% or more against.
- (d) FWS >= 0.5, or TOP3 >= 0.5 with CONC >= 0.8.
- (e) The window ends before the news interval, and the case's ts_precision is minute or hour. For
  day precision, the window must end before 00:00Z of the date, or the case is excluded.
- (f) The dominant side was vindicated.

**L2, possibly informed.** (a), (b), (e) and (f), plus either (c) or (d).

**N1, pattern but wrong (hard negative).** (a) through (e) hold, but (f) fails. Example: PM-012.

**N0, quiet.** A baseline window.

**R, reaction.** A post-news window. It is excluded from "pre" training and kept for evaluation.

**Calibration.** Check that seeds PM-001, PM-003, PM-007 and PM-011 come out L1 at some L. Check that
PM-012 comes out N1, PM-022 is not L1, and PM-023 lands only in R.
- Report how X = 5 and the thresholds in (b) and (c) trade off seed recall against the N1/L1 ratio on
  unlabelled in-scope markets.
- (d) must stay a disjunction. PM-008 shows an experienced account can carry the signal.

## 3. News timestamps at scale (markets without a reported case)
Options, from most to least reliable:

1. **Scheduled-event text.** The market description and resolutionSource (Gamma) give scheduled
   times, e.g. Nobel at 11:00 Oslo, FOMC 14:00 ET, BLS 08:30 ET, poll close. Exact and reliable, but
   only for scheduled events.
2. **Manual curation via web search** (as in this seed set). Best quality, slow. Use it to audit a
   sample of the others.
3. **Price-jump onset.** The first 1-minute bar where |dp| >= 0.15 within 30 minutes, followed by a
   breadth surge (at least 20 distinct wallets trading within 30 minutes). Scales to every market, but
   it is circular: the informed flow itself can create the "jump". Use the breadth surge, not the first
   move. Validate against the six fine-timestamp seed events; on PM-001, for instance, the jump came
   after 06:15Z while the pre-news drift was earlier.
4. **Gamma closedTime and UMA resolution timestamps; Kalshi settlement_ts.** Upper bounds only. They
   can lag by hours or weeks (Warsh: news Jan 29, close Mar 4). Never use one as t_cut, except as a
   bound.
5. **News APIs.**
   - GDELT (api.gdeltproject.org) returns 403 from this sandbox.
   - General news sites are unreachable through WebFetch (DNS failure).
   - Allowed hosts are only the Polymarket APIs, api.elections.kalshi.com, api.goldsky.com and
     huggingface.co. Hugging Face news datasets with timestamps are the only possibly-reachable corpus;
     their coverage and latency are untested.
   - Requesting network access for GDELT is the cheapest real fix.

## 4. Leakage risks
- **(f) vindication uses the outcome.** It is a label only, never a feature. The same applies to
  realised PnL, redemptions, `outcome_vindicated` and resolution status.
- **Price-jump t_cut** uses prices after the window. Use it only to place labels, with a margin of
  W ending at least 1h before the jump onset. Any feature like "minutes until jump" is forbidden.
- **closedTime** encodes when resolution happened. Use it only as a bound.
- **Fresh-wallet and CONC definitions** must use only a wallet's history up to the trade time, never
  its later activity. Cluster links found from later withdrawals or exit behaviour (PM-005, PM-010,
  PM-015) are labels or context only.
- **Baseline windows** must lie entirely before t_cut.
- **Selection bias.** Reported cases are selected on being vindicated and newsworthy. Seed recall
  therefore overstates real recall, and negatives are scarce. Oversample N1 windows mined
  automatically, and keep PM-012, 013, 014, 002 and 022 in every evaluation.
- **Contract semantics** (PM-002) can make informed flow look wrong. Add a label-noise flag when the
  resolution was disputed or ruled on wording.
- **Day-precision cases** are ambiguous across time zones. The loader pads to 12:00Z of the next day;
  exclude them from L1 unless they have an upper bound.
