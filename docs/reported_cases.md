# Reported cases of suspected informed trading (seed set)

Labeling agent, 2026-11-17. Machine-readable: `artifacts/labeling/reported_cases.csv`; load with
`pmflow.labeling.cases.load_cases()`.

Flow patterns only. No names of charged or disciplined individuals are recorded, even though the
sources name them. Handles appear only as opaque IDs and only where the reports publish them.
One more negative case (a large losing NO bet on the Feb-28-2026 Iran strike) was excluded because
the reporting is about attributing the account to a named private person, and the handle contains
that name.

## Method and main limitation
- Sources were read through web-search result extracts only. WebFetch to news hosts failed DNS in this
  sandbox (npr.org, theblock.co, wtop.com). Every row cites the extracted URLs; see the CSV column
  `source_access`. Spot-check primary articles before relying on any number.
- Market slug, condition id, Gamma id, resolution and closedTime come from the Gamma API
  (2026-11-17). `match_confidence` is high when the report names the market, medium when inferred
  from the event and date, and low when picked among sibling markets.
- In Postgres, `pm_markets` was still being ingested. Only 3 cases (5 markets) are present yet, so
  `pg_market_id` is blank. Re-run `python -m pmflow.labeling.cases --write-matches
  artifacts/labeling/case_market_matches.csv`. The Gamma market_id equals `pm_markets.market_id`.
- `public_news_ts_utc` is the earliest credible public report, not the resolution time. Official
  confirmation is a separate column. Day precision means a local date. Upper bounds (publication
  times, closedTime) are in `public_news_ts_upper_bound_utc`. UTC conversions are explained in
  `ts_note`. No value was guessed.

## Summary
25 cases (23 Polymarket, 2 Kalshi); 19 in scope; grades A=11, B=10, C=4.

| case | type | market | in scope | news ts (UTC) | prec | grade | vindicated |
|---|---|---|---|---|---|---|---|
| PM-001 | suspected_informed | Maduro out by January 31, 2026? | yes | 2026-01-03T06:00Z | hour | A | yes |
| PM-002 | contrast_resolution_semantics | Will the U.S. invade Venezuela by Jan 31, 2026? | yes | 2026-01-03T06:00Z | hour | B | NO |
| PM-003 | suspected_informed | Nobel Peace Prize 2025: Machado | yes (borderline) | 2025-10-10T09:00Z | minute | A | yes |
| PM-004 | suspected_informed | Israeli strike on Iran, June 2025 (indictment) | yes | 2025-06-12T23:30Z | hour | A | yes |
| PM-005 | suspected_informed | Israel strike timing, June 2025 (Bubblemaps cluster) | yes | 2025-06-12T23:30Z | hour | B | yes |
| PM-006 | suspected_informed | US strike on Iranian nuclear sites, June 2025 | yes | <=2025-06-22T02:00Z | day/bound | B | yes |
| PM-007 | suspected_informed | US strikes Iran by Feb 28, 2026? (six fresh wallets) | yes | 2026-02-28T06:00Z | hour | A | yes |
| PM-008 | suspected_informed | same market, single experienced account | yes | 2026-02-28T06:00Z | hour | A | yes |
| PM-009 | suspected_informed | Khamenei out as Supreme Leader by March 31? | yes | 2026-02-28 | day | B | yes |
| PM-010 | suspected_informed | nine-account cluster across US-Iran war markets | yes | per event | - | A | yes |
| PM-011 | suspected_informed | US x Iran ceasefire by April 7? | yes | 2026-04-07T22:32Z | minute | A | yes |
| PM-012 | NEGATIVE fresh wallets wrong | US x Iran ceasefire by March 31? | yes | n/a | - | B | NO |
| PM-013 | NEGATIVE whale wrong | Feb-28 strike markets (large NO book) | yes | 2026-02-28T06:00Z | hour | C | NO |
| PM-014 | NEGATIVE whale wrong | Iran-war markets (~$1.5M wallet) | yes | unknown | - | C | NO |
| PM-015 | suspected_informed | Biden last-day pardons | yes | 2025-01-20 (<=17:00Z) | day | B | yes |
| PM-016 | suspected_informed | Will Joe pardon Hunter Biden? | yes | 2024-12-01 | day | B | yes |
| PM-017 | ambiguous | Will Trump pardon CZ in 2025? | yes | 2025-10-23 (<=17:22Z) | day | C | yes |
| PM-018 | suspected_informed (charged) | Google Year in Search 2025 | no | 2025-12-04 | day | A | yes |
| PM-019 | suspected_informed | OpenAI browser by October 31? | no | 2025-10-21 (<=19:34Z) | day | B | yes |
| PM-020 | suspected_informed (weak) | Taylor Swift & Kelce engaged in 2025? | no | 2025-08-26 (<=17:14Z) | day | B | yes |
| KX-001 | Kalshi discipline | MrBeast video markets | no | per event | - | A | yes |
| KX-002 | CFTC action | Kalshi presidential mention markets | no | per event | - | A | yes |
| PM-021 | suspected, pending | Israel 2026 election next PM / largest party | yes | 2026-10-27T20:00Z (assumed) | hour | B | pending |
| PM-022 | CONTRAST whale w/ research | 2024 US presidential winner (French whale) | no | 2024-11-06 | day | A | yes |
| PM-023 | CONTRAST reaction to public report | Trump nominates Warsh as Fed chair? | yes | 2026-01-29 | day | C | yes |

10 rows have minute or hour timestamps, covering 6 distinct events. The two high-confidence minute
timestamps are Nobel 09:00Z and the ceasefire post 22:32Z. 21 of 25 rows are Gamma-matched
(9 high, 9 medium, 3 low).

## Patterns
- **Core positive: fresh wallets, one-sided flow at long odds, hours before unscheduled government
  action.**
  - PM-001: account about one week old with a default username; 13 bets ($33k) between Dec 26 and
    Jan 2 at 5-12%; $409,881 profit; Venezuela markets only; DOJ/CFTC charges (allegations). The price
    drifted from 6% to 12.5% before the 06:00Z explosion reports and jumped after 06:15Z.
  - PM-007: six wallets funded less than 24h before the strike; $1.0-1.2M; full exit afterwards.
  - PM-011: wallets created the same day; first bets 1.5-26h before the post at 2.9-10.3%; one wallet
    created 12 minutes before the post; at least 50 new wallets.
  - PM-003: account opened that month; buying began about 11h before a scheduled announcement;
    3.7% rose to over 70%.
- **Experienced, not fresh:** PM-008 and PM-009 (Magamyman). The first trade came about 71 minutes
  before the news, at 17%, from an account with a history of strike-timing bets. So a fresh-wallet
  condition must not be mandatory.
- **Clusters (account-level):** PM-010 (9 accounts, 98% win rate over more than 80 bets) and
  PM-015/016 (2 accounts linked by a shared exchange deposit wallet; $316k).
- **Negatives and contrasts:**
  - PM-012 has the same surface pattern as PM-011 but resolved NO.
  - PM-013 and PM-014 are whales with no edge.
  - PM-002 had the right information but the wrong contract wording.
  - PM-022 is a vindicated whale with no MNPI.
  - PM-023 is a reaction to a press report; its resolution came 5 weeks later.
  - PM-021: profit was taken on the exit-poll move, but the final outcome is unresolved and currently
    leans the other way.
- **Out of scope:** PM-018, PM-019, PM-020, KX-001 and KX-002. Kalshi has no wallet data, so those two
  cannot be window-labelled.

## Aggregate catalogues (not rows)
- Mitts & Ofir, "From Iran to Taylor Swift" (Harvard Corp Gov forum, 2026-03-25). Polymarket
  Feb 2024 to Feb 2026: about 210k flagged wallet-market pairs, 69.9% win rate, about $143M
  anomalous profit. Not proof. The SSRN paper could not be reached.
- ACDC, "Insider Risks in Polymarket Political Markets" (April 2026, PDF on acdatacollective.org).
- ACDC "military Orcas" (Reuters/Al-Monitor, Aug 2026): 152 wallets, about $8M, 97.2% win rate.
  The best expansion source if the wallet list can be obtained.
- Polymarket referred more than 90 wallets (cryptometer); Kalshi made more than 50 referrals in
  Q2-Q3 2026.

## Searched, nothing found
Fed, BLS and CPI markets; Assad's fall (moved on public news); Russia-Ukraine ceasefire;
cabinet changes; tariffs. The Zelensky-suit and Ukraine-minerals disputes were oracle disputes, not
flow. The Kalshi James Bond market is unresolved.
