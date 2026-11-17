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
