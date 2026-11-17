# Informed-flow signal for geopolitical prediction markets

Research prototype, public data only. We classify observable flow patterns; we never
identify, accuse or contact individuals. Wallet addresses are treated as opaque IDs and
are not linked to real-world identities. Sports markets are excluded. No live trading.

## Shared conventions (all agents)

- Python package: `src/pmflow/`. Each role owns one subpackage and must not edit the others:
  `data/` (ingestion, taxonomy, schema), `labeling/`, `features/`, `modeling/`, `evaluation/`.
- Virtualenv: `/home/user/prediction-market-analysis/.venv` (create with `python3 -m venv` if
  missing; `pip install -e .`). Add dependencies to `pyproject.toml` under your own
  optional-dependency group if another agent may be editing it at the same time.
- Postgres 16, local: DSN `postgresql://pmflow:pmflow@localhost:5432/pmflow`
  (env `PMFLOW_DSN` overrides). Schema is owned by `src/pmflow/data/schema.sql`; other roles
  create their own tables with their role prefix (`lbl_`, `feat_`, `mdl_`, `eval_`).
- Bulky raw data goes in `data/` (gitignored). Small, reviewable outputs (CSV/Parquet under
  ~20 MB, reports, figures) go in `artifacts/<role>/` and are committed.
- Findings and blockers: append to `docs/shared_log.md`. Flag blockers early; never work
  around them silently and never substitute mocked or synthetic data for real data.
- Reports go in `docs/`.
- Network: only gamma-api/data-api/clob.polymarket.com, api.elections.kalshi.com,
  api.goldsky.com, huggingface.co and package registries are reachable from shell. Other
  hosts fail with a proxy 403: log it as a blocker rather than retrying.
- Git: do NOT commit, push or switch branches. The orchestrator commits.
- Strict time discipline: no feature may use information after its window cutoff.
