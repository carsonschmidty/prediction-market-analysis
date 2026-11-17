#!/usr/bin/env bash
# Full resumable Polymarket pipeline. Safe to re-run at any time (all stages are checkpointed).
#   nohup scripts/run_polymarket_pipeline.sh > data/logs/pipeline.nohup 2>&1 &
# Progress:  .venv/bin/python -m pmflow.data.ingest_polymarket --status ; tail -f data/logs/pipeline.log
set -uo pipefail
cd /home/user/prediction-market-analysis
PY=.venv/bin/python
LOG=data/logs/pipeline.log
mkdir -p data/logs
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
scripts/start_db.sh >> "$LOG" 2>&1

# 1. metadata for all closed markets (taxonomy is applied inline); trades stage re-polled while it runs
if ! pgrep -f "pmflow.data.ingest_gamma" >/dev/null; then
  say "starting gamma metadata ingest"
  $PY -m pmflow.data.ingest_gamma --resume --workers 10 >> data/logs/ingest_gamma.nohup 2>&1 &
fi
while pgrep -f "pmflow.data.ingest_gamma" >/dev/null; do
  say "gamma running; trades pass on markets classified so far"
  $PY -m pmflow.data.ingest_polymarket --stage trades --workers 6 >> data/logs/trades.nohup 2>&1
  sleep 120
done
say "gamma finished; re-applying taxonomy"
$PY -m pmflow.data.taxonomy --apply >> "$LOG" 2>&1

# 2. taker trades for in-scope (then macro comparison) markets, by volume desc
say "trades stage"; $PY -m pmflow.data.ingest_polymarket --stage trades --workers 6 >> data/logs/trades.nohup 2>&1
# 3. wallet histories for wallets with a >= $1k taker trade in an in-scope market
say "wallets stage"; $PY -m pmflow.data.ingest_polymarket --stage wallets --workers 6 --min-wallet-usd 1000 >> data/logs/wallets.nohup 2>&1
# 4. maker attribution (all fills) for in-scope markets
say "fills stage"; PMFLOW_MIN_FREE_GB=8 $PY -m pmflow.data.ingest_polymarket --stage fills --workers 6 >> data/logs/fills.nohup 2>&1
# 5. CLOB price history (daily + hourly)
say "prices stage"; $PY -m pmflow.data.ingest_polymarket --stage prices --workers 4 >> data/logs/prices.nohup 2>&1
# 6. derived tables + exports
say "derive + export"; $PY -m pmflow.data.derive >> "$LOG" 2>&1; $PY -m pmflow.data.export >> "$LOG" 2>&1
say "pipeline finished"
