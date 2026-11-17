#!/usr/bin/env bash
# Idempotently start the local Postgres 16 instance used by pmflow.
# Creates the cluster, role and database on first run; safe to re-run after a container restart.
set -euo pipefail
PGBIN=/usr/lib/postgresql/16/bin
PGDATA=${PGDATA:-/home/user/pgdata}
LOG=/home/user/prediction-market-analysis/data/logs/postgres.log
REPO=/home/user/prediction-market-analysis
mkdir -p "$(dirname "$LOG")"
as_pg() { if [ "$(id -un)" = postgres ]; then "$@"; else su postgres -s /bin/bash -c "$(printf '%q ' "$@")"; fi; }

if [ ! -s "$PGDATA/PG_VERSION" ]; then
  mkdir -p "$PGDATA"; chown postgres:postgres "$PGDATA"; chmod 700 "$PGDATA"
  as_pg "$PGBIN/initdb" -D "$PGDATA" -U postgres --auth-local=trust --auth-host=scram-sha-256 -E UTF8 >/dev/null
fi
if ! grep -q "pmflow tuning" "$PGDATA/postgresql.conf"; then
  cat >> "$PGDATA/postgresql.conf" <<CONF
# pmflow tuning
listen_addresses = 'localhost'
port = 5432
shared_buffers = 2GB
work_mem = 64MB
maintenance_work_mem = 512MB
max_wal_size = 1GB
min_wal_size = 256MB
synchronous_commit = off
CONF
fi
touch "$LOG"; chown postgres:postgres "$LOG"
if ! as_pg "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
  rm -f "$PGDATA/postmaster.pid" 2>/dev/null || true
  as_pg "$PGBIN/pg_ctl" -D "$PGDATA" -l "$LOG" -w start >/dev/null
fi
for i in $(seq 1 30); do "$PGBIN/pg_isready" -h localhost -p 5432 -q && break; sleep 1; done
as_pg "$PGBIN/psql" -h /var/run/postgresql -p 5432 -U postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='pmflow'" | grep -q 1 \
  || as_pg "$PGBIN/psql" -h /var/run/postgresql -U postgres -c "CREATE ROLE pmflow LOGIN PASSWORD 'pmflow' CREATEDB"
as_pg "$PGBIN/psql" -h /var/run/postgresql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='pmflow'" | grep -q 1 \
  || as_pg "$PGBIN/psql" -h /var/run/postgresql -U postgres -c "CREATE DATABASE pmflow OWNER pmflow"
# Apply schema (idempotent: CREATE ... IF NOT EXISTS)
if [ -f "$REPO/src/pmflow/data/schema.sql" ]; then
  PGPASSWORD=pmflow "$PGBIN/psql" -q -h localhost -U pmflow -d pmflow -v ON_ERROR_STOP=1 -f "$REPO/src/pmflow/data/schema.sql" >/dev/null
fi
echo "postgres up: postgresql://pmflow:pmflow@localhost:5432/pmflow"
