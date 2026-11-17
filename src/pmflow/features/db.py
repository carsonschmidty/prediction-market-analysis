"""DB helpers and table-name configuration for the features role."""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_DSN = "postgresql://pmflow:pmflow@localhost:5432/pmflow"


def dsn() -> str:
    return os.environ.get("PMFLOW_DSN", DEFAULT_DSN)


def connect(**kw):
    import psycopg

    conn = psycopg.connect(dsn(), **kw)
    with conn.cursor() as cur:
        cur.execute("set work_mem = '128MB'")
    return conn


@dataclass(frozen=True)
class Sources:
    """Names of the relations the feature code reads.

    The leakage test swaps these for temp tables that contain only rows with ts < cutoff,
    so every query in the feature code must go through these names."""

    trades: str = "pm_trades"                 # compact taker/maker rows (role 0 = taker)
    meta: str = "feat_market_meta"            # corpus markets (in-scope + macro) metadata
    hist: str = "feat_wallet_hist"            # per-wallet in-corpus history arrays (base.py)
    pmw: str = "pm_wallets"                   # full Polymarket history (partial coverage)
    wms: str = "pm_wallet_market_summary"     # full history per wallet-market (partial)


FULL = Sources()
