"""Leakage-audit utilities.

1. ``check_source_timestamps``: a feature table that records, per snapshot, the max timestamp of any source
   row it used (column ``max_source_ts``) must have max_source_ts < cutoff_ts for every row.
2. ``audit_sql_feature``: re-computes a SQL feature for a sample of snapshots with all trades at or after the
   cutoff *physically removed*: within one session a TEMP table ``pm_trades`` (and a temp ``pm_trades_v``)
   containing only rows with ts < cutoff shadows the real table (pg_temp is searched first for relations).
   The feature must be identical. The SQL must reference ``pm_trades`` / ``pm_trades_v`` unqualified and take
   parameters ``%(market_id)s`` and ``%(cutoff_ts)s``; it must return exactly one row.
   Scope of the truncated copy: the snapshot's own market (default) or all markets within ``lookback`` of the
   cutoff (``all_markets=True``; for cross-market wallet features). Rows outside that scope are absent in the
   truncated run, so a feature that legitimately reaches further back will be flagged (conservative).
3. ``audit_python_feature``: same idea for a Python function ``f(trades_df, cutoff_ts) -> dict``: compares
   f(full history) to f(history truncated at cutoff).
4. ``check_snapshot_selection``: asserts eval_snapshots invariants (cutoff on grid, cutoff < closed_time,
   last trade < cutoff, split rules).

CLI: ``python -m pmflow.evaluation.leakage --selfcheck`` runs (4) and audits the snapshot price features.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from .harness import connect


def check_source_timestamps(df: pd.DataFrame, ts_col: str = "max_source_ts", snaps: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return violating rows (empty DataFrame = pass). df needs snapshot_id and ts_col (and cutoff_ts, or snaps)."""
    if "cutoff_ts" not in df:
        if snaps is None:
            with connect() as c:
                cur = c.execute("SELECT snapshot_id, cutoff_ts FROM eval_snapshots WHERE snapshot_id = ANY(%s)",
                                [list(map(int, df.snapshot_id))])
                snaps = pd.DataFrame(cur.fetchall(), columns=["snapshot_id", "cutoff_ts"])
        df = df.merge(snaps[["snapshot_id", "cutoff_ts"]], on="snapshot_id")
    ts = pd.to_datetime(df[ts_col], utc=True)
    return df[ts >= pd.to_datetime(df.cutoff_ts, utc=True)]


def _equal(a, b, tol=1e-9):
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return (a is None or (isinstance(a, float) and math.isnan(a))) == (b is None or (isinstance(b, float) and math.isnan(b)))
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)))
    except (TypeError, ValueError):
        return a == b


def sample_snapshots(n: int = 50, seed: int = 0, where: str = "is_live") -> pd.DataFrame:
    with connect() as c:
        c.execute("SELECT setseed(%s)", [((seed % 1000) / 1000.0)])
        cur = c.execute(f"SELECT snapshot_id, market_id, cutoff_ts FROM eval_snapshots WHERE {where} "
                        f"ORDER BY random() LIMIT %s", [n])
        return pd.DataFrame(cur.fetchall(), columns=["snapshot_id", "market_id", "cutoff_ts"])


def audit_sql_feature(sql: str, sample: pd.DataFrame | None = None, all_markets: bool = False,
                      lookback: str = "30 days") -> pd.DataFrame:
    """Returns per-snapshot comparison with column ``ok``. Every row must be ok."""
    sample = sample if sample is not None else sample_snapshots()
    rows = []
    for r in sample.itertuples(index=False):
        prm = dict(market_id=int(r.market_id), cutoff_ts=r.cutoff_ts)
        with connect() as c:
            cur = c.execute(sql, prm)
            cols = [d.name for d in cur.description]
            full = cur.fetchone()
            c.execute("BEGIN")
            if all_markets:
                c.execute("CREATE TEMP TABLE pm_trades ON COMMIT DROP AS SELECT * FROM public.pm_trades "
                          "WHERE ts < %(cutoff_ts)s AND ts >= %(cutoff_ts)s - %(lb)s::interval", dict(prm, lb=lookback))
            else:
                c.execute("CREATE TEMP TABLE pm_trades ON COMMIT DROP AS SELECT * FROM public.pm_trades "
                          "WHERE market_id = %(market_id)s AND ts < %(cutoff_ts)s", prm)
            vdef = c.execute("SELECT pg_get_viewdef('public.pm_trades_v'::regclass)").fetchone()[0]
            vdef = vdef.replace("FROM pm_trades t", "FROM pg_temp.pm_trades t")
            c.execute(f"CREATE TEMP VIEW pm_trades_v AS {vdef}")
            trunc = c.execute(sql, prm).fetchone()
            c.execute("ROLLBACK")
        diffs = [k for k, a, b in zip(cols, full, trunc) if not _equal(a, b)]
        rows.append(dict(snapshot_id=r.snapshot_id, market_id=r.market_id, cutoff_ts=r.cutoff_ts,
                         ok=not diffs, differing=",".join(diffs)))
    return pd.DataFrame(rows)


def audit_python_feature(f, sample: pd.DataFrame | None = None) -> pd.DataFrame:
    """f(trades_df, cutoff_ts) -> dict. trades_df has pm_trades columns for the snapshot's market."""
    sample = sample if sample is not None else sample_snapshots()
    rows = []
    with connect() as c:
        for r in sample.itertuples(index=False):
            cur = c.execute("SELECT market_id, outcome_index, is_buy, price, size, encode(wallet,'hex') wallet, ts, role "
                            "FROM pm_trades WHERE market_id = %s", [int(r.market_id)])
            tr = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
            tr["ts"] = pd.to_datetime(tr.ts, utc=True)
            cut = pd.Timestamp(r.cutoff_ts)
            a = f(tr, cut)
            b = f(tr[tr.ts < cut].copy(), cut)
            diffs = [k for k in set(a) | set(b) if not _equal(a.get(k), b.get(k))]
            rows.append(dict(snapshot_id=r.snapshot_id, ok=not diffs, differing=",".join(sorted(diffs))))
    return pd.DataFrame(rows)


SNAPSHOT_INVARIANTS = {
    "cutoff_not_on_6h_grid": "SELECT count(*) FROM eval_snapshots WHERE extract(epoch FROM cutoff_ts)::bigint % 21600 <> 0",
    "cutoff_not_before_close": "SELECT count(*) FROM eval_snapshots s JOIN eval_markets m USING (market_id) WHERE s.cutoff_ts >= m.closed_time",
    "last_trade_not_before_cutoff": "SELECT count(*) FROM eval_snapshots WHERE last_trade_age_h <= 0",
    "train_event_closes_after_train_end": "SELECT count(*) FROM eval_snapshots s JOIN eval_event_splits e USING (event_id) WHERE s.split='train' AND e.event_close >= '2026-04-01'",
    "val_cutoff_outside_period": "SELECT count(*) FROM eval_snapshots WHERE split='val' AND (cutoff_ts < '2026-04-01' OR cutoff_ts >= '2026-07-01')",
    "test_cutoff_before_test_start": "SELECT count(*) FROM eval_snapshots WHERE split='test' AND cutoff_ts < '2026-07-01'",
    "event_in_two_scored_splits": "SELECT count(*) FROM (SELECT event_id FROM eval_snapshots WHERE split IN ('train','val','test') GROUP BY 1 HAVING count(DISTINCT split) > 1) x",
    "live_price_outside_band": "SELECT count(*) FROM eval_snapshots WHERE is_live AND (price_yes < 0.03 OR price_yes > 0.97)",
}

PRICE_FEATURE_SQL = """
SELECT (SELECT CASE WHEN outcome_index = 0 THEN price ELSE 1 - price END FROM pm_trades
         WHERE market_id = %(market_id)s AND role = 0 AND ts < %(cutoff_ts)s ORDER BY ts DESC, key_hash LIMIT 1) AS last_p,
       (SELECT count(*) FROM pm_trades WHERE market_id = %(market_id)s AND role = 0
         AND ts >= %(cutoff_ts)s::timestamptz - interval '7 days' AND ts < %(cutoff_ts)s) AS n7d
"""


def check_snapshot_selection() -> dict:
    with connect() as c:
        return {k: c.execute(q).fetchone()[0] for k, q in SNAPSHOT_INVARIANTS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    inv = check_snapshot_selection()
    print("snapshot invariants (all must be 0):", inv)
    sample = sample_snapshots(a.n)
    res = audit_sql_feature(PRICE_FEATURE_SQL, sample)
    print(f"truncation audit of price features: {res.ok.sum()}/{len(res)} identical")
    # cross-check stored eval_snapshots values against direct recomputation
    with connect() as c:
        bad = 0
        for r in sample.itertuples(index=False):
            lp, n7 = c.execute(PRICE_FEATURE_SQL, dict(market_id=int(r.market_id), cutoff_ts=r.cutoff_ts)).fetchone()
            sp, sn = c.execute("SELECT price_yes, n_trades_7d FROM eval_snapshots WHERE snapshot_id=%s", [int(r.snapshot_id)]).fetchone()
            # ties at the same ts can order differently: compare n7d exactly, price loosely
            bad += int(n7 != sn or abs(float(lp) - float(sp)) > 0.05)
        print(f"stored vs recomputed mismatches: {bad}/{len(sample)}")
    if any(v for v in inv.values()) or not res.ok.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
