"""Leakage test: features computed on the full DB must equal features computed on a copy of
the DB from which every row with ts >= cutoff has been removed (and resolutions of markets that
closed at/after the cutoff are unknown).

For each sampled (market, cutoff) we build session-temp tables:
  cut_trades : pm_trades rows with ts < cutoff for the market and for every wallet that traded
               it before the cutoff (all their corpus markets)
  cut_meta   : feat_market_meta with closed_time / resolution nulled where closed_time >= cutoff
  cut_hist   : feat_wallet_hist rebuilt from cut_trades + cut_meta with the production builder
  cut_pmw    : pm_wallets rows with first_trade_ts < cutoff
  cut_wms    : pm_wallet_market_summary rows with first_ts < cutoff
and compare every output column.
"""
import numpy as np
import pandas as pd
import pytest

from pmflow.features.base import build_hist
from pmflow.features.core import compute_window_features
from pmflow.features.db import Sources

# (market_id, cutoff): reported-case markets at their news lower edge plus non-case markets
# at arbitrary times (incl. a non-midnight cutoff, an early 2024 cutoff and a macro market).
SAMPLE = [
    (916440, "2026-01-03T05:00:00Z"),     # PM-001 Maduro out
    (1198423, "2026-02-28T05:00:00Z"),    # PM-007 US strikes Iran by Feb 28
    (1706788, "2026-04-07T13:17:41Z"),    # PM-011 ceasefire, arbitrary second
    (532742, "2025-06-12T22:30:00Z"),     # PM-004 Israel strike
]


def _random_sample(conn, n=4, seed=7):
    with conn.cursor() as cur:
        cur.execute(
            """select m.market_id, s.first_trade_ts + (s.last_trade_ts - s.first_trade_ts) * 0.6
               from feat_market_meta m join pm_market_trade_stats s using (market_id)
               where s.n_taker between 2000 and 60000
               order by md5(m.market_id::text || %s) limit %s""", (str(seed), n))
        return [(int(a), pd.Timestamp(b).floor("s").isoformat()) for a, b in cur.fetchall()]


def _cut_sources(conn, market_id, cutoff):
    c = pd.Timestamp(cutoff)
    with conn.cursor() as cur:
        for t in ("cut_trades", "cut_meta", "cut_pmw", "cut_wms", "cut_hist", "cut_w"):
            cur.execute(f"drop table if exists {t}")
        cur.execute("create temp table cut_w as select distinct wallet from pm_trades"
                    " where market_id = %s and ts < %s", (market_id, c))
        cur.execute("create temp table cut_trades as select t.* from pm_trades t"
                    " where t.ts < %s and (t.market_id = %s or t.wallet in (select wallet from cut_w))",
                    (c, market_id))
        cur.execute("create index on cut_trades (market_id, ts)")
        cur.execute("create temp table cut_meta as select * from feat_market_meta")
        cur.execute("update cut_meta set closed_time = null, resolution_status = null,"
                    " winning_outcome_index = null, resolved_yes = null"
                    " where closed_time >= %s or closed_time is null", (c,))
        cur.execute("create temp table cut_pmw as select * from pm_wallets where first_trade_ts < %s", (c,))
        cur.execute("create temp table cut_wms as select * from pm_wallet_market_summary"
                    " where first_ts < %s and wallet in (select wallet from cut_w)", (c,))
    build_hist(conn, target="cut_hist", trades="cut_trades", meta="cut_meta", temp=True)
    return Sources(trades="cut_trades", meta="cut_meta", hist="cut_hist", pmw="cut_pmw", wms="cut_wms")


def _compare(a: pd.DataFrame, b: pd.DataFrame):
    bad = []
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            if not np.allclose(xv, yv, rtol=1e-5, atol=1e-6, equal_nan=True):
                bad.append(col)
        elif not x.astype(str).equals(y.astype(str)):
            bad.append(col)
    return bad


def test_features_equal_on_truncated_db(conn):
    sample = SAMPLE + _random_sample(conn)
    for mid, cut in sample:
        full = compute_window_features(mid, cut, conn=conn)
        src = _cut_sources(conn, mid, cut)
        trunc = compute_window_features(mid, cut, conn=conn, src=src)
        assert len(full) == 4 and full["n_trades_total"].iloc[0] > 0, (mid, cut)
        bad = _compare(full, trunc)
        assert not bad, f"{mid} @ {cut}: columns differ {bad}"
    conn.rollback()


def test_future_trades_do_not_change_features(conn):
    """Adding a later cutoff to the same batch must not change the earlier cutoff's row."""
    from pmflow.features.core import compute_features_batch
    a = compute_features_batch([(916440, "2026-01-01T00:00:00Z")], conn=conn)
    b = compute_features_batch([(916440, "2026-01-01T00:00:00Z"), (916440, "2026-01-03T12:00:00Z")], conn=conn)
    b = b[b["cutoff_ts"] == pd.Timestamp("2026-01-01T00:00:00Z")].reset_index(drop=True)
    assert not _compare(a, b)
