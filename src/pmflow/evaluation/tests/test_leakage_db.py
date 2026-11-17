"""DB-backed tests (skipped when Postgres / eval tables are unavailable)."""
import numpy as np
import pandas as pd
import pytest

try:
    from pmflow.evaluation.harness import compare, connect, load_snapshots
    from pmflow.evaluation import leakage as L
    with connect() as _c:
        _c.execute("SELECT 1 FROM eval_snapshots LIMIT 1")
    HAVE_DB = True
except Exception:  # noqa: BLE001
    HAVE_DB = False

pytestmark = pytest.mark.skipif(not HAVE_DB, reason="needs pmflow Postgres with eval_snapshots")

LEAKY_SQL = """SELECT count(*) AS n FROM pm_trades WHERE market_id = %(market_id)s
               AND ts < %(cutoff_ts)s::timestamptz + interval '1 day'"""
CLEAN_SQL = """SELECT count(*) AS n, max(price) AS mx FROM pm_trades_v WHERE market_id = %(market_id)s
               AND ts < %(cutoff_ts)s"""


def test_truncation_audit_flags_leaky_and_passes_clean():
    s = L.sample_snapshots(8, seed=5, where="is_live AND hours_to_close_diag > 48 AND n_trades_24h > 0")
    assert L.audit_sql_feature(CLEAN_SQL, s).ok.all()
    leaky = L.audit_sql_feature(LEAKY_SQL, s)
    assert (~leaky.ok).sum() >= 1


def test_python_audit():
    s = L.sample_snapshots(3, seed=6, where="is_live AND n_trades_24h > 0")
    good = lambda tr, cut: {"n": int((tr.ts < cut).sum())}
    bad = lambda tr, cut: {"n": len(tr)}
    assert L.audit_python_feature(good, s).ok.all()
    assert not L.audit_python_feature(bad, s).ok.all()


def test_source_timestamp_check():
    s = L.sample_snapshots(5, seed=7)
    ok = s.assign(max_source_ts=pd.to_datetime(s.cutoff_ts, utc=True) - pd.Timedelta("1s"))
    assert L.check_source_timestamps(ok).empty
    bad = s.assign(max_source_ts=pd.to_datetime(s.cutoff_ts, utc=True))
    assert len(L.check_source_timestamps(bad)) == len(s)


def test_snapshot_invariants():
    assert all(v == 0 for v in L.check_snapshot_selection().values())


def test_harness_price_vs_itself_is_zero():
    snaps = load_snapshots("is_live AND split = 'test' AND grid_24h")
    pred = snaps[["snapshot_id"]].assign(p_model=snaps.price_yes)
    rep = compare(pred, snaps, B=50, splits=["test"])
    assert np.allclose(rep["summary"].d_brier, 0)
