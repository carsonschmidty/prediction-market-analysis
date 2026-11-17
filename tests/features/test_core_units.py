import numpy as np
import pandas as pd

from pmflow.features.core import _burstiness, _robust_z, compute_features_batch, to_wide


def test_robust_z_handles_flat_baseline():
    z, med, n = _robust_z(100.0, np.zeros(10))
    assert n == 10 and med == 0 and z == 100.0          # $1 scale floor, not inf
    z, _, n = _robust_z(5.0, np.array([1.0, 2.0]))
    assert np.isnan(z) and n == 2                         # too few baseline windows


def test_burstiness_regular_vs_bursty():
    assert _burstiness(np.arange(0, 100, 10.0)) < -0.9   # perfectly regular -> -1
    assert _burstiness(np.array([0, 1, 2, 3, 1000, 1001, 1002.0])) > 0.3


def test_real_market_shapes_and_invariants(conn):
    df = compute_features_batch([(1198423, "2026-02-28T05:00:00Z"), (916440, "2026-01-02T00:00:00Z")],
                                conn=conn)
    assert len(df) == 8
    ok = df["gv"] > 0
    assert (df.loc[ok, "os"].between(0, 1)).all()
    assert (df.loc[ok, "p_entry"].between(0, 1)).all()
    assert (df["gv"].abs() + 1e-6 >= df["nsn"].abs()).all()
    for c in ("fws72", "top3_share", "conc_top3", "full_cov_share", "cluster_share"):
        v = df[c].dropna()
        assert ((v >= -1e-9) & (v <= 1 + 1e-9)).all(), c
    # gross volume grows with lookback
    for _, g in df.groupby("market_id"):
        assert g.sort_values("lookback_h")["gv"].is_monotonic_increasing
    w = to_wide(df)
    assert len(w) == 2 and "nsn_72h" in w and "px_last" in w


def test_yes_equivalent_direction(conn):
    """nsn must equal a direct SQL computation of YES-equivalent signed notional."""
    with conn.cursor() as cur:
        cur.execute("""select sum(case when (outcome_index = 0) = is_buy then 1 else -1 end * price * size)
                       from pm_trades where market_id = 916440 and role = 0
                       and ts >= '2026-01-02T00:00:00Z' and ts < '2026-01-03T00:00:00Z'""")
        ref = cur.fetchone()[0]
    df = compute_features_batch([(916440, "2026-01-03T00:00:00Z")], lookbacks=(24,), conn=conn)
    assert abs(df["nsn"].iloc[0] - ref) < 1e-6 * max(1, abs(ref))
