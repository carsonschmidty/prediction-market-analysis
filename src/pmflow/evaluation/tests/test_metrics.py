"""Unit tests for evaluation metrics. Run: .venv/bin/python -m pytest src/pmflow/evaluation/tests -q"""
import numpy as np
import pandas as pd
import pytest

from pmflow.evaluation import metrics as M


def test_brier_logloss_known_values():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    assert M.brier(y, p) == pytest.approx((0.04 + 0.04 + 0.16 + 0.16) / 4)
    assert M.log_loss(y, p) == pytest.approx(-(2 * np.log(0.8) + 2 * np.log(0.6)) / 4)
    # perfect forecast
    assert M.brier(y, y.astype(float)) == 0
    assert M.log_loss(y, y.astype(float)) == pytest.approx(-np.log(1 - M.EPS))


def test_weights():
    y = np.array([1, 0])
    p = np.array([0.5, 0.0])
    assert M.brier(y, p, w=[1, 0]) == pytest.approx(0.25)
    assert M.brier(y, p, w=[1, 3]) == pytest.approx(0.25 / 4)


def test_reliability_and_ece_perfectly_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, len(p)) < p).astype(int)
    assert M.ece(y, p) < 0.01
    t = M.reliability_table(y, p)
    assert t.n.sum() == len(p)
    assert np.all(np.abs(t.gap) < 0.02)
    a, b = M.calibration_slope_intercept(y, p)
    assert abs(a) < 0.05 and abs(b - 1) < 0.05


def test_ece_miscalibrated():
    p = np.full(1000, 0.7)
    y = np.r_[np.ones(500), np.zeros(500)]
    assert M.ece(y, p) == pytest.approx(0.2)


def test_platt_recovers_distortion():
    rng = np.random.default_rng(2)
    true = rng.uniform(0.02, 0.98, 100_000)
    y = (rng.uniform(size=len(true)) < true).astype(int)
    # favourite-longshot style distortion: observed prices less extreme than truth
    z = M.logit(true) * 0.7 + 0.1
    price = M.expit(z)
    a, b = M.fit_platt(y, price)
    assert b == pytest.approx(1 / 0.7, rel=0.05)
    assert a == pytest.approx(-0.1 / 0.7, abs=0.05)
    assert M.brier(y, M.apply_platt(price, (a, b))) < M.brier(y, price)


def test_weight_schemes():
    df = pd.DataFrame(dict(market_id=[1, 1, 1, 2, 3], event_id=[10, 10, 10, 10, 20]))
    np.testing.assert_allclose(M.snapshot_weights(df, "market"), [1 / 3, 1 / 3, 1 / 3, 1, 1])
    w = M.snapshot_weights(df, "event")
    # event 10 has 2 markets -> each market weight 1/2; event 20 total 1
    assert w[:4].sum() == pytest.approx(1) and w[4] == pytest.approx(1)


def test_bootstrap_ratio_clusters():
    # one big cluster dominates -> CI must be wide; point estimate equals plain mean
    vals = np.r_[np.ones(1000), np.zeros(10)]
    groups = np.r_[np.zeros(1000), np.arange(1, 11)]
    pt, lo, hi, _ = M.bootstrap_ratio(groups, vals, np.ones_like(vals), B=500)
    assert pt == pytest.approx(1000 / 1010)
    assert lo < 0.5  # resampling the 11 clusters often drops the big one


def test_paired_test_detects_and_nulls():
    rng = np.random.default_rng(3)
    G, per = 400, 5
    g = np.repeat(np.arange(G), per)
    base = rng.uniform(0, 0.5, G * per)
    # identical model -> delta 0, p ~ 1
    r0 = M.paired_test(base, base, g, B=200)
    assert r0["delta"] == 0
    # model better by 0.02 with event-level noise
    noise = np.repeat(rng.normal(0, 0.05, G), per)
    r1 = M.paired_test(base, base - 0.02 + noise, g, B=500)
    assert r1["delta"] == pytest.approx(0.02, abs=0.01)
    assert r1["p_cluster_t"] < 0.01 and r1["ci_lo"] > 0


def test_cluster_se_larger_than_naive_when_correlated():
    rng = np.random.default_rng(4)
    G, per = 200, 50
    g = np.repeat(np.arange(G), per)
    d = np.repeat(rng.normal(0, 1, G), per) + rng.normal(0, 0.1, G * per)
    r = M.paired_test(d, np.zeros_like(d), g, B=100)
    naive = d.std() / np.sqrt(len(d))
    assert r["se_cluster"] > 3 * naive
