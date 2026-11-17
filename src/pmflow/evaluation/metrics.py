"""Probability-forecast metrics with weights and event-clustered bootstrap.

All functions take numpy arrays. ``w`` are non-negative weights (default 1). ``groups`` are cluster ids
(event ids) used for the bootstrap: events are resampled with replacement, never individual snapshots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

EPS = 1e-4


def _w(w, n):
    return np.ones(n) if w is None else np.asarray(w, dtype=float)


def clip(p, eps: float = EPS):
    return np.clip(np.asarray(p, dtype=float), eps, 1 - eps)


def brier_terms(y, p):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def logloss_terms(y, p, eps: float = EPS):
    p = clip(p, eps)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(y, p, w=None) -> float:
    w = _w(w, len(y))
    return float(np.sum(w * brier_terms(y, p)) / np.sum(w))


def log_loss(y, p, w=None, eps: float = EPS) -> float:
    w = _w(w, len(y))
    return float(np.sum(w * logloss_terms(y, p, eps)) / np.sum(w))


def reliability_table(y, p, w=None, bins=10) -> pd.DataFrame:
    """Equal-width bins on [0,1] (or explicit edges). Columns: bin, n, weight, mean_p, freq_y, gap."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = _w(w, len(y))
    edges = np.linspace(0, 1, bins + 1) if np.isscalar(bins) else np.asarray(bins, float)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        sw = w[m].sum()
        if m.sum() == 0:
            rows.append((b, edges[b], edges[b + 1], 0, 0.0, np.nan, np.nan, np.nan))
            continue
        mp = np.sum(w[m] * p[m]) / sw
        fy = np.sum(w[m] * y[m]) / sw
        rows.append((b, edges[b], edges[b + 1], int(m.sum()), float(sw), mp, fy, fy - mp))
    return pd.DataFrame(rows, columns=["bin", "lo", "hi", "n", "weight", "mean_p", "freq_y", "gap"])


def ece(y, p, w=None, bins=10) -> float:
    """Expected calibration error: weight-share-weighted |freq_y - mean_p| over equal-width bins."""
    t = reliability_table(y, p, w, bins)
    t = t[t.n > 0]
    return float(np.sum(t.weight * np.abs(t.gap)) / t.weight.sum())


def logit(p, eps: float = EPS):
    p = clip(p, eps)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def fit_platt(y, p, w=None, l2: float = 1e-6):
    """Fit P(y=1) = expit(a + b*logit(p)) by weighted max likelihood. Returns (a, b)."""
    x = logit(p)
    y = np.asarray(y, float)
    w = _w(w, len(y))
    w = w / w.mean()

    def nll(theta):
        z = theta[0] + theta[1] * x
        # log(1+exp(z)) stable
        ll = y * z - np.logaddexp(0, z)
        return -np.sum(w * ll) + l2 * np.sum(theta ** 2)

    def grad(theta):
        z = theta[0] + theta[1] * x
        r = w * (expit(z) - y)
        return np.array([r.sum(), (r * x).sum()]) + 2 * l2 * theta

    res = optimize.minimize(nll, np.array([0.0, 1.0]), jac=grad, method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def apply_platt(p, ab):
    return expit(ab[0] + ab[1] * logit(p))


def calibration_slope_intercept(y, p, w=None):
    """Calibration-in-the-large intercept and slope: logistic fit of y on logit(p). Perfect = (0, 1)."""
    return fit_platt(y, p, w, l2=0.0)


def summarize(y, p, w=None, bins=10) -> dict:
    a, b = calibration_slope_intercept(y, p, w)
    return dict(n=len(y), brier=brier(y, p, w), log_loss=log_loss(y, p, w), ece=ece(y, p, w, bins),
                calib_intercept=a, calib_slope=b, mean_p=float(np.average(p, weights=_w(w, len(y)))),
                base_rate=float(np.average(y, weights=_w(w, len(y)))))


# ---------------------------------------------------------------- weights


def snapshot_weights(df: pd.DataFrame, scheme: str, market_col="market_id", event_col="event_id"):
    """'snapshot' (1 each), 'market' (each market sums to 1), 'event' (each event sums to 1, split
    equally across its markets in df, then across each market's snapshots)."""
    if scheme == "snapshot":
        return np.ones(len(df))
    n_m = df.groupby(market_col)[market_col].transform("size").to_numpy(float)
    if scheme == "market":
        return 1.0 / n_m
    if scheme == "event":
        k = df.groupby(event_col)[market_col].transform("nunique").to_numpy(float)
        return 1.0 / (n_m * k)
    raise ValueError(scheme)


# ---------------------------------------------------------------- clustered bootstrap


def _cluster_sums(groups, *cols):
    g = pd.factorize(np.asarray(groups))[0]
    G = g.max() + 1
    return g, [np.bincount(g, weights=np.asarray(c, float), minlength=G) for c in cols]


def bootstrap_ratio(groups, num, den, B=2000, seed=0, ci=0.95):
    """Cluster bootstrap of sum(num)/sum(den), resampling clusters. Returns (point, lo, hi, draws)."""
    _, (sn, sd) = _cluster_sums(groups, num, den)
    G = len(sn)
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for i in range(0, B, 200):
        b = min(200, B - i)
        c = rng.multinomial(G, np.full(G, 1 / G), size=b).astype(float)
        draws[i:i + b] = (c @ sn) / (c @ sd)
    a = (1 - ci) / 2
    return float(sn.sum() / sd.sum()), float(np.quantile(draws, a)), float(np.quantile(draws, 1 - a)), draws


def clustered_mean_ci(values, groups, w=None, B=2000, seed=0, ci=0.95):
    w = _w(w, len(values))
    pt, lo, hi, _ = bootstrap_ratio(groups, w * np.asarray(values, float), w, B, seed, ci)
    return pt, lo, hi


def paired_test(loss_base, loss_model, groups, w=None, B=2000, seed=0):
    """Paired comparison of per-snapshot losses (positive delta = model better).

    Returns dict with mean delta, event-clustered bootstrap 95% CI, bootstrap two-sided p-value, and a
    cluster-robust t-test (event-level totals; CR1 with G-1 df)."""
    d = np.asarray(loss_base, float) - np.asarray(loss_model, float)
    w = _w(w, len(d))
    pt, lo, hi, draws = bootstrap_ratio(groups, w * d, w, B, seed)
    p_boot = float(min(1.0, 2 * min(np.mean(draws <= 0), np.mean(draws >= 0))))
    g, (sd, sw) = _cluster_sums(groups, w * d, w)
    G = len(sd)
    # cluster-robust SE of the ratio estimator
    W = sw.sum()
    resid = sd - pt * sw
    se = float(np.sqrt(G / (G - 1) * np.sum(resid ** 2)) / W) if G > 1 else np.nan
    t = pt / se if se and se > 0 else np.nan
    p_t = float(2 * stats.t.sf(abs(t), G - 1)) if np.isfinite(t) else np.nan
    return dict(delta=pt, ci_lo=lo, ci_hi=hi, p_boot=p_boot, se_cluster=se, t=t, p_cluster_t=p_t,
                n=len(d), n_clusters=G)


def min_detectable_effect(loss_base, loss_model_proxy, groups, w=None, alpha=0.05, power=0.8):
    """MDE for the mean paired loss delta, given the clustered SE of the delta.

    Without a real model we use a proxy delta series (e.g. raw price vs recalibrated price) to estimate
    the clustered SD of per-event delta totals; MDE = (z_{1-a/2} + z_power) * SE."""
    r = paired_test(loss_base, loss_model_proxy, groups, w, B=200)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * r["se_cluster"]
