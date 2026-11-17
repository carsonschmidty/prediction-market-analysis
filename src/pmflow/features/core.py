"""Point-in-time window features for informed-flow research (features role).

Public API
----------
compute_window_features(market_id, cutoff_ts, lookbacks=(1, 6, 24, 72)) -> DataFrame
compute_features_batch(requests, lookbacks=(1, 6, 24, 72)) -> DataFrame
    ``requests``: iterable of (market_id, cutoff_ts) or a DataFrame with those columns
    (plus optional ``snapshot_id``). Returns one row per (market, cutoff, lookback) with the
    window features, and the cutoff-level features repeated on every row.
to_wide(long_df) -> one row per (market, cutoff) with ``<feature>_<L>h`` columns.

As-of rule (enforced by construction; see tests/features/test_leakage.py)
------------------------------------------------------------------------
Every number for cutoff c is a function of rows with ts < c only:
* trades of the market: ``ts < c`` (windows are [c - L, c)).
* wallet attributes: first in-corpus trade time (<= the trade time by definition), corpus
  markets whose first trade by the wallet is < c, cumulative notional through UTC days that
  ended <= c, realised PnL on markets with closed_time < c computed from trades before that
  closed_time, full-history rows (pm_wallets / pm_wallet_market_summary) only for wallets whose
  first in-scope >= $1k trade is < c and only market rows with first_ts < c.
* market metadata: created_at, scheduled end_date, category/geo_subtype (known ex ante).
  closed_time / resolution are never used for the market itself.

YES-equivalent convention: outcome index 0 is "YES" (for the ~0.3% of corpus markets that are
not Yes/No, the first listed outcome). A trade's YES direction d = +1 for BUY outcome 0 or SELL
outcome 1, else -1. Signed notional = d * price * size (USD actually exchanged). YES-equivalent
price = price for outcome 0, 1 - price for outcome 1.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .db import FULL, Sources, connect

LOOKBACKS_H = (1, 6, 24, 72)
H = 3600.0
FRESH_S = 72 * H                 # labeling_plan §1: first-ever trade < 72h before the trade
FRESH24_S = 24 * H
CLUSTER_GAP_S = 30 * 60          # co-timed fresh wallets: first-ever trades <= 30 min apart
CLUSTER_MIN = 3
K_BASE = 20                      # baseline windows for robust z
BASE_GAP_S = 7 * 86400           # weak-label baseline must end >= 7 days before cutoff
MIN_BASE = 3
CENSOR_BEFORE = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
RETENTION_START = datetime(2023, 11, 18, tzinfo=timezone.utc).timestamp()
HIST_MKTS = 20                   # whale filter: "high-history" = >= 20 prior corpus markets
MIN_RES = 5                      # ... with >= 5 resolved prior markets
MIXED = (0.35, 0.65)             # ... and win rate in this band = "mixed track record"
CONC_HI = 0.8
URG_N = 20                       # reference price = median YES-eq price of previous 20 trades

CUTOFF_FEATURES = [
    "hours_since_open", "hours_since_first_trade", "hours_to_end", "hour_utc", "dow_utc",
    "wallet_age_censored", "market_history_truncated", "px_last", "log_gv_total", "log_gv_7d",
    "n_trades_total", "n_wallets_total", "category", "geo_subtype", "in_scope",
]


def _ts(x) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.timestamp()


def _robust_z(x: float, base: np.ndarray) -> tuple[float, float, int]:
    """(z, median, n). scale = max(1.4826*MAD, 1.2533*mean|x-med|, $1) so that windows of a
    quiet market (MAD = 0) do not give infinite z."""
    n = len(base)
    if n < MIN_BASE:
        return np.nan, np.nan, n
    med = float(np.median(base))
    dev = np.abs(base - med)
    scale = max(1.4826 * float(np.median(dev)), 1.2533 * float(dev.mean()), 1.0)
    return (x - med) / scale, med, n


def _burstiness(tw: np.ndarray) -> float:
    if len(tw) < 3:
        return np.nan
    g = np.diff(tw)
    m, s = g.mean(), g.std()
    return (s - m) / (s + m) if (s + m) > 0 else np.nan


def _wq(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(values) == 0 or weights.sum() <= 0:
        return np.nan
    o = np.argsort(values)
    cw = np.cumsum(weights[o])
    return float(values[o][np.searchsorted(cw, q * cw[-1])])


class _Wallets:
    """Per-wallet history arrays fetched for a set of wallets, queried as of a cutoff."""

    def __init__(self, conn, wallets: list[bytes], src: Sources):
        self.h: dict[bytes, tuple] = {}
        self.full: dict[bytes, tuple] = {}
        if not wallets:
            return
        with conn.cursor() as cur:
            for i in range(0, len(wallets), 20000):
                chunk = wallets[i:i + 20000]
                cur.execute(
                    f"select wallet, first_t, first_big_t, mkt_ids, mkt_first_t, res_t, res_pnl,"
                    f" day_end_t, cum_usd from {src.hist} where wallet = any(%s)", (chunk,))
                for w, ft, fb, mi, mf, rt, rp, de, cu in cur.fetchall():
                    rt = np.asarray(rt or [], float)
                    rp = np.asarray(rp or [], float)
                    self.h[bytes(w)] = (
                        ft, fb, np.asarray(mi, np.int64), np.asarray(mf, float), rt,
                        np.concatenate([[0.0], np.cumsum(rp)]),
                        np.concatenate([[0], np.cumsum(rp > 0)]),
                        np.asarray(de, float), np.concatenate([[0.0], np.asarray(cu, float)]))
                cur.execute(
                    f"select wallet, extract(epoch from first_trade_ts)::float8 from {src.pmw}"
                    f" where wallet = any(%s) and activity_complete and first_trade_ts is not null",
                    (chunk,))
                pf = {bytes(w): t for w, t in cur.fetchall()}
                if pf:
                    cur.execute(
                        f"select wallet, market_id, extract(epoch from first_ts)::float8 from {src.wms}"
                        f" where wallet = any(%s) and first_ts is not null", (list(pf),))
                    rows: dict[bytes, list] = {}
                    for w, m, t in cur.fetchall():
                        rows.setdefault(bytes(w), []).append((m if m is not None else -1, t))
                    for w, t in pf.items():
                        r = rows.get(w, [])
                        self.full[w] = (t, np.array([x[0] for x in r], np.int64),
                                        np.array([x[1] for x in r], float))

    def first_t(self, w: bytes) -> float:
        r = self.h.get(w)
        return r[0] if r is not None else np.nan

    def asof_many(self, w: bytes, cs: np.ndarray, market_id: int) -> np.ndarray:
        """Matrix (len(cs) x 8) of wallet w's attributes as of each cutoff in cs:
        prior_mkts (excl. market_id), res_n, res_wins, res_pnl, cum_usd_prior_days, full_ok,
        full_first_t, full_prior_mkts. Only history strictly before each cutoff is used."""
        out = np.full((len(cs), 8), np.nan)
        out[:, 5] = 0.0
        r = self.h.get(w)
        if r is None:
            return out
        ft, fb, mi, mf, rt, rpc, rwc, de, cu = r
        k = np.searchsorted(mf, cs, "left")
        here = np.flatnonzero(mi == market_id)
        own = (mf[here[0]] < cs).astype(float) if len(here) else 0.0
        out[:, 0] = k - own
        kr = np.searchsorted(rt, cs, "left")
        out[:, 1] = kr
        out[:, 2] = rwc[kr]
        out[:, 3] = rpc[kr]
        out[:, 4] = cu[np.searchsorted(de, cs, "right")]
        f = self.full.get(w)
        if f is not None and fb is not None:
            ok = (fb < cs) & (f[0] < cs)
            if ok.any():
                fts = np.sort(f[2])
                fo = f[2][f[1] == market_id]
                fpm = np.searchsorted(fts, cs, "left") - (fo.min() < cs if len(fo) else 0)
                out[ok, 5] = 1.0
                out[ok, 6] = f[0]
                out[ok, 7] = fpm[ok]
        return out


def _load_market(conn, market_id: int, max_cutoff: float, src: Sources):
    with conn.cursor() as cur:
        cur.execute(
            f"select extract(epoch from ts)::float8, outcome_index, is_buy, price, size, wallet"
            f" from {src.trades} where market_id = %s and role = 0 and ts < to_timestamp(%s)"
            f" order by ts, key_hash", (market_id, max_cutoff))
        rows = cur.fetchall()
        cur.execute(
            f"select extract(epoch from created_at)::float8, extract(epoch from end_date)::float8,"
            f" category, geo_subtype, in_scope from {src.meta} where market_id = %s", (market_id,))
        meta = cur.fetchone()
    return rows, meta


def _market_features(conn, market_id: int, cutoffs: list[float], lookbacks, src: Sources) -> list[dict]:
    lbs = sorted(set(lookbacks))
    maxl = max(lbs) * H
    rows, meta = _load_market(conn, market_id, max(cutoffs), src)
    created, end_t, category, geo_subtype, in_scope = meta if meta else (np.nan, np.nan, None, None, None)
    created = np.nan if created is None else created
    end_t = np.nan if end_t is None else end_t
    n = len(rows)
    if n:
        t = np.fromiter((r[0] for r in rows), float, n)
        oi = np.fromiter((r[1] for r in rows), np.int8, n)
        buy = np.fromiter((r[2] for r in rows), bool, n)
        px = np.fromiter((r[3] for r in rows), float, n)
        sz = np.fromiter((r[4] for r in rows), float, n)
        wcode, wuniq = pd.factorize(pd.Series([bytes(r[5]) for r in rows], dtype=object))
        wuniq = list(wuniq)
    else:
        t = np.zeros(0); oi = np.zeros(0, np.int8); buy = np.zeros(0, bool)
        px = np.zeros(0); sz = np.zeros(0); wcode = np.zeros(0, np.int64); wuniq = []
    del rows
    d = np.where((oi == 0) == buy, 1.0, -1.0)
    usd = px * sz
    snot = d * usd
    ypx = np.where(oi == 0, px, 1.0 - px)
    cgv = np.concatenate([[0.0], np.cumsum(usd)])
    cns = np.concatenate([[0.0], np.cumsum(snot)])
    ref = pd.Series(ypx).rolling(URG_N, min_periods=5).median().shift(1).to_numpy()
    prem = d * (ypx - ref)
    # first index of each wallet in this market (for n_wallets_total and "first trade here")
    nw = len(wuniq)
    first_idx = np.full(nw, n, np.int64)
    np.minimum.at(first_idx, wcode, np.arange(n))
    first_idx_sorted = np.sort(first_idx)

    C = np.asarray(sorted(cutoffs), float)
    nc = len(C)
    # (wallet, cutoff) pairs: wallet traded in [c - maxL, c)  <=>  c in (t_j, t_j + maxL]
    lo = np.searchsorted(C, t, "right")
    hi = np.searchsorted(C, t + maxl, "right")
    cnt = hi - lo
    if cnt.sum():
        rep = np.repeat(np.arange(n), cnt)
        ci = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + lo[rep]
        pair = np.unique(wcode[rep].astype(np.int64) * nc + ci)
        del rep, ci
    else:
        pair = np.zeros(0, np.int64)
    pk, pc = pair // nc, pair % nc
    in_any = np.zeros(nw, bool)
    in_any[pk] = True
    W = _Wallets(conn, [wuniq[k] for k in np.flatnonzero(in_any)], src)
    wfirst = np.array([W.first_t(w) if in_any[k] else np.nan for k, w in enumerate(wuniq)], float)
    age = t - wfirst[wcode] if n else np.zeros(0)
    # attribute matrix per pair: 8 wallet as-of columns + concentration proxy
    M = np.full((len(pair), 9), np.nan)
    if len(pair):
        order = np.argsort(wcode, kind="stable")
        bounds = np.searchsorted(wcode[order], np.arange(nw + 1), "left")
        pstart = np.searchsorted(pk, np.arange(nw + 1), "left")
        dstart = np.floor(C / 86400) * 86400
        for k in np.flatnonzero(in_any):
            a, b = pstart[k], pstart[k + 1]
            cs = C[pc[a:b]]
            M[a:b, :8] = W.asof_many(wuniq[k], cs, market_id)
            rows_k = order[bounds[k]:bounds[k + 1]]
            tk = t[rows_k]
            cu = np.concatenate([[0.0], np.cumsum(usd[rows_k])])
            in_m = cu[np.searchsorted(tk, cs, "left")]
            today = in_m - cu[np.searchsorted(tk, dstart[pc[a:b]], "left")]
            den = M[a:b, 4] + today
            with np.errstate(invalid="ignore", divide="ignore"):
                M[a:b, 8] = np.where(den > 0, np.minimum(in_m / den, 1.0), np.nan)
    by_c = np.argsort(pc, kind="stable")
    cbounds = np.searchsorted(pc[by_c], np.arange(nc + 1), "left")
    pos = np.full(nw, -1, np.int64)

    out = []
    for ci_, c in enumerate(C):
        i1 = int(np.searchsorted(t, c, "left"))
        base = {
            "market_id": market_id, "cutoff_ts": c,
            "hours_since_open": (c - created) / H if not np.isnan(created) else np.nan,
            "hours_since_first_trade": (c - t[0]) / H if i1 > 0 else np.nan,
            "hours_to_end": (end_t - c) / H if not np.isnan(end_t) else np.nan,
            "hour_utc": datetime.fromtimestamp(c, timezone.utc).hour,
            "dow_utc": datetime.fromtimestamp(c, timezone.utc).weekday(),
            "wallet_age_censored": bool(c < CENSOR_BEFORE),
            "market_history_truncated": bool(np.isnan(created) or created < RETENTION_START + 86400),
            "px_last": float(ypx[i1 - 1]) if i1 > 0 else np.nan,
            "log_gv_total": math.log1p(cgv[i1]),
            "log_gv_7d": math.log1p(cgv[i1] - cgv[np.searchsorted(t, c - 7 * 86400, "left")]),
            "n_trades_total": i1,
            "n_wallets_total": int(np.searchsorted(first_idx_sorted, i1, "left")),
            "category": category, "geo_subtype": geo_subtype, "in_scope": in_scope,
        }
        rr = by_c[cbounds[ci_]:cbounds[ci_ + 1]]
        pos[pk[rr]] = rr
        A = (pos, M)
        for L in lbs:
            rec = dict(base)
            rec["lookback_h"] = L
            rec.update(_window(t, d, usd, snot, ypx, sz, prem, wcode, age, wfirst, first_idx, cgv, cns,
                               c, L * H, A, i1))
            out.append(rec)
        pos[pk[rr]] = -1
    return out


def _attr(A, ks, j, default=np.nan):
    pos, M = A
    idx = pos[ks]
    return np.where(idx >= 0, M[np.maximum(idx, 0), j], default)


def _window(t, d, usd, snot, ypx, sz, prem, wcode, age, wfirst, first_idx, cgv, cns, c, Ls, A, i1):
    f: dict = {}
    i0 = int(np.searchsorted(t, c - Ls, "left"))
    nt = i1 - i0
    gv = cgv[i1] - cgv[i0]
    nsn = cns[i1] - cns[i0]
    f["n_trades"] = nt
    f["gv"] = gv
    f["nsn"] = nsn
    f["log_gv"] = math.log1p(gv)
    f["os"] = abs(nsn) / gv if gv > 0 else np.nan
    f["yes_share"] = (gv + nsn) / (2 * gv) if gv > 0 else np.nan

    # baselines: K preceding non-overlapping windows (recent) and K windows ending >= 7d earlier
    def base_vals(end0):
        ends = end0 - Ls * np.arange(K_BASE)
        starts = ends - Ls
        ok = ends > (t[0] if len(t) else np.inf)
        e = np.searchsorted(t, ends[ok], "left"); s = np.searchsorted(t, starts[ok], "left")
        return cgv[e] - cgv[s], cns[e] - cns[s]

    bg, bn = base_vals(c - Ls)
    f["gv_z"], med_gv, f["n_base"] = _robust_z(gv, bg)
    f["nsn_z"], _, _ = _robust_z(nsn, bn)
    f["gv_ratio_base"] = gv / med_gv if (med_gv == med_gv and med_gv > 0) else np.nan
    _, bn7 = base_vals(c - BASE_GAP_S)
    f["wl_z_nsn"], _, f["wl_n_base"] = _robust_z(nsn, bn7)
    f["wl_base_med_abs_nsn"] = float(np.median(np.abs(bn7))) if len(bn7) >= MIN_BASE else np.nan

    f["px_start"] = float(ypx[i0 - 1]) if i0 > 0 else np.nan
    f["px_end"] = float(ypx[i1 - 1]) if i1 > 0 else np.nan
    f["dp"] = f["px_end"] - f["px_start"]
    f["impact_per_10k"] = f["dp"] / (nsn / 1e4) if abs(nsn) >= 100 else np.nan

    dom = 1.0 if nsn > 0 else (-1.0 if nsn < 0 else 0.0)
    f["dom"] = dom
    f["dp_dom"] = f["dp"] * dom if dom else np.nan

    names_nan = ["p_entry", "vwap_yes", "max_trade", "q50_trade", "q90_trade", "top5_trade_share",
                 "n_ge1k", "n_ge10k", "share_ge1k", "burst_B", "peak10m_share", "urg_prem", "urg_share_1c",
                 "fws72", "fws24", "fresh_nsn", "fresh_nsn_share", "dom_gv", "dom_n_wallets", "dom_age_med_d",
                 "dom_newmkt_share", "dom_log_prior_mkts", "dom_winrate", "dom_winrate_cov", "dom_prior_pnl",
                 "top3_share", "top1_share", "conc_top3", "conc80_share", "n_new_wallets_dom", "n_clusters3",
                 "max_cluster", "cluster_share", "whale_hist_share", "whale_hist_mixed_share",
                 "hist_good_share", "fresh_conc_share", "full_cov_share", "fws72_full",
                 "dom_log_prior_mkts_full", "n_wallets"]
    for k in names_nan:
        f[k] = np.nan
    if nt == 0:
        f["n_trades"] = 0
        return f
    sl = slice(i0, i1)
    u, dd, ww, tw, yp, ag = usd[sl], d[sl], wcode[sl], t[sl], ypx[sl], age[sl]
    f["n_wallets"] = len(np.unique(ww))
    f["vwap_yes"] = float((yp * sz[sl]).sum() / sz[sl].sum())
    f["max_trade"] = float(u.max())
    f["q50_trade"] = float(np.quantile(u, 0.5))
    f["q90_trade"] = float(np.quantile(u, 0.9))
    f["top5_trade_share"] = float(np.sort(u)[-5:].sum() / gv) if gv > 0 else np.nan
    f["n_ge1k"] = int((u >= 1000).sum())
    f["n_ge10k"] = int((u >= 10000).sum())
    f["share_ge1k"] = float(u[u >= 1000].sum() / gv) if gv > 0 else np.nan
    f["burst_B"] = _burstiness(tw)
    bins = np.floor((tw - (c - Ls)) / 600).astype(np.int64)
    f["peak10m_share"] = float(np.bincount(bins).max() / nt)
    pr = prem[sl]
    okp = ~np.isnan(pr)
    if okp.any() and u[okp].sum() > 0:
        f["urg_prem"] = float((pr[okp] * u[okp]).sum() / u[okp].sum())
        f["urg_share_1c"] = float(u[okp][pr[okp] > 0.01].sum() / u[okp].sum())
    fr = ag < FRESH_S
    f["fresh_nsn"] = float((snot[sl] * fr).sum())
    f["fresh_nsn_share"] = f["fresh_nsn"] / gv if gv > 0 else np.nan
    if dom == 0:
        return f
    m = dd == dom
    ud, wd, agd = u[m], ww[m], ag[m]
    dgv = ud.sum()
    f["dom_gv"] = float(dgv)
    own = yp[m] if dom > 0 else 1.0 - yp[m]
    f["p_entry"] = float((own * sz[sl][m]).sum() / sz[sl][m].sum())
    f["fws72"] = float(ud[agd < FRESH_S].sum() / dgv)
    f["fws24"] = float(ud[agd < FRESH24_S].sum() / dgv)
    f["dom_age_med_d"] = _wq(agd / 86400, ud, 0.5)
    ks, inv = np.unique(wd, return_inverse=True)
    wn = np.bincount(inv, ud)
    f["dom_n_wallets"] = len(ks)
    order = np.argsort(-wn)
    f["top1_share"] = float(wn[order[:1]].sum() / dgv)
    f["top3_share"] = float(wn[order[:3]].sum() / dgv)
    prior = _attr(A, ks, 0); rn = _attr(A, ks, 1); rw = _attr(A, ks, 2); rpnl = _attr(A, ks, 3)
    conc = _attr(A, ks, 8)
    full_ok = _attr(A, ks, 5, 0.0) > 0
    full_first = _attr(A, ks, 6); full_pm = _attr(A, ks, 7)
    share = wn / dgv
    okp = ~np.isnan(prior)
    if okp.any():
        f["dom_newmkt_share"] = float(share[okp & (prior == 0)].sum() / share[okp].sum())
        f["dom_log_prior_mkts"] = float((np.log1p(prior[okp]) * share[okp]).sum() / share[okp].sum())
    wr = np.where(rn > 0, rw / np.maximum(rn, 1), np.nan)
    okr = rn >= 3
    f["dom_winrate_cov"] = float(share[okr].sum())
    if okr.any():
        f["dom_winrate"] = float((wr[okr] * share[okr]).sum() / share[okr].sum())
    f["dom_prior_pnl"] = float(np.nansum(rpnl))
    top3 = order[:3]
    okc = ~np.isnan(conc[top3])
    if okc.any():
        f["conc_top3"] = float((conc[top3][okc] * wn[top3][okc]).sum() / wn[top3][okc].sum())
    f["conc80_share"] = float(share[np.nan_to_num(conc) >= CONC_HI].sum())
    hist = np.nan_to_num(prior) >= HIST_MKTS
    mixed = hist & (rn >= MIN_RES) & (wr >= MIXED[0]) & (wr <= MIXED[1])
    good = (rn >= MIN_RES) & (np.nan_to_num(wr) > MIXED[1])
    f["whale_hist_share"] = float(share[hist].sum())
    f["whale_hist_mixed_share"] = float(share[mixed].sum())
    f["hist_good_share"] = float(share[good].sum())
    # fresh at the wallet's first dominant-side trade in the window, or <= 1 prior corpus market
    wfa = np.full(len(ks), np.inf)
    np.minimum.at(wfa, inv, agd)
    fresh_w = (wfa < FRESH_S) | (np.nan_to_num(prior, nan=99) <= 1)
    f["fresh_conc_share"] = float(share[fresh_w & (np.nan_to_num(conc) >= CONC_HI)].sum())
    # co-timed fresh-wallet clusters: wallets whose first-ever corpus trade is in this window,
    # in this market, and on the dominant side
    gidx = np.arange(i0, i1)[m]
    newk = []
    for j, k in enumerate(ks):
        fi = first_idx[k]
        if i0 <= fi < i1 and d[fi] == dom and abs(t[fi] - wfirst[k]) < 1.0:
            newk.append((t[fi], j))
    f["n_new_wallets_dom"] = len(newk)
    f["n_clusters3"], f["max_cluster"], f["cluster_share"] = 0, 0, 0.0
    if newk:
        newk.sort()
        tt = np.array([x[0] for x in newk]); jj = np.array([x[1] for x in newk])
        grp = np.concatenate([[0], np.cumsum(np.diff(tt) > CLUSTER_GAP_S)])
        sizes = np.bincount(grp)
        f["max_cluster"] = int(sizes.max())
        big = sizes >= CLUSTER_MIN
        f["n_clusters3"] = int(big.sum())
        f["cluster_share"] = float(share[jj[big[grp]]].sum())
    # full Polymarket history (pm_wallets), where available and eligible as of c
    f["full_cov_share"] = float(share[full_ok].sum())
    if full_ok.any():
        fage = np.full(len(ks), np.nan)
        # age at the wallet's first dominant-side trade in the window
        wft = np.full(len(ks), np.inf)
        np.minimum.at(wft, inv, tw[m])
        fage[full_ok] = wft[full_ok] - full_first[full_ok]
        f["fws72_full"] = float(share[full_ok & (fage < FRESH_S)].sum() / share[full_ok].sum())
        f["dom_log_prior_mkts_full"] = float(
            (np.log1p(np.maximum(full_pm[full_ok], 0)) * share[full_ok]).sum() / share[full_ok].sum())
    del gidx
    return f


# ------------------------------------------------------------------ public API
def compute_features_batch(requests, lookbacks=LOOKBACKS_H, conn=None, src: Sources = FULL,
                           progress: bool = False) -> pd.DataFrame:
    if isinstance(requests, pd.DataFrame):
        req = requests.copy()
    else:
        req = pd.DataFrame(list(requests), columns=["market_id", "cutoff_ts"])
    req["_c"] = [_ts(x) for x in req["cutoff_ts"]]
    own = conn is None
    conn = conn or connect()
    out = []
    try:
        groups = list(req.groupby("market_id"))
        for gi, (mid, g) in enumerate(groups):
            cs = sorted(set(g["_c"]))
            out.extend(_market_features(conn, int(mid), cs, lookbacks, src))
            if progress and gi % 100 == 0:
                print(f"  market {gi+1}/{len(groups)}", flush=True)
    finally:
        if own:
            conn.close()
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df = df.rename(columns={"cutoff_ts": "_c"})
    keep = [c for c in req.columns if c not in ("cutoff_ts",)]
    df = req[keep + ["cutoff_ts"]].merge(df, on=["market_id", "_c"], how="left")
    df["cutoff_ts"] = pd.to_datetime(df["_c"], unit="s", utc=True)
    return df.drop(columns="_c")


def compute_window_features(market_id: int, cutoff_ts, lookbacks=LOOKBACKS_H, conn=None,
                            src: Sources = FULL) -> pd.DataFrame:
    return compute_features_batch([(market_id, cutoff_ts)], lookbacks, conn=conn, src=src)


def to_wide(df: pd.DataFrame, key=("market_id", "cutoff_ts")) -> pd.DataFrame:
    """Pivot the long output to one row per key with ``<feature>_<L>h`` window columns."""
    key = [k for k in key if k in df.columns]
    ids = [c for c in ("snapshot_id", "market_id", "cutoff_ts") if c in df.columns and c not in key]
    cut = [c for c in CUTOFF_FEATURES if c in df.columns]
    win = [c for c in df.columns if c not in set(key) | set(ids) | set(cut) | {"lookback_h"}]
    first = df.drop_duplicates(key)[key + ids + cut]
    piv = df.set_index(key + ["lookback_h"])[win].unstack("lookback_h")
    piv.columns = [f"{a}_{int(b)}h" for a, b in piv.columns]
    return first.merge(piv.reset_index(), on=key, how="left")
