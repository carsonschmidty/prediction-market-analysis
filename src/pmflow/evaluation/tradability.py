"""Tradability simulator: enter after a signal at cutoff with a realistic fill, hold to resolution.

No historical order books exist, so fills are reconstructed from public taker trades (role 0):

* Entry is allowed from ``cutoff_ts + latency``. The reference fill is the share-weighted price of the
  first ``n_ref`` taker trades after that time, within ``max_wait``, whose aggressor bought OUR side
  (buy YES = BUY outcome 0 or SELL outcome 1; buy NO = the opposite). Those trades executed at the ask,
  so their price already contains the spread.
* If no same-side trade happens within ``max_wait`` and ``fallback`` is 'any', the first trade of any side
  is used: an opposite-side trade (executed at the bid) gets ``+ spread_proxy``, where spread_proxy is the
  median absolute YES-equivalent price jump between consecutive opposite-aggressor trades among the last
  ``spread_lookback`` trades before the cutoff (an effective-spread proxy; floor ``min_spread``).
  Otherwise the signal is recorded as unfilled.
* Size impact: ``impact_coef * sqrt(stake / max(usd_24h, impact_floor_usd))`` added to the price (usd_24h =
  pre-cutoff 24h taker notional), plus a flat ``extra_slippage``. Price capped at ``max_price``.
* Fees: ``fee_model='bps'``: fee = fee_bps/1e4 * stake; ``'prob'``: fee per share = fee_rate*min(p,1-p)
  (shape of Polymarket's taker-fee formula where fees apply). Default 0: geopolitical markets on Polymarket
  have generally been fee-free; run sensitivity with non-zero fees.
* Hold to resolution: payoff 1 per share if our side wins. Return = pnl / stake.
* ``dedupe='first_per_market'`` (default) keeps the first signal per (market, direction); 'all' keeps every
  flagged snapshot (overlapping bets; inflates n but CIs are event-clustered either way).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import metrics as M
from .harness import connect


@dataclass
class TradeParams:
    stake_usd: float = 100.0
    latency_s: float = 60.0
    max_wait_h: float = 6.0
    n_ref: int = 3
    fallback: str = "any"            # 'any' | 'none'
    spread_lookback: int = 50
    min_spread: float = 0.005
    impact_coef: float = 0.05
    impact_floor_usd: float = 500.0
    extra_slippage: float = 0.0
    max_price: float = 0.995
    fee_model: str = "bps"           # 'bps' | 'prob'
    fee_bps: float = 0.0
    fee_rate: float = 0.0
    dedupe: str = "first_per_market"


SQL_POST = """
WITH s AS (
  SELECT * FROM unnest(%(sid)s::bigint[]) AS u(snapshot_id)
)
SELECT e.snapshot_id, t.ts, CASE WHEN t.outcome_index = 0 THEN t.price ELSE 1 - t.price END AS p_yes,
       ((t.outcome_index = 0) = t.is_buy) AS buys_yes, t.size
FROM s JOIN eval_snapshots e USING (snapshot_id)
JOIN eval_markets m USING (market_id)
JOIN LATERAL (SELECT * FROM pm_trades t WHERE t.market_id = e.market_id AND t.role = 0
              AND t.ts >= e.cutoff_ts + make_interval(secs => %(lat)s)
              AND t.ts < LEAST(e.cutoff_ts + make_interval(secs => %(lat)s) + make_interval(secs => %(wait)s * 3600),
                               m.closed_time)
              ORDER BY t.ts LIMIT 200) t ON true
"""

SQL_PRE = """
WITH s AS (SELECT * FROM unnest(%(sid)s::bigint[]) AS u(snapshot_id))
SELECT e.snapshot_id, t.ts, CASE WHEN t.outcome_index = 0 THEN t.price ELSE 1 - t.price END AS p_yes,
       ((t.outcome_index = 0) = t.is_buy) AS buys_yes
FROM s JOIN eval_snapshots e USING (snapshot_id)
JOIN LATERAL (SELECT * FROM pm_trades t WHERE t.market_id = e.market_id AND t.role = 0 AND t.ts < e.cutoff_ts
              ORDER BY t.ts DESC LIMIT %(lb)s) t ON true
"""


def _spread_proxy(pre: pd.DataFrame, min_spread: float) -> pd.Series:
    pre = pre.sort_values(["snapshot_id", "ts"])
    dp = pre.groupby("snapshot_id").p_yes.diff().abs()
    flip = pre.groupby("snapshot_id").buys_yes.shift() != pre.buys_yes
    v = pre.assign(j=dp.where(flip & dp.notna())).groupby("snapshot_id").j.median()
    return v.fillna(min_spread).clip(lower=min_spread)


def simulate(signals: pd.DataFrame, params: TradeParams | None = None, chunk: int = 2000) -> pd.DataFrame:
    """signals: DataFrame(snapshot_id, direction in {+1,-1}). Returns one row per (deduped) signal."""
    tp = params or TradeParams()
    with connect() as c:
        cur = c.execute("SELECT snapshot_id, market_id, event_id, universe, split, cutoff_ts, y, price_yes, usd_24h "
                        "FROM eval_snapshots WHERE snapshot_id = ANY(%s)", [list(map(int, signals.snapshot_id))])
        snap = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    sig = signals.merge(snap, on="snapshot_id").sort_values("cutoff_ts")
    sig = sig[sig.direction != 0]
    if tp.dedupe == "first_per_market":
        sig = sig.drop_duplicates(["market_id", "direction"], keep="first")
    out = []
    ids = sig.snapshot_id.astype("int64").tolist()
    with connect() as c:
        for i in range(0, len(ids), chunk):
            part = ids[i:i + chunk]
            cur = c.execute(SQL_POST, dict(sid=part, lat=tp.latency_s, wait=tp.max_wait_h))
            post = pd.DataFrame(cur.fetchall(), columns=["snapshot_id", "ts", "p_yes", "buys_yes", "size"])
            cur = c.execute(SQL_PRE, dict(sid=part, lb=tp.spread_lookback))
            pre = pd.DataFrame(cur.fetchall(), columns=["snapshot_id", "ts", "p_yes", "buys_yes"])
            sp = _spread_proxy(pre, tp.min_spread) if len(pre) else pd.Series(dtype=float)
            out.append(_fills(sig[sig.snapshot_id.isin(part)], post, sp, tp))
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    return res


def _fills(sig: pd.DataFrame, post: pd.DataFrame, sp: pd.Series, tp: TradeParams) -> pd.DataFrame:
    post = post.merge(sig[["snapshot_id", "direction"]], on="snapshot_id")
    post["our_side_price"] = np.where(post.direction > 0, post.p_yes, 1 - post.p_yes)
    post["same_side"] = np.where(post.direction > 0, post.buys_yes, ~post.buys_yes.astype(bool))
    post = post.sort_values(["snapshot_id", "ts"])
    same = post[post.same_side].groupby("snapshot_id").head(tp.n_ref)
    ref_same = same.groupby("snapshot_id").apply(
        lambda g: np.average(g.our_side_price, weights=g["size"]), include_groups=False)
    first_any = post.groupby("snapshot_id").head(1).set_index("snapshot_id")
    rows = []
    for r in sig.itertuples(index=False):
        sid = r.snapshot_id
        p_cut = r.price_yes if r.direction > 0 else 1 - r.price_yes
        if sid in ref_same.index:
            ref, how = float(ref_same[sid]), "same_side"
        elif tp.fallback == "any" and sid in first_any.index:
            f = first_any.loc[sid]
            ref = float(f.our_side_price) + (0.0 if f.same_side else float(sp.get(sid, tp.min_spread)))
            how = "any_plus_spread"
        else:
            rows.append(dict(snapshot_id=sid, filled=False)); continue
        impact = tp.impact_coef * np.sqrt(tp.stake_usd / max(float(r.usd_24h or 0), tp.impact_floor_usd))
        price = min(ref + impact + tp.extra_slippage, tp.max_price)
        shares = tp.stake_usd / price
        fee = (tp.fee_bps / 1e4 * tp.stake_usd) if tp.fee_model == "bps" else tp.fee_rate * min(price, 1 - price) * shares
        win = int(r.y == 1) if r.direction > 0 else int(r.y == 0)
        pnl = shares * win - tp.stake_usd - fee
        rows.append(dict(snapshot_id=sid, filled=True, fill_how=how, price_at_cutoff=p_cut, ref_price=ref,
                         impact=impact, fill_price=price, fee=fee, win=win, pnl=pnl, ret=pnl / tp.stake_usd))
    res = pd.DataFrame(rows).merge(sig[["snapshot_id", "market_id", "event_id", "universe", "split", "direction",
                                        "cutoff_ts"]], on="snapshot_id")
    return res


def summarize_trades(trades: pd.DataFrame, B: int = 1000) -> pd.DataFrame:
    rows = []
    if trades.empty:
        return pd.DataFrame()
    for (u, s), d in trades.groupby(["universe", "split"]):
        f = d[d.filled]
        row = dict(universe=u, split=s, n_signals=len(d), fill_rate=float(d.filled.mean()))
        if len(f):
            pt, lo, hi = M.clustered_mean_ci(f.ret.to_numpy(), f.event_id.to_numpy(), B=B)
            row.update(n_filled=len(f), n_evt=f.event_id.nunique(), hit_rate=float(f.win.mean()),
                       mean_fill=float(f.fill_price.mean()), mean_cost_vs_cutoff=float((f.fill_price - f.price_at_cutoff).mean()),
                       ret_per_dollar=pt, ret_lo=lo, ret_hi=hi, total_pnl=float(f.pnl.sum()),
                       share_fallback=float((f.fill_how != "same_side").mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def params_dict(tp: TradeParams) -> dict:
    return asdict(tp)
