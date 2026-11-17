"""Final TEST tradability simulation (evaluation role), at the FROZEN flag thresholds.

    python -m pmflow.evaluation.final_trade

Uses `tradability.simulate` unchanged (taker-trade fills, role 0 only). Adds: event-level dedupe (by
pre-filtering signals), one-at-a-time sensitivity around the default cost assumptions, the price-only control,
a naive "buy YES at the candidate's flag rate" control (random draws) and "buy YES at each market's first test
snapshot", edge decay between cutoff and fill, and event-clustered CIs for return per $ and total PnL.
Outputs: artifacts/evaluation/final/trade_*.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import metrics as M
from .final_audit import OUT
from .tradability import TradeParams, simulate

MODELS = ["lgbm_anchored", "stack_s1", "ridge_flow", "ctrl_lgbm_price_only"]
DEFAULT = dict(impact_coef=0.05, stake_usd=100.0, latency_s=60.0, fee_bps=0.0)
GRID = ([("impact_coef", v) for v in (0.0, 0.1, 0.2)] + [("fee_bps", v) for v in (50.0, 100.0, 200.0)] +
        [("latency_s", v) for v in (3600.0, 21600.0)] + [("stake_usd", v) for v in (10.0, 1000.0, 10000.0)])
N_NAIVE = 10


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def clustered_total(pnl, groups, B=2000, seed=0):
    g, (sp,) = M._cluster_sums(groups, pnl)
    G = len(sp)
    rng = np.random.default_rng(seed)
    draws = (rng.multinomial(G, np.full(G, 1 / G), size=B) @ sp)
    return float(sp.sum()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def run(signals: pd.DataFrame, dedupe: str, **kw) -> pd.DataFrame:
    sig = signals.sort_values("cutoff_ts")
    if dedupe == "first_per_event":
        sig = sig.drop_duplicates(["event_id", "direction"], keep="first")
        dd = "all"
    else:
        dd = dedupe
    p = dict(DEFAULT, **kw)
    tp = TradeParams(stake_usd=p["stake_usd"], latency_s=p["latency_s"], impact_coef=p["impact_coef"],
                     fee_bps=p["fee_bps"], dedupe=dd)
    if sig.empty:
        return pd.DataFrame()
    return simulate(sig[["snapshot_id", "direction"]], tp)


def summarize(tr: pd.DataFrame, sig: pd.DataFrame, label: dict) -> list[dict]:
    rows = []
    if tr.empty:
        return [dict(label, universe=u, n_signals=0) for u in ("in_scope", "macro")]
    tr = tr.merge(sig[["snapshot_id", "p_side"]], on="snapshot_id", how="left")
    for u, d in tr.groupby("universe"):
        f = d[d.filled]
        row = dict(label, universe=u, n_signals=len(d), fill_rate=float(d.filled.mean()))
        if len(f):
            g = f.event_id.to_numpy()
            pt, lo, hi = M.clustered_mean_ci(f.ret.to_numpy(), g, B=2000)
            tot, tlo, thi = clustered_total(f.pnl.to_numpy(), g)
            row.update(n_filled=len(f), n_mkt=f.market_id.nunique(), n_evt=f.event_id.nunique(),
                       share_yes=float((f.direction > 0).mean()), hit_rate=float(f.win.mean()),
                       mean_price_cutoff=float(f.price_at_cutoff.mean()), mean_fill=float(f.fill_price.mean()),
                       mean_cost_vs_cutoff=float((f.fill_price - f.price_at_cutoff).mean()),
                       edge_at_cutoff=float((f.p_side - f.price_at_cutoff).mean()) if f.p_side.notna().any() else np.nan,
                       edge_at_fill=float((f.p_side - f.fill_price).mean()) if f.p_side.notna().any() else np.nan,
                       realized_edge_at_fill=float((f.win - f.fill_price).mean()),
                       ret_per_dollar=pt, ret_lo=lo, ret_hi=hi, total_pnl=tot, pnl_lo=tlo, pnl_hi=thi,
                       stake_total=float(len(f) * label.get("stake_usd", DEFAULT["stake_usd"])))
        rows.append(row)
    return rows


def signals_for(s: pd.DataFrame, m: str) -> pd.DataFrame:
    f = s[s[f"flag_{m}"] != 0]
    d = np.sign(f[f"flag_{m}"]).astype(int)
    return pd.DataFrame(dict(snapshot_id=f.snapshot_id, market_id=f.market_id, event_id=f.event_id,
                             cutoff_ts=f.cutoff_ts, direction=d,
                             p_side=np.where(d > 0, f[m], 1 - f[m]), universe=f.universe))


def main():
    s = pd.read_parquet(OUT / "test_scored_frame.parquet")
    rows = []
    trades_default = []
    for m in MODELS:
        sig = signals_for(s, m)
        log(f"{m}: {len(sig)} flagged test snapshots")
        for dd in ("first_per_market", "first_per_event", "all"):
            tr = run(sig, dd)
            rows += summarize(tr, sig, dict(model=m, dedupe=dd, varied="default", value=np.nan, **DEFAULT))
            if dd == "first_per_market" and not tr.empty:
                trades_default.append(tr.assign(model=m))
        for k, v in GRID:
            tr = run(sig, "first_per_market", **{k: v})
            rows += summarize(tr, sig, dict(model=m, dedupe="first_per_market", varied=k, value=v,
                                            **dict(DEFAULT, **{k: v})))
    # naive YES controls at each flagging model's rate (per universe), random draws
    rng = np.random.default_rng(20261117)
    for m in ("lgbm_anchored", "stack_s1", "ctrl_lgbm_price_only"):
        for i in range(N_NAIVE):
            picks = []
            for u in ("in_scope", "macro"):
                su = s[s.universe == u]
                n = int((su[f"flag_{m}"] != 0).sum())
                if n:
                    picks.append(su.iloc[rng.choice(len(su), n, replace=False)])
            p = pd.concat(picks)
            sig = pd.DataFrame(dict(snapshot_id=p.snapshot_id, market_id=p.market_id, event_id=p.event_id,
                                    cutoff_ts=p.cutoff_ts, direction=1, p_side=np.nan, universe=p.universe))
            tr = run(sig, "first_per_market")
            rows += summarize(tr, sig, dict(model=f"naive_yes_rate_of_{m}", dedupe="first_per_market",
                                            varied="draw", value=i, **DEFAULT))
        log(f"naive draws for {m} done")
    first = s.sort_values("cutoff_ts").drop_duplicates("market_id")
    sig = pd.DataFrame(dict(snapshot_id=first.snapshot_id, market_id=first.market_id, event_id=first.event_id,
                            cutoff_ts=first.cutoff_ts, direction=1, p_side=np.nan, universe=first.universe))
    rows += summarize(run(sig, "first_per_market"), sig, dict(model="buy_yes_first_test_snapshot_all_markets",
                                                              dedupe="first_per_market", varied="default",
                                                              value=np.nan, **DEFAULT))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "trade_summary.csv", index=False)
    if trades_default:
        pd.concat(trades_default).to_csv(OUT / "trade_default_trades.csv", index=False)
    cols = ["model", "dedupe", "varied", "value", "universe", "n_filled", "n_evt", "share_yes", "hit_rate",
            "mean_fill", "edge_at_cutoff", "edge_at_fill", "ret_per_dollar", "ret_lo", "ret_hi", "total_pnl"]
    print(R[[c for c in cols if c in R]].round(4).to_string())


if __name__ == "__main__":
    main()
