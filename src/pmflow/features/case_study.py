"""Prototype on the reported cases: is the reported informed-flow footprint visible in our
taker-trade data, and how often does the same footprint appear in ordinary windows?

For every reported case matched to a corpus market with trades:
  pre       cutoff = lower edge of the case's news interval (ReportedCase.news_window()[0]);
            day-precision cases therefore use 00:00Z of the reported date (conservative).
  baseline  same market, cutoffs 7..36 days before the pre cutoff (daily, same time of day).
  reaction  windows [upper edge, upper edge + L) (for the contrast cases, e.g. PM-023).
  matched   non-case in-scope snapshots from feat_snapshot_features (00:00Z grid, live),
            the 50 nearest by (px_last, log_gv_7d) for the same lookback.
Pool base rate = share of ALL in-scope grid snapshots (non-case markets) where the pattern fires.

Claim checks compare the reports' sizes / timing / fresh-account claims with the taker trades in
the reported flow window (aggregate statistics only; no wallet identifiers are written).

Outcome (vindication) is used here descriptively only (which side the reports say was informed).

CLI: python -m pmflow.features.case_study
"""
from __future__ import annotations

import re
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from pmflow.labeling.cases import load_cases

from .core import compute_features_batch
from .db import connect
from .weak_labels import pattern_flags

OUT = "artifacts/features"
LBS = (1, 6, 24, 72)
KEY = ["nsn", "gv", "os", "p_entry", "dp_dom", "gv_z", "wl_z_nsn", "fws72", "dom_age_med_d",
       "dom_newmkt_share", "top3_share", "conc_top3", "n_new_wallets_dom", "max_cluster",
       "n_clusters3", "fresh_conc_share", "whale_hist_mixed_share", "dom_winrate", "urg_prem",
       "max_trade", "full_cov_share", "fws72_full"]
DIRECTIONAL = {"nsn", "wl_z_nsn"}   # multiplied by dom (dominant-side magnitude)
# Descriptive, label-oriented (uses the resolved outcome; NOT features): net flow toward the
# eventual winner, overall and from fresh (<72h in-corpus) wallets. Pool controls use eval y.
TO_WINNER = {"nsn_to_winner": "nsn", "fresh_nsn_to_winner": "fresh_nsn"}
KEY = KEY + list(TO_WINNER)


def _case_rows(conn):
    cases = load_cases()
    from pmflow.labeling.cases import match_to_markets
    match_to_markets(cases)
    meta = pd.read_sql_query(
        "select m.market_id, m.question, m.winning_outcome_index, m.in_scope, m.macro_comparison,"
        " s.n_taker, s.first_trade_ts from feat_market_meta m left join pm_market_trade_stats s using (market_id)",
        conn).set_index("market_id")
    rows, skipped = [], []
    for c in cases:
        nw = c.news_window()
        mids = [m for m in c.pg_market_ids if m in meta.index and (meta.loc[m, "n_taker"] or 0) > 0]
        if not mids or nw is None:
            why = ("no news timestamp" if nw is None else
                   "market not in pm_markets" if not c.pg_market_ids else "no trades in corpus")
            skipped.append((c.case_id, c.case_type, why))
            continue
        for m in mids:
            rows.append(dict(case_id=c.case_id, case_type=c.case_type, market_id=m, t_lo=nw[0], t_hi=nw[1],
                             precision=c.ts_precision, vindicated=c.outcome_vindicated,
                             win_idx=meta.loc[m, "winning_outcome_index"], case=c))
    return rows, skipped


def _requests(rows):
    req = []
    for r in rows:
        lo, hi = pd.Timestamp(r["t_lo"]), pd.Timestamp(r["t_hi"])
        req.append((r["case_id"], r["market_id"], "pre", 0, lo))
        for k in range(30):
            req.append((r["case_id"], r["market_id"], "baseline", k, lo - timedelta(days=7 + k)))
        for L in LBS:
            req.append((r["case_id"], r["market_id"], f"reaction_{L}h", 0, hi + timedelta(hours=L)))
    return pd.DataFrame(req, columns=["case_id", "market_id", "window", "k", "cutoff_ts"])


def _epoch(s: pd.Series) -> pd.Series:
    return (pd.to_datetime(s, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def _signed(df):
    df = df.copy()
    for f in DIRECTIONAL:
        df[f] = df[f] * df["dom"]
    return df


def _pctile(x, ref):
    ref = ref[~np.isnan(ref)]
    if len(ref) == 0 or np.isnan(x):
        return np.nan
    return float((ref < x).mean() + 0.5 * (ref == x).mean())


def _rz(x, ref):
    ref = ref[~np.isnan(ref)]
    if len(ref) < 3 or np.isnan(x):
        return np.nan
    med = np.median(ref)
    s = max(1.4826 * np.median(np.abs(ref - med)), 1.2533 * np.mean(np.abs(ref - med)), 1e-9)
    return float((x - med) / s)


def _pool(conn, case_mids):
    cols = []
    for L in LBS:
        cols += [f"{f}_{L}h" for f in (set(KEY) - set(TO_WINNER)) | {"dom", "wl_base_med_abs_nsn", "fresh_nsn"}]
    q = (f"select f.snapshot_id, f.market_id, f.px_last, f.log_gv_7d, e.split, e.y, {', '.join(cols)}"
         f" from feat_snapshot_features f join eval_snapshots e using (snapshot_id)"
         f" where f.in_scope and not (f.market_id = any(%(m)s))")
    pool = pd.read_sql_query(q, conn, params={"m": list(case_mids)})
    sgn = 2.0 * pool["y"] - 1.0
    for L in LBS:
        for k, base in TO_WINNER.items():
            pool[f"{k}_{L}h"] = pool[f"{base}_{L}h"] * sgn
    return pool


def _claims(conn, r, reported_end=False):
    """Aggregate checks of the report's claims in the reported flow window.

    reported_end=False: window ends at the conservative news lower edge (what a feature may see).
    reported_end=True : window ends at the report's flow_window_end / public news time, to test
                        the reports' own claims (NOT usable as a feature window)."""
    c = r["case"]
    end = pd.Timestamp(r["t_lo"])
    if reported_end:
        e2 = c.flow_window_end or c.public_news_ts
        if e2 is None or c.ts_precision not in ("minute", "hour"):
            return None
        end = pd.Timestamp(e2)
    start = pd.Timestamp(c.flow_window_start) if c.flow_window_start else end - timedelta(hours=72)
    win = r["win_idx"]
    if win not in (0, 1):
        return None
    # reported side: the winning side for vindicated/pending cases, the losing side for negatives
    side_yes = (win == 0) if r["vindicated"] is not False else (win == 1)
    s = +1 if side_yes else -1
    t = pd.read_sql_query(
        """select t.ts, t.outcome_index, t.is_buy, t.price, t.size, t.wallet, h.first_t, h.mkt_first_t,
                  h.mkt_ids, extract(epoch from w.first_trade_ts) full_first_t, w.activity_complete
           from pm_trades t join feat_wallet_hist h using (wallet) left join pm_wallets w using (wallet)
           where t.market_id = %(m)s and t.role = 0 and t.ts >= %(a)s and t.ts < %(b)s""",
        conn, params={"m": r["market_id"], "a": start, "b": end})
    if t.empty:
        return dict(case_id=r["case_id"], market_id=r["market_id"], flow_start=start, flow_end=end, n_trades=0)
    t["d"] = np.where((t.outcome_index == 0) == t.is_buy, 1, -1)
    t["usd"] = t["price"] * t["size"]
    t["own"] = np.where(t.outcome_index == 0, t.price, 1 - t.price)
    t.loc[t.d < 0, "own"] = 1 - t.loc[t.d < 0, "own"]
    t["tsec"] = _epoch(t.ts)
    side = t[t.d == s].copy()
    gv = t.usd.sum()
    out = dict(case_id=r["case_id"], market_id=r["market_id"], flow_start=start, flow_end=end,
               window_end_rule="reported" if reported_end else "news_lower_edge",
               reported_side="YES" if side_yes else "NO", n_trades=len(t), gv=gv,
               side_usd=side.usd.sum(), side_share=side.usd.sum() / gv if gv else np.nan)
    if side.empty:
        return out
    side["wid"] = side.wallet.map(bytes)
    side["shares"] = side["size"]
    g = side.groupby("wid")
    w = pd.DataFrame({"usd": g.usd.sum(), "n": g.size(), "t0": g.tsec.min(),
                      "vwap": g.apply(lambda x: (x.own * x.shares).sum() / x.shares.sum()),
                      "first_t": g.first_t.first(), "full_first_t": g.full_first_t.first(),
                      "complete": g.activity_complete.first()})
    w["age_d"] = (w.t0 - w.first_t) / 86400
    w["full_age_d"] = (w.t0 - w.full_first_t) / 86400
    mk, mi = g.mkt_first_t.first(), g.mkt_ids.first()
    w["prior_mkts"] = [int(np.sum((np.asarray(mk[i]) < w.t0[i]) & (np.asarray(mi[i]) != r["market_id"])))
                       for i in w.index]
    w = w.sort_values("usd", ascending=False)
    top = w.iloc[0]
    out.update(
        side_vwap=float((side.own * side.shares).sum() / side.shares.sum()),
        side_n_wallets=len(w), side_n_ge10k=int((w.usd >= 1e4).sum()),
        side_new_wallets_7d=int((w.age_d < 7).sum()), side_new_usd_share_7d=float(w.usd[w.age_d < 7].sum() / w.usd.sum()),
        side_new_wallets_72h=int((w.age_d < 3).sum()),
        top_usd=float(top.usd), top_n_trades=int(top.n), top_vwap=float(top.vwap), top_age_d=float(top.age_d),
        top_prior_corpus_mkts=int(top.prior_mkts),
        top_full_age_d=float(top.full_age_d) if top.complete is True else np.nan,
        top_first_trade_h_before_news=float((end.timestamp() - top.t0) / 3600),
        top3_usd_share=float(w.usd.iloc[:3].sum() / w.usd.sum()),
    )
    m = re.match(r"[~>]?\s*([\d,]+)", c.reported_size_usd or "")
    if m:
        S = float(m.group(1).replace(",", ""))
        near = w[(w.usd >= 0.5 * S) & (w.usd <= 1.5 * S)]
        out.update(reported_size=S, n_wallets_near_reported_size=len(near),
                   n_near_size_age_lt7d=int((near.age_d < 7).sum()),
                   near_size_min_age_d=float(near.age_d.min()) if len(near) else np.nan,
                   near_size_vwap_range=(f"{near.vwap.min():.3f}-{near.vwap.max():.3f}" if len(near) else ""))
    return out


def run():
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    conn = connect()
    rows, skipped = _case_rows(conn)
    req = _requests(rows)
    feats = compute_features_batch(req, conn=conn)
    feats = pd.concat([feats, pattern_flags(feats)], axis=1)
    wsign = {r["market_id"]: (1.0 if r["win_idx"] == 0 else -1.0 if r["win_idx"] == 1 else np.nan) for r in rows}
    ws = feats.market_id.map(wsign)
    for k, base in TO_WINNER.items():
        feats[k] = feats[base] * ws
    feats = _signed(feats)
    for k in ("pat_strong", "pat_possible", "pat_loose"):
        feats[k] = feats[k].astype(bool)
    # reaction rows: keep the matching lookback only
    rm = feats.window.str.startswith("reaction")
    feats = feats[~rm | (feats.window == "reaction_" + feats.lookback_h.astype(str) + "h")]
    case_mids = sorted({r["market_id"] for r in rows})
    pool = _pool(conn, case_mids)
    pool_flags = {L: pattern_flags(pool, f"_{L}h") for L in LBS}

    info = {(r["case_id"], r["market_id"]): r for r in rows}
    out = []
    for (cid, mid), g in feats.groupby(["case_id", "market_id"], sort=False):
        r = info[(cid, mid)]
        for L in LBS:
            gl = g[g.lookback_h == L]
            pre = gl[gl.window == "pre"].iloc[0]
            base = gl[(gl.window == "baseline") & (gl.n_trades_total > 0)]
            reac = gl[gl.window.str.startswith("reaction")]
            # matched controls
            P = pool[[f"gv_{L}h"]].notna().to_numpy()[:, 0]
            pz = pool[P]
            dist = (((pz.px_last - pre.px_last) / 0.1) ** 2 + ((pz.log_gv_7d - pre.log_gv_7d) / 1.0) ** 2)
            mt = pz.loc[dist.nsmallest(50).index]
            for f in KEY:
                pv = mt[f"{f}_{L}h"].to_numpy(float)
                if f in DIRECTIONAL:
                    pv = pv * mt[f"dom_{L}h"].to_numpy(float)
                bv = base[f].to_numpy(float)
                out.append(dict(
                    case_id=cid, case_type=r["case_type"], market_id=mid, precision=r["precision"],
                    vindicated=r["vindicated"], pre_cutoff=pre.cutoff_ts, lookback_h=L, feature=f,
                    pre_value=pre[f], baseline_median=np.nanmedian(bv) if len(bv) else np.nan,
                    baseline_pct=_pctile(pre[f], bv), baseline_robust_z=_rz(pre[f], bv), n_baseline=len(bv),
                    matched_median=np.nanmedian(pv) if len(pv) else np.nan, matched_pct=_pctile(pre[f], pv),
                    n_matched=len(mt),
                    reaction_value=reac[f].iloc[0] if len(reac) else np.nan,
                    pre_pat_strong=bool(pre.pat_strong), pre_pat_possible=bool(pre.pat_possible),
                    pre_pat_loose=bool(pre.pat_loose),
                    pre_cond=("".join(k for k, v in zip("abcd", [pre.cond_a, pre.cond_b, pre.cond_c, pre.cond_d]) if v)),
                    baseline_rate_possible=float(base.pat_possible.mean()) if len(base) else np.nan,
                    baseline_rate_loose=float(base.pat_loose.mean()) if len(base) else np.nan,
                    reaction_pat_possible=bool(reac.pat_possible.iloc[0]) if len(reac) else None,
                    reaction_pat_loose=bool(reac.pat_loose.iloc[0]) if len(reac) else None,
                    pre_dom_is_winner=(None if r["win_idx"] not in (0, 1) or pre.dom == 0 else
                                       bool((pre.dom > 0) == (r["win_idx"] == 0))),
                    pool_rate_strong=float(pool_flags[L].pat_strong[P].mean()),
                    pool_rate_possible=float(pool_flags[L].pat_possible[P].mean()),
                    pool_rate_loose=float(pool_flags[L].pat_loose[P].mean()),
                ))
    comp = pd.DataFrame(out)
    comp.to_csv(f"{OUT}/case_feature_comparison.csv", index=False)
    claims = pd.DataFrame([x for x in [_claims(conn, r, e) for r in rows for e in (False, True)] if x])
    claims.to_csv(f"{OUT}/case_claim_checks.csv", index=False)
    pd.DataFrame(skipped, columns=["case_id", "case_type", "reason"]).to_csv(f"{OUT}/case_skipped.csv", index=False)
    # pool base rates by split and lookback
    br = []
    for L in LBS:
        P = pool[f"gv_{L}h"].notna()
        fl = pool_flags[L][P]
        for sp, idx in pool[P].groupby("split").groups.items():
            br.append(dict(lookback_h=L, split=sp, n=len(idx), **{k: float(fl.loc[idx, k].mean())
                      for k in ("cond_a", "cond_b", "cond_c", "cond_d", "pat_strong", "pat_possible", "pat_loose")}))
    pd.DataFrame(br).to_csv(f"{OUT}/pattern_base_rates.csv", index=False)
    _plots(conn, feats, comp, rows, pool, pool_flags)
    return comp, claims, skipped


def _plots(conn, feats, comp, rows, pool, pool_flags):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    # 1. percentile heatmap: pre window vs matched controls (24h lookback)
    sel = comp[comp.lookback_h == 24]
    feats_show = ["nsn", "os", "wl_z_nsn", "p_entry", "fws72", "fresh_nsn_to_winner", "nsn_to_winner",
                  "dom_newmkt_share", "top3_share",
                  "conc_top3", "n_new_wallets_dom", "max_cluster", "fresh_conc_share", "whale_hist_mixed_share"]
    lab = sel.case_id + " " + sel.market_id.astype(str)
    piv = sel.assign(lab=lab).pivot_table(index="lab", columns="feature", values="matched_pct", sort=False)[feats_show]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(piv) + 1.6))
    im = ax.imshow(piv.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_xticks(range(len(feats_show)), feats_show, rotation=40, ha="right")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if v == v:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else INK)
    ax.set_title("Pre-news 24h window: percentile vs 50 matched non-case in-scope snapshots", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.025, label="percentile")
    fig.tight_layout(); fig.savefig(f"{OUT}/case_percentiles_24h.png", dpi=130); plt.close(fig)

    # 2. 72h pre-news trajectories: YES price and cumulative signed notional (fresh vs other)
    show = [r for r in rows if r["precision"] in ("minute", "hour")]
    seen, sh = set(), []
    for r in show:
        if r["market_id"] not in seen:
            seen.add(r["market_id"]); sh.append(r)
    n = len(sh)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.0 * n + 0.6), squeeze=False)
    for i, r in enumerate(sh):
        end = pd.Timestamp(r["t_lo"]); start = end - timedelta(hours=72)
        t = pd.read_sql_query(
            """select t.ts, t.outcome_index, t.is_buy, t.price, t.size, h.first_t
               from pm_trades t join feat_wallet_hist h using (wallet)
               where t.market_id = %(m)s and t.role = 0 and t.ts >= %(a)s and t.ts < %(b)s order by t.ts""",
            conn, params={"m": r["market_id"], "a": start, "b": end + timedelta(hours=3)})
        if t.empty:
            continue
        t["h"] = (pd.to_datetime(t.ts, utc=True) - end).dt.total_seconds() / 3600
        d = np.where((t.outcome_index == 0) == t.is_buy, 1, -1)
        t["s"] = d * t["price"] * t["size"]
        t["y"] = np.where(t.outcome_index == 0, t.price, 1 - t.price)
        fresh = (_epoch(t.ts) - t.first_t) < 72 * 3600
        pre = t.h < 0
        a1, a2 = axes[i]
        a1.plot(t.h, t.y, color=BLUE, lw=1.2)
        a1.axvline(0, color=GRAY, lw=1, ls="--")
        a1.set_ylabel("YES price"); a1.set_xlim(-72, 3)
        a1.set_title(f"{r['case_id']} market {r['market_id']}: YES-eq price (news lower edge = 0h)",
                     loc="left", fontsize=8, color=INK)
        a2.plot(t.h[pre], (t.s * fresh)[pre].cumsum() / 1e3, color=ORANGE, lw=1.6, label="fresh wallets (<72h)")
        a2.plot(t.h[pre], (t.s * ~fresh)[pre].cumsum() / 1e3, color=BLUE, lw=1.6, label="other wallets")
        a2.axhline(0, color="#c9c8c3", lw=0.8)
        a2.set_ylabel("cum. signed $k"); a2.set_xlim(-72, 0)
        a2.set_title("cumulative YES-signed taker notional before news", loc="left", fontsize=8, color=INK)
        if i == 0:
            a2.legend(frameon=False, fontsize=7, loc="upper left")
    axes[-1][0].set_xlabel("hours relative to news"); axes[-1][1].set_xlabel("hours relative to news")
    fig.tight_layout(); fig.savefig(f"{OUT}/case_trajectories_72h.png", dpi=130); plt.close(fig)

    # 3. pattern rates: pool vs cases (pre) vs same-market baselines
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(LBS)); wdt = 0.26
    pool_rate = [pool_flags[L].pat_loose[pool[f"gv_{L}h"].notna()].mean() for L in LBS]
    pre_rate = [comp[(comp.lookback_h == L) & (comp.feature == "nsn")].pre_pat_loose.mean() for L in LBS]
    base_rate = [comp[(comp.lookback_h == L) & (comp.feature == "nsn")].baseline_rate_loose.mean() for L in LBS]
    for j, (v, c, lb) in enumerate(((pre_rate, ORANGE, "case pre-news windows"),
                                    (base_rate, GRAY, "same-market baselines (>=7d earlier)"),
                                    (pool_rate, BLUE, "all in-scope grid snapshots"))):
        ax.bar(x + (j - 1) * wdt, v, wdt - 0.02, color=c, label=lb)
        for xi, vi in zip(x, v):
            ax.text(xi + (j - 1) * wdt, vi + 0.005, f"{vi*100:.1f}%", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [f"{L}h" for L in LBS]); ax.set_ylabel("share of windows")
    ax.set_title("How often the (loose) informed-flow pattern fires", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/pattern_rates.png", dpi=130); plt.close(fig)


def main():
    comp, claims, skipped = run()
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_rows", 400):
        print("skipped:", skipped)
        print(comp[comp.feature.isin(["nsn", "fws72", "os", "p_entry"])][
            ["case_id", "market_id", "lookback_h", "feature", "pre_value", "baseline_median", "baseline_pct",
             "matched_median", "matched_pct", "pre_cond", "pre_pat_loose", "pre_dom_is_winner"]].to_string(index=False))
        print(claims.T.to_string())


if __name__ == "__main__":
    main()
