"""Baseline A: market price alone as P(YES), plus reference predictors and sensitivity analyses.

Run: ``python -m pmflow.evaluation.baseline`` -> CSVs + figures in artifacts/evaluation/, and
``eval_baseline_preds`` (snapshot_id, p_price, p_vwap24, p_clim, p_platt, p_iso) for reuse.
"""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression

from . import metrics as M
from .harness import ART, SPLITS, UNIVERSES, connect, load_snapshots

log = logging.getLogger("pmflow.evaluation.baseline")

PRICE_EDGES = [0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97 + 1e-9]
TTC_EDGES_H = [0, 24, 72, 168, 720, 2160, np.inf]
TTC_LABELS = ["<1d", "1-3d", "3-7d", "7-30d", "30-90d", ">90d"]
FORECASTS = ["p_price", "p_clim", "p_platt", "p_iso", "p_vwap24"]


# ---------------------------------------------------------------- reference predictors


def fit_isotonic(y, p, w=None):
    o = np.argsort(p)
    ps, ys = np.asarray(p)[o], np.asarray(y, float)[o]
    ws = np.ones(len(ps)) if w is None else np.asarray(w)[o]
    fit = isotonic_regression(ys, weights=ws, increasing=True).x
    # collapse ties to knots, clip away exact 0/1
    knots_x, idx = np.unique(ps, return_index=True)
    knots_y = np.clip(fit[idx], 0.005, 0.995)
    return knots_x, knots_y


def add_reference_forecasts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit on train (per universe) only; apply everywhere."""
    df = df.copy()
    df["p_price"] = df.price_yes
    df["p_vwap24"] = df.vwap_24h.fillna(df.price_yes).clip(0.001, 0.999)
    params = {}
    for u in UNIVERSES:
        m = df.universe == u
        tr = m & (df.split == "train")
        br = float(df.loc[tr, "y"].mean())
        ab = M.fit_platt(df.loc[tr, "y"], df.loc[tr, "price_yes"])
        kx, ky = fit_isotonic(df.loc[tr, "y"].to_numpy(), df.loc[tr, "price_yes"].to_numpy())
        df.loc[m, "p_clim"] = br
        df.loc[m, "p_platt"] = M.apply_platt(df.loc[m, "price_yes"], ab)
        df.loc[m, "p_iso"] = np.interp(df.loc[m, "price_yes"], kx, ky)
        params[u] = dict(train_base_rate=br, platt_a=ab[0], platt_b=ab[1], iso_knots=len(kx))
    return df, params


# ---------------------------------------------------------------- scoring tables


def score(d: pd.DataFrame, fc: str, weight="snapshot", B=1000, seed=0) -> dict:
    y = d.y.to_numpy(float)
    p = d[fc].to_numpy(float)
    w = M.snapshot_weights(d, weight)
    g = d.event_id.to_numpy()
    bpt, blo, bhi, _ = M.bootstrap_ratio(g, w * M.brier_terms(y, p), w, B, seed)
    lpt, llo, lhi, _ = M.bootstrap_ratio(g, w * M.logloss_terms(y, p), w, B, seed)
    s = M.summarize(y, p, w)
    return dict(n_snap=len(d), n_mkt=d.market_id.nunique(), n_evt=d.event_id.nunique(), base_rate=s["base_rate"],
                mean_p=s["mean_p"], brier=bpt, brier_lo=blo, brier_hi=bhi, log_loss=lpt, log_loss_lo=llo,
                log_loss_hi=lhi, ece=s["ece"], calib_intercept=s["calib_intercept"], calib_slope=s["calib_slope"])


def grouped(df, by, fcs=("p_price",), weight="snapshot", B=500, min_evt=3):
    rows = []
    for key, d in df.groupby(by, observed=True):
        if d.event_id.nunique() < min_evt:
            continue
        key = key if isinstance(key, tuple) else (key,)
        for fc in fcs:
            rows.append(dict(zip(by, key), forecast=fc, **score(d, fc, weight, B)))
    return pd.DataFrame(rows)


def paired(df, by, base="p_price", alts=("p_platt", "p_iso", "p_vwap24", "p_clim"), weight="snapshot", B=1000):
    rows = []
    for key, d in df.groupby(by, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        y = d.y.to_numpy(float)
        w = M.snapshot_weights(d, weight)
        g = d.event_id.to_numpy()
        for alt in alts:
            for lname, lf in (("brier", M.brier_terms), ("log_loss", M.logloss_terms)):
                r = M.paired_test(lf(y, d[base]), lf(y, d[alt]), g, w, B=B)
                rows.append(dict(zip(by, key), weighting=weight, alt=alt, loss=lname, **r))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures


def reliability_figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    cols = {"p_price": ("#2a78d6", "raw price (Baseline A)"), "p_platt": ("#eb6834", "Platt-recalibrated price")}
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for i, u in enumerate(UNIVERSES):
        for j, s in enumerate(SPLITS):
            ax = axes[i, j]
            d = df[(df.universe == u) & (df.split == s)]
            ax.plot([0, 1], [0, 1], color=ink2, lw=1, ls="--", zorder=1)
            for fc, (c, lab) in cols.items():
                t = M.reliability_table(d.y, d[fc], bins=10)
                t = t[t.n > 0]
                ax.plot(t.mean_p, t.freq_y, color=c, lw=2, marker="o", ms=5, label=lab, zorder=3,
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5)
            ax.set_title(f"{u} / {s}  (n={len(d):,}, events={d.event_id.nunique():,})", fontsize=10, color=ink)
            ax.grid(color=grid, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=ink2, labelsize=8)
            if j == 0:
                ax.set_ylabel("observed YES frequency", color=ink2)
            if i == 1:
                ax.set_xlabel("forecast P(YES), 10 equal-width bins", color=ink2)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Reliability of market price vs train-fitted recalibration (primary live snapshots)", color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------- sensitivity


def filter_sensitivity(conn_where="split IN ('train','val','test')") -> pd.DataFrame:
    allc = load_snapshots(where=conn_where)
    variants = {
        "primary (band .03-.97, age<=72h, n7d>=3, usd7d>=200)": allc.is_live,
        "band .01-.99, primary activity": (allc.price_yes.between(.01, .99) & (allc.last_trade_age_h <= 72)
                                            & (allc.n_trades_7d >= 3) & (allc.usd_7d >= 200)),
        "band .05-.95, primary activity": allc.is_live & allc.price_yes.between(.05, .95),
        "band .10-.90, primary activity": allc.is_live & allc.price_yes.between(.10, .90),
        "band .03-.97, no activity filter": allc.price_yes.between(.03, .97),
        "band .03-.97, strict activity (age<=24h, n7d>=20, usd7d>=5k)": (allc.is_live & (allc.last_trade_age_h <= 24)
                                                                       & (allc.n_trades_7d >= 20) & (allc.usd_7d >= 5000)),
        "primary, 24h grid only": allc.is_live & allc.grid_24h,
        "primary, excl. last 24h before close (diag)": allc.is_live & (allc.hours_to_close_diag > 24),
        "primary, excl. last 7d before close (diag)": allc.is_live & (allc.hours_to_close_diag > 168),
    }
    rows = []
    for name, mask in variants.items():
        d = allc[mask]
        d, prm = add_reference_forecasts(d)
        for (u, s), ds in d.groupby(["universe", "split"]):
            r = score(ds, "p_price", "snapshot", B=300)
            y = ds.y.to_numpy(float)
            pt = M.paired_test(M.brier_terms(y, ds.p_price), M.brier_terms(y, ds.p_platt), ds.event_id.to_numpy(), B=300)
            rows.append(dict(variant=name, universe=u, split=s, **{k: r[k] for k in
                        ("n_snap", "n_mkt", "n_evt", "base_rate", "brier", "brier_lo", "brier_hi", "log_loss", "ece", "calib_slope")},
                        platt_gain=pt["delta"], platt_gain_lo=pt["ci_lo"], platt_gain_hi=pt["ci_hi"]))
    # price definition variants on the primary set
    d = allc[allc.is_live].copy()
    d["vwap_6h_f"] = d.vwap_6h.fillna(d.price_yes)
    d["vwap_24h_f"] = d.vwap_24h.fillna(d.price_yes)
    for fc in ("vwap_6h_f", "vwap_24h_f"):
        for (u, s), ds in d.groupby(["universe", "split"]):
            r = score(ds, fc, "snapshot", B=300)
            rows.append(dict(variant=f"primary, price = {fc}", universe=u, split=s, **{k: r[k] for k in
                        ("n_snap", "n_mkt", "n_evt", "base_rate", "brier", "brier_lo", "brier_hi", "log_loss", "ece", "calib_slope")}))
    return pd.DataFrame(rows)


def diag_time_to_close() -> pd.DataFrame:
    with connect() as c:
        cur = c.execute("SELECT * FROM eval_diag_snapshots WHERE price_yes IS NOT NULL")
        d = pd.DataFrame(cur.fetchall(), columns=[x.name for x in cur.description])
    d["price_yes"] = d.price_yes.astype(float)
    d["p_price"] = d.price_yes
    rows = []
    for (a, h, u, s), ds in d.groupby(["anchor", "hours_before", "universe", "event_split"]):
        rows.append(dict(anchor=a, hours_before=h, universe=u, event_split=s, **score(ds, "p_price", "snapshot", B=300),
                         share_price_outside_03_97=float((~ds.price_yes.between(.03, .97)).mean())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- MDE


def mde_table(df: pd.DataFrame) -> pd.DataFrame:
    """Minimum detectable mean Brier improvement (alpha=.05 two-sided, power=.8), from the event-clustered SE of
    paired deltas between price and real alternative forecasts (proxies for 'a model that moves prices by about
    this much'). Also an analytic 'sigma-scaling' MDE from the per-event dispersion of the price Brier terms."""
    rows = []
    for (u, s), d in df.groupby(["universe", "split"]):
        y = d.y.to_numpy(float)
        g = d.event_id.to_numpy()
        for alt in ("p_platt", "p_iso", "p_vwap24"):
            for wname in ("snapshot", "event"):
                w = M.snapshot_weights(d, wname)
                se = M.paired_test(M.brier_terms(y, d.p_price), M.brier_terms(y, d[alt]), g, w, B=100)["se_cluster"]
                rows.append(dict(universe=u, split=s, proxy=alt, weighting=wname, n_evt=d.event_id.nunique(),
                                 mean_abs_move=float(np.average(np.abs(d[alt] - d.p_price), weights=w)),
                                 se_delta=se, mde_brier=2.8016 * se,
                                 brier_price=float(np.average(M.brier_terms(y, d.p_price), weights=w)),
                                 mde_rel=2.8016 * se / float(np.average(M.brier_terms(y, d.p_price), weights=w))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load_snapshots()
    df, params = add_reference_forecasts(df)
    (ART / "baseline_params.json").write_text(json.dumps(params, indent=2))
    log.info("loaded %d primary snapshots (%.0fs)", len(df), time.time() - t0)

    # split counts table
    cnt = (df.groupby(["universe", "split"]).agg(n_snap=("snapshot_id", "size"), n_mkt=("market_id", "nunique"),
                                                  n_evt=("event_id", "nunique"), base_rate_snap=("y", "mean"),
                                                  mean_price=("price_yes", "mean"),
                                                  first_cutoff=("cutoff_ts", "min"), last_cutoff=("cutoff_ts", "max"))
           .reset_index())
    mk = df.drop_duplicates("market_id").groupby(["universe", "split"]).y.mean().rename("base_rate_market")
    cnt = cnt.merge(mk.reset_index(), on=["universe", "split"])
    cnt.to_csv(ART / "split_counts.csv", index=False)

    # headline
    head = []
    for wname in ("snapshot", "market", "event"):
        for (u, s), d in df.groupby(["universe", "split"]):
            for fc in FORECASTS:
                head.append(dict(universe=u, split=s, weighting=wname, forecast=fc, **score(d, fc, wname, B=2000)))
    head = pd.DataFrame(head)
    head.to_csv(ART / "baseline_headline.csv", index=False)
    log.info("headline done (%.0fs)", time.time() - t0)

    pr = pd.concat([paired(df, ["universe", "split"], weight=w, B=2000) for w in ("snapshot", "market", "event")])
    pr.to_csv(ART / "baseline_paired_vs_price.csv", index=False)

    df["year"] = df.cutoff_ts.dt.year
    df["price_bucket"] = pd.cut(df.price_yes, PRICE_EDGES, right=False)
    df["ttc_bucket_diag"] = pd.cut(df.hours_to_close_diag, TTC_EDGES_H, labels=TTC_LABELS, right=False)
    df["geo_subtype"] = df.geo_subtype.fillna("none")
    grouped(df, ["universe", "year"], ("p_price", "p_platt")).to_csv(ART / "baseline_by_year.csv", index=False)
    grouped(df, ["universe", "split", "geo_subtype"], ("p_price", "p_platt")).to_csv(ART / "baseline_by_subtype.csv", index=False)
    grouped(df, ["universe", "split", "price_bucket"], ("p_price", "p_platt")).to_csv(ART / "baseline_by_price_bucket.csv", index=False)
    grouped(df, ["universe", "split", "ttc_bucket_diag"], ("p_price", "p_platt")).to_csv(ART / "baseline_by_time_to_close_DIAG.csv", index=False)
    grouped(df, ["universe", "split", "neg_risk"], ("p_price", "p_platt")).to_csv(ART / "baseline_by_negrisk.csv", index=False)
    log.info("group tables done (%.0fs)", time.time() - t0)

    rel = []
    for (u, s), d in df.groupby(["universe", "split"]):
        for fc in ("p_price", "p_platt"):
            t = M.reliability_table(d.y, d[fc], bins=10)
            t.insert(0, "forecast", fc); t.insert(0, "split", s); t.insert(0, "universe", u)
            rel.append(t)
    pd.concat(rel).to_csv(ART / "baseline_reliability.csv", index=False)
    reliability_figure(df, ART / "reliability_price_vs_platt.png")

    mde_table(df).to_csv(ART / "mde.csv", index=False)
    log.info("mde done (%.0fs)", time.time() - t0)
    diag_time_to_close().to_csv(ART / "baseline_diag_anchored_DIAG.csv", index=False)
    filter_sensitivity().to_csv(ART / "baseline_filter_sensitivity.csv", index=False)
    log.info("sensitivity done (%.0fs)", time.time() - t0)

    # persist reference predictions for other agents (small: ~650k rows x 6 cols)
    out = df[["snapshot_id", "p_price", "p_vwap24", "p_clim", "p_platt", "p_iso"]]
    with connect() as c:
        c.execute("DROP TABLE IF EXISTS eval_baseline_preds")
        c.execute("CREATE TABLE eval_baseline_preds (snapshot_id bigint PRIMARY KEY, p_price real, p_vwap24 real, "
                  "p_clim real, p_platt real, p_iso real)")
        with c.cursor().copy("COPY eval_baseline_preds FROM STDIN") as cp:
            for r in out.itertuples(index=False):
                cp.write_row(tuple(float(x) if i else int(x) for i, x in enumerate(r)))
        c.execute("COMMENT ON TABLE eval_baseline_preds IS 'Baseline A and reference forecasts per primary snapshot; "
                  "p_clim/p_platt/p_iso fitted on train split only, per universe.'")
    log.info("all done (%.0fs)", time.time() - t0)


if __name__ == "__main__":
    main()
