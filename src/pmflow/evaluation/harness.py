"""Reusable comparison harness: model predictions vs Baseline A (market price) on the fixed splits.

Usage (CLI)::

    python -m pmflow.evaluation.harness --pred artifacts/modeling/preds.parquet --name my_model
    python -m pmflow.evaluation.harness --pred table:mdl_predictions --name my_model [--flag-col flag]
    python -m pmflow.evaluation.harness --pred preds.parquet --name m --trade --edge 0.05

Input: (snapshot_id, p_model[, flag]) for any subset of ``eval_snapshots``. Only snapshots in the primary
set (is_live, split in train/val/test) are scored; rows of purged/non-live snapshots are reported and ignored.
Snapshots without a prediction are reported as coverage (scoring is on the intersection; Baseline A is
re-scored on exactly the same rows, so deltas are always paired).

Python::

    from pmflow.evaluation.harness import load_snapshots, compare
    snaps = load_snapshots()
    report = compare(pred_df, snaps)   # dict of DataFrames
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from . import metrics as M

DSN = os.environ.get("PMFLOW_DSN", "postgresql://pmflow:pmflow@localhost:5432/pmflow")
REPO = Path(__file__).resolve().parents[3]
ART = REPO / "artifacts" / "evaluation"
SPLITS = ["train", "val", "test"]
UNIVERSES = ["in_scope", "macro"]
SNAP_COLS = ("snapshot_id, market_id, event_id, universe, geo_subtype, neg_risk, y, cutoff_ts, grid_24h, "
             "price_yes, vwap_6h, vwap_24h, last_trade_age_h, n_trades_24h, usd_24h, n_trades_7d, usd_7d, "
             "hours_since_open, hours_to_close_diag, split, is_live")


def connect():
    return psycopg.connect(DSN, autocommit=True)


def load_snapshots(where: str = "is_live AND split IN ('train','val','test')", cols: str = SNAP_COLS) -> pd.DataFrame:
    with connect() as c:
        cur = c.execute(f"SELECT {cols} FROM eval_snapshots WHERE {where}")
        df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    for k in ("price_yes", "vwap_6h", "vwap_24h", "last_trade_age_h", "usd_24h", "usd_7d",
              "hours_since_open", "hours_to_close_diag"):
        if k in df:
            df[k] = df[k].astype(float)
    df["cutoff_ts"] = pd.to_datetime(df["cutoff_ts"], utc=True)
    return df


def load_predictions(src: str) -> pd.DataFrame:
    if src.startswith("table:") or src.startswith("sql:"):
        q = f"SELECT * FROM {src[6:]}" if src.startswith("table:") else src[4:]
        with connect() as c:
            cur = c.execute(q)
            return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])
    if src.endswith(".csv"):
        return pd.read_csv(src)
    return pd.read_parquet(src)


# ------------------------------------------------------------------ core comparison


def _block(d: pd.DataFrame, pcol: str, bcol: str, weight: str, B: int, seed: int) -> dict:
    y = d["y"].to_numpy(float)
    w = M.snapshot_weights(d, weight)
    g = d["event_id"].to_numpy()
    pm, pb = d[pcol].to_numpy(float), d[bcol].to_numpy(float)
    out = dict(n_snap=len(d), n_mkt=d.market_id.nunique(), n_evt=d.event_id.nunique(),
               base_rate=float(np.average(y, weights=w)))
    for name, p in (("model", pm), ("base", pb)):
        s = M.summarize(y, p, w)
        for k in ("brier", "log_loss", "ece", "calib_intercept", "calib_slope"):
            out[f"{k}_{name}"] = s[k]
    tb = M.paired_test(M.brier_terms(y, pb), M.brier_terms(y, pm), g, w, B=B, seed=seed)
    tl = M.paired_test(M.logloss_terms(y, pb), M.logloss_terms(y, pm), g, w, B=B, seed=seed + 1)
    out.update(d_brier=tb["delta"], d_brier_lo=tb["ci_lo"], d_brier_hi=tb["ci_hi"], d_brier_p=tb["p_cluster_t"],
               d_brier_pboot=tb["p_boot"], d_logloss=tl["delta"], d_logloss_lo=tl["ci_lo"], d_logloss_hi=tl["ci_hi"],
               d_logloss_p=tl["p_cluster_t"])
    # skill relative to price (positive = better than price)
    out["brier_skill_vs_price"] = 1 - out["brier_model"] / out["brier_base"]
    return out


def compare(pred: pd.DataFrame, snaps: pd.DataFrame | None = None, pcol: str = "p_model",
            bcol: str = "price_yes", weights=("snapshot", "market", "event"), B: int = 2000, seed: int = 0,
            flag_col: str | None = None, splits=SPLITS) -> dict[str, pd.DataFrame]:
    """Compare predictions with Baseline A. Returns dict of DataFrames:
    'coverage', 'summary' (universe x split x weighting), 'by_year', 'by_subtype', 'reliability', 'flags'."""
    if snaps is None:
        snaps = load_snapshots()
    pred = pred[["snapshot_id", pcol] + ([flag_col] if flag_col else [])].copy()
    if pred.snapshot_id.duplicated().any():
        raise ValueError("duplicate snapshot_id in predictions")
    if pred[pcol].isna().any() or (pred[pcol] < 0).any() or (pred[pcol] > 1).any():
        raise ValueError(f"{pcol} must be in [0,1] with no NaN")
    d = snaps.merge(pred, on="snapshot_id", how="inner")
    if bcol not in d:
        raise KeyError(bcol)
    cov = (snaps.assign(has_pred=snaps.snapshot_id.isin(pred.snapshot_id))
           .groupby(["universe", "split"]).has_pred.agg(["size", "sum", "mean"]).reset_index()
           .rename(columns={"size": "n_primary", "sum": "n_pred", "mean": "coverage"}))
    cov["n_pred_outside_primary"] = int((~pred.snapshot_id.isin(snaps.snapshot_id)).sum())
    rows, yrs, subs, rel = [], [], [], []
    for u in UNIVERSES:
        for s in splits:
            ds = d[(d.universe == u) & (d.split == s)]
            if len(ds) == 0:
                continue
            for wname in weights:
                rows.append(dict(universe=u, split=s, weighting=wname, **_block(ds, pcol, bcol, wname, B, seed)))
            for yr, dy in ds.groupby(ds.cutoff_ts.dt.year):
                yrs.append(dict(universe=u, split=s, year=yr, **_block(dy, pcol, bcol, "snapshot", 500, seed)))
            for st, dg in ds.groupby(ds.geo_subtype.fillna("none")):
                if dg.event_id.nunique() >= 5:
                    subs.append(dict(universe=u, split=s, geo_subtype=st, **_block(dg, pcol, bcol, "snapshot", 500, seed)))
            for name, col in (("model", pcol), ("base", bcol)):
                t = M.reliability_table(ds.y, ds[col])
                t.insert(0, "forecast", name); t.insert(0, "split", s); t.insert(0, "universe", u)
                rel.append(t)
    out = dict(coverage=cov, summary=pd.DataFrame(rows), by_year=pd.DataFrame(yrs),
               by_subtype=pd.DataFrame(subs), reliability=pd.concat(rel, ignore_index=True) if rel else pd.DataFrame())
    out["flags"] = flag_precision(d, pcol, bcol, flag_col)
    return out


def flag_precision(d: pd.DataFrame, pcol="p_model", bcol="price_yes", flag_col=None,
                   edges=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """High-confidence flags. If flag_col is given (values +1 = 'YES underpriced', -1 = 'NO underpriced',
    0 = none) use it; otherwise flag |p_model - price| >= edge for each edge. For flagged rows report
    precision (share resolving in the flagged direction) vs the price-implied expectation of that share
    (mean price of the flagged side): precision > implied means the flag beats the market."""
    rows = []
    specs = [("flag_col", None)] if flag_col else [(f"edge>={e}", e) for e in edges]
    for u in UNIVERSES:
        for s in SPLITS:
            ds = d[(d.universe == u) & (d.split == s)]
            for name, e in specs:
                direction = (np.sign(ds[flag_col]) if flag_col else
                             np.where(ds[pcol] - ds[bcol] >= e, 1, np.where(ds[bcol] - ds[pcol] >= e, -1, 0)))
                direction = np.asarray(direction)
                m = direction != 0
                if m.sum() == 0:
                    rows.append(dict(universe=u, split=s, flag=name, n_flag=0)); continue
                f = ds[m]
                dr = direction[m]
                hit = np.where(dr > 0, f.y, 1 - f.y).astype(float)
                implied = np.where(dr > 0, f[bcol], 1 - f[bcol]).astype(float)
                g = f.event_id.to_numpy()
                pt, lo, hi = M.clustered_mean_ci(hit - implied, g, B=500)
                rows.append(dict(universe=u, split=s, flag=name, n_flag=int(m.sum()), n_evt=f.event_id.nunique(),
                                 n_mkt=f.market_id.nunique(), share_yes_flags=float(np.mean(dr > 0)),
                                 precision=float(hit.mean()), implied=float(implied.mean()),
                                 excess=pt, excess_lo=lo, excess_hi=hi))
    return pd.DataFrame(rows)


def write_report(rep: dict[str, pd.DataFrame], name: str, outdir: Path | None = None) -> Path:
    outdir = outdir or (ART / "runs" / name)
    outdir.mkdir(parents=True, exist_ok=True)
    for k, v in rep.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(outdir / f"{k}.csv", index=False)
    s = rep["summary"]
    cols = ["universe", "split", "weighting", "n_snap", "n_evt", "brier_base", "brier_model", "d_brier",
            "d_brier_lo", "d_brier_hi", "d_brier_p", "log_loss_base", "log_loss_model", "d_logloss",
            "d_logloss_lo", "d_logloss_hi", "ece_base", "ece_model"]
    txt = [f"# Harness report: {name}", "", "Delta = baseline loss - model loss (positive = model better). "
           "CIs: event-clustered bootstrap 95%; p: cluster-robust t over events.", "",
           s[cols].round(5).to_markdown(index=False) if hasattr(s, "to_markdown") else s[cols].round(5).to_string()]
    (outdir / "summary.md").write_text("\n".join(txt))
    return outdir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="parquet/csv path, table:<name> or sql:<query>")
    ap.add_argument("--name", required=True)
    ap.add_argument("--pcol", default="p_model")
    ap.add_argument("--flag-col", default=None)
    ap.add_argument("--baseline", default="price_yes", help="baseline column of eval_snapshots")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--trade", action="store_true", help="also run the tradability simulator")
    ap.add_argument("--edge", type=float, default=0.05, help="trade when |p_model - price| >= edge")
    ap.add_argument("--stake", type=float, default=100.0)
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--impact-coef", type=float, default=0.05)
    ap.add_argument("--latency-s", type=float, default=60.0)
    ap.add_argument("--dedupe", default="first_per_market", choices=["first_per_market", "all"])
    a = ap.parse_args()
    pred = load_predictions(a.pred)
    snaps = load_snapshots()
    rep = compare(pred, snaps, pcol=a.pcol, bcol=a.baseline, B=a.B, flag_col=a.flag_col,
                  splits=a.splits.split(","))
    if a.trade:
        from .tradability import TradeParams, simulate, summarize_trades
        d = snaps.merge(pred, on="snapshot_id")
        d = d[d.split.isin(a.splits.split(","))]
        sig = d.assign(direction=np.where(d[a.pcol] - d.price_yes >= a.edge, 1,
                                          np.where(d.price_yes - d[a.pcol] >= a.edge, -1, 0)))
        sig = sig[sig.direction != 0]
        tp = TradeParams(stake_usd=a.stake, fee_bps=a.fee_bps, impact_coef=a.impact_coef,
                         latency_s=a.latency_s, dedupe=a.dedupe)
        trades = simulate(sig[["snapshot_id", "direction"]], tp)
        rep["trades_summary"] = summarize_trades(trades)
        rep["trades"] = trades
    out = write_report(rep, a.name)
    print((out / "summary.md").read_text())
    if "trades_summary" in rep:
        print(rep["trades_summary"].to_string())
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
