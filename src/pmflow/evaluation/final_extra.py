"""Final-phase extras (evaluation role): informed-flow score / weak-pattern flags on test, and the reported-case
check. Run after final_score.

    python -m pmflow.evaluation.final_extra

* pat_* (feat_weak_labels, outcome-free pattern flags): for each pattern x lookback, the dominant side's hit
  rate vs (a) its price at the cutoff, (b) its p_entry, (c) the price-only control's probability; counts of
  snapshots / markets / events; event-clustered CIs. train/val shown for context, test is the held-out read.
* s1_informed_score (stack_s1 file): thresholds at the TRAIN 99th/95th percentile (fixed from train only); for
  test snapshots above them, the dominant 24h side's hit rate vs its price.
* Reported cases: seed-case markets with scored val/test snapshots before the public-news timestamp: price and each
  candidate at each such snapshot (anecdotal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .final_audit import OUT, PRED_FILES, q
from .harness import ART, REPO

LOOK = (1, 6, 24, 72)


def _excess(hit, ref, groups, B=2000):
    if len(hit) == 0:
        return (np.nan, np.nan, np.nan)
    return M.clustered_mean_ci(np.asarray(hit, float) - np.asarray(ref, float), groups, B=B)


def patterns() -> pd.DataFrame:
    cols = ", ".join(f"w.pat_{p}_{L}h, w.dom_{L}h, w.p_entry_{L}h" for p in ("strong", "possible", "loose") for L in LOOK)
    d = q(f"SELECT w.snapshot_id, s.market_id, s.event_id, s.universe, s.split, s.y, s.price_yes, {cols} "
          "FROM feat_weak_labels w JOIN eval_snapshots s USING (snapshot_id) "
          "WHERE s.is_live AND s.split IN ('train','val','test')")
    d = d.loc[:, ~d.columns.duplicated()]
    d["price_yes"] = d.price_yes.astype(float)
    ctrl = pd.read_parquet(ART.parent / "modeling" / PRED_FILES["ctrl_lgbm_price_only"], columns=["snapshot_id", "p_model"])
    d = d.merge(ctrl.rename(columns={"p_model": "p_ctrl"}), on="snapshot_id", how="left")
    rows = []
    for u in ("in_scope", "macro"):
        for sp in ("train", "val", "test"):
            ds = d[(d.universe == u) & (d.split == sp)]
            for p in ("strong", "possible", "loose"):
                for L in list(LOOK) + ["any"]:
                    if L == "any":
                        m = np.zeros(len(ds), bool)
                        dom = np.zeros(len(ds))
                        pe = np.full(len(ds), np.nan)
                        for LL in reversed(LOOK):          # shortest lookback that fires sets the direction
                            f = ds[f"pat_{p}_{LL}h"].fillna(False).astype(bool).to_numpy()
                            dom = np.where(f, ds[f"dom_{LL}h"].to_numpy(float), dom)
                            pe = np.where(f, ds[f"p_entry_{LL}h"].astype(float).to_numpy(), pe)
                            m |= f
                    else:
                        m = ds[f"pat_{p}_{L}h"].fillna(False).astype(bool).to_numpy()
                        dom = ds[f"dom_{L}h"].to_numpy(float)
                        pe = ds[f"p_entry_{L}h"].astype(float).to_numpy()
                    m = np.asarray(m, bool) & (np.nan_to_num(dom) != 0)
                    f = ds[m]
                    dm = dom[m]
                    hit = np.where(dm > 0, f.y, 1 - f.y)
                    px = np.where(dm > 0, f.price_yes, 1 - f.price_yes)
                    pc = np.where(dm > 0, f.p_ctrl, 1 - f.p_ctrl)
                    g = f.event_id.to_numpy()
                    e1, e2, e3 = _excess(hit, px, g), _excess(hit, pe[m], g), _excess(hit, pc, g)
                    rows.append(dict(universe=u, split=sp, pattern=p, lookback=L, n_snap=int(m.sum()),
                                     n_mkt=f.market_id.nunique(), n_evt=f.event_id.nunique(),
                                     share_yes=float(np.mean(dm > 0)) if m.sum() else np.nan,
                                     hit_rate=float(np.mean(hit)) if m.sum() else np.nan,
                                     mean_price_dom=float(np.mean(px)) if m.sum() else np.nan,
                                     mean_p_entry=float(np.nanmean(pe[m])) if m.sum() else np.nan,
                                     excess_vs_price=e1[0], lo=e1[1], hi=e1[2],
                                     excess_vs_p_entry=e2[0], lo_pe=e2[1], hi_pe=e2[2],
                                     excess_vs_ctrl=e3[0], lo_ctrl=e3[1], hi_ctrl=e3[2]))
    return pd.DataFrame(rows)


def s1_score() -> pd.DataFrame:
    st = pd.read_parquet(ART.parent / "modeling" / PRED_FILES["stack_s1"],
                         columns=["snapshot_id", "split", "universe", "s1_informed_score"])
    d = q("SELECT s.snapshot_id, s.market_id, s.event_id, s.y, s.price_yes, w.dom_24h, w.pat_loose_24h "
          "FROM eval_snapshots s JOIN feat_weak_labels w USING (snapshot_id) WHERE s.is_live AND s.split IN ('train','val','test')")
    d = d.loc[:, ~d.columns.duplicated()].merge(st, on="snapshot_id")
    d["price_yes"] = d.price_yes.astype(float)
    tr = d[d.split == "train"].s1_informed_score
    rows = []
    for qq in (0.95, 0.99):
        th = float(tr.quantile(qq))
        for u in ("in_scope", "macro"):
            for sp in ("train", "val", "test"):
                ds = d[(d.universe == u) & (d.split == sp)]
                m = (ds.s1_informed_score >= th).to_numpy() & (ds.dom_24h.to_numpy(float) != 0)
                f = ds[m]
                dm = f.dom_24h.to_numpy(float)
                hit = np.where(dm > 0, f.y, 1 - f.y)
                px = np.where(dm > 0, f.price_yes, 1 - f.price_yes)
                e = _excess(hit, px, f.event_id.to_numpy())
                rows.append(dict(train_quantile=qq, threshold=th, universe=u, split=sp, n_snap=int(m.sum()),
                                 n_mkt=f.market_id.nunique(), n_evt=f.event_id.nunique(),
                                 share_with_pat_loose_24h=float(f.pat_loose_24h.fillna(False).mean()) if len(f) else np.nan,
                                 hit_rate=float(np.mean(hit)) if len(f) else np.nan,
                                 mean_price_dom=float(np.mean(px)) if len(f) else np.nan,
                                 excess_vs_price=e[0], lo=e[1], hi=e[2]))
    return pd.DataFrame(rows)


def cases() -> pd.DataFrame:
    cs = pd.read_csv(REPO / "artifacts" / "labeling" / "reported_cases.csv")
    mt = pd.read_csv(REPO / "artifacts" / "features" / "case_market_matches.csv")
    mt = mt.dropna(subset=["pg_market_ids"]).assign(market_id=lambda x: x.pg_market_ids.astype(str).str.split(";"))
    mt = mt.explode("market_id")
    mt["market_id"] = mt.market_id.astype(int)
    cs = cs.merge(mt[["case_id", "market_id"]], on="case_id")
    cs["news"] = pd.to_datetime(cs.public_news_ts_utc, utc=True, errors="coerce")
    snaps = q("SELECT snapshot_id, market_id, event_id, split, cutoff_ts, price_yes, y FROM eval_snapshots "
              "WHERE is_live AND grid_24h AND split IN ('val','test') AND market_id = ANY(%s)", [cs.market_id.tolist()])
    snaps["cutoff_ts"] = pd.to_datetime(snaps.cutoff_ts, utc=True)
    out = snaps.merge(cs[["case_id", "case_type", "market_id", "news", "public_news_ts_utc", "ts_precision"]],
                      on="market_id")
    out = out[out.cutoff_ts < out.news]
    for k, f in PRED_FILES.items():
        p = pd.read_parquet(ART.parent / "modeling" / f, columns=["snapshot_id", "p_model", "flag"])
        out = out.merge(p.rename(columns={"p_model": k, "flag": f"flag_{k}"}), on="snapshot_id", how="left")
    out["hours_before_news"] = (out.news - out.cutoff_ts).dt.total_seconds() / 3600
    return out.sort_values(["case_id", "cutoff_ts"])


def main():
    pt = patterns()
    pt.to_csv(OUT / "extra_patterns.csv", index=False)
    s1 = s1_score()
    s1.to_csv(OUT / "extra_s1_informed_score.csv", index=False)
    cs = cases()
    cs.to_csv(OUT / "extra_cases.csv", index=False)
    pd.set_option("display.width", 250)
    print(pt[(pt.split == "test") & (pt.lookback.astype(str) == "any")].round(3).to_string())
    print(s1.round(3).to_string())
    print(cs.round(3).to_string())


if __name__ == "__main__":
    main()
