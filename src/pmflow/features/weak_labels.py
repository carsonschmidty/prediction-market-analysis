"""Weak-label window quantities and labels (docs/labeling_plan.md §1-2) for eval snapshots.

Window quantities per lookback L (already in feat_snapshot_features, see core.py):
    NSN = nsn_L, GV = gv_L, OS = os_L, p_entry = p_entry_L, FWS = fws72_L, TOP3 = top3_share_L,
    CONC = conc_top3_L (notional-weighted CONC of the top-3 dominant-side wallets),
    z = wl_z_nsn_L (robust z of NSN vs <= 20 same-length windows ending >= 7 days before cutoff).

Pattern conditions (no outcome information; usable as features/scores):
    (a) dom*z >= 4 and |NSN| >= 5 x median|NSN_baseline| and |NSN| >= $10k
    (b) OS >= 0.70
    (c) p_entry <= 0.30
    (d) FWS >= 0.5, or TOP3 >= 0.5 with CONC >= 0.8
    pat_strong = a & b & c & d          pat_possible = a & b & (c | d)
    pat_loose  = (dom*z >= 2, |NSN| >= $5k) & b & (c | d)   [sensitivity variant, not in the plan]
Condition (e) (window ends before the news interval) holds by construction for snapshot windows,
which all end at the cutoff; there is no news timestamp for generic snapshots.

Labels (use the outcome: LABEL ONLY, never features):
    lbl_vindicated = dominant side won (dom=+1 and y=1, or dom=-1 and y=0)
    lbl_L1 = pat_strong & vindicated      lbl_N1 = pat_strong & not vindicated
    lbl_L2 = pat_possible & vindicated    lbl_N2 = pat_possible & not vindicated
(L2/N2 include L1/N1 rows.) Thresholds are the plan's fixed values, not tuned on any split.

CLI: python -m pmflow.features.weak_labels   (builds feat_weak_labels, prints counts by split)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .db import connect

LBS = (1, 6, 24, 72)


def pattern_flags(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Python version of the pattern conditions for a frame with core feature names
    (optionally suffixed, e.g. '_24h'). NaN conditions are False."""
    g = lambda c: df[c + suffix].astype(float)  # noqa: E731
    nsn, dom, z = g("nsn"), g("dom"), g("wl_z_nsn")
    med = g("wl_base_med_abs_nsn")
    a = ((dom * z) >= 4) & (nsn.abs() >= 5 * med) & (nsn.abs() >= 1e4)
    a_l = ((dom * z) >= 2) & (nsn.abs() >= 5e3)
    b = g("os") >= 0.7
    c = g("p_entry") <= 0.3
    d = (g("fws72") >= 0.5) | ((g("top3_share") >= 0.5) & (g("conc_top3") >= 0.8))
    out = pd.DataFrame({"cond_a": a, "cond_b": b, "cond_c": c, "cond_d": d}, index=df.index).fillna(False)
    out["pat_strong"] = out.cond_a & out.cond_b & out.cond_c & out.cond_d
    out["pat_possible"] = out.cond_a & out.cond_b & (out.cond_c | out.cond_d)
    out["pat_loose"] = a_l.fillna(False) & out.cond_b & (out.cond_c | out.cond_d)
    return out


def _sql() -> str:
    sel = ["f.snapshot_id", "f.market_id", "f.cutoff_ts", "e.split", "e.universe", "e.is_live",
           "f.wallet_age_censored"]
    for L in LBS:
        s = f"_{L}h"
        a = (f"(f.dom{s} * f.wl_z_nsn{s} >= 4 and abs(f.nsn{s}) >= 5 * f.wl_base_med_abs_nsn{s}"
             f" and abs(f.nsn{s}) >= 10000)")
        al = f"(f.dom{s} * f.wl_z_nsn{s} >= 2 and abs(f.nsn{s}) >= 5000)"
        b = f"(f.os{s} >= 0.7)"
        c = f"(f.p_entry{s} <= 0.3)"
        d = f"(f.fws72{s} >= 0.5 or (f.top3_share{s} >= 0.5 and f.conc_top3{s} >= 0.8))"
        vind = (f"(case when e.y is null or f.dom{s} = 0 or f.dom{s} is null then null"
                f" else (f.dom{s} > 0) = (e.y = 1) end)")
        sel += [
            f"f.nsn{s} as nsn{s}", f"f.gv{s} as gv{s}", f"f.os{s} as os{s}", f"f.p_entry{s} as p_entry{s}",
            f"f.fws72{s} as fws{s}", f"f.top3_share{s} as top3{s}", f"f.conc_top3{s} as conc{s}",
            f"f.wl_z_nsn{s} as z{s}", f"f.dom{s} as dom{s}",
            f"coalesce({a}, false) as cond_a{s}", f"coalesce({b}, false) as cond_b{s}",
            f"coalesce({c}, false) as cond_c{s}", f"coalesce({d}, false) as cond_d{s}",
            f"coalesce({a} and {b} and {c} and {d}, false) as pat_strong{s}",
            f"coalesce({a} and {b} and ({c} or {d}), false) as pat_possible{s}",
            f"coalesce({al} and {b} and ({c} or {d}), false) as pat_loose{s}",
            f"{vind} as lbl_vindicated{s}",
        ]
        strong, poss = f"({a} and {b} and {c} and {d})", f"({a} and {b} and ({c} or {d}))"
        for lab, pat, v in (("L1", strong, "true"), ("N1", strong, "false"),
                            ("L2", poss, "true"), ("N2", poss, "false")):
            sel.append(f"coalesce({pat} and {vind} = {v}, false) as lbl_{lab}{s}")
    body = ",\n  ".join(sel)
    return f"create table feat_weak_labels as select\n  {body}\nfrom feat_snapshot_features f join eval_snapshots e using (snapshot_id)"


def build(conn):
    with conn.cursor() as cur:
        cur.execute("drop table if exists feat_weak_labels")
        cur.execute(_sql())
        cur.execute("alter table feat_weak_labels add primary key (snapshot_id)")
    conn.commit()


def summary(conn) -> pd.DataFrame:
    rows = []
    for L in LBS:
        s = f"_{L}h"
        q = f"""select split, universe, count(*) n,
                  sum(pat_strong{s}::int) strong, sum(lbl_L1{s}::int) l1, sum(lbl_N1{s}::int) n1,
                  sum(pat_possible{s}::int) possible, sum(lbl_L2{s}::int) l2, sum(lbl_N2{s}::int) n2,
                  sum(pat_loose{s}::int) loose,
                  sum((pat_loose{s} and lbl_vindicated{s})::int) loose_vind,
                  avg(case when pat_strong{s} then p_entry{s} end) strong_mean_p_entry,
                  avg(case when pat_possible{s} then p_entry{s} end) possible_mean_p_entry,
                  avg(case when pat_loose{s} then p_entry{s} end) loose_mean_p_entry,
                  count(distinct case when pat_strong{s} then market_id end) strong_mkts,
                  count(distinct case when pat_possible{s} then market_id end) possible_mkts
                from feat_weak_labels group by 1, 2 order by 2, 1"""
        d = pd.read_sql_query(q, conn)
        d.insert(0, "lookback_h", L)
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    out["prec_strong"] = out.l1 / out.strong.where(out.strong > 0)
    out["prec_possible"] = out.l2 / out.possible.where(out.possible > 0)
    out["prec_loose"] = out.loose_vind / out.loose.where(out.loose > 0)
    out["N1_L1_ratio"] = out.n1 / out.l1.where(out.l1 > 0)
    out["N2_L2_ratio"] = out.n2 / out.l2.where(out.l2 > 0)
    return out


def precision(conn) -> pd.DataFrame:
    """Hit rate of each pattern vs the mean entry price of the dominant side (what the market
    already implied), by universe and split."""
    rows = []
    for L in LBS:
        for pat in ("pat_strong", "pat_possible", "pat_loose"):
            d = pd.read_sql_query(
                f"select universe, split, count(*) n_fire, count(distinct market_id) n_mkts,"
                f" avg(lbl_vindicated_{L}h::int) hit_rate, avg(p_entry_{L}h) mean_p_entry"
                f" from feat_weak_labels where {pat}_{L}h group by 1, 2", conn)
            d.insert(0, "pattern", pat)
            d.insert(0, "lookback_h", L)
            rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["excess_over_p_entry"] = d.hit_rate - d.mean_p_entry
    return d


def main():
    import warnings
    warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
    conn = connect()
    build(conn)
    s = summary(conn)
    s.to_csv("artifacts/features/weak_label_summary.csv", index=False)
    precision(conn).to_csv("artifacts/features/weak_label_precision.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
