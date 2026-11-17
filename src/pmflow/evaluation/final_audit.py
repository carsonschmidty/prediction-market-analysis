"""Final-phase leakage / integrity audit (evaluation role). Runs BEFORE any test metric is computed.

    python -m pmflow.evaluation.final_audit [--n-test 40 --n-val 30 --n-train 10]

Nothing in this module scores a model on test: it only (a) re-derives inputs and predictions, (b) checks
invariants, and (c) describes the test split with outcome vs PRICE (Baseline-A-type quantities) to size the
window-selection artefact. Outputs: artifacts/evaluation/final/audit_*.csv and audit_summary.json.

Sections
 A  feature reproduction: stored feat_snapshot_features rows vs a recomputation on a DB copy truncated at the
    cutoff (feature agent's truncation procedure, fresh random sample incl. large markets), and vs a full-DB
    recomputation (separates leakage from build-time drift such as pm_wallets coverage growth)
 B  model inputs: candidate feature lists vs forbidden columns (lbl_*, *_diag, hours_to_end, full_*)
 C  refit reproduction: refit the candidates/control on TRAIN rows only with the modeling code and compare
    the frozen val/test predictions (proves the fitted row set excludes val/test)
 D  splits: invariants, purged rows, topic / template overlap between train and test
 E  pre-registration: hashes, row counts, thresholds, flag recomputation
 F  maker rows: role filters, recomputed taker counts on a sample
 G  window-selection bias on test (y vs price only) and the bias-robust subset definition
 H  text / timestamp checks (question text, end_date vs closed_time)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import topics as T
from .harness import ART, REPO, connect

OUT = ART / "final"
DATA_END = pd.Timestamp("2026-11-17", tz="UTC")
TEST_START = pd.Timestamp("2026-07-01", tz="UTC")
# bias-robust set: market's scheduled end_date at least this long before data end, so that NO-resolving
# markets would also have closed (and resolved) by data end; fixed here before scoring
SAFE_MARGIN_DAYS = 7
PRED_FILES = {
    "lgbm_anchored": "preds_lgbm_anchored.parquet",
    "stack_s1": "preds_stack_s1.parquet",
    "ridge_flow": "preds_ridge_flow.parquet",
    "ctrl_lgbm_price_only": "preds_ctrl_lgbm_price_only.parquet",
}
THRESH = {"lgbm_anchored": 0.03, "stack_s1": 0.05, "ridge_flow": 0.05, "ctrl_lgbm_price_only": 0.05}
FULL_HISTORY_COLS = ("full_cov_share", "fws72_full", "dom_log_prior_mkts_full")


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def q(sql, params=None) -> pd.DataFrame:
    with connect() as c:
        cur = c.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description])


# ------------------------------------------------------------------ A. feature reproduction


def _compare_row(stored: pd.Series, w: pd.DataFrame) -> list[str]:
    bad = []
    for col in stored.index:
        if col in ("snapshot_id", "cutoff_ts", "market_id"):
            continue
        wc = col if col in w else col.replace("burst_b", "burst_B")
        if wc not in w:
            bad.append(col + "(missing)")
            continue
        a, b = stored[col], w[wc].iloc[0]
        if isinstance(a, (str, bool, np.bool_)) or a is None:
            if str(a) != str(b) and not (pd.isna(a) and pd.isna(b)):
                bad.append(col)
        else:
            try:
                fa, fb = float(a), float(b)
            except (TypeError, ValueError):
                bad.append(col)
                continue
            if not np.isclose(fa, fb, rtol=2e-5, atol=1e-5, equal_nan=True):
                bad.append(col)
    return bad


def audit_features(n_test=40, n_val=30, n_train=10, seed="final-eval") -> pd.DataFrame:
    sys.path.insert(0, str(REPO / "tests" / "features"))
    from test_leakage import _cut_sources  # the feature agent's truncation procedure (read-only use)

    from pmflow.features.core import compute_window_features, to_wide
    from pmflow.features.db import connect as fconnect
    conn = fconnect()
    rows = []
    for split, n in (("test", n_test), ("val", n_val), ("train", n_train)):
        ids = pd.read_sql_query(
            "select f.snapshot_id from feat_snapshot_features f join eval_snapshots e using (snapshot_id) "
            f"where e.split = '{split}' order by md5(f.snapshot_id::text || '{seed}') limit {n}", conn)
        for sid in ids.snapshot_id:
            stored = pd.read_sql_query(f"select * from feat_snapshot_features where snapshot_id = {int(sid)}",
                                       conn).iloc[0]
            mid, cut = int(stored.market_id), stored.cutoff_ts
            t0 = time.time()
            full = to_wide(compute_window_features(mid, cut, conn=conn))
            src = _cut_sources(conn, mid, cut)
            trunc = to_wide(compute_window_features(mid, cut, conn=conn, src=src))
            conn.rollback()
            b_tr = _compare_row(stored, trunc)
            b_full = _compare_row(stored, full)
            leak = [c for c in b_tr if c not in b_full]          # differs only when the future is removed
            drift = [c for c in b_full]                          # stored vs today's full DB
            rows.append(dict(split=split, snapshot_id=int(sid), market_id=mid, cutoff_ts=cut,
                             n_cols=len(stored) - 3, n_diff_trunc=len(b_tr), n_diff_full=len(b_full),
                             diff_trunc=",".join(b_tr), diff_full=",".join(b_full),
                             diff_trunc_not_full=",".join(leak),
                             only_full_history_cols=all(any(c.startswith(f) for f in FULL_HISTORY_COLS)
                                                        for c in b_tr + b_full),
                             secs=round(time.time() - t0, 1)))
            log(f"A {split} {sid}: trunc-diff {len(b_tr)} full-diff {len(b_full)} ({rows[-1]['secs']}s)")
    conn.close()
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B/C. model inputs and refit reproduction


def audit_model_inputs() -> dict:
    from pmflow.modeling import data as D
    cand_feats = {"lgbm_anchored": ["price_logit"] + D.GROUPS["all"], "ctrl_lgbm_price_only": ["price_logit"],
                  "ridge_flow": list(D.RIDGE_FEATURES)}
    forbidden_rx = re.compile(r"^lbl_|_diag$|^hours_to_end|^full_|fws72_full|_full_|closed_time|resolution|^y$")
    hits = {k: [f for f in v if forbidden_rx.search(f)] for k, v in cand_feats.items()}
    # engineered columns: which raw columns feed them (from add_engineered source)
    import inspect
    src = inspect.getsource(D.add_engineered)
    raw_refs = sorted(set(re.findall(r'd\[f?"([a-z0-9_{}]+)"\]', src)) | set(re.findall(r"d\.([a-z_0-9]+)", src)))
    raw_bad = [r for r in raw_refs if forbidden_rx.search(r.replace("{L}", "24").replace("{c}", "x"))]
    frame_cols = pd.read_parquet(D.CACHE, columns=None).columns
    frame_bad = [c for c in frame_cols if forbidden_rx.search(c) and c not in ("hours_to_end",)]
    return dict(candidate_feature_counts={k: len(v) for k, v in cand_feats.items()},
                forbidden_in_candidate_features=hits, raw_columns_used_by_engineering=raw_refs,
                forbidden_raw_refs=raw_bad, forbidden_cols_in_model_frame=frame_bad,
                hours_to_end_in_frame="hours_to_end" in frame_cols,
                hours_to_end_in_any_candidate=any("hours_to_end" in v for v in cand_feats.values()))


def audit_refit() -> tuple[pd.DataFrame, dict]:
    """Refit candidates on TRAIN rows with the modeling code; compare to the frozen predictions."""
    from pmflow.modeling import data as D
    from pmflow.modeling.stage2_tabular import fit_lgbm_full, fit_ridge_full
    from pmflow.modeling.stage3_system_one import _pca_fit
    sel2 = json.loads((D.ART / "stage2_selected.json").read_text())
    d = D.load_frame().reset_index(drop=True)
    tr = d[d.split == "train"]
    D.assert_fit_rows(tr)
    out, info = [], {}
    fitted_rows = dict(n_train_rows=len(tr), n_train_events=int(tr.event_id.nunique()),
                       max_train_cutoff=str(tr.cutoff_ts.max()), max_train_event_close=str(tr.event_close.max()))
    info["refit_rows"] = fitted_rows
    preds = {k: pd.read_parquet(D.ART / v) for k, v in PRED_FILES.items()}

    def cmp(name, p_new):
        f = preds[name].set_index("snapshot_id").p_model
        p_new = pd.Series(np.clip(p_new, 1e-4, 1 - 1e-4), index=d.snapshot_id)
        for s in ("train", "val", "test"):
            ids = d.snapshot_id[d.split == s]
            diff = np.abs(f.loc[ids].to_numpy() - p_new.loc[ids].to_numpy())
            out.append(dict(model=name, split=s, n=len(ids), max_abs_diff=float(diff.max()),
                            mean_abs_diff=float(diff.mean())))

    lg = fit_lgbm_full(d, ["price_logit"] + D.GROUPS["all"], sel2["lgbm_all"]["n_rounds"])
    z_lgbm = lg.resid_logit(d)
    cmp("lgbm_anchored", D.expit(d.price_logit + z_lgbm))
    lp = fit_lgbm_full(d, ["price_logit"], sel2["lgbm_price_only"]["n_rounds"])
    cmp("ctrl_lgbm_price_only", D.expit(d.price_logit + lp.resid_logit(d)))
    rn = fit_ridge_full(d, D.RIDGE_FEATURES, sel2["ridge_noint"]["l2"], fit_intercept=False)
    z_r = rn.resid_logit(d[D.RIDGE_FEATURES].to_numpy(float), d.price_logit)
    cmp("ridge_flow", D.expit(d.price_logit + z_r))
    info["ridge_flow_implicit_constant_logit"] = float(-(rn.coef_ * rn.scaler.mu / rn.scaler.sd).sum())
    info["ridge_flow_mean_abs_resid_test"] = float(np.abs(z_r[(d.split == "test").to_numpy()]).mean())
    # system-one frozen head: PCA + ridge on TRAIN embeddings
    try:
        E = np.load(REPO / "data" / "modeling" / "s1_emb.npy").astype(np.float32)
        ids = np.load(REPO / "data" / "modeling" / "s1_emb_ids.npy")
        assert (ids == d.snapshot_id.to_numpy()).all()
        tr_m = (d.split == "train").to_numpy()
        mu, V = _pca_fit(E[tr_m])
        P = (E - mu) @ V.T
        cols = [f"pc{i}" for i in range(P.shape[1])]
        dd = pd.concat([d, pd.DataFrame(P, columns=cols, index=d.index)], axis=1)
        sel3 = json.loads((D.ART / "stage3_selected.json").read_text())
        m = fit_ridge_full(dd, cols, sel3["s1_text_desc"]["l2"], fit_intercept=True)
        z_s1 = m.resid_logit(dd[cols].to_numpy(float), dd.price_logit)
        st = preds["stack_s1"].set_index("snapshot_id")
        for s in ("val", "test"):
            msk = (d.split == s).to_numpy()
            info[f"z_s1_max_abs_diff_{s}"] = float(np.abs(st.loc[d.snapshot_id[msk], "z_s1"].to_numpy() - z_s1[msk]).max())
            info[f"z_lgbm_max_abs_diff_{s}"] = float(np.abs(st.loc[d.snapshot_id[msk], "z_lgbm"].to_numpy() - z_lgbm[msk]).max())
        del E, P, dd
    except Exception as e:  # noqa: BLE001
        info["z_s1_refit_error"] = repr(e)
    # stack: exact linear structure resid = c + b1 z_lgbm + b2 z_s1 on val/test (implicit constant from centring)
    st = preds["stack_s1"]
    st = st[st.split.isin(["val", "test"])]
    pr = d.set_index("snapshot_id").loc[st.snapshot_id, "price_yes"].to_numpy(float)
    resid = D.logit(st.p_model) - D.logit(pr)
    X = np.column_stack([np.ones(len(st)), st.z_lgbm, st.z_s1])
    coef, *_ = np.linalg.lstsq(X, resid, rcond=None)
    info["stack_resid_fit"] = dict(const=float(coef[0]), b_lgbm=float(coef[1]), b_s1=float(coef[2]),
                                   max_abs_err=float(np.abs(X @ coef - resid).max()))
    return pd.DataFrame(out), info


# ------------------------------------------------------------------ D. splits and topics


def audit_splits() -> tuple[dict, pd.DataFrame]:
    from .leakage import check_snapshot_selection
    inv = check_snapshot_selection()
    res = dict(invariants=inv)
    ids = pd.concat([pd.read_parquet(ART.parent / "modeling" / f, columns=["snapshot_id", "split"])
                     .assign(model=k) for k, f in PRED_FILES.items()])
    s = q("SELECT snapshot_id, split AS split_db, is_live, grid_24h FROM eval_snapshots WHERE snapshot_id = ANY(%s)",
          [ids.snapshot_id.unique().tolist()])
    m = ids.merge(s, on="snapshot_id", how="left")
    res["pred_rows_missing_in_db"] = int(m.split_db.isna().sum())
    res["pred_split_mismatch"] = int((m.split != m.split_db).sum())
    res["pred_rows_purged"] = int(m.split_db.fillna("").str.startswith("purged").sum())
    res["pred_rows_not_live"] = int((~m.is_live.fillna(False).astype(bool)).sum())
    res["feat_rows_purged_test"] = int(q("SELECT count(*) n FROM feat_snapshot_features f JOIN eval_snapshots e "
                                         "USING (snapshot_id) WHERE e.split='purged_test'").n[0])
    # events straddling scored splits (all snapshots, not only live)
    res["events_in_two_scored_splits_all_rows"] = int(q(
        "SELECT count(*) n FROM (SELECT event_id FROM eval_snapshots WHERE split IN ('train','val','test') "
        "GROUP BY 1 HAVING count(DISTINCT split) > 1) x").n[0])
    # markets whose closed_time in train events is >= any test cutoff (label known before test?)
    res["train_markets_closing_after_2026_04_01"] = int(q(
        "SELECT count(*) n FROM eval_markets m JOIN eval_event_splits e USING (event_id) "
        "WHERE e.split='train' AND m.closed_time >= '2026-04-01'").n[0])
    # topics
    tp = T.load()
    snaps = q("SELECT snapshot_id, market_id, event_id, universe, split FROM eval_snapshots "
              "WHERE is_live AND grid_24h AND split IN ('train','val','test')").merge(tp, on="market_id")
    rows = []
    for u in ("in_scope", "macro"):
        su = snaps[snaps.universe == u]
        tr = su[su.split == "train"]
        for s in ("val", "test"):
            ss = su[su.split == s]
            for level in ("topic", "theatre", "template"):
                inter = ss[level].isin(set(tr[level]))
                ev = ss.drop_duplicates("event_id")
                rows.append(dict(universe=u, split=s, level=level, n_clusters=ss[level].nunique(),
                                 share_snap_in_train_cluster=float(inter.mean()),
                                 share_evt_in_train_cluster=float(ev[level].isin(set(tr[level])).mean()),
                                 largest_cluster=ss[level].value_counts().index[0],
                                 largest_cluster_share=float(ss[level].value_counts(normalize=True).iloc[0])))
    return res, pd.DataFrame(rows)


# ------------------------------------------------------------------ E. pre-registration


def audit_prereg() -> pd.DataFrame:
    rows = []
    for k, f in PRED_FILES.items():
        p = ART.parent / "modeling" / f
        d = pd.read_parquet(p)
        price = d.p_model - d.edge
        th = THRESH[k]
        fl = np.where(d.edge >= th, 1, np.where(-d.edge >= th, -1, 0))
        rows.append(dict(model=k, file=f, sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         mtime_utc=pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").isoformat(),
                         n_rows=len(d), n_train=int((d.split == "train").sum()), n_val=int((d.split == "val").sum()),
                         n_test=int((d.split == "test").sum()), threshold=th,
                         flag_recomputed_mismatch=int((fl != d.flag).sum()),
                         n_test_flags=int((d[d.split == "test"].flag != 0).sum()),
                         p_nan=int(d.p_model.isna().sum()), price_range=f"{price.min():.3f}-{price.max():.3f}"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ F. maker rows


def audit_maker(n=300) -> dict:
    r = dict(role_counts=q("SELECT role, count(*) n FROM pm_trades GROUP BY 1").set_index("role").n.to_dict())
    r["test_markets_with_maker_rows"] = int(q(
        "SELECT count(DISTINCT t.market_id) n FROM pm_trades t JOIN eval_markets m USING (market_id) "
        "JOIN eval_event_splits e USING (event_id) WHERE t.role = 1 AND e.split='test'").n[0])
    s = q(f"SELECT snapshot_id, market_id, cutoff_ts, n_trades_cum, price_yes FROM eval_snapshots WHERE is_live AND "
          f"split IN ('val','test') ORDER BY md5(snapshot_id::text || 'mk') LIMIT {n}")
    bad_taker, n_with_maker, bad_all = 0, 0, 0
    with connect() as c:
        for x in s.itertuples():
            a, b = c.execute("SELECT count(*) FILTER (WHERE role = 0), count(*) FROM pm_trades WHERE market_id=%s "
                             "AND ts < %s", [int(x.market_id), x.cutoff_ts]).fetchone()
            bad_taker += int(a != x.n_trades_cum)
            n_with_maker += int(b != a)
            bad_all += int(b != x.n_trades_cum)
    r.update(n_checked=len(s), taker_count_mismatch=bad_taker, snapshots_whose_market_now_has_maker_rows_before_cutoff=n_with_maker,
             mismatch_if_role_not_filtered=bad_all)
    return r


# ------------------------------------------------------------------ G. window-selection bias (y vs price only)


def window_bias() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = q("""SELECT s.snapshot_id, s.market_id, s.event_id, s.universe, s.split, s.cutoff_ts, s.y, s.price_yes,
                    m.end_date, m.closed_time, m.created_at, e.event_close
             FROM eval_snapshots s JOIN eval_markets m ON m.market_id = s.market_id
             JOIN eval_event_splits e ON e.event_id = s.event_id
             WHERE s.is_live AND s.grid_24h AND s.split IN ('train','val','test')""")
    for c in ("end_date", "closed_time", "cutoff_ts", "event_close", "created_at"):
        s[c] = pd.to_datetime(s[c], utc=True)
    s["price_yes"] = s.price_yes.astype(float)
    s["w_mkt"] = 1.0 / s.groupby("market_id").market_id.transform("size")
    safe_end = DATA_END - pd.Timedelta(days=SAFE_MARGIN_DAYS)
    split_end = {"train": pd.Timestamp("2026-04-01", tz="UTC"), "val": pd.Timestamp("2026-07-01", tz="UTC"),
                 "test": DATA_END}
    s["window_end"] = s.split.map(split_end)
    s["end_rel"] = np.select(
        [s.end_date.isna(), s.end_date <= s.window_end - pd.Timedelta(days=SAFE_MARGIN_DAYS),
         s.end_date <= s.window_end + pd.Timedelta(days=30)],
        ["no_end_date", "end_date_safely_inside_window", "end_date_near_window_end(<=+30d)"],
        "end_date_after_window(+30d)")
    s["bias_robust"] = s.end_rel.eq("end_date_safely_inside_window")
    days = (s.event_close - s.cutoff_ts).dt.total_seconds() / 86400
    s["tte_bucket_DIAG"] = pd.cut(days, [-1, 7, 30, 90, 180, 1e5], labels=["<=7d", "7-30d", "30-90d", "90-180d", ">180d"])
    s["early_close_days"] = (s.end_date - s.closed_time).dt.total_seconds() / 86400

    def agg(g):
        w = g.w_mkt
        return pd.Series(dict(n_snap=len(g), n_mkt=g.market_id.nunique(), n_evt=g.event_id.nunique(),
                              ybar_mkt=np.average(g.y, weights=w), price_mkt=np.average(g.price_yes, weights=w),
                              yes_excess_mkt=np.average(g.y - g.price_yes, weights=w),
                              yes_excess_snap=float((g.y - g.price_yes).mean())))
    rows = []
    for keys in (["universe", "split"], ["universe", "split", "tte_bucket_DIAG"], ["universe", "split", "end_rel"],
                 ["universe", "split", "bias_robust"]):
        t = s.groupby(keys, observed=True).apply(agg, include_groups=False).reset_index()
        t.insert(0, "breakdown", "+".join(keys[2:]) or "all")
        rows.append(t)
    tab = pd.concat(rows, ignore_index=True)
    # end_date plausibility: does end_date look edited to the close date for YES markets?
    mk = s.drop_duplicates("market_id")
    ed = mk.groupby(["universe", "split", "y"]).apply(lambda g: pd.Series(dict(
        n=len(g), med_end_minus_close_d=g.early_close_days.median(),
        share_close_ge3d_before_end=float((g.early_close_days >= 3).mean()),
        share_end_within_1d_of_close=float((g.early_close_days.abs() <= 1).mean()))), include_groups=False).reset_index()
    return tab, ed, s[["snapshot_id", "bias_robust", "end_rel"]]


# ------------------------------------------------------------------ H. text checks


def audit_text() -> pd.DataFrame:
    m = q("SELECT m.market_id, m.question, m.y, e.split FROM eval_markets m JOIN eval_event_splits e USING (event_id) "
          "WHERE e.split IN ('val','test')")
    rx = re.compile(r"resolved|resolves (yes|no)|\bwon\b|has (been )?(happened|occurred)|confirmed|already|"
                    r"clarification|update[d:]|\(yes\)|\(no\)|outcome", re.I)
    m["hit"] = m.question.fillna("").str.contains(rx)
    return m[m.hit][["market_id", "split", "y", "question"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--skip-features", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {"run_utc": pd.Timestamp.now(tz="UTC").isoformat(), "note": "no test metric computed in this module"}
    log("E prereg");  pre = audit_prereg(); pre.to_csv(OUT / "audit_prereg.csv", index=False)
    log("B inputs");  summ["model_inputs"] = audit_model_inputs()
    log("D splits");  sp, tp = audit_splits(); summ["splits"] = sp; tp.to_csv(OUT / "audit_topic_overlap.csv", index=False)
    log("F maker");   summ["maker"] = audit_maker()
    log("G bias");    wb, ed, robust = window_bias()
    wb.to_csv(OUT / "audit_window_bias.csv", index=False); ed.to_csv(OUT / "audit_end_date.csv", index=False)
    robust.to_parquet(OUT / "bias_robust_rows.parquet", index=False)
    log("H text");    tx = audit_text(); tx.to_csv(OUT / "audit_text_hits.csv", index=False); summ["text_hits"] = len(tx)
    log("C refit");   rf, info = audit_refit(); rf.to_csv(OUT / "audit_refit.csv", index=False); summ["refit"] = info
    (OUT / "audit_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    if not a.skip_features:
        log("A features")
        fa = audit_features(a.n_test, a.n_val, a.n_train)
        fa.to_csv(OUT / "audit_features.csv", index=False)
        summ["features"] = dict(n=len(fa), rows_with_trunc_only_diffs=int((fa.diff_trunc_not_full != "").sum()),
                                rows_any_diff_trunc=int((fa.n_diff_trunc > 0).sum()),
                                rows_any_diff_full=int((fa.n_diff_full > 0).sum()))
    (OUT / "audit_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
