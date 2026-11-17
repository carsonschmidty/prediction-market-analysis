"""Final held-out TEST scoring (evaluation role). Run ONCE, after final_audit and the frozen analysis plan
(artifacts/evaluation/final/analysis_plan.json).

    python -m pmflow.evaluation.final_score

Scores the frozen prediction files on the test split, in-scope and macro separately, against raw price, the
price-only control, constant-shift controls and train-fitted recalibrations; event-clustered and topic/theatre-
clustered inference; calibration; breakdowns; flag precision. Writes artifacts/evaluation/final/score_*.csv,
figures, and score_summary.json. No parameter here is chosen from test results.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics as M
from . import topics as T
from .final_audit import OUT, PRED_FILES, THRESH, DATA_END, q
from .harness import ART, load_snapshots

CANDS = ["lgbm_anchored", "stack_s1", "ridge_flow"]
CTRL = "ctrl_lgbm_price_only"
CTRL_SHIFT_TRAIN = 0.11114814096619798          # stage2_selected.json ctrl_intercept (fitted on TRAIN)
WEIGHTS = ("market", "snapshot", "event")
B = 2000


def load_test() -> pd.DataFrame:
    s = load_snapshots(where="is_live AND split = 'test'")
    base = q("SELECT snapshot_id, p_iso, p_platt FROM eval_baseline_preds WHERE snapshot_id = ANY(%s)",
             [s.snapshot_id.tolist()])
    s = s.merge(base, on="snapshot_id", how="left")
    for k, f in PRED_FILES.items():
        p = pd.read_parquet(ART.parent / "modeling" / f)
        p = p[p.split == "test"][["snapshot_id", "p_model", "flag"] + (["s1_informed_score", "z_lgbm", "z_s1"]
                                                                        if k == "stack_s1" else [])]
        p = p.rename(columns={"p_model": k, "flag": f"flag_{k}"})
        s = s.merge(p, on="snapshot_id", how="inner" if k == CTRL else "left")
    assert s[CANDS + [CTRL]].notna().all().all()
    tp = T.load()
    s = s.merge(tp, on="market_id", how="left")
    rb = pd.read_parquet(OUT / "bias_robust_rows.parquet")
    s = s.merge(rb, on="snapshot_id", how="left")
    em = q("SELECT market_id, end_date FROM eval_markets")
    s = s.merge(em, on="market_id", how="left")
    s["end_date"] = pd.to_datetime(s.end_date, utc=True)
    s["bias_robust_21d"] = s.end_date <= DATA_END - pd.Timedelta(days=21)
    s["p_iso"] = s.p_iso.astype(float)
    s["p_platt"] = s.p_platt.astype(float)
    lp = M.logit(s.price_yes)
    s["ctrl_shift_train"] = M.expit(lp + CTRL_SHIFT_TRAIN)
    # level-only version of each candidate: its own mean residual on test, per universe (label-free)
    for c in CANDS + [CTRL]:
        r = M.logit(s[c]) - lp
        s[f"level_only_{c}"] = np.nan
        for u in ("in_scope", "macro"):
            mu = s.universe == u
            w = M.snapshot_weights(s[mu], "market")
            s.loc[mu, f"level_only_{c}"] = M.expit(lp[mu] + np.average(r[mu], weights=w))
    s["month"] = s.cutoff_ts.dt.strftime("%Y-%m")
    s["price_bucket"] = pd.cut(s.price_yes, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    return s.sort_values("snapshot_id").reset_index(drop=True)


def _paired(d, pc, mc, w, groups, B=B, seed=0):
    y = d.y.to_numpy(float)
    tb = M.paired_test(M.brier_terms(y, d[pc]), M.brier_terms(y, d[mc]), groups, w, B=B, seed=seed)
    tl = M.paired_test(M.logloss_terms(y, d[pc]), M.logloss_terms(y, d[mc]), groups, w, B=B, seed=seed + 1)
    return tb, tl


def paired_table(s: pd.DataFrame, subset_col: str | None = None, tag="all") -> pd.DataFrame:
    comps = ["price_yes", CTRL, "ctrl_shift_train", "p_iso", "p_platt"]
    rows = []
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    for u in ("in_scope", "macro"):
        d = s[s.universe == u]
        if subset_col:
            d = d[d[subset_col].fillna(False).astype(bool)]
        if d.empty:
            continue
        for m in CANDS + [CTRL, "ctrl_shift_train", "p_iso", "p_platt"]:
            cs = comps + ([f"level_only_{m}"] if m in CANDS + [CTRL] else [])
            for c in cs:
                if c == m:
                    continue
                for wn in WEIGHTS:
                    w = M.snapshot_weights(d, wn)
                    tb, tl = _paired(d, c, m, w, d.event_id.to_numpy())
                    row = dict(subset=tag, universe=u, model=m, comparator=c, weighting=wn, n_snap=len(d),
                               n_mkt=d.market_id.nunique(), n_evt=d.event_id.nunique(),
                               brier_model=M.brier(d.y, d[m], w), brier_comp=M.brier(d.y, d[c], w),
                               d_brier=tb["delta"], d_brier_lo=tb["ci_lo"], d_brier_hi=tb["ci_hi"],
                               d_brier_p=tb["p_cluster_t"], d_brier_pboot=tb["p_boot"],
                               mde_brier=z * tb["se_cluster"],
                               ll_model=M.log_loss(d.y, d[m], w), ll_comp=M.log_loss(d.y, d[c], w),
                               d_ll=tl["delta"], d_ll_lo=tl["ci_lo"], d_ll_hi=tl["ci_hi"], d_ll_p=tl["p_cluster_t"],
                               mde_ll=z * tl["se_cluster"],
                               mean_abs_move=float(np.average(np.abs(d[m] - d.price_yes), weights=w)),
                               mean_resid_logit=float(np.average(M.logit(d[m]) - M.logit(d.price_yes), weights=w)))
                    # coarse clusters (market weighting only, to save time)
                    if wn == "market":
                        for lvl in ("topic", "theatre"):
                            g = d[lvl].to_numpy()
                            tbc, tlc = _paired(d, c, m, w, g, B=1000, seed=7)
                            row.update({f"d_brier_lo_{lvl}": tbc["ci_lo"], f"d_brier_hi_{lvl}": tbc["ci_hi"],
                                        f"d_brier_p_{lvl}": tbc["p_cluster_t"], f"n_{lvl}": tbc["n_clusters"],
                                        f"d_ll_lo_{lvl}": tlc["ci_lo"], f"d_ll_hi_{lvl}": tlc["ci_hi"],
                                        f"d_ll_p_{lvl}": tlc["p_cluster_t"]})
                    rows.append(row)
    return pd.DataFrame(rows)


def leave_one_theatre_out(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for u in ("in_scope", "macro"):
        d0 = s[s.universe == u]
        for th in ["(none)"] + sorted(d0.theatre.unique()):
            d = d0 if th == "(none)" else d0[d0.theatre != th]
            w = M.snapshot_weights(d, "market")
            for m in CANDS:
                for c in ("price_yes", CTRL):
                    tb, _ = _paired(d, c, m, w, d.event_id.to_numpy(), B=500)
                    rows.append(dict(universe=u, dropped=th, model=m, comparator=c, n_snap=len(d),
                                     n_evt=d.event_id.nunique(), d_brier=tb["delta"], lo=tb["ci_lo"], hi=tb["ci_hi"]))
    return pd.DataFrame(rows)


def calibration(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, rel = [], []
    rng = np.random.default_rng(0)
    for u in ("in_scope", "macro"):
        d = s[s.universe == u].reset_index(drop=True)
        w = M.snapshot_weights(d, "market")
        g = pd.factorize(d.event_id)[0]
        G = g.max() + 1
        by = [np.flatnonzero(g == k) for k in range(G)]
        picks = [np.concatenate([by[k] for k in rng.integers(0, G, G)]) for _ in range(300)]
        for m in ["price_yes"] + CANDS + [CTRL, "ctrl_shift_train", "p_iso"]:
            y, p = d.y.to_numpy(float), d[m].to_numpy(float)
            a, b = M.calibration_slope_intercept(y, p, w)
            e = M.ece(y, p, w)
            eb = [M.ece(y[k], p[k], w[k]) for k in picks]
            sb = [M.calibration_slope_intercept(y[k], p[k], w[k])[1] for k in picks[:200]]
            rows.append(dict(universe=u, model=m, ece=e, ece_lo=np.quantile(eb, .025), ece_hi=np.quantile(eb, .975),
                             calib_intercept=a, calib_slope=b, slope_lo=np.quantile(sb, .025),
                             slope_hi=np.quantile(sb, .975), mean_p=float(np.average(p, weights=w)),
                             ybar=float(np.average(y, weights=w))))
            t = M.reliability_table(y, p, w)
            rel.append(t.assign(universe=u, model=m))
    return pd.DataFrame(rows), pd.concat(rel, ignore_index=True)


def breakdowns(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for u in ("in_scope", "macro"):
        d0 = s[s.universe == u]
        for col in ("geo_subtype", "price_bucket", "month", "theatre", "end_rel"):
            for k, d in d0.groupby(d0[col].astype(str), observed=True):
                if d.event_id.nunique() < 3:
                    continue
                w = M.snapshot_weights(d, "market")
                for m in CANDS:
                    for c in ("price_yes", CTRL):
                        tb, _ = _paired(d, c, m, w, d.event_id.to_numpy(), B=500)
                        rows.append(dict(universe=u, by=col, level=k, model=m, comparator=c, n_snap=len(d),
                                         n_evt=d.event_id.nunique(), brier_price=M.brier(d.y, d.price_yes, w),
                                         ybar=float(np.average(d.y, weights=w)),
                                         mean_price=float(np.average(d.price_yes, weights=w)),
                                         d_brier=tb["delta"], lo=tb["ci_lo"], hi=tb["ci_hi"]))
    return pd.DataFrame(rows)


def flags(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for u in ("in_scope", "macro"):
        d = s[s.universe == u]
        for m in CANDS + [CTRL]:
            f = d[d[f"flag_{m}"] != 0]
            base = dict(universe=u, model=m, threshold=THRESH[m], n_test_snap=len(d))
            if f.empty:
                rows.append(dict(base, n_flag=0))
                continue
            dr = np.sign(f[f"flag_{m}"]).to_numpy()
            hit = np.where(dr > 0, f.y, 1 - f.y).astype(float)
            imp = np.where(dr > 0, f.price_yes, 1 - f.price_yes).astype(float)
            mod = np.where(dr > 0, f[m], 1 - f[m]).astype(float)
            pt, lo, hi = M.clustered_mean_ci(hit - imp, f.event_id.to_numpy(), B=B)
            ptm, lom, him = M.clustered_mean_ci(hit - imp, f.event_id.to_numpy(), w=1 / f.groupby("market_id")
                                                .market_id.transform("size").to_numpy(), B=B)
            # control-matched comparison: same rows, control's probability as the "implied" rate
            ctl = np.where(dr > 0, f[CTRL], 1 - f[CTRL]).astype(float)
            ptc, loc, hic = M.clustered_mean_ci(hit - ctl, f.event_id.to_numpy(), B=B)
            # robust subset
            fr = f[f.bias_robust.fillna(False).astype(bool)]
            if len(fr):
                drr = np.sign(fr[f"flag_{m}"]).to_numpy()
                hr = np.where(drr > 0, fr.y, 1 - fr.y) - np.where(drr > 0, fr.price_yes, 1 - fr.price_yes)
                ptr, lor, hir = M.clustered_mean_ci(hr.astype(float), fr.event_id.to_numpy(), B=B)
            else:
                ptr = lor = hir = np.nan
            rows.append(dict(base, n_flag=len(f), flag_rate=len(f) / len(d), n_mkt=f.market_id.nunique(),
                             n_evt=f.event_id.nunique(), share_yes=float((dr > 0).mean()),
                             precision=float(hit.mean()), implied_price=float(imp.mean()), model_prob=float(mod.mean()),
                             excess_vs_price=pt, lo=lo, hi=hi, excess_vs_price_mktw=ptm, lo_mktw=lom, hi_mktw=him,
                             excess_vs_ctrl_prob=ptc, lo_ctrl=loc, hi_ctrl=hic,
                             n_flag_robust=len(fr), excess_robust=ptr, lo_robust=lor, hi_robust=hir))
    return pd.DataFrame(rows)


def holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * v))
        out[k] = run
    return out


def plot_reliability(rel: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    style = {"price_yes": ("#444444", "raw price"), "lgbm_anchored": ("#1f5fa8", "lgbm_anchored (primary)"),
             "stack_s1": ("#c4561d", "stack_s1"), CTRL: ("#2e8b57", "price-only control")}
    for ax, u in zip(axes, ("in_scope", "macro")):
        for m, (c, lab) in style.items():
            t = rel[(rel.model == m) & (rel.universe == u) & (rel.n > 0)]
            ax.plot(t.mean_p, t.freq_y, "o-", ms=3.5, lw=1.4, color=c, label=lab)
        ax.plot([0, 1], [0, 1], ":", color="#999999", lw=1)
        ax.set_title(f"TEST reliability, {u} (market-weighted)", fontsize=10)
        ax.set_xlabel("mean forecast in bin")
        ax.set_ylabel("observed YES rate")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "score_reliability_test.png", dpi=120)
    plt.close(fig)


def main():
    plan = json.loads((OUT / "analysis_plan.json").read_text())
    s = load_test()
    print({u: len(s[s.universe == u]) for u in ("in_scope", "macro")})
    P = pd.concat([paired_table(s), paired_table(s, "bias_robust", "bias_robust_7d"),
                   paired_table(s, "bias_robust_21d", "bias_robust_21d")], ignore_index=True)
    s["non_robust"] = ~s.bias_robust.fillna(False).astype(bool)
    P = pd.concat([P, paired_table(s, "non_robust", "non_robust_DESCRIPTIVE")], ignore_index=True)
    P.to_csv(OUT / "score_paired.csv", index=False)
    loo = leave_one_theatre_out(s)
    loo.to_csv(OUT / "score_leave_one_theatre_out.csv", index=False)
    cal, rel = calibration(s)
    cal.to_csv(OUT / "score_calibration.csv", index=False)
    rel.to_csv(OUT / "score_reliability.csv", index=False)
    plot_reliability(rel)
    breakdowns(s).to_csv(OUT / "score_breakdowns.csv", index=False)
    fl = flags(s)
    fl.to_csv(OUT / "score_flags.csv", index=False)
    # primary + Holm over the three candidate-vs-control tests (in-scope, market-weighted, all rows)
    sel = P[(P.subset == "all") & (P.universe == "in_scope") & (P.weighting == "market") & (P.comparator == CTRL)
            & P.model.isin(CANDS)].set_index("model")
    summ = dict(plan_frozen_utc=plan["frozen_utc"], scored_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                primary=sel.loc["lgbm_anchored", ["d_brier", "d_brier_lo", "d_brier_hi", "d_brier_p", "mde_brier",
                                                  "d_ll", "d_ll_lo", "d_ll_hi"]].to_dict(),
                holm_vs_ctrl=holm(sel.d_brier_p.to_dict()))
    (OUT / "score_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    s.drop(columns=["price_bucket"]).to_parquet(OUT / "test_scored_frame.parquet", index=False)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
