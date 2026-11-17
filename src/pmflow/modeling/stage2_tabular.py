"""Stage 2: Baseline B — market-anchored models on engineered features (TRAIN walk-forward OOF + VAL only).

    python -m pmflow.modeling.stage2_tabular            # everything below, ~10-20 min on 3 threads

Models (all predict logit p = logit(price) + correction):
  controls    intercept-only shift; Platt (free slope + intercept); LightGBM on price only (re-learned calibration)
  ridge       OffsetLogit on RIDGE_FEATURES (pre-specified), lambda chosen on walk-forward OOF log loss
  lgbm_<grp>  AnchoredLGBM on price_logit + group features; n_rounds chosen on walk-forward OOF (can be 0)
Also: weighting sensitivity, shuffled-feature placebo (5 seeds), SHAP + group-permutation importance on val,
and a level-shift diagnostic (how much of a val gain is a uniform shift toward YES).
Outputs: artifacts/modeling/stage2_*.csv, stage2_*.png, data/modeling/stage2_oof.parquet (component residuals).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from pmflow.evaluation import metrics as M

from . import data as D
from .evaluate import fmt, score
from .models import AnchoredLGBM, OffsetLogit, _wll

MAX_ROUNDS = 1200
RIDGE_L2 = [1e-1, 1e-2, 1e-3, 1e-4]
MONO = {f: 1 for f in ["s_nsn_6h", "s_nsn_24h", "s_nsn_72h", "s_fresh_nsn_6h", "s_fresh_nsn_24h",
                       "s_fresh_nsn_72h", "s_fws72_6h", "s_fws72_24h", "s_fws72_72h", "s_pat_loose_any",
                       "s_pat_possible_any"]}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------ walk-forward helpers


def lgbm_select_rounds(d, feats, wscheme="market", monotone=None, params=None):
    """Average (over walk-forward folds) of the OOF weighted log-loss improvement curve; returns
    (best_rounds, curve_df, oof_resid) where oof_resid holds per-fold OOF residual logits at best_rounds."""
    curves = []
    folds = list(D.walk_forward_folds(d))
    models = []
    for k, fit, sc in folds:
        f, s = d[fit], d[sc]
        wf, ws = D.weights(f, wscheme), D.weights(s, wscheme)
        rec = {}
        m = AnchoredLGBM(feats, n_rounds=MAX_ROUNDS, monotone=monotone, **({"params": params} if params else {}))
        m.fit(f, f.price_logit, f.y, wf, valid=(s, s.price_logit, s.y, ws), record=rec)
        base = _wll(s.y.to_numpy(float), s.price_logit.to_numpy(float), ws)
        ll = np.array(rec["v"]["binary_logloss"])
        curves.append(pd.DataFrame(dict(fold=k, it=np.arange(1, len(ll) + 1), d_ll=base - ll, w=ws.sum())))
        models.append((k, sc, m))
    cv = pd.concat(curves)
    # equal weight per fold (folds differ a lot in size; avoid letting one period dominate)
    avg = cv.groupby("it").d_ll.mean()
    best = int(avg.idxmax()) if avg.max() > 0 else 0
    oof = np.full(len(d), np.nan)
    for k, sc, m in models:
        oof[sc] = m.resid_logit(d[sc], n_rounds=best) if best > 0 else 0.0
    return best, cv, oof


def fit_lgbm_full(d, feats, n_rounds, wscheme="market", monotone=None, params=None):
    tr = d[d.split == "train"]
    m = AnchoredLGBM(feats, n_rounds=n_rounds, monotone=monotone, **({"params": params} if params else {}))
    return m.fit(tr, tr.price_logit, tr.y, D.weights(tr, wscheme))


def ridge_wf(d, feats, wscheme="market", l2_grid=RIDGE_L2, **kw):
    """lambda by pooled OOF log loss (equal fold weight); returns best l2, oof resid, per-lambda table."""
    folds = [(k, f, s) for k, f, s in D.walk_forward_folds(d) if f.sum() >= 200 and s.sum() > 0]
    res, oofs = [], {}
    for l2 in l2_grid:
        oof = np.full(len(d), np.nan)
        dl = []
        for k, fit, sc in folds:
            f, s = d[fit], d[sc]
            X = f[feats].to_numpy(float) if feats else None
            m = OffsetLogit(l2=l2, **kw).fit(X, f.price_logit, f.y, D.weights(f, wscheme), frame=f)
            r = m.resid_logit(s[feats].to_numpy(float) if feats else None, s.price_logit)
            oof[sc] = r
            ws = D.weights(s, wscheme)
            yv, o = s.y.to_numpy(float), s.price_logit.to_numpy(float)
            dl.append(_wll(yv, o, ws) - _wll(yv, o + r, ws))
        res.append(dict(l2=l2, d_ll_mean_fold=float(np.mean(dl)), **{f"fold{k}": v for k, v in enumerate(dl)}))
        oofs[l2] = oof
    t = pd.DataFrame(res)
    best = float(t.loc[t.d_ll_mean_fold.idxmax(), "l2"])
    return best, oofs[best], t


def fit_ridge_full(d, feats, l2, wscheme="market", **kw):
    tr = d[d.split == "train"]
    return OffsetLogit(l2=l2, **kw).fit(tr[feats].to_numpy(float) if feats else None, tr.price_logit, tr.y,
                                        D.weights(tr, wscheme), frame=tr)


def oof_scored(d, oof):
    m = np.isfinite(oof)
    return d[m], oof[m]


def fold_table(d, oof, name):
    rows = []
    for k, fit, sc in D.walk_forward_folds(d):
        s = d[sc]
        r = oof[sc]
        for u in ("in_scope", "macro"):
            mu = (s.universe == u).to_numpy()
            if mu.sum() == 0:
                continue
            su, ru = s[mu], r[mu]
            w = M.snapshot_weights(su, "market")
            y, o = su.y.to_numpy(float), su.price_logit.to_numpy(float)
            pb, pm = D.expit(o), D.expit(o + ru)
            rows.append(dict(model=name, fold=k, universe=u, n=len(su), n_evt=su.event_id.nunique(),
                             d_brier=M.brier(y, pb, w) - M.brier(y, pm, w),
                             d_ll=M.log_loss(y, pb, w) - M.log_loss(y, pm, w)))
    return rows


# ------------------------------------------------------------------ main


def main(B=1000, n_placebo=5):
    t0 = time.time()
    D.ART.mkdir(parents=True, exist_ok=True)
    d = D.no_test(D.load_frame()).reset_index(drop=True)
    tr_m, va_m = (d.split == "train").to_numpy(), (d.split == "val").to_numpy()
    va = d[va_m]
    val_rows, oof_rows, fold_rows, sel = [], [], [], {}
    resid_oof = pd.DataFrame({"snapshot_id": d.snapshot_id})
    resid_val = pd.DataFrame({"snapshot_id": va.snapshot_id.to_numpy()})

    def record(name, oof, val_resid, extra=None):
        resid_oof[name] = oof
        resid_val[name] = val_resid
        so, ro = oof_scored(d, oof)
        if len(so):
            oof_rows.append(score(so, D.expit(so.price_logit + ro), name, weighting=("market", "snapshot"), B=B,
                                  extra=extra))
        fold_rows.extend(fold_table(d, oof, name))
        val_rows.append(score(va, D.expit(va.price_logit + val_resid), name, B=B, extra=extra))
        v = val_rows[-1]
        vv = v[v.weighting == "market"][["universe", "d_brier", "d_brier_lo", "d_brier_hi", "d_ll"]].round(4)
        log(f"{name}: val(market-w) {vv.to_dict('records')}")

    # ---- controls
    for name, kw in (("ctrl_intercept", dict(penalize_intercept=False)),
                     ("ctrl_platt", dict(penalize_intercept=False, free_slope=True))):
        l2, oof, t = ridge_wf(d, [], l2_grid=[0.0], **kw)
        m = fit_ridge_full(d, [], 0.0, **kw)
        sel[name] = dict(intercept=m.intercept_, slope=m.slope_)
        record(name, oof, m.resid_logit(None, va.price_logit))

    # ---- ridge (simplest model)
    for ws in ("market", "event", "snapshot"):
        l2, oof, t = ridge_wf(d, D.RIDGE_FEATURES, wscheme=ws)
        t.assign(weights=ws).to_csv(D.ART / f"stage2_ridge_lambda_{ws}.csv", index=False)
        m = fit_ridge_full(d, D.RIDGE_FEATURES, l2, wscheme=ws)
        name = "ridge" if ws == "market" else f"ridge_w{ws}"
        sel[name] = dict(l2=l2, intercept=m.intercept_, coef=dict(zip(D.RIDGE_FEATURES, map(float, m.coef_))))
        record(name, oof, m.resid_logit(va[D.RIDGE_FEATURES].to_numpy(float), va.price_logit))
    # ridge with no intercept (pure feature effects)
    l2, oof, t = ridge_wf(d, D.RIDGE_FEATURES, fit_intercept=False)
    m = fit_ridge_full(d, D.RIDGE_FEATURES, l2, fit_intercept=False)
    sel["ridge_noint"] = dict(l2=l2, coef=dict(zip(D.RIDGE_FEATURES, map(float, m.coef_))))
    record("ridge_noint", oof, m.resid_logit(va[D.RIDGE_FEATURES].to_numpy(float), va.price_logit))

    # ---- anchored LightGBM, group ablations
    curves = []
    variants = {f"lgbm_{g}": (["price_logit"] + cols, None, "market") for g, cols in D.GROUPS.items()}
    variants["lgbm_all_monotone"] = (["price_logit"] + D.GROUPS["all"], MONO, "market")
    variants["lgbm_all_wevent"] = (["price_logit"] + D.GROUPS["all"], None, "event")
    variants["lgbm_all_wsnapshot"] = (["price_logit"] + D.GROUPS["all"], None, "snapshot")
    variants["lgbm_all_noprice"] = (D.GROUPS["all"], None, "market")
    variants["DIAG_lgbm_all_plus_hours_to_end"] = (["price_logit"] + D.GROUPS["all"] + D.DIAG_ONLY, None, "market")
    full_models = {}
    for name, (feats, mono, ws) in variants.items():
        best, cv, oof = lgbm_select_rounds(d, feats, ws, mono)
        curves.append(cv.assign(model=name))
        m = fit_lgbm_full(d, feats, best, ws, mono)
        full_models[name] = m
        sel[name] = dict(n_rounds=best, n_features=len(feats), weights=ws)
        record(name, oof, m.resid_logit(va), extra=dict(n_rounds=best))
    pd.concat(curves).to_csv(D.ART / "stage2_lgbm_wf_curves.csv", index=False)

    # ---- placebo: shuffle feature rows jointly (train and val separately), keep price and y
    plac = []
    feats = ["price_logit"] + D.GROUPS["all"]
    fcols = D.GROUPS["all"]
    for s in range(n_placebo):
        rng = np.random.default_rng(D.SEED + s)
        dp = d.copy()
        for msk in (tr_m, va_m):
            idx = np.flatnonzero(msk)
            dp.loc[idx, fcols] = dp.loc[rng.permutation(idx), fcols].to_numpy()
        best, cv, oof = lgbm_select_rounds(dp, feats)
        m = fit_lgbm_full(dp, feats, best)
        vr = m.resid_logit(dp[va_m])
        r = score(dp[va_m], D.expit(dp[va_m].price_logit + vr), f"placebo_lgbm_all_s{s}", B=300,
                  weighting=("market", "snapshot"), extra=dict(n_rounds=best))
        so, ro = oof_scored(dp, oof)
        ro_ = score(so, D.expit(so.price_logit + ro), f"placebo_lgbm_all_s{s}", B=300, weighting=("market",),
                    extra=dict(n_rounds=best)) if len(so) else pd.DataFrame()
        plac.append(r.assign(where="val"))
        plac.append(ro_.assign(where="train_oof"))
        # ridge placebo
        l2, oofr, _ = ridge_wf(dp, D.RIDGE_FEATURES)
        mr = fit_ridge_full(dp, D.RIDGE_FEATURES, l2)
        vrr = mr.resid_logit(dp[va_m][D.RIDGE_FEATURES].to_numpy(float), dp[va_m].price_logit)
        plac.append(score(dp[va_m], D.expit(dp[va_m].price_logit + vrr), f"placebo_ridge_s{s}", B=300,
                          weighting=("market", "snapshot")).assign(where="val"))
        log(f"placebo {s}: lgbm rounds {best}")
    pd.concat(plac).to_csv(D.ART / "stage2_placebo.csv", index=False)

    # ---- importance for lgbm_all (val): SHAP and group permutation
    m = full_models["lgbm_all"]
    imp = []
    if m.booster is not None:
        sh = m.shap(va)[:, :-1]
        imp_shap = pd.DataFrame(dict(feature=m.features, mean_abs_shap_val=np.abs(sh).mean(0),
                                     gain=m.booster.feature_importance("gain")))
        imp_shap.sort_values("mean_abs_shap_val", ascending=False).to_csv(D.ART / "stage2_importance_shap.csv",
                                                                          index=False)
        base_r = m.resid_logit(va)
        rng = np.random.default_rng(D.SEED)
        for g, cols in [("price_logit", ["price_logit"])] + [(g, c) for g, c in D.GROUPS.items()
                                                               if g not in ("all", "price_only")]:
            for u in ("in_scope", "macro"):
                mu = (va.universe == u).to_numpy()
                vu = va[mu]
                w = M.snapshot_weights(vu, "market")
                y, o = vu.y.to_numpy(float), vu.price_logit.to_numpy(float)
                ll0 = _wll(y, o + base_r[mu], w)
                dls = []
                for rep in range(5):
                    vp = vu.copy()
                    vp[cols] = vp[cols].to_numpy()[rng.permutation(len(vp))]
                    dls.append(_wll(y, o + m.resid_logit(vp), w) - ll0)
                imp.append(dict(group=g, universe=u, perm_increase_ll=float(np.mean(dls)),
                                perm_sd=float(np.std(dls)), ll_model=ll0, ll_price=_wll(y, o, w)))
        pd.DataFrame(imp).to_csv(D.ART / "stage2_importance_perm_groups.csv", index=False)

    # ---- level-shift diagnostic on val: replace each model's residual by its val-mean (label-free) constant
    shift = []
    for name in resid_val.columns[1:]:
        r = resid_val[name].to_numpy(float)
        for u in ("in_scope", "macro"):
            mu = (va.universe == u).to_numpy()
            vu = va[mu]
            w = M.snapshot_weights(vu, "market")
            const = np.full(mu.sum(), np.average(r[mu], weights=w))
            y, o = vu.y.to_numpy(float), vu.price_logit.to_numpy(float)
            shift.append(dict(model=name, universe=u, mean_resid=float(const[0]),
                              d_brier_model=M.brier(y, D.expit(o), w) - M.brier(y, D.expit(o + r[mu]), w),
                              d_brier_const_shift=M.brier(y, D.expit(o), w) - M.brier(y, D.expit(o + const), w),
                              d_brier_model_vs_shift=M.brier(y, D.expit(o + const), w) - M.brier(y, D.expit(o + r[mu]), w)))
    pd.DataFrame(shift).to_csv(D.ART / "stage2_val_level_shift_diag.csv", index=False)

    V = pd.concat(val_rows, ignore_index=True)
    O = pd.concat(oof_rows, ignore_index=True)
    V.to_csv(D.ART / "stage2_val_results.csv", index=False)
    O.to_csv(D.ART / "stage2_train_oof_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(D.ART / "stage2_train_oof_by_fold.csv", index=False)
    (D.ART / "stage2_selected.json").write_text(json.dumps(sel, indent=1, default=float))
    resid_oof.to_parquet(D.REPO / "data" / "modeling" / "stage2_oof_resid.parquet", index=False)
    resid_val.to_parquet(D.REPO / "data" / "modeling" / "stage2_val_resid.parquet", index=False)
    print(fmt(V[V.weighting == "market"]))
    _plots(V, imp)
    increments(B)
    log(f"done in {time.time() - t0:.0f}s")


INCREMENT_PAIRS = [(m, "lgbm_price_only") for m in ("lgbm_flow", "lgbm_wallet", "lgbm_flags", "lgbm_context",
                                                     "lgbm_all", "lgbm_all_monotone", "lgbm_all_noprice")] + \
                  [("lgbm_price_only", "ctrl_platt"), ("lgbm_all", "ctrl_platt"), ("ridge", "ctrl_intercept"),
                   ("ridge_wevent", "ctrl_intercept"), ("ridge_noint", "price")]


def increments(B=1000):
    """Paired val/OOF deltas of each model vs its *control* (not vs raw price): does a feature group add anything
    beyond what re-calibrating / shifting the price already gives? Positive = model better than control."""
    d = D.no_test(D.load_frame()).reset_index(drop=True)
    out = []
    for where, fn in (("val", "stage2_val_resid.parquet"), ("train_oof", "stage2_oof_resid.parquet")):
        R = pd.read_parquet(D.REPO / "data" / "modeling" / fn)
        dd = d.set_index("snapshot_id").loc[R.snapshot_id].reset_index()
        for mname, cname in INCREMENT_PAIRS:
            ok = np.isfinite(R[mname]).to_numpy() & (np.isfinite(R[cname]).to_numpy() if cname != "price" else True)
            for u in ("in_scope", "macro"):
                mu = ok & (dd.universe == u).to_numpy()
                du = dd[mu]
                y, o = du.y.to_numpy(float), du.price_logit.to_numpy(float)
                pm = D.expit(o + R.loc[mu, mname].to_numpy())
                pc = D.expit(o + (R.loc[mu, cname].to_numpy() if cname != "price" else 0.0))
                for wn in ("market", "snapshot"):
                    w = M.snapshot_weights(du, wn)
                    tb = M.paired_test(M.brier_terms(y, pc), M.brier_terms(y, pm), du.event_id, w, B=B)
                    tl = M.paired_test(M.logloss_terms(y, pc), M.logloss_terms(y, pm), du.event_id, w, B=B, seed=1)
                    out.append(dict(where=where, model=mname, control=cname, universe=u, weighting=wn, n=len(du),
                                    n_evt=du.event_id.nunique(), d_brier=tb["delta"], d_brier_lo=tb["ci_lo"],
                                    d_brier_hi=tb["ci_hi"], d_ll=tl["delta"], d_ll_lo=tl["ci_lo"], d_ll_hi=tl["ci_hi"]))
    t = pd.DataFrame(out)
    t.to_csv(D.ART / "stage2_increments_vs_control.csv", index=False)
    return t


def _plots(V, imp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    v = V[V.weighting == "market"]
    names = list(dict.fromkeys(v.model))
    for ax, u in zip(axes, ("in_scope", "macro")):
        vu = v[v.universe == u].set_index("model").loc[names]
        yy = np.arange(len(names))
        ax.errorbar(vu.d_brier, yy, xerr=[vu.d_brier - vu.d_brier_lo, vu.d_brier_hi - vu.d_brier], fmt="o",
                    color="#2a5d9f", ecolor="#9fb6d6", capsize=2)
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_yticks(yy, names, fontsize=8)
        ax.set_title(f"VAL {u}: ΔBrier vs price (market-weighted, 95% event CI)", fontsize=9)
        ax.set_xlabel("Brier(price) − Brier(model)  (positive = model better)")
    fig.tight_layout()
    fig.savefig(D.ART / "stage2_val_dbrier.png", dpi=110)
    plt.close(fig)
    try:
        sh = pd.read_csv(D.ART / "stage2_importance_shap.csv").head(20)[::-1]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(sh.feature, sh.mean_abs_shap_val, color="#2a5d9f")
        ax.set_xlabel("mean |SHAP| on VAL (logit units), lgbm_all")
        fig.tight_layout()
        fig.savefig(D.ART / "stage2_importance_shap.png", dpi=110)
        plt.close(fig)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    import sys
    if "--increments" in sys.argv:
        print(increments().round(4).to_string())
    else:
        main()
