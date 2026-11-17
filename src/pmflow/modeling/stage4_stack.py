"""Stages 4-6: market-anchored stack, per-component calibration, high-confidence flag, frozen predictions.

    python -m pmflow.modeling.stage4_stack          # needs stage2 + stage3 outputs

* Components are residual logits vs logit(price): z_lgbm (lgbm_all), z_s1 (frozen-encoder head, text+descriptors),
  each with walk-forward OOF values on TRAIN and train-fit values on VAL/TEST.
* Calibration (stage 5): Platt on the residual, i.e. logit p = logit(price) + a + b*z, fitted on TRAIN OOF rows only
  (no isotonic). Reliability/ECE on VAL.
* Stack (stage 4): logit p = logit(price) + b0 + b1*z_lgbm + b2*z_s1, L2-shrunk, fitted on TRAIN OOF rows; VAL check.
* Flag (stage 6): flag = sign(p_model - price) if |p_model - price| >= threshold; threshold chosen on VAL by a fixed
  rule, then frozen.
* Frozen predictions for ALL splits are written for the pre-registered candidates. No test metric is computed here:
  every scoring call goes through `evaluate.score`, which refuses test rows.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pmflow.evaluation import metrics as M

from . import data as D
from .evaluate import score
from .models import OffsetLogit
from .stage2_tabular import fit_lgbm_full, fit_ridge_full, ridge_wf

FLAG_GRID = [0.02, 0.03, 0.05, 0.075, 0.10]
STACK_L2 = [1.0, 1e-1, 1e-2, 1e-3]


def _fit_meta(frame, Z, l2, fit_intercept=True):
    return OffsetLogit(l2=l2, fit_intercept=fit_intercept).fit(Z, frame.price_logit, frame.y,
                                                               D.weights(frame, "market"), frame=frame)


def _meta_wf(dev, zcols, fit_intercept=True):
    """Choose meta l2 by walk-forward over the TRAIN OOF rows themselves (folds 1..4 fitted on earlier OOF)."""
    ok = dev[zcols].notna().all(axis=1).to_numpy() & (dev.split == "train").to_numpy()
    sub = dev[ok].reset_index(drop=True)
    l2, _, t = ridge_wf(sub, zcols, l2_grid=STACK_L2, fit_intercept=fit_intercept)
    return l2, t, sub


def main(B=1000):
    D.ART.mkdir(parents=True, exist_ok=True)
    sel2 = json.loads((D.ART / "stage2_selected.json").read_text())
    d = D.load_frame().reset_index(drop=True)
    dev_m = d.split.isin(["train", "val"]).to_numpy()
    tr = d[d.split == "train"]
    # ---- component 1: anchored LightGBM (all groups), refit deterministically on TRAIN
    feats = ["price_logit"] + D.GROUPS["all"]
    lg = fit_lgbm_full(d[dev_m], feats, sel2["lgbm_all"]["n_rounds"])
    z_lgbm_all = lg.resid_logit(d)
    oof2 = pd.read_parquet(D.REPO / "data" / "modeling" / "stage2_oof_resid.parquet")
    val2 = pd.read_parquet(D.REPO / "data" / "modeling" / "stage2_val_resid.parquet")
    chk = d[["snapshot_id"]].assign(z=z_lgbm_all).merge(val2[["snapshot_id", "lgbm_all"]], on="snapshot_id")
    det_err = float(np.abs(chk.z - chk.lgbm_all).max())
    assert det_err < 1e-6, f"lgbm refit not deterministic ({det_err})"
    # controls / simplest model, refit on TRAIN for all splits
    lp = fit_lgbm_full(d[dev_m], ["price_logit"], sel2["lgbm_price_only"]["n_rounds"])
    z_price_only = lp.resid_logit(d)
    rn = fit_ridge_full(d[dev_m], D.RIDGE_FEATURES, sel2["ridge_noint"]["l2"], fit_intercept=False)
    z_ridge_noint = rn.resid_logit(d[D.RIDGE_FEATURES].to_numpy(float), d.price_logit)
    # ---- component 2: system-one frozen-encoder head
    s3 = pd.read_parquet(D.REPO / "data" / "modeling" / "stage3_resid.parquet")
    d = d.merge(oof2[["snapshot_id", "lgbm_all"]].rename(columns={"lgbm_all": "z_lgbm_oof"}), on="snapshot_id",
                how="left")
    d = d.merge(s3[["snapshot_id", "s1_text_desc_oof", "s1_text_desc_all", "s1_aux_informed_score"]],
                on="snapshot_id", how="left")
    d["z_lgbm_all"] = z_lgbm_all
    # component value used downstream: OOF on train, train-fit on val/test
    is_tr = (d.split == "train").to_numpy()
    d["z_lgbm"] = np.where(is_tr, d.z_lgbm_oof, d.z_lgbm_all)
    d["z_s1"] = np.where(is_tr, d.s1_text_desc_oof, d.s1_text_desc_all)
    dev = d[d.split.isin(["train", "val"])].reset_index(drop=True)
    va = dev[dev.split == "val"]
    rows, cal, rel, sel = [], [], [], dict(lgbm_refit_max_abs_diff=det_err)

    # ---- stage 5: per-component Platt-on-residual calibration, fitted on TRAIN OOF
    cal_models = {}
    for comp in ("z_lgbm", "z_s1"):
        ok = dev[comp].notna().to_numpy() & (dev.split == "train").to_numpy()
        fr = dev[ok]
        m = OffsetLogit(l2=1e-4, penalize_intercept=False).fit(fr[[comp]].to_numpy(float), fr.price_logit, fr.y,
                                                              D.weights(fr, "market"), frame=fr)
        cal_models[comp] = m
        sel[f"calib_{comp}"] = dict(a=float(m.intercept_), b_on_std=float(m.coef_[0]),
                                    std=float(m.scaler.sd[0]), n_train_oof=int(ok.sum()))
        for nm, p in ((comp + "_raw", D.expit(va.price_logit + va[comp])),
                      (comp + "_platt", m.predict_proba(va[[comp]].to_numpy(float), va.price_logit))):
            rows.append(score(va, p, nm, B=B))
            for u in ("in_scope", "macro"):
                mu = (va.universe == u).to_numpy()
                t = M.reliability_table(va.y[mu], np.asarray(p)[mu], M.snapshot_weights(va[mu], "market"))
                rel.append(t.assign(model=nm, universe=u))
                # ECE with event-bootstrap CI
                e = _ece_ci(va[mu], np.asarray(p)[mu], B=300)
                cal.append(dict(model=nm, universe=u, **e))
    for u in ("in_scope", "macro"):
        mu = (va.universe == u).to_numpy()
        rel.append(M.reliability_table(va.y[mu], va.price_yes[mu], M.snapshot_weights(va[mu], "market"))
                   .assign(model="price", universe=u))
        cal.append(dict(model="price", universe=u, **_ece_ci(va[mu], va.price_yes.to_numpy()[mu], B=300)))

    # ---- stage 4: stack
    stack = {}
    for nm, zc, fi in (("stack", ["z_lgbm", "z_s1"], True), ("stack_noint", ["z_lgbm", "z_s1"], False),
                       ("stack_lgbm_only", ["z_lgbm"], True)):
        l2, t, sub = _meta_wf(dev, zc, fi)
        m = _fit_meta(sub, sub[zc].to_numpy(float), l2, fi)
        stack[nm] = (m, zc)
        sel[nm] = dict(l2=l2, intercept=float(m.intercept_), coef_on_std=dict(zip(zc, map(float, m.coef_))),
                       n_train_oof=len(sub), wf=t.to_dict("records"))
        rows.append(score(va, m.predict_proba(va[zc].to_numpy(float), va.price_logit), nm, B=B))
    # reference rows
    rows.append(score(va, D.expit(va.price_logit + z_price_only[(d.split == "val").to_numpy()]), "ctrl_lgbm_price_only",
                      B=B))
    rows.append(score(va, D.expit(va.price_logit + z_ridge_noint[(d.split == "val").to_numpy()]), "ridge_noint", B=B))

    # ---- frozen candidate predictions (all splits)
    cand = {
        "lgbm_anchored": D.expit(d.price_logit + d.z_lgbm_all),             # PRIMARY
        # no free intercept: an intercept fitted on walk-forward OOF rows learns the YES-excess created by
        # selecting events that close inside the fold window (docs/modeling.md, "window-selection bias")
        "stack_s1": stack["stack_noint"][0].predict_proba(
            np.column_stack([d.z_lgbm_all, d.s1_text_desc_all]), d.price_logit),
        "ridge_flow": D.expit(d.price_logit + z_ridge_noint),
    }
    controls = {"ctrl_lgbm_price_only": D.expit(d.price_logit + z_price_only)}
    val_m_all = (d.split == "val").to_numpy()

    # ---- level-shift diagnostic on VAL for candidates (label-free constant = candidate's mean val residual)
    shift = []
    for cname, p in cand.items():
        for u in ("in_scope", "macro"):
            mu = val_m_all & (d.universe == u).to_numpy()
            sub = d[mu]
            w = M.snapshot_weights(sub, "market")
            r = D.logit(np.asarray(p)[mu]) - sub.price_logit.to_numpy()
            c = np.average(r, weights=w)
            y, o = sub.y.to_numpy(float), sub.price_logit.to_numpy(float)
            shift.append(dict(model=cname, universe=u, mean_resid=float(c),
                              d_brier_model=M.brier(y, D.expit(o), w) - M.brier(y, D.expit(o + r), w),
                              d_brier_const_shift=M.brier(y, D.expit(o), w) - M.brier(y, D.expit(o + c), w)))
    pd.DataFrame(shift).to_csv(D.ART / "stage4_val_level_shift_diag.csv", index=False)

    # ---- stage 6: flag threshold on VAL (rule: among thresholds with >= 10 flagged val events in BOTH directions
    # pooled, take the one with the highest pooled val excess precision whose clustered 95% CI lower bound > 0;
    # if none qualifies, freeze 0.05 (the harness default) and declare the flag expected-uninformative)
    flag_tab, flag_sel = [], {}
    val_m = (d.split == "val").to_numpy()
    for cname, p in {**cand, **controls}.items():
        best = None
        for th in FLAG_GRID:
            for u in ("pooled", "in_scope", "macro"):
                sub = d[val_m] if u == "pooled" else d[val_m & (d.universe == u).to_numpy()]
                pv = np.asarray(p)[sub.index]
                fl = np.where(pv - sub.price_yes >= th, 1, np.where(sub.price_yes - pv >= th, -1, 0))
                f = fl != 0
                if f.sum() == 0:
                    flag_tab.append(dict(model=cname, threshold=th, universe=u, n_flag=0))
                    continue
                hit = np.where(fl[f] > 0, sub.y[f], 1 - sub.y[f]).astype(float)
                imp = np.where(fl[f] > 0, sub.price_yes[f], 1 - sub.price_yes[f]).astype(float)
                pt, lo, hi = M.clustered_mean_ci(hit - imp, sub.event_id[f].to_numpy(), B=500)
                r = dict(model=cname, threshold=th, universe=u, n_flag=int(f.sum()),
                         n_evt=int(sub.event_id[f].nunique()), share_yes=float((fl[f] > 0).mean()),
                         precision=float(hit.mean()), implied=float(imp.mean()), excess=pt, excess_lo=lo, excess_hi=hi)
                flag_tab.append(r)
                if u == "pooled" and r["n_evt"] >= 10 and lo > 0 and (best is None or pt > best[1]):
                    best = (th, pt)
        flag_sel[cname] = dict(threshold=best[0] if best else 0.05, rule_met=best is not None)
    sel["flags"] = flag_sel

    # write frozen parquet files (all splits)
    for cname, p in {**cand, **controls}.items():
        th = flag_sel.get(cname, dict(threshold=0.05))["threshold"]
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        diff = p - d.price_yes.to_numpy(float)
        out = pd.DataFrame(dict(snapshot_id=d.snapshot_id, p_model=p,
                                flag=np.where(diff >= th, 1, np.where(-diff >= th, -1, 0)).astype(int),
                                edge=diff, split=d.split, universe=d.universe,
                                fit_kind=np.where(d.split == "train", "in_sample_train_fit", "out_of_sample_train_fit")))
        if cname == "stack_s1":
            out["s1_informed_score"] = d.s1_aux_informed_score.to_numpy()
            out["z_lgbm"] = d.z_lgbm_all.to_numpy()
            out["z_s1"] = d.s1_text_desc_all.to_numpy()
        if cname == "lgbm_anchored":
            oof_p = D.expit(d.price_logit + d.z_lgbm_oof)
            out["p_model_train_oof"] = np.where(d.split == "train", oof_p, np.nan)
        assert out.p_model.notna().all() and out.snapshot_id.is_unique
        out.to_parquet(D.ART / f"preds_{cname}.parquet", index=False)

    R = pd.concat(rows, ignore_index=True)
    R.to_csv(D.ART / "stage4_val_results.csv", index=False)
    pd.DataFrame(cal).to_csv(D.ART / "stage5_val_ece.csv", index=False)
    pd.concat(rel).to_csv(D.ART / "stage5_val_reliability.csv", index=False)
    pd.DataFrame(flag_tab).to_csv(D.ART / "stage6_val_flags.csv", index=False)
    (D.ART / "stage4_selected.json").write_text(json.dumps(sel, indent=1, default=float))
    _reliability_plot(pd.concat(rel))
    v = R[R.weighting == "market"]
    print(v[["model", "universe", "d_brier", "d_brier_lo", "d_brier_hi", "d_ll", "d_ll_lo", "d_ll_hi"]].round(4)
          .to_string())
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "wf"} for k, v in sel.items()
                      if isinstance(v, dict)}, indent=1, default=float))


def _ece_ci(df, p, B=300):
    w = M.snapshot_weights(df, "market")
    e = M.ece(df.y.to_numpy(float), p, w)
    g = pd.factorize(df.event_id)[0]
    G = g.max() + 1
    rng = np.random.default_rng(0)
    draws = []
    idx_by = [np.flatnonzero(g == k) for k in range(G)]
    for _ in range(B):
        pick = np.concatenate([idx_by[k] for k in rng.integers(0, G, G)])
        draws.append(M.ece(df.y.to_numpy(float)[pick], np.asarray(p)[pick], w[pick]))
    return dict(ece=e, ece_lo=float(np.quantile(draws, 0.025)), ece_hi=float(np.quantile(draws, 0.975)))


def _reliability_plot(rel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, u in zip(axes, ("in_scope", "macro")):
        for nm, c in (("price", "#555"), ("z_lgbm_raw", "#2a5d9f"), ("z_lgbm_platt", "#d08a1f"),
                      ("z_s1_platt", "#3a9a5b")):
            t = rel[(rel.model == nm) & (rel.universe == u) & (rel.n > 0)]
            ax.plot(t.mean_p, t.freq_y, "o-", ms=3, color=c, label=nm)
        ax.plot([0, 1], [0, 1], ":", color="#999")
        ax.set_title(f"VAL reliability ({u}, market-weighted)", fontsize=9)
        ax.set_xlabel("mean forecast")
        ax.set_ylabel("observed YES rate")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(D.ART / "stage5_val_reliability.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
