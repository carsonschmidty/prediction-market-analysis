"""Development-time scoring (train OOF / val only). Refuses test rows.

Delta convention as in the harness: delta = loss(price) - loss(model); positive = model better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pmflow.evaluation import metrics as M

UNIVERSES = ("in_scope", "macro")


class DevScoringRefused(AssertionError):
    pass


def _guard(d: pd.DataFrame):
    if d["split"].eq("test").any() or d["split"].str.startswith("purged").any():
        raise DevScoringRefused("development scoring must not touch test/purged rows")


def score(d: pd.DataFrame, p, name: str, base_col: str = "price_yes", weighting=("snapshot", "market", "event"),
          B: int = 1000, seed: int = 0, by_universe: bool = True, extra: dict | None = None) -> pd.DataFrame:
    """Paired comparison of p (aligned with d) vs price on d's rows, per universe and weighting."""
    _guard(d)
    d = d.assign(_p=np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4))
    rows = []
    groups = [(u, d[d.universe == u]) for u in UNIVERSES] if by_universe else [("all", d)]
    for u, du in groups:
        if len(du) == 0:
            continue
        y, pm, pb, g = du.y.to_numpy(float), du._p.to_numpy(), du[base_col].to_numpy(float), du.event_id.to_numpy()
        for wn in weighting:
            w = M.snapshot_weights(du, wn)
            tb = M.paired_test(M.brier_terms(y, pb), M.brier_terms(y, pm), g, w, B=B, seed=seed)
            tl = M.paired_test(M.logloss_terms(y, pb), M.logloss_terms(y, pm), g, w, B=B, seed=seed + 1)
            rows.append(dict(model=name, universe=u, weighting=wn, n_snap=len(du), n_evt=du.event_id.nunique(),
                             brier_price=M.brier(y, pb, w), brier_model=M.brier(y, pm, w),
                             d_brier=tb["delta"], d_brier_lo=tb["ci_lo"], d_brier_hi=tb["ci_hi"],
                             d_brier_p=tb["p_cluster_t"],
                             ll_price=M.log_loss(y, pb, w), ll_model=M.log_loss(y, pm, w),
                             d_ll=tl["delta"], d_ll_lo=tl["ci_lo"], d_ll_hi=tl["ci_hi"], d_ll_p=tl["p_cluster_t"],
                             ece_model=M.ece(y, pm, w), ece_price=M.ece(y, pb, w),
                             mean_abs_move=float(np.average(np.abs(pm - pb), weights=w)),
                             mean_resid_logit=float(np.average(M.logit(pm) - M.logit(pb), weights=w)),
                             **(extra or {})))
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame, cols=("model", "universe", "weighting", "n_snap", "n_evt", "brier_price", "brier_model",
                                "d_brier", "d_brier_lo", "d_brier_hi", "d_ll", "d_ll_lo", "d_ll_hi")) -> str:
    cols = [c for c in cols if c in df]
    t = df[cols].copy()
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda v: f"{v:+.4f}" if c.startswith("d_") else f"{v:.4f}")
    return t.to_markdown(index=False)
