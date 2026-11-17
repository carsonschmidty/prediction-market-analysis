"""Stage 1: Baseline A (raw price) on exactly the modeling rows (24h grid subset), TRAIN and VAL only.

Sanity: the evaluation harness, given p_model = price on the full 6-hourly primary set (train/val only),
must reproduce the published Baseline A numbers; then the same on our 24h-grid rows.
    python -m pmflow.modeling.stage1_baseline
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pmflow.evaluation import metrics as M
from pmflow.evaluation.harness import compare, load_snapshots

from .data import ART, load_frame, no_test


def main():
    ART.mkdir(parents=True, exist_ok=True)
    snaps = load_snapshots()
    dev = snaps[snaps.split.isin(["train", "val"])]           # test rows never passed in
    rows = []
    rep = compare(dev[["snapshot_id"]].assign(p_model=dev.price_yes.to_numpy()), dev, B=500, splits=["train", "val"])
    s = rep["summary"]
    rows.append(s[s.weighting == "snapshot"].assign(rows="full_6h_grid")[["rows", "universe", "split", "n_snap",
                                                                            "n_evt", "brier_base", "log_loss_base"]])
    d = no_test(load_frame())
    rep2 = compare(d[["snapshot_id"]].assign(p_model=d.price_yes.astype(float).to_numpy()), dev, B=500,
                   splits=["train", "val"])
    s2 = rep2["summary"]
    rows.append(s2.assign(rows="modeling_24h_grid")[["rows", "universe", "split", "weighting", "n_snap", "n_evt",
                                                     "brier_base", "log_loss_base", "ece_base"]])
    # event-clustered CIs of the absolute Brier on our rows
    ci = []
    for (u, sp), g in d.groupby(["universe", "split"]):
        y, p = g.y.to_numpy(float), g.price_yes.to_numpy(float)
        pt, lo, hi, _ = M.bootstrap_ratio(g.event_id.to_numpy(), M.brier_terms(y, p), np.ones(len(g)), B=1000)
        lpt, llo, lhi, _ = M.bootstrap_ratio(g.event_id.to_numpy(), M.logloss_terms(y, p), np.ones(len(g)), B=1000)
        ci.append(dict(universe=u, split=sp, n_snap=len(g), n_mkt=g.market_id.nunique(), n_evt=g.event_id.nunique(),
                       base_rate=y.mean(), mean_price=p.mean(), brier=pt, brier_lo=lo, brier_hi=hi,
                       log_loss=lpt, ll_lo=llo, ll_hi=lhi))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(ART / "stage1_baseline_sanity.csv", index=False)
    pd.DataFrame(ci).to_csv(ART / "stage1_baseline_modeling_rows.csv", index=False)
    print(out.round(4).to_string())
    print(pd.DataFrame(ci).round(4).to_string())


if __name__ == "__main__":
    main()
